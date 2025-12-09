"""
LSTM model with static parameter conditioning for VISIT emulation.

Implements two conditioning methods:
1. init_state: Static params -> MLP -> (h0, c0)
2. concat_input: Static params -> embedding -> concat with dynamic input
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMStaticConditioning(nn.Module):
    """
    LSTM with static parameter conditioning for multi-horizon time series prediction.
    
    Architecture:
    - Base LSTM: processes dynamic inputs (meteorology, CO2, state)
    - Static conditioning layer 1: MLP to initialize LSTM hidden states
    - Static conditioning layer 2: embedding concatenated with dynamic inputs
    - Multi-output head: 4 regression outputs (GPP, NPP, NEP, Rh)
    """
    
    def __init__(
        self,
        dynamic_dim: int = 10,         # meteorology (9) + aCO2 (1)
        static_dim: int = 36,          # Number of static parameters
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 4,           # GPP, NPP, NEP, Rh
        static_mlp_dims: Tuple[int] = (128,),
        static_emb_dim: int = 64
    ):
        """
        Args:
            dynamic_dim: Number of dynamic input features
            static_dim: Number of static parameters
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_dim: Number of output targets
            static_mlp_dims: MLP hidden dimensions for init_state conditioning
            static_emb_dim: Embedding dimension for concat_input conditioning
        """
        super().__init__()
        
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.static_emb_dim = static_emb_dim
        
        # Static conditioning 1: MLP to initialize LSTM states
        mlp_layers = []
        in_dim = static_dim
        for hidden_dim in static_mlp_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Project to (h0, c0) for all layers
        mlp_layers.append(nn.Linear(in_dim, 2 * num_layers * hidden_size))
        self.static_mlp = nn.Sequential(*mlp_layers)
        
        # Static conditioning 2: Embedding to concat with dynamic input
        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, static_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM: input is dynamic features + static embedding
        self.lstm = nn.LSTM(
            input_size=dynamic_dim + static_emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_dim)
        )
    
    def init_hidden(self, static_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden states from static parameters.
        
        Args:
            static_x: (batch_size, static_dim)
        
        Returns:
            h0: (num_layers, batch_size, hidden_size)
            c0: (num_layers, batch_size, hidden_size)
        """
        batch_size = static_x.shape[0]
        
        # MLP projection
        init_state = self.static_mlp(static_x)  # (batch_size, 2*num_layers*hidden_size)
        init_state = init_state.view(batch_size, 2, self.num_layers, self.hidden_size)
        
        h0 = init_state[:, 0, :, :].transpose(0, 1).contiguous()  # (num_layers, batch_size, hidden_size)
        c0 = init_state[:, 1, :, :].transpose(0, 1).contiguous()
        
        return h0, c0
    
    def forward(
        self,
        context_x: torch.Tensor,
        static_x: torch.Tensor,
        future_known: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        target_y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.
        
        Args:
            context_x: (batch_size, context_len, dynamic_dim) - historical inputs
            static_x: (batch_size, static_dim) - static parameters
            future_known: (batch_size, prediction_len, known_dim) - known future inputs (meteorology + CO2)
            teacher_forcing_ratio: Probability of using true previous output vs predicted
            target_y: (batch_size, prediction_len, output_dim) - targets for teacher forcing
        
        Returns:
            predictions: (batch_size, prediction_len, output_dim)
        """
        batch_size, context_len, _ = context_x.shape
        device = context_x.device
        
        # Initialize hidden states from static parameters
        h, c = self.init_hidden(static_x)
        
        # Embed static parameters for concatenation
        static_emb = self.static_embedding(static_x)  # (batch_size, static_emb_dim)
        static_emb_expanded = static_emb.unsqueeze(1)  # (batch_size, 1, static_emb_dim)
        
        # Process context with LSTM
        context_static = static_emb_expanded.expand(-1, context_len, -1)
        context_input = torch.cat([context_x, context_static], dim=-1)  # (batch_size, context_len, dynamic_dim + static_emb_dim)
        
        context_output, (h, c) = self.lstm(context_input, (h, c))
        
        # Autoregressive prediction
        predictions = []
        if future_known is None:
            raise ValueError("future_known inputs are required for autoregressive prediction.")

        if future_known.shape[-1] != self.dynamic_dim:
            raise ValueError(
                f"Expected future_known features={self.dynamic_dim}, got {future_known.shape[-1]}"
            )

        prediction_len = future_known.shape[1]
        
        # Last output from context
        last_output = self.output_head(context_output[:, -1, :])  # (batch_size, output_dim)

        for t in range(prediction_len):
            if teacher_forcing_ratio > 0 and target_y is not None and t > 0:
                if torch.rand(1, device=device).item() < teacher_forcing_ratio:
                    last_output = target_y[:, t-1, :]
            
            known_input = future_known[:, t, :]  # (batch_size, dynamic_dim)
            next_input = known_input
            
            # Add static conditioning
            next_input_with_static = torch.cat([
                next_input.unsqueeze(1),  # (batch_size, 1, dynamic_dim)
                static_emb_expanded       # (batch_size, 1, static_emb_dim)
            ], dim=-1)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(next_input_with_static, (h, c))
            
            # Predict next output
            next_output = self.output_head(lstm_out[:, 0, :])  # (batch_size, output_dim)
            
            predictions.append(next_output)
            last_output = next_output
        
        predictions = torch.stack(predictions, dim=1)  # (batch_size, prediction_len, output_dim)
        
        return predictions


class LSTMWrapper(nn.Module):
    """
    Wrapper for easy training and inference.
    """
    
    def __init__(self, model: LSTMStaticConditioning):
        super().__init__()
        self.model = model
    
    def forward(self, batch: dict, teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        Forward pass for training/validation.
        
        Args:
            batch: dict with "context_x", "static_x", "future_known", "target_y"
            teacher_forcing_ratio: Teacher forcing probability
        
        Returns:
            predictions: (batch_size, prediction_len, output_dim)
        """
        return self.model(
            context_x=batch["context_x"],
            static_x=batch["static_x"],
            future_known=batch["future_known"],
            teacher_forcing_ratio=teacher_forcing_ratio,
            target_y=batch.get("target_y", None)
        )


def create_model(
    dynamic_dim: int = 11,
    static_dim: int = 26,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    output_dim: int = 4,
    device: str = "cuda"
) -> LSTMStaticConditioning:
    """
    Factory function to create model.
    """
    model = LSTMStaticConditioning(
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_dim,
        static_mlp_dims=(128,),
        static_emb_dim=64
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Model: LSTMStaticConditioning")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    return model


if __name__ == "__main__":
    # Test model
    batch_size = 16
    context_len = 180
    prediction_len = 30
    dynamic_dim = 11
    static_dim = 26
    
    model = create_model(device="cpu")
    
    # Create dummy batch
    batch = {
        "context_x": torch.randn(batch_size, context_len, dynamic_dim),
        "static_x": torch.randn(batch_size, static_dim),
        "future_known": torch.randn(batch_size, prediction_len, 8),
        "target_y": torch.randn(batch_size, prediction_len, 4)
    }
    
    # Forward pass
    predictions = model(
        context_x=batch["context_x"],
        static_x=batch["static_x"],
        future_known=batch["future_known"],
        teacher_forcing_ratio=0.5,
        target_y=batch["target_y"]
    )
    
    print(f"Input shapes:")
    print(f"  context_x: {batch['context_x'].shape}")
    print(f"  static_x: {batch['static_x'].shape}")
    print(f"  future_known: {batch['future_known'].shape}")
    print(f"\nOutput shape: {predictions.shape}")
