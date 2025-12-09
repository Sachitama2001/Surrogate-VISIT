"""
Training script for VISIT LSTM emulator with:
- Hybrid loss: 0.7 * one-step MSE + 0.3 * rollout MSE
- Scheduled sampling teacher forcing
- AdamW optimizer with cosine warmup
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import create_model, LSTMStaticConditioning
from dataset import create_dataloaders


class HybridLoss(nn.Module):
    """
    Hybrid loss combining one-step and rollout predictions.
    
    Loss = 0.7 * MSE(y_pred_t, y_true_t) + 0.3 * MSE(rollout[:30], y_true[:30])
    """
    
    def __init__(
        self,
        one_step_weight: float = 0.7,
        rollout_weight: float = 0.3,
        rollout_len: int = 30
    ):
        super().__init__()
        self.one_step_weight = one_step_weight
        self.rollout_weight = rollout_weight
        self.rollout_len = rollout_len
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (batch_size, prediction_len, output_dim)
            targets: (batch_size, prediction_len, output_dim)
        
        Returns:
            dict with "total_loss", "one_step_loss", "rollout_loss"
        """
        # One-step loss: all time steps
        one_step_loss = self.mse(predictions, targets)
        
        # Rollout loss: first rollout_len steps
        rollout_predictions = predictions[:, :self.rollout_len, :]
        rollout_targets = targets[:, :self.rollout_len, :]
        rollout_loss = self.mse(rollout_predictions, rollout_targets)
        
        # Combined loss
        total_loss = (
            self.one_step_weight * one_step_loss +
            self.rollout_weight * rollout_loss
        )
        
        return {
            "total_loss": total_loss,
            "one_step_loss": one_step_loss,
            "rollout_loss": rollout_loss
        }


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
):
    """
    Cosine learning rate schedule with warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Trainer for VISIT LSTM model.
    """
    
    def __init__(
        self,
        model: LSTMStaticConditioning,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = HybridLoss(
            one_step_weight=config["loss"]["one_step_weight"],
            rollout_weight=config["loss"]["rollout_weight"],
            rollout_len=config["loss"]["rollout_len"]
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"]
        )
        
        # Scheduler
        num_training_steps = len(train_loader) * config["num_epochs"]
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["scheduler"]["warmup_steps"],
            num_training_steps=num_training_steps
        )
        
        # Scheduled sampling
        self.teacher_forcing_start = config["teacher_forcing"]["start_prob"]
        self.teacher_forcing_end = config["teacher_forcing"]["end_prob"]
        self.teacher_forcing_decay_steps = config["teacher_forcing"]["decay_steps"]
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        # Save directory
        self.save_dir = Path(config["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def get_teacher_forcing_ratio(self) -> float:
        """Compute current teacher forcing ratio with linear decay."""
        if self.global_step >= self.teacher_forcing_decay_steps:
            return self.teacher_forcing_end
        
        progress = self.global_step / self.teacher_forcing_decay_steps
        return self.teacher_forcing_start - progress * (
            self.teacher_forcing_start - self.teacher_forcing_end
        )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_one_step_losses = []
        epoch_rollout_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get teacher forcing ratio
            teacher_forcing_ratio = self.get_teacher_forcing_ratio()
            
            # Forward pass
            predictions = self.model(
                context_x=batch["context_x"],
                static_x=batch["static_x"],
                future_known=batch["future_known"],
                teacher_forcing_ratio=teacher_forcing_ratio,
                target_y=batch["target_y"]
            )
            
            # Compute loss
            loss_dict = self.criterion(predictions, batch["target_y"])
            loss = loss_dict["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get("gradient_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clip"]
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            epoch_one_step_losses.append(loss_dict["one_step_loss"].item())
            epoch_rollout_losses.append(loss_dict["rollout_loss"].item())
            
            self.global_step += 1
            
            # Log progress
            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"TF ratio: {teacher_forcing_ratio:.3f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_one_step = np.mean(epoch_one_step_losses)
        avg_rollout = np.mean(epoch_rollout_losses)
        
        print(f"\n[Epoch {epoch}] Train Loss: {avg_loss:.4f} "
              f"(One-step: {avg_one_step:.4f}, Rollout: {avg_rollout:.4f})")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate on validation set."""
        self.model.eval()
        
        val_losses = []
        val_one_step_losses = []
        val_rollout_losses = []
        
        for batch in self.val_loader:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass (no teacher forcing)
            predictions = self.model(
                context_x=batch["context_x"],
                static_x=batch["static_x"],
                future_known=batch["future_known"],
                teacher_forcing_ratio=0.0,
                target_y=None
            )
            
            # Compute loss
            loss_dict = self.criterion(predictions, batch["target_y"])
            
            val_losses.append(loss_dict["total_loss"].item())
            val_one_step_losses.append(loss_dict["one_step_loss"].item())
            val_rollout_losses.append(loss_dict["rollout_loss"].item())
        
        avg_val_loss = np.mean(val_losses)
        avg_one_step = np.mean(val_one_step_losses)
        avg_rollout = np.mean(val_rollout_losses)
        
        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f} "
              f"(One-step: {avg_one_step:.4f}, Rollout: {avg_rollout:.4f})\n")
        
        # Save best model
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint(epoch, is_best=True)
            print(f"✓ New best model saved (Val Loss: {avg_val_loss:.4f})\n")
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / "checkpoint_latest.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / "checkpoint_best.pt")
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Save directory: {self.save_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.config["num_epochs"] + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(
                self.scheduler.get_last_lr()[0]
            )
            
            # Save checkpoint
            self.save_checkpoint(epoch)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} completed in {epoch_time/60:.1f} minutes\n")
            print("=" * 60 + "\n")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save training history
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {self.save_dir / 'training_history.json'}")


def main():
    """Main training script."""
    
    # Configuration
    config = {
        # Data
        "data_dir": "/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS",
        "batch_size": 64,
        "context_len": 180,
        "prediction_len": 30,
        "num_workers": 4,
        
        # Model
        "dynamic_dim": None,  # Will be auto-detected from data
        "static_dim": None,   # Will be auto-detected from data
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "output_dim": 4,
        "known_future_dim": None,
        
        # Training
        "num_epochs": 50,
        "optimizer": {
            "lr": 2e-3,
            "weight_decay": 1e-4
        },
        "scheduler": {
            "warmup_steps": 500
        },
        "teacher_forcing": {
            "start_prob": 1.0,
            "end_prob": 0.3,
            "decay_steps": 20000
        },
        "loss": {
            "one_step_weight": 0.7,
            "rollout_weight": 0.3,
            "rollout_len": 30
        },
        "gradient_clip": 1.0,
        
        # Logging
        "save_dir": "/mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM/artifacts"
    }
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        context_len=config["context_len"],
        prediction_len=config["prediction_len"],
        num_workers=config["num_workers"]
    )
    
    train_dataset = dataloaders["train"].dataset
    config["dynamic_dim"] = train_dataset.dynamic_dim
    config["static_dim"] = train_dataset.static_dim
    config["known_future_dim"] = train_dataset.future_known_dim
    config["num_samples"] = train_dataset.num_samples
    config["dynamic_features"] = train_dataset.dynamic_feature_names
    config["future_known_features"] = train_dataset.future_known_feature_names
    print(
        f"\nAuto-detected dims -> dynamic: {config['dynamic_dim']}, "
        f"static: {config['static_dim']}, future_known: {config['known_future_dim']}"
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        dynamic_dim=config["dynamic_dim"],
        static_dim=config["static_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
        device=device
    )
    
    # Save config
    config_save_path = Path(config["save_dir"]) / "config.json"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_save_path}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config,
        device=device
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
