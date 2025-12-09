"""
Quick test script to verify dataset loading and model forward pass.
"""

import sys
sys.path.append("/mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM")

import torch
from dataset import VISITTimeSeriesDataset
from model import create_model

def test_dataset():
    """Test dataset loading."""
    print("\n" + "="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    data_dir = "/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS"
    
    # Create train dataset
    train_dataset = VISITTimeSeriesDataset(
        data_dir=data_dir,
        split="train",
        context_len=180,
        prediction_len=30,
        fit_scaler=True
    )
    
    print(f"\n✓ Dataset created successfully")
    print(f"  Total windows: {len(train_dataset)}")
    
    # Test one sample
    sample = train_dataset[0]
    print(f"\n✓ Sample data:")
    for key, val in sample.items():
        print(f"  {key}: {val.shape}")
    
    return train_dataset

def test_model(sample):
    """Test model forward pass."""
    print("\n" + "="*60)
    print("Testing Model Forward Pass")
    print("="*60)
    
    # Get dimensions from sample
    dynamic_dim = sample["context_x"].shape[-1]
    static_dim = sample["static_x"].shape[0]
    
    print(f"\nDimensions:")
    print(f"  dynamic_dim: {dynamic_dim}")
    print(f"  static_dim: {static_dim}")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        device=device
    )
    
    # Create batch
    batch = {
        k: v.unsqueeze(0).to(device) for k, v in sample.items()
    }
    
    # Forward pass
    print(f"\n✓ Running forward pass...")
    predictions = model(
        context_x=batch["context_x"],
        static_x=batch["static_x"],
        future_known=batch["future_known"],
        teacher_forcing_ratio=0.5,
        target_y=batch["target_y"]
    )
    
    print(f"✓ Forward pass successful")
    print(f"  Input: context_x {batch['context_x'].shape}")
    print(f"  Output: predictions {predictions.shape}")
    
    return model

def main():
    """Main test function."""
    try:
        # Test dataset
        dataset = test_dataset()
        
        # Test model
        sample = dataset[0]
        model = test_model(sample)
        
        print("\n" + "="*60)
        print("✓✓✓ All tests passed! ✓✓✓")
        print("="*60)
        print("\nYou can now run training with:")
        print("  python train.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
