"""
Time-series dataset for VISIT model emulation with static parameter conditioning.

Loads the latest VISIT-LHP perturbation batch (currently 180 samples, 2010-2019
daily) and creates sliding window samples for LSTM training.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import pickle

TIME_COLUMNS = ["year", "doy", "month", "day"]
METEOROLOGY_COLUMNS = [
    "tmp_sfc",
    "tmp_2m",
    "tmp10_soil",
    "tmp200_soil",
    "dswrf_sfc",
    "tcdc_clm",
    "prate_sfc",
    "spfh_2m",
    "wind_10m",
]
DYNAMIC_COLUMNS = METEOROLOGY_COLUMNS + ["a_co2"]
FUTURE_KNOWN_COLUMNS = list(DYNAMIC_COLUMNS)
FLUX_COLUMNS = ["gpp", "npp", "er", "nep"]
PARAMETER_COLUMNS = [f"param_{i:02d}" for i in range(51)]
DAILY_COLUMN_NAMES = TIME_COLUMNS + DYNAMIC_COLUMNS + FLUX_COLUMNS + PARAMETER_COLUMNS
DAILY_FILE_PATTERN = "*_daily_*.txt"

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class VISITTimeSeriesDataset(Dataset):
    """
    Dataset for VISIT model emulation with:
    - Dynamic features: meteorology (9) + aCO2 (1) = 10 features
    - Static features: continuous parameter vector (36 in current summary)
    - Targets: GPP, NPP, NEP, Rh (4 outputs)
    
    Sliding window approach:
    - context_len: historical steps to condition on
    - prediction_len: future steps to predict
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        context_len: int = 180,
        prediction_len: int = 30,
        train_range: Tuple[str, str] = ("2010-01-01", "2017-12-31"),
        val_range: Tuple[str, str] = ("2018-01-01", "2018-12-31"),
        test_range: Tuple[str, str] = ("2019-01-01", "2019-12-31"),
        scaler_dynamic: Optional[StandardScaler] = None,
        scaler_static: Optional[StandardScaler] = None,
        fit_scaler: bool = False
    ):
        """
        Args:
            data_dir: Path to VISIT perturbation outputs (e.g., OUTPUT_PERTURBATION_LHP_V2_LHS)
            split: "train", "val", or "test"
            context_len: Number of historical time steps
            prediction_len: Number of future time steps to predict
            train_range, val_range, test_range: Date ranges for splits
            scaler_dynamic, scaler_static: Pre-fitted scalers (for val/test)
            fit_scaler: Whether to fit scalers (only True for train split)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.fit_scaler = fit_scaler
        
        # Define date ranges
        self.date_ranges = {
            "train": pd.date_range(train_range[0], train_range[1], freq="D"),
            "val": pd.date_range(val_range[0], val_range[1], freq="D"),
            "test": pd.date_range(test_range[0], test_range[1], freq="D")
        }
        
        # Load data
        print(f"\n{'='*60}")
        print(f"Loading {split.upper()} split from {data_dir}")
        print(f"Date range: {self.date_ranges[split][0]} to {self.date_ranges[split][-1]}")
        print(f"Context length: {context_len}, Prediction length: {prediction_len}")
        
        self.samples_data, self.static_params, self.sample_ids = self._load_all_samples()
        self.dynamic_feature_names = list(DYNAMIC_COLUMNS)
        self.future_known_feature_names = list(FUTURE_KNOWN_COLUMNS)
        self.dynamic_dim = len(self.dynamic_feature_names)
        self.future_known_dim = len(self.future_known_feature_names)
        self.static_dim = self.static_params.shape[1]
        self.future_known_indices = list(range(self.future_known_dim))
        self.num_samples = len(self.samples_data)
        
        # Extract time indices for this split
        self.split_indices = self._get_split_indices()
        
        # Scale data
        if fit_scaler:
            self.scaler_dynamic = StandardScaler()
            self.scaler_static = StandardScaler()
            self._fit_scalers()
        else:
            assert scaler_dynamic is not None and scaler_static is not None, \
                "Must provide pre-fitted scalers for val/test splits"
            self.scaler_dynamic = scaler_dynamic
            self.scaler_static = scaler_static
        
        self._apply_scaling()
        
        # Create sliding windows
        self.windows = self._create_windows()
        
        print(f"Created {len(self.windows)} windows for {split} split")
        print(f"Dynamic features shape: {self.dynamic_features.shape}")
        print(f"Static params shape: {self.static_params.shape}")
        print(f"Targets shape: {self.targets.shape}")
        print(f"{'='*60}\n")
    
    def _index_daily_files(self) -> Dict[int, Path]:
        """Create a mapping from sample_id -> daily output path."""
        file_map: Dict[int, Path] = {}
        for path in sorted(self.data_dir.glob(DAILY_FILE_PATTERN)):
            tokens = path.stem.split("_")
            for token in reversed(tokens):
                if token.isdigit():
                    file_map[int(token)] = path
                    break
        return file_map

    def _load_all_samples(self) -> Tuple[List[pd.DataFrame], np.ndarray, List[int]]:
        """Load all daily output files and the parameter summary."""
        param_file = self.data_dir / "parameter_summary.csv"
        param_df = pd.read_csv(param_file)
        static_params = param_df.drop(columns=["sample_id"]).values
        n_params_perturbed = static_params.shape[1]
        sample_ids = param_df["sample_id"].astype(int).tolist()
        
        file_map = self._index_daily_files()
        missing_files = [sid for sid in sample_ids if sid not in file_map]
        if missing_files:
            raise FileNotFoundError(
                f"Daily output files missing for sample IDs: {missing_files[:5]}"
            )
        
        samples_data = []
        for sample_id in sample_ids:
            file_path = file_map[sample_id]
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=DAILY_COLUMN_NAMES)
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01") + pd.to_timedelta(df["doy"], unit="D")
            df = df.set_index("date")
            samples_data.append(df)
        
        print(f"Detected {n_params_perturbed} perturbed parameters from parameter_summary.csv")
        print(f"Loaded {len(samples_data)} sample files")
        print(f"Each sample has {len(samples_data[0])} time steps")
        print(f"Static parameters array: {static_params.shape}")
        
        return samples_data, static_params, sample_ids
    
    def _get_split_indices(self) -> List[int]:
        """Get time indices for this split."""
        split_dates = self.date_ranges[self.split]
        
        # Get indices in the full time series (2010-2019)
        full_dates = pd.date_range("2010-01-01", "2019-12-31", freq="D")
        indices = [full_dates.get_loc(date) for date in split_dates]
        
        return indices
    
    def _fit_scalers(self):
        """Fit scalers on training data."""
        # Collect all training data across samples
        all_dynamic = []
        for sample_df in self.samples_data:
            sample_df_train = sample_df.iloc[self.split_indices]
            dynamic_feat = sample_df_train[self.dynamic_feature_names].values
            all_dynamic.append(dynamic_feat)
        
        all_dynamic = np.vstack(all_dynamic)  # (n_samples * n_train_days, 11)
        
        self.scaler_dynamic.fit(all_dynamic)
        self.scaler_static.fit(self.static_params)
        
        print(f"Fitted dynamic scaler: mean={self.scaler_dynamic.mean_[:3]}, std={self.scaler_dynamic.scale_[:3]}")
        print(f"Fitted static scaler: mean={self.scaler_static.mean_[:3]}, std={self.scaler_static.scale_[:3]}")
    
    def _apply_scaling(self):
        """Apply scaling to dynamic and static features."""
        # Scale dynamic features for each sample
        scaled_samples = []
        for sample_df in self.samples_data:
            dynamic_feat = sample_df[self.dynamic_feature_names].values
            scaled_dynamic = self.scaler_dynamic.transform(dynamic_feat)
            targets = sample_df[FLUX_COLUMNS].values
            scaled_samples.append({
                "dynamic": scaled_dynamic,
                "targets": targets
            })
        
        self.dynamic_features = np.stack([s["dynamic"] for s in scaled_samples])
        self.targets = np.stack([s["targets"] for s in scaled_samples])
        
        # Scale static parameters
        self.static_params_scaled = self.scaler_static.transform(self.static_params)  # (100, n_params)
    
    def _create_windows(self) -> List[Dict]:
        """
        Create sliding windows for each sample and time range.
        
        Returns:
            List of dicts with:
                - sample_id: which perturbed sample
                - start_idx: start index in full time series
                - end_idx: end index for prediction
        """
        windows = []
        
        for sample_id in range(len(self.samples_data)):
            # Get valid time indices for this split
            valid_indices = self.split_indices
            
            # Create sliding windows
            for i in range(len(valid_indices) - self.context_len - self.prediction_len + 1):
                start_idx = valid_indices[i]
                context_end_idx = valid_indices[i + self.context_len - 1]
                pred_end_idx = valid_indices[i + self.context_len + self.prediction_len - 1]
                
                windows.append({
                    "sample_id": sample_id,
                    "start_idx": start_idx,
                    "context_end_idx": context_end_idx,
                    "pred_end_idx": pred_end_idx
                })
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - context_x: (context_len, dynamic_dim) dynamic features
                - static_x: (n_params,) static parameters
                - future_known: (prediction_len, future_known_dim) known future inputs
                - target_y: (prediction_len, 4) target outputs
        """
        window = self.windows[idx]
        sample_id = window["sample_id"]
        start_idx = window["start_idx"]
        context_end_idx = window["context_end_idx"]
        pred_end_idx = window["pred_end_idx"]
        
        # Context: historical dynamic features
        context_x = self.dynamic_features[
            sample_id, 
            start_idx:context_end_idx+1, 
            :
        ]  # (context_len, dynamic_dim)
        
        # Static parameters
        static_x = self.static_params_scaled[sample_id]  # (n_params,)
        
        # Future known inputs: meteorology (7) + CO2 (1)
        future_known = self.dynamic_features[
            sample_id,
            context_end_idx+1:pred_end_idx+1,
            :
        ][:, self.future_known_indices]  # (prediction_len, future_known_dim)
        
        # Targets
        target_y = self.targets[
            sample_id,
            context_end_idx+1:pred_end_idx+1,
            :
        ]  # (prediction_len, 4)
        
        return {
            "context_x": torch.FloatTensor(context_x),
            "static_x": torch.FloatTensor(static_x),
            "future_known": torch.FloatTensor(future_known),
            "target_y": torch.FloatTensor(target_y)
        }
    
    def get_scalers(self) -> Tuple[StandardScaler, StandardScaler]:
        """Return fitted scalers for saving."""
        return self.scaler_dynamic, self.scaler_static


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    context_len: int = 180,
    prediction_len: int = 30,
    num_workers: int = 4
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Returns:
        dict with "train", "val", "test" DataLoader objects
    """
    print("\nCreating dataloaders...")
    # Create train dataset and fit scalers
    train_dataset = VISITTimeSeriesDataset(
        data_dir=data_dir,
        split="train",
        context_len=context_len,
        prediction_len=prediction_len,
        fit_scaler=True
    )
    
    # Get fitted scalers
    scaler_dynamic, scaler_static = train_dataset.get_scalers()
    
    # Create val/test datasets with fitted scalers
    val_dataset = VISITTimeSeriesDataset(
        data_dir=data_dir,
        split="val",
        context_len=context_len,
        prediction_len=prediction_len,
        scaler_dynamic=scaler_dynamic,
        scaler_static=scaler_static,
        fit_scaler=False
    )
    
    test_dataset = VISITTimeSeriesDataset(
        data_dir=data_dir,
        split="test",
        context_len=context_len,
        prediction_len=prediction_len,
        scaler_dynamic=scaler_dynamic,
        scaler_static=scaler_static,
        fit_scaler=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Save scalers
    save_dir = Path(data_dir).parent / "ML_LHP_LSTM" / "data"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "scaler_dynamic.pkl", "wb") as f:
        pickle.dump(scaler_dynamic, f)
    
    with open(save_dir / "scaler_static.pkl", "wb") as f:
        pickle.dump(scaler_static, f)
    
    print(f"Saved scalers to {save_dir}")
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


if __name__ == "__main__":
    # Test dataset creation
    data_dir = "/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS"
    
    dataloaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        context_len=180,
        prediction_len=30,
        num_workers=0  # Use 0 for testing
    )
    
    # Test one batch
    for batch in dataloaders["train"]:
        print("\nBatch shapes:")
        for key, val in batch.items():
            print(f"  {key}: {val.shape}")
        break
