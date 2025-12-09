"""
静的パラメータ逆推定モジュール

学習済みLSTMモデルを用いて、観測データ(NEP)から
静的パラメータ（現在36次元）を逆推定する。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import pickle


class StaticParameterInverter:
    """
    静的パラメータの逆推定を行うクラス
    
    固定:
        - LSTMモデルの重み
        - 動的入力(気象・CO2・状態変数)
        - スケーラー
    
    最適化:
        - 静的パラメータベクトル
    """
    
    def __init__(
        self,
        model: nn.Module,
        static_scaler,
        dynamic_scaler,
        config: Dict,
        device: str = "cuda",
        param_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: 学習済みLSTMモデル
            static_scaler: 静的パラメータのスケーラー
            dynamic_scaler: 動的特徴量のスケーラー
            config: 逆推定の設定
            device: 計算デバイス
        """
        self.model = model
        self.model.eval()  # 評価モード
        self.model.train()  # LSTMのbackwardのためtraining modeに設定
        
        # モデルの重みを固定
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.static_scaler = static_scaler
        self.dynamic_scaler = dynamic_scaler
        self.config = config
        self.device = device
        self.param_names = param_names or []
        self.param_name_to_idx = {name: idx for idx, name in enumerate(self.param_names)}
        
        # 設定の取得
        self.lr = config.get("lr", 0.001)  # より小さい学習率
        self.max_iters = config.get("max_iters", 300)  # 短めに
        self.lambda_prior = config.get("lambda_prior", 10.0)  # より強い正則化
        self.target_variable = config.get("target_variable", "nep")
        self.clip_enabled = config.get("clip_raw_params", {}).get("enabled", True)
        self._setup_param_bounds()
        
        print("="*70)
        print("StaticParameterInverter initialized")
        print("="*70)
        print(f"Learning rate: {self.lr}")
        print(f"Max iterations: {self.max_iters}")
        print(f"Prior regularization: λ={self.lambda_prior}")
        print(f"Target variable: {self.target_variable.upper()}")
        print(f"Parameter clipping: {self.clip_enabled}")
        print("="*70)
    
    def normalize_params(self, params_raw: np.ndarray) -> np.ndarray:
        """パラメータを正規化"""
        return self.static_scaler.transform(params_raw.reshape(1, -1))[0]
    
    def denormalize_params(self, params_norm: np.ndarray) -> np.ndarray:
        """パラメータを非正規化"""
        return self.static_scaler.inverse_transform(params_norm.reshape(1, -1))[0]
    
    def _setup_param_bounds(self) -> None:
        bounds_cfg = self.config.get("param_bounds")
        if not bounds_cfg or not self.param_name_to_idx:
            self.bounds_min = None
            self.bounds_max = None
            self.bounds_depend = {}
            self.has_explicit_bounds = False
            return

        num_params = len(self.param_name_to_idx)
        self.bounds_min = np.full(num_params, -np.inf, dtype=np.float64)
        self.bounds_max = np.full(num_params, np.inf, dtype=np.float64)
        self.bounds_depend = {}
        applied = False

        for name, rule in bounds_cfg.items():
            idx = self.param_name_to_idx.get(name)
            if idx is None or rule is None:
                continue
            if "min" in rule:
                self.bounds_min[idx] = rule["min"]
                applied = True
            if "max" in rule:
                self.bounds_max[idx] = rule["max"]
                applied = True
            if "max_from" in rule:
                ref_name = rule["max_from"]
                ref_idx = self.param_name_to_idx.get(ref_name)
                if ref_idx is None:
                    raise ValueError(f"max_from references unknown parameter '{ref_name}'")
                self.bounds_depend[idx] = ref_idx
                applied = True

        if not applied:
            self.bounds_min = None
            self.bounds_max = None
            self.bounds_depend = {}
        self.has_explicit_bounds = applied

    def apply_param_constraints(self, params_raw: np.ndarray, params_prior_raw: np.ndarray) -> np.ndarray:
        """
        パラメータに制約を適用（簡易版）
        
        各パラメータを事前値の±30%の範囲にクリップ。
        """
        if not self.clip_enabled:
            return params_raw

        if getattr(self, "has_explicit_bounds", False):
            clipped = params_raw.copy()
            if self.bounds_min is not None:
                clipped = np.maximum(clipped, self.bounds_min)
            if self.bounds_max is not None:
                clipped = np.minimum(clipped, self.bounds_max)
            for idx, ref_idx in self.bounds_depend.items():
                clipped[idx] = min(clipped[idx], clipped[ref_idx])
            return clipped

        # 事前値から±30%の範囲にクリップ（従来挙動）
        min_vals = params_prior_raw * 0.7
        max_vals = params_prior_raw * 1.3
        return np.clip(params_raw, min_vals, max_vals)
    
    def compute_loss(
        self,
        params_norm: torch.Tensor,
        params_prior_norm: torch.Tensor,
        dynamic_input: torch.Tensor,
        future_known: torch.Tensor,
        y_obs: torch.Tensor,
        obs_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        損失を計算
        
        Args:
            params_norm: 正規化された静的パラメータ (static_dim,)
            params_prior_norm: 事前値(正規化) (static_dim,)
            dynamic_input: コンテキスト期間の動的入力 (batch, context_len, dynamic_dim)
            future_known: 予測期間の既知入力 (batch, prediction_len, dynamic_dim)
            y_obs: 観測データ (batch, prediction_len, 4) [GPP, NPP, ER, NEP]
            obs_mask: 観測マスク (batch, prediction_len)
        
        Returns:
            total_loss: 総損失
            loss_dict: 各損失成分の辞書
        """
        batch_size = dynamic_input.shape[0]
        context_x = dynamic_input
        future_x = future_known
        static_x = params_norm.unsqueeze(0).repeat(batch_size, 1)

        # モデルで予測
        predictions = self.model(context_x, static_x, future_x)  # (batch, prediction_len, 4)

        # NEPのみを使用（index=3）
        pred_nep = predictions[:, :, 3]
        obs_nep = y_obs[:, :, 3]
        
        # 観測損失（マスクされた時点のみ）
        if obs_mask.sum() > 0:
            masked_pred = pred_nep[obs_mask]
            masked_obs = obs_nep[obs_mask]
            loss_obs = torch.mean((masked_pred - masked_obs) ** 2)
        else:
            loss_obs = torch.tensor(0.0, device=self.device)
        
        # 事前値からの乖離正則化（unconstrainedの場合はlambda=0として計算）
        # ただし、params_prior_normが同じものを指すようにしているので、実質0になる
        loss_prior = self.lambda_prior * torch.mean((params_norm - params_prior_norm) ** 2)
        
        # 総損失
        total_loss = loss_obs + loss_prior
        
        loss_dict = {
            "total": total_loss.item(),
            "obs": loss_obs.item(),
            "prior": loss_prior.item()
        }
        
        return total_loss, loss_dict
    
    def invert(
        self,
        params_prior_raw: np.ndarray,
        dynamic_input: np.ndarray,
        future_known: np.ndarray,
        y_obs: np.ndarray,
        obs_mask: Optional[np.ndarray] = None,
        unconstrained: bool = False,
        record_history: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        静的パラメータを逆推定
        
        Args:
            params_prior_raw: 事前パラメータ (static_dim,) 物理空間
            dynamic_input: コンテキスト期間の動的入力 (batch, context_len, dynamic_dim) 正規化済み
            future_known: 予測期間の既知入力 (batch, prediction_len, dynamic_dim) 正規化済み
            y_obs: 観測フラックス (batch, prediction_len, 4)
            obs_mask: 観測マスク (batch, prediction_len) Noneの場合は全て使用
            unconstrained: Trueの場合、全ての制約を外す
        
        Returns:
            params_opt_raw: 推定されたパラメータ (static_dim,) 物理空間
            info: 推定情報(損失履歴など)
        """
        print("\n" + "="*70)
        if unconstrained:
            print("Starting parameter inversion (UNCONSTRAINED)...")
            print("  ✗ Gradient clipping: OFF")
            print("  ✗ Parameter bounds: OFF")
            print("  ✗ Prior regularization: lambda=0.0")
        else:
            print("Starting parameter inversion...")
        print("="*70)
        
        # 観測マスクの処理
        if obs_mask is None:
            obs_mask = np.ones(y_obs.shape[:-1], dtype=bool)

        obs_valid = int(obs_mask.sum())
        obs_total = int(obs_mask.size)
        print(f"Observations: {obs_valid}/{obs_total} time steps")
        
        # 正規化
        params_prior_norm = self.normalize_params(params_prior_raw)
        
        # Tensorに変換
        params_norm = torch.tensor(
            params_prior_norm.copy(), 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        params_prior_norm_t = torch.tensor(
            params_prior_norm, 
            dtype=torch.float32, 
            device=self.device
        )
        dynamic_input_t = torch.tensor(
            dynamic_input, 
            dtype=torch.float32, 
            device=self.device
        )
        future_known_t = torch.tensor(
            future_known, 
            dtype=torch.float32, 
            device=self.device
        )
        y_obs_t = torch.tensor(
            y_obs, 
            dtype=torch.float32, 
            device=self.device
        )
        obs_mask_t = torch.tensor(
            obs_mask, 
            dtype=torch.bool, 
            device=self.device
        )
        
        # オプティマイザ
        optimizer = torch.optim.Adam([params_norm], lr=self.lr)
        param_history = []
        
        # 履歴
        loss_history = []
        best_loss = float('inf')
        best_params = params_norm.detach().clone()

        if record_history:
            with torch.no_grad():
                param_history.append(self.denormalize_params(params_norm.detach().cpu().numpy()))
        
        # 最適化ループ
        print("\nOptimization progress:")
        print("-"*70)
        
        for iteration in range(self.max_iters):
            optimizer.zero_grad()
            
            # 損失計算
            loss, loss_dict = self.compute_loss(
                params_norm,
                params_prior_norm_t if not unconstrained else params_norm,  # 正則化なしの場合はpriorを無視
                dynamic_input_t,
                future_known_t,
                y_obs_t,
                obs_mask_t
            )
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング（unconstrainedの場合はスキップ）
            if not unconstrained:
                torch.nn.utils.clip_grad_norm_([params_norm], max_norm=1.0)
            
            optimizer.step()
            
            # パラメータ制約の適用（unconstrainedの場合はスキップ）
            if not unconstrained:
                with torch.no_grad():
                    params_raw_current = self.denormalize_params(params_norm.cpu().numpy())
                    params_raw_clipped = self.apply_param_constraints(params_raw_current, params_prior_raw)
                    params_norm_clipped = self.normalize_params(params_raw_clipped)
                    params_norm.copy_(torch.tensor(params_norm_clipped, device=self.device))
            
            if record_history:
                with torch.no_grad():
                    param_history.append(self.denormalize_params(params_norm.detach().cpu().numpy()))
            
            # 履歴記録
            loss_history.append(loss_dict)
            
            # ベスト更新
            if loss_dict["total"] < best_loss:
                best_loss = loss_dict["total"]
                best_params = params_norm.detach().clone()
            
            # 進捗表示
            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"Iter {iteration+1:4d}/{self.max_iters}: "
                      f"Loss={loss_dict['total']:.6f} "
                      f"(Obs={loss_dict['obs']:.6f}, Prior={loss_dict['prior']:.6f})")
        
        print("-"*70)
        print(f"Optimization completed. Best loss: {best_loss:.6f}")
        print("="*70)
        
        # 最良パラメータを物理空間に戻す
        params_opt_raw = self.denormalize_params(best_params.cpu().numpy())
        
        # パラメータ変化量
        param_change = np.abs(params_opt_raw - params_prior_raw)
        # ゼロ除算を避ける
        param_change_pct = 100 * param_change / (np.abs(params_prior_raw) + 1e-8)
        
        print(f"\nParameter changes:")
        print(f"  Mean absolute change: {np.mean(param_change):.6f}")
        print(f"  Max absolute change: {np.max(param_change):.6f}")
        print(f"  Mean relative change: {np.mean(param_change_pct):.2f}%")
        print(f"  Max relative change: {np.max(param_change_pct):.2f}%")
        
        info = {
            "loss_history": loss_history,
            "final_loss": best_loss,
            "iterations": self.max_iters,
            "param_change_abs": param_change,
            "param_change_pct": param_change_pct
        }
        if record_history:
            info["param_history_raw"] = np.stack(param_history, axis=0)
        
        return params_opt_raw, info


def load_inversion_config(config_path: Optional[Path] = None) -> Dict:
    """逆推定の設定を読み込み"""
    default_config = {
        "lr": 0.001,  # 控えめな学習率
        "max_iters": 300,
        "lambda_prior": 10.0,  # 強めの正則化
        "target_variable": "nep",
        "use_mask": True,
        "clip_raw_params": {
            "enabled": True
        },
        "param_bounds": None
    }
    
    if config_path and config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        overrides = user_config.get("inversion", {})

        bounds_file = overrides.pop("bounds_file", None)
        if bounds_file:
            bounds_path = Path(bounds_file)
            if not bounds_path.is_absolute():
                bounds_path = (config_path.parent / bounds_path).resolve()
            overrides["param_bounds"] = load_parameter_bounds(bounds_path)
        default_config.update(overrides)
    
    return default_config


def load_parameter_bounds(bounds_path: Path) -> Dict[str, Dict[str, float]]:
    if not bounds_path.exists():
        raise FileNotFoundError(f"Parameter bounds file not found: {bounds_path}")
    import yaml
    with open(bounds_path, "r") as f:
        data = yaml.safe_load(f) or {}
    bounds = data.get("parameter_bounds", {})
    if not isinstance(bounds, dict):
        raise ValueError("parameter_bounds must be a mapping of parameter name -> constraints")
    return bounds
