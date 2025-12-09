"""
静的パラメータ逆推定のテスト実行

NEPのみを使って1サンプルの逆推定を行い、結果を可視化する。
"""

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from model import create_model
from dataset import VISITTimeSeriesDataset
from inverse_estimator import StaticParameterInverter, load_inversion_config

# パス設定
BASE_DIR = Path("/mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM")
DATA_DIR = Path("/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
VIS_DIR = BASE_DIR / "inverse_results_unconstrained"
VIS_DIR.mkdir(exist_ok=True)

# 設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SAMPLE_ID = 0  # テスト用サンプルID (0〜179)
TEST_WINDOW_START = 100  # テストウィンドウの開始インデックス

print("="*70)
print("Static Parameter Inversion Test (UNCONSTRAINED)")
print("="*70)
print("\n【制約設定】")
print("  ✗ パラメータ範囲制約: なし")
print("  ✗ 勾配クリッピング: なし")
print("  ✗ 事前値正則化: なし")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Test sample ID: {TEST_SAMPLE_ID}")
print("="*70)

# ステップ1: モデルとスケーラー読み込み
print("\n[Step 1] Loading model and scalers...")
print("-"*70)

# 設定読み込み
import json
with open(ARTIFACTS_DIR / "config.json", 'r') as f:
    config = json.load(f)

# モデル作成
model = create_model(
    dynamic_dim=config["dynamic_dim"],
    static_dim=config["static_dim"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    dropout=config["dropout"],
    output_dim=config["output_dim"],
    device=DEVICE
)

# チェックポイント読み込み
checkpoint = torch.load(
    ARTIFACTS_DIR / "checkpoint_best.pt",
    map_location=DEVICE,
    weights_only=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded (Epoch {checkpoint['epoch']})")

# スケーラー読み込み
with open(BASE_DIR / "data/scaler_dynamic.pkl", "rb") as f:
    scaler_dynamic = pickle.load(f)
with open(BASE_DIR / "data/scaler_static.pkl", "rb") as f:
    scaler_static = pickle.load(f)

print(f"✓ Scalers loaded")

# ステップ2: テストデータ準備
print("\n[Step 2] Preparing test data...")
print("-"*70)

# テストデータセット作成
test_dataset = VISITTimeSeriesDataset(
    data_dir=str(DATA_DIR),
    split="test",
    context_len=config["context_len"],
    prediction_len=config["prediction_len"],
    scaler_dynamic=scaler_dynamic,
    scaler_static=scaler_static,
    fit_scaler=False
)

print(f"✓ Test dataset loaded: {len(test_dataset)} windows")

# 特定のウィンドウを選択
test_window = test_dataset[TEST_WINDOW_START]
print(f"✓ Selected window {TEST_WINDOW_START}")

# パラメータサマリー読み込み（物理空間の事前値）
param_summary = np.loadtxt(
    DATA_DIR / "parameter_summary.csv",
    delimiter=',',
    skiprows=1
)
params_prior_raw = param_summary[TEST_SAMPLE_ID, 1:]  # 最初の列はsample_id

print(f"✓ Prior parameters loaded: shape={params_prior_raw.shape}")

# データ抽出
context_x = test_window['context_x'].numpy()
future_known = test_window['future_known'].numpy()
target_y = test_window['target_y'].numpy()  # (30, 4)

dynamic_dim = context_x.shape[-1]
future_dim = future_known.shape[-1]
print(f"✓ Context shape: {context_x.shape}  # dynamic_dim={dynamic_dim}")
print(f"✓ Future known shape: {future_known.shape}  # future_dim={future_dim}")
print(f"✓ Target shape: {target_y.shape}")

# 観測データとして target_y を使用（実験）
# 実際の応用では観測サイトのデータを使う
y_obs = target_y
obs_mask = np.ones(len(y_obs), dtype=bool)  # 全時点で観測あり

print(f"✓ Observation data prepared")
print(f"  NEP range: [{y_obs[:, 3].min():.4f}, {y_obs[:, 3].max():.4f}]")

# ステップ3: 逆推定実行
print("\n[Step 3] Running parameter inversion...")
print("-"*70)

# 逆推定設定
inversion_config = load_inversion_config()
print(f"Inversion config: {inversion_config}")

# Inverter初期化
inverter = StaticParameterInverter(
    model=model,
    static_scaler=scaler_static,
    dynamic_scaler=scaler_dynamic,
    config=inversion_config,
    device=DEVICE
)

# 逆推定実行（制約なし）
params_opt_raw, info = inverter.invert(
    params_prior_raw=params_prior_raw,
    dynamic_input=context_x,
    future_known=future_known,
    y_obs=y_obs,
    obs_mask=obs_mask,
    unconstrained=True  # 制約を全て外す
)

print("\n✓ Inversion completed!")

# ステップ4: 結果評価
print("\n[Step 4] Evaluating results...")
print("-"*70)

# 事前パラメータで予測
params_prior_norm = scaler_static.transform(params_prior_raw.reshape(1, -1))[0]
with torch.no_grad():
    context_x_t = torch.tensor(context_x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    params_prior_t = torch.tensor(params_prior_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    future_known_t = torch.tensor(future_known, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    
    pred_prior = model(context_x_t, params_prior_t, future_known_t)
    pred_prior = pred_prior.cpu().numpy()[0]

# 最適パラメータで予測
params_opt_norm = scaler_static.transform(params_opt_raw.reshape(1, -1))[0]
with torch.no_grad():
    params_opt_t = torch.tensor(params_opt_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    
    pred_opt = model(context_x_t, params_opt_t, future_known_t)
    pred_opt = pred_opt.cpu().numpy()[0]

# RMSE計算
rmse_prior = np.sqrt(np.mean((pred_prior[:, 3] - y_obs[:, 3])**2))
rmse_opt = np.sqrt(np.mean((pred_opt[:, 3] - y_obs[:, 3])**2))

print(f"NEP RMSE:")
print(f"  Prior:     {rmse_prior:.6f}")
print(f"  Optimized: {rmse_opt:.6f}")
print(f"  Improvement: {100*(rmse_prior - rmse_opt)/rmse_prior:.2f}%")

# ステップ5: 可視化
print("\n[Step 5] Creating visualizations...")
print("-"*70)

# 図1: NEP予測比較
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

days = np.arange(config["prediction_len"])

# 上段: NEP時系列
ax = axes[0]
ax.plot(days, y_obs[:, 3], 'g-', linewidth=2.5, label='Observation', alpha=0.8)
ax.plot(days, pred_prior[:, 3], 'b--', linewidth=2, label=f'Prior (RMSE={rmse_prior:.6f})', alpha=0.7)
ax.plot(days, pred_opt[:, 3], 'r--', linewidth=2, label=f'Optimized (RMSE={rmse_opt:.6f})', alpha=0.7)
ax.set_xlabel('Days ahead', fontsize=12, fontweight='bold')
ax.set_ylabel('NEP (gC/m²/day)', fontsize=12, fontweight='bold')
ax.set_title('NEP Prediction: Before vs After Parameter Inversion', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

# 下段: 誤差
ax = axes[1]
error_prior = pred_prior[:, 3] - y_obs[:, 3]
error_opt = pred_opt[:, 3] - y_obs[:, 3]
ax.plot(days, error_prior, 'b-', linewidth=2, label='Prior error', alpha=0.7)
ax.plot(days, error_opt, 'r-', linewidth=2, label='Optimized error', alpha=0.7)
ax.axhline(0, color='black', linestyle=':', linewidth=1)
ax.fill_between(days, 0, error_prior, alpha=0.2, color='blue')
ax.fill_between(days, 0, error_opt, alpha=0.2, color='red')
ax.set_xlabel('Days ahead', fontsize=12, fontweight='bold')
ax.set_ylabel('Prediction Error (gC/m²/day)', fontsize=12, fontweight='bold')
ax.set_title('Prediction Errors', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIS_DIR / "nep_comparison.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: nep_comparison.png")
plt.close()

# 図2: 損失履歴
fig, ax = plt.subplots(figsize=(12, 6))
iterations = range(len(info['loss_history']))
total_losses = [h['total'] for h in info['loss_history']]
obs_losses = [h['obs'] for h in info['loss_history']]
prior_losses = [h['prior'] for h in info['loss_history']]

ax.plot(iterations, total_losses, 'k-', linewidth=2.5, label='Total Loss', alpha=0.8)
ax.plot(iterations, obs_losses, 'g-', linewidth=2, label='Observation Loss', alpha=0.7)
ax.plot(iterations, prior_losses, 'b-', linewidth=2, label='Prior Loss', alpha=0.7)
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Loss History During Parameter Inversion', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(VIS_DIR / "loss_history.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: loss_history.png")
plt.close()

# 図3: パラメータ変化
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

param_indices = np.arange(len(params_prior_raw))
param_change_abs = info['param_change_abs']
param_change_pct = info['param_change_pct']

# 上段: 絶対変化量
ax = axes[0]
bars = ax.bar(param_indices, param_change_abs, alpha=0.7, edgecolor='black')
ax.set_xlabel('Parameter Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Absolute Change', fontsize=12, fontweight='bold')
ax.set_title('Absolute Parameter Changes', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 下段: 相対変化量(%)
ax = axes[1]
bars = ax.bar(param_indices, param_change_pct, alpha=0.7, edgecolor='black', color='orange')
ax.set_xlabel('Parameter Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative Change (%)', fontsize=12, fontweight='bold')
ax.set_title('Relative Parameter Changes', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VIS_DIR / "parameter_changes.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: parameter_changes.png")
plt.close()

# 結果サマリー保存
print("\n[Step 6] Saving results...")
print("-"*70)

with open(VIS_DIR / "inversion_summary.txt", "w") as f:
    f.write("="*70 + "\n")
    f.write("Parameter Inversion Test Results\n")
    f.write("="*70 + "\n\n")
    f.write(f"Test Sample ID: {TEST_SAMPLE_ID}\n")
    f.write(f"Test Window: {TEST_WINDOW_START}\n")
    f.write(f"Prediction Length: {config['prediction_len']} days\n\n")
    f.write("--- NEP RMSE ---\n")
    f.write(f"Prior:     {rmse_prior:.6f}\n")
    f.write(f"Optimized: {rmse_opt:.6f}\n")
    f.write(f"Improvement: {100*(rmse_prior - rmse_opt)/rmse_prior:.2f}%\n\n")
    f.write("--- Parameter Changes ---\n")
    f.write(f"Mean absolute change: {np.mean(param_change_abs):.6f}\n")
    f.write(f"Max absolute change:  {np.max(param_change_abs):.6f}\n")
    f.write(f"Mean relative change: {np.mean(param_change_pct):.2f}%\n")
    f.write(f"Max relative change:  {np.max(param_change_pct):.2f}%\n\n")
    f.write("--- Optimization ---\n")
    f.write(f"Iterations: {info['iterations']}\n")
    f.write(f"Final loss: {info['final_loss']:.6f}\n")
    f.write("="*70 + "\n")

print(f"✓ Saved: inversion_summary.txt")

print("\n" + "="*70)
print("Test completed successfully!")
print(f"Results saved in: {VIS_DIR}")
print("="*70)
