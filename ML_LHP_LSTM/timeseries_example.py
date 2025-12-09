"""
timeseries_predictions.png を理解するための実例コード
実際のデータを使って各ステップを詳細に出力
"""

import numpy as np
import pickle
import torch
from pathlib import Path
from dataset import VISITTimeSeriesDataset
from model import create_model

# パス設定
BASE_DIR = Path("/mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM")
DATA_DIR = Path("/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS")

print("=" * 70)
print("timeseries_predictions.png 作成プロセスの詳細")
print("=" * 70)

# ステップ1: スケーラー読み込み
print("\n【ステップ1】スケーラー読み込み")
print("-" * 70)
with open(BASE_DIR / "data/scaler_dynamic.pkl", "rb") as f:
    scaler_dynamic = pickle.load(f)
with open(BASE_DIR / "data/scaler_static.pkl", "rb") as f:
    scaler_static = pickle.load(f)

print(f"✓ 動的スケーラー: {scaler_dynamic.n_features_in_} 特徴量")
print(f"  平均値: {scaler_dynamic.mean_[:3]} ... (最初の3要素)")
print(f"  標準偏差: {scaler_dynamic.scale_[:3]} ... (最初の3要素)")
print(f"\n✓ 静的スケーラー: {scaler_static.n_features_in_} パラメータ")

# ステップ2: テストデータセット作成
print("\n【ステップ2】テストデータセット作成")
print("-" * 70)
test_dataset = VISITTimeSeriesDataset(
    data_dir=str(DATA_DIR),
    split="test",
    context_len=180,
    prediction_len=30,
    scaler_dynamic=scaler_dynamic,
    scaler_static=scaler_static,
    fit_scaler=False
)
print(f"✓ テストウィンドウ数: {len(test_dataset)}")

# ステップ3: ランダムサンプル選択
print("\n【ステップ3】ランダムサンプル選択")
print("-" * 70)
np.random.seed(42)  # 再現性のため
sample_idx = np.random.choice(len(test_dataset), 1)[0]
print(f"✓ 選択されたウィンドウインデックス: {sample_idx}")

# ステップ4: データ取得
print("\n【ステップ4】データ取得")
print("-" * 70)
sample = test_dataset[sample_idx]
print(f"✓ context_x shape: {sample['context_x'].shape}")
dynamic_dim = sample['context_x'].shape[-1]
static_dim = sample['static_x'].shape[0]
future_dim = sample['future_known'].shape[-1]
print(f"  → 過去180日分の動的特徴 ({dynamic_dim} 次元: 気象9 + aCO2)")
print(f"\n✓ static_x shape: {sample['static_x'].shape}")
print(f"  → {static_dim} 個の静的パラメータ")
print(f"\n✓ future_known shape: {sample['future_known'].shape}")
print(f"  → 未来30日分の動的特徴 ({future_dim} 次元)")
print(f"\n✓ target_y shape: {sample['target_y'].shape}")
print(f"  → 未来30日分の [GPP, NPP, ER, NEP] = 4フラックス")

# ステップ5: データ内容の詳細
print("\n【ステップ5】データ内容の詳細")
print("-" * 70)
print("target_y（実測値）:")
target = sample['target_y'].numpy()
print(f"  最初の5日分:")
print(f"    GPP: {target[:5, 0]}")
print(f"    NPP: {target[:5, 1]}")
print(f"    ER:  {target[:5, 2]}")
print(f"    NEP: {target[:5, 3]}")

# ステップ6: モデル読み込み
print("\n【ステップ6】モデル読み込みと予測")
print("-" * 70)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_model(
    dynamic_dim=dynamic_dim,
    static_dim=static_dim,
    hidden_size=256,
    num_layers=2,
    dropout=0.1,
    output_dim=4,
    device=device
)
checkpoint = torch.load(
    BASE_DIR / "artifacts/checkpoint_best.pt",
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ モデル読み込み完了 (Epoch {checkpoint['epoch']})")

# ステップ7: 予測実行
print("\n【ステップ7】予測実行")
print("-" * 70)
with torch.no_grad():
    context_x = sample['context_x'].unsqueeze(0).to(device)
    static_x = sample['static_x'].unsqueeze(0).to(device)
    future_known = sample['future_known'].unsqueeze(0).to(device)
    
    print(f"入力shape:")
    print(f"  context_x: {context_x.shape}  # (batch=1, time=180, features={dynamic_dim})")
    print(f"  static_x: {static_x.shape}   # (batch=1, params={static_dim})")
    print(f"  future_known: {future_known.shape}  # (batch=1, time=30, features={future_dim})")
    
    prediction = model(context_x, static_x, future_known)
    prediction = prediction.cpu().numpy()[0]
    
    print(f"\n出力shape: {prediction.shape}  # (time=30, fluxes=4)")

# ステップ8: 予測結果
print("\n【ステップ8】予測結果")
print("-" * 70)
print(f"予測値（最初の5日分）:")
print(f"  GPP: {prediction[:5, 0]}")
print(f"  NPP: {prediction[:5, 1]}")
print(f"  ER:  {prediction[:5, 2]}")
print(f"  NEP: {prediction[:5, 3]}")

# ステップ9: 精度評価
print("\n【ステップ9】このサンプルの精度")
print("-" * 70)
for i, flux_name in enumerate(['GPP', 'NPP', 'ER', 'NEP']):
    rmse = np.sqrt(np.mean((prediction[:, i] - target[:, i])**2))
    mae = np.mean(np.abs(prediction[:, i] - target[:, i]))
    print(f"{flux_name}: RMSE={rmse:.6f}, MAE={mae:.6f}")

# ステップ10: プロットデータ
print("\n【ステップ10】プロット用の時間軸")
print("-" * 70)
context_days = np.arange(-180, 0)
pred_days = np.arange(0, 30)
print(f"コンテキスト期間: {context_days[0]} ~ {context_days[-1]} 日")
print(f"予測期間: {pred_days[0]} ~ {pred_days[-1]} 日")
print(f"\n縦線（X=0）が予測開始点を示します")

print("\n" + "=" * 70)
print("これらのデータを使って timeseries_predictions.png が作成されます")
print("=" * 70)
