# ML_LHP_LSTM ディレクトリ構造

## 概要
VISIT生態系モデルの36パラメータ摂動実験(LHP: Latin Hypercube Perturbation)の結果を学習したLSTMエミュレータ

## ディレクトリ構造

```
ML_LHP_LSTM/
├── README.md                   # プロジェクト概要
├── STRUCTURE.md               # このファイル（ディレクトリ構造説明）
├── requirements.txt           # Python依存パッケージ
│
├── model.py                   # LSTMモデル定義
├── dataset.py                 # データローダー
├── train.py                   # 学習スクリプト
├── evaluate_model.py          # 評価・可視化スクリプト
├── test_setup.py             # セットアップテスト
│
├── configs/                   # 設定ファイル
│   └── LHP_architecture.yaml  # モデルアーキテクチャ設定
│
├── data/                      # スケーラー保存先
│   ├── scaler_dynamic.pkl     # 動的特徴量のスケーラー
│   └── scaler_static.pkl      # 静的パラメータのスケーラー
│
├── artifacts/                 # 学習済みモデル
│   ├── checkpoint_best.pt     # 最良モデル (Epoch 15, Val Loss: 8.9e-5)
│   ├── checkpoint_latest.pt   # 最終モデル (Epoch 50)
│   ├── config.json           # 学習時の設定
│   └── training_history.json # 学習履歴
│
├── visualizations/            # 評価結果の可視化
│   ├── learning_curves.png           # 学習曲線
│   ├── test_metrics.png              # テスト性能メトリクス
│   ├── prediction_vs_actual.png      # 予測vs実測
│   ├── timeseries_predictions.png    # 時系列予測
│   ├── error_distribution.png        # 誤差分布
│   └── evaluation_report.txt         # 評価レポート
│
├── training.log               # 学習ログ (321KB)
├── logs/                      # 追加ログ保存先
├── models/                    # （空）将来のモデルバリエーション用
│
└── old_dataset_pytorch/       # 旧データセット（参考用、1.6GB）
    ├── visit_pytorch_metadata.json
    ├── visit_pytorch_scalers.pkl
    ├── visit_pytorch_train_X.npy
    ├── visit_pytorch_train_y.npy
    ├── visit_pytorch_val_X.npy
    ├── visit_pytorch_val_y.npy
    ├── visit_pytorch_test_X.npy
    └── visit_pytorch_test_y.npy

```

## データフロー

### 入力データ
- **データソース**: `/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS/`
  - 180サンプル × 3652日 (2010-2019)
  - 各サンプル: daily出力 (time列 + 気象9 + aCO2 + flux4 + param36)
  - `parameter_summary.csv` (36パラメータ + sample_id)

### 学習データ
- **Train**: 2010-2017 (8年間)
- **Validation**: 2018 (1年間)
- **Test**: 2019 (1年間)

### モデル
- **入力**:
  - コンテキスト期間: 180日
  - 動的特徴量: 10次元 (気象9 + aCO2)
  - 静的パラメータ: 36次元
- **出力**:
  - 予測期間: 30日
  - フラックス: 4変数 (GPP, NPP, ER, NEP)
- **アーキテクチャ**:
  - LSTM: 256 hidden units × 2 layers
  - パラメータ数: 1,042,820

static_x ──┐
           ├─ Static MLP ─────────────┐
context_x ─┼─ concat(+static_emb) ── LSTM layers ──┐
future_known ───────────────┘                      │
                                                   ▼
                 ┌────────────┬────────────┬────────────┬────────────┐
                 │ GPP head   │ NPP head   │ Rh head    │ NEP head    │
                 └────────────┴────────────┴────────────┴────────────┘
                     (各head: hidden→128→1、NEPはRhとNPPから整合算出可)

## 使用方法

### 1. 学習
```bash
cd /mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM
conda activate deepl
python train.py
```

### 2. 評価・可視化
```bash
python evaluate_model.py
```

### 3. 予測
```python
from model import create_model
import torch

# モデル読み込み
model = create_model(...)
checkpoint = torch.load("artifacts/checkpoint_best.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 性能

### テスト性能 (再学習予定)

OUTPUT_PERTURBATION_LHP_V2_LHS (180サンプル) 向けの再学習をこれから実施するため、最新メトリクスは未確定です。旧100サンプルバッチでの参考値は `visualizations/evaluation_report.txt` を参照してください。

## ファイルサイズ

| フォルダ/ファイル | サイズ | 説明 |
|------------------|--------|------|
| old_dataset_pytorch | 1.6GB | 旧データ（参考用） |
| artifacts | 24MB | 学習済みモデル |
| visualizations | 1.5MB | 評価結果の図 |
| training.log | 321KB | 学習ログ |
| その他 | ~100KB | スクリプト等 |

## 環境

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **Conda環境**: deepl
- **GPU**: CUDA対応 (オプション)

## 関連ファイル

- **摂動実験**: `/mnt/d/VISIT/honban/point/ex/LHP_parameter_perturbation/scripts/run_perturbations.py`
- **可視化**: `/mnt/d/VISIT/honban/point/ex/visualize_perturbation_detailed.py`
- **解析**: `/mnt/d/VISIT/honban/point/ex/analyze_perturbation_results.py`

## 更新履歴

- 2025-11-13: ディレクトリ構造を整理、visualizationsとold_dataset_pytorchを統合
- 2025-11-12: 初回学習完了 (50 epochs, best at epoch 15)
- 2025-09-19: 36パラメータ摂動実験完了
