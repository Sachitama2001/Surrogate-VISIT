# VISIT LSTM Emulator with Static Parameter Conditioning

VISIT生態系モデルのLSTMエミュレータ(静的パラメータconditioning付き)

## プロジェクト概要

このプロジェクトは、VISITモデル(LHP熱帯林サイト)の入出力をLSTMで模倣し、日次のGPP/NPP/NEP/Rhを多ホライズン予測するための深層学習モデルです。

### 主な特徴

- **静的パラメータconditioning**: 最新の parameter_summary.csv (36列) を静的特徴として使用
  - `init_state`: MLPでLSTMの初期hidden stateを生成
  - `concat_input`: 埋め込みベクトルを動的入力と結合
- **ハイブリッド損失関数**: 0.7×one-step MSE + 0.3×rollout MSE@30
- **Scheduled sampling**: teacher forcing確率を1.0→0.3に線形減衰
- **既知未来入力**: 気象データ(9変数) + aCO2濃度を使用

## データセット

- **サンプル数**: 180個の摂動実験 (各サンプルは2010-2019の連続系列)
- **時系列長**: 3652日 (2010-01-01〜2019-12-31の日次データ)
- **入力特徴**:
  - 動的特徴 (10次元): 気象(9) + aCO2(1)
  - 静的特徴 (36次元): Tree(15), C3(15), Soil(6)
- **出力**: GPP, NPP, NEP, Rh (4変数)

### データ分割

- **Train**: 2010-2017 (2922日)
- **Validation**: 2018 (365日)
- **Test**: 2019 (365日)

## 観測データ (逆推定用)

- **ソース**: `ex/LHP_observation/FLX_MY-LHP_JapanFLUX2024_ALLVARS_DD_2010-2020_1-3.csv` (JapanFlux 日次フラックス, 2010-01-01〜2019-12-31, 3652行)
- **NEPの定義**: フラックスの符号規約は NEE>0 が大気への放出、NEE<0 が吸収なので、NEP は `NEP_obs = -1 × NEE_xxx / 100`。`prepare_observation_nep.py` の `--nee-column` で `NEE_vUT` (デフォルト) か `NEE_PI_F` など任意カラムを選択できる。
- **欠損とQC**:
  - `NEE_vUT` および `NEE_vUT_USTAR{05,50,95}` は -9999 が欠損。全期間で約32%が欠損。
  - 年次の有効率: 2012-2016は100%、2017は69%、2018は30%、2019は0% (終盤は観測欠落)。2010-2011も40%前後と低いため、逆推定は基本的に2012-2016のウィンドウを選ぶ。
  - `NEE_vUT_QC` は 0〜1 のgap-filling率。0.5以下を優先し、それ以外は `obs_mask` で除外する運用を推奨。
- **前処理メモ**:
  - `TIMESTAMP` は YYYYMMDD (ローカル日付) なので `pd.to_datetime(format="%Y%m%d")` 等でUTC補正なしに解釈する。
  - 欠損日は `obs_mask=False` とし、`y_obs` の該当行は任意値でも構わない (損失計算で無視される)。
  - ダイナミック入力 `context_x`/`future_known` は VISIT訓練と同じスケーリングを適用する必要がある。観測期間の気象は `OUTPUT_PERTURBATION_LHP_V2_LHS/*_daily_*.txt` か `CODE_LHP/INPUT/LHP_ERA5.txt` から抽出し、既存の `scaler_dynamic.pkl` で正規化する。
  - 逆推定ツール (`inverse_estimator.py`) では `params_prior_raw` に `parameter_summary.csv` の該当行を入れる。観測サイト固有の初期値が無い場合は平均ベクトルを使い、クリップ範囲±30%を超えないよう注意する。

> **参考統計** (NEE_vUT, 2010-2019)
> - 平均 -2.71 gC/m²/day, 標準偏差 2.05, 最小 -9.35, 最大 7.57
> - 欠損の最長連続 556 日 (2018/2019 に跨る)。連続ウィンドウ抽出時はギャップ判定を必ず実施する。

### フラックスの単位について

- VISIT本体の構造体定義 (`ex/CODE_LHP/structure.h`) では `Flux` の `gpp/npp/er/nep` など主要フラックスが **Mg C ha⁻¹ day⁻¹** で出力されることが明記されている。
- シミュレーション出力 (`OUTPUT_PERTURBATION_LHP_V2_LHS/LHP_*.txt`) および本LSTMの学習データも同じ単位をそのまま使用している。例: `gpp=0.10` は 0.10 Mg C ha⁻¹ day⁻¹ (= 10 g C m⁻² day⁻¹)。
- 変換係数: `1 Mg C ha⁻¹ day⁻¹ = 10⁶ g / 10⁴ m² = 100 g C m⁻² day⁻¹`。
- 観測データ（JapanFlux）は gC m⁻² day⁻¹ で出力されるため、逆推定に使う際は `NEP_obs_MgHa = NEP_obs_gpm2 / 100` に変換してから `inverse_estimator` に渡す。逆に、VISIT/LSTMの出力を観測と直接比較する場合は ×100 して gC m⁻² day⁻¹ に揃える。
- 観測NEP処理パイプライン:
  - `prepare_observation_nep.py --start 2012-01-01 --end 2016-12-31 --nee-column NEE_PI_F --nee-qc-column ''` のように、NEEソースとQC列を切り替えて `observations/nep_obs_2012_2016.csv` を生成（欠損・QCフィルタ済、`obs_mask`列付き）。
  - `build_observation_windows.py --daily ../OUTPUT_PERTURBATION_LHP_V2_LHS/LHP_20250919_daily_0000.txt --obs observations/nep_obs_2012_2016.csv` で VISIT気象を既存スケーラーで正規化し、観測対応の `(context_x, future_known, y_obs, obs_mask)` を `observations/obs_windows_2012_2016.npz` に生成。
  - NEE 列を PI ベースに切り替えた NPZ (`observations/obs_windows_2012_2016_pi.npz` など) も同じ構造で保存されるため、`test_inversion.py --obs-npz /mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM/observations/obs_windows_2012_2016_pi.npz ...` のように `--obs-npz` で差し替えるだけで PI 版データを使用できる。カスタム観測セットを追加した場合は、それぞれの NPZ を `--obs-npz` で指すだけで他のオプションは共通。
  - `test_inversion.py --mode observation --obs-index <n> --result-dir inverse_results/obs_2012_2016/window_<n>` を実行すると、観測ウィンドウに対する LSTM + StaticParameterInverter の結果を PNG/テキストで保存できる。`--obs-prior-sample` を省略した場合は parameter_summary の平均を事前値とする。
  - 例: `obs-index 0 (2012-01-01開始)` では有効観測 16/30 日、NEP RMSE が 0.0478 → 0.0464 MgC/ha/day (約2.8% 改善) を確認済み。成果物は `inverse_results/obs_2012_2016/window_000/` に配置。
  - `configs/inversion_config.yaml` で学習率・反復数・正則化・パラメータ境界を一括管理できる。`bounds_file: parameter_bounds.yaml` を指定すると、`range_of_paras.csv` 由来の min/max (一部は `max_from` で別パラメータに依存) を `StaticParameterInverter` が自動適用し、物理的な許容レンジを超えた更新を防げる。
  - `test_inversion.py` は複数の観測ウィンドウをまとめて投入できるようになり、2012年などの1年分を30日ウィンドウのスタックとして扱える。`--obs-range-start/--obs-range-end` でウィンドウ範囲を日付指定し、`--obs-range-step` で間引き、`--max-obs-windows` で上限を設けられる。内部では `(batch, 30, feature)` 形式に自動整形され、RMSE計算・可視化もバッチ全体を対象とする。
  - 例: 2012年全期間の127ウィンドウを使うには `python test_inversion.py --mode observation --obs-range-start 2012-01-01 --obs-range-end 2012-12-31 --result-dir inverse_results/obs_2012/full_year --config configs/inversion_config.yaml` を実行する。結果フォルダには全ウィンドウを連結した NEP 比較グラフ、損失履歴、パラメータ変化、RMSE/観測数サマリが出力される。
  - `--unconstrained` を付けるとパラメータクリッピングと事前正則化を完全に無効化し、`inverse_results/obs_2012_2016/window_000_unreg` と同じ条件を再現できる（物理範囲外ジャンプを許容するため、使用時は注意）。
  - **パラメータ推定→年間評価の2段階実験**: `test_inversion.py` は推定済みパラメータの保存/再利用に対応。単一ウィンドウで推定した後、`--param-load` でそのパラメータを固定し年間の観測NEPと突き合わせることができる。
    1. 最初の30日で推定し、結果を保存: `python test_inversion.py --mode observation --obs-index 0 --config configs/inversion_config_bounds.yaml --result-dir inverse_results/obs_2012/window000_stage`
    2. 保存された `inverse_results/obs_2012/window000_stage/params_snapshot.json` を読み込み、年間評価: `python test_inversion.py --mode observation --obs-range-start 2012-01-01 --obs-range-end 2012-12-31 --config configs/inversion_config_bounds.yaml --param-load inverse_results/obs_2012/window000_stage/params_snapshot.json --result-dir inverse_results/obs_2012/full_year_from_window0`
    - **PI版 + 2012-06-01開始ウィンドウの再現手順**: NEE_PI_F 由来の観測NPZを使い、6月開始ウィンドウ (index=152) で推定→年間比較を行う場合は下記コマンドを順に実行する。
      1. 30日ウィンドウの逆推定 (結果: `inverse_results/obs_2012_pi/window152_jun01/`):
         ```bash
         python test_inversion.py \
           --mode observation \
           --obs-npz observations/obs_windows_2012_2016_pi.npz \
           --obs-index 152 \
           --config configs/inversion_config_bounds.yaml \
           --result-dir inverse_results/obs_2012_pi/window152_jun01
         ```
      2. 推定パラメータで2012年全期間を評価 (結果: `inverse_results/obs_2012_pi/full_year_from_window152_jun01/`):
         ```bash
         python test_inversion.py \
           --mode observation \
           --obs-npz observations/obs_windows_2012_2016_pi.npz \
           --obs-range-start 2012-01-01 \
           --obs-range-end 2012-12-31 \
           --config configs/inversion_config_bounds.yaml \
           --param-load inverse_results/obs_2012_pi/window152_jun01/params_snapshot.json \
           --result-dir inverse_results/obs_2012_pi/full_year_from_window152_jun01
         ```
  - すべての実行結果ディレクトリには `parameter_values.csv`（prior/optimized/差分/割合）と `params_snapshot.json`（パラメータセット＋メタデータ）が自動で出力される。`--param-save` を指定すると保存先を任意パスに変更できる。

### 観測気象入力の準備メモ

今後、ERA5 の動的入力をサイト観測で置き換えるために、フラックス CSV (`LHP_observation/FLX_MY-LHP_JapanFLUX2024_ALLVARS_DD_2010-2020_1-3.csv`) に含まれる気象列と VISIT/LSTM が期待する 9 変数 ( + aCO2 ) の対応状況を整理した。CSV は 2010-01-01〜2020-12-31 の連続日付、欠損値は -9999、`*_QC` 列が 0=良/1=補間のフラグ。

| VISIT/LSTM動的入力 | 元の単位/意味 | 観測CSVの候補列 | 追加作業メモ |
| --- | --- | --- | --- |
| `tmp_sfc_d` (地表気温, K) | Kelvin (ERA5: ~300K) | `TA_F` (℃) / `TA_F_DAY` / `TA_F_NIGHT` | °C→K で +273.15。キャノピー/塔測温のため厳密には地表温度ではない。`TA_F_QC` でQC>0は無効化。|
| `tmp_2m_d` (2m 気温, K) | Kelvin | `TA_F` | 観測では地表との区別が無いため暫定的に同じ列を流用予定。|
| `tmp10_soil_d` (10cm 土壌温, K) | Kelvin | **該当列なし** | CSVは土壌水分 (`SWC_*`) のみ。別ロガーの土壌温か ERA5 の値を引き続き採用する必要あり。|
| `tmp200_soil_d` (200cm 土壌温, K) | Kelvin | **該当列なし** | 上に同じ。|
| `dswrf_sfc_d` (下向き短波, W m⁻²) | W m⁻² | `SW_IN_F` (`SW_IN_PI_F` で補間可) | 単位は一致。`SW_IN_F_QC` でフィルタ。|
| `tcdc_clm_d` (全天雲量, 0-1) | ratio | **直接列なし** | `SW_IN_F / SW_IN_POT` や `LW_IN_F` を使った経験式で推定するか、ERA5雲量を残す。|
| `prate_sfc_d` (降水, mm day⁻¹) | mm day⁻¹ | `P_F` / `P_PI_F` | 数値レンジから mm/day と判断。`P_F_QC` を参照。必要なら kg m⁻² day⁻¹ へ 1:1 変換。|
| `spfh_2m_d` (2m 比湿, kg kg⁻¹) | kg kg⁻¹ | **直接列なし** (関連: `VPD_F`, `PA_F`, `TA_F`) | `TA_F` から飽和水蒸気圧 `e_s` を計算し、`VPD_F` (kPa) で `e = e_s - VPD`、`PA_F` (kPa→Pa) を使って `q = 0.622 e / (p - 0.378 e)` を求める。VPD列のQCも確認。|
| `wind_10m_d` (10m 風速, m s⁻¹) | m s⁻¹ | `WS_F` | タワー高さが10 mと異なる場合は対数風速補正を検討。`WS_F_QC` で外れ値除去。|

補足:

- CSVには長波放射 (`LW_IN_F`), 潜在短波 (`SW_IN_POT`), PPFD (`PPFD_IN_F`), 大気圧 (`PA_F`), CO₂濃度 (`CO2_F`) なども含まれる。ERA5 でしか得られない変数は当面ブレンド運用とし、差分を別列に保持すると検証が楽。
- `TIMESTAMP` は `YYYYMMDD`。現行パイプライン同様ローカル日付で処理すれば問題ないが、時刻分解能を上げる際はタイムゾーン管理が必要。
- 気象を観測値に切り替えても既存 `scaler_dynamic.pkl` に合わせる必要があるため、列順 `[[tmp_sfc, tmp_2m, tmp10_soil, tmp200_soil, dswrf, tcdc, prate, spfh, wind], aCO2]` を守り、欠損は `-9999→NaN→mask` の手順で管理する。
- 比湿や雲量のように派生計算が必要な量は、専用の `prepare_observation_meteorology.py` を用意してロジックを共有すると安全。土壌温については別途センサーを探すか ERA5 を継続利用する方針が現実的。
- 観測値から `tmp_2m / dswrf / 降水 / 風速` を一括で上書きする場合は `LHP_observation/blend_observation_meteorology.py` を利用する。例:

```bash
cd /mnt/d/VISIT/honban/point/ex/LHP_observation
python blend_observation_meteorology.py \
  --era5-input ../CODE_LHP/INPUT/LHP_ERA5.txt \
  --obs-csv FLX_MY-LHP_JapanFLUX2024_ALLVARS_DD_2010-2020_1-3.csv \
  --output ../CODE_LHP/INPUT/LHP_BLEND.txt
```

`LHP_BLEND.txt` を `init_site.c` で参照する設定に変えれば、QCを見ずに観測列へ完全置換した強制力がそのまま VISIT に渡る。欠測が含まれる日だけ ERA5 が残る点に注意。

## アーキテクチャ

```
Input:
  - context_x: (batch, 180, 10)  # 過去180日の動的特徴
  - static_x: (batch, 36)        # 静的パラメータ
  - future_known: (batch, 30, 10) # 未来30日の既知入力（気象9 + aCO2）

Model:
  1. Static MLP: static_x → (h0, c0) for LSTM initialization
  2. Static Embedding: static_x → embedding (64-dim)
  3. LSTM: [context_x + static_emb] → hidden states
  4. Autoregressive prediction with known future inputs
  5. Output Head: hidden → (GPP, NPP, NEP, Rh)

Output:
  - predictions: (batch, 30, 4)  # 未来30日の予測
```

### モデル仕様

- **LSTM**: hidden_size=256, num_layers=2, dropout=0.1
- **Static conditioning**:
  - init_state MLP: [36 → 128] → 2×num_layers×hidden_size
  - concat_input embedding: [36 → 64]
- **Output head**: [hidden_size → 128 → 4]

## ディレクトリ構造

```
ML_LHP_LSTM/
├── README.md                  # このファイル
├── requirements.txt           # 依存パッケージ
├── LHP_architecture.yaml      # アーキテクチャ仕様
├── dataset.py                 # データローディング
├── model.py                   # LSTMモデル実装
├── train.py                   # 学習スクリプト
├── data/                      # 前処理済みデータ
│   ├── scaler_dynamic.pkl
│   └── scaler_static.pkl
├── artifacts/                 # 学習結果
│   ├── config.json
│   ├── checkpoint_best.pt
│   ├── checkpoint_latest.pt
│   └── training_history.json
├── logs/                      # ログファイル
├── models/                    # 保存済みモデル
└── configs/                   # 設定ファイル
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
cd /mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM
pip install -r requirements.txt
```

### 2. データの確認

学習データは`OUTPUT_PERTURBATION_LHP_V2_LHS/`にあることを確認:
- `LHP_20250919_daily_0000.txt` ~ `LHP_20250919_daily_0179.txt` (180ファイル)
- `parameter_summary.csv` (180行 × [sample_id + 36パラメータ])

## 使用方法

### 学習の実行

```bash
python train.py
```

学習パラメータは`train.py`の`config`辞書で設定:

```python
config = {
    "batch_size": 64,
    "context_len": 180,      # 過去のコンテキスト長
    "prediction_len": 30,    # 予測ホライズン
    "num_epochs": 50,
    "optimizer": {
        "lr": 2e-3,
        "weight_decay": 1e-4
    },
    # ... その他の設定
}
```

`train.py`実行時に `dynamic_dim` / `static_dim` / `known_future_dim` はデータローダから自動検出されるため、configを手で合わせ直す必要はありません。

> **バックグラウンドで実行する場合**
>
> `nohup conda run ...` だと標準出力がリダイレクトされずログが空になるため、以下のように環境の Python を直接指定してください。
>
> ```bash
> nohup /home/sakastation/anaconda3/envs/deepl/bin/python -u train.py \
>   > logs/train_20251123.log 2>&1 &
> echo $! > logs/train_20251123.pid
> ```
>
> `python -u` もしくは `PYTHONUNBUFFERED=1` を付ければ `tail -f logs/...` でリアルタイムに進捗を確認できます。

### データセットのテスト

```bash
python dataset.py
```

### モデルのテスト

```bash
python model.py
```

## 学習詳細

### ハイブリッド損失関数

```python
L_total = 0.7 × MSE_one_step + 0.3 × MSE_rollout@30

where:
  MSE_one_step = MSE(y_pred[:30], y_true[:30])  # 全30ステップの損失
  MSE_rollout@30 = MSE(y_pred[:30], y_true[:30])  # ロールアウト損失
```

### Scheduled Sampling

Teacher forcing確率をステップ数に応じて線形減衰:

```
TF_ratio(step) = 1.0 - (step / 20000) × (1.0 - 0.3)
               = 1.0  (step=0)
               → 0.3  (step≥20000)
```

### 学習スケジュール

- **Optimizer**: AdamW (lr=2e-3, weight_decay=1e-4)
- **Scheduler**: Cosine warmup (warmup_steps=500)
- **Gradient clipping**: max_norm=1.0
- **Early stopping**: val/Rollout_RMSE@30で判定 (patience=10)

## 評価指標

- **Primary metric**: NEP_RMSE (同化の主対象)
- **Other metrics**: MAE, RMSE, MAPE, Rollout_RMSE@30

各出力変数(GPP, NPP, NEP, Rh)に対して計算します。

## 学習結果の確認

学習完了後、以下のファイルが生成されます:

1. **checkpoint_best.pt**: 最良モデル(validation loss最小)
2. **checkpoint_latest.pt**: 最新エポックのモデル
3. **training_history.json**: 学習履歴(loss, learning rate)
4. **config.json**: 学習時の設定

```python
# モデルのロード例
import torch
from model import create_model

checkpoint = torch.load("artifacts/checkpoint_best.pt")
model = create_model(
  dynamic_dim=10,
  static_dim=36,
    hidden_size=256,
    num_layers=2,
    dropout=0.1,
    device="cuda"
)
model.load_state_dict(checkpoint["model_state_dict"])
```

## トラブルシューティング

### メモリ不足

- `batch_size`を32または16に減らす
- `num_workers`を0に設定
- GPUメモリが不足する場合は`device="cpu"`で実行

### 学習が不安定

- Learning rateを下げる (1e-3など)
- `gradient_clip`の値を小さくする (0.5など)
- Dropoutを増やす (0.2など)

### 予測精度が低い

- `context_len`を増やす (240や360など)
- `hidden_size`を増やす (512など)
- エポック数を増やす
- データの正規化を確認

## 参考

- **データ生成**: `../LHP_parameter_perturbation/scripts/run_perturbations.py`
- **アーキテクチャ仕様**: `LHP_architecture.yaml`
- **実験ログ**: `OUTPUT_PERTURBATION_LHP_V2_LHS/experiment_results.csv`

## ライセンス

Internal use only

## 逆推定再現プロセス (Surrogate-based Parameter Recovery)

このリポジトリには、VISIT 摂動実験の任意シナリオに対して静的36次元パラメータを再推定し、推定値と真値を比較する完全な再現手順を含めています。最新ワークフローの概要は以下の通りです。

1. **評価スクリプト**: `evaluate_parameter_recovery.py`
   - `artifacts/` 配下の学習済みサロゲートとスケーラーをロードし、`OUTPUT_PERTURBATION_LHP_V2_LHS` からテストウィンドウを取り出して逆推定を実行。
   - 代表的なコマンド:

     ```bash
     cd /mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM
     conda run -n deepl python evaluate_parameter_recovery.py \
       --window-index 0 \
       --prior-mode mean \
       --trace-params tree_topt tree_pmax tree_sla
     ```

   - 出力物 (`inverse_recovery/...`):
     - `*_params.csv`: prior / true / estimated /差分のテーブル
     - `*_summary.json`: RMSE, MAE,最終損失などのサマリ
     - `*_overview.png`: 真値 vs 推定散布図 + 誤差ヒスト + NEP系列
     - `*_param_bars.png`: true/prior/estimated の棒グラフ
     - `*_param_traces.png`: `--trace-params` で指定したパラメータの学習過程 (iter vs 値)。`--trace-stride` で間引き可能

2. **逆推定器**: `inverse_estimator.py`
   - `StaticParameterInverter` が NEP 損失 + 事前正則化で静的パラメータを最適化。
   - `record_history=True` を渡すと毎イテレーションのパラメータを保存し、可視化に利用。

3. **可視化ユーティリティ**: `plot_param_scatter.py`
   - `evaluate_parameter_recovery.py` の CSV を読み込み、各パラメータを **個別レンジ** (`configs/range_of_paras.csv`) で 0-1 正規化した真値 vs 推定の散布図を描画。
   - 例:

     ```bash
     conda run -n deepl python plot_param_scatter.py \
       inverse_recovery/sample000_win0000_mean_20251127_182556_params.csv \
       --include-prior \
       --range-csv configs/range_of_paras.csv
     ```

   - `range_of_paras.csv` には `max_from` で別パラメータを参照するレンジも定義済み。スクリプト内で依存関係を解決し、全パラメータを [0,1] に正規化した上で散布図 (`*_params_norm_scatter.png`) を再生成する。

4. **再現結果の確認**: 最新実行では `sample000_win0000` に対し MAE≈4.16, RMSE≈8.76, 最大誤差≈33.7 (MgC ha⁻¹ day⁻¹換算) を得ており、トレース図/散布図込みの成果物は `inverse_recovery/` で確認できる。別ウィンドウや prior 設定で再現したい場合は `--window-index`, `--prior-mode`, `--prior-sample-id` などを変更するだけで同じパイプラインが利用可能。

> **補足**: 逆推定対象の NEP 観測を外部から与える場合は `test_inversion.py --mode observation` を使用し、`evaluate_parameter_recovery.py` と同一の `StaticParameterInverter` を通じて推定 + 可視化を行う。必要なコンテキスト/未来入力/観測 NEP は `observations/obs_windows_*.npz` 形式で準備する。
