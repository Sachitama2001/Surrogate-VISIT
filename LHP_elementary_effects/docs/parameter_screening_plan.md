# LHP Elementary Effects パラメータスクリーニング計画

## 対象と目的
- **対象モデル**: VISIT LHP セット。ベースラインは `/mnt/d/VISIT/honban/point/ex/CODE_LHP/INPUT/parameter_LHP_backup.txt`。
- **指標**: 年平均 `GPP`, `NEP`, `ER`, `NPP`。訪問出力から年次集計して Elementary Effects 指標 (μ*, σ) を計算し、次段階の Sobol 解析に渡す。
- **パラメータ候補**: `docs/parameter_metadata.csv` に整理済み（121 変数）。固定・コピー・派生扱いのものは除外し、114 変数が Morris 設計に載っている。

## Morris 設計の仕様
- **設計生成ツール**: `scripts/prepare_morris_design.py`（SALib 1.5.2 依存）。
- **設定ファイル**: `configs/elementary_effects_config.json`。
- **主な設定値**:
  - 軌跡数 `num_trajectories = 20`
  - グリッドレベル `num_levels = 6`
  - ローカル最適化 `local_optimization = true`
  - シード `seed = 20251207`
- **設計CSV**: `configs/morris_design_lhp.csv`  (行数 230 = 20 × (114 + 1)). 先頭列 `sample_id`、以降に各パラメータ値が並ぶ。

## ワークフロー概要
1. **設計生成**
   - `python scripts/prepare_morris_design.py --config configs/elementary_effects_config.json`
   - 既に SALib を `deepl` 環境へインストール済み。
2. **サンプル実行**
   - 今後作成するランナー `scripts/run_morris_experiment.py`（仮）を使用予定。
   - 流れ: 設計CSV読込 → ベースラインをコピー → 1 サンプルずつ `parameter_LHP.txt` を更新 → VISIT 実行 → 年平均指標を抽出 → 結果を `results/` に保存。
   - 失敗時はログと入力状態を保持し、リトライ可能な構造にする。
3. **年平均計算**
   - 既存の出力解析コード（例: `ex/advanced_sensitivity_visualization.py`）を参考に、GPP/NEP/ER/NPP を年単位に集計。
   - `results/run_<timestamp>/summary.csv` のような共通フォーマットを想定。
4. **Elementary Effects 集計**
   - SALib の `morris.analyze` もしくは自前実装で μ*, σ を計算。計算結果は `results/analysis_<timestamp>.csv` に書き出す。

## 実装メモ
- VISIT 実行箇所は既存 `ex/LHP_parameter_perturbation/scripts/run_perturbations.py` の `generate_perturbed_parameters`/`run_one_sample` を参考にする。
- モリス法では 1 パラメータずつステップが進むが、設計 CSV には各ステップで全変数の値が並ぶため、「差分がある変数だけ`parameter_LHP.txt`を更新」する処理で効率化できる。
- 成果物:
  - `scripts/run_morris_experiment.py`（設計→VISIT→指標）
  - `scripts/aggregate_morris_results.py`（μ*, σ 計算）
- 長時間バッチのため、ログと途中結果の保存（例: JSON で進捗管理）が必須。

## TODO
- [x] パラメータメタデータ作成 (`docs/parameter_metadata.csv`)
- [x] Morris 設計 CSV 作成 (`configs/morris_design_lhp.csv`)
- [ ] ランナー実装 (`scripts/run_morris_experiment.py`)
- [ ] 指標集計 + μ*/σ 計算 (`scripts/aggregate_morris_results.py`)
- [ ] `results/README.md` に出力フォーマットを整理
