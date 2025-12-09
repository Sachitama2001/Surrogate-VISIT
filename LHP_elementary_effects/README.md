# LHP Elementary Effects スクリーニング

VISIT LHP パラメータセットを対象に、Elementary Effects（Morris 法）で先行スクリーニングを行うための専用ワークスペースです。

## 目的
- `/mnt/d/VISIT/honban/point/ex/CODE_LHP/INPUT/parameter_LHP_backup.txt` に保存されている正規化済みパラメータ値をベースラインとして採用する。
- LHS へ進む前に、Morris 法で 1 変数ずつ摂動し、影響度の大きいパラメータ候補を抽出する。
- 評価指標は年間平均の `GPP` `NEP` `ER` `NPP`。これらの結果を次段階（Sobol 解析など）へ引き継ぐ。

## ディレクトリ構成
```
LHP_elementary_effects/
├─ README.md                ← 本ファイル
├─ configs/                 ← 軌跡長・デルタ・シード等の設定ファイル
├─ scripts/                 ← パラメータ更新・VISIT 実行・指標抽出のスクリプト群
├─ docs/                    ← パラメータ一覧や設計メモなどの補足資料
└─ results/                 ← 各スクリーニングバッチの生ログと集計結果
```

## 初期計画
1. **ベースライン整理**
   - `parameter_LHP_backup.txt` をパースし、対象 120 変数の名称・行番号・初期値・推奨ステップ幅を `docs/` に CSV で保存する。
2. **Morris 設計**
   - SALib などを使った Morris サンプリング、または自前実装で軌跡を生成（グリッドレベル p=4〜6 程度）。
   - 軌跡上では 1 ステップごとに 1 変数のみ変更するため、既存の VISIT コンパイル動線を再利用できる。
3. **モデル実行**
   - 各サンプル点で `parameter_LHP.txt` を上書き → VISIT 実行 → 年次時系列を取得。
   - 実行後は必ずベースラインファイルを復元（既存の摂動スクリプトと同じ手順）。
4. **指標計算**
   - 出力から年平均 GPP/NEP/ER/NPP を算出し、`results/` にサンプル ID・変数名・デルタ量・4 指標を記録。
5. **Elementary Effects 解析**
   - パラメータごとに μ*（平均絶対効果）と σ（効果のばらつき）を算出し、次段階へ渡す優先候補を決定する。

### Morris 設計 CSV の生成元
   ```bash
   cd /mnt/d/VISIT/honban/point/ex/LHP_elementary_effects
   conda run -n deepl python scripts/prepare_morris_design.py --config configs/elementary_effects_config.json
   ```
   ※ config の `num_trajectories` と `num_levels` に応じて行数が変わる（現行 CSV は 114 変数 × 2 軌跡 → 230 行）。
 最新実行メモ (2025-12-08):
 - 古い成果物を削除し、設計を再生成（115 変数 × 3 軌跡 → 348 サンプル）。ステージング時に `f_hm_a/i/p` を正規化するロジックを導入。
 - `scripts/run_morris_experiments.py --config configs/elementary_effects_config.json --start-id 0` で 348 サンプルすべて実行（`status=success`）。
 - `scripts/analyze_morris_effects.py` で μ* を再計算し、`results/morris_effects/morris_effects_<metric>.csv` と JSON を更新。
 - `scripts/visualize_morris_effects.py --top-n 15` で μ* 上位の棒グラフを `results/morris_effects/plots/` に保存。
 - `scripts/aggregate_morris_timeseries_metrics.py` で年平均 + 年次SD + 日次SD を集約し、`results/morris_summary_timeseries_metrics.csv` を出力。
 - `scripts/visualize_timeseries_metrics.py` で平均・年次SD・日次SDのヒスト/散布図を生成（`results/morris_effects/plots/`）。
 - `scripts/analyze_variability_drivers.py --top-k 15` で SD 指標に対するパラメータの Spearman 相関を算出し、`results/morris_effects/variability_drivers_*.csv` と対応するバー図を出力。
### 土壌腐植分配の正規化（警告対策）
- ステージング時に `f_hm_a` / `f_hm_i` / `f_hm_p` の和が 1 になるよう `f_hm_p = 1 - a - i` を再計算して書き戻す。Morris 設計で `f_hm_p` を振っても上書きされる。
- `f_hm_i` は `docs/parameter_metadata.csv` に追記済み（line_index=138）。サンプルに含まれない場合はベースライン値を用いる。

## 次のステップ
1. `docs/parameter_screening_plan.md` を作成し、対象 120 変数の許容範囲と Morris グリッド解像度をまとめる。
2. `scripts/` に「Morris 設計を読み込み → パラメータ適用 → VISIT 実行 → 年平均抽出」を一気通貫で行うランナーを実装する。
3. `configs/` に軌跡数・グリッドレベル・デルタ幅・ランダムシードを JSON などで整理し、再現性を確保する。

## ランナー実装状況
- `scripts/run_morris_experiments.py`: `results/morris_samples/` にステージ済みの `parameter_LHP.txt` を読み取り、指定したサンプル ID 範囲について `CODE_LHP/visitb` を順次実行する。各サンプルのログと `LHP_*` 出力は `results/morris_runs/sample_####/` 以下に整理し、`results/morris_metrics_summary.csv` に GPP/NEP/ER/NPP の年平均を追記する。
- 実行例:
   ```bash
   cd /mnt/d/VISIT/honban/point/ex/LHP_elementary_effects
   conda run -n deepl python scripts/run_morris_experiments.py \
      --config configs/elementary_effects_config.json \
      --start-id 0 --max-samples 5
   ```
   既定では `make` を 1 度実行して `visitb` を再コンパイルする。既存バイナリを使い回す場合は `--no-compile` を付与する。

### 出力ディレクトリ
- `results/morris_samples/`: Morris 設計のパラメータファイル（既存）
- `results/morris_runs/`: ランナーが自動生成するサンプル別の VISIT 出力（annual/daily/spinup/txt, visit ログ、使用したパラメータ）
- `results/morris_metrics_summary.csv`: 各サンプルの実行状況と GPP/NEP/ER/NPP 年平均を一覧化したサマリーファイル
- `results/morris_effects/`: SALib による Morris 効果量の集計（`morris_effects_<metric>.csv` と `morris_effects_all_metrics.json`）。
- `results/morris_effects/plots/`: Morris 効果の μ* 上位を棒グラフ化した PNG（`scripts/visualize_morris_effects.py` で生成）。

最新実行メモ:
- 2025-12-07: 230 サンプル（旧設計）を実行し、`scripts/analyze_morris_effects.py` で効果量算出、`visualize_morris_effects.py --top-n 15` で μ* 上位を図表化。
- 2025-12-08: 設計を再生成（115 パラメータ×3 軌跡=348 サンプル）、ステージング時に `f_hm_p=1-a-i` 正規化を導入し humification 警告を解消。全 348 サンプルを `status=success` で完走。
   - 集計: `results/morris_metrics_summary.csv`（年平均）に加え、`aggregate_morris_timeseries_metrics.py` で日次・年次の変動指標を含む `results/morris_summary_timeseries_metrics.csv` を生成。
   - 可視化: `visualize_timeseries_metrics.py` で mean/SD のヒスト・散布図を `results/morris_effects/plots/` に出力。
   - 変動ドライバ解析: `analyze_variability_drivers.py` で年次・日次 SD とパラメータの Spearman 相関を計算し、CSV とバー図（`results/morris_effects/variability_drivers_*.csv` と `plots/variability_drivers_*.png`）を生成。

## 年次出力の指標列（`CODE_LHP/ansis.c`）
`output_ansis_ann` の `fprintf` を確認した結果、`LHP_*_annual*.txt` の列は以下の並びになっている。

| 列番号 (1-based) | 指標 | `ansis_ann` | 備考 |
| --- | --- | --- | --- |
| 2 | GPP | `[0]` | 年間総 GPP |
| 3 | NPP | `[1]` | 年間総 NPP |
| 4 | NEP | `[2]` | 正の値で吸収 |
| 15 | ER | `[4]` | Total ecosystem respiration |

ランナーのメトリクス抽出ロジックもこの列対応に基づいており、年間値を単純平均（年数で割り算）して μ* 計算の入力に利用する。

## コミュニケーション方針
今後のエージェントとのやり取り（指示・結果報告・質疑応答）は基本的に日本語で行う。

この構成により、既存の LHS 摂動コードや VISIT バイナリを再利用しつつ、Morris 法による先行スクリーニングを独立したプロジェクトとして管理できる。
