# LHPパラメータ摂動プロジェクト v2

`LHP_parameter_perturbation_v2` は、直近の VISIT-LHP 再出力と 9 つの気象変数（+ aCO2）に合わせてサロゲート用データを再構築するための新しいワークスペースです。元の `LHP_parameter_perturbation` を踏襲しつつ、以下の点を追加で管理します。

- **動的特徴の刷新**: `idea_for_next.txt` に列挙した 9 個の ERA5 系変数（tmp_sfc_d, tmp_2m_d, tmp10_soil_d, tmp200_soil_d, dswrf_sfc_d, tcdc_clm_d, prate_sfc_d, spfh_2m_d, wind_10m_d）に絞り、`aCO2` を動的列として明示的に出力／保存します。
- **再実行専用の成果置き場**: 既存結果と区別できるよう、出力ルートを `OUTPUT_PERTURBATION_LHP_V2` に変更しています。
- **LHS 設定はそのまま**: サンプル数、シード、範囲ファイルは従来と同じです。`configs/lhp_experiment_config.json` の `sampling` ブロックを更新する必要はありません。

## ディレクトリ構成
| Path | 用途 |
|------|------|
| `configs/` | LHS サンプリングや実行条件をまとめた JSON。デフォルトは `OUTPUT_PERTURBATION_LHP_V2_LHS` を指します。 |
| `docs/` | 動的変数や今後のタスクをメモしたファイル。`idea_for_next.txt` をこの場所にコピーしています。 |
| `scripts/` | 摂動実験ランチャー。現時点では v1 の `run_perturbations.py` を再利用する薄いラッパーです。必要に応じてここで v2 専用の変更を追加してください。 |

## 使い方
1. **Python 環境を有効化**（例: `conda run -n deepl python ...`）。
2. **VISIT 本体をビルド**: `cd /mnt/d/VISIT/honban/point/ex/CODE_LHP && make visitb`。
3. **摂動を実行**:
   ```bash
   cd /mnt/d/VISIT/honban/point/ex/LHP_parameter_perturbation_v2
    conda run -n deepl python scripts/run_perturbations_v2.py
   ```
   `--n_samples` や `--std_ratio` などの CLI オプションは従来通り利用できます。

## v1 との違い
- 出力先が `OUTPUT_PERTURBATION_LHP_V2(LHS)` になるため、旧データを温存したまま並行で解析可能です。
- `docs/idea_for_next.txt` に沿って `ansis.c` やサロゲートの動的列を 9 + aCO2 に制限する前提で進めます。
- `scripts/run_perturbations_v2.py` が v1 実装をラップしつつ、v2用 config (`configs/lhp_experiment_config.json`) と `docs/range_of_paras_v2.csv` を自動で指定します。CLIで独自の設定を渡したい場合はこれまで通りフラグを追加してください（`--range_file` を指定した場合はそちらが優先されます）。

## 今後のタスク
- `CODE_LHP/ansis.c` を編集して 9 変数 + aCO2 だけを書き出す。
- `OUTPUT_PERTURBATION_LHP_V2_LHS` を生成し、`ML_LHP_SIMPLE_PLUS` のデータパイプラインを再作成する。
- サロゲート v4 系列を再学習し、Rh の改善度を記録する。
