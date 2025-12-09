# timeseries_predictions.png ガイド (V2 LHP バッチ)

`evaluate_model.py` の `plot_time_series_predictions` が生成する `visualizations/timeseries_predictions.png` について、最新版の挙動をまとめます。

## 1. 図の概要

- 行: テストデータからランダムに選んだ `n_samples` 個のウィンドウ（デフォルト3）。
- 列: フラックス変数 (GPP / NPP / ER / NEP)。
- 各サブプロットは **未来30日 (day 0〜29)** のみを描画し、実測値 (緑) と予測値 (赤点線) を直接比較します。過去180日の履歴は数値としてモデルに渡されますが、図には表示しません。

## 2. 軸と凡例

- X軸: 予測ホライズン (0 = 予測開始日, 29 = 30日目)。
- Y軸: フラックス (gC/m²/day)。
- 緑実線: VISIT 出力 (正解)。
- 赤破線: LSTM 予測。
- グレー塗り: 予測と実測の差分（誤差帯）。

## 3. データパイプライン

`VISITTimeSeriesDataset` が以下のテンソルを返します。

| key | shape | 内容 |
| --- | --- | --- |
| `context_x` | (180, 10) | 気象9 + aCO2 の履歴（スケーリング済み） |
| `static_x` | (36,) | parameter_summary.csv の静的パラメータ |
| `future_known` | (30, 10) | 予測期間の既知入力（気象9 + aCO2） |
| `target_y` | (30, 4) | VISIT フラックス (GPP/NPP/ER/NEP) |

描画時は以下の処理を行います。

```python
with torch.no_grad():
      pred = model(
            context_x.unsqueeze(0).to(device),
            static_x.unsqueeze(0).to(device),
            future_known.unsqueeze(0).to(device)
      )
prediction = pred.cpu().numpy()[0]  # (30, 4)
target = sample['target_y'].numpy() # (30, 4)
```

## 4. 再現手順

```bash
cd /mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM
conda run -n deepl python evaluate_model.py \
   --plot-timeseries --timeseries-count 3
```

`evaluate_model.py` をそのまま実行すると以下の可視化も含めて一括生成します。

- `learning_curves.png` / `test_metrics.png` / `prediction_vs_actual.png`
- `timeseries_predictions.png`
- `error_distribution.png`

## 5. カスタマイズ

`evaluate_model.py` 内 `plot_time_series_predictions` の引数で変更可能です。

- `n_samples`: 行数（表示するウィンドウ数）。
- `config["prediction_len"]`: 横軸の長さ。訓練時と揃える必要があります。
- `sample_indices`: ランダム抽選ではなく特定インデックスを指定したい場合は、関数内で上書きしてください。

## 6. 読み取りポイント

1. **短期 vs 長期**: day 0〜10 に比べて day 20 以降で誤差が増えやすいかを確認。
2. **フラックスごとの差**: ER は滑らかで追従しやすい一方、NEP は差分量のためノイズに弱い。
3. **季節イベント**: 30日窓内に季節遷移が入ると GPP/NPP の立ち上がりを外すことがあるため、必要に応じて `timeseries_count` を増やして複数例をチェック。

## 7. テストセットのサイズ感

- OUTPUT_PERTURBATION_LHP_V2_LHS では 180 サンプル。
- 各サンプルから 2019 年 365 日分のスライディング窓を作るため、テストウィンドウ数は約 33k（180 × (365 - 30 - 180 + 1)）。
- その中からランダム抽出した `n_samples` 件を図にしています。

## 8. よくある質問

- **Q. 過去180日のフラックス線が無いのは？**  
   A. 現行実装では入力となる動的特徴は気象+aCO2のみであり、フラックス履歴は `target_y` にのみ存在します。図は予測区間の比較に集中させるため、コンテキストは表示していません。

- **Q. 旧100サンプル時代の図との違いは？**  
   A. 旧版は状態変数も含めた11次元入力とフラックス履歴を同一軸に描いていました。新版は10次元 (気象9 + aCO2) に統一し、ラベル・凡例もアップデート済みです。

---

このガイドの内容は `evaluate_model.py` (2025-11-23 時点) を前提にしています。コードを変更した場合は、本ファイルも併せて更新してください。
