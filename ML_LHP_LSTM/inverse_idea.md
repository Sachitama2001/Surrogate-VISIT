静的特徴量 逆推定モジュール 仕様書（ドラフト）
1. 目的

訓練済み LSTM モデル（VISIT-LHPの入出力を模倣する NN）を用いて、
静的特徴量（36 次元パラメタ）を観測データに合わせて逆推定する。

NN の重み・動的入力（気象9変数 + aCO₂）群は固定し、
静的特徴ベクトルだけを最適化する。

2. 入出力
2.1 入力

訓練済みモデル

インターフェース（例）

model(static_features, dynamic_features) -> outputs

出力：[GPP, NPP, NEP, Rh] の4変数（日次系列）

静的パラメタの事前値

s_prior_raw: 形状 (static_dim=36,) のベクトル

VISIT 元のパラメタ、もしくは標準設定値。

動的入力時系列

X_dyn[t]: 形状 (T, dyn_feat_dim)

内容：同じLSTM訓練時に使用した スケーリング済み or スケーリング前 の気象9列 + aCO₂
（どちらを渡すかは、訓練済みモデルの仕様に合わせる）

観測データ

主に NEP の観測（GPP/NPP/Rh があれば利用可）

Y_obs[t]: 形状 (T, obs_dim)（最低でも NEP を含む）

観測が存在する時刻を示す obs_mask[t] （ブール or 0/1）

スケーラ・正規化パラメタ

静的特徴用：μ_s, σ_s（StandardScaler 相当）

必要なら動的特徴用・出力用も参照可能（ただし逆推定の更新対象は静的のみ）

設定（Config）

学習率、イテレーション数、正則化係数など（下記参照）。

2.2 出力

推定された静的パラメタ

s_opt_raw: 形状 (36,) のベクトル（物理空間）

必要に応じて s_opt_norm（正規化空間）も返す。

最終損失・履歴（任意）

final_loss: スカラー

loss_history: イテレーション毎の損失（デバッグ用）

3. 前提条件

LSTM モデルは すでに学習済み で、
逆推定時には パラメタ更新しない（requires_grad=False）。

訓練時に使用した スケーリング手法・パラメタ は保存済みで再利用可能。

逆推定は基本的に 1 サンプル（1サイト or 1パラメタセット）単位 で実行する。

4. 損失関数設計（ざっくり仕様）
4.1 観測データ損失 
𝐿
obs
L
obs
	​


基本ライン：NEP のみ MSE を使う。

𝐿
obs
=
1
∣
Ω
∣
∑
𝑡
∈
Ω
(
NEP
^
𝑡
−
NEP
𝑡
obs
)
2
L
obs
	​

=
∣Ω∣
1
	​

t∈Ω
∑
	​

(
NEP
^
t
	​

−NEP
t
obs
	​

)
2

Ω
Ω：観測が存在する時刻の集合（obs_mask==True）。

拡張案（観測があれば）：

GPP/NPP/Rh も使う場合、NEP に大きい重みを付けて足し合わせる。
（実装者には「NEP重視・他はオプション」と伝える）

4.2 事前パラメタからの乖離正則化 
𝐿
prior
L
prior
	​


事前値 s_prior_norm から離れ過ぎないように L2 正則化：

𝐿
prior
=
𝜆
prior
 
∥
𝑠
norm
−
𝑠
prior,norm
∥
2
L
prior
	​

=λ
prior
	​

∥s
norm
	​

−s
prior,norm
	​

∥
2

λ_prior は Config で指定（例：0.1〜10 の範囲で調整）。

4.3 総損失
𝐿
=
𝐿
obs
+
𝐿
prior
L=L
obs
	​

+L
prior
	​

5. 最適化の設計
5.1 更新対象

静的特徴ベクトル（正規化後）

s_norm: 形状 (36,)

これを nn.Parameter または同等の「勾配付き変数」として扱う。

5.2 固定するもの

LSTM モデルの全重み・バイアス。

動的入力時系列 X_dyn。

スケーラ（μ_s, σ_sなど）。

5.3 アルゴリズム

推奨：Adam または AdamW

lr: 1e-2 前後（Configで指定）

max_iters: 100〜1000 の範囲でConfig指定

各イテレーションで行う処理（高レベル）：

s_norm から静的入力を構成（必要ならパラメタ制約適用）。

model(s_norm, X_dyn) を実行して

𝑌
^
1
:
𝑇
Y
^
1:T
	​

（GPP/NPP/NEP/Rh）を取得。

NEP を取り出し、obs_mask を使って L_obs を計算。

L_prior を計算。

L = L_obs + L_prior。

L を逆伝播し、∂L/∂s_norm を取得。

オプティマイザで s_norm を更新。

必要ならパラメタ制約（クリップなど）を適用。

5.4 制約の扱い（簡易仕様）

パラメタごとに「妥当な範囲」がある場合は、物理空間でクリップする簡単な方式でよい：

s_norm → s_raw に逆変換。

s_raw[i] = clamp(s_raw[i], min_i, max_i) を適用。

再度 s_raw → s_norm に変換して戻す。

詳細な変数変換（softplus, sigmoid 等）はオプションとし、実装者判断に任せる。

6. Config 例（ざっくり）
inversion:
  lr: 0.02
  max_iters: 500
  lambda_prior: 1.0        # 事前との距離の強さ
  use_multi_output_loss: false  # trueならGPP/NPP/Rhも使う
  target_variable: "nep"   # 主対象
  use_mask: true           # 観測のある時刻のみ損失計算
  clip_raw_params:
    enabled: true
    # パラメタ別の[min,max]は別途テーブルで与える想定

7. モジュール構造（ざっくり）
7.1 クラス構成案

StaticParameterInverter

責務：1サンプル分の静的パラメタを逆推定する。

コンストラクタ引数：

訓練済みモデル model

スケーラ情報 static_scaler（μ, σ）

Config inversion_config

メソッド：

invert(s_prior_raw, X_dyn, Y_obs, obs_mask) -> s_opt_raw, info

info には最終損失やイテレーション数など任意情報。

必要に応じて：

apply_param_constraints(s_raw)：レンジクリップ関数

build_loss(outputs, Y_obs, obs_mask)：損失計算関数

7.2 想定フロー（利用側）

訓練済みモデルを読み込む。

スケーラを読み込む。

StaticParameterInverter を初期化。

逆推定したいサンプルごとに：

s_prior_raw, X_dyn, Y_obs, obs_mask を渡して invert(...) を実行。

得られた s_opt_raw を保存 or VISIT本体に渡して再実行などに利用。

8. 評価の観点（高レベル）

NEPの再現性

逆推定前後で NEP RMSE がどれだけ改善したか。

パラメタの変動量

||s_opt_raw - s_prior_raw|| が妥当な範囲か。

生態学的な整合性（オプション）

GPP/NPP/Rh のパターンが極端に不自然でないか。