"""
学習済みLSTMモデルの評価と可視化

評価内容:
1. 学習曲線（Loss推移）
2. テストデータでの予測性能
3. パラメータ別の予測精度
4. 時系列予測の可視化
5. 誤差分布の解析
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from model import create_model
from dataset import VISITTimeSeriesDataset

# 設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# パス設定
BASE_DIR = Path("/mnt/d/VISIT/honban/point/ex/ML_LHP_LSTM")
DATA_DIR = Path("/mnt/d/VISIT/honban/point/ex/OUTPUT_PERTURBATION_LHP_V2_LHS")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
VIS_DIR = BASE_DIR / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

FLUX_NAMES = ['GPP', 'NPP', 'ER', 'NEP']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_training_history():
    """学習履歴を読み込み"""
    history_file = ARTIFACTS_DIR / "training_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print(f"✓ Loaded training history: {len(history['train_loss'])} epochs")
    return history

def load_model_and_config():
    """学習済みモデルと設定を読み込み"""
    # 設定読み込み
    config_file = ARTIFACTS_DIR / "config.json"
    with open(config_file, 'r') as f:
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
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
    
    return model, config, checkpoint

def plot_learning_curves(history):
    """学習曲線のプロット"""
    print("\n" + "="*60)
    print("1. 学習曲線の可視化")
    print("="*60)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Loss曲線
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 最小値をマーク
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    axes[0].axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)
    axes[0].text(best_epoch, best_val_loss, f'  Best: Epoch {best_epoch}\n  Loss={best_val_loss:.6f}',
                fontsize=10, verticalalignment='bottom')
    
    # 学習率の推移
    axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / "learning_curves.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: learning_curves.png")
    plt.close()
    
    # 詳細統計
    print(f"\n学習統計:")
    print(f"  Total epochs: {len(epochs)}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.6f}")

def evaluate_on_test_data(model, config):
    """テストデータで評価"""
    print("\n" + "="*60)
    print("2. テストデータでの評価")
    print("="*60)
    
    # スケーラーをファイルから読み込み
    scaler_dir = BASE_DIR / "data"
    with open(scaler_dir / "scaler_dynamic.pkl", "rb") as f:
        scaler_dynamic = pickle.load(f)
    with open(scaler_dir / "scaler_static.pkl", "rb") as f:
        scaler_static = pickle.load(f)
    
    print(f"✓ Loaded scalers from {scaler_dir}")
    
    # テストデータセット作成
    test_dataset = VISITTimeSeriesDataset(
        data_dir=config["data_dir"],
        split="test",
        context_len=config["context_len"],
        prediction_len=config["prediction_len"],
        scaler_dynamic=scaler_dynamic,
        scaler_static=scaler_static,
        fit_scaler=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Test dataset: {len(test_dataset)} windows")
    
    # 評価
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            context_x = batch['context_x'].to(DEVICE)
            static_x = batch['static_x'].to(DEVICE)
            future_known = batch['future_known'].to(DEVICE)
            target_y = batch['target_y'].to(DEVICE)
            
            # 予測
            predictions = model(context_x, static_x, future_known)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target_y.cpu().numpy())
    
    # 結合
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Targets shape: {targets.shape}")
    
    # メトリクス計算
    mse = np.mean((predictions - targets)**2, axis=(0, 1))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets), axis=(0, 1))
    
    # R2スコア
    ss_res = np.sum((targets - predictions)**2, axis=(0, 1))
    ss_tot = np.sum((targets - targets.mean(axis=(0, 1), keepdims=True))**2, axis=(0, 1))
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\nテスト性能 (30日予測):")
    print(f"{'Flux':<8} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 56)
    for i, flux_name in enumerate(FLUX_NAMES):
        print(f"{flux_name:<8} {mse[i]:<12.6f} {rmse[i]:<12.6f} {mae[i]:<12.6f} {r2[i]:<12.4f}")
    
    return predictions, targets, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_prediction_accuracy(predictions, targets, metrics):
    """予測精度の可視化"""
    print("\n" + "="*60)
    print("3. 予測精度の可視化")
    print("="*60)
    
    # メトリクスのバープロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('RMSE', metrics['rmse']),
        ('MAE', metrics['mae']),
        ('R²', metrics['r2']),
        ('MSE', metrics['mse'])
    ]
    
    for idx, (metric_name, values) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(FLUX_NAMES, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                     alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} by Flux Variable', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}' if metric_name == 'R²' else f'{value:.5f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_metrics.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: test_metrics.png")
    plt.close()

def plot_prediction_vs_actual(predictions, targets):
    """予測値 vs 実測値の散布図"""
    print("\n" + "="*60)
    print("4. 予測値 vs 実測値の散布図")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, flux_name in enumerate(FLUX_NAMES):
        pred_flat = predictions[:, :, i].flatten()
        target_flat = targets[:, :, i].flatten()
        
        # サンプリング（点が多すぎる場合）
        n_points = len(pred_flat)
        if n_points > 10000:
            idx = np.random.choice(n_points, 10000, replace=False)
            pred_flat = pred_flat[idx]
            target_flat = target_flat[idx]
        
        # 散布図
        axes[i].scatter(target_flat, pred_flat, alpha=0.3, s=10, color='steelblue')
        
        # 対角線（完全一致）
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        # 相関係数
        corr = np.corrcoef(target_flat, pred_flat)[0, 1]
        
        axes[i].set_xlabel('Actual (gC/m²/day)', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Predicted (gC/m²/day)', fontsize=11, fontweight='bold')
        axes[i].set_title(f'{flux_name} (R={corr:.4f})', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')
    
    plt.suptitle('Predicted vs Actual Values', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "prediction_vs_actual.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: prediction_vs_actual.png")
    plt.close()

def plot_time_series_predictions(model, config, n_samples=3):
    """時系列予測の可視化"""
    print("\n" + "="*60)
    print("5. 時系列予測の可視化")
    print("="*60)
    
    # スケーラーをファイルから読み込み
    scaler_dir = BASE_DIR / "data"
    with open(scaler_dir / "scaler_dynamic.pkl", "rb") as f:
        scaler_dynamic = pickle.load(f)
    with open(scaler_dir / "scaler_static.pkl", "rb") as f:
        scaler_static = pickle.load(f)
    
    # テストデータセット
    test_dataset = VISITTimeSeriesDataset(
        data_dir=config["data_dir"],
        split="test",
        context_len=config["context_len"],
        prediction_len=config["prediction_len"],
        scaler_dynamic=scaler_dynamic,
        scaler_static=scaler_static,
        fit_scaler=False
    )
    
    # ランダムにサンプルを選択
    sample_indices = np.random.choice(len(test_dataset), n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, len(FLUX_NAMES), figsize=(20, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            sample = test_dataset[idx]
            
            # バッチ次元を追加
            context_x = sample['context_x'].unsqueeze(0).to(DEVICE)
            static_x = sample['static_x'].unsqueeze(0).to(DEVICE)
            future_known = sample['future_known'].unsqueeze(0).to(DEVICE)
            target_y = sample['target_y'].cpu().numpy()
            
            # 予測
            prediction = model(context_x, static_x, future_known)
            prediction = prediction.cpu().numpy()[0]
            
            for col, flux_name in enumerate(FLUX_NAMES):
                ax = axes[row, col]
                
                # 予測期間のみ表示
                pred_days = np.arange(0, config["prediction_len"])
                ax.plot(pred_days, target_y[:, col], 'g-', linewidth=2.5, 
                       label='Actual', alpha=0.8)
                ax.plot(pred_days, prediction[:, col], 'r--', linewidth=2.5, 
                       label='Predicted', alpha=0.8)
                
                # 誤差を塗りつぶし
                ax.fill_between(pred_days, target_y[:, col], prediction[:, col],
                               alpha=0.2, color='gray', label='Error')
                
                ax.set_xlabel('Days ahead', fontsize=10)
                ax.set_ylabel(f'{flux_name} (gC/m²/day)', fontsize=10)
                ax.set_title(f'Window {idx} - {flux_name}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('30-Day Flux Predictions (3 Random Test Windows)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "timeseries_predictions.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: timeseries_predictions.png")
    plt.close()

def plot_error_distribution(predictions, targets):
    """誤差分布の解析"""
    print("\n" + "="*60)
    print("6. 誤差分布の解析")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, flux_name in enumerate(FLUX_NAMES):
        errors = (predictions[:, :, i] - targets[:, :, i]).flatten()
        
        # ヒストグラム
        axes[i].hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        # 統計情報
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        axes[i].set_xlabel('Prediction Error (gC/m²/day)', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[i].set_title(f'{flux_name} Error Distribution\nμ={mean_error:.5f}, σ={std_error:.5f}', 
                         fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Prediction Error Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "error_distribution.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: error_distribution.png")
    plt.close()

def plot_continuous_timeseries(predictions, targets):
    """連続した時系列での予測精度を可視化"""
    print("\n" + "="*60)
    print("7. 連続時系列での予測比較")
    print("="*60)
    
    # 最初のN個の予測ウィンドウを連続表示（オーバーラップを考慮）
    n_windows = 12  # 約1年分（30日×12=360日）
    
    fig, axes = plt.subplots(len(FLUX_NAMES), 1, figsize=(20, 12))
    
    for i, flux_name in enumerate(FLUX_NAMES):
        # 各ウィンドウの最初の予測のみを使用して連続時系列を構築
        continuous_actual = []
        continuous_pred = []
        
        for w in range(n_windows):
            # 各ウィンドウの最初の1日だけを使用（オーバーラップを避ける）
            continuous_actual.append(targets[w, 0, i])
            continuous_pred.append(predictions[w, 0, i])
        
        # 最後のウィンドウは全期間を追加
        continuous_actual.extend(targets[n_windows-1, 1:, i])
        continuous_pred.extend(predictions[n_windows-1, 1:, i])
        
        days = np.arange(len(continuous_actual))
        
        axes[i].plot(days, continuous_actual, 'g-', linewidth=2, 
                    label='Actual', alpha=0.8)
        axes[i].plot(days, continuous_pred, 'r--', linewidth=2, 
                    label='Predicted', alpha=0.8)
        
        axes[i].fill_between(days, continuous_actual, continuous_pred, 
                            alpha=0.2, color='gray')
        
        # メトリクス計算
        rmse = np.sqrt(np.mean((np.array(continuous_pred) - np.array(continuous_actual))**2))
        mae = np.mean(np.abs(np.array(continuous_pred) - np.array(continuous_actual)))
        
        axes[i].set_ylabel(f'{flux_name} (gC/m²/day)', fontsize=12, fontweight='bold')
        axes[i].set_title(f'{flux_name} - Continuous Prediction (RMSE={rmse:.5f}, MAE={mae:.5f})', 
                         fontsize=13, fontweight='bold')
        axes[i].legend(fontsize=11, loc='best')
        axes[i].grid(True, alpha=0.3)
        
        if i == len(FLUX_NAMES) - 1:
            axes[i].set_xlabel('Days from start of test period', fontsize=12, fontweight='bold')
    
    plt.suptitle('Continuous Time Series Prediction (First ~40 days of 2019)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "continuous_timeseries.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: continuous_timeseries.png")
    plt.close()

def plot_multi_step_accuracy(predictions, targets):
    """複数ステップ先予測の精度変化を可視化"""
    print("\n" + "="*60)
    print("8. 予測ステップ数と精度の関係")
    print("="*60)
    
    # 各予測ステップ(1日目、2日目、...、30日目)でのRMSEを計算
    n_steps = predictions.shape[1]  # 30
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, flux_name in enumerate(FLUX_NAMES):
        rmse_by_step = []
        mae_by_step = []
        
        for step in range(n_steps):
            pred_step = predictions[:, step, i]
            target_step = targets[:, step, i]
            
            rmse = np.sqrt(np.mean((pred_step - target_step)**2))
            mae = np.mean(np.abs(pred_step - target_step))
            
            rmse_by_step.append(rmse)
            mae_by_step.append(mae)
        
        steps = np.arange(1, n_steps + 1)
        
        ax = axes[i]
        ax2 = ax.twinx()
        
        line1 = ax.plot(steps, rmse_by_step, 'b-', linewidth=2.5, 
                       label='RMSE', marker='o', markersize=4, alpha=0.8)
        line2 = ax2.plot(steps, mae_by_step, 'r-', linewidth=2.5, 
                        label='MAE', marker='s', markersize=4, alpha=0.8)
        
        ax.set_xlabel('Prediction Step (days ahead)', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE (gC/m²/day)', fontsize=11, fontweight='bold', color='b')
        ax2.set_ylabel('MAE (gC/m²/day)', fontsize=11, fontweight='bold', color='r')
        
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title(f'{flux_name} - Prediction Error by Step', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 凡例を統合
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=10, loc='upper left')
    
    plt.suptitle('Multi-Step Prediction Accuracy', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "multi_step_accuracy.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: multi_step_accuracy.png")
    plt.close()

def create_evaluation_report(history, metrics, checkpoint):
    """評価レポート作成"""
    print("\n" + "="*60)
    print("7. 評価レポート作成")
    print("="*60)
    
    report = []
    report.append("="*60)
    report.append("LSTM Model Evaluation Report")
    report.append("="*60)
    report.append("")
    report.append("--- Training Information ---")
    report.append(f"Total Epochs: {len(history['train_loss'])}")
    report.append(f"Best Epoch: {np.argmin(history['val_loss']) + 1}")
    report.append(f"Best Validation Loss: {checkpoint['best_val_loss']:.6f}")
    report.append(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
    report.append(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    report.append("")
    report.append("--- Test Performance (30-day prediction) ---")
    report.append(f"{'Flux':<8} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    report.append("-" * 44)
    for i, flux_name in enumerate(FLUX_NAMES):
        report.append(f"{flux_name:<8} {metrics['rmse'][i]:<12.6f} "
                     f"{metrics['mae'][i]:<12.6f} {metrics['r2'][i]:<12.4f}")
    report.append("")
    report.append("="*60)
    
    # 保存
    with open(VIS_DIR / "evaluation_report.txt", "w") as f:
        f.write("\n".join(report))
    
    print(f"✓ Saved: evaluation_report.txt")
    print("\n" + "\n".join(report))

def main():
    """メイン処理"""
    print("="*60)
    print("LSTM Model Evaluation and Visualization")
    print("="*60)
    
    # 1. 学習履歴読み込み
    history = load_training_history()
    
    # 2. モデル読み込み
    model, config, checkpoint = load_model_and_config()
    
    # 3. 学習曲線
    plot_learning_curves(history)
    
    # 4. テストデータで評価
    predictions, targets, metrics = evaluate_on_test_data(model, config)
    
    # 5. 予測精度の可視化
    plot_prediction_accuracy(predictions, targets, metrics)
    
    # 6. 予測 vs 実測
    plot_prediction_vs_actual(predictions, targets)
    
    # 7. 時系列予測
    plot_time_series_predictions(model, config, n_samples=3)
    
    # 8. 誤差分布
    plot_error_distribution(predictions, targets)
    
    # 9. 連続時系列
    plot_continuous_timeseries(predictions, targets)
    
    # 10. マルチステップ精度
    plot_multi_step_accuracy(predictions, targets)
    
    # 11. 評価レポート
    create_evaluation_report(history, metrics, checkpoint)
    
    print("\n" + "="*60)
    print("評価完了!")
    print(f"結果は {VIS_DIR} に保存されました")
    print("="*60)

if __name__ == "__main__":
    main()
