#!/usr/bin/env python3
"""
Boston GPU Training Log Analyzer
Extracts metrics from TensorBoard events and creates visualization plots
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from tensorboard.compat.proto.event_pb2 import Event
    
    # TensorBoardイベントファイルを読み込む
    event_file = "/root/nuplan/exp/exp/training/pluto_boston/2026.01.21.22.21.56/events.out.tfevents.1769001734.f6f9ac4bc616.49070.0"
    
    print(f"📖 TensorBoardイベント読み込み: {event_file}")
    
    epochs = []
    train_losses = []
    val_minADEs = []
    val_minFDEs = []
    val_MRs = []
    
    with open(event_file, 'rb') as f:
        while True:
            try:
                # イベントのサイズを読む
                len_bytes = f.read(8)
                if not len_bytes:
                    break
                    
                event_size = int.from_bytes(len_bytes[:4], byteorder='little')
                event_bytes = f.read(event_size)
                
                # イベントをパース
                event = Event()
                event.ParseFromString(event_bytes)
                
                # スカラーメトリクスを抽出
                if event.HasField('summary'):
                    for value in event.summary.value:
                        tag = value.tag
                        scalar_value = value.simple_value if value.HasField('simple_value') else None
                        
                        if scalar_value is not None:
                            # 各メトリクスをキャプチャ
                            if 'train_loss' in tag or 'loss' in tag and 'train' in tag:
                                if hasattr(event, 'step'):
                                    pass
                            if 'val/minADE' in tag:
                                pass
                            if 'val/minFDE' in tag:
                                pass
                                
            except Exception as e:
                break
    
    print("⚠️ TensorBoard解析は複雑です")
    print("代替案: ログテキストから直接抽出")
    
except Exception as e:
    print(f"TensorBoard抽出エラー: {e}")

# ログテキストから直接抽出する方法に切り替え
import re

log_file = "/workspace/pluto/docs/train_trial/analyze_boston_gpu_train_10epochs.log"
print(f"\n📖 ログファイル読み込み: {log_file}")

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# 各行から データを抽出
epochs_data = {}

for i, line in enumerate(lines):
    # ANSIコード削除
    line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
    
    # Epoch番号を探す
    epoch_match = re.search(r'Epoch\s+(\d+)/9', line_clean)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        if current_epoch not in epochs_data:
            epochs_data[current_epoch] = {}
    
    # メトリクスを探す
    if 'loss/train_loss' in line_clean:
        # 数値を抽出
        num_match = re.search(r'([0-9.]+)\s*$', line_clean.strip())
        if num_match:
            try:
                value = float(num_match.group(1))
                if 0 < value < 1000:  # 妥当な範囲
                    if current_epoch in epochs_data:
                        epochs_data[current_epoch]['train_loss'] = value
            except:
                pass
    
    if 'val/minADE:' in line_clean or 'val/minADE ' in line_clean:
        num_match = re.search(r'([0-9.]+)', line_clean)
        if num_match:
            try:
                value = float(num_match.group(1))
                if 0 < value < 1000:
                    if current_epoch in epochs_data:
                        epochs_data[current_epoch]['minADE'] = value
            except:
                pass
    
    if 'val/minFDE:' in line_clean or 'val/minFDE ' in line_clean:
        # 次の行に値があることが多い
        num_match = re.search(r'([0-9.]+)', line_clean)
        if num_match:
            try:
                value = float(num_match.group(1))
                if 0 < value < 1000:
                    if current_epoch in epochs_data:
                        epochs_data[current_epoch]['minFDE'] = value
            except:
                pass
    
    if 'val/MR:' in line_clean or 'val/MR ' in line_clean:
        num_match = re.search(r'([0-9.]+)', line_clean)
        if num_match:
            try:
                value = float(num_match.group(1))
                if 0 < value < 100:
                    if current_epoch in epochs_data:
                        epochs_data[current_epoch]['MR'] = value
            except:
                pass

# 取得したデータをソート
epochs = sorted(epochs_data.keys())
train_losses = [epochs_data[e].get('train_loss', None) for e in epochs]
val_minADEs = [epochs_data[e].get('minADE', None) for e in epochs]
val_minFDEs = [epochs_data[e].get('minFDE', None) for e in epochs]
val_MRs = [epochs_data[e].get('MR', None) for e in epochs]

# Noneを削除
epochs = [e for i, e in enumerate(epochs) if train_losses[i] is not None]
train_losses = [v for v in train_losses if v is not None]
val_minADEs = [v for v in val_minADEs if v is not None]
val_minFDEs = [v for v in val_minFDEs if v is not None]

print(f"✅ 抽出されたEpoch数: {len(epochs)}\n")

print("抽出されたデータ:")
print(f"  Epochs: {epochs}")
print(f"  Train Losses: {[f'{x:.4f}' for x in train_losses]}")
print(f"  Val minADE: {[f'{x:.4f}' for x in val_minADEs]}")
print(f"  Val minFDE: {[f'{x:.4f}' if x else 'N/A' for x in val_minFDEs]}\n")

# グラフ化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Boston Dataset - GPU Training Results (10 Epochs)', fontsize=16, fontweight='bold')

# 1. Training Loss
ax = axes[0, 0]
ax.plot(epochs, train_losses, 'o-', linewidth=2, markersize=8, color='#FF6B6B', label='Train Loss')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss Progression', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
for i, (x, y) in enumerate(zip(epochs, train_losses)):
    ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

# 2. Validation minADE
ax = axes[0, 1]
ax.plot(epochs, val_minADEs, 's-', linewidth=2, markersize=8, color='#4ECDC4', label='Val minADE')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('minADE', fontsize=12)
ax.set_title('Validation minADE Progression', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
for i, (x, y) in enumerate(zip(epochs, val_minADEs)):
    ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

# 3. Validation minFDE
ax = axes[1, 0]
if all(v is not None for v in val_minFDEs):
    ax.plot(epochs, val_minFDEs, '^-', linewidth=2, markersize=8, color='#95E1D3', label='Val minFDE')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('minFDE', fontsize=12)
    ax.set_title('Validation minFDE Progression', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    for i, (x, y) in enumerate(zip(epochs, val_minFDEs)):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

# 4. Summary Table
ax = axes[1, 1]
ax.axis('off')

summary_data = []
summary_data.append("📊 SUMMARY STATISTICS")
summary_data.append("="*40)
summary_data.append(f"Total Epochs: {len(epochs)}")
summary_data.append(f"Initial Train Loss: {train_losses[0]:.4f}")
summary_data.append(f"Final Train Loss: {train_losses[-1]:.4f}")
summary_data.append(f"Loss Reduction: {(train_losses[0] - train_losses[-1]):.4f}")
summary_data.append(f"Improvement: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%")
summary_data.append("")
summary_data.append(f"Initial Val minADE: {val_minADEs[0]:.4f}")
summary_data.append(f"Final Val minADE: {val_minADEs[-1]:.4f}")
summary_data.append(f"minADE Reduction: {(val_minADEs[0] - val_minADEs[-1]):.4f}")
summary_data.append(f"Improvement: {(1 - val_minADEs[-1]/val_minADEs[0])*100:.1f}%")
if all(v is not None for v in val_minFDEs):
    summary_data.append("")
    summary_data.append(f"Initial Val minFDE: {val_minFDEs[0]:.4f}")
    summary_data.append(f"Final Val minFDE: {val_minFDEs[-1]:.4f}")
    summary_data.append(f"minFDE Reduction: {(val_minFDEs[0] - val_minFDEs[-1]):.4f}")
    summary_data.append(f"Improvement: {(1 - val_minFDEs[-1]/val_minFDEs[0])*100:.1f}%")

summary_text = "\n".join(summary_data)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/workspace/pluto/docs/train_trial/boston_training_metrics.png', dpi=150, bbox_inches='tight')
print("✅ グラフ保存: /workspace/pluto/docs/train_trial/boston_training_metrics.png")

# 詳細なメトリクステーブル
print("\n" + "="*90)
print("📊 詳細メトリクステーブル")
print("="*90)
print(f"{'Epoch':<8} {'Train Loss':<15} {'Val minADE':<15} {'Val minFDE':<15} {'Val MR':<15}")
print("-"*90)
for i, epoch in enumerate(epochs):
    fde_str = f"{val_minFDEs[i]:.4f}" if val_minFDEs[i] is not None else "N/A"
    mr_str = f"{val_MRs[i]:.4f}" if val_MRs[i] is not None else "N/A"
    print(f"{epoch:<8} {train_losses[i]:<15.4f} {val_minADEs[i]:<15.4f} {fde_str:<15} {mr_str:<15}")
print("="*90)

print("\n✨ ログ解析完了！")
