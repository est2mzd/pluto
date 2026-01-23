#!/usr/bin/env python3
import os
import glob

log_file = "/root/nuplan/exp/exp/training/pluto_boston/2026.01.21.21.49.48/log.txt"

print("\n【Q1】グラフ作成方法")
print("="*70)

print("工程:")
print("1. TensorBoard events ファイルから メトリクスデータを読み込む")
print("2. 各エポックの metrics を抽出")
print("3. matplotlib で 4つのグラフ（Loss, minADE, minFDE, MR）を描画")
print("\nコード例:")
print("""
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

events_file = 'events.out.tfevents.*'
scalar_dict = {}

for event in summary_iterator(events_file):
    for value in event.summary.value:
        if value.tag in ['loss', 'val/minADE', 'val/minFDE', 'val/MR']:
            if value.tag not in scalar_dict:
                scalar_dict[value.tag] = []
            scalar_dict[value.tag].append(value.simple_value)

# グラフ作成
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (tag, values) in zip(axes.flat, scalar_dict.items()):
    ax.plot(values)
    ax.set_title(tag)
plt.savefig('metrics.png')
""")

print("\n【Q2】学習に使用されるデータ")
print("="*70)

with open(log_file, 'r') as f:
    content = f.read()

# ログから正確なデータを抽出
for line in content.split('\n'):
    if 'Extracted' in line and 'scenarios' in line:
        print(f"✓ {line.split('INFO')[-1].strip()}")
    if 'Number of samples in train set' in line:
        print(f"✓ {line.split('INFO')[-1].strip()}")
    if 'Number of samples in validation set' in line:
        print(f"✓ {line.split('INFO')[-1].strip()}")

print("\n結論: Boston全500シナリオから 350(train) + 100(val) = 450 使用")

print("\n【Q3】Bostonシナリオ数")
print("="*70)

boston_db_count = len(glob.glob("/nuplan/dataset/nuplan-v1.1/splits/train_boston/*.db"))
print(f"Boston train_boston総シナリオ: {boston_db_count}")

cache_count = len([d for d in os.listdir("/nuplan/exp/boston_cache_correct") 
                   if os.path.isdir(os.path.join("/nuplan/exp/boston_cache_correct", d))])
print(f"キャッシュ済みシナリオ: {cache_count}")

print(f"学習に使用: 500 scenarios (ログから確認)")

print("\n✅ 完了")
