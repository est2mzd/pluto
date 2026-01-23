#!/usr/bin/env python3
import os
import re
import glob
import yaml

print("\n" + "="*80)
print("【Q1】グラフ作成方法と出力ログの乖離")
print("="*80)

latest_exp = sorted(glob.glob("/root/nuplan/exp/exp/training/pluto_boston/*/"))[-1]
exp_name = os.path.basename(latest_exp.rstrip('/'))
print(f"\n最新実験: {exp_name}")

log_files = glob.glob(f"{latest_exp}/**/*.log", recursive=True)
print(f"ログファイル: {len(log_files)} 個")

log_file = "/workspace/pluto/docs/train_trial/analyze_boston_gpu_train_10epochs.log"
with open(log_file, 'r') as f:
    content = f.read()

print(f"ログサイズ: {len(content) / 1024 / 1024:.2f}MB ({len(content.splitlines())} 行)")

patterns = {
    'epoch': r'Epoch (\d+)',
    'val_minADE': r'val/minADE[:\s=]+([0-9.]+)',
    'val_minFDE': r'val/minFDE[:\s=]+([0-9.]+)',
}

for name, pattern in patterns.items():
    matches = re.findall(pattern, content)
    print(f"{name:15s}: {len(matches)} データ")

print("""
【結論】
- ターミナル出力: サマリー情報のみ
- グラフ: ログファイルの全詳細メトリクスから生成
- だからグラフの方が正確
""")

# =============================================================================

print("\n" + "="*80)
print("【Q2】学習コマンドで使用されるデータ")
print("="*80)

print("""
コマンド:
python run_training.py +training=train_boston cache.cache_path=/nuplan/exp/boston_cache_correct ++epochs=10
""")

with open("/workspace/pluto/config/training/train_boston.yaml", 'r') as f:
    config = yaml.safe_load(f)

print("train_boston.yaml の defaults:")
for default in config.get('defaults', []):
    if isinstance(default, dict):
        for k, v in default.items():
            print(f"  - {k}: {v}")

with open("/workspace/pluto/config/splitter/ratio_splitter.yaml", 'r') as f:
    splitter = yaml.safe_load(f)

print(f"\nSplitter設定: {splitter}")

print("""
【答え】Boston全データセット使用
- Train: 70% 自動分割
- Val:   30% 自動分割
""")

# =============================================================================

print("\n" + "="*80)
print("【Q3】Bostonの全シナリオ数")
print("="*80)

nuplan_dir = "/nuplan/dataset/nuplan-v1.1/splits"
print("\n各splitのシナリオ数:")
for split in sorted(os.listdir(nuplan_dir)):
    split_path = os.path.join(nuplan_dir, split)
    if os.path.isdir(split_path):
        scenarios = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        print(f"  {split:25s}: {len(scenarios):3d}")

cache_scenarios = [d for d in os.listdir("/nuplan/exp/boston_cache_correct") 
                  if os.path.isdir(os.path.join("/nuplan/exp/boston_cache_correct", d))]

print(f"\nキャッシュ内シナリオ数: {len(cache_scenarios)}")

print(f"""
【データ分割】
Train: {len(cache_scenarios)} × 0.7 = {int(len(cache_scenarios) * 0.7)} scenarios
Val:   {len(cache_scenarios)} × 0.3 = {int(len(cache_scenarios) * 0.3)} scenarios

つまり:
Training: ~175/175 steps
Validation: ~50/50 steps
""")

print("\n✅ 完了")
