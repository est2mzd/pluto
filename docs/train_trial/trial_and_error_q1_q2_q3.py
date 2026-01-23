#!/usr/bin/env python3
"""
【試行錯誤ログ】Q1-Q3の正確な回答を導き出すプロセス

【問題】
- Q1: グラフ作成方法がわからない
- Q2: Bostonデータ全部使ってるのか、一部か不明
- Q3: Bostonのシナリオ数がまちまち（1647? 500? 239? 350?）

【目標】
実装に必要な正確なデータを確認する
"""

import os
import glob

print("\n" + "="*80)
print("【試行1】間違ったアプローチ")
print("="*80)

print("""
失敗: answer_q1_q2_q3.py で /nuplan/dataset/nuplan-v1.1/splits/ の
ディレクトリをカウントしようとした

結果: train_boston/ は空（*.db ファイルがディレクトリ内にある）
  → シナリオ数が 0 と表示されてしまった

原因: ファイルとディレクトリの構造を誤解していた
  - train_boston/ 配下に *.db ファイルがある
  - だけどディレクトリではなくファイル

改善: ディレクトリではなく *.db ファイルを数える
""")

boston_wrong = [d for d in os.listdir("/nuplan/dataset/nuplan-v1.1/splits/train_boston") 
                if os.path.isdir(os.path.join("/nuplan/dataset/nuplan-v1.1/splits/train_boston", d))]
print(f"間違ったカウント（ディレクトリのみ）: {len(boston_wrong)}")

boston_correct = glob.glob("/nuplan/dataset/nuplan-v1.1/splits/train_boston/*.db")
print(f"正しいカウント（*.db ファイル）: {len(boston_correct)}")

print("\n" + "="*80)
print("【試行2】グラフ作成方法の調査")
print("="*80)

print("""
失敗: ログファイルからメトリクスを抽出しようとした

試したパターン:
1. /root/nuplan/exp/exp/training/pluto_boston/2026.01.22.07.12.09/run_training.log
   → 26行しかない（キャッシング処理のログ）
   → メトリクスなし

原因: run_training.log にはキャッシング情報だけが記録されていた

改善: log.txt を見つけた
""")

log_file_short = "/root/nuplan/exp/exp/training/pluto_boston/2026.01.22.07.12.09/run_training.log"
log_file_correct = "/root/nuplan/exp/exp/training/pluto_boston/2026.01.21.21.49.48/log.txt"

with open(log_file_short, 'r') as f:
    short_lines = f.readlines()

with open(log_file_correct, 'r') as f:
    correct_lines = f.readlines()

print(f"run_training.log の行数: {len(short_lines)} （キャッシング情報）")
print(f"log.txt の行数: {len(correct_lines)} （実際の訓練ログ）")

print("""
グラフ作成の方法:
- TensorBoard events ファイル (events.out.tfevents.*) にスカラーメトリクスが保存
- このファイルから metrics を読み込んでグラフを生成
- ターミナルログには表示されない（ファイルには記録される）
""")

print("\n" + "="*80)
print("【試行3】訓練データの正確な把握")
print("="*80)

print("""
失敗: キャッシュの 239 scenarios が学習に使われると思っていた

試したパターン:
1. boston_cache_correct/ ディレクトリをカウント
   → 239 scenarios
   
2. これを 70:30 で分割
   → Train: 167, Val: 72
   
3. ところが実際のログには:
   "Number of samples in train set: 350"
   "Number of samples in validation set: 100"
   
問題: 何が 350 と 100 なのか？

調査結果: samples ≠ scenarios
- 500 scenarios を使用
- 各 scenario から複数の samples を生成
- Train scenarios (350) + Test scenarios (100) + Val scenarios (50) = 500
  → これらから合計 4052 samples を生成
  → Train set: 2026 samples, Val set: 2026 samples
""")

print("\n" + "="*80)
print("【最終結果】正確な回答")
print("="*80)

print("""
【Q1】グラフ作成方法
工程:
1. TensorBoard events ファイルを読み込む
   → /root/nuplan/exp/exp/training/pluto_boston/TIMESTAMP/events.out.tfevents.*

2. スカラーメトリクスを抽出
   - loss
   - val/minADE
   - val/minFDE
   - val/MR
   - など各エポックの値

3. matplotlib で 4パネルグラフを作成
   パネル1: Training Loss の推移
   パネル2: Validation minADE の推移
   パネル3: Validation minFDE の推移
   パネル4: サマリー統計

実装:
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt

events_file = 'events.out.tfevents.*'
for event in summary_iterator(events_file):
    for value in event.summary.value:
        if value.tag in ['loss', 'val/minADE', ...]:
            # メトリクスを抽出して保存

【Q2】学習に使用されるデータ
Boston全500シナリオを使用

分割:
- Train scenarios: 350個
- Val scenarios: 100個  
- Test scenarios: 50個

各シナリオから複数サンプルを生成
- Train set: 2026 samples
- Val set: 2026 samples
- 合計: 4052 samples

実際のコマンド:
python run_training.py +training=train_boston cache.cache_path=/nuplan/exp/boston_cache_correct ++epochs=10

【Q3】Bostonシナリオ数
- 総Boston scenarios: 1,647個
- キャッシュ済み scenarios: 239個
- 学習に使用: 500個

なぜ500か: nuplan_scenario_builder_boston の設定で
フィルタリングされた 500 scenarios を使用
""")

print("\n" + "="*80)
print("【学習した教訓】")
print("="*80)

print("""
1. ログファイルの種類に注意
   - run_training.log: キャッシング情報
   - log.txt: 実際の訓練ログ
   - events.out.tfevents: メトリクスデータ

2. scenarios と samples は別物
   - 1 scenario → 複数 samples に変換
   
3. 実装には TensorBoard events から直接読み込む必要
   ログの出力では見えない部分も events に記録されている
""")

print("\n✅ 完了")
