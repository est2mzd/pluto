# train_boston キャッシュ作成・学習実行ログ

📌 **[← 戻る: README.md](README.md)** | このアーカイブは execution_summary.md 完成前の初期ログです

**作成日時**: 2026-01-21
**目的**: train_bostonデータセット用のキャッシュ作成と学習実行の完全ログ
**注記**: このドキュメントは途中で停止しており、完全な実行ログは [execution_summary.md](execution_summary.md) を参照してください。

---

## 初期環境確認

### データセット構成
- **mini**: 64ファイル（開発用小規模）
- **train_boston**: 1,647ファイル（本番用大規模）
- **その他**: train_pittsburgh, train_singapore, test, val

### 既存設定ファイル
- ✅ `/workspace/pluto/config/scenario_filter/training_scenarios_boston.yaml` - 既に存在
- ✅ `/workspace/nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan_boston.yaml` - nuplan-devkit内に存在
- ✅ `/workspace/pluto/config/training/train_pluto.yaml` - 既に存在（miniを使用）

### 計画
1. train_boston用の training yaml を作成
2. キャッシュ作成コマンドを実行
3. 学習コマンドを実行
4. 各ステップの結果を記録

---

## Step 1: train_boston用のtraining設定ファイル作成

**実行内容**:
- `/workspace/pluto/config/training/train_boston.yaml` を作成
- scenario_builder を nuplan_boston に指定
- scenario_filter を training_scenarios_boston に指定
- training_scenarios_boston.yaml の limit_total_scenarios を 100 に設定（開発用）

**ファイル内容**:
```yaml
# @package _global_
job_name: pluto_boston
py_func: train
objective_aggregate_mode: mean

defaults:
  - override /data_augmentation: contrastive_scenario_generator
  - override /splitter: nuplan
  - override /model: pluto_model
  - override /scenario_builder: nuplan_boston
  - override /scenario_filter: training_scenarios_boston
  - override /custom_trainer: pluto_trainer
  - override /lightning: custom_lightning
```

**結果**: ✅ 成功

---

## Step 2: キャッシュ作成実行

**実行コマンド**:
```bash
cd /workspace/pluto && python run_training.py \
  py_func=cache \
  +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_100 \
  cache.cleanup_cache=true \
  worker=sequential
```

**実行開始**:

**実行結果**: ✅ 実行中（バックグラウンド）
- PID: 45233
- ログ: /tmp/boston_cache_full.log
- 進捗: Scenario 1-34/100 が確認され、正常に実行中
- 特になしの警告: RuntimeWarning（nuplan_map/utils.py内）は既知の問題

**進捗状況**（5分後に確認）:

**終了ステータス**: ✅ 成功（2026-01-21 00:25:36）
- 処理時間: 約8分
- 処理シナリオ数: 100/100
- 失敗数: 0 out of 200
- メタデータ: `/nuplan/exp/boston_cache_100_v2/metadata/boston_cache_100_v2_metadata_node_0.csv`

**作成されたキャッシュ構成**:
- 各シナリオごとにディレクトリ作成（100個）
- 各ディレクトリ内に `features/` と `targets/` サブディレクトリ
- キャッシュファイル形式: `.pkl.gz`

---

## Step 3: 学習実行

**実行コマンド**:
```bash
cd /workspace/pluto && python run_training.py \
  py_func=train \
  +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_100_v2 \
  worker=sequential \
  epochs=5
```

**設定説明**:
- `py_func=train`: 学習モード
- `+training=train_boston`: train_boston用設定
- `cache.cache_path`: キャッシュ読み込み元
- `worker=sequential`: シングルスレッド処理
- `epochs=5`: 5エポック（開発用・短縮）

**実行開始**:




