# キャッシュ作成コマンドの詳細解説

このドキュメントでは、キャッシュ作成コマンドの各引数について、設定ファイルの場所、定義内容、動作を詳しく解説します。

---

## 1. 実行したコマンド

```bash
python run_training.py \
   py_func=cache \
   +training=train_pluto \
   scenario_builder=nuplan_mini \
   cache.cache_path=/nuplan/exp/sanity_check \
   cache.cleanup_cache=true \
   scenario_filter=training_scenarios_tiny \
   worker=sequential
```

**このコマンドの目的**:
- nuplanデータベースから50個のシナリオを読み込む
- 各シナリオから機械学習用の特徴量とターゲットを計算
- 計算結果を `/nuplan/exp/sanity_check` に保存（キャッシュ化）
- 次回以降の学習時に高速にデータを読み込めるようにする

**実行時間**: 約5〜10分（50シナリオの場合）

---

## 2. `py_func=cache`

### 2.1. 概要

**引数の目的**: 実行モードを指定する。`cache`はシナリオから特徴量を計算してディスクに保存する。

---

### 2.2. 設定ファイルの場所

**デフォルト定義**: `/workspace/pluto/config/default_training.yaml` 行49

```yaml
# Mandatory parameters
py_func: ???                                          # Function to be run inside main (can be "train", "test", "cache")
```

**設定の意味**:
- `???` = 必須パラメータ（コマンドラインで必ず指定する必要がある）
- コメントに記載の通り、`train`, `test`, `cache` のいずれかを指定

### 2.3. コード内での処理

**ファイル**: `/workspace/pluto/run_training.py` 行99-103

```python
elif cfg.py_func == "cache":
    # Precompute and cache all features
    logger.info("Starting caching...")
    with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
        cache_data(cfg=cfg, worker=worker)
    return None
```

**処理の流れ**:
1. `py_func == "cache"` の場合、`cache_data()` 関数を実行
2. この関数は `/workspace/nuplan-devkit/nuplan/planning/training/experiments/caching.py` に定義されている
3. シナリオを読み込み、特徴量とターゲットを計算し、ディスクに保存

### 2.4. 選択肢と違い

| 値 | 動作 | 用途 |
|---|------|------|
| `cache` | キャッシュ作成のみ（学習しない） | データ前処理、開発初期 |
| `train` | キャッシュから読み込んで学習 | モデル学習 |
| `test` | キャッシュから読み込んでテスト | モデル評価 |
| `validate` | キャッシュから読み込んで検証 | モデル検証 |

### 2.5. なぜ `cache` が必要か

**問題**: 生データからの特徴量計算は時間がかかる
- 1シナリオあたり5〜10秒
- 50,000シナリオなら70〜140時間

**解決**: 一度計算してキャッシュに保存
- 初回: 計算 + 保存（時間がかかる）
- 2回目以降: 読み込みのみ（数秒〜数分）

---

## 3. `+training=train_pluto`

### 3.1. 概要

**引数の目的**: Pluto用の設定（モデル、トレーナー、フィルタなど6項目）を1行でまとめて適用する。

---

### 3.2. 設定ファイルの場所と内容

**ファイル**: `/workspace/pluto/config/training/train_pluto.yaml`

```yaml
defaults:
  - override /data_augmentation: contrastive_scenario_generator
  - override /splitter: nuplan
  - override /model: pluto_model
  - override /scenario_filter: training_scenarios_tiny
  - override /custom_trainer: pluto_trainer
  - override /lightning: custom_lightning
```

### この1行で設定される6つの項目

| 設定項目 | 値 | 役割 |
|---------|---|------|
| モデル | `pluto_model` | Plutoのニューラルネットワーク |
| シナリオフィルタ | `training_scenarios_tiny` | 50シナリオに制限 |
| データ分割 | `nuplan` | train/val/test分割 |
| トレーナー | `pluto_trainer` | 学習ロジック |
| データ拡張 | `contrastive_scenario_generator` | データ拡張手法 |
| Lightning設定 | `custom_lightning` | GPU設定など |

### `+` の意味

**Hydraの文法**:
- `+` = 新しいグループを追加
- `+training=train_pluto` で `config/training/train_pluto.yaml` を読み込む

### なぜこれを使うのか？

**使わない場合**（長い）:
```bash
python run_training.py py_func=cache \
  model=pluto_model \
  scenario_filter=training_scenarios_tiny \
  splitter=nuplan \
  custom_trainer=pluto_trainer \
  data_augmentation=contrastive_scenario_generator \
  lightning=custom_lightning \
  ...
```

**使う場合**（短い）:
```bash
python run_training.py py_func=cache +training=train_pluto ...
```

→ **コマンドが短く、タイプミスも減る**

---

## 4. `scenario_builder=nuplan_mini`

### 4.1. 概要

**引数の目的**: nuplanデータベースの読み込み元を指定する。`nuplan_mini`は小規模な開発用データセット。

---

### 4.2. 設定ファイルの場所

**ファイル**: `/workspace/nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan_mini.yaml`

```yaml
_target_: nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.NuPlanScenarioBuilder
_convert_: 'all'

data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini
map_root: ${oc.env:NUPLAN_MAPS_ROOT}
sensor_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs

db_files: null  # if db file(s) exist locally, the data_root is ignored

map_version: nuplan-maps-v1.0

include_cameras: false # Include camera data in the scenarios.

max_workers: null
verbose: ${verbose}

defaults:
  - vehicle_parameters: nuplan_vehicle_parameters
  - scenario_mapping: nuplan_scenario_mapping
```

### パラメータの意味

| パラメータ | 値 | 説明 |
|----------|---|------|
| `_target_` | `NuPlanScenarioBuilder` | 使用するPythonクラス |
| `data_root` | `.../splits/mini` | **miniデータセット**のパス |
| `map_root` | `NUPLAN_MAPS_ROOT` 環境変数 | 地図データのパス |
| `sensor_root` | `.../sensor_blobs` | センサーデータのパス |
| `include_cameras` | `false` | カメラ画像を含めない（軽量化） |

### `nuplan_mini` とは

**miniデータセット**:
- 小規模なサブセット（開発・テスト用）
- 数百〜数千のシナリオ
- データサイズ: 数GB
- ダウンロードと処理が高速

**vs 完全版データセット**:
- `nuplan.yaml` または `nuplan_boston.yaml` を使用
- 数万〜数十万のシナリオ
- データサイズ: 数百GB
- 本番学習用

### 他のscenario_builderオプション

**利用可能な設定**（`/workspace/nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/` 配下）:

| ファイル名 | 用途 | データセット |
|----------|------|-----------|
| `nuplan_mini.yaml` | 開発・テスト | miniデータセット |
| `nuplan.yaml` | 本番学習 | 全データセット |
| `nuplan_boston.yaml` | ボストン限定 | ボストンのみ |
| `nuplan_challenge.yaml` | チャレンジ用 | チャレンジデータ |

### コード内での使用

**処理フロー**:
1. Hydraが `nuplan_mini.yaml` を読み込み
2. `_target_` に基づいて `NuPlanScenarioBuilder` クラスをインスタンス化
3. `data_root` のパスからSQLiteデータベースを開く
4. シナリオ情報をメモリに読み込む

**参照**: `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_building_builder.py` 行12-23

---

## 5. `cache.cache_path=/nuplan/exp/sanity_check`

### 5.1. 概要

**引数の目的**: 計算した特徴量とターゲットを保存するディレクトリパスを指定する。

---

### 5.2. 設定ファイルの場所

**デフォルト定義**: `/workspace/pluto/config/default_training.yaml` 行42-46

```yaml
# Cache parameters
cache:
  cache_path:                                         # Local/remote path to store all preprocessed artifacts from the data pipeline
  use_cache_without_dataset: false                    # Load all existing features from a local/remote cache without loading the dataset
  force_feature_computation: false                    # Recompute features even if a cache exists
  cleanup_cache: false                                # Cleanup cached data in the cache_path, this ensures that new data are generated if the same cache_path is passed
```

### パラメータの意味

**`cache.cache_path`**:
- デフォルト: 空（値なし）
- コマンドラインで指定: `/nuplan/exp/sanity_check`
- 特徴量とターゲットを保存するディレクトリ

### キャッシュディレクトリの構造

```
/nuplan/exp/sanity_check/
├── 2021.05.12.22.00.38_veh-35_01008_01518/    ← log_name（記録ログ名）
│   ├── lane_following/                         ← scenario_type（シナリオタイプ）
│   │   ├── abc123def456.../                    ← scenario_token（シナリオ固有ID）
│   │   │   ├── agents.pkl.gz                   ← 特徴量: 周囲の車両情報
│   │   │   ├── vector_map.pkl.gz               ← 特徴量: 地図情報（車線など）
│   │   │   ├── trajectory.pkl.gz               ← ターゲット: 正解の走行軌跡
│   │   │   └── ...（その他の特徴量）
│   │   ├── xyz789.../                          ← 別のシナリオ
│   │   └── ...（合計50個のシナリオ）
│   └── lane_change/                            ← 別のシナリオタイプ
│       └── ...
├── 2021.06.09.17.23.18_veh-28_00773_01140/    ← 別のログ
│   └── ...
└── cache_metadata_0.csv                        ← メタデータ（統計情報）
```

### ファイル形式

**拡張子**: `.pkl.gz`
- `.pkl` = Pythonのpickle形式（オブジェクトをシリアライズ）
- `.gz` = gzip圧縮（サイズを削減）

**例**: `agents.pkl.gz` の中身（概念）
```python
{
    'ego': {  # 自車の情報
        'position': [[x1, y1], [x2, y2], ...],  # 過去の位置
        'velocity': [[vx1, vy1], [vx2, vy2], ...],
        'heading': [θ1, θ2, ...],
    },
    'agents': [  # 周囲の車両
        {'position': [...], 'velocity': [...], 'heading': [...]},
        {'position': [...], 'velocity': [...], 'heading': [...]},
        ...  # 最大48台
    ]
}
```

### パスの選び方

**推奨**:
- 開発用: `/nuplan/exp/dev_cache` または `/nuplan/exp/sanity_check`
- 本番用: `/nuplan/exp/production_cache` または `/nuplan/exp/cache_1M`

**注意**:
- パスを変えると、異なるキャッシュとして扱われる
- 同じパスを使えば、既存のキャッシュを再利用できる

---

## 6. `cache.cleanup_cache=true`

### 6.1. 概要

**引数の目的**: キャッシュ作成前に既存のキャッシュディレクトリを削除するかどうかを制御する。

---

### 6.2. 設定ファイルの場所

**デフォルト定義**: `/workspace/pluto/config/default_training.yaml` 行46

```yaml
cache:
  cleanup_cache: false                                # Cleanup cached data in the cache_path, this ensures that new data are generated if the same cache_path is passed
```

### パラメータの意味

| 値 | 動作 | 用途 |
|---|------|------|
| `true` | 既存のキャッシュを**削除**してから作成 | クリーンスタート、設定変更後 |
| `false` | 既存のキャッシュを**保持**して追加作成 | キャッシュの追加、再利用 |

### 処理の流れ

**`cleanup_cache=true` の場合**:

```python
# /workspace/pluto/src/custom_training/custom_training_builder.py 行52-56
if cfg.cache.cleanup_cache and Path(cfg.cache.cache_path).exists():
    rmtree(cfg.cache.cache_path)

Path(cfg.cache.cache_path).mkdir(parents=True, exist_ok=True)
```

1. `/nuplan/exp/sanity_check` が存在するかチェック
2. 存在する場合、ディレクトリごと削除（`rmtree`）
3. 新しく空のディレクトリを作成
4. キャッシュ作成を開始

### いつ `true` にすべきか

**`cleanup_cache=true` を使う場合**:
- ✅ 初回実行
- ✅ モデル設定を変更した（特徴量ビルダーのパラメータなど）
- ✅ シナリオフィルタを変更した
- ✅ 古いキャッシュが破損している可能性がある
- ✅ ディスク容量を節約したい

**`cleanup_cache=false` を使う場合**:
- ✅ 既存のキャッシュを保持したい
- ✅ 新しいシナリオを追加でキャッシュしたい
- ✅ 設定を変更していない

### 注意点

**データの消失**:
- `cleanup_cache=true` は**完全削除**
- 作成に時間がかかったキャッシュも削除される
- 実行前に、本当に削除してよいか確認する

**推奨**:
- 開発初期: `cleanup_cache=true`（常にクリーン）
- 安定後: `cleanup_cache=false`（キャッシュ再利用）

---

## 7. `scenario_filter=training_scenarios_tiny`

### 7.1. 概要

**引数の目的**: 読み込むシナリオの選択条件を定義する。`training_scenarios_tiny`は50シナリオに制限。

---

### 7.2. 設定ファイルの場所

**ファイル**: `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml`

```yaml
_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
_convert_: 'all'

scenario_types: null                # List of scenario types to include
scenario_tokens: null               # List of scenario tokens to include

log_names: null                     # Filter scenarios by log names
map_names: null                     # Filter scenarios by map names

num_scenarios_per_type: null        # Number of scenarios per type
limit_total_scenarios: 50           # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
timestamp_threshold_s: null         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
ego_displacement_minimum_m: null    # Whether to remove scenarios where the ego moves less than a certain amount
ego_start_speed_threshold: null     # Limit to scenarios where the ego reaches a certain speed from below
ego_stop_speed_threshold: null      # Limit to scenarios where the ego reaches a certain speed from above
speed_noise_tolerance: null         # Value at or below which a speed change between two timepoints should be ignored as noise.

expand_scenarios: true              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
remove_invalid_goals: true          # Whether to remove scenarios where the mission goal is invalid
shuffle: true                       # Whether to shuffle the scenarios
```

### 主要パラメータの詳細

#### `limit_total_scenarios: 50`

**最も重要な設定**:
- データセット全体から**最大50個**のシナリオを選択
- `null` にすると全シナリオを使用

**開発フェーズごとの推奨値**:

| フェーズ | 推奨値 | 理由 | 処理時間 |
|--------|-------|------|---------|
| 開発・デバッグ | `50` | 高速フィードバック | 数分 |
| 検証 | `500` | より多くのデータでテスト | 数十分 |
| 本番学習 | `null`（制限なし） | 最高性能を目指す | 数時間〜数日 |

#### `shuffle: true`

**ランダム選択**:
- `true`: データセットからランダムに50個を選択
- `false`: 最初の50個を選択（常に同じシナリオ）

**推奨**: `true`（多様なシナリオでテスト）

#### `expand_scenarios: true`

**シナリオの分割**:
- 長いシナリオ（例: 30秒）を短い単位（例: 10秒 × 3個）に分割
- より多くの学習サンプルを生成

#### `remove_invalid_goals: true`

**品質管理**:
- ゴール地点が不正なシナリオを除外
- 学習の品質向上

### 他のフィルタオプション

#### シナリオタイプで絞り込み

```yaml
scenario_types:
  - lane_following        # 車線追従
  - lane_change_left      # 左車線変更
  - lane_change_right     # 右車線変更
```

#### 地図で絞り込み

```yaml
map_names:
  - us-ma-boston          # ボストン
  - us-nv-las-vegas-strip # ラスベガス
```

#### ログ名で絞り込み

```yaml
log_names:
  - 2021.05.12.22.00.38_veh-35_01008_01518
```

### コード内での処理

**処理フロー**:
1. ScenarioBuilderが全シナリオを読み込み
2. ScenarioFilterが条件に基づいて絞り込み
3. 最終的に50個のシナリオがリストアップされる

**参照**: `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_builder.py`

---

## 8. `worker=sequential`

### 8.1. 概要

**引数の目的**: データ処理の並列化方法を指定する。`sequential`は1つずつ順番に処理。

---

### 8.2. 設定ファイルの場所

**ファイル**: `/workspace/nuplan-devkit/nuplan/planning/script/config/common/worker/sequential.yaml`

```yaml
_target_: nuplan.planning.utils.multithreading.worker_sequential.Sequential
_convert_: 'all'
```

### パラメータの意味

**`worker`**: 並列処理の方法を指定

### 利用可能なワーカータイプ

**利用可能な設定**（`/workspace/nuplan-devkit/nuplan/planning/script/config/common/worker/` 配下）:

#### 8.2.1. `sequential`（シーケンシャル）

```yaml
_target_: nuplan.planning.utils.multithreading.worker_sequential.Sequential
```

**特徴**:
- ✅ 1つずつ順番に処理
- ✅ 並列処理なし
- ✅ デバッグしやすい（エラーメッセージが明確）
- ✅ メモリ使用量が少ない
- ❌ 処理時間が長い（50シナリオで5〜10分）

**推奨用途**:
- 開発初期
- デバッグ時
- エラーの原因を特定したいとき
- 小規模データ（50シナリオ程度）

#### 2. `single_machine_thread_pool`（マルチスレッド）

```yaml
_target_: nuplan.planning.utils.multithreading.worker_pool.WorkerPool
```

**特徴**:
- ✅ 複数のCPUコアを使用
- ✅ 高速（sequentialの3〜8倍）
- ✅ 1台のマシンで完結
- ⚠️ メモリ使用量が増加
- ⚠️ エラーメッセージが読みにくい場合がある

**推奨用途**:
- 中規模データ（500〜5,000シナリオ）
- 開発が安定してから
- 単一マシンでの高速化

#### 3. `ray_distributed`（分散処理）

**ファイル**: `/workspace/nuplan-devkit/nuplan/planning/script/config/common/worker/ray_distributed.yaml`

```yaml
_target_: nuplan.planning.utils.multithreading.worker_ray.RayDistributed
_convert_: 'all'
master_node_ip: null    # Set to a master node IP if you desire to connect to cluster remotely
threads_per_node: null  # Number of CPU threads to use per node, "null" means all threads available
debug_mode: false       # If true all tasks will be executed serially, mainly for testing
log_to_driver: true     # If true, all printouts from ray threads will be displayed in driver
logs_subdir: 'logs'     # Subdirectory to store logs inside the experiment directory
use_distributed: false  # Whether to use the built-in distributed mode of ray
```

**特徴**:
- ✅ 複数のマシン（クラスタ）で処理
- ✅ 超高速（数万シナリオでも数時間）
- ✅ 本番環境向け
- ❌ 設定が複雑
- ❌ デバッグが困難

**推奨用途**:
- 大規模データ（1万シナリオ以上）
- 本番学習
- クラスタ環境がある場合

### 処理時間の比較

**50シナリオの場合**:

| Worker | 処理時間 | CPU使用率 | メモリ |
|--------|---------|----------|--------|
| `sequential` | 5〜10分 | 1コア（10%程度） | 2〜4GB |
| `single_machine_thread_pool` | 2〜3分 | 複数コア（80%程度） | 8〜16GB |
| `ray_distributed` | 1〜2分 | 複数マシン | 状況による |

**5,000シナリオの場合**:

| Worker | 処理時間 | 推奨環境 |
|--------|---------|---------|
| `sequential` | 8〜16時間 | 非推奨 |
| `single_machine_thread_pool` | 2〜4時間 | 推奨 |
| `ray_distributed` | 30分〜1時間 | 本番環境 |

### コード内での使用

**処理フロー**:
1. Hydraが `worker=sequential` を読み込み
2. `Sequential` クラスをインスタンス化
3. `cache_data()` 関数内で、各シナリオを順番に処理

```python
# 疑似コード
for scenario in scenarios:  # 50個のシナリオ
    features, targets = compute_features(scenario)
    save_to_cache(features, targets)
```

**参照**: `/workspace/nuplan-devkit/nuplan/planning/utils/multithreading/worker_sequential.py`

### 推奨設定

**開発時（このコマンド）**:
```bash
worker=sequential
```

**本番時**:
```bash
worker=single_machine_thread_pool  # または ray_distributed
```

---

## まとめ: 各引数の役割一覧

| 引数 | 値 | 役割 | 設定ファイル |
|-----|---|------|------------|
| `py_func` | `cache` | キャッシュ作成モードで実行 | `config/default_training.yaml` |
| `+training` | `train_pluto` | Pluto固有設定を一括適用 | `config/training/train_pluto.yaml` |
| `scenario_builder` | `nuplan_mini` | miniデータセットから読み込み | `nuplan-devkit/.../scenario_builder/nuplan_mini.yaml` |
| `cache.cache_path` | `/nuplan/exp/sanity_check` | キャッシュ保存先 | `config/default_training.yaml` |
| `cache.cleanup_cache` | `true` | 既存キャッシュを削除 | `config/default_training.yaml` |
| `scenario_filter` | `training_scenarios_tiny` | 50シナリオに制限 | `config/scenario_filter/training_scenarios_tiny.yaml` |
| `worker` | `sequential` | 1つずつ順番に処理 | `nuplan-devkit/.../worker/sequential.yaml` |

---

## 実行結果: 生成されるファイル

```
/nuplan/exp/sanity_check/
├── [log_name]/
│   ├── [scenario_type]/
│   │   ├── [scenario_token]/
│   │   │   ├── agents.pkl.gz          (周囲の車両情報)
│   │   │   ├── vector_map.pkl.gz      (地図情報)
│   │   │   ├── trajectory.pkl.gz      (正解軌跡)
│   │   │   └── ...
│   │   └── ... (合計50シナリオ)
│   └── ...
└── cache_metadata_0.csv               (統計情報)
```

**次のステップ**: このキャッシュを使って学習を実行
```bash
python run_training.py py_func=train +training=train_pluto \
  cache.cache_path=/nuplan/exp/sanity_check \
  cache.use_cache_without_dataset=true
```

### 引数の種類と定義場所

| 引数 | 値 | 定義ファイル | 説明 |
|-----|----|-----------|----|
| `py_func` | `cache` / `train` | `/workspace/pluto/config/default_training.yaml` 行49 | キャッシュ作成か学習かを選択 |
| `+training` | `train_pluto` | `/workspace/pluto/config/training/train_pluto.yaml` | 学習設定グループの追加（`+`は新規追加） |
| `scenario_builder` | `nuplan_mini` | nuplan-devkit の設定 | シナリオを読み込むデータセット選択 |
| `scenario_filter` | `training_scenarios_tiny` | `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml` | シナリオのフィルタリング設定 |
| `cache.cache_path` | `/nuplan/exp/sanity_check` | `/workspace/pluto/config/default_training.yaml` 行43 | キャッシュの保存先パス |
| `cache.cleanup_cache` | `true` | `/workspace/pluto/config/default_training.yaml` 行46 | 既存キャッシュを削除するか |
| `worker` | `sequential` | Hydra フレームワークの設定 | 並列処理方法（sequential=1個ずつ） |

---

## 2. 設定ファイルの階層構造と定義

### 2.1 ベース設定ファイル

**ファイル**: `/workspace/pluto/config/default_training.yaml` (行1-64)

このファイルは全学習設定の基盤です。

```yaml
# Hydra 設定
hydra:
  run:
    dir: ${output_dir}
  output_subdir: ${output_dir}/code/hydra
  searchpath:
    - pkg://nuplan.planning.script.config.common
    - pkg://nuplan.planning.script.config.training
    - pkg://nuplan.planning.script.experiments
    - config/training

# 設定グループの継承チェーン
defaults:
  - default_experiment              # nuplan-devkit の基本設定
  - default_common                  # nuplan-devkit の共通設定
  - lightning: custom_lightning     # Lightning 設定
  - callbacks: default_callbacks
  - optimizer: adam
  - lr_scheduler: null
  - warm_up_lr_scheduler: null
  - data_loader: default_data_loader
  - splitter: ???                   # 必須パラメータ（上書き必須）
  - objective:
  - training_metric:
  - data_augmentation: null
  - data_augmentation_scheduler: null
  - scenario_type_weights: default_scenario_type_weights
  - custom_trainer: null

# 基本パラメータ
nuplan_trainer: false
experiment_name: 'training'
objective_aggregate_mode: ???       # 必須パラメータ

# キャッシュ設定
cache:
  cache_path:                              # 保存先（コマンドラインで上書き）
  use_cache_without_dataset: false         # キャッシュのみ使用するか
  force_feature_computation: false         # 強制再計算するか
  cleanup_cache: false                     # 既存キャッシュ削除するか

# 学習パラメータ
py_func: ???                        # 必須：cache/train/test/validate
epochs: 25
warmup_epochs: 3
lr: 1e-3
weight_decay: 0.0001
checkpoint:

# Weights & Biases ログ設定
wandb:
  mode: disable
  project: nuplan-pluto
  name: ${experiment_name}
  log_model: all
  artifact:
  run_id:
```

**参照**: `/workspace/pluto/config/default_training.yaml` 行1-64

---

### 2.2 トレーニング設定の上書き

**ファイル**: `/workspace/pluto/config/training/train_pluto.yaml`

このファイルはコマンドの `+training=train_pluto` で指定され、デフォルト設定を上書きします。

```yaml
# @package _global_     # グローバルに適用（ネストなし）
job_name: pluto
py_func: train          # train モード に上書き
objective_aggregate_mode: mean

defaults:
  # 以下の設定を上書き
  - override /data_augmentation: contrastive_scenario_generator
  - override /splitter: nuplan
  - override /model: pluto_model
  - override /scenario_filter: training_scenarios_tiny
  - override /custom_trainer: pluto_trainer
  - override /lightning: custom_lightning
```

**参照**: `/workspace/pluto/config/training/train_pluto.yaml`

**`+training=train_pluto` の意味**:
- `+` = 新規グループを追加（既存グループの上書きではなく追加）
- `training=train_pluto` = `config/training/train_pluto.yaml` を読み込む
- これにより、上記の `defaults` 内のすべての設定が反映される

---

### 2.3 シナリオフィルタ設定

**ファイル**: `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml`

このファイルはシナリオの選択方法を定義します。

```yaml
_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
_convert_: 'all'

# シナリオ選択条件
scenario_types: null                # シナリオタイプで絞らない
scenario_tokens: null               # 特定のシナリオトークンを指定しない
log_names: null                     # ログ名で絞らない
map_names: null                     # 地図で絞らない

# 数量制限（最重要）
num_scenarios_per_type: null        # タイプごとの数を制限しない
limit_total_scenarios: 50           # ★ 合計50シナリオに制限 ★

# 時間・距離フィルタ
timestamp_threshold_s: null
ego_displacement_minimum_m: null
ego_start_speed_threshold: null
ego_stop_speed_threshold: null
speed_noise_tolerance: null

# シナリオ処理方法
expand_scenarios: true              # 長いシナリオを短く分割
remove_invalid_goals: true          # 不正なゴールのシナリオを除外
shuffle: true                       # ランダムに順序を変更
```

**参照**: `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml`

---

### 2.4 モデル設定

**ファイル**: `/workspace/pluto/config/model/pluto_model.yaml`

Plutoモデルのアーキテクチャとハイパーパラメータを定義します。

```yaml
_target_: src.models.pluto.pluto_model.PlanningModel
_convert_: "all"

# モデルアーキテクチャ
dim: 128                    # 埋め込み次元
state_channel: 6           # 状態チャネル数
polygon_channel: 6         # ポリゴンチャネル数
history_channel: 9         # 過去情報のチャネル数
history_steps: 21          # 過去のステップ数（2秒分）
future_steps: 80           # 予測する未来のステップ数（8秒分）
encoder_depth: 4           # エンコーダ層数
decoder_depth: 4           # デコーダ層数
drop_path: 0.2            # DropPath確率
dropout: 0.1              # ドロップアウト確率
num_heads: 4              # Attention のヘッド数
num_modes: 12             # 予測する走行経路の候補数
state_dropout: 0.75
use_ego_history: false
state_attn_encoder: true
use_hidden_proj: false

# 特徴量ビルダ設定
feature_builder:
  _target_: src.feature_builders.pluto_feature_builder.PlutoFeatureBuilder
  _convert_: "all"
  radius: 120              # 周囲の捜索範囲（メートル）
  history_horizon: 2       # 過去2秒分のデータを使用
  future_horizon: 8        # 未来8秒分を予測対象
  sample_interval: 0.1     # 0.1秒間隔でサンプリング
  max_agents: 48           # 周囲の最大エージェント数
  build_reference_line: true
```

**参照**: `/workspace/pluto/config/model/pluto_model.yaml` 行1-40

---

### 2.5 カスタムトレーナ設定

**ファイル**: `/workspace/pluto/config/custom_trainer/pluto_trainer.yaml`

```yaml
_target_: src.models.pluto.pluto_trainer.LightningTrainer
```

このファイルはシンプルに、カスタムトレーナクラスを指定するだけです。

**参照**: `/workspace/pluto/config/custom_trainer/pluto_trainer.yaml`

---

### 2.6 Lightning設定

**ファイル**: `/workspace/pluto/config/lightning/custom_lightning.yaml` (行1-30)

PyTorch Lightning トレーナのパラメータを定義します。

```yaml
distributed_training:
  equal_variance_scaling_strategy: true

trainer:
  checkpoint:
    resume_training: false
    save_top_k: 5                    # ベスト5モデルを保存
    monitor: loss/val_loss           # 検証ロスで監視
    mode: min                        # ロスを最小化

  params:
    max_epochs: ${epochs}            # 25エポック（default_training.yaml から）
    val_check_interval: 1.0          # 毎エポック検証実行
    
    limit_train_batches:             # デフォルトですべて使用
    limit_val_batches:
    limit_test_batches:
    
    devices: -1                      # すべてのGPUを使用
    accelerator: gpu                 # GPU加速
    precision: 32                    # 単精度（FP32）
```

**参照**: `/workspace/pluto/config/lightning/custom_lightning.yaml` 行1-30

---

## 3. 入出力ファイルの関係

### 3.1 全体のフロー図

```
【コマンド実行】
        ↓
python run_training.py \
  py_func=cache \
  +training=train_pluto \
  scenario_builder=nuplan_mini \
  scenario_filter=training_scenarios_tiny \
  cache.cache_path=/nuplan/exp/sanity_check \
  cache.cleanup_cache=true \
  worker=sequential
        ↓
【Hydra 設定構築】
        ↓
config/default_training.yaml（ベース）
  + config/training/train_pluto.yaml（上書き）
    - config/model/pluto_model.yaml（モデル定義）
    - config/scenario_filter/training_scenarios_tiny.yaml（フィルタ）
    - config/custom_trainer/pluto_trainer.yaml（トレーナ）
    - config/lightning/custom_lightning.yaml（Lightning 設定）
        ↓
【入力データ】
        ↓
nuplan-devkit データベース
  ↓ scenario_builder=nuplan_mini で読み込み
  ↓ scenario_filter=training_scenarios_tiny で50シナリオに絞る
        ↓
シナリオオブジェクト（50個）
        ↓
【特徴量抽出】
        ↓
feature_builder（model/pluto_model.yaml で定義）
  → 各シナリオから特徴量を抽出
  → 各シナリオからターゲットを抽出
        ↓
【出力データ（キャッシュ）】
        ↓
/nuplan/exp/sanity_check/
  ├── {log_name}/
  │   ├── {scenario_type}/
  │   │   ├── {scenario_token}/
  │   │   │   ├── agents.pkl.gz         ← feature
  │   │   │   ├── vector_map.pkl.gz     ← feature
  │   │   │   └── trajectory.pkl.gz     ← target
  │   │   └── ...（49個のシナリオ分）
```

---

### 3.2 キャッシュ作成モード（py_func=cache）の詳細フロー

```
【ステップ1】設定読み込み
━━━━━━━━━━━━━━━━━━━━━━━
コマンドライン引数
  ↓ Hydra が処理
  ↓ default_training.yaml + train_pluto.yaml を統合
  ↓
統合設定オブジェクト（cfg）
  - cache.cache_path: /nuplan/exp/sanity_check
  - cache.cleanup_cache: true
  - scenario_filter: training_scenarios_tiny
  - model: pluto_model
  - py_func: cache
  - ...（その他100以上のパラメータ）

【ステップ2】前処理
━━━━━━━━━━━━━━━━━━━━━━━
cache.cleanup_cache=true → /nuplan/exp/sanity_check を削除
                        → ディレクトリを新規作成

【ステップ3】シナリオ読み込み
━━━━━━━━━━━━━━━━━━━━━━━
scenario_builder=nuplan_mini
  ↓ nuplan-devkit のデータベースから読み込み
  ↓ 全シナリオをメモリに読み込み
    （nuplan_mini は小規模なため高速）

scenario_filter=training_scenarios_tiny
  ↓ limit_total_scenarios: 50 で絞る
  ↓ shuffle: true でランダム順序に
  ↓ expand_scenarios: true で長いシナリオを分割
  ↓ remove_invalid_goals: true で不正なものを除外
  ↓
50個のシナリオオブジェクト

【ステップ4】特徴量抽出
━━━━━━━━━━━━━━━━━━━━━━━
model: pluto_model.yaml
  ├─ PlutoFeatureBuilder を生成
  │   ├─ radius: 120 （周囲120m以内の車両を認識）
  │   ├─ history_horizon: 2 （過去2秒）
  │   ├─ future_horizon: 8 （未来8秒）
  │   └─ max_agents: 48 （最大48台の車両を追跡）
  │
  └─ 各シナリオに対して:
     1. 自車の過去軌跡を抽出 → agents.pkl.gz
     2. 周囲の車両情報を抽出 → agents.pkl.gz
     3. 地図情報（車線など）を抽出 → vector_map.pkl.gz
     4. 実際の走行軌跡（正解）を抽出 → trajectory.pkl.gz

【ステップ5】キャッシュ保存
━━━━━━━━━━━━━━━━━━━━━━━
/nuplan/exp/sanity_check/
  ├── 2021.05.12.22.00.38_veh-35.../   ← log_name
  │   ├── lane_following/              ← scenario_type
  │   │   ├── abc123def456.../         ← scenario_token
  │   │   │   ├── agents.pkl.gz        （グリップ圧縮）
  │   │   │   ├── vector_map.pkl.gz
  │   │   │   ├── trajectory.pkl.gz
  │   │   └── (計50個)
  │   └── lane_change/
  │       └── ...
  └── 2021.06.09.17.23.18_.../        （その他のログ）

メタデータファイルも同時作成:
  cache_metadata_0.csv  ← 計算結果の統計情報
```

---

### 3.3 トレーニングモード（py_func=train）の詳細フロー

```
【ステップ1】キャッシュの読み込み
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cache.cache_path: /nuplan/exp/sanity_check
  ↓ この配下のすべてのキャッシュファイルを検索
  ↓ ファイル一覧を取得
  ↓
キャッシュファイルメタデータ

【ステップ2】データ分割（Splitter）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
50個のシナリオ（キャッシュから読み込み）
  ↓ config/training/train_pluto.yaml の defaults で指定
  ↓ override /splitter: nuplan
  ↓
nuplan splitter（nuplan-devkit の標準設定）
  ├─ train: 70% （35個）
  ├─ val:   15% （7個）
  └─ test:  15% （8個）

【ステップ3】バッチ作成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
default_data_loader.yaml
  batch_size: 32（推定値）
  num_workers: 複数のワーカで並列読み込み
  ↓
各バッチ:
  {
    "features": {
      "agents": Tensor[32, 48, 6],         # 32サンプル × 48エージェント × 6次元
      "vector_map": Tensor[32, ..., 6],
    },
    "targets": {
      "trajectory": Tensor[32, 80, 2],     # 32サンプル × 80ステップ（8秒） × 2次元
    }
  }

【ステップ4】モデル定義
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
config/model/pluto_model.yaml から
PlanningModel を生成

モデルのアーキテクチャ:
  入力: features
    ↓
  Encoder（encoder_depth=4 層）
    ↓ dim=128 に埋め込み
    ↓
  Attention（num_heads=4）
    ↓
  Decoder（decoder_depth=4 層）
    ↓
  出力: num_modes=12 個の走行経路候補
        各経路の確率スコア

【ステップ5】損失関数・最適化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
config/training/train_pluto.yaml の defaults:
  - override /custom_trainer: pluto_trainer
    ↓ LightningTrainer クラスで定義
    ↓
損失計算:
  予測軌跡 と 正解軌跡（targets） の差
  ↓
optimizer: adam
  lr: 1e-3（学習率）
  weight_decay: 0.0001

【ステップ6】学習ループ（Lightning 管理）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
config/lightning/custom_lightning.yaml から

for epoch in range(25):  # epochs: 25
  
  【Training フェーズ】
  for batch in train_dataloader:
    predictions = model(batch["features"])
    loss = compute_loss(predictions, batch["targets"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  【Validation フェーズ】
  for batch in val_dataloader:
    predictions = model(batch["features"])
    val_loss = compute_loss(predictions, batch["targets"])
    logger.log(val_loss)
    
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_model()  # ベストモデルを保存

【ステップ7】出力
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
./checkpoints/
  ├── epoch=0-val_minFDE=1.234.ckpt
  ├── epoch=1-val_minFDE=1.145.ckpt
  ├── epoch=2-val_minFDE=1.098.ckpt  ← ベストモデル
  ├── epoch=3-val_minFDE=1.102.ckpt
  └── last.ckpt

${output_dir}/code/hydra/
  └── (Hydra が出力した最終設定ファイル)
```

---

## 4. ファイル一覧と役割

### 設定ファイル（入力）

| ファイルパス | 役割 | キー設定 |
|-----------|------|--------|
| `/workspace/pluto/config/default_training.yaml` | ベース設定 | `py_func`, `cache.*`, `epochs`, `lr` |
| `/workspace/pluto/config/training/train_pluto.yaml` | Pluto固有の上書き設定 | `py_func=train`, モデル・スプリッター選択 |
| `/workspace/pluto/config/model/pluto_model.yaml` | モデルアーキテクチャ | `dim=128`, `num_modes=12`, `radius=120` |
| `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml` | シナリオ選択 | `limit_total_scenarios=50` |
| `/workspace/pluto/config/custom_trainer/pluto_trainer.yaml` | カスタムトレーナ指定 | `_target_` クラスパス |
| `/workspace/pluto/config/lightning/custom_lightning.yaml` | Lightning設定 | `max_epochs=${epochs}`, `devices=-1` |

### データファイル（入出力）

| ファイル/ディレクトリ | 用途 | 作成タイミング |
|-----------------|------|-----------|
| `/nuplan/exp/sanity_check/` | キャッシュ保存先 | `py_func=cache` 実行時 |
| `{log_name}/{scenario_type}/{token}/*.pkl.gz` | 特徴量・ターゲット | キャッシュ作成時 |
| `cache_metadata_*.csv` | キャッシュメタデータ | キャッシュ作成時 |
| `./checkpoints/*.ckpt` | 学習済みモデル | `py_func=train` 実行時 |
| `${output_dir}/code/hydra/` | Hydra出力ファイル | 常に生成 |

---

## 5. コマンドライン引数が設定に反映される仕組み

### Hydra の処理順序

```python
# Hydra が実行時に行うこと

# ステップ1: ベース設定を読み込み
cfg = load_config("default_training.yaml")

# ステップ2: コマンドラインで上書き
cfg = override_config(cfg, {
    "py_func": "cache",
    "training": "train_pluto",           # +training により追加
    "scenario_builder": "nuplan_mini",
    "scenario_filter": "training_scenarios_tiny",
    "cache.cache_path": "/nuplan/exp/sanity_check",
    "cache.cleanup_cache": True,
    "worker": "sequential"
})

# ステップ3: config/training/train_pluto.yaml を適用
# train_pluto.yaml の defaults セクションが実行:
#   - override /data_augmentation: contrastive_scenario_generator
#   - override /splitter: nuplan
#   - override /model: pluto_model
#   - override /scenario_filter: training_scenarios_tiny
#   - override /custom_trainer: pluto_trainer
#   - override /lightning: custom_lightning
# これらが cfg に適用される

# ステップ4: 補間（変数の展開）
# ${epochs} → 25（default_training.yaml から）
# ${output_dir} → 実行時に計算

# ステップ5: main() 関数に cfg を渡す
main(cfg)
```

---

## 6. デバッグ方法：最終設定を確認

Hydra が最終的に決定した設定を確認したい場合は、以下のコマンドを使用：

```bash
python run_training.py --cfg job
```

このコマンドで、統合されたすべての設定がYAML形式で表示されます。

**参照**: `/workspace/pluto/docs/TRAINING_GUIDE.md`（Hydra のデバッグ方法セクション）

---

## 7. 重要なポイント

### ポイント1: `+training=train_pluto` の役割

- `+` なし: 既存グループを上書き
- `+` あり: 新規グループを追加
- `train_pluto.yaml` は `defaults` セクションで複数の設定を一括上書き

```yaml
# train_pluto.yaml
defaults:
  - override /data_augmentation: contrastive_scenario_generator  ← 上書き
  - override /splitter: nuplan                                   ← 上書き
  - override /model: pluto_model                                 ← 上書き
  - override /scenario_filter: training_scenarios_tiny           ← 上書き
  - override /custom_trainer: pluto_trainer                      ← 上書き
  - override /lightning: custom_lightning                        ← 上書き
```

### ポイント2: `limit_total_scenarios: 50` の効果

- 全データセットから **50個だけ** を選択
- 開発時は高速（キャッシュ作成 数分）
- 本番では `null` に変更すれば全データを使用

### ポイント3: キャッシュの再利用

- キャッシュ作成後、`scenario_filter` を変更しなければ
- `py_func=train` で同じキャッシュを再利用
- 高速フィードバック（1回の学習実験が数分で完了）

### ポイント4: ファイルの継承チェーン

```
nuplan-devkit の設定
  ↑
pluto/config/default_training.yaml（ベース）
  ↑
pluto/config/training/train_pluto.yaml（Pluto固有）
  ↑
コマンドラインの上書き（最優先）
  ↑
最終的な cfg オブジェクト
```

---

---

# 学習コマンドの詳細解説

## 実行したコマンド

```bash
python run_training.py \
  py_func=train \
  +training=train_pluto \
  scenario_builder=nuplan_mini \
  scenario_filter=training_scenarios_tiny \
  cache.cache_path=/nuplan/exp/sanity_check \
  worker=sequential
```

**このコマンドの目的**:
- 既存のキャッシュから特徴量とターゲットを読み込む
- データをtrain/val/testに分割
- Plutoモデルを学習
- 学習済みモデルを保存

**実行時間**: 約10〜30分（50シナリオ、25エポックの場合）

---

## キャッシュコマンドとの違い

| 項目 | キャッシュ作成 | 学習 |
|-----|------------|------|
| `py_func` | `cache` | `train` |
| `cache.cleanup_cache` | `true` | **指定なし**（既存を保持） |
| 主な処理 | 特徴量計算・保存 | キャッシュ読込・学習 |
| 出力 | `.pkl.gz`ファイル | `.ckpt`ファイル |

---

## 1. `py_func=train`

### 引数の目的

実行モードを学習に設定する。既存キャッシュからデータを読み込み、モデルを学習する。

### コード内での処理

**ファイル**: `/workspace/pluto/run_training.py` 行54-67

```python
if cfg.py_func == "train":
    # Build training engine
    engine = build_training_engine(cfg, worker)
    
    # Run training
    logger.info("Starting training...")
    engine.trainer.fit(
        model=engine.model,
        datamodule=engine.datamodule,
        ckpt_path=cfg.checkpoint,
    )
    return engine
```

### 学習の流れ

```
1. キャッシュからデータ読み込み
   ↓ cache.cache_path から .pkl.gz ファイルを読む
   
2. データ分割（Splitter）
   ↓ train: 70%, val: 15%, test: 15%
   
3. モデル構築
   ↓ pluto_model.yaml の定義に従う
   
4. 学習ループ（25エポック）
   ↓ optimizer, loss計算, バックプロパゲーション
   
5. モデル保存
   ↓ ./checkpoints/*.ckpt
```

---

## 2. `cache.cleanup_cache` の省略

### 引数の意図

学習時は既存キャッシュを**保持して読み込む**ため、`cleanup_cache`を指定しない。

### キャッシュ作成との違い

| 状況 | cleanup_cache | 動作 |
|-----|--------------|------|
| **キャッシュ作成** | `true` | 既存削除→新規作成 |
| **学習** | 指定なし（falseがデフォルト） | 既存を読み込む |

### もし学習時にcleanup_cache=trueを指定すると？

```bash
# ❌ 誤った使用例
python run_training.py py_func=train cache.cleanup_cache=true ...
```

**結果**: 
1. 既存キャッシュが削除される
2. キャッシュが存在しないためエラー
3. または再度キャッシュ作成が始まる（時間の無駄）

**正しい使用**: 学習時は`cleanup_cache`を指定しない

---

## 3. キャッシュ読み込みの仕組み

### cache.use_cache_without_dataset

**デフォルト設定**: `/workspace/pluto/config/default_training.yaml` 行44

```yaml
cache:
  use_cache_without_dataset: false
```

### 動作モード

| use_cache_without_dataset | 動作 | 用途 |
|--------------------------|------|------|
| `false`（デフォルト） | キャッシュ**と**データベース両方使用 | キャッシュがない場合は計算 |
| `true` | キャッシュ**のみ**使用、DBアクセスなし | 高速、DB不要環境 |

### 推奨設定

**開発時（このコマンド）**:
```bash
# use_cache_without_dataset=false（デフォルト）
# キャッシュ作成直後なので、キャッシュが存在する
# → 高速にキャッシュから読み込まれる
```

**本番時**:
```bash
python run_training.py py_func=train \
  cache.use_cache_without_dataset=true \
  ...
```
→ データベース不要、完全にキャッシュのみで学習

---

## 4. 学習の出力ファイル

### 保存されるファイル

```
./checkpoints/
├── epoch=0-val_minFDE=1.234.ckpt
├── epoch=5-val_minFDE=1.145.ckpt
├── epoch=10-val_minFDE=1.098.ckpt    ← ベストモデル
├── epoch=15-val_minFDE=1.102.ckpt
└── last.ckpt                         ← 最終エポック

./logs/
└── tensorboard または wandb のログ
```

### ファイル名の意味

```
epoch=10-val_minFDE=1.098.ckpt
  │      └─ 検証データでのminFDE（最小最終変位誤差）
  └─ エポック番号
```

### Lightning設定による制御

**ファイル**: `/workspace/pluto/config/lightning/custom_lightning.yaml` 行6-9

```yaml
checkpoint:
  save_top_k: 5              # ベスト5モデルを保存
  monitor: loss/val_loss     # 検証ロスで監視
  mode: min                  # ロスを最小化
```

---

## 5. 学習時の重要な設定

### エポック数

**デフォルト**: `/workspace/pluto/config/default_training.yaml` 行50

```yaml
epochs: 25
```

**変更方法**:
```bash
python run_training.py py_func=train epochs=50 ...
```

### バッチサイズ

**設定場所**: data_loaderの設定（詳細は`train_pluto.yaml`が指定）

### GPU設定

**ファイル**: `/workspace/pluto/config/lightning/custom_lightning.yaml` 行21-23

```yaml
devices: -1        # すべてのGPUを使用
accelerator: gpu   # GPU加速
precision: 32      # 単精度（FP32）
```

---

## まとめ: キャッシュ作成 vs 学習

| 項目 | キャッシュ作成 | 学習 |
|-----|------------|------|
| **py_func** | `cache` | `train` |
| **cleanup_cache** | `true` | 指定なし |
| **主な処理** | 特徴量計算・保存 | キャッシュ読込・学習 |
| **入力** | nuplan DB | キャッシュ（.pkl.gz） |
| **出力** | キャッシュ（.pkl.gz） | モデル（.ckpt） |
| **実行時間** | 5〜10分 | 10〜30分 |
| **実行頻度** | 1回 | 何度でも |

### 開発の流れ

```
1. キャッシュ作成（1回だけ）
   python run_training.py py_func=cache cache.cleanup_cache=true ...
   ↓ 5〜10分

2. 学習（何度でも試行）
   python run_training.py py_func=train ...
   ↓ 10〜30分
   
3. パラメータ調整して再学習
   python run_training.py py_func=train epochs=50 lr=0.0001 ...
   ↓ 繰り返し

※ キャッシュは再利用されるので、2回目以降は高速
```
