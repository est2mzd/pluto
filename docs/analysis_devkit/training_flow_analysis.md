# nuplan-devkit トレーニングフロー解析

## 概要

`/workspace/pluto/run_training.py` が nuplan-devkit を使用してデータ抽出と学習を行う流れを解説する。
すべての記述は実際のコードに基づく。

---

## 1. エントリーポイント

### ファイル
- `/workspace/pluto/run_training.py` (行1-110)

### 主要な処理フロー
```python
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
```

参照: `/workspace/pluto/run_training.py` 行34

### 設定
- `CONFIG_PATH = "./config"`
- `CONFIG_NAME = "default_training"`

参照: `/workspace/pluto/run_training.py` 行29-30

---

## 2. py_func による動作モード

### 2.1 キャッシュモード (py_func="cache")

コマンド例:
```bash
python run_training.py py_func=cache +training=train_pluto scenario_builder=nuplan_mini cache.cache_path=/nuplan/exp/sanity_check cache.cleanup_cache=true scenario_filter=training_scenarios_tiny worker=sequential
```

#### 処理内容
```python
elif cfg.py_func == "cache":
    logger.info("Starting caching...")
    with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
        cache_data(cfg=cfg, worker=worker)
    return None
```

参照: `/workspace/pluto/run_training.py` 行99-103

#### cache_data関数の役割
場所: `/workspace/nuplan-devkit/nuplan/planning/training/experiments/caching.py` 行115

機能:
1. シナリオビルダーを構築 (`build_scenario_builder`)
2. シナリオをフィルタリングして取得
3. 各シナリオに対してfeatureとtargetを計算し、キャッシュに保存

参照: `/workspace/nuplan-devkit/nuplan/planning/training/experiments/caching.py` 行115-176

---

### 2.2 トレーニングモード (py_func="train")

#### 処理内容
```python
if cfg.py_func == "train":
    with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
        engine = build_training_engine(cfg, worker)
    
    logger.info("Starting training...")
    with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
        engine.trainer.fit(
            model=engine.model,
            datamodule=engine.datamodule,
            ckpt_path=cfg.checkpoint,
        )
    return engine
```

参照: `/workspace/pluto/run_training.py` 行54-67

---

## 3. TrainingEngine の構築

### 3.1 カスタム実装版

場所: `/workspace/pluto/src/custom_training/custom_training_builder.py` 行238

```python
def build_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    logger.info("Building training engine...")
    
    trainer = build_custom_trainer(cfg)
    torch_module_wrapper = build_torch_module_wrapper(cfg.model)
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)
    model = build_lightning_module(cfg, torch_module_wrapper)
    
    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)
    return engine
```

参照: `/workspace/pluto/src/custom_training/custom_training_builder.py` 行238-260

### 3.2 TrainingEngine の構成要素

```python
@dataclass(frozen=True)
class TrainingEngine:
    trainer: pl.Trainer  # Trainer for models
    model: pl.LightningModule  # Module describing NN model, loss, metrics, visualization
    datamodule: pl.LightningDataModule  # Loading data
```

参照: `/workspace/pluto/src/custom_training/custom_training_builder.py` 行74-82

---

## 4. データモジュールの構築

### 4.1 build_lightning_datamodule

場所: `/workspace/pluto/src/custom_training/custom_training_builder.py` 行88-138

#### 主要な処理ステップ

1. **Feature BuildersとTarget Buildersの取得**
   ```python
   feature_builders = model.get_list_of_required_feature()
   target_builders = model.get_list_of_computed_target()
   ```
   参照: 行102-103

2. **Splitterの構築**
   ```python
   splitter = build_splitter(cfg.splitter)
   ```
   参照: 行106
   
   訓練データ/検証データ/テストデータの分割を担当

3. **FeaturePreprocessorの作成**
   ```python
   feature_preprocessor = FeaturePreprocessor(
       cache_path=cfg.cache.cache_path,
       force_feature_computation=cfg.cache.force_feature_computation,
       feature_builders=feature_builders,
       target_builders=target_builders,
   )
   ```
   参照: 行109-114

4. **シナリオの構築**
   ```python
   scenarios = build_scenarios(cfg, worker, model)
   ```
   参照: 行124

5. **DataModuleの作成**
   ```python
   datamodule: pl.LightningDataModule = CustomDataModule(
       feature_preprocessor=feature_preprocessor,
       splitter=splitter,
       all_scenarios=scenarios,
       dataloader_params=cfg.data_loader.params,
       augmentors=augmentors,
       worker=worker,
       scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights,
       **cfg.data_loader.datamodule,
   )
   ```
   参照: 行127-136

---

## 5. シナリオの構築フロー

### 5.1 build_scenarios

場所: `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_builder.py` 行162-177

```python
def build_scenarios(cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper) -> List[AbstractScenario]:
    scenarios = (
        extract_scenarios_from_cache(cfg, worker, model)
        if cfg.cache.use_cache_without_dataset
        else extract_scenarios_from_dataset(cfg, worker)
    )
    
    logger.info(f'Extracted {len(scenarios)} scenarios for training')
    assert len(scenarios) > 0, 'No scenarios were retrieved for training, check the scenario_filter parameters!'
    
    return scenarios
```

#### 2つのシナリオ取得方法

1. **キャッシュから取得** (`extract_scenarios_from_cache`)
   - `cfg.cache.use_cache_without_dataset = true` の場合
   - 場所: 行97-143
   - 事前に計算済みのfeature/targetをキャッシュから読み込む

2. **データセットから取得** (`extract_scenarios_from_dataset`)
   - `cfg.cache.use_cache_without_dataset = false` の場合
   - 場所: 行146-159
   - ScenarioBuilderとScenarioFilterを使用してデータベースから直接読み込む

参照: `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_builder.py` 行162-177

---

## 6. ScenarioBuilderとScenarioFilter

### 6.1 ScenarioBuilderの構築

場所: `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_building_builder.py` 行12-23

```python
def build_scenario_builder(cfg: DictConfig) -> AbstractScenarioBuilder:
    logger.info('Building AbstractScenarioBuilder...')
    scenario_builder = instantiate(cfg.scenario_builder)
    validate_type(scenario_builder, AbstractScenarioBuilder)
    logger.info('Building AbstractScenarioBuilder...DONE!')
    return scenario_builder
```

Hydraの`instantiate`を使用して、設定ファイルから動的にインスタンスを生成

### 6.2 ScenarioFilterの例

場所: `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml`

```yaml
_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
_convert_: 'all'

scenario_types: null                # List of scenario types to include
scenario_tokens: null               # List of scenario tokens to include

log_names: null                     # Filter scenarios by log names
map_names: null                     # Filter scenarios by map names

num_scenarios_per_type: null        # Number of scenarios per type
limit_total_scenarios: 50           # Limit total scenarios (float = fraction, int = num)
timestamp_threshold_s: null         # Filter scenarios to ensure scenarios have more than X seconds between their initial lidar timestamps
ego_displacement_minimum_m: null    # Whether to remove scenarios where the ego moves less than a certain amount
ego_start_speed_threshold: null     # Limit to scenarios where the ego reaches a certain speed from below
ego_stop_speed_threshold: null      # Limit to scenarios where the ego reaches a certain speed from above
speed_noise_tolerance: null         # Value at or below which a speed change between two timepoints should be ignored as noise.

expand_scenarios: true              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
remove_invalid_goals: true          # Whether to remove scenarios where the mission goal is invalid
shuffle: true                       # Whether to shuffle the scenarios
```

参照: `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml`

---

## 7. FeaturePreprocessor

### 7.1 役割

場所: `/workspace/nuplan-devkit/nuplan/planning/training/preprocessing/feature_preprocessor.py` 行19-129

```python
class FeaturePreprocessor:
    """
    Compute features and targets for a scenario. This class also manages cache. If a feature/target
    is not present in a cache, it is computed, otherwise it is loaded
    """
```

参照: 行19-23

### 7.2 compute_features メソッド

```python
def compute_features(self, scenario: AbstractScenario) -> Tuple[FeaturesType, TargetsType, List[CacheMetadataEntry]]:
    all_features, all_feature_cache_metadata = self._compute_all_features(scenario, self._feature_builders)
    all_targets, all_targets_cache_metadata = self._compute_all_features(scenario, self._target_builders)
    
    all_cache_metadata = all_feature_cache_metadata + all_targets_cache_metadata
    return all_features, all_targets, all_cache_metadata
```

参照: `/workspace/nuplan-devkit/nuplan/planning/training/preprocessing/feature_preprocessor.py` 行77-96

### 7.3 _compute_all_features メソッド

各builder（FeatureBuilderまたはTargetBuilder）に対して:
```python
for builder in builders:
    feature, feature_metadata_entry = compute_or_load_feature(
        scenario, self._cache_path, builder, self._storing_mechanism, self._force_feature_computation
    )
    all_features[builder.get_feature_unique_name()] = feature
    all_features_metadata_entries.append(feature_metadata_entry)
```

参照: 行108-127

---

## 8. モデルの構築

### 8.1 build_torch_module_wrapper

場所: `/workspace/nuplan-devkit/nuplan/planning/script/builders/model_builder.py` 行12-23

```python
def build_torch_module_wrapper(cfg: DictConfig) -> TorchModuleWrapper:
    logger.info('Building TorchModuleWrapper...')
    model = instantiate(cfg)
    validate_type(model, TorchModuleWrapper)
    logger.info('Building TorchModuleWrapper...DONE!')
    return model
```

Hydraの`instantiate`を使用して、設定ファイルからモデルを動的に生成

### 8.2 Plutoモデルの設定例

場所: `/workspace/pluto/config/model/pluto_model.yaml` (一部)

```yaml
_target_: src.models.pluto.pluto_model.PlanningModel

# ... (model parameters) ...

feature_builders:
  - _target_: src.feature_builders.pluto_feature_builder.PlutoFeatureBuilder
```

参照: `/workspace/pluto/config/model/pluto_model.yaml` 行1, 22

---

## 9. キャッシュシステム

### 9.1 キャッシュ生成プロセス

場所: `/workspace/nuplan-devkit/nuplan/planning/training/experiments/caching.py`

#### cache_scenarios関数 (行27-94)

```python
def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
    cfg: DictConfig = args[0]["cfg"]
    
    model = build_torch_module_wrapper(cfg.model)
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()
    
    preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )
    
    for idx, scenario in enumerate(scenarios):
        features, targets, file_cache_metadata = preprocessor.compute_features(scenario)
        # ... (process results) ...
```

参照: 行27-90

### 9.2 キャッシュの構造

キャッシュパス構造（コードから推測される構造）:
```
{cache_path}/
  ├── {log_name}/
  │   ├── {scenario_type}/
  │   │   ├── {scenario_token}/
  │   │   │   ├── {feature_name}.gz
  │   │   │   ├── {target_name}.gz
```

参照: `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_builder.py` 行70-88 のキャッシュマップ構造

---

## 10. 設定ファイルの階層構造

### 10.1 default_training.yaml

場所: `/workspace/pluto/config/default_training.yaml`

#### Hydra searchpath
```yaml
hydra:
  searchpath:
    - pkg://nuplan.planning.script.config.common
    - pkg://nuplan.planning.script.config.training
    - pkg://nuplan.planning.script.experiments
    - config/training
```

参照: 行5-9

#### defaults (設定の継承構造)
```yaml
defaults:
  - default_experiment
  - default_common
  - lightning: custom_lightning
  - callbacks: default_callbacks
  - optimizer: adam
  - lr_scheduler: null
  - warm_up_lr_scheduler: null
  - data_loader: default_data_loader
  - splitter: ???
  - objective:
  - training_metric:
  - data_augmentation: null
  - data_augmentation_scheduler: null
  - scenario_type_weights: default_scenario_type_weights
  - custom_trainer: null
```

参照: 行12-36

#### キャッシュ設定
```yaml
cache:
  cache_path:                                         # Local/remote path to store all preprocessed artifacts
  use_cache_without_dataset: false                    # Load all existing features from cache without loading dataset
  force_feature_computation: false                    # Recompute features even if cache exists
  cleanup_cache: false                                # Cleanup cached data in the cache_path
```

参照: 行42-46

---

## 11. 処理フロー全体図（テキスト表現）

```
1. run_training.py main()
   ├─ Hydraで設定をロード (default_training.yaml)
   ├─ py_func による分岐
   │
   ├─ [py_func="cache" の場合]
   │  └─ cache_data()
   │     ├─ build_scenario_builder() → ScenarioBuilder作成
   │     ├─ build_scenarios_from_config() → シナリオ取得
   │     ├─ build_torch_module_wrapper() → モデル作成
   │     ├─ model.get_list_of_required_feature() → FeatureBuilders取得
   │     ├─ model.get_list_of_computed_target() → TargetBuilders取得
   │     ├─ FeaturePreprocessor作成
   │     └─ 各シナリオに対して:
   │        └─ preprocessor.compute_features() → feature/target計算・保存
   │
   └─ [py_func="train" の場合]
      └─ build_training_engine()
         ├─ build_torch_module_wrapper() → モデル作成
         ├─ build_lightning_datamodule()
         │  ├─ model.get_list_of_required_feature()
         │  ├─ model.get_list_of_computed_target()
         │  ├─ build_splitter() → データ分割
         │  ├─ FeaturePreprocessor作成
         │  ├─ build_scenarios()
         │  │  ├─ [use_cache_without_dataset=true]
         │  │  │  └─ extract_scenarios_from_cache() → キャッシュから読み込み
         │  │  └─ [use_cache_without_dataset=false]
         │  │     └─ extract_scenarios_from_dataset()
         │  │        ├─ build_scenario_builder()
         │  │        ├─ build_scenario_filter()
         │  │        └─ scenario_builder.get_scenarios()
         │  └─ CustomDataModule作成
         ├─ build_lightning_module() → LightningModule作成
         └─ build_custom_trainer() → Trainer作成
```

---

## 12. まとめ

### データ抽出の仕組み

1. **ScenarioBuilder**: nuplan-devkitのデータベースから運転シナリオを読み込む
2. **ScenarioFilter**: シナリオをフィルタリング（タイプ、トークン、ログ名などで）
3. **FeatureBuilder**: 各シナリオから学習用の特徴量を抽出
4. **TargetBuilder**: 各シナリオから学習用のターゲット（正解ラベル）を抽出
5. **FeaturePreprocessor**: feature/targetの計算とキャッシュ管理を担当

### 学習の仕組み

1. **PyTorch Lightning**: 学習フレームワーク
2. **TrainingEngine**: Trainer、Model、DataModuleをまとめたデータクラス
3. **CustomDataModule**: シナリオデータをPyTorch Lightningの形式で提供
4. **Splitter**: データをtrain/val/testに分割

### キャッシュの役割

- 一度計算したfeature/targetをディスクに保存
- 次回以降の学習で再計算が不要になり、高速化
- `cache.use_cache_without_dataset=true` でデータベースアクセスなしで学習可能

---

## 参照元ファイル一覧

1. `/workspace/pluto/run_training.py` (行1-110)
2. `/workspace/pluto/src/custom_training/custom_training_builder.py` (行1-260)
3. `/workspace/nuplan-devkit/nuplan/planning/training/experiments/caching.py` (行1-176)
4. `/workspace/nuplan-devkit/nuplan/planning/training/experiments/training.py` (行1-100)
5. `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_building_builder.py` (行1-23)
6. `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_builder.py` (行1-205)
7. `/workspace/nuplan-devkit/nuplan/planning/script/builders/model_builder.py` (行1-23)
8. `/workspace/nuplan-devkit/nuplan/planning/training/preprocessing/feature_preprocessor.py` (行1-129)
9. `/workspace/pluto/config/default_training.yaml` (行1-64)
10. `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml` (全体)
11. `/workspace/pluto/config/model/pluto_model.yaml` (行1, 22)
