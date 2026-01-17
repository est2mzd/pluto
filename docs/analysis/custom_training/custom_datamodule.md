# custom_datamodule.py 詳細解説

## 📝 ファイル概要

このファイルは、PyTorch Lightning の `LightningDataModule` を拡張し、PLUTOトレーニング用のデータ管理を行います。

**主な責務：**
- 複数のシナリオをtrain/val/testに分割
- 各セットの特徴量を計算・キャッシュ
- DataLoader を作成して PyTorch Lightning に提供

---

## 🔧 主要クラス・関数

### 1. `create_dataset()` 関数

#### 役割
複数のシナリオから、指定された割合でサンプリングしてDatasetを作成します。

#### シグネチャ
```python
def create_dataset(
    samples: List[AbstractScenario],           # 全シナリオ
    feature_preprocessor: FeaturePreprocessor, # 特徴量計算エンジン
    dataset_fraction: float,                   # 使用割合（0.0～1.0）
    dataset_name: str,                         # データセット名（train/val/test）
    augmentors: Optional[List[AbstractAugmentor]] = None  # データ拡張
) -> torch.utils.data.Dataset:
```

#### 処理フロー
```
入力: 100個のシナリオ、dataset_fraction=0.5
     ↓
1. num_keep = 100 * 0.5 = 50
2. 100個からランダムに50個をサンプリング
     ↓
出力: 50個のシナリオを含む Dataset
```

#### 例
```python
# 訓練データセットを作成（全シナリオの60%を使用）
train_dataset = create_dataset(
    samples=train_scenarios,
    feature_preprocessor=preprocessor,
    dataset_fraction=0.6,
    dataset_name="train",
    augmentors=[AugmentorA(), AugmentorB()]
)
```

---

### 2. `distributed_weighted_sampler_init()` 関数

#### 役割
シナリオタイプごとに異なる確率でサンプリングする「重み付きサンプラー」を作成します。

#### 背景
訓練データには、以下のようなシナリオタイプが混在しています：
- 直線走行シナリオ: 多数
- 交差点シナリオ: 少数
- 渋滞シナリオ: 少数

**問題**: そのまま学習すると、直線走行シナリオばかり学習します

**解決**: 重み付きサンプリング
```
直線走行: 重み=1.0 → 50%の確率で選ばれる
交差点:   重み=2.0 → 40%の確率で選ばれる
渋滞:     重み=2.0 → 10%の確率で選ばれる
```

#### シグネチャ
```python
def distributed_weighted_sampler_init(
    scenario_dataset: ScenarioDataset,  # 対象Dataset
    scenario_sampling_weights: Dict[str, float],  # シナリオタイプ→重みの辞書
    replacement: bool = True  # 復元抽出（Trueなら同じシナリオを複数回選択可能）
) -> WeightedRandomSampler:
```

#### 例
```python
# シナリオタイプ別の重み
weights = {
    "straight": 1.0,
    "intersection": 2.0,
    "congestion": 2.0
}

sampler = distributed_weighted_sampler_init(
    scenario_dataset=dataset,
    scenario_sampling_weights=weights,
    replacement=True
)
```

---

### 3. `CustomDataModule` クラス

#### 役割
PyTorch Lightning フレームワークに必要なデータ管理をすべて実装します。

#### クラス図
```
LightningDataModule（抽象基底クラス）
    ↑
    │ 継承
    │
CustomDataModule
```

#### 主要なメソッド

##### `__init__()` - 初期化
```python
def __init__(
    self,
    feature_preprocessor: FeaturePreprocessor,     # 特徴量計算
    splitter: AbstractSplitter,                    # train/val/test分割器
    all_scenarios: List[AbstractScenario],         # 全シナリオ
    train_fraction: float,                         # 訓練データ使用割合
    val_fraction: float,                           # 検証データ使用割合
    test_fraction: float,                          # テストデータ使用割合
    dataloader_params: Dict[str, Any],             # DataLoaderパラメータ
    scenario_type_sampling_weights: DictConfig,    # シナリオ重み設定
    worker: WorkerPool,                            # マルチプロセッシング用
    augmentors: Optional[List[AbstractAugmentor]] = None  # データ拡張
) -> None:
```

**パラメータ解説:**

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `feature_preprocessor` | 特徴量計算エンジン | `FeaturePreprocessor(...)` |
| `splitter` | シナリオを分割 | `RandomSplitter()` |
| `all_scenarios` | 全シナリオリスト | `[scenario1, scenario2, ...]` |
| `train_fraction` | 訓練に使う割合 | `0.7` = 70% |
| `val_fraction` | 検証に使う割合 | `0.15` = 15% |
| `test_fraction` | テストに使う割合 | `0.15` = 15% |
| `dataloader_params` | DataLoaderの設定 | `{"batch_size": 32, "num_workers": 4}` |
| `scenario_type_sampling_weights` | シナリオタイプ別重み | `{"straight": 1.0, "intersection": 2.0}` |
| `worker` | 並列処理用ワーカー | `WorkerPool(num_workers=8)` |
| `augmentors` | データ拡張手法 | `[RandomRotation(), RandomFlip()]` |

##### `setup()` - データセットの準備
```python
def setup(self, stage: Optional[str] = None) -> None:
```

**呼ばれるタイミング:**
- `stage="fit"` → 訓練・検証データセットを準備
- `stage="validate"` → 検証データセットのみ準備
- `stage="test"` → テストデータセットを準備

**内部処理:**
```
stage="fit" の場合:
    ↓
1. splitter.get_train_samples() → 訓練用シナリオ取得
2. create_dataset() → 訓練データセット作成
3. splitter.get_val_samples() → 検証用シナリオ取得
4. create_dataset() → 検証データセット作成
    ↓
self._train_set と self._val_set に保存
```

##### `train_dataloader()` - 訓練用DataLoader生成
```python
def train_dataloader(self) -> torch.utils.data.DataLoader:
```

**動作:**
1. 訓練データセットが準備済みか確認
2. シナリオ重み設定が有効なら、重み付きサンプラーを作成
3. DataLoader を返す

**重要:** 訓練時のシャッフル方法
```python
# 重み付きサンプリングが有効 → 重み付きサンプラー使用
if self._scenario_type_sampling_weights.enable:
    sampler = distributed_weighted_sampler_init(...)
    return DataLoader(shuffle=False, sampler=sampler, ...)  # samplerを使用

# 重み付きサンプリングが無効 → ランダムシャッフル
else:
    return DataLoader(shuffle=True, sampler=None, ...)  # ランダムシャッフル
```

##### `val_dataloader()` - 検証用DataLoader生成
```python
def val_dataloader(self) -> torch.utils.data.DataLoader:
```

**特徴:**
- シャッフルなし（常に同じ順序）
- 重み付きサンプリングなし（順序通りに使用）

##### `test_dataloader()` - テスト用DataLoader生成
```python
def test_dataloader(self) -> torch.utils.data.DataLoader:
```

**特徴:**
- 検証時と同様、シャッフルなし

##### `transfer_batch_to_device()` - GPU/CPU転送
```python
def transfer_batch_to_device(
    self,
    batch: Tuple[FeaturesType, ...],  # バッチ
    device: torch.device,              # 転送先（GPU/CPU）
    dataloader_idx: int
) -> Tuple[FeaturesType, ...]:
```

**役割:**
バッチをGPUメモリに転送（複数のテンソルを正しく転送）

```python
# PyTorch Lightning が自動で呼び出す
batch = (features_tensor, targets_tensor, metadata)
batch_on_gpu = module.transfer_batch_to_device(batch, device=torch.device('cuda'))
```

---

## 🔄 データフロー（詳細版）

```
【初期化】
custom_training_builder.py
  ↓
build_lightning_datamodule()
  ↓
CustomDataModule.__init__()  ← 初期化（データセット未作成）


【学習開始】
PyTorch Lightning
  ↓
trainer.fit(model, datamodule)
  ↓
datamodule.setup(stage="fit")  ← ここでデータセット作成
  ├─ splitter.get_train_samples()
  ├─ create_dataset() → self._train_set
  ├─ splitter.get_val_samples()
  └─ create_dataset() → self._val_set


【各エポック】
for epoch in range(max_epochs):
  ├─ train_dataloader() → バッチ取得
  │   ├─ 重み付きサンプラーで優先度付きサンプリング
  │   ├─ バッチ作成
  │   └─ transfer_batch_to_device() で GPU転送
  │
  └─ val_dataloader() → バッチ取得
      ├─ 順序通りにサンプリング
      ├─ バッチ作成
      └─ transfer_batch_to_device() で GPU転送
```

---

## 💡 実装のポイント

### 1. Lazy Initialization パターン
```python
# __init__では None で初期化
self._train_set: Optional[torch.utils.data.Dataset] = None

# setup() が呼ばれるまで実際のデータセットを作成しない
# メリット: 初期化が高速
```

### 2. 分散学習対応
```python
# DistributedSamplerWrapper を使用
distributed_weighted_sampler = DistributedSamplerWrapper(weighted_sampler)

# 複数GPUで学習する場合も、各GPUが正しくデータを分担
```

### 3. フィーチャ計算の並列化
```python
# feature_preprocessor が効率的に特徴量を計算
# キャッシュを活用して再計算を避ける
```

---

## 🎨 カスタマイズ例

### 例1: データ拡張の追加
```python
augmentors = [
    RandomRotation(degrees=15),
    RandomNoise(std=0.1),
    RandomCrop(size=(128, 128))
]

datamodule = CustomDataModule(
    ...,
    augmentors=augmentors
)
```

### 例2: 不均衡データセットの処理
```python
# シナリオタイプ別の重み設定
scenario_type_sampling_weights = {
    "highway": 1.0,       # 一般的
    "intersection": 3.0,  # 重要度高
    "accident": 5.0,      # 最も重要
}

# 重いシナリオほど多く訓練に使用
```

---

## 🐛 よくあるエラーと対処

### Error: `DataModuleNotSetupError`
```
原因: setup() が呼ばれる前に train_dataloader() を呼び出した
解決: PyTorch Lightning が自動的に setup() を呼ぶので、通常は発生しない
```

### Error: `AssertionError: Train fraction has to be larger than 0!`
```
原因: train_fraction=0.0 で初期化した
解決: train_fraction > 0 にする
```

### Warning: `All scenario sampling weights must be positive`
```
原因: シナリオタイプの重みが負の値
解決: 全ての重みを正の値にする
```

---

## 📚 関連ファイル

- [custom_training_builder.py](./custom_training_builder.md) - 全体統合
- [../../../config/default_training.yaml](../../../config/default_training.yaml) - 設定
- [../../../run_training.py](../../../run_training.py) - エントリーポイント
