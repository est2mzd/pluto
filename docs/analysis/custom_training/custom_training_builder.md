# custom_training_builder.py 詳細解説

## 📝 ファイル概要

このファイルは、PyTorch Lightning を使用した学習エンジン全体を構築します。

**主な責務：**
- データモジュール（DataModule）の構築
- モデル・損失関数・評価指標の構築
- トレーナー（学習ループ管理）の構築
- ログ記録システム（WandB/TensorBoard）の設定

**比喩:** 「工場の組立ラインの設計図」のようなもの

---

## 🔧 主要クラス・関数

### 1. `TrainingEngine` クラス（データクラス）

#### 役割
学習に必要な3つのコンポーネントを1つにまとめる容器。

#### 定義
```python
@dataclass(frozen=True)
class TrainingEngine:
    """Lightning training engine dataclass wrapping the lightning trainer, model and datamodule."""
    trainer: pl.Trainer              # 学習ループ管理
    model: pl.LightningModule        # ニューラルネット
    datamodule: pl.LightningDataModule  # データ管理
```

#### 使用例
```python
# 学習エンジンの取得
engine = build_training_engine(cfg, worker)

# 各コンポーネントにアクセス
engine.trainer      # トレーナー
engine.model        # モデル
engine.datamodule   # データモジュール

# 学習実行
engine.trainer.fit(engine.model, engine.datamodule)
```

#### `frozen=True` の意味
```python
# 作成後は変更不可（不変オブジェクト）
engine.trainer = new_trainer  # ❌ エラー

# これにより、意図しない変更を防ぐ
```

---

### 2. `update_config_for_training()` 関数

#### 役割
設定の妥当性をチェックして、必要に応じて修正する前処理。

#### シグネチャ
```python
def update_config_for_training(cfg: DictConfig) -> None:
```

#### 処理内容

```
入力: cfg（omegaconf 設定オブジェクト）
 ↓
1. 設定を編集可能にする
   OmegaConf.set_struct(cfg, False)
 ↓
2. キャッシュパス処理
   - None なら警告ログ
   - ローカルパスなら自動作成
   - S3パスなら何もしない
 ↓
3. キャッシュ削除が有効なら実行
   Path(cache_path).rmtree()  ← 古いキャッシュを消す
 ↓
4. オーバーフィッティング検査有効なら
   num_workers = 0  ← シングルスレッドに変更
 ↓
5. 設定を読み込み専用に
   OmegaConf.set_struct(cfg, True)
 ↓
6. 最終設定をログ出力
```

#### 具体例

```python
# 【シナリオ1】キャッシュパスが None
cfg.cache.cache_path = None
update_config_for_training(cfg)
# ログ: "Parameter cache_path is not set, caching is disabled"

# 【シナリオ2】ローカルキャッシュパス
cfg.cache.cache_path = "/tmp/cache"
cfg.cache.cleanup_cache = True
update_config_for_training(cfg)
# 動作: /tmp/cache が存在すれば削除
#     新規に /tmp/cache を作成

# 【シナリオ3】S3キャッシュパス
cfg.cache.cache_path = "s3://my-bucket/cache"
update_config_for_training(cfg)
# 動作: S3パスなのでスキップ（クラウド側で管理）
```

---

### 3. `build_lightning_datamodule()` 関数

#### 役割
PyTorch Lightning のデータモジュールを構築する。

#### シグネチャ
```python
def build_lightning_datamodule(
    cfg: DictConfig,           # 設定
    worker: WorkerPool,        # マルチプロセッシング
    model: TorchModuleWrapper  # モデル（必要な特徴量を取得）
) -> pl.LightningDataModule:
```

#### 処理フロー
```
入力: cfg, worker, model
 ↓
1. モデルが必要とする特徴量を取得
   feature_builders = model.get_list_of_required_feature()
 ↓
2. モデルが出力する目標値を取得
   target_builders = model.get_list_of_computed_target()
 ↓
3. train/val/test分割器を構築
   splitter = build_splitter(cfg.splitter)
 ↓
4. 特徴量計算エンジンを構築
   feature_preprocessor = FeaturePreprocessor(...)
 ↓
5. データ拡張（オプション）を構築
   augmentors = build_agent_augmentor(cfg.data_augmentation)
 ↓
6. シナリオを読み込み
   scenarios = build_scenarios(cfg, worker, model)
 ↓
7. カスタムデータモジュールを構築
   datamodule = CustomDataModule(...)
 ↓
出力: datamodule
```

#### 具体的な処理

```python
# 【キー処理】モデル依存の設定
feature_builders = model.get_list_of_required_feature()
# 例えば PLUTOモデルが必要とする:
# - エージェント位置情報
# - マップ情報
# - 過去の軌跡情報
# などを自動認識

# 【キー処理】特徴量計算
feature_preprocessor = FeaturePreprocessor(
    cache_path=cfg.cache.cache_path,  # 前回の計算結果を再利用
    force_feature_computation=cfg.cache.force_feature_computation,
    feature_builders=feature_builders,  # モデルが必要な特徴量
    target_builders=target_builders      # 学習の目標値
)
```

---

### 4. `build_lightning_module()` 関数

#### 役割
PyTorch Lightning モジュール（モデル + 損失関数 + 最適化器）を構築する。

#### シグネチャ
```python
def build_lightning_module(
    cfg: DictConfig,
    torch_module_wrapper: TorchModuleWrapper  # ベースとなるニューラルネット
) -> pl.LightningModule:
```

#### 処理フロー

```
入力: cfg, torch_module_wrapper
 ↓
設定に `custom_trainer` セクションがあるか?
 │
 ├─ YES（カスタム訓練器がある）
 │   ↓
 │  カスタム訓練器を使用
 │   model = instantiate(
 │       cfg.custom_trainer,
 │       model=torch_module_wrapper,
 │       lr=cfg.lr,
 │       weight_decay=cfg.weight_decay,
 │       epochs=cfg.epochs,
 │       warmup_epochs=cfg.warmup_epochs
 │   )
 │
 └─ NO（標準設定）
     ↓
    標準 LightningModuleWrapper を使用
     - 損失関数を build_objectives() で構築
     - 評価指標を build_training_metrics() で構築
     - 最適化器を cfg.optimizer で指定
     - 学習率スケジューラーを設定
     ↓
出力: model
```

#### 具体例

```python
# 【シナリオA】カスタム訓練器使用
# config.yaml に以下がある場合:
# custom_trainer:
#   _target_: src.models.pluto.pluto_trainer.PlutoTrainer

model = PLUTOTrainer(
    model=pluto_model,
    lr=1e-3,
    weight_decay=0.0001,
    epochs=25,
    warmup_epochs=3
)

# 【シナリオB】標準設定
# config.yaml に custom_trainer がない場合:

objectives = build_objectives(cfg)  # [Loss1(), Loss2(), ...]
metrics = build_training_metrics(cfg)  # [Metric1(), Metric2(), ...]

model = LightningModuleWrapper(
    model=pluto_model,
    objectives=objectives,
    metrics=metrics,
    batch_size=32,
    optimizer=cfg.optimizer,
    lr_scheduler=cfg.lr_scheduler,
    warm_up_lr_scheduler=cfg.warm_up_lr_scheduler
)
```

---

### 5. `build_custom_trainer()` 関数

#### 役割
PyTorch Lightning トレーナーを構築。学習ループ、ロギング、チェックポイント保存を管理する。

#### シグネチャ
```python
def build_custom_trainer(cfg: DictConfig) -> pl.Trainer:
```

#### 処理フロー

```
入力: cfg
 ↓
1. トレーナーパラメータ取得
   params = cfg.lightning.trainer.params
   例: max_epochs, gpus, num_nodes など
 ↓
2. コールバック設定
   └─ ModelCheckpoint
      └─ 最良モデルを保存
   └─ RichModelSummary
      └─ モデル構造を表示
   └─ RichProgressBar
      └─ 進捗バーを表示
   └─ LearningRateMonitor
      └─ 学習率の変化を記録
 ↓
3. ロギング設定
   ├─ WandB有効 → WandbLogger
   │  - WandB にリアルタイムログ送信
   │  - アーティファクト（モデル）を保存
   │
   └─ WandB無効 → TensorBoardLogger
      - ローカルに TensorBoard ログ保存
 ↓
4. Trainer 生成
   trainer = pl.Trainer(
       callbacks=callbacks,
       logger=training_logger,
       **params
   )
 ↓
出力: trainer
```

#### WandB 設定の詳細

```python
if cfg.wandb.mode == "disable":
    # WandB を使わない
    training_logger = TensorBoardLogger(...)
else:
    # WandB を使う
    
    # 【前回の実験を続ける場合】
    if cfg.wandb.artifact is not None:
        # アーティファクトを取得
        os.system(f"wandb artifact get {cfg.wandb.artifact}")
        
        # 前回のチェックポイントを読み込む
        checkpoint = os.path.join(os.getcwd(), f"artifacts/{artifact}/model.ckpt")
        run_id = artifact.split(":")[0][-8:]
        
        cfg.checkpoint = checkpoint
        cfg.wandb.run_id = run_id
    
    # WandbLogger を初期化
    training_logger = WandbLogger(
        save_dir=cfg.group,
        project=cfg.wandb.project,      # プロジェクト名
        name=cfg.wandb.name,            # ラン名
        mode=cfg.wandb.mode,            # "online" or "offline"
        log_model=cfg.wandb.log_model,  # "all", "best", or None
        resume=cfg.checkpoint is not None,  # 前回の続きか
        id=cfg.wandb.run_id             # ラン ID
    )
```

#### チェックポイント設定の詳細

```python
ModelCheckpoint(
    dirpath=os.path.join(os.getcwd(), "checkpoints"),  # 保存先
    filename="{epoch}-{val_minFDE:.3f}",               # ファイル名
    monitor=cfg.lightning.trainer.checkpoint.monitor,  # 監視対象（例: val_loss）
    mode=cfg.lightning.trainer.checkpoint.mode,        # "min" or "max"
    save_top_k=cfg.lightning.trainer.checkpoint.save_top_k,  # 上位K個を保存
    save_last=True                                      # 最後のエポックも保存
)

# 例えば save_top_k=3 なら、最良の3個のモデルのみ保存
# 古いモデルは自動削除されてディスク節約
```

---

### 6. `build_training_engine()` 関数（メイン構築関数）

#### 役割
全てのコンポーネントを統合して、`TrainingEngine` オブジェクトを返す。

#### シグネチャ
```python
def build_training_engine(
    cfg: DictConfig,
    worker: WorkerPool
) -> TrainingEngine:
```

#### 処理フロー（全体統合）

```
入力: cfg, worker
 ↓
【ステップ1】前処理
update_config_for_training(cfg)
 ├─ キャッシュパスの妥当性チェック
 └─ 設定値の補正
 ↓
【ステップ2】トレーナー構築
trainer = build_custom_trainer(cfg)
 ├─ コールバック設定
 ├─ ログ記録設定（WandB/TensorBoard）
 └─ GPU/CPU設定
 ↓
【ステップ3】モデル構築
torch_module_wrapper = build_torch_module_wrapper(cfg.model)
 └─ ニューラルネットワークの基本構造
 ↓
【ステップ4】データモジュール構築
datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)
 ├─ 特徴量計算エンジン
 ├─ train/val/test分割
 └─ DataLoader 生成
 ↓
【ステップ5】ライトニングモジュール構築
model = build_lightning_module(cfg, torch_module_wrapper)
 ├─ 損���関数
 ├─ 評価指標
 └─ 最適化器
 ↓
【ステップ6】統合
engine = TrainingEngine(
    trainer=trainer,
    model=model,
    datamodule=datamodule
)
 ↓
出力: engine
```

#### 使用例

```python
# run_training.py などから呼び出し
engine = build_training_engine(cfg, worker)

# 学習実行
engine.trainer.fit(engine.model, engine.datamodule)

# テスト実行（オプション）
engine.trainer.test(engine.model, engine.datamodule)
```

---

## 🔄 全体的なデータフロー

```
【設定ファイル読み込み】
config/default_training.yaml
  ├─ epochs: 25
  ├─ lr: 1e-3
  ├─ batch_size: 32
  ├─ wandb.mode: online
  └─ model, data_loader 等
        ↓
【build_training_engine() 呼び出し】
        ↓
┌─────────────────────────────────────┐
│  update_config_for_training()       │
│  └─ キャッシュパス準備など         │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  build_custom_trainer()             │
│  ├─ callbacks: [checkpoint, ...]    │
│  ├─ logger: WandbLogger             │
│  └─ trainer パラメータ統合          │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  build_torch_module_wrapper()       │
│  └─ ニューラルネット基本構造        │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  build_lightning_datamodule()       │
│  ├─ feature_preprocessor           │
│  ├─ splitter                       │
│  ├─ scenarios                      │
│  └─ CustomDataModule               │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  build_lightning_module()           │
│  ├─ objectives (損失関数)           │
│  ├─ metrics (評価指標)              │
│  └─ optimizer                      │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  TrainingEngine 統合                │
│  ├─ trainer                        │
│  ├─ model                          │
│  └─ datamodule                     │
└──────────┬──────────────────────────┘
           ↓
【学習実行】
trainer.fit(model, datamodule)
  ├─ for epoch in range(max_epochs):
  │   ├─ train loop
  │   │   └─ 各バッチで重みを更新
  │   ├─ val loop
  │   │   └─ 検証データで評価
  │   └─ ログ記録（WandB）
  └─ チェックポイント保存
```

---

## 💡 重要なポイント

### 1. 設定駆動設計（Configuration-driven Design）
```python
# コードの変更なしに、設定ファイルだけで動作変更
- 訓練パラメータ（学習率、エポック数）
- ロギング設定（WandB or TensorBoard）
- モデルアーキテクチャ
- データ拡張方法
# すべて YAML で管理
```

### 2. ビルダーパターン（Builder Pattern）
```python
# 複雑なオブジェクトの生成を段階的に行う
build_custom_trainer()           # Step 1
build_torch_module_wrapper()     # Step 2
build_lightning_datamodule()     # Step 3
build_lightning_module()         # Step 4
TrainingEngine(...)              # Step 5 統合
```

### 3. 依存性注入（Dependency Injection）
```python
# 各関数が必要な依存性を明示的に受け取る
def build_lightning_datamodule(cfg, worker, model):
    # cfg, worker, model があれば動作可能
    # テスト時も簡単にモック可能
```

---

## 🎯 実践的な使用例

### 例1: WandB を使って学習
```python
# run_training.py が内部で実行
cfg = load_config()
cfg.wandb.mode = "online"
cfg.wandb.project = "nuplan-pluto"
cfg.wandb.name = "experiment_1"

engine = build_training_engine(cfg, worker)
engine.trainer.fit(engine.model, engine.datamodule)

# WandB ダッシュボードでリアルタイムログを確認
```

### 例2: 前回の実験を続ける
```python
cfg = load_config()
cfg.wandb.mode = "online"
cfg.wandb.artifact = "my_project/my_run/model-v1:latest"

engine = build_training_engine(cfg, worker)
# 前回のモデルから訓練再開
engine.trainer.fit(engine.model, engine.datamodule)
```

### 例3: TensorBoard でログ記録
```python
cfg = load_config()
cfg.wandb.mode = "disable"  # WandB 無効

engine = build_training_engine(cfg, worker)
# TensorBoard ログが ./logs に保存される
engine.trainer.fit(engine.model, engine.datamodule)

# ブラウザで確認:
# tensorboard --logdir logs
```

---

## 📚 関連ファイル

- [custom_datamodule.py](./custom_datamodule.md) - データ管理
- [../../../config/default_training.yaml](../../../config/default_training.yaml) - 設定
- [../../../run_training.py](../../../run_training.py) - エントリーポイント
- [../../../README.md](../../../README.md) - 使用方法
