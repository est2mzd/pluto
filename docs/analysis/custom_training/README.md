# custom_training モジュール解説

## 📋 概要

`custom_training` モジュールは、PLUTOモデルの学習（トレーニング）を行うための中核機能を提供します。PyTorch Lightning フレームワークを使用して、データローディング、モデル構築、学習ループの管理を行います。

### 🎯 主な役割

```
入力: 設定ファイル（Hydra Config）
        ↓
   [custom_training モジュール]
        ↓
   - データの準備と読み込み
   - モデル・最適化器の構築
   - 学習ループの実行
   - ログ記録（WandB/TensorBoard）
        ↓
出力: 学習済みモデル（チェックポイント）
```

---

## 📁 ファイル構成

| ファイル | 役割 | 詳細ドキュメント |
|---------|------|---------|
| `custom_datamodule.py` | データ管理・ローディング | [詳細](./custom_datamodule.md) |
| `custom_training_builder.py` | 学習エンジン構築 | [詳細](./custom_training_builder.md) |
| `__init__.py` | モジュール定義 | 外部公開インターフェース |

---

## 🔄 データフロー

### 学習パイプラインの流れ

```
1. 【設定読み込み】
   config/default_training.yaml → Hydra Config
           ↓
2. 【データ準備】(custom_datamodule.py)
   - シナリオの分割（train/val/test）
   - 特徴量の計算と前処理
   - データセットの作成
           ↓
3. 【モデル構築】(custom_training_builder.py)
   - ニューラルネットワークモデル
   - 損失関数・評価指標
   - 最適化器・学習率スケジューラー
           ↓
4. 【トレーニング】(PyTorch Lightning)
   - 各エポックで学習ループ実行
   - 検証データで評価
   - ログ記録（WandB）
           ↓
5. 【チェックポイント保存】
   checkpoints/ ← 最良モデル
```

---

## 🔑 キーコンセプト

### 1. **PyTorch Lightning を使う理由**

通常のPyTorchでは、以下を手書きする必要があります：
```python
# 通常のPyTorch（複雑）
for epoch in range(num_epochs):
    for batch in train_loader:
        # 前処理、学習ステップ、損失計算...
        pass
    for val_batch in val_loader:
        # 検証...
        pass
```

PyTorch Lightningでは、簡潔に記述できます：
```python
# PyTorch Lightning（シンプル）
trainer = pl.Trainer(max_epochs=25)
trainer.fit(model, datamodule)
```

### 2. **Hydra Config システム**

設定をYAMLファイルで管理し、コマンドラインから動的に変更できます：
```bash
python run_training.py \
  py_func=train \
  lr=1e-3 \
  batch_size=32 \
  wandb.mode=online
```

### 3. **データ分割戦略**

- **Train Set**: モデルの重みを更新するデータ
- **Val Set**: 学習中にモデル性能を評価するデータ  
- **Test Set**: 最終的な性能評価用データ（学習には使わない）

---

## 📊 主要クラス・関数

### custom_datamodule.py

| 名前 | 種類 | 説明 |
|------|------|------|
| `CustomDataModule` | クラス | PyTorch Lightning対応のデータモジュール |
| `create_dataset()` | 関数 | データセット作成 |
| `distributed_weighted_sampler_init()` | 関数 | シナリオタイプ別の重み付きサンプリング |

### custom_training_builder.py

| 名前 | 種類 | 説明 |
|------|------|------|
| `TrainingEngine` | クラス | trainer, model, datamoduleを統合 |
| `build_lightning_datamodule()` | 関数 | DataModuleの構築 |
| `build_lightning_module()` | 関数 | モデル・損失関数の構築 |
| `build_custom_trainer()` | 関数 | Trainerの構築（ログ記録を含む） |
| `build_training_engine()` | 関数 | 全体統合（メイン構築関数） |
| `update_config_for_training()` | 関数 | 設定の妥当性チェック |

---

## 🚀 使用例

### 基本的な学習実行

```bash
cd /home/takuya/work/autonomous/pluto

# 最小限の設定で学習
python run_training.py \
  py_func=train \
  +training=train_pluto \
  scenario_builder=nuplan \
  cache.cache_path=/path/to/cache \
  cache.use_cache_without_dataset=true
```

#### 引数の説明

| 引数 | 説明 | 例 |
|------|------|-----|
| `py_func=train` | 実行モード を「訓練」に指定 | `train` / `cache` / `test` |
| `+training=train_pluto` | PLUTOモデル用訓練設定を読み込み（ファイル: `config/training/train_pluto.yaml`） | `train_pluto` |
| `scenario_builder=nuplan` | nuPlanデータセットを使用 | `nuplan` |
| `cache.cache_path=/path/to/cache` | 特徴量キャッシュの保存先 | `/nuplan/exp/cache` |
| `cache.use_cache_without_dataset=true` | キャッシュがあれば、元の .db ファイルなしで学習 | `true` / `false` |

**詳細説明：**
- **`py_func=train`**: スクリプトが「学習」「キャッシュ生成」「テスト」のどれを実行するか指定
  - `train`: モデルの重みを更新
  - `cache`: 特徴量を計算・キャッシュに保存（`use_cache_without_dataset=true` の前に実行が必要）
  - `test`: 学習済みモデルで性能を評価

- **`+training=train_pluto`**: `+` は新しいキー追加を意味する。PLUTO用の訓練設定（損失関数、メトリクスなど）を読み込む

- **`cache.cache_path`**: 以下を含むディレクトリ
  - `/path/to/cache/train/`: 訓練用特徴量
  - `/path/to/cache/val/`: 検証用特徴量
  - `/path/to/cache/test/`: テスト用特徴量

- **`cache.use_cache_without_dataset=true`**: キャッシュから直接読み込む。元の nuPlan .db ファイルが不要になるので、デバイスの空き容量節約

---

### WandB を有効にして実行

```bash
wandb login  # 初回のみ、APIキーを入力（https://wandb.ai/authorize から取得）

python run_training.py \
  py_func=train \
  +training=train_pluto \
  scenario_builder=nuplan \
  cache.cache_path=/path/to/cache \
  cache.use_cache_without_dataset=true \
  wandb.mode=online \
  wandb.project=nuplan-pluto \
  wandb.name=my_experiment
```

#### WandB 引数の説明

| 引数 | 説明 | 例 |
|------|------|-----|
| `wandb.mode=online` | WandBを有効化（`disable` で無効） | `online` / `offline` / `disable` |
| `wandb.project=nuplan-pluto` | WandBプロジェクト名 | 任意（ないなら自動作成） |
| `wandb.name=my_experiment` | ランの表示名（WandBダッシュボードで見える） | `exp_1`, `baseline`, など |

**WandB の動作：**
1. `wandb login` でアカウント認証
2. `mode=online` で、訓練中のメトリクスをリアルタイムでWandBサーバーに送信
3. ブラウザで https://wandb.ai にアクセス → プロジェクトを確認 → グラフを可視化

**`mode=offline` との違い：**
- `online`: リアルタイムログ（インターネット接続が必要）
- `offline`: ローカル保存のみ（後で `wandb sync` で同期可能）
- `disable`: WandB使わない（TensorBoard使用）

---

### 設定パラメータのカスタマイズ

```bash
python run_training.py \
  py_func=train \
  +training=train_pluto \
  scenario_builder=nuplan \
  cache.cache_path=/path/to/cache \
  cache.use_cache_without_dataset=true \
  lr=5e-4 \
  epochs=50 \
  warmup_epochs=5 \
  data_loader.params.batch_size=64 \
  data_loader.params.num_workers=8
```

#### 訓練パラメータの説明

| 引数 | 説明 | デフォルト | 調整のヒント |
|------|------|----------|-----------|
| `lr` | 学習率（大きいと学習が早いが不安定、小さいと遅い） | `1e-3` | GPU VRAM 不足なら小さくする |
| `epochs` | 訓練エポック数 | `25` | 多いほど精度向上（計算時間増） |
| `warmup_epochs` | ウォームアップエポック数（最初は学習率を低くする） | `3` | 不安定さを軽減 |
| `data_loader.params.batch_size` | バッチサイズ（大きいと学習が安定） | `32` | GPU メモリに応じて調整 |
| `data_loader.params.num_workers` | データローディングの並列ワーカー数 | 設定依存 | CPU コア数に応じて調整 |

**具体的な調整例：**

```bash
# GPU メモリが限られている場合
python run_training.py ... \
  data_loader.params.batch_size=16 \
  data_loader.params.num_workers=4

# 高速に訓練したい場合
python run_training.py ... \
  lr=2e-3 \
  epochs=100 \
  data_loader.params.batch_size=128 \
  data_loader.params.num_workers=16

# 精密に訓練したい場合（時間がかかる）
python run_training.py ... \
  lr=1e-4 \
  epochs=200 \
  warmup_epochs=10 \
  data_loader.params.batch_size=16
```

---

### チェックポイントから再開

```bash
# 前回の最良モデルから訓練を続ける
python run_training.py \
  py_func=train \
  +training=train_pluto \
  scenario_builder=nuplan \
  cache.cache_path=/path/to/cache \
  cache.use_cache_without_dataset=true \
  checkpoint=./checkpoints/epoch-24-val_minFDE=0.123.ckpt
```

#### チェックポイント引数

| 引数 | 説明 |
|------|------|
| `checkpoint=./checkpoints/...ckpt` | 保存されたモデルファイルのパス |

**チェックポイント位置:**
```
./checkpoints/
├── epoch-20-val_minFDE=0.145.ckpt  ← 20エポック目
├── epoch-24-val_minFDE=0.123.ckpt  ← 24エポック目（最良）
└── last.ckpt                        ← 最後のエポック
```

---

## ⚙️ 設定ファイル

### 主要な設定項目（YAML）

[default_training.yaml](../../../config/default_training.yaml) に記述される、デフォルト設定：

| 項目 | デフォルト | 説明 |
|------|----------|------|
| `epochs` | `25` | 学習エポック数（繰り返し回数） |
| `warmup_epochs` | `3` | ウォームアップエポック（最初の数エポック） |
| `lr` | `1e-3` | 学習率（重みの更新量） |
| `weight_decay` | `0.0001` | L2正則化（過学習を抑える） |
| `data_loader.params.batch_size` | `32` | 一度に処理するサンプル数 |
| `data_loader.params.num_workers` | 設定依存 | データ読み込みの並列数 |
| `cache.cache_path` | 未設定 | 特徴量キャッシュの保存先 |
| `cache.use_cache_without_dataset` | `false` | キャッシュのみで訓練するか |
| `wandb.mode` | `disable` | `online` (有効) / `disable` (無効) |
| `wandb.project` | `nuplan-pluto` | WandB プロジェクト名 |

### 設定の優先順位

```
コマンドライン引数  > config/default_training.yaml > 各種デフォルト値

例:
python run_training.py lr=5e-4           # ← コマンドライン引数が優先
# (default_training.yaml の lr=1e-3 を上書き)
```

### よく使う設定組み合わせ

#### パターン1: 最小限の設定
```yaml
# default_training.yaml 的な最小設定
py_func: train
epochs: 25
lr: 1e-3
batch_size: 32
wandb:
  mode: disable
```

#### パターン2: 本格的な訓練
```yaml
epochs: 50
lr: 5e-4
warmup_epochs: 5
weight_decay: 0.0001
batch_size: 64
num_workers: 16
wandb:
  mode: online
  project: nuplan-pluto
  name: production_v1
```

#### パターン3: 高速な実験用
```yaml
epochs: 10
lr: 1e-3
batch_size: 128
num_workers: 8
wandb:
  mode: offline  # ローカルでのみ保存
```

---

## 🔧 引数の見方・読み方

### ドット記法（dot notation）

```bash
# ❌ 間違い
python run_training.py batch_size=32

# ✅ 正し
python run_training.py data_loader.params.batch_size=32
```

**理由：** YAML 設定が階層構造になっているため

```yaml
# config.yaml のファイル構造
data_loader:        # レベル1
  params:           # レベル2
    batch_size: 32  # レベル3
```

### `+` と `~` 記号

| 記号 | 意味 | 例 |
|------|------|-----|
| `+key=value` | キーを追加（存在しなかったら追加） | `+training=train_pluto` |
| `~key` | キーを削除 | `~wandb.artifact` |
| `key=value` | キーを上書き | `lr=1e-3` |

### 型の指定

```bash
# 文字列
python run_training.py wandb.project=my_project

# 数値
python run_training.py lr=0.001

# 真偽値
python run_training.py cache.use_cache_without_dataset=true

# リスト
python run_training.py 'worker=single_machine_thread_pool'
```

---

## 🔍 デバッグのヒント

### 問題: メモリ不足
**原因**: バッチサイズが大きすぎる  
**解決**: 以下を減らす
```bash
data_loader.params.batch_size=32  # 元: 64
```

### 問題: データローディングが遅い
**原因**: ワーカー数が不適切  
**解決**: GPU数に応じて調整
```bash
data_loader.params.num_workers=4  # 元: 16
```

### 問題: WandB へのアップロードが遅い
**原因**: ログ頻度が高すぎる  
**解決**: チェックポイント保存頻度を調整
```yaml
lightning:
  trainer:
    checkpoint:
      save_top_k: 3  # 最良モデル3つのみ保存
```

---

## 📚 参考資料

- [PyTorch Lightning 公式ドキュメント](https://lightning.ai/)
- [Hydra 公式ドキュメント](https://hydra.cc/)
- [WandB 公式ドキュメント](https://docs.wandb.ai/)

---

## 次のステップ

詳細は各ファイルのドキュメントを参照してください：

- 📄 [custom_datamodule.py の詳細](./custom_datamodule.md)
- 📄 [custom_training_builder.py の詳細](./custom_training_builder.md)
