# 初心者向け：nuplan-devkit 学習システムの仕組み

## はじめに：このシステムは何をしているのか？

自動運転車を賢くするためには、**実際の運転データから学習させる**必要があります。
このシステムは以下の3つのことを行います：

1. 📦 **データの準備**: 大量の運転記録から、学習に使える形式にデータを変換する
2. 🧠 **モデルの学習**: 変換したデータを使って、AIモデルに運転の仕方を教える
3. 💾 **効率化**: 一度変換したデータを保存して、次回から素早く使えるようにする

---

## なぜ複雑なシステムが必要なのか？

### 問題1: データがそのままでは使えない

自動運転車が記録する生データは、以下のような情報です：
- 🚗 車の位置、速度
- 🗺️ 地図情報
- 🚶 周囲の歩行者や他の車の位置
- 📹 カメラ画像
- 📡 センサーデータ

しかし、AIモデルに学習させるには：
- 「この状況で次にどう動くべきか」という**答え（正解ラベル）**
- AIが理解できる**数値の形式**に変換された特徴量

が必要です。

### 問題2: データ量が膨大

- nuplanデータセットには数千〜数万の運転シナリオ
- 1つのシナリオを変換するのに数秒〜数十秒かかる
- 毎回変換していたら、学習開始まで何時間もかかる

### 解決策: nuplan-devkitの設計思想

nuplan-devkitは、この問題を**段階的に処理**することで解決します：

```
生データ → フィルタリング → 特徴量抽出 → キャッシュ保存 → 学習
```

---

## 全体の流れ：2つの主要なモード

### モード1: キャッシュ作成モード（準備段階）

**やること**: データを変換して保存する

```bash
python run_training.py py_func=cache \
  cache.cache_path=/nuplan/exp/sanity_check \
  scenario_filter=training_scenarios_tiny
```

**引数の意味**:
- `py_func=cache`: キャッシュ作成モードで実行
- `cache.cache_path=/nuplan/exp/sanity_check`: キャッシュの保存先ディレクトリ
- `scenario_filter=training_scenarios_tiny`: 使用するシナリオフィルタ設定（50シナリオに制限）
  - **50シナリオ制限の根拠**: 開発時は高速フィードバック重視。全データ（数万シナリオ）でのキャッシュ作成は1時間以上かかるが、50シナリオなら数分で完了。デバッグやパラメータ調整の試行錯誤が効率的（詳細は「2. ScenarioFilter」の「『50シナリオ制限』の根拠」を参照）

このコマンドは：
1. データベースから運転シナリオを読み込む
2. 各シナリオから特徴量（features）とターゲット（targets）を計算
3. 計算結果をディスクに保存

**なぜ必要？**
- 毎回データ変換すると時間がかかりすぎる
- 一度変換すれば、何度でも再利用できる
- 学習の試行錯誤が高速になる

**参照**: `/workspace/pluto/run_training.py` 行99-103

---

### モード2: 学習モード（本番）

**やること**: 保存したデータでモデルを学習

```bash
python run_training.py py_func=train \
  cache.cache_path=/nuplan/exp/sanity_check \
  cache.use_cache_without_dataset=true
```

**引数の意味**:
- `py_func=train`: 学習モードで実行
- `cache.cache_path=/nuplan/exp/sanity_check`: 読み込むキャッシュのディレクトリ
- `cache.use_cache_without_dataset=true`: データベースを使わずキャッシュのみ使用

このコマンドは：
1. 保存済みのキャッシュからデータを読み込む
2. データを学習用/検証用に分割
3. AIモデルに学習させる

**なぜ必要？**
- キャッシュから読み込むので高速
- モデルやパラメータを変えて何度も試せる

**参照**: `/workspace/pluto/run_training.py` 行54-67

---

## 詳細解説：各コンポーネントの役割

### 1. ScenarioBuilder（シナリオビルダー）

#### 何をするもの？
nuplanデータベースから**運転シナリオ**を読み込むツール

#### 運転シナリオとは？
- 実際の道路で記録された、数秒〜数十秒の運転の様子
- 例：「交差点で左折する」「高速道路で車線変更する」など

#### なぜ必要？
データベースは巨大で複雑な構造なので、直接読むのは大変。
ScenarioBuilderが、必要なデータだけを取り出してくれる。

#### 設定例
```yaml
scenario_builder: nuplan_mini  # miniデータセットを使う
```

**参照**: `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_building_builder.py` 行12-23

---

### 2. ScenarioFilter（シナリオフィルタ）

#### 何をするもの？
読み込んだシナリオを**絞り込む**ツール

#### なぜ絞り込むのか？

**理由1: 開発時は少量で試したい**
- 全データで学習すると時間がかかる
- まずは少量（50個など）で動作確認

**理由2: 特定の状況だけ学習させたい**
- 「車線変更のシナリオだけ」
- 「ボストン市内のデータだけ」など

#### 設定例
```yaml
# /workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml
limit_total_scenarios: 50          # 最大50個まで
shuffle: true                      # ランダムに選ぶ
expand_scenarios: true             # 長いシナリオを短く分割
remove_invalid_goals: true         # 不正なゴール地点のシナリオを除外
```

#### 具体的にできること

| パラメータ | 用途 | 例 |
|----------|------|-----|
| `limit_total_scenarios` | シナリオ数を制限 | 開発時は50個、本番は10000個 |
| `scenario_types` | シナリオの種類で絞る | 車線変更、右折、左折など |
| `log_names` | 特定の記録ログだけ | 特定の日時や場所のデータ |
| `map_names` | 特定の地図だけ | ボストン、ラスベガスなど |

**参照**: `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml`

#### 「50シナリオ制限」の根拠

**training_scenarios_tiny.yaml の設定**:
```yaml
limit_total_scenarios: 50
```

**なぜ50シナリオに制限するのか？**

| 理由 | 詳細 | 実際の時間 |
|-----|------|----------|
| **開発効率** | 全データ（数万シナリオ）でテストするのは時間がかかる | キャッシュ作成: 1時間 vs 20時間 |
| **デバッグしやすさ** | エラーが出たときに、少ないデータなら問題箇所の特定が早い | バグ修正: 数分 vs 数時間 |
| **メモリ節約** | 少ないシナリオなら、低スペック開発マシンでも実行可能 | メモリ使用量: 1GB vs 50GB |
| **高速フィードバック** | コード変更 → テスト → 結果確認 のループが高速 | 1サイクル: 数分 vs 1時間以上 |
| **コスト削減** | 開発用GPUサーバー不要（CPU付きPCで十分） | 開発マシン: ノートPC vs 高性能サーバー |

**使い分けの基準**:

```
【開発フェーズ】
↓ limit_total_scenarios: 50
↓ 機能確認、バグ修正、パラメータ調整
↓ 高速サイクル重視

【検証フェーズ】
↓ limit_total_scenarios: 500～1,000
↓ より多くのデータで動作確認
↓ 本番に近い環境での検証

【本番フェーズ】
↓ limit_total_scenarios: null（制限なし）
↓ 全データを使用（数万シナリオ）
↓ 最高の性能を目指す
```

**実例**:
```yaml
# 開発用
limit_total_scenarios: 50          # training_scenarios_tiny.yaml

# テスト用（仮想）
# limit_total_scenarios: 500      # training_scenarios_500.yaml (存在しない場合は自作)

# 本番用
# limit_total_scenarios: null     # training_scenarios_full.yaml (制限なし)
```

**重要**: 開発では少量のデータで高速テストを繰り返し、本番では全データを使って最高性能を引き出す。この使い分けが開発効率の鍵です。

---

### 3. FeatureBuilder（特徴量ビルダー）

#### 何をするもの？
シナリオから**AIが理解できる数値データ**を作る

#### なぜ必要？

AIモデルは数値しか理解できません。生データをそのまま渡しても学習できません。

#### 変換の例

**元のデータ（人間には分かる）**:
- 自車の位置: 緯度42.123, 経度-71.456
- 前の車との距離: 15.3メートル
- 速度: 時速50km

**変換後（AIが処理できる）**:
- 自車を中心とした座標系での相対位置
- 正規化された速度 (0〜1の範囲)
- 時系列データ（過去2秒分の軌跡）

#### Plutoモデルの例

PlutoFeatureBuilderは以下を作成：
- 周囲の車両の位置と速度
- 道路の車線情報
- 交通信号の状態
- 過去の軌跡

**参照**: `/workspace/pluto/config/model/pluto_model.yaml` 行22

---

### 4. TargetBuilder（ターゲットビルダー）

#### 何をするもの？
**正解データ（教師データ）**を作る

#### なぜ必要？

AIの学習には「入力」と「正解」の両方が必要です。
- 入力 = FeatureBuilder が作る特徴量
- 正解 = TargetBuilder が作るターゲット

#### 具体例

**シナリオの状況**:
- 現在時刻: 0秒
- 記録データには20秒先までの実際の走行軌跡がある

**TargetBuilderがすること**:
- 未来8秒間の理想的な走行軌跡を抽出
- モデルの予測と比較できる形式に変換

**学習の流れ**:
1. モデル: 「次の8秒間、こう走るべきだと予測します」
2. ターゲット: 「実際にはこう走りました」
3. 両者の差を計算して、モデルを修正

---

### 5. FeaturePreprocessor（特徴量プリプロセッサ）

#### 何をするもの？
FeatureBuilderとTargetBuilderを**管理**し、**キャッシュ機能**を提供

#### 主な役割

**役割1: 複数のBuilderを統合**
- 1つのシナリオに対して、複数のFeatureBuilder/TargetBuilderを実行
- 結果をまとめて返す

**役割2: キャッシュの読み書き**
```python
# 疑似コード
if キャッシュに保存済み:
    return キャッシュから読み込み
else:
    計算する
    キャッシュに保存
    return 計算結果
```

**役割3: エラーハンドリング**
- 変換に失敗したシナリオをスキップ
- エラー情報をログに記録

**なぜ必要？**
- 個々のBuilderは単機能だが、実際は複数のBuilderを組み合わせる
- キャッシュ機能により、2回目以降が高速化
- エラーがあっても処理を続行できる

**参照**: `/workspace/nuplan-devkit/nuplan/planning/training/preprocessing/feature_preprocessor.py` 行19-129

---

### 6. Splitter（スプリッター）

#### 何をするもの？
全シナリオを**学習用/検証用/テスト用**に分割

#### なぜ3つに分けるのか？

**学習用（Train）**: モデルを訓練するデータ
- 例: 全体の70%

**検証用（Validation）**: 学習中に性能を確認
- 学習中に定期的に検証
- 「学習がうまくいっているか」をチェック
- 例: 全体の15%

**テスト用（Test）**: 最終的な性能評価
- 学習が終わった後に1回だけ使う
- 「本当に性能が出ているか」を確認
- 例: 全体の15%

#### なぜ分ける必要がある？

**過学習（オーバーフィッティング）を防ぐため**

- 学習用データだけで評価すると、答えを暗記しているだけかもしれない
- 未知のデータ（検証用/テスト用）で評価して、本当に賢くなったか確認

**参照**: `/workspace/pluto/src/custom_training/custom_training_builder.py` 行106

---

### 7. DataModule（データモジュール）

#### 何をするもの？
学習フレームワーク（PyTorch Lightning）にデータを渡すための**インターフェース**

#### 具体的な役割

**役割1: データの準備**
```python
def prepare_data():
    # キャッシュからシナリオを読み込む
    # or データベースからシナリオを読み込む
```

**役割2: データの分割**
```python
def setup():
    # Splitterを使って train/val/test に分割
```

**役割3: バッチの作成**
```python
def train_dataloader():
    # 学習用データを32個ずつまとめて返す（バッチサイズ32の場合）
```

#### なぜ必要？

PyTorch Lightningという学習フレームワークの**決まった形式**に合わせるため。
- フレームワークが自動でデータを読み込んでくれる
- 並列処理、GPU転送なども自動化

**参照**: `/workspace/pluto/src/custom_training/custom_training_builder.py` 行127-136

---

### 8. Model（モデル）

#### 何をするもの？
**実際のAI（ニューラルネットワーク）**

#### モデルの役割

**入力**: FeatureBuilderが作った特徴量
- 周囲の車両情報
- 道路情報
- 過去の軌跡

**処理**: ニューラルネットワークで計算

**出力**: 未来の走行計画
- 「次の8秒間、このように走るべき」という軌跡

#### 2つのレイヤー

**1. TorchModuleWrapper（モデル本体）**
- ニューラルネットワークの構造
- 推論処理（予測）
- 必要な特徴量/ターゲットの定義

**2. LightningModuleWrapper（学習ロジック）**
- 損失関数（予測と正解の誤差計算）
- オプティマイザ（モデルの修正方法）
- 学習率スケジューラ（学習率の調整）

**なぜ2層に分かれている？**
- TorchModule: モデルのコアロジック（推論だけでも使える）
- LightningModule: 学習に必要な追加機能（学習時のみ使う）

**参照**: 
- TorchModuleWrapper: `/workspace/nuplan-devkit/nuplan/planning/script/builders/model_builder.py` 行12-23
- LightningModuleWrapper: `/workspace/pluto/src/custom_training/custom_training_builder.py` 行161-176

---

### 9. Trainer（トレーナー）

#### 何をするもの？
学習プロセス全体を**管理**するコントローラー

#### 具体的にやること

**学習ループの実行**:
```python
for epoch in range(epochs):  # 25エポック繰り返す
    for batch in train_dataloader:  # 学習データを順番に処理
        予測 = model(batch)
        誤差 = 予測と正解の差
        モデルを修正
    
    for batch in val_dataloader:  # 検証
        検証スコアを計算
    
    ベストモデルを保存
```

**その他の機能**:
- GPU/CPUの管理
- チェックポイント保存（途中で中断しても再開可能）
- ログの記録（TensorBoard, Weights & Biasesなど）
- 学習率の調整
- Early Stopping（改善しなくなったら自動停止）

**なぜ必要？**
- 学習の定型処理を自動化
- ベストプラクティスが組み込まれている
- コードがシンプルになる

**参照**: `/workspace/pluto/src/custom_training/custom_training_builder.py` 行181-236

---

## キャッシュシステムの詳細

### なぜキャッシュが重要なのか？

#### 計算時間の比較

**キャッシュなし（毎回計算）**:
```
シナリオ読込: 1秒
特徴量計算: 5秒
学習: 1時間
----------------
合計: 1時間 + (1秒+5秒) × シナリオ数
```

50,000シナリオの場合: **約85時間**

**キャッシュあり（1回目だけ計算）**:
```
【1回目】
シナリオ読込: 1秒
特徴量計算: 5秒
キャッシュ保存: 1秒
----------------
合計: 7秒 × 50,000 = 約97時間（一度だけ）

【2回目以降】
キャッシュ読込: 0.1秒
学習: 1時間
----------------
合計: 約1時間
```

### キャッシュの仕組み

#### ディレクトリ構造

```
/nuplan/exp/sanity_check/          # cache_path
├── 2021.05.12.22.00.38_veh-35_01008_01518/    # log_name
│   ├── lane_following/                         # scenario_type
│   │   ├── abc123.../                          # scenario_token
│   │   │   ├── agents.pkl.gz                   # feature
│   │   │   ├── vector_map.pkl.gz               # feature
│   │   │   ├── trajectory.pkl.gz               # target
│   │   └── def456.../
│   └── lane_change/
└── 2021.06.09.17.23.18_veh-28_00773_01140/
```

**参照**: `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_builder.py` 行70-88

#### キャッシュの利用方法

**方法1: キャッシュだけ使う（データベース不要）**
```yaml
cache:
  cache_path: /nuplan/exp/sanity_check
  use_cache_without_dataset: true     # データベースにアクセスしない
```

メリット:
- 超高速（ディスクからの読み込みのみ）
- データベースが不要（他の環境でも使える）

デメリット:
- キャッシュ作成時と同じscenario_filterしか使えない

**方法2: データベースから読んで、キャッシュも使う**
```yaml
cache:
  cache_path: /nuplan/exp/sanity_check
  use_cache_without_dataset: false    # データベースも使う
```

メリット:
- scenario_filterを変えられる
- キャッシュがあるものは読み込み、ないものは計算

デメリット:
- データベースへのアクセスが必要

**参照**: `/workspace/pluto/config/default_training.yaml` 行42-46

---

## 設定ファイル（Hydra）の役割

### Hydraとは？

設定を**YAMLファイル**で管理し、**コマンドラインから簡単に上書き**できる仕組み

### なぜ設定ファイルが必要？

#### 問題: パラメータが多すぎる

学習には100以上のパラメータがあります：
- モデルのアーキテクチャ
- 学習率、エポック数
- データの分割方法
- キャッシュの設定
- ログの設定
- など

#### 解決: 階層的な設定管理

```
default_training.yaml (ベース設定)
  ├─ model: pluto_model.yaml
  ├─ splitter: boston_splitter.yaml
  ├─ scenario_filter: training_scenarios_tiny.yaml
  ├─ optimizer: adam.yaml
  └─ lightning: custom_lightning.yaml
```

### 使い方の例

#### 基本的な使い方
```bash
# デフォルト設定で実行
python run_training.py
```

#### 設定の上書き
```bash
# エポック数を変更
python run_training.py epochs=50

# 複数の設定を変更
python run_training.py epochs=50 lr=0.001 batch_size=64
```

#### 設定グループの切り替え
```bash
# モデルを変更
python run_training.py model=other_model

# シナリオフィルタを変更
python run_training.py scenario_filter=training_scenarios_1M
```

#### 設定の追加
```bash
# カスタム設定を追加（+をつける）
python run_training.py +training=train_pluto
```

**参照**: `/workspace/pluto/config/default_training.yaml` 行1-11

---

## 実際の実行例

### 例1: 開発時（少量データで動作確認）

```bash
python run_training.py \
  py_func=cache \
  scenario_builder=nuplan_mini \
  scenario_filter=training_scenarios_tiny \
  cache.cache_path=/nuplan/exp/dev_test \
  cache.cleanup_cache=true \
  worker=sequential
```

**各引数の意味**:
- `py_func=cache`: キャッシュ作成モードで実行
- `scenario_builder=nuplan_mini`: miniデータセット（小規模）を使用
- `scenario_filter=training_scenarios_tiny`: シナリオを50個に制限
- `cache.cache_path=/nuplan/exp/dev_test`: キャッシュの保存先
- `cache.cleanup_cache=true`: 既存キャッシュを削除してクリーンスタート
- `worker=sequential`: 並列処理なし（1つずつ順番に処理、デバッグしやすい）

**何をしている？**
1. miniデータセット（小規模）を使用
2. 50シナリオだけ抽出
3. 既存キャッシュを削除してクリーンスタート
4. 並列処理なし（デバッグしやすい）
5. キャッシュを作成

### 例2: 本番学習（大規模データ）

**ステップ1: キャッシュ作成**
```bash
python run_training.py \
  py_func=cache \
  scenario_builder=nuplan \
  scenario_filter=training_scenarios_1M \
  cache.cache_path=/nuplan/exp/production \
  worker=ray_distributed
```

**引数の意味**:
- `scenario_builder=nuplan`: 完全版データセット（大規模）を使用
- `scenario_filter=training_scenarios_1M`: 100万シナリオを使用
- `worker=ray_distributed`: Ray分散処理で並列実行（高速化）

**ステップ2: 学習実行**
```bash
python run_training.py \
  py_func=train \
  model=pluto_model \
  splitter=boston_splitter \
  cache.cache_path=/nuplan/exp/production \
  cache.use_cache_without_dataset=true \
  epochs=25 \
  batch_size=32
```

**引数の意味**:
- `py_func=train`: 学習モードで実行
- `model=pluto_model`: Plutoモデルを使用
- `splitter=boston_splitter`: ボストンデータ用のデータ分割設定
- `cache.use_cache_without_dataset=true`: データベース不要、キャッシュのみ使用
- `epochs=25`: 25エポック（全データを25回繰り返し学習）
- `batch_size=32`: 1バッチ32サンプル（一度に32個のシナリオを処理）

**何が違う？**
- 100万シナリオを使用
- 分散処理で高速化（ray_distributed）
- キャッシュ作成と学習を分離
- 大規模なバッチサイズ

---

## よくある質問

### Q1: なぜこんなに複雑なの？シンプルにできない？

**A**: 自動運転のデータは特殊です

**一般的な画像認識**:
- 画像ファイルをそのまま読めばOK
- データ変換が簡単

**自動運転**:
- データベースから複数のテーブルを結合
- 時系列データの処理
- 地図情報との照合
- 複雑な特徴量エンジニアリング

この複雑さを**段階的に処理**することで、各部分を理解しやすくしています。

---

### Q2: キャッシュを使わずに、毎回計算したらダメ？

**A**: 可能ですが、非常に遅くなります

**実験の回数**:
- モデル開発中は100回以上の学習実験が必要
- パラメータ調整、アーキテクチャ変更など

**時間の比較**:
- キャッシュあり: 1実験 1時間 × 100回 = 100時間
- キャッシュなし: 1実験 10時間 × 100回 = 1000時間

**差**: 900時間（約37日）の節約

---

### Q3: scenario_filterとsplitterの違いは？

**scenario_filter**: データセット全体からシナリオを選ぶ
- 「どのシナリオを使うか」を決める
- キャッシュ作成前に実行
- 例: 「ボストンのデータだけ」「車線変更シナリオだけ」

**splitter**: 選んだシナリオを分割する
- 「選んだシナリオをどう分けるか」を決める
- 学習開始前に実行
- 例: 「70%を学習用、15%を検証用、15%をテスト用」

```
全データセット (10万シナリオ)
    ↓ scenario_filter
選択されたシナリオ (5万シナリオ)
    ↓ splitter
学習用 (3.5万) / 検証用 (0.75万) / テスト用 (0.75万)
```

---

### Q4: feature_buildersとtarget_buildersの違いは？

**feature_builders**: AIへの「入力」を作る
- 現在の状況（自車、他車、道路など）
- 過去の軌跡
- センサー情報

**target_builders**: AIの「正解」を作る
- 実際にどう走ったか（未来の軌跡）
- 理想的な行動

**学習の流れ**:
```
features → [AIモデル] → 予測
targets → 正解

誤差 = |予測 - 正解|
→ モデルを修正
```

---

### Q5: py_func=cacheは毎回実行する必要がある？

**A**: いいえ、1回だけでOK（条件が変わらない限り）

**再実行が必要な場合**:
- scenario_filterを変更した
- feature_builders/target_buildersを変更した
- データセットを更新した
- キャッシュファイルを削除した

**再実行が不要な場合**:
- モデルのアーキテクチャを変更
- 学習率やエポック数を変更
- splitterを変更

キャッシュは**入力データ**に依存し、**学習パラメータ**には依存しません。

---

## まとめ：全体像の理解

### データフロー（簡略版）

```
📦 nuplanデータベース
  ↓
🔍 ScenarioBuilder（読み込み）
  ↓
🎯 ScenarioFilter（絞り込み）
  ↓
🔧 FeatureBuilder（特徴量作成）
🔧 TargetBuilder（正解作成）
  ↓
💾 キャッシュ保存
  ↓
📊 Splitter（データ分割）
  ↓
🎓 DataModule（バッチ作成）
  ↓
🧠 Model（学習）
```

### 各コンポーネントの存在理由

| コンポーネント | なぜ必要？ |
|------------|---------|
| ScenarioBuilder | DBの複雑さを隠蔽 |
| ScenarioFilter | 開発効率化、特定状況の学習 |
| FeatureBuilder | AIが理解できる形式に変換 |
| TargetBuilder | 教師データの作成 |
| FeaturePreprocessor | Builderの管理、キャッシュ機能 |
| キャッシュ | 計算の再利用、高速化 |
| Splitter | 過学習防止、正確な性能評価 |
| DataModule | フレームワークとの橋渡し |
| Model | 実際の学習・推論 |
| Trainer | 学習プロセスの自動化 |

### 2段階アプローチの利点

**段階1: キャッシュ作成** (`py_func=cache`)
- 時間がかかるが、1回だけ
- データ変換の正しさを確認できる
- 並列処理で高速化可能

**段階2: 学習** (`py_func=train`)
- キャッシュから読み込むので高速
- 何度でも試行錯誤できる
- モデルやパラメータの実験に集中

この分離により、**開発サイクルが大幅に高速化**されます。

---

## 次のステップ

このガイドで全体像が理解できたら：

1. **小規模データで試す**: `training_scenarios_tiny.yaml`（50シナリオ）
2. **キャッシュの動作確認**: `py_func=cache`でキャッシュ作成
3. **学習の実行**: `py_func=train`で学習開始
4. **パラメータ調整**: 学習率やエポック数を変えて実験
5. **大規模化**: データ量を増やして本格的な学習

---

## 参照

この文書は以下のソースコードに基づいています：

- `/workspace/pluto/run_training.py`
- `/workspace/pluto/src/custom_training/custom_training_builder.py`
- `/workspace/nuplan-devkit/nuplan/planning/training/experiments/caching.py`
- `/workspace/nuplan-devkit/nuplan/planning/script/builders/scenario_builder.py`
- `/workspace/nuplan-devkit/nuplan/planning/training/preprocessing/feature_preprocessor.py`
- `/workspace/pluto/config/default_training.yaml`
- `/workspace/pluto/config/scenario_filter/training_scenarios_tiny.yaml`

詳細な実装は [training_flow_analysis.md](training_flow_analysis.md) を参照してください。
