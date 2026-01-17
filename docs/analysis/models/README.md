# models モジュール解説

## 📋 概要

`models` は、PLUTOの **ニューラルネットワークモデル** を実装するモジュールです。

**主な役割：**
- エンコーダー：入力特徴量を圧縮・抽象化
- デコーダー：将来軌跡を生成
- 損失関数：モデルの学習目標

---

## 📁 ファイル構成

```
models/
├── pluto/                        # PLUTOモデル本体
│   ├── pluto_model.py           # メイン: PLUTOモデル
│   ├── pluto_trainer.py         # 訓練エンジン
│   ├── layers/                  # ネットワークレイヤー
│   │   ├── common_layers.py     # 共通レイヤー
│   │   ├── embedding.py         # 埋め込み層
│   │   ├── fourier_embedding.py # フーリエ埋め込み
│   │   ├── mlp_layer.py        # 多層パーセプトロン
│   │   └── transformer.py       # Transformer
│   ├── loss/                    # 損失関数
│   │   └── esdf_collision_loss.py # 衝突回避損失
│   └── modules/                 # モジュール群
│       ├── agent_encoder.py     # エージェント処理
│       ├── agent_predictor.py   # 軌跡予測
│       ├── map_encoder.py       # 地図処理
│       ├── planning_decoder.py  # 計画デコーダ
│       └── static_objects_encoder.py # 静止物体
```

---

## 🔑 PLUTO アーキテクチャ

### 全体図

```
【入力】PlutoFeature
  ├─ agent: (batch, max_agents, time, feat_dim)
  ├─ map: (batch, num_layers, ...)
  └─ current_state: (batch, state_dim)
        ↓
   ┌─────┴─────┬──────────┐
   ↓           ↓          ↓
 Agent       Map        Current
Encoder    Encoder      State
   │           │          │
   └─────┬─────┴──────────┘
         ↓
    Fusion Module
   (マルチモーダル統合)
         ↓
  Planning Decoder
   (軌跡生成)
         ↓
【出力】
  - trajectory: (batch, k=6, time, 2) 複数軌跡
  - probability: (batch, k=6)        確率
```

### 各モジュールの役割

#### 1. Agent Encoder
```
エージェント情報を処理:
  - 位置・速度・形状
  - 時系列情報
  → 圧縮表現
```

#### 2. Map Encoder
```
地図情報を処理:
  - 交通信号
  - ポリゴン
  - コスト地図
  → 空間表現
```

#### 3. Planning Decoder
```
統合された表現から軌跡を生成:
  - 複数の候補軌跡（k=6）
  - 各軌跡の確率
```

---

## 💡 主要テクノロジー

### 1. Transformer
```
自己注意機構 (Self-Attention):
  - エージェント間の相互作用を学習
  - 時系列の長期依存性を捉える
```

### 2. フーリエ埋め込み
```
周期的なパターンを学習:
  - 道路の曲率
  - エージェント間の距離
```

### 3. ESDF 衝突損失
```
衝突回避を学習:
  - Euclidean Signed Distance Field
  - 生成軌跡が障害物から離れるように学習
```

---

## 🚀 使用例

### モデル初期化

```python
from src.models.pluto.pluto_model import PlutoModel

model = PlutoModel(
    history_steps=21,
    future_steps=80,
    num_trajectory_samples=6
)
```

### 推論

```python
with torch.no_grad():
    features, targets = batch
    
    trajectory, probability = model(features)
    # trajectory: (batch, 6, 80, 2)
    # probability: (batch, 6)
```

### 訓練

```python
from src.models.pluto.pluto_trainer import PlutoTrainer

trainer_module = PlutoTrainer(
    model=model,
    lr=1e-3,
    weight_decay=0.0001,
    epochs=25,
    warmup_epochs=3
)
```

---

## 📊 モデルサイズ

| 項目 | 値 |
|------|-----|
| パラメータ数 | ~10M |
| 推論時間 | ~10-50ms |
| GPU メモリ | ~2GB |

---

## 📚 関連ファイル

- [post_processing/README.md](../post_processing/README.md) - 後処理
- [metrics/README.md](../metrics/README.md) - 性能評価
- [planners/README.md](../planners/README.md) - 推論エンジン
