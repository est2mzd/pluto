# features モジュール解説

## 📋 概要

`features` は、PLUTOモデルが処理する **データ構造（PlutoFeature）を定義** するモジュールです。

**主な役割：**
- 特徴量、ターゲット、メタデータを1つの構造に統合
- バッチ処理時のデータ結合（collate）機能
- デバイス（GPU/CPU）間のデータ転送

---

## 📁 ファイル構成

| ファイル | 役割 |
|---------|------|
| `pluto_feature.py` | メイン：PlutoFeature クラス定義 |

---

## 🔑 `PlutoFeature` クラス

### 役割
PyTorch Lightning の `AbstractModelFeature` を拡張し、PLUTO用のデータコンテナとして機能。

### 属性

```python
@dataclass
class PlutoFeature(AbstractModelFeature):
    data: Dict[str, Any]        # アンカーサンプル（元のシナリオ）
    data_p: Dict[str, Any]      # ポジティブサンプル（対比学習用）
    data_n: Dict[str, Any]      # ネガティブサンプル（対比学習用）
    data_n_info: Dict[str, Any] # ネガティブのメタ情報
```

### データ構造

```
PlutoFeature
├─ data: {
│   "agent": {position, heading, velocity, ...},
│   "map": {交通信号, ポリゴンなど},
│   "current_state": [x, y, yaw, v, ...],
│   "cost_maps": occupancy grid,
│   "causal": {相互作用情報}
│   }
│
├─ data_p: {同じ構造}  # ポジティブサンプル
│
├─ data_n: {同じ構造}  # ネガティブサンプル
│
└─ data_n_info: {
    "valid_mask": bool,  # 有効か
    "type": int          # 修正方法（0=MAP, 1=AGENT）
}
```

---

## 🔄 `collate()` メソッド（バッチ処理）

### 役割
複数の PlutoFeature インスタンスをバッチに結合。

### 処理フロー

```
入力: feature_list = [PlutoFeature1, PlutoFeature2, ..., PlutoFeatureBatch_Size]
       ↓
【判定】ネガティブサンプルがあるか？
       ├─ YES: 3倍のバッチサイズ（anchor + positive + negative）
       ├─ PARTIAL: 2倍のバッチサイズ（anchor + positive）
       └─ NO: 1倍のバッチサイズ（anchor のみ）
       ↓
【処理】各特徴量を以下のいずれかで結合:
  - pad_sequence(): 可変長データをパディング
  - torch.stack(): 固定長データをスタック
       ↓
出力: batch_data（バッチ化されたテンソル）
```

### 具体例

```python
# 【シナリオ】バッチサイズ 32、ポジティブ+ネガティブあり
feature_list = [PlutoFeature(...), ...] * 32  # 32個

# 【collate 実行】
batch = PlutoFeature.collate(feature_list)

# 【結果】
batch["agent"]["position"].shape
# (96, max_time_steps, 2)  ← 96 = 32 * 3 (anchor + pos + neg)

batch["current_state"].shape
# (96, state_dim)

batch["data_n_valid_mask"].shape
# (32,)  ← ネガティブの有効フラグ（32個のサンプル）
```

### パディングルール

```python
pad_keys = ["agent", "map"]  # 可変長（エージェント数が異なる）
stack_keys = ["current_state", "origin", "angle", "cost_maps"]  # 固定長
```

**例:**
```
Sample A: agent数 = 5
Sample B: agent数 = 8
Sample C: agent数 = 6

パディング後:
├─ Sample A: [5個を8個にパディング（0で埋める）]
├─ Sample B: [8個]
└─ Sample C: [6個を8個にパディング（0で埋める）]

結果: (batch_size=3, max_agents=8, time_steps, 2)
```

---

## 🔄 対比学習時のバッチ構造

### ネガティブサンプルがある場合

```
バッチ構成:
[anchor_1, anchor_2, ..., anchor_32,
 pos_1,    pos_2,    ..., pos_32,
 neg_1,    neg_2,    ..., neg_32]

バッチサイズ: 96 (32*3)

メモリ構成:
data[0:32]:   anchorサンプル
data[32:64]:  positiveサンプル
data[64:96]:  negativeサンプル

メタ情報:
data_n_valid_mask[0:32]: neg_1～neg_32 の有効フラグ
data_n_type[0:32]:       neg_1～neg_32 の修正方法
```

### ネガティブサンプルがない場合

```
バッチ構成:
[anchor_1, anchor_2, ..., anchor_32,
 pos_1,    pos_2,    ..., pos_32]

バッチサイズ: 64 (32*2)
```

---

## 💡 実装のポイント

### 1. Pad Sequence による可変長処理

```python
pad_sequence(
    [f.data["agent"] for f in feature_list],
    batch_first=True
)
# 最も長いシーケンスに合わせてパディング
```

**メリット:**
- メモリ効率が良い（不要なパディングなし）
- 複数 GPU での分散学習に対応

### 2. Stack による固定長処理

```python
torch.stack([
    f.data["current_state"] for f in feature_list
], dim=0)
# すべてのテンソルを新しい次元に積み重ねる
```

**メリット:**
- 高速（コピーなし）
- GPU 計算に最適化

### 3. 対比学習対応

```python
if feature_list[0].data_n is not None:
    # ネガティブサンプルを含める
    batch_data["agent"] = {
        k: pad_sequence(
            [f.data[k] for f in feature_list]
            + [f.data_p[k] for f in feature_list]  # positive
            + [f.data_n[k] for f in feature_list], # negative
            batch_first=True
        )
    }
```

---

## 🚀 使用例

### 基本的な使用

```python
from src.features.pluto_feature import PlutoFeature

# 単一サンプル
feature = PlutoFeature(
    data={...},  # 特徴量データ
    data_p={...} # ポジティブサンプル
)

# バッチ処理
feature_list = [feature1, feature2, ..., featureN]
batch = PlutoFeature.collate(feature_list)
```

### データローダーとの連携

```python
from torch.utils.data import DataLoader

# DataLoader で自動的に collate 関数を呼び出し
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=PlutoFeature.collate  # ← 自動でバッチ化
)

for batch in dataloader:
    # batch は PlutoFeature.collate() の出力
    print(batch["agent"]["position"].shape)
```

---

## 📊 データ型の詳細

### numpy vs torch

```python
# feature_builders から出力
data["agent"]["position"]  # numpy.ndarray

# collate() 後
batch["agent"]["position"]  # torch.Tensor

# GPU 転送前
batch["agent"]["position"].device  # cpu

# GPU 転送後（trainer が自動実行）
batch["agent"]["position"].device  # cuda:0
```

---

## 🔧 正規化・変換

### `PlutoFeature.normalize()` メソッド

```python
normalized_feature = PlutoFeature.normalize(feature.data)
# 値を 0～1 の範囲に正規化
```

### デバイス転送

```python
# CPU → GPU
batch = batch.to(device='cuda:0')

# GPU → CPU
batch = batch.to(device='cpu')
```

---

## 📚 関連ファイル

- [feature_builders/README.md](../feature_builders/README.md) - 特徴量計算
- [custom_training/README.md](../custom_training/README.md) - 訓練パイプライン
