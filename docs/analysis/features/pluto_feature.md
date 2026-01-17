# PlutoFeature データ構造と変換 詳細ガイド

## 📋 概要

`PlutoFeature` は、PLUTO モデルが処理する **全データの統一フォーマット** を定義します。

---

## 🏗️ クラス構造

### データクラス定義

```python
@dataclass
class PlutoFeature(AbstractModelFeature):
    data: Dict[str, Any]  # すべての特徴量情報
    
    @staticmethod
    def collate(features: List["PlutoFeature"]) -> Dict:
        """
        バッチ内の複数特徴量を統合
        
        入力: [feature_1, feature_2, ..., feature_batch_size]
        出力: 各フィールドが一次元大きくなったdict
        """
```

---

## 📦 内部構造の詳細

### data フィールドの完全構造

```python
data = {
    # ============ Ego 状態情報 ============
    "current_state": torch.tensor([
        ego_x, ego_y,           # 位置 [m]
        ego_yaw,                # 向き [rad]
        ego_vel_x, ego_vel_y,   # 速度 [m/s]
        ego_acc_x, ego_acc_y,   # 加速度 [m/s²]
        ego_steer,              # ステアリング角度 [rad]
        ego_steer_rate          # ステアリング速度 [rad/s]
    ], dtype=torch.float32),    # Shape: (10,)
    
    # 座標系の原点（グローバル座標）
    "origin": torch.tensor([
        ego_x, ego_y
    ], dtype=torch.float32),    # Shape: (2,)
    
    # Ego の向き（回転マトリックスのための角度）
    "angle": torch.tensor(ego_yaw, dtype=torch.float32),  # Shape: ()
    
    # ============ エージェント情報 ============
    "agent": {
        # 位置履歴: (max_agents=64, time_steps=101, 2)
        "position": torch.zeros((64, 101, 2), dtype=torch.float32),
        
        # 向き履歴: (64, 101)
        "heading": torch.zeros((64, 101), dtype=torch.float32),
        
        # 速度履歴: (64, 101, 2)
        "velocity": torch.zeros((64, 101, 2), dtype=torch.float32),
        
        # エージェントサイズ: (64, 2) = [width, length]
        "shape": torch.zeros((64, 2), dtype=torch.float32),
        
        # エージェント種類: (64,) = {0: 車, 1: 歩行者, 2: 自転車, ...}
        "category": torch.zeros((64,), dtype=torch.int64),
        
        # エージェント有効フラグ: (64,) = {True: 有効, False: パディング}
        "valid_mask": torch.zeros((64,), dtype=torch.bool),
        
        # 将来の正解軌跡: (64, future_steps=80, 3) = (x, y, yaw)
        "target": torch.zeros((64, 80, 3), dtype=torch.float32),
    },
    
    # ============ マップ情報 ============
    "map": {
        # 各セマンティックレイヤーのポリゴン
        "polygon_lane": [...],              # レーン境界
        "polygon_road_edge": [...],         # 道路端
        "polygon_crosswalk": [...],         # 横断歩道
        "polygon_stop_line": [...],         # 停止線
        
        # 交通信号
        "polygon_tl_status": torch.tensor([...]),  # ステータス
        "polygon_tl_id": torch.tensor([...]),      # 信号 ID
        
        # 標識
        "polygon_sign": [...],
    },
    
    # ============ オプション: コスト地図 ============
    "cost_maps": torch.zeros((500, 500), dtype=torch.float32),
    # occupancy grid: 0=自由, 1=障害物
    
    # ============ 因果関係情報 ============
    "causal": {
        "interaction_label": torch.zeros((64,), dtype=torch.int64),
        "leading_agent_mask": torch.zeros((64,), dtype=torch.bool),
        ...
    },
    
    # ============ メタ情報 ============
    "scenario_name": "mini_demo_scenario_0",
    "log_name": "dataset_v1.1",
    "timestamp": 123456789
}
```

---

## 🔄 変換メソッド

### 1️⃣ `to_tensor()` - NumPy から Tensor への変換

```python
def to_tensor(
    self,
    device: str = "cpu"
) -> "PlutoFeature":
    """
    すべての配列を PyTorch Tensor に変換
    """
    
    converted_data = {}
    
    for key, value in self.data.items():
        if isinstance(value, dict):
            # ネストされたdict（エージェント情報など）
            converted_data[key] = {
                subkey: torch.from_numpy(subval).to(device)
                if isinstance(subval, np.ndarray)
                else subval
                for subkey, subval in value.items()
            }
        elif isinstance(value, np.ndarray):
            # NumPy配列 → Tensor
            converted_data[key] = torch.from_numpy(value).to(device)
        else:
            # すでに Tensor か スカラー
            converted_data[key] = value
    
    return PlutoFeature(data=converted_data)

# 使用例
feature_gpu = feature_cpu.to_tensor(device="cuda:0")
```

### 2️⃣ `to_numpy()` - Tensor から NumPy への変換

```python
def to_numpy(self) -> "PlutoFeature":
    """
    すべての Tensor を NumPy 配列に変換
    """
    
    converted_data = {}
    
    for key, value in self.data.items():
        if isinstance(value, dict):
            # ネストされたdict
            converted_data[key] = {
                subkey: subval.cpu().numpy()
                if isinstance(subval, torch.Tensor)
                else subval
                for subkey, subval in value.items()
            }
        elif isinstance(value, torch.Tensor):
            # Tensor → NumPy
            converted_data[key] = value.cpu().numpy()
        else:
            # NumPy か スカラー
            converted_data[key] = value
    
    return PlutoFeature(data=converted_data)

# 使用例
feature_np = feature_tensor.to_numpy()
```

### 3️⃣ `to_device()` - デバイス間の移動

```python
def to_device(self, device: str) -> "PlutoFeature":
    """
    Tensor をデバイス間で移動（CPU ↔ GPU）
    """
    
    converted_data = {}
    
    for key, value in self.data.items():
        if isinstance(value, dict):
            converted_data[key] = {
                subkey: subval.to(device)
                if isinstance(subval, torch.Tensor)
                else subval
                for subkey, subval in value.items()
            }
        elif isinstance(value, torch.Tensor):
            converted_data[key] = value.to(device)
        else:
            converted_data[key] = value
    
    return PlutoFeature(data=converted_data)

# 使用例
feature_gpu = feature_cpu.to_device("cuda:0")
feature_cpu2 = feature_gpu.to_device("cpu")
```

---

## 🎁 `collate()` メソッド - バッチ統合の詳細

### 目的

複数のシナリオから生成された特徴量を、**モデルが処理可能なバッチ形式に統合**

### 実装例

```python
@staticmethod
def collate(features: List["PlutoFeature"]) -> Dict[str, Any]:
    """
    入力: 
        features = [
            PlutoFeature(data={...}),
            PlutoFeature(data={...}),
            PlutoFeature(data={...}),
        ]  # batch_size = 3
    
    処理: 各フィールドのstackingまたはconcatenation
    
    出力: バッチ化されたdict
    """
    
    batch_size = len(features)
    collated = {}
    
    # ============ スカラー値 ============
    # Ego の状態を batch_size 分スタック
    collated["current_state"] = torch.stack([
        f.data["current_state"] for f in features
    ])
    # Shape: (3, 10) ← (batch_size, state_dim)
    
    collated["angle"] = torch.stack([
        f.data["angle"] for f in features
    ])
    # Shape: (3,) ← (batch_size,)
    
    # ============ エージェント情報 ============
    collated["agent"] = {}
    
    for key in ["position", "heading", "velocity", "shape", "category", "valid_mask", "target"]:
        collated["agent"][key] = torch.stack([
            f.data["agent"][key] for f in features
        ])
    
    # 例: position
    # Shape: (3, 64, 101, 2) ← (batch_size, max_agents, time, 2)
    
    # ============ マップ情報 ============
    # マップはシナリオごとに異なるため、リストで保持
    collated["map"] = [
        f.data["map"] for f in features
    ]
    # Length: 3 (batch_size)
    
    # ============ メタ情報 ============
    collated["scenario_names"] = [
        f.data["scenario_name"] for f in features
    ]
    
    return collated

# 使用例
features = [
    builder(scenario_1, iteration=0),
    builder(scenario_2, iteration=0),
    builder(scenario_3, iteration=0),
]

batch = PlutoFeature.collate(features)

print(batch["current_state"].shape)  # (3, 10)
print(batch["agent"]["position"].shape)  # (3, 64, 101, 2)
print(len(batch["map"]))  # 3
```

---

## 🧮 データ型の詳細

### 推奨 Dtype

```python
# 位置・速度・加速度
torch.float32  # 精度とメモリのバランス

# カテゴリー・フラグ
torch.int64 / torch.bool  # 分類用

# マスク（有効性）
torch.bool  # True/False のみ

# 座標
torch.float32  # グローバル座標は大きい値
```

### メモリ消費量の計算

```python
# 単一特徴量
agent_position = 64 * 101 * 2 * 4 bytes = 51.6 KB
agent_heading = 64 * 101 * 1 * 4 bytes = 25.8 KB
map_polygons ≈ 100 KB

# 単一特徴量の合計 ≈ 200 KB

# バッチサイズ 32
batch_memory = 200 KB * 32 ≈ 6.4 MB
```

---

## 🚀 実装例

### 訓練ループでの使用

```python
from src.features.pluto_feature import PlutoFeature

# シナリオ → 特徴量への変換
scenarios = load_scenarios()
features = [builder(s, iteration=0) for s in scenarios[:32]]

# バッチ化
batch = PlutoFeature.collate(features)

# GPU に移動
batch = PlutoFeature(data=batch).to_device("cuda:0")

# モデル入力
outputs = model(batch)

# 後処理
predictions = outputs["prediction"]  # (batch_size, max_agents, future_steps, 2)
confidence = outputs["confidence"]    # (batch_size, max_agents, num_modes)
```

### 評価での使用

```python
from src.metrics import MinADE, MinFDE

# テストバッチ
test_feature = builder(test_scenario, iteration=0)
test_feature = test_feature.to_tensor(device="cuda:0")

# 推論
with torch.no_grad():
    output = model(test_feature)

# メトリック計算
target = test_feature.data["agent"]["target"]  # (1, 64, 80, 3)
prediction = output["prediction"]              # (1, 64, 80, 3)

metric = MinADE()
ade = metric(prediction, target)
print(f"minADE: {ade.item():.2f}")
```

---

## 🔍 デバッグ・検証

### 特徴量の整合性チェック

```python
def validate_pluto_feature(feature):
    """特徴量の妥当性検証"""
    
    # 形状チェック
    assert feature.data["current_state"].shape == (10,), "current_state の形状が不正"
    assert feature.data["agent"]["position"].shape == (64, 101, 2), "position の形状が不正"
    assert feature.data["agent"]["target"].shape == (64, 80, 3), "target の形状が不正"
    
    # 値の範囲チェック
    positions = feature.data["agent"]["position"]
    assert positions[positions[:,:,0].isfinite()].min() >= -10000, "位置が異常に小さい"
    assert positions[positions[:,:,0].isfinite()].max() <= 10000, "位置が異常に大きい"
    
    # 有効性マスク チェック
    valid_mask = feature.data["agent"]["valid_mask"]
    assert valid_mask[0] == True, "Ego が有効でない"
    
    print("✓ 特徴量が妥当")

validate_pluto_feature(feature)
```

### バッチ化の確認

```python
def validate_batch(batch):
    """バッチ化データの検証"""
    
    batch_size = batch["current_state"].shape[0]
    
    # すべてのテンソルが同じ batch_size を持つ
    assert batch["agent"]["position"].shape[0] == batch_size
    assert len(batch["map"]) == batch_size
    assert len(batch["scenario_names"]) == batch_size
    
    print(f"✓ バッチサイズ {batch_size} が統一されている")

validate_batch(batch)
```

---

## 📚 関連ファイル

- [../feature_builders/pluto_feature_builder.md](../feature_builders/pluto_feature_builder.md) - 特徴量の生成
- [../custom_training/custom_datamodule.md](../custom_training/custom_datamodule.md) - DataModule での使用
