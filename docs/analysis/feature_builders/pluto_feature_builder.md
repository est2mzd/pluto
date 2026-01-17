# PlutoFeatureBuilder 詳細実装ガイド

## 📋 概要

`PlutoFeatureBuilder` は、nuPlan シナリオから PLUTO モデルが必要とする **構造化された特徴量** を計算する中核エンジンです。

---

## 🔧 クラス構造

### 初期化パラメータの詳細

```python
class PlutoFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        radius: float = 100,                    # Ego 周辺 [m]
        history_horizon: float = 2,             # 過去 [秒]
        future_horizon: float = 8,              # 将来 [秒]
        sample_interval: float = 0.1,           # サンプリング間隔 [秒]
        max_agents: int = 64,                   # 最大エージェント数
        max_static_obstacles: int = 10,         # 最大障害物数
        build_reference_line: bool = False,     # リファレンスライン計算
        disable_agent: bool = False             # エージェント除外
    ):
```

### パラメータ計算

```python
# 内部で自動計算されるパラメータ
self.history_samples = int(history_horizon / sample_interval)
# 例: int(2.0 / 0.1) = 20 ステップ

self.future_samples = int(future_horizon / sample_interval)
# 例: int(8.0 / 0.1) = 80 ステップ
```

---

## 📊 `__call__()` メソッド - 特徴量計算フロー

### シグネチャ

```python
def __call__(
    self,
    scenario: AbstractScenario,
    iteration: int = 0
) -> AbstractModelFeature:
```

### 詳細処理フロー

```
【ステップ1】Ego 軌跡の抽出
    ├─ 過去軌跡: scenario.get_ego_past_trajectory()
    │   → history_samples = 20 ステップ
    ├─ 現在状態: scenario.initial_ego_state
    └─ 将来軌跡: scenario.get_ego_future_trajectory()
        → future_samples = 80 ステップ
    
    結果: ego_state_list (101要素)
    [過去20, 現在1, 将来80]

【ステップ2】周辺エージェント軌跡の抽出
    ├─ 過去: scenario.get_past_tracked_objects()
    ├─ 現在: scenario.initial_tracked_objects
    └─ 将来: scenario.get_future_tracked_objects()
    
    結果: tracked_objects_list (101要素)
    各要素に複数のエージェント情報

【ステップ3】ルート・地図情報の抽出
    ├─ ルート: scenario.get_route_roadblock_ids()
    ├─ 地図: scenario.map_api
    ├─ ミッション: scenario.get_mission_goal()
    └─ 交通信号: scenario.get_traffic_light_status_at_iteration()

【ステップ4】特徴量ビルド
    └─ self._build_feature() で統合
        → PlutoFeature オブジェクト生成
```

---

## 🔄 `_build_feature()` メソッド - 特徴量構築

### 目的
```
生のシナリオデータを
PlutoFeature の標準形式に変換
```

### 処理内容

```python
def _build_feature(
    self,
    present_idx: int,              # 現在時刻のインデックス (20)
    ego_state_list: List,          # Ego 軌跡 (101)
    tracked_objects_list: List,    # エージェント軌跡 (101)
    route_roadblocks_ids: List,    # ルート
    map_api: AbstractMap,          # 地図 API
    mission_goal: Point2D,         # 目的地
    traffic_light_status: Dict     # 交通信号
) -> Dict[str, Any]:
```

### 出力形式の詳細

```python
return {
    # Ego の現在状態
    "current_state": torch.tensor([
        ego_x, ego_y, ego_yaw,      # 位置・向き
        ego_vel, ego_acc,           # 速度・加速度
        ego_steer, ego_steer_rate   # ステアリング
    ]),
    
    # 原点（座標系の基準）
    "origin": torch.tensor([
        ego_x, ego_y
    ]),
    
    # Ego の向き
    "angle": torch.tensor(ego_yaw),
    
    # エージェント情報（すべて正規化）
    "agent": {
        "position": (max_agents, time_steps=101, 2),
        "heading": (max_agents, 101),
        "velocity": (max_agents, 101, 2),
        "shape": (max_agents, 2),  # [width, length]
        "category": (max_agents,),
        "valid_mask": (max_agents,),
        "target": (max_agents, future_steps=80, 3)  # 将来位置・yaw
    },
    
    # マップ情報
    "map": {
        "polygon_tl_status": (num_tl,),
        "polygon_tl_id": (num_tl,),
        "polygon_road_edge": (num_edges,),
        ...他のマップレイヤー
    },
    
    # オプション: コスト地図
    "cost_maps": (H=500, W=500),  # occupancy grid
    
    # 因果関係情報
    "causal": {
        "interaction_label": (max_agents,),
        "leading_agent_mask": (max_agents,),
        ...
    }
}
```

---

## 🎯 エージェント処理の詳細

### ステップ1: エージェント抽出

```python
# radius 内のエージェントのみを保持
for agent in all_agents:
    distance_to_ego = euclidean_distance(agent.position, ego.position)
    if distance_to_ego <= radius:
        candidates.append(agent)

# 結果: candidates (数十個)
```

### ステップ2: ソート & パディング

```python
# Ego を第0要素に
agents = [ego] + candidates

# 最大エージェント数にパディング
while len(agents) < max_agents:
    agents.append(EmptyAgent)  # 有効フラグ = False

# 結果: agents (max_agents=64)
```

### ステップ3: 軌跡の統合

```python
# 各エージェントの時系列データを統合
for agent_idx, agent in enumerate(agents):
    for time_idx in range(101):
        agent_position[agent_idx, time_idx] = agent.position_at_time[time_idx]
        agent_heading[agent_idx, time_idx] = agent.heading_at_time[time_idx]
        agent_velocity[agent_idx, time_idx] = agent.velocity_at_time[time_idx]

# 結果: (max_agents=64, time=101, 2/1/2)
```

---

## 📍 座標系の正規化

### 変換前（グローバル座標）

```
地図座標系:
  (0, 0) ─────→ x (東)
   │
   │
   ↓ y (北)
   
エージェント: (100, 200)
```

### 変換後（Ego 中心座標）

```
Ego 座標系:
  (0, 0) は Ego の前方
  x軸: Ego の前方方向
  y軸: Ego の左方向
  
変換: 回転 + 平行移動
  x_ego = (x - x_ego) * cos(yaw) + (y - y_ego) * sin(yaw)
  y_ego = -(x - x_ego) * sin(yaw) + (y - y_ego) * cos(yaw)
```

### メリット

```
✓ モデルが Ego 中心で学習
✓ 異なる場所での汎化性能向上
✓ 回転不変性の改善
```

---

## 🗺️ マップ情報の処理

### 抽出されるマップレイヤー

```python
semantic_layers = [
    SemanticMapLayer.LANE,
    SemanticMapLayer.ROAD_EDGE,
    SemanticMapLayer.TRAFFIC_LIGHT,
    SemanticMapLayer.CROSSWALK,
    SemanticMapLayer.STOP_LINE,
    ...
]

for layer in semantic_layers:
    polygons = map_api.get_proximal_map_objects(
        ego_position,
        radius,
        [layer]
    )
    # 各ポリゴンをテンソル化
```

### 交通信号処理

```python
# 交通信号の位置と状態を統合
for tl_id, tl_status in traffic_light_status.items():
    tl_position = map_api.get_traffic_light_position(tl_id)
    tl_state = convert_to_enum(tl_status)  # RED/GREEN/YELLOW/...
    
    # 保存
    tl_positions.append(tl_position)
    tl_statuses.append(tl_state)
```

---

## 💡 正規化テクニック

### ミン・マックス正規化

```python
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# 例: 速度を 0～10 m/s から 0～1 に正規化
normalized_vel = (velocity - 0) / (10 - 0)
```

### Z スコア正規化

```python
def normalize_zscore(values, mean, std):
    return (values - mean) / std

# 統計的に正規化
```

---

## 🚀 実装例

### シナリオから特徴量を計算

```python
from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder
from nuplan.planning.scenario_builder.scenario_builder import ScenarioBuilder

# シナリオロード
scenario_builder = ScenarioBuilder(...)
scenario = scenario_builder.build_scenario("mini_demo_scenario")

# ビルダー作成
builder = PlutoFeatureBuilder(
    radius=100,
    history_horizon=2.0,
    future_horizon=8.0,
    max_agents=64
)

# 特徴量計算
feature = builder(scenario, iteration=0)

# 結果確認
print(feature.data.keys())
# dict_keys(['agent', 'map', 'current_state', 'origin', 'angle', 'cost_maps', 'causal'])

print(feature.data["agent"]["position"].shape)
# torch.Size([64, 101, 2])
```

---

## 🔍 デバッグ・検証

### 特徴量のサイズチェック

```python
def validate_feature(feature):
    assert feature.data["agent"]["position"].shape[0] == 64  # max_agents
    assert feature.data["agent"]["position"].shape[1] == 101  # 過去20 + 現在1 + 将来80
    assert feature.data["agent"]["position"].shape[2] == 2   # (x, y)
    
    print("✓ 特徴量形状が正常")
```

### 値の範囲チェック

```python
def check_value_ranges(feature):
    pos = feature.data["agent"]["position"]
    assert pos.min() >= -1000  # グローバル座標の妥当性
    assert pos.max() <= 1000
    
    vel = feature.data["agent"]["velocity"]
    assert vel.min() >= -50  # 速度の妥当性
    assert vel.max() <= 50
    
    print("✓ 値の範囲が正常")
```

### 有効性マスク確認

```python
def check_valid_mask(feature):
    valid_agents = feature.data["agent"]["valid_mask"].sum()
    print(f"有効なエージェント数: {valid_agents}")
    
    # 第0要素は必ず Ego（有効）
    assert feature.data["agent"]["valid_mask"][0] == True
    
    print("✓ 有効性マスクが正常")
```

---

## 📚 関連ファイル

- [../features/README.md](../features/README.md) - 出力形式
- [../custom_training/README.md](../custom_training/README.md) - 訓練での使用
