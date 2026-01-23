# シナリオ管理・データ処理 詳細ガイド

## 📋 概要

`ScenarioManager` は、nuPlan シナリオの **ルート管理・占有グリッド処理・コスト地図生成** を担当します。

---

## 🗺️ OccupancyMap - 占有グリッド表現

### 目的

```
連続空間 → グリッド離散化 → バイナリグリッド表現
```

### グリッド表現

```
物理空間:               グリッド空間:

Y ↑                     [0,0]  →  [0, W-1]
  │  (x₁, y₁)            ↓
  │    ●                [H-1,0]  [H-1,W-1]
  │
  0 ────────→ X

解像度: 0.2 m/grid
グリッドサイズ: 500 × 500 = 100 m × 100 m
```

### 実装

```python
class OccupancyMap:
    def __init__(
        self,
        width: int = 500,           # グリッド幅 [pixel]
        height: int = 500,          # グリッド高さ
        resolution: float = 0.2     # 1グリッド当たりのメートル数
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # バイナリグリッド: 0=自由, 1=占有
        self.grid = np.zeros((height, width), dtype=np.uint8)
    
    def world_to_grid(
        self,
        world_x: float,
        world_y: float,
        origin_x: float = 0,
        origin_y: float = 0
    ) -> Tuple[int, int]:
        """
        物理座標 → グリッド座標への変換
        
        例:
          世界座標: (1.5, 2.0) m
          原点: (0, 0)
          解像度: 0.2 m/grid
          
          grid_x = (1.5 - 0) / 0.2 = 7
          grid_y = (2.0 - 0) / 0.2 = 10
        """
        
        grid_x = int((world_x - origin_x) / self.resolution)
        grid_y = int((world_y - origin_y) / self.resolution)
        
        # 境界チェック
        grid_x = max(0, min(grid_x, self.width - 1))
        grid_y = max(0, min(grid_y, self.height - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(
        self,
        grid_x: int,
        grid_y: int,
        origin_x: float = 0,
        origin_y: float = 0
    ) -> Tuple[float, float]:
        """グリッド座標 → 物理座標への変換"""
        
        world_x = grid_x * self.resolution + origin_x
        world_y = grid_y * self.resolution + origin_y
        
        return world_x, world_y
    
    def add_obstacle(
        self,
        world_x: float,
        world_y: float,
        radius: float,
        origin_x: float = 0,
        origin_y: float = 0
    ):
        """障害物を円形で追加"""
        
        grid_x, grid_y = self.world_to_grid(world_x, world_y, origin_x, origin_y)
        grid_radius = int(radius / self.resolution)
        
        # 円形領域を埋める
        y_min = max(0, grid_y - grid_radius)
        y_max = min(self.height, grid_y + grid_radius + 1)
        x_min = max(0, grid_x - grid_radius)
        x_max = min(self.width, grid_x + grid_radius + 1)
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dist = math.sqrt((x - grid_x)**2 + (y - grid_y)**2)
                if dist <= grid_radius:
                    self.grid[y, x] = 1
```

---

## 💰 CostMapManager - コスト地図生成

### 目的

各グリッドセルの **通行コスト** を計算

```
占有グリッド → コスト計算 → 最短経路探索
```

### コスト計算式

```python
cost[y, x] = (
    occupancy_cost[y, x]      # 占有コスト
    + collision_risk[y, x]    # 衝突リスク
    + off_road_cost[y, x]     # オフロードペナルティ
    + distance_to_boundary[y, x] * boundary_cost  # 境界コスト
)
```

### 実装

```python
class CostMapManager:
    def __init__(self, occupancy_map: OccupancyMap):
        self.occupancy_map = occupancy_map
        self.cost_map = np.zeros_like(occupancy_map.grid, dtype=np.float32)
    
    def compute_cost_map(
        self,
        occupied_cells: np.ndarray,
        road_mask: np.ndarray,
        boundary_distance: np.ndarray
    ) -> np.ndarray:
        """
        総合コスト地図を計算
        
        Args:
            occupied_cells: (H, W) 占有グリッド
            road_mask: (H, W) 走行可能エリア
            boundary_distance: (H, W) 道路端までの距離
        """
        
        cost_map = np.zeros_like(occupied_cells, dtype=np.float32)
        
        # コンポーネント1: 占有コスト
        cost_map[occupied_cells == 1] = 1000  # 占有セルは通路不可
        
        # コンポーネント2: オフロードペナルティ
        cost_map[road_mask == 0] = 100  # 道路外は高コスト
        
        # コンポーネント3: 境界コスト（道路端から遠いほど安全）
        boundary_cost = np.exp(-boundary_distance / 2)  # 指数減衰
        cost_map += boundary_cost * 10
        
        # コンポーネント4: 衝突リスク（周辺エージェント考慮）
        # ... エージェント位置から膨張処理で追加 ...
        
        return cost_map
```

---

## 🛣️ RouteManager - ルート管理

### 目的

シナリオの **目的地までの最適ルート** を計算・維持

### Dijkstra アルゴリズムによるルート計算

```python
class RouteManager:
    def __init__(self, map_api):
        self.map_api = map_api
        self.route_cache = {}
    
    def compute_route(
        self,
        start_position: Tuple[float, float],
        goal_position: Tuple[float, float],
        cost_map: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Dijkstra アルゴリズムで最短経路を計算
        
        例:
          スタート: (0, 0)
          ゴール: (50, 50)
          コスト地図: 障害物周辺のコストが高い
          
          結果: [(0,0), (5,2), (10,5), ..., (50,50)]
        """
        
        grid_start = self.occupancy_map.world_to_grid(*start_position)
        grid_goal = self.occupancy_map.world_to_grid(*goal_position)
        
        # Dijkstra実行
        distances, predecessors = self._dijkstra(
            grid_start,
            grid_goal,
            cost_map
        )
        
        # 経路を復元
        path_grid = self._reconstruct_path(grid_start, grid_goal, predecessors)
        
        # グリッド座標を世界座標に変換
        path_world = [
            self.occupancy_map.grid_to_world(gx, gy)
            for gx, gy in path_grid
        ]
        
        return path_world
    
    def _dijkstra(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dijkstra アルゴリズムの実装
        
        計算量: O(N log N) where N = grid cells
        """
        
        H, W = cost_map.shape
        distances = np.full((H, W), np.inf)
        predecessors = np.full((H, W), None, dtype=object)
        
        distances[start] = 0
        unvisited = {start}
        
        while unvisited:
            # 未訪問ノードで最小距離を選ぶ
            current = min(unvisited, key=lambda n: distances[n])
            
            if current == goal:
                break  # ゴール到達
            
            unvisited.remove(current)
            
            # 隣接セルの距離更新
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 境界チェック
                if not (0 <= neighbor[0] < H and 0 <= neighbor[1] < W):
                    continue
                
                # 占有セルスキップ
                if cost_map[neighbor] >= 1000:
                    continue
                
                # 距離計算（斜めは√2倍）
                dist_multiplier = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
                new_distance = distances[current] + cost_map[neighbor] * dist_multiplier
                
                # より短い経路が見つかったら更新
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current
                    unvisited.add(neighbor)
        
        return distances, predecessors
```

---

## 🔄 ScenarioManager - 統合管理

### 全体構造

```python
class ScenarioManager:
    def __init__(self, scenario: AbstractScenario):
        self.scenario = scenario
        self.occupancy_map = OccupancyMap()
        self.cost_map_manager = CostMapManager(self.occupancy_map)
        self.route_manager = RouteManager(scenario.map_api)
        
        # キャッシュ
        self.route_cache = {}
    
    def update(self, current_ego_state: EgoState):
        """シナリオ状態の更新"""
        
        # Step 1: 占有グリッドの更新
        self._update_occupancy_map(current_ego_state)
        
        # Step 2: コスト地図の再計算
        self._update_cost_map()
        
        # Step 3: ルートの再計算（必要時）
        self._update_route(current_ego_state)
    
    def _update_occupancy_map(self, ego_state: EgoState):
        """占有グリッドを更新"""
        
        self.occupancy_map.grid.fill(0)  # リセット
        
        # 静的障害物
        for obstacle in self.scenario.get_all_static_obstacles():
            self.occupancy_map.add_obstacle(
                obstacle.position[0],
                obstacle.position[1],
                obstacle.get_radius(),
                ego_state.position[0],
                ego_state.position[1]
            )
        
        # 動的障害物（他の車）
        for agent in self.scenario.get_tracked_objects().values():
            self.occupancy_map.add_obstacle(
                agent.position[0],
                agent.position[1],
                agent.get_radius(),
                ego_state.position[0],
                ego_state.position[1]
            )
    
    def _update_cost_map(self):
        """コスト地図を再計算"""
        
        road_mask = self._get_road_mask()
        boundary_distance = self._get_boundary_distance()
        
        self.cost_map = self.cost_map_manager.compute_cost_map(
            self.occupancy_map.grid,
            road_mask,
            boundary_distance
        )
    
    def _update_route(self, ego_state: EgoState):
        """ルートを更新"""
        
        goal = self.scenario.get_mission_goal()
        
        route = self.route_manager.compute_route(
            ego_state.position,
            goal.position,
            self.cost_map
        )
        
        self.current_route = route
    
    def get_cost_map(self) -> np.ndarray:
        """コスト地図を取得"""
        return self.cost_map
    
    def get_route(self) -> List[Tuple[float, float]]:
        """現在のルートを取得"""
        return self.current_route
```

---

## 🚀 使用例

### シナリオの管理と更新

```python
from src.scenario_manager.scenario_manager import ScenarioManager

# シナリオ読み込み
scenario = load_scenario("mini_demo_scenario_0")

# マネージャー作成
manager = ScenarioManager(scenario)

# シミュレーションループ
for iteration in range(1000):
    # Ego の現在状態取得
    ego_state = scenario.get_ego_state_at_iteration(iteration)
    
    # シナリオ状態更新
    manager.update(ego_state)
    
    # コスト地図を取得
    cost_map = manager.get_cost_map()
    
    # ルートを取得
    route = manager.get_route()
    
    # コスト地図とルートを計画に使用
    planned_trajectory = planner.plan(ego_state, cost_map, route)
```

---

## 📊 可視化

```python
import matplotlib.pyplot as plt

def visualize_scenario_manager(manager, ego_state):
    """シナリオ管理状態の可視化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # プロット1: 占有グリッド
    ax1.imshow(manager.occupancy_map.grid, cmap='gray')
    ax1.set_title("Occupancy Map")
    ax1.set_xlabel("Grid X")
    ax1.set_ylabel("Grid Y")
    
    # プロット2: コスト地図
    im2 = ax2.imshow(manager.cost_map, cmap='hot')
    plt.colorbar(im2, ax=ax2, label='Cost')
    ax2.set_title("Cost Map")
    
    # プロット3: ルート
    route = manager.get_route()
    route_xs, route_ys = zip(*route)
    ax3.plot(route_xs, route_ys, 'b-', linewidth=2, label='Route')
    ax3.plot(ego_state.position[0], ego_state.position[1], 'ro', markersize=10, label='Ego')
    ax3.set_title("Computed Route")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # プロット4: コスト分布
    ax4.hist(manager.cost_map.flatten(), bins=50)
    ax4.set_title("Cost Distribution")
    ax4.set_xlabel("Cost Value")
    ax4.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("scenario_manager_viz.png")
    plt.show()
```

---

## 📚 関連ファイル

- [../planners/pluto_planner.md](../planners/pluto_planner.md) - 推論エンジン
- [../post_processing/trajectory_evaluation.md](../post_processing/trajectory_evaluation.md) - 軌跡検証
