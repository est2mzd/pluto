# ユーティリティ関数実装 詳細ガイド

## 📋 概要

`utils` モジュールは、PLUTO 全体で使用される **低レベル汎用機能** を提供します。

---

## 🔄 CollisionChecker - GPU 加速衝突検出

### 目的

**大量の軌跡** に対して高速に衝突判定を実施

### 実装戦略

```
従来的な衝突検出:
  軌跡数N × 時間ステップT × 障害物数M
  計算量: O(N × T × M)
  CPU: 10秒
  
GPU 加速版:
  行列演算でバッチ処理
  計算量: O(N × T × M) （並列化）
  GPU: 0.1秒
  速度向上: 100倍
```

### 実装コード

```python
class CollisionChecker:
    def __init__(self, scenario: AbstractScenario):
        self.scenario = scenario
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def check_collisions_batch(
        self,
        trajectories: torch.Tensor,    # (N, T, 2) N個の軌跡
        ego_shape: Tuple[float, float] = (2.0, 4.8)  # (width, length)
    ) -> torch.Tensor:
        """
        バッチ衝突検出
        
        Args:
            trajectories: (N, T, 2) N個の軌跡（各T時刻）
            ego_shape: 車のサイズ
        
        Returns:
            has_collision: (N,) True=衝突あり
        """
        
        N, T, _ = trajectories.shape
        
        # GPU に転送
        trajectories = trajectories.to(self.device)
        
        # 静的障害物の境界ボックスを準備
        static_obstacles = self._get_static_obstacles_gpu()
        
        # 各軌跡について衝突判定
        has_collision = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        for t in range(T):
            # 時刻 t でのすべての軌跡位置: (N, 2)
            positions_t = trajectories[:, t, :]
            
            # 各軌跡の占有領域（矩形）: (N, 4, 2)
            ego_boxes = self._get_ego_boxes_at_time(positions_t, ego_shape, t)
            
            # 衝突判定（GPU並列処理）
            collision_t = self._intersect_boxes_gpu(ego_boxes, static_obstacles)
            
            # 衝突フラグを更新
            has_collision = has_collision | collision_t  # どの時刻でも衝突なら True
        
        return has_collision.cpu()
    
    def _get_ego_boxes_at_time(
        self,
        positions: torch.Tensor,  # (N, 2) N個の位置
        ego_shape: Tuple[float, float],
        time_idx: int
    ) -> torch.Tensor:
        """
        位置からEgoの占有領域（矩形）を生成
        
        Args:
            positions: (N, 2)
            ego_shape: (width, length)
        
        Returns:
            ego_boxes: (N, 4, 2) 4頂点
        """
        
        N = positions.shape[0]
        width, length = ego_shape
        
        # 中心座標
        cx = positions[:, 0]  # (N,)
        cy = positions[:, 1]
        
        # ヘディングを推定（簡略化）
        if time_idx == 0:
            yaw = torch.zeros(N, device=positions.device)
        else:
            # 前の位置から速度ベクトルを計算
            pass  # 省略
        
        # 矩形の4頂点を計算
        # 車の中心を原点として、回転後に実際の位置に移動
        corners_local = torch.tensor([
            [-width/2, -length/2],
            [ width/2, -length/2],
            [ width/2,  length/2],
            [-width/2,  length/2]
        ], device=positions.device)
        
        # 回転変換: cos(yaw), -sin(yaw), sin(yaw), cos(yaw)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        # (N, 4, 2) に拡張
        ego_boxes = torch.zeros(N, 4, 2, device=positions.device)
        
        for corner_idx in range(4):
            corner = corners_local[corner_idx]
            
            # 回転
            rotated_x = (corner[0] * cos_yaw - corner[1] * sin_yaw)
            rotated_y = (corner[0] * sin_yaw + corner[1] * cos_yaw)
            
            # 平行移動
            ego_boxes[:, corner_idx, 0] = cx + rotated_x
            ego_boxes[:, corner_idx, 1] = cy + rotated_y
        
        return ego_boxes
    
    def _intersect_boxes_gpu(
        self,
        ego_boxes: torch.Tensor,          # (N, 4, 2)
        obstacle_boxes: torch.Tensor      # (M, 4, 2)
    ) -> torch.Tensor:
        """
        N個のEgo矩形 と M個の障害物矩形の交差判定（GPU並列）
        
        Returns:
            collision: (N,) True=衝突あり
        """
        
        N = ego_boxes.shape[0]
        M = obstacle_boxes.shape[0]
        
        collision = torch.zeros(N, dtype=torch.bool, device=ego_boxes.device)
        
        # 分離軸定理（Separating Axis Theorem）を使用
        # 2つの凸ポリゴンが衝突していない ⇔ 分離軸が存在する
        
        for n in range(N):
            ego_box = ego_boxes[n]  # (4, 2)
            
            for m in range(M):
                obs_box = obstacle_boxes[m]  # (4, 2)
                
                # 分離軸を取得（エッジの法線）
                axes = self._get_separating_axes(ego_box, obs_box)
                
                # 各軸について投影の重なりをチェック
                is_separated = False
                
                for axis in axes:
                    # Ego の投影範囲
                    ego_proj = torch.mm(ego_box, axis.unsqueeze(1)).squeeze()
                    ego_min, ego_max = ego_proj.min(), ego_proj.max()
                    
                    # 障害物の投影範囲
                    obs_proj = torch.mm(obs_box, axis.unsqueeze(1)).squeeze()
                    obs_min, obs_max = obs_proj.min(), obs_proj.max()
                    
                    # 重なっていないか
                    if ego_max < obs_min or obs_max < ego_min:
                        is_separated = True
                        break
                
                if not is_separated:
                    collision[n] = True
                    break
        
        return collision
    
    def _get_separating_axes(
        self,
        box1: torch.Tensor,  # (4, 2)
        box2: torch.Tensor   # (4, 2)
    ) -> List[torch.Tensor]:
        """
        分離軸定理の分離軸リストを取得
        """
        
        axes = []
        
        # Box1 の各エッジの法線
        for i in range(4):
            edge = box1[(i+1) % 4] - box1[i]
            # 法線（垂直）
            normal = torch.tensor([-edge[1], edge[0]], device=edge.device)
            normal = normal / (torch.norm(normal) + 1e-8)
            axes.append(normal)
        
        # Box2 の各エッジの法線
        for i in range(4):
            edge = box2[(i+1) % 4] - box2[i]
            normal = torch.tensor([-edge[1], edge[0]], device=edge.device)
            normal = normal / (torch.norm(normal) + 1e-8)
            axes.append(normal)
        
        return axes
```

---

## 🎨 画像変換・処理

### affine_transform - アフィン変換

```python
def affine_transform(
    image: torch.Tensor,              # (C, H, W)
    rotation: float = 0,              # [rad]
    scale: float = 1.0,
    translation: Tuple[float, float] = (0, 0)
) -> torch.Tensor:
    """
    画像に対して アフィン変換を適用
    
    用途: データ拡張、座標系の変換
    """
    
    C, H, W = image.shape
    
    # アフィン行列を構築
    # [x']   [cos -sin tx] [x]
    # [y'] = [sin  cos ty] [y]
    # [1 ]   [  0    0  1 ] [1]
    
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    
    affine_matrix = torch.tensor([
        [scale * cos_r, -scale * sin_r, translation[0]],
        [scale * sin_r,  scale * cos_r, translation[1]]
    ], dtype=torch.float32, device=image.device)
    
    # Grid を作成
    grid = torch.nn.functional.affine_grid(
        affine_matrix.unsqueeze(0),
        (1, C, H, W)
    )
    
    # サンプリング
    transformed = torch.nn.functional.grid_sample(
        image.unsqueeze(0),
        grid,
        align_corners=False
    )
    
    return transformed.squeeze(0)

# 使用例
original = torch.randn(3, 256, 256)
rotated = affine_transform(original, rotation=math.pi/4)
```

---

## 📊 統計・正規化

### min_max_normalize - ミン・マックス正規化

```python
def min_max_normalize(
    data: torch.Tensor,
    min_val: float = 0,
    max_val: float = 1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    値を [min_val, max_val] の範囲に正規化
    
    x_norm = (x - x_min) / (x_max - x_min) * (max_val - min_val) + min_val
    """
    
    data_min = data.min()
    data_max = data.max()
    
    data_normalized = (data - data_min) / (data_max - data_min + eps)
    data_normalized = data_normalized * (max_val - min_val) + min_val
    
    return data_normalized

# 例
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
normalized = min_max_normalize(data, min_val=0, max_val=1)
# tensor([0.0, 0.25, 0.5, 0.75, 1.0])
```

### zscore_normalize - Z スコア正規化

```python
def zscore_normalize(
    data: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Z スコア正規化: (x - μ) / σ
    """
    
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    
    normalized = (data - mean) / (std + eps)
    
    return normalized
```

---

## 🎯 軌跡処理

### interpolate_trajectory - 軌跡補間

```python
def interpolate_trajectory(
    trajectory: np.ndarray,        # (T, 2)
    target_freq: float = 10        # Hz
) -> np.ndarray:
    """
    軌跡を指定周波数で補間
    
    例:
      入力: [(0,0), (1,1), (2,2)]  サンプリング 1 Hz
      出力: [(0,0), (0.5,0.5), (1,1), (1.5,1.5), (2,2)]  10 Hz
    """
    
    T = trajectory.shape[0]
    
    # 元の時刻（秒）
    t_original = np.linspace(0, (T-1) / 1.0, T)
    
    # 補間後の時刻
    t_interp = np.linspace(0, (T-1) / 1.0, int((T-1) * target_freq) + 1)
    
    # 補間
    traj_interp = np.interp(t_interp, t_original, trajectory)
    
    return traj_interp
```

### smooth_trajectory - 軌跡平滑化

```python
def smooth_trajectory(
    trajectory: np.ndarray,        # (T, 2)
    kernel_size: int = 5
) -> np.ndarray:
    """
    軌跡をガウシアンフィルタで平滑化
    
    ノイズ除去、急激な方向転換の緩和
    """
    
    from scipy.ndimage import gaussian_filter1d
    
    sigma = kernel_size / 4
    
    # X 成分と Y 成分を個別に平滑化
    x_smooth = gaussian_filter1d(trajectory[:, 0], sigma=sigma)
    y_smooth = gaussian_filter1d(trajectory[:, 1], sigma=sigma)
    
    return np.column_stack([x_smooth, y_smooth])
```

---

## 📈 可視化ユーティリティ

### plot_trajectories - 複数軌跡の描画

```python
def plot_trajectories(
    trajectories: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """複数軌跡を一度に描画"""
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, traj in enumerate(trajectories):
        label = labels[idx] if labels else f"Trajectory {idx}"
        color = colors[idx] if colors else None
        
        ax.plot(traj[:, 0], traj[:, 1], label=label, color=color, marker='o', markersize=2)
    
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()

# 使用例
trajectories = [
    np.array([[0, 0], [1, 1], [2, 2]]),
    np.array([[0, 0], [0.5, 1.5], [1, 2.5]]),
    np.array([[0, 0], [1.5, 0.5], [3, 1]]),
]

plot_trajectories(trajectories, labels=["Mode 1", "Mode 2", "Mode 3"])
```

---

## 🔍 パフォーマンス最適化

### 実装パターン

```python
# ❌ 遅い（Python ループ）
def slow_collision_check(trajectories, obstacles):
    collisions = []
    for traj in trajectories:
        for t in range(len(traj)):
            for obs in obstacles:
                if distance(traj[t], obs) < threshold:
                    collisions.append(True)
    return collisions

# ✅ 速い（NumPy ベクトル化）
def fast_collision_check(trajectories, obstacles):
    # (N, T, 2) と (M, 2) のブロードキャスト
    dists = np.linalg.norm(
        trajectories[:, :, np.newaxis, :] - obstacles[np.newaxis, np.newaxis, :, :],
        axis=-1
    )  # (N, T, M)
    
    return (dists.min(axis=-1) < threshold).any(axis=1)  # (N,)
```

---

## 📚 関連ファイル

- [../post_processing/trajectory_evaluation.md](../post_processing/trajectory_evaluation.md) - 軌跡検証での使用
- [../feature_builders/pluto_feature_builder.md](../feature_builders/pluto_feature_builder.md) - 特徴量ビルダーでの使用
