# 軌跡評価・安全性検証 詳細ガイド

## 📋 概要

`TrajectoryEvaluator` と `EmergencyBrake` は、**生成された軌跡の妥当性・安全性を検証・改善** するコンポーネントです。

---

## 🔍 TrajectoryEvaluator - 軌跡検証

### 目的

```
入力軌跡 → 妥当性検査 → 安全性評価 → 改善提案
```

### 実装構造

```python
class TrajectoryEvaluator:
    def __init__(self, scenario: AbstractScenario):
        self.scenario = scenario
        self.collision_checker = CollisionChecker(scenario)
        self.comfort_checker = ComfortMetricsCalculator()
    
    def evaluate(
        self,
        trajectory: Trajectory
    ) -> TrajectoryEvaluationResult:
        """軌跡を多角的に評価"""
        
        result = TrajectoryEvaluationResult()
        
        # 評価1: 衝突検査
        result.has_collision = self._check_collision(trajectory)
        
        # 評価2: 快適性
        result.comfort_score = self._evaluate_comfort(trajectory)
        
        # 評価3: 妥当性
        result.feasibility_score = self._evaluate_feasibility(trajectory)
        
        # 評価4: ルート準拠
        result.off_route = self._check_off_route(trajectory)
        
        return result
```

---

## ⚠️ 衝突検査の詳細

### アルゴリズム

```
【ステップ1】Ego 軌跡の包囲枠生成
    各時刻 t で、Ego の占有領域を矩形で表現
    位置: (x_t, y_t)
    サイズ: 車長 × 車幅 (4.8 m × 2.0 m)
    向き: yaw_t
    
    ↓ 時系列で矩形リストを生成

【ステップ2】障害物検出
    static_obstacles (建物、街灯など) の位置取得
    tracked_objects (他の車など) の時系列位置
    
    ↓ 時刻同期

【ステップ3】交差判定
    各時刻で、Ego 矩形 ∩ 障害物 ≠ ∅ ?
    
    if 交差:
        collision = True
        collision_time = t
        return False
    
    ↓ すべての時刻をチェック

【ステップ4】結果返却
    collision = False ⇒ 軌跡は安全
    collision = True ⇒ 軌跡は危険
```

### 実装例

```python
def _check_collision(self, trajectory: Trajectory) -> bool:
    """軌跡が衝突しているか判定"""
    
    states = trajectory.states
    T = len(states)
    
    # 車の大きさ [m]
    EGO_WIDTH = 2.0
    EGO_LENGTH = 4.8
    
    for t, state in enumerate(states):
        # Ego の位置と向き
        x, y = state.position
        yaw = state.heading
        
        # Ego の占有領域（矩形）
        ego_box = get_oriented_bounding_box(
            center=(x, y),
            width=EGO_WIDTH,
            length=EGO_LENGTH,
            angle=yaw
        )
        
        # 静的障害物との衝突確認
        for obstacle in self.scenario.get_all_static_obstacles():
            obstacle_box = obstacle.get_oriented_bounding_box()
            
            if ego_box.intersects(obstacle_box):
                print(f"Collision with static obstacle at t={t}")
                return True
        
        # 動的障害物（他の車）との衝突確認
        tracked_objects = self.scenario.get_tracked_objects_at_time(t)
        
        for agent in tracked_objects.values():
            agent_box = get_oriented_bounding_box(
                center=agent.position,
                width=agent.width,
                length=agent.length,
                angle=agent.heading
            )
            
            if ego_box.intersects(agent_box):
                print(f"Collision with agent {agent.id} at t={t}")
                return True
    
    print("No collision detected")
    return False
```

---

## 🛑 EmergencyBrake - 安全性改善

### 目的

衝突予測時に、**フルブレーキで衝突を回避**

```
衝突軌跡 → 衝突時刻検出 → フルブレーキ計算 → 修正軌跡
```

### アルゴリズム

```python
class EmergencyBrake:
    def __init__(self, deceleration: float = -5.0):  # m/s²
        self.deceleration = deceleration  # 最大減速度
    
    def apply_emergency_brake(
        self,
        original_trajectory: Trajectory,
        collision_time: int
    ) -> Trajectory:
        """
        衝突時刻以降の軌跡をブレーキに置き換え
        """
        
        states = list(original_trajectory.states)
        T = len(states)
        dt = 0.1  # サンプリング間隔
        
        # 衝突時刻直前の状態から開始
        collision_state = states[collision_time - 1]
        x = collision_state.position[0]
        y = collision_state.position[1]
        vx = collision_state.velocity[0]
        vy = collision_state.velocity[1]
        yaw = collision_state.heading
        
        # 衝突時刻以降を修正
        for t in range(collision_time, T):
            # 速度の更新（フルブレーキ）
            v_mag = math.sqrt(vx**2 + vy**2)
            
            if v_mag > 0.1:  # 0に近い場合は停止
                # 速度方向を保持しながら減速
                decel_mag = self.deceleration * dt
                vx = vx * (1 + decel_mag / v_mag)
                vy = vy * (1 + decel_mag / v_mag)
            else:
                vx, vy = 0, 0
            
            # 位置更新
            x += vx * dt
            y += vy * dt
            
            # 状態を更新
            new_state = State(
                position=(x, y),
                heading=yaw,
                velocity=(vx, vy),
                acceleration=(self.deceleration, 0)
            )
            
            states[t] = new_state
        
        return Trajectory(states)
```

---

## 😊 快適性メトリクス

### 加速度の評価

```python
def evaluate_comfort(trajectory: Trajectory) -> float:
    """
    乗車快適性を評価
    
    基準:
      加速度 < 3 m/s²: 快適
      加速度 3-5 m/s²: 許容範囲
      加速度 > 5 m/s²: 不快
    """
    
    states = trajectory.states
    T = len(states)
    dt = 0.1
    
    max_accel = 0
    max_jerk = 0
    
    for t in range(1, T):
        prev_state = states[t - 1]
        curr_state = states[t]
        
        # 加速度計算
        ax = (curr_state.velocity[0] - prev_state.velocity[0]) / dt
        ay = (curr_state.velocity[1] - prev_state.velocity[1]) / dt
        accel = math.sqrt(ax**2 + ay**2)
        
        max_accel = max(max_accel, accel)
    
    # スコア: 加速度が小さいほど良い (0-1)
    comfort_score = max(0, 1 - max_accel / 5.0)
    
    return comfort_score
```

---

## 🗺️ ForwardSimulator - 運動学シミュレーション

### 目的

軌跡が **運動学的に実現可能か** を検証

```
軌跡 → 運動学モデル → シミュレーション → 実現可能性判定
```

### 2輪モデル（自転車モデル）

```python
class BicycleModel:
    """車両の2輪モデル"""
    
    def __init__(
        self,
        wheelbase: float = 2.7  # 前後軸間距離
    ):
        self.wheelbase = wheelbase
    
    def step(
        self,
        x: float,           # 位置 X
        y: float,           # 位置 Y
        yaw: float,         # 向き
        v: float,           # 速度 [m/s]
        steer: float,       # ステアリング角 [rad]
        dt: float = 0.1     # 時間ステップ
    ) -> Tuple[float, float, float]:
        """
        一ステップシミュレーション
        
        キネマティック方程式:
          dx = v * cos(yaw)
          dy = v * sin(yaw)
          dyaw = (v / wheelbase) * tan(steer)
        """
        
        dx = v * math.cos(yaw)
        dy = v * math.sin(yaw)
        dyaw = (v / self.wheelbase) * math.tan(steer)
        
        x_new = x + dx * dt
        y_new = y + dy * dt
        yaw_new = yaw + dyaw * dt
        
        return x_new, y_new, yaw_new

def simulate_trajectory(
    trajectory: Trajectory,
    model: BicycleModel,
    dt: float = 0.1
) -> List[Tuple[float, float]]:
    """軌跡をシミュレーションして実現可能性を検証"""
    
    simulated_positions = [(0, 0)]
    x, y, yaw = 0, 0, 0
    
    for t in range(1, len(trajectory.states)):
        state = trajectory.states[t]
        v = math.sqrt(state.velocity[0]**2 + state.velocity[1]**2)
        
        # ステアリング角を推定（ヘディングの変化から）
        prev_yaw = trajectory.states[t-1].heading
        yaw_diff = state.heading - prev_yaw
        steer = math.atan(self.wheelbase * yaw_diff / (v * dt))
        
        # シミュレーション
        x, y, yaw = model.step(x, y, yaw, v, steer, dt)
        simulated_positions.append((x, y))
    
    return simulated_positions
```

---

## 🔗 評価パイプライン全体

```python
class TrajectoryPostProcessor:
    def __init__(self, scenario: AbstractScenario):
        self.evaluator = TrajectoryEvaluator(scenario)
        self.emergency_brake = EmergencyBrake()
        self.simulator = ForwardSimulator()
    
    def process_trajectory(
        self,
        trajectory: Trajectory
    ) -> Tuple[Trajectory, Dict]:
        """軌跡の評価と改善"""
        
        # Step 1: 評価
        eval_result = self.evaluator.evaluate(trajectory)
        
        # Step 2: 衝突回避
        if eval_result.has_collision:
            trajectory = self.emergency_brake.apply_emergency_brake(
                trajectory,
                eval_result.collision_time
            )
        
        # Step 3: 実現可能性検証
        feasible = self.simulator.check_feasibility(trajectory)
        
        # Step 4: 結果リポート
        report = {
            "original_collision": eval_result.has_collision,
            "after_brake_collision": self.evaluator.evaluate(trajectory).has_collision,
            "comfort_score": eval_result.comfort_score,
            "feasible": feasible
        }
        
        return trajectory, report

# 使用例
post_processor = TrajectoryPostProcessor(scenario)
safe_trajectory, report = post_processor.process_trajectory(raw_trajectory)

if report["after_brake_collision"]:
    print("警告: ブレーキ後も衝突予測")
else:
    print("✓ 軌跡は安全")

print(f"快適性スコア: {report['comfort_score']:.2f}")
print(f"実現可能: {report['feasible']}")
```

---

## 📊 評価結果の可視化

```python
import matplotlib.pyplot as plt

def visualize_trajectory_evaluation(scenario, trajectory, eval_result):
    """評価結果を可視化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # プロット1: 軌跡と衝突判定
    states = trajectory.states
    xs = [s.position[0] for s in states]
    ys = [s.position[1] for s in states]
    
    ax1.plot(xs, ys, 'b-', linewidth=2, label='Trajectory')
    
    if eval_result.has_collision:
        collision_idx = eval_result.collision_time
        ax1.plot(xs[collision_idx], ys[collision_idx], 'rx', markersize=15, label='Collision')
    
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title("Trajectory and Collision Detection")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # プロット2: 速度プロファイル
    velocities = [math.sqrt(s.velocity[0]**2 + s.velocity[1]**2) for s in states]
    ax2.plot(velocities)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Velocity [m/s]")
    ax2.set_title("Velocity Profile")
    ax2.grid(True, alpha=0.3)
    
    # プロット3: 加速度プロファイル
    accelerations = [math.sqrt(s.acceleration[0]**2 + s.acceleration[1]**2) for s in states]
    ax3.plot(accelerations)
    ax3.axhline(y=3, color='g', linestyle='--', alpha=0.5, label='Comfortable (3 m/s²)')
    ax3.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Limit (5 m/s²)')
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Acceleration [m/s²]")
    ax3.set_title("Acceleration Profile")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # プロット4: スコア表示
    scores = {
        "Comfort": eval_result.comfort_score,
        "Feasibility": eval_result.feasibility_score,
        "On-Route": 1.0 if not eval_result.off_route else 0.0
    }
    
    ax4.bar(scores.keys(), scores.values())
    ax4.set_ylabel("Score")
    ax4.set_ylim([0, 1])
    ax4.set_title("Evaluation Scores")
    
    plt.tight_layout()
    plt.savefig("trajectory_evaluation.png")
    plt.show()
```

---

## 📚 関連ファイル

- [../planners/pluto_planner.md](../planners/pluto_planner.md) - 推論エンジン
- [../utils/utility_functions.md](../utils/utility_functions.md) - CollisionChecker
