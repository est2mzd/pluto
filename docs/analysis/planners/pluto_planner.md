# PlutoPlanner 推論エンジン 詳細ガイド

## 📋 概要

`PlutoPlanner` は、**訓練済み PLUTO モデルを nuPlan シミュレーション環境に統合する推論エンジン** です。

---

## 🏗️ クラス構造

### 初期化パラメータ

```python
class PlutoPlanner(AbstractPlanner):
    def __init__(
        self,
        config: DictConfig,                    # Hydra 設定
        model_params: Dict[str, Any],         # モデル重み
        checkpoint_path: Optional[str] = None # 事前訓練重み
    ):
```

### 初期化処理の詳細

```
【ステップ1】設定の解析
    ├─ モデルアーキテクチャ設定
    ├─ 推論パラメータ
    └─ データ正規化パラメータ

【ステップ2】モデル構築
    ├─ PlutoModel インスタンス作成
    └─ チェックポイント読み込み（指定時）

【ステップ3】デバイス設定
    ├─ GPU/CPU 選択
    └─ モデルをデバイスに転送

【ステップ4】内部状態初期化
    ├─ 過去軌跡バッファ
    ├─ シナリオ管理オブジェクト
    └─ 軌跡評価機
```

---

## 🔄 `compute_planner_trajectory()` メソッド - 推論フロー

### メソッドシグネチャ

```python
def compute_planner_trajectory(
    self,
    current_input: PlannerInput
) -> List[TrajectoryWithProba]:
    """
    Args:
        current_input: PlannerInput
            ├─ iteration: シミュレーションステップ
            ├─ history: 過去軌跡
            ├─ traffic_light_status: 交通信号
            └─ scenario: nuPlan Scenario

    Returns:
        List[TrajectoryWithProba]
            各モードの軌跡と確率
    """
```

### 詳細処理フロー

```
【ステップ1】入力データの抽出
    ├─ current_input から Scenario を取得
    ├─ 現在のイテレーション番号を確認
    └─ 交通信号状態を読み込み

【ステップ2】特徴量ビルド
    ├─ PlutoFeatureBuilder を実行
    │   ├─ 過去軌跡の抽出 (2秒)
    │   ├─ 周辺エージェント検出
    │   ├─ マップ情報抽出
    │   └─ 座標正規化
    └─ PlutoFeature オブジェクト生成

【ステップ3】推論
    ├─ feature を GPU に転送
    ├─ model(feature) を実行
    │   ├─ Encoder: 入力データを潜在表現に変換
    │   ├─ Decoder: 複数の軌跡モードを生成
    │   └─ Head: 各モードの確率を計算
    └─ 出力を抽出
        ├─ prediction: (1, max_agents, num_modes, T, 3)
        ├─ confidence: (1, max_agents, num_modes)
        └─ auxiliary: その他の情報

【ステップ4】出力処理
    ├─ Ego 軌跡のみを抽出
    │   (max_agents=64 中の第0要素)
    │
    ├─ モード軌跡を逆正規化
    │   ├─ Ego 中心座標 → グローバル座標
    │   └─ 座標変換行列を使用
    │
    ├─ 軌跡を TrajectoryWithProba に変換
    │   ├─ 位置列
    │   ├─ ヘディング列
    │   ├─ 速度列
    │   └─ 確率値
    │
    └─ 複数モードをリストで返却

【ステップ5】軌跡評価（オプション）
    ├─ 衝突検出
    ├─ 安全性チェック
    └─ 不適切な軌跡をフィルタリング
```

---

## 🔧 入出力形式の詳細

### 入力: PlannerInput

```python
class PlannerInput:
    iteration: int                    # シミュレーションステップ (0, 1, 2, ...)
    history: Tuple[...],             # 過去軌跡バッファ
    traffic_light_status: Dict,       # 交通信号状態
    scenario: AbstractScenario        # nuPlan シナリオ
```

### 出力: TrajectoryWithProba

```python
class TrajectoryWithProba:
    trajectory: Trajectory            # 軌跡オブジェクト
    probability: float                # モードの確率 [0, 1]
    
    # Trajectory の内容:
    trajectory.states: List[State]    # T個の State
    # State.position, State.velocity, State.acceleration など

# 返り値例
[
    TrajectoryWithProba(trajectory=Trajectory_mode_1, probability=0.6),
    TrajectoryWithProba(trajectory=Trajectory_mode_2, probability=0.3),
    TrajectoryWithProba(trajectory=Trajectory_mode_3, probability=0.1),
]
```

---

## 🎯 座標変換の詳細

### 正規化座標 → グローバル座標

```python
def denormalize_trajectory(
    self,
    normalized_pred: torch.Tensor,    # (num_modes, T, 3) in Ego frame
    ego_state: EgoState,              # Ego の現在状態
) -> torch.Tensor:
    """
    Args:
        normalized_pred: Ego 中心座標系での予測
            [
                [ego_x_offset, ego_y_offset, yaw]_t=0,
                [ego_x_offset, ego_y_offset, yaw]_t=1,
                ...
            ]
        
        ego_state: Ego の現在状態
            position: (x, y)
            heading: yaw [rad]
    
    Returns:
        global_pred: グローバル座標系での予測
    """
    
    ego_x, ego_y = ego_state.position
    ego_yaw = ego_state.heading
    
    # 回転行列
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    
    num_modes, T = normalized_pred.shape[:2]
    global_pred = torch.zeros_like(normalized_pred)
    
    for m in range(num_modes):
        for t in range(T):
            # Ego 座標系の位置
            local_x = normalized_pred[m, t, 0]
            local_y = normalized_pred[m, t, 1]
            
            # グローバル座標に変換
            global_x = ego_x + local_x * cos_yaw - local_y * sin_yaw
            global_y = ego_y + local_x * sin_yaw + local_y * cos_yaw
            
            # ヘディングはそのまま加算
            global_yaw = ego_yaw + normalized_pred[m, t, 2]
            
            global_pred[m, t] = torch.tensor([global_x, global_y, global_yaw])
    
    return global_pred
```

---

## 🚀 実装例

### 基本的な推論

```python
from src.models.pluto.pluto_model import PlutoModel
from src.planners.pluto_planner import PlutoPlanner
from hydra import compose, initialize_config_dir
import os

# 設定読み込み
config_dir = "/home/takuya/work/autonomous/pluto/config"
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="default_training")

# プランナー作成
planner = PlutoPlanner(
    config=cfg,
    checkpoint_path="/path/to/pluto_checkpoint.ckpt"
)

# 推論
planner_input = PlannerInput(
    iteration=0,
    history=...,
    scenario=scenario,
    traffic_light_status={}
)

trajectories = planner.compute_planner_trajectory(planner_input)

# 結果の処理
for idx, traj_with_proba in enumerate(trajectories):
    trajectory = traj_with_proba.trajectory
    probability = traj_with_proba.probability
    
    print(f"Mode {idx}: probability={probability:.3f}")
    print(f"  Final position: {trajectory.states[-1].position}")
    print(f"  Final velocity: {trajectory.states[-1].velocity}")
```

### nuPlan シミュレーションでの使用

```python
from nuplan.planning.scenario_builder.scenario_builder import ScenarioBuilder
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

# シナリオ読み込み
scenario_builder = ScenarioBuilder(...)
scenarios = scenario_builder.get_scenarios(
    scenario_types=["mini_demo_scenario"],
    limit=10
)

# シミュレーション初期化
simulation = Simulation(...)

# プランナーを使用した計画
for scenario in scenarios:
    planner_input = simulation.build_planner_input(scenario)
    
    # PLUTO による軌跡生成
    trajectories = planner.compute_planner_trajectory(planner_input)
    
    # 最も確率の高いモードを選択
    best_trajectory = max(trajectories, key=lambda x: x.probability).trajectory
    
    # シミュレーションに入力
    simulation.step(best_trajectory)
```

---

## 📊 マルチモーダル予測の扱い

### モード統合戦略

```python
class MultimodalTrajectorySelector:
    """複数モードから最終軌跡を選択"""
    
    @staticmethod
    def select_best_mode(trajectories: List[TrajectoryWithProba]) -> Trajectory:
        """確率が最も高いモードを選択"""
        best = max(trajectories, key=lambda x: x.probability)
        return best.trajectory
    
    @staticmethod
    def select_safest_mode(
        trajectories: List[TrajectoryWithProba],
        collision_checker
    ) -> Trajectory:
        """衝突のないモードの中で確率が最も高いものを選択"""
        
        safe_trajectories = [
            t for t in trajectories
            if not collision_checker.has_collision(t.trajectory)
        ]
        
        if not safe_trajectories:
            # 衝突回避が不可能な場合、確率最高を選択
            return max(trajectories, key=lambda x: x.probability).trajectory
        
        return max(safe_trajectories, key=lambda x: x.probability).trajectory
    
    @staticmethod
    def sample_from_distribution(
        trajectories: List[TrajectoryWithProba]
    ) -> Trajectory:
        """確率分布からサンプリング"""
        
        probs = [t.probability for t in trajectories]
        idx = np.random.choice(len(trajectories), p=probs)
        
        return trajectories[idx].trajectory
```

---

## 🔍 推論時のデバッグ

### 推論の可視化

```python
import matplotlib.pyplot as plt

def visualize_prediction(
    scenario, 
    current_state,
    trajectories,
    map_api
):
    """予測軌跡の可視化"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # マップを描画
    plot_map(ax, map_api, current_state.position)
    
    # Ego を描画
    ego_x, ego_y = current_state.position
    ax.plot(ego_x, ego_y, 'ko', markersize=10, label='Ego')
    
    # 予測軌跡を描画
    colors = ['r', 'g', 'b', 'orange', 'purple']
    for idx, traj_with_proba in enumerate(trajectories):
        traj = traj_with_proba.trajectory
        proba = traj_with_proba.probability
        
        # 軌跡点を抽出
        positions = [state.position for state in traj.states]
        xs, ys = zip(*positions)
        
        # プロット
        color = colors[idx % len(colors)]
        ax.plot(xs, ys, color=color, alpha=0.7, 
                label=f'Mode {idx} (p={proba:.2f})')
    
    # 周辺エージェント
    plot_agents(ax, scenario, current_state)
    
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("prediction_viz.png", dpi=150)
    plt.show()

# 使用例
visualize_prediction(scenario, ego_state, trajectories, map_api)
```

---

## 🔗 関連処理パイプライン

```
入力 (PlannerInput)
    ↓
【特徴量抽出】
    PlutoFeatureBuilder.build()
    ↓
【正規化】
    Ego中心座標系に変換
    ↓
【推論】
    model(feature)
    ↓
【逆正規化】
    グローバル座標に変換
    ↓
【軌跡評価】（オプション）
    TrajectoryEvaluator
    ↓
【出力】(List[TrajectoryWithProba])
    - trajectory: Trajectory オブジェクト
    - probability: 各モードの確率
    ↓
【モード選択】
    MultimodalTrajectorySelector
    ↓
【最終軌跡】(Trajectory)
    シミュレーション実行
```

---

## 📈 推論性能の最適化

### バッチ推論

```python
def batch_inference(
    planner: PlutoPlanner,
    scenarios: List[AbstractScenario],
    batch_size: int = 32
) -> Dict[str, Any]:
    """複数シナリオの一括推論"""
    
    all_trajectories = {}
    
    for batch_start in range(0, len(scenarios), batch_size):
        batch_end = min(batch_start + batch_size, len(scenarios))
        batch = scenarios[batch_start:batch_end]
        
        # バッチの特徴量を抽出
        features = [
            planner.feature_builder(s, iteration=0) 
            for s in batch
        ]
        
        # バッチ化
        batched_features = PlutoFeature.collate(features)
        
        # 一括推論
        with torch.no_grad():
            outputs = planner.model(batched_features)
        
        # 各シナリオの結果を分離
        for i, scenario in enumerate(batch):
            all_trajectories[scenario.scenario_name] = outputs[i]
    
    return all_trajectories
```

---

## 📚 関連ファイル

- [../feature_builders/pluto_feature_builder.md](../feature_builders/pluto_feature_builder.md) - 特徴量抽出
- [../models/pluto_model.md](../models/pluto_model.md) - モデルアーキテクチャ
- [../post_processing/trajectory_evaluation.md](../post_processing/trajectory_evaluation.md) - 軌跡検証
