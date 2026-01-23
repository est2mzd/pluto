# post_processing モジュール解説

## 📋 概要

`post_processing` は、モデルの **出力軌跡の処理・評価** を行うモジュールです。

**主な役割：**
- 予測軌跡の検証・フィルタリング
- 衝突回避チェック
- 軌跡の滑らかさ評価
- 快適性メトリクス計算

---

## 📁 ファイル構成

```
post_processing/
├── trajectory_evaluator.py       # 軌跡評価エンジン
├── emergency_brake.py            # 緊急ブレーキ
├── common/
│   ├── enum.py                   # 定数定義
│   └── geometry.py               # 幾何学計算
├── evaluation/
│   └── comfort_metrics.py        # 快適性指標
├── forward_simulation/           # 物理シミュレーション
│   ├── batch_kinematic_bicycle.py # 車両モデル
│   ├── batch_lqr.py             # 最適制御
│   ├── batch_lqr_utils.py       # LQR補助関数
│   └── forward_simulator.py      # シミュレータ
└── observation/
    └── world_from_prediction.py  # 予測から世界を構築
```

---

## 🔑 主要コンポーネント

### 1. TrajectoryEvaluator

#### 役割
```
軌跡の妥当性を評価:
  ✓ 衝突なし
  ✓ 車線内
  ✓ 加速度制限内
  ✓ 回転角速度制限内
```

### 2. EmergencyBrake

#### 役割
```
危険を検出したら緊急ブレーキ:
  - 衝突予測
  - 急制動
  - フェイルセーフ
```

### 3. ComfortMetrics

#### 役割
```
乗客の快適性を評価:
  - 横加速度
  - 縦加速度
  - ジャーク（加速度の変化率）
```

### 4. ForwardSimulator

#### 役割
```
軌跡の物理的シミュレーション:
  - Kinematic Bicycle Model（車両モデル）
  - LQR制御（最適制御）
  - 実現可能性チェック
```

---

## 📊 計算フロー

```
【入力】
  - 予測軌跡（(80 steps, 2D)）
  - 周辺オブジェクト
        ↓
【Step 1】衝突チェック
  TrajectoryEvaluator で障害物との衝突判定
        ↓
【Step 2】物理シミュレーション
  ForwardSimulator で実行可能性チェック
        ↓
【Step 3】快適性評価
  ComfortMetrics で乗客快適性を計測
        ↓
【Step 4】安全チェック
  危険ならEmergencyBrake を作動
        ↓
【出力】
  - 検証済み軌跡
  - 安全性スコア
  - 快適性スコア
```

---

## 💡 実装のポイント

### 1. Batch Processing

```python
# 複数軌跡を並列評価
batch_trajectories = (batch_size, k=6, 80, 2)

# GPU で高速化
evaluated = evaluate_trajectories_batch(
    batch_trajectories,
    batch_objects
)
```

### 2. LQR 制御

```
軌跡が実現可能か確認:
  - 目標軌跡に追従可能？
  - 制御入力は妥当？
  → LQR (Linear Quadratic Regulator)
     で検証
```

### 3. Kinematic Bicycle Model

```
車両の運動モデル:
  x_dot = v * cos(θ)
  y_dot = v * sin(θ)
  θ_dot = v * tan(δ) / L
  
  v: 速度
  δ: ステアリング角
  L: ホイールベース
  θ: 向き
```

---

## 🚀 使用例

### 軌跡の評価

```python
from src.post_processing.trajectory_evaluator import TrajectoryEvaluator

evaluator = TrajectoryEvaluator()

# 軌跡の妥当性をチェック
is_valid = evaluator.evaluate(
    trajectory=predicted_trajectory,
    ego_state=current_ego,
    objects=surrounding_objects
)

if not is_valid:
    print("軌跡が不安全です")
```

### 快適性の計算

```python
from src.post_processing.evaluation.comfort_metrics import ComfortMetrics

comfort = ComfortMetrics()

scores = comfort.compute(trajectory)
# {
#   "max_lateral_acceleration": 0.5,  # m/s²
#   "max_longitudinal_acceleration": 1.0,
#   "max_jerk": 0.2,
#   "comfort_score": 0.85
# }
```

### 物理シミュレーション

```python
from src.post_processing.forward_simulation.forward_simulator import ForwardSimulator

simulator = ForwardSimulator(
    vehicle_model="kinematic_bicycle"
)

# 軌跡を制御で実現可能か確認
is_feasible, control_inputs = simulator.simulate(
    trajectory=predicted_trajectory,
    ego_state=current_ego
)
```

---

## 📊 快適性指標の基準

| 指標 | 快適 | 普通 | 不快 |
|------|------|------|------|
| 横加速度 | < 0.5 m/s² | 0.5-1.0 | > 1.0 |
| 縦加速度 | < 1.0 m/s² | 1.0-2.0 | > 2.0 |
| ジャーク | < 0.2 m/s³ | 0.2-0.5 | > 0.5 |

---

## 📚 関連ファイル

- [models/README.md](../models/README.md) - モデル出力
- [metrics/README.md](../metrics/README.md) - 性能評価
- [planners/README.md](../planners/README.md) - 推論エンジン
