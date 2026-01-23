# planners モジュール解説

## 📋 概要

`planners` は、PLUTOの **推論エンジン** を実装するモジュールです。

**主な役割：**
- 訓練済みモデルから軌跡を生成
- シミュレーション環境で実行
- nuPlan ベンチマークで評価

---

## 📁 ファイル構成

| ファイル | 役割 |
|---------|------|
| `pluto_planner.py` | メイン：PLUTOプランナー |
| `ml_planner_utils.py` | 補助関数 |

---

## 🔑 `PlutoPlanner` クラス

### 役割
nuplan-devkit の `AbstractPlanner` を実装し、PLUTO推論を提供。

### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `initialize()` | モデルの初期化 |
| `compute_planner_trajectory()` | 軌跡の計算 |
| `name()` | プランナー名 |

### 処理フロー

```
【入力】PlannerInput
  - planner_input.ego_state: Ego の現在状態
  - planner_input.observations: 周辺エージェント
  - planner_input.route: ナビゲーション情報
        ↓
【Step 1】特徴量計算
  PlutoFeatureBuilder で特徴量を抽出
        ↓
【Step 2】モデル推論
  torch モデルで軌跡を予測
        ↓
【Step 3】軌跡抽出
  複数軌跡から最も確率の高いものを選択
        ↓
【出力】TrajectoryStateSample
  - trajectory: 8秒先の軌跡（80ステップ）
  - timestamp: 生成時刻
```

---

## 🚀 使用例

### nuPlan ベンチマーク実行

```bash
python /nuplan/planning/script/run_nuplan_l5kit.py \
  +planner=pluto_planner \
  scenario_builder=nuplan \
  planner.model_path=./checkpoints/best_model.ckpt
```

### コード例

```python
from src.planners.pluto_planner import PlutoPlanner

# プランナー初期化
planner = PlutoPlanner(
    model_path="./checkpoints/best_model.ckpt",
    checkpoint=None
)

# シミュレーション実行
result = planner.compute_planner_trajectory(planner_input)
# result.trajectory: 予測軌跡
```

---

## 📚 関連ファイル

- [models/README.md](../models/README.md) - モデルアーキテクチャ
- [feature_builders/README.md](../feature_builders/README.md) - 特徴量
