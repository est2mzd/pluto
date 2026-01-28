# /workspace/pluto/notebook README

このREADMEは **READMEだけで理解できること** を目標に、以下の3つを
**ユーザーが実際に可視化・実行できる粒度** でまとめたチュートリアルです。

1. **nuPlan datasetのデータ構造と読み取り方（可視化まで）**  
2. **nuPlan devkitの使い方（何をどうすると何ができるか）**  
3. **Bostonデータを使った学習手順（NaN回避〜GPU学習まで）**

---

## 1. nuPlan datasetのデータ構造と読み取り方（可視化まで）

### 1.1 物理構成（ユーザーが迷わないための最小定義）
- **Log**: `.db` 1ファイル（SQLite）  
- **Scenario**: Logの中の連続した時間区間  
- **Frame / Iteration**: シナリオ内の1タイムステップ（Lidar 1フレーム）

### 1.2 必須の環境変数
nuPlan devkitは以下を前提に動きます。
- `NUPLAN_DATA_ROOT` : `.db` と `splits` の起点
- `NUPLAN_MAPS_ROOT` : mapデータのルート
- `NUPLAN_MAP_VERSION` : mapバージョン

### 1.3 どこに何があるか（Bostonの例）
```
/nuplan/dataset/nuplan-v1.1/
  splits/
    train_boston/
      *.db   ← この数がログ数（シナリオ数ではない）
```

**シナリオ数を把握したい場合**は、まず `.db` を数えるのが正しい出発点です。  
（ディレクトリ数は0で誤りになる）

### 1.4 `.db` を読み、シナリオを1つ取り出す
以下は「1つのシナリオを取得して中身を確認する」最小コードです。

```python
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

# 1) ScenarioBuilder を作る（環境変数が必要）
scenario_builder = NuPlanScenarioBuilder(
    data_root=os.environ["NUPLAN_DATA_ROOT"],
    map_root=os.environ["NUPLAN_MAPS_ROOT"],
    map_version=os.environ["NUPLAN_MAP_VERSION"],
)

# 2) ScenarioFilter を作る（最小で1件だけ）
scenario_filter = ScenarioFilter(
    log_names=None,
    map_names=None,
    scenario_tokens=None,
    num_scenarios_per_type=1,
    limit_total_scenarios=1,
    timestamp_threshold_s=0.0,
    ego_displacement_minimum_m=0.0,
    ego_start_speed_threshold=0.0,
    ego_stop_speed_threshold=0.0,
    expand_scenarios=True,
    remove_invalid_goals=True,
    shuffle=True,
)

worker = Sequential()
scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
scenario = scenarios[0]
```

### 1.5 シナリオから「可視化に必要な最低限の情報」を取る
**ユーザーが可視化できるレベルの情報**は以下です。

```python
# フレーム数
num_frames = scenario.get_number_of_iterations()

# Lidarフレームのトークン列（可視化に使う）
tokens = scenario.get_scenario_tokens()

# Egoの軌跡（位置・向き）
ego_xy = []
for i in range(num_frames):
    ego_state = scenario.get_ego_state_at_iteration(i)
    ego_xy.append((ego_state.rear_axle.x, ego_state.rear_axle.y))
```

### 1.6 Ego軌跡を簡単に可視化する
```python
import matplotlib.pyplot as plt

xs = [p[0] for p in ego_xy]
ys = [p[1] for p in ego_xy]

plt.figure(figsize=(6,6))
plt.plot(xs, ys, linewidth=2)
plt.axis("equal")
plt.title("Ego Trajectory")
plt.show()
```

### 1.7 データ解像度の注意点（8点 vs 80ステップ）
- **8点の軌跡**は「可視化用の粗サンプル」。  
- 学習・推論は **0.1秒刻みの高解像度**（未来 8秒 = 80ステップ）を前提。  
  - **入力**: 過去 2秒 → 21ステップ  
  - **出力**: 未来 8秒 → 80ステップ  
- **8点軌跡 = 学習用の正規データではない** と理解することが重要。

---

## 2. nuPlan devkitの使い方（何をどうしたら何ができるか）

### 2.1 devkitでできること（具体例）
1. `.db` からシナリオを抽出  
2. ScenarioFilter で「特定条件のシナリオだけ」を抽出  
3. シナリオのフレームを可視化（map + Lidar）  
4. GIFとしてアニメーション出力  
5. Egoの速度・位置・軌跡などを時系列で可視化

### 2.2 何をどうすると「シナリオ抽出」できるか
**やること**: `ScenarioFilter` を作り、`scenario_builder.get_scenarios()` で抽出する。

```python
scenario_filter = ScenarioFilter(
    log_names=["2021.08.12.12.35.09_veh-28_02678_02925"],  # 例: 特定ログのみ
    map_names=None,
    scenario_tokens=None,
    num_scenarios_per_type=1,
    limit_total_scenarios=1,
    timestamp_threshold_s=0.0,
    ego_displacement_minimum_m=2.0,  # 2m以上動いたシナリオのみ
    ego_start_speed_threshold=0.0,
    ego_stop_speed_threshold=0.0,
    expand_scenarios=True,
    remove_invalid_goals=True,
    shuffle=True,
)
```

### 2.3 何をどうすると「map + Lidar を描画」できるか
**やること**: `LidarPc.render` を呼ぶ。  
（内部で `render_on_map` が呼ばれ、map + 点群 + アノテーションが描画される）

```python
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc

token = tokens[0]  # 先頭フレームのLidarトークン
lidar = LidarPc(token, scenario._db)  # scenario._db は内部で持つDB
fig = lidar.render()
```

### 2.4 何をどうすると「GIFアニメーション」が作れるか
**やること**: 各フレームを `LidarPc.render` で描き、`FuncAnimation` でGIF化する。

```python
import matplotlib.pyplot as plt
from matplotlib import animation

fig = plt.figure(figsize=(6,6))

def update(i):
    token = tokens[i]
    lidar = LidarPc(token, scenario._db)
    fig.clear()
    lidar.render(ax=plt.gca())

ani = animation.FuncAnimation(fig, update, frames=len(tokens))
ani.save("/workspace/pluto/notebook/outputs/scenario.gif", writer="pillow")
```

---

## 3. Bostonデータを使った学習手順（NaN回避〜GPU学習まで）

### 3.1 重要な前提: TrajectorySamplingの不整合を修正
**問題**: `future_steps=80` なのに `TrajectorySampling(num_poses=8, interval_length=1)`  
→ **shape mismatch (8,3) vs (80,3)** で NaN が発生。

**修正（必須）**:
```python
# /workspace/pluto/src/models/pluto/pluto_model.py
trajectory_sampling = TrajectorySampling(
    num_poses=80,
    time_horizon=8,
    interval_length=0.1,
)
```

この修正により **CPU学習でNaNが消えることを確認済み**。

### 3.2 notebook環境での学習実行方針
- **subprocess.run() は使わない**  
  - Hydra設定が反映されず誤解を生むため。
- **Python APIで実行する**  
  - `initialize_config_dir` + `compose` + `run_training.main(cfg)`

最小例:
```python
from hydra import initialize_config_dir, compose
from nuplan.planning.script.run_training import main as training_main

config_dir = "/workspace/pluto/config"
overrides = [
    "scenario_filter=training_scenarios_boston",
    "splitter=ratio_splitter",
    "scenario_builder=nuplan_boston",
    "model=pluto_model",
]

with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
    cfg = compose(config_name="default_training", overrides=overrides)
    training_main(cfg)
```

### 3.3 notebookで必要なDDP設定
- `ddp_find_unused_parameters_false` は **notebookでは非対応**。  
- notebook環境では `ddp_notebook` を使う。

```python
overrides += ["lightning.trainer.params.strategy=ddp_notebook"]
```

### 3.4 小規模でパイプライン確認
目的: データローダ・モデル・学習ループが動くか確認。  
方法: まず少量バッチで動作確認。

```python
overrides += [
    "lightning.trainer.params.max_epochs=1",
    "lightning.trainer.params.limit_train_batches=0.01",
    "lightning.trainer.params.limit_val_batches=0.01",
]
```

### 3.5 キャッシュ構造の整合性を確認する
- `pluto_model` は **`feature.gz`** を期待。  
- `simple_vector_model` は **`agents.gz` / `vector_map.gz` / `trajectory.gz`** 構造。  
- モデルとキャッシュ構造が一致しないと学習が失敗する。

### 3.6 分割キャッシュ戦略（大規模学習の基本）
- 1647シナリオを一括キャッシュすると重すぎる。  
- **分割（例: 400/400/400/447）でキャッシュ作成 → 学習** が妥当。  
- `scenario_filter` で部分集合を指定できるため、分割キャッシュは実装上問題ない。

### 3.7 1647シナリオ全データでの学習コマンド（GPU）
**キャッシュ作成**
```bash
python run_training.py \
  py_func=cache \
  +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_1647_full \
  cache.force_feature_computation=True
```

**学習実行**
```bash
python run_training.py \
  py_func=train \
  +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_1647_full \
  lightning.trainer.params.max_epochs=100 \
  lightning.trainer.params.accelerator=gpu \
  lightning.trainer.params.devices=1
```

### 3.8 失敗パターン（回避事項）
1. **subprocess.run() を使う**  
   - Hydraの新規YAMLが認識されず、誤解を招くエラーが出る。
2. **モデルとキャッシュ構造の不整合**  
   - `pluto_model` 用キャッシュは `feature.gz` を含む必要がある。
3. **8点軌跡を学習用に使えると誤解する**  
   - 学習は 80ステップ前提。

