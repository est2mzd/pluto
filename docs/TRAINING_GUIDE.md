# PLUTO Training ガイド

PLUTOのトレーニングは2段階に分かれています：
1. **Feature Cache ステップ**（データ前処理）
2. **Training ステップ**（モデル学習）

---

## 1. Mini data

### 1.1. キャッシュ作成 for mini
- splietter: ./nuplan-devkit/nuplan/planning/script/config/common/splitter/
- scenario_builder: ./nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/

```bash
python run_training.py \
    py_func=cache  \
    +training=train_pluto \
    scenario_builder=nuplan_mini \
    scenario_filter=training_scenarios_tiny \
    cache.cache_path=/nuplan/exp/sanity_check \
    cache.cleanup_cache=true \
    worker=sequential
```

### 1.2. 学習 for mini
```bash
python run_training.py \
    py_func=train \
    +training=train_pluto \
    scenario_builder=nuplan_mini \
    scenario_filter=training_scenarios_tiny \
    cache.cache_path=/nuplan/exp/sanity_check \
    worker=sequential
```




python run_training.py
  py_func=cache +training=train_pluto
  scenario_builder=nuplan_mini
  scenario_filter=training_scenarios_tiny
  cache.cache_path=/nuplan/exp/sanity_check
  cache.cleanup_cache=true
  worker=sequential


## 2. Boston data

### 2.1. キャッシュ作成 for Boston
```bash
python run_training.py \
   py_func=cache \
   +training=train_pluto \
   scenario_builder=nuplan_mini \
   scenario_filter=training_scenarios_tiny \
   cache.cache_path=/nuplan/exp/sanity_check \
   cache.cleanup_cache=true \
   worker=sequential
```

### 2.2. 学習 for Boston
```bash
python run_training.py \
  py_func=train \
  +training=train_pluto \
  scenario_builder=nuplan_mini \
  scenario_filter=training_scenarios_tiny \
  cache.cache_path=/nuplan/exp/sanity_check \
  worker=sequential
```

```bash
python run_training.py \
  py_func=train \
  +training=train_pluto \
  splitter=nuplan_boston \
  scenario_builder=nuplan_boston \
  scenario_filter=training_scenarios_boston \
  cache.cache_path=/nuplan/exp/cache_boston \
  worker=sequential
```

### 各引数の説明と根拠


### 目的

- ✅ 環境構築が正しくできているか確認
- ✅ nuplan-devkit との連携が正常か確認
- ✅ GPU メモリ使用量を確認
- ✅ キャッシュ生成プロセスが正常か確認

### 実行時間

通常 **5～10分程度**

---

## 2. 本体 Training

### コマンド : mini


### コマンド : boston
- splietter: ./nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan_boston.yaml
- scenario_builder: ./nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan_boston.yaml




### Hydra の デバッグ方法
- 「Hydra が最終的に作った設定の完成形をそのまま表示する」ための公式デバッグ手段です
```
python run_training.py --cfg job
```











## 3. Simulation
- pluto_1M_aux_cil.ckpt は READMEにリンクがある

```bash
python run_simulation.py \
  planner=pluto_planner \
  planner.ckpt_path=checkpoints/pluto_1M_aux_cil.ckpt \
  scenario_builder=nuplan_mini \
  scenario_filter=training_scenarios_tiny
```