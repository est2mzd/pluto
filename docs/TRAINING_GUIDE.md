# PLUTO Training ガイド

PLUTOのトレーニングは2段階に分かれています：
1. **Feature Cache ステップ**（データ前処理）
2. **Training ステップ**（モデル学習）

---

## 1. Sanity Check（Feature Cache）

### コマンド
```bash
python run_training.py \
   py_func=cache +training=train_pluto \
   scenario_builder=nuplan_mini \
   cache.cache_path=/nuplan/exp/sanity_check \
   cache.cleanup_cache=true \
   scenario_filter=training_scenarios_tiny \
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

### コマンド
```bash
python run_training.py \
  py_func=train \
  +training=train_pluto \
  scenario_builder=nuplan_mini \
  scenario_filter=training_scenarios_tiny \
  cache.cache_path=/nuplan/exp/sanity_check \
  worker=sequential
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