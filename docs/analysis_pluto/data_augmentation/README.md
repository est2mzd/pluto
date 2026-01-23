# data_augmentation モジュール解説

## 📋 概要

`data_augmentation` モジュールは、PLUTOの学習データを拡張・生成するための機能を提供します。

**主な役割：**
- 既存のシナリオから、類似だが異なるシナリオを生成
- ポジティブサンプル（似たシナリオ）とネガティブサンプル（対比シナリオ）を生成
- 対比学習（Contrastive Learning）で、モデルの表現学習を強化

### 🎯 なぜデータ拡張が必要？

**問題**: 訓練データが少ない
```
実際のシナリオ: 1000個
        ↓
これだけでは、モデルが過学習してしまう
```

**解決**: データ拡張
```
実際のシナリオ: 1000個
        ↓
各シナリオから ポジティブ/ネガティブ サンプル生成
        ↓
訓練データ: 3000個以上に増加
        ↓
モデルがより堅牢に学習
```

---

## 📁 ファイル構成

| ファイル | 役割 | 詳細ドキュメント |
|---------|------|---------|
| `contrastive_scenario_generator.py` | データ拡張エンジン | [詳細](./contrastive_scenario_generator.md) |
| `__init__.py` | モジュール定義 | 外部公開インターフェース |

---

## 🔄 対比学習（Contrastive Learning）とは

### 概念図

```
【元のシナリオ】
  Ego車: 位置(x, y), 向き(yaw), 速度v
  他のエージェント: 複数
  地図情報: 交通信号など
        ↓
   ┌─────┴─────┐
   ↓           ↓
【ポジティブサンプル】 【ネガティブサンプル】
Ego を少し移動        Ego を大きく移動
  + 衝突チェック       or 交通信号反転
  + 無関係エージェント  or インタラクティブ
    をドロップ         エージェント削除
        ↓                  ↓
  似ているシナリオ   全く異なるシナリオ
        ↓                  ↓
    ┌───────────────────┘
    ↓
モデルは学習:
「ポジティブは類似表現」
「ネガティブは異なる表現」
```

### 対比学習が学習効率を上げる理由

```
【通常の監視学習】
Scenario A → モデル → 軌跡 A ✓
Scenario B → モデル → 軌跡 B ✓
...
      ↓
表現学習: 不十分

【対比学習】
Scenario A → モデル → 表現 A
Pos(A)    → モデル → 表現 A'  ← 類似！
Neg(A)    → モデル → 表現 A'' ← 異なる！
      ↓
表現学習: 強化！
モデルが「何が重要か」を学べる
```

---

## 🎨 主な拡張方法

### 1. ポジティブサンプル生成

#### 方法: Ego車の小さな摂動
```
元: Ego = (x: 10.0, y: 20.0, yaw: 0.5 rad, v: 5.0 m/s)
    ↓
ノイズ生成: Δx ∈ [-1.5, 2.0], Δy ∈ [-1.5, 1.5], Δyaw ∈ [-0.35, 0.35]
    ↓
新: Ego = (x: 10.5, y: 20.3, yaw: 0.52 rad, v: 5.1 m/s)
    ↓
衝突チェック: 他の車と衝突してないか確認
    ↓
出力: ポジティブサンプル
```

**特徴:**
- Ego車の位置・向き・速度を少しだけ変更
- 衝突チェックで安全性を確保
- 無関係なエージェント（インタラクティブでない）を確率的にドロップ
- コスト地図（occupancy grid）も回転・シフト

### 2. ネガティブサンプル生成

#### 方法A: 交通信号反転
```
【前】赤信号で停止中
  Ego car: 停止（v=0）
  信号: RED
    ↓
【後】信号を GREEN に反転
  Ego car: 同じ（v=0）→ 走行すべき
  信号: GREEN ← 矛盾！
    ↓
モデル: 「この矛盾を検出できるか？」
```

#### 方法B: インタラクティブエージェント削除
```
【前】他の車と相互作用
  Ego: 加速中（v=2 m/s）
  他の車: 急ブレーキ（v=3→0）
  -> Ego は危険を避ける
    ↓
【後】相互作用している車を削除
  Ego: 加速中（v=2 m/s）
  他の車: なし
  -> 同じ行動は矛盾！
    ↓
モデル: 「隠された脅威を検出できるか？」
```

#### 方法C: エージェント挿入
```
【前】自由な経路
  Ego: 前方に障害物なし
  -> 直進
    ↓
【後】自由な経路上にエージェント追加
  Ego: 前方に新車登場（速度類似）
  -> 同じ行動は矛盾！
    ↓
モデル: 「新しい障害物を処理できるか？」
```

---

## 💡 実装のポイント

### 1. 衝突チェック（CollisionChecker）

```python
# Ego 車の円形 or 矩形で、他車との衝突判定
def safety_check(
    ego_position,    # (x, y)
    ego_heading,     # yaw角度
    agents_position, # (N, 2)
    agents_heading,  # (N,)
    agents_shape     # (N, 2) = [width, length]
) -> bool:
    # 衝突なし: True, 衝突あり: False
```

**背景**: PLUTOは実自動車なので、生成されたシナリオが物理的に妥当である必要がある

### 2. スケーリング機構

```python
# ノイズが大きすぎて衝突した場合
num_tries = 0
while num_tries < 5:
    if not self.safety_check(...):
        scale *= 0.5  # ノイズを半減
        num_tries += 1
    else:
        break
```

**背景**: 小さなノイズを試していくことで、安全な拡張を探索

### 3. エージェント生成

```python
# ネガティブサンプルで、新しいエージェントを挿入する際
# 既存エージェントと速度が似ているものをテンプレートとして使用

agents_velocity_diff = abs(others_velocity - ego_velocity)
copy_agent_idx = argmin(agents_velocity_diff)  # 最も似た速度のエージェント
```

**背景**: ランダムなエージェントより、既存の動きを参考にしたほうが自然

---

## 🔧 使用方法

### 設定ファイル

[config/data_augmentation/contrastive_scenario_generator.yaml](../../../config/data_augmentation/contrastive_scenario_generator.yaml):

```yaml
_target_: src.data_augmentation.contrastive_scenario_generator.ContrastiveScenarioGenerator
history_steps: 21           # 過去21ステップを使用
max_interaction_horizon: 40 # 40ステップ以内の相互作用を検出
low: [0.0, -1.5, -0.35, -1, -0.5, -0.2, -0.2]     # ノイズ下限
high: [2.0, 1.5, 0.35, 1, 0.5, 0.2, 0.2]          # ノイズ上限
use_negative_sample: true   # ネガティブサンプルを生成
```

### 訓練コマンド

```bash
# データ拡張を有効にして訓練
python run_training.py \
  py_func=train \
  +training=train_pluto \
  +data_augmentation=contrastive_scenario_generator \
  scenario_builder=nuplan \
  cache.cache_path=/path/to/cache
```

### パラメータ調整

| パラメータ | 説明 | 調整のヒント |
|-----------|------|-----------|
| `history_steps` | 過去何ステップを使用するか | 長いほど過去の情報を利用（計算量増） |
| `max_interaction_horizon` | インタラクティブと判定する最大距離 | 大きいほど、より多くのエージェントを重要と認識 |
| `low / high` | ノイズの範囲 | 大きいほど激しく拡張（訓練は難化） |
| `use_negative_sample` | ネガティブサンプル生成するか | `true`: 対比学習 / `false`: ポジティブのみ |

---

## 📊 期待される効果

### データ拡張なし
```
訓練データ: 1000シナリオ
      ↓
モデル学習:
  - 訓練精度: 95%
  - 検証精度: 70%  ← 過学習！
```

### データ拡張あり
```
訓練データ: 1000シナリオ
  ↓
拡張後: 3000シナリオ（3倍！）
      ↓
モデル学習:
  - 訓練精度: 85%
  - 検証精度: 82%  ← バランス改善！
```

### 対比学習の効果
```
通常学習:
  - 訓練精度: 85%
  - 検証精度: 82%
  
対比学習:
  - 訓練精度: 83%
  - 検証精度: 85%  ← 検証性能向上！
```

---

## 🐛 よくあるエラーと対処

### Error: `CollisionChecker` がインポートできない
```
原因: pytorch のバージョン不一致
解決: src/utils/collision_checker.py を確認
```

### Warning: 衝突チェックで全てのノイズが失敗
```
原因: max_interaction_horizon が小さすぎる
解決: max_interaction_horizon を大きくする
```

### 生成されたシナリオが不自然
```
原因: ノイズ範囲が大きすぎる
解決: low / high パラメータを縮小
```

---

## 🚀 応用例

### 例1: 難しい訓練用データセット作成
```bash
# ノイズ範囲を大きくして、より変わったシナリオを生成
python run_training.py \
  data_augmentation.low=[-2.0, -3.0, -0.7, -2, -1.0, -0.4, -0.4] \
  data_augmentation.high=[3.0, 3.0, 0.7, 2, 1.0, 0.4, 0.4]
```

### 例2: 対比学習を無効にしてポジティブのみ使用
```bash
python run_training.py \
  data_augmentation.use_negative_sample=false
```

### 例3: ネガティブサンプルの種類を限定
```python
# contrastive_scenario_generator.py を編集して、
# available_generators から方法を選別
```

---

## 📚 参考資料

- [PyTorch Lightning での Data Augmentation](https://lightning.ai/docs/pytorch/stable/advanced/augmentation.html)
- [Contrastive Learning 論文](https://arxiv.org/abs/2002.05709)
- [nuPlan Dataset](https://www.nuplandataset.org/)

---

## 次のステップ

詳細は以下のドキュメントを参照してください：

- 📄 [contrastive_scenario_generator.py の詳細](./contrastive_scenario_generator.md)
