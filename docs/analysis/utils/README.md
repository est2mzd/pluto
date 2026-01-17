# utils モジュール解説

## 📋 概要

`utils` は、プロジェクト全体で共通的に使用される **補助関数・ユーティリティ** を集約するモジュールです。

**主な役割：**
- 衝突判定
- 画像処理（クロップ、回転）
- 可視化
- 一般的なユーティリティ

---

## 📁 ファイル構成

| ファイル | 役割 |
|---------|------|
| `collision_checker.py` | 衝突判定エンジン |
| `utils.py` | 汎用補助関数 |
| `vis.py` | 可視化ツール |

---

## 🔑 主要モジュール

### 1. `CollisionChecker`

#### 役割
```
Ego 車と周辺オブジェクトの衝突判定:
  - 円形衝突判定
  - 矩形衝突判定
  - GPU 高速化対応
```

#### 使用例

```python
from src.utils.collision_checker import CollisionChecker

checker = CollisionChecker()

# 衝突判定
collisions = checker.collision_check(
    ego_state=ego_tensor,        # (batch, 3): [x, y, θ]
    objects=objects_tensor,      # (batch, N, 3): [x, y, θ]
    objects_width=width_tensor,  # (batch, N)
    objects_length=length_tensor # (batch, N)
)
# 出力: collisions (batch, N) bool テンソル
```

#### 内部処理

```
【入力】
  Ego: 位置(x, y)と向き(θ)
  Object: 複数の位置・向き・サイズ
        ↓
【Step 1】Ego の矩形を生成
  向き(θ)に基づいて回転した矩形
        ↓
【Step 2】各 Object との衝突判定
  回転矩形と矩形のIntersection判定
        ↓
【出力】
  collisions[i] = True/False （i番目オブジェクト）
```

---

### 2. `utils.py` 補助関数

#### 主要関数

| 関数 | 説明 |
|------|------|
| `crop_img_from_center()` | 画像を中心からクロップ |
| `shift_and_rotate_img()` | 画像をシフト・回転 |
| `to_tensor()` | NumPy → PyTorch |
| `to_numpy()` | PyTorch → NumPy |
| `to_device()` | デバイス転送 |

#### 具体例

```python
from src.utils.utils import crop_img_from_center, shift_and_rotate_img

# 画像をクロップ（500x500）
cropped = crop_img_from_center(image, (500, 500))

# 画像をシフト・回転
transformed = shift_and_rotate_img(
    img=image,
    shift=[1.0, 2.0, 0],      # (Δx, Δy, Δz)
    angle=0.5,                # 回転角（ラジアン）
    resolution=0.2,           # メートル/ピクセル
    cval=-200                 # パディング値
)
```

#### 変換処理の説明

```python
# Affine 変換で画像を変形
# 例: Ego が (x, y) だけシフト＆θ回転

# シフト: [shift_x, shift_y]
#        - resolution で正規化
#        - ピクセル単位に変換

# 回転: angle ラジアン
#      - 回転行列を生成
#      - 画像に適用
```

---

### 3. `vis.py` 可視化ツール

#### 役割
```
シナリオの可視化:
  - Ego 車
  - 周辺エージェント
  - 地図・車線
  - 予測軌跡
  - コスト地図
```

#### 使用例

```python
from src.utils.vis import plot_scenario

# シナリオをプロット
plot_scenario(
    scenario=scenario,
    trajectory=predicted_trajectory,
    savepath="./output/scenario_vis.png"
)
```

#### 出力例

```
┌─────────────────────────────┐
│         シナリオ可視化       │
│                             │
│    [車線] [Ego●]→           │
│           [他車]            │
│           [予測軌跡 ...]    │
│                             │
│    [障害物] [X]             │
└─────────────────────────────┘
```

---

## 💡 実装のポイント

### 1. GPU 対応

```python
# CPU と GPU の両方で動作
collisions = checker.collision_check(
    ego_state=ego.to(device),      # GPU に転送
    objects=objects.to(device)
)
```

### 2. バッチ処理

```python
# 複数の状態を同時に処理
batch_ego = (batch_size, 3)
batch_objects = (batch_size, num_objects, 3)

collisions = checker.collision_check(
    ego_state=batch_ego,
    objects=batch_objects,
    ...
)
# 出力: (batch_size, num_objects)
```

### 3. 効率的な変換

```python
# 冗長な copy を避ける
tensor = to_tensor(numpy_array)  # メモリ効率的
array = to_numpy(tensor)
```

---

## 🚀 使用例

### 衝突チェック付きシミュレーション

```python
from src.utils.collision_checker import CollisionChecker
from src.utils.utils import to_tensor

checker = CollisionChecker()

# シミュレーション
for t in range(num_steps):
    ego_state = [x, y, theta]
    
    # 衝突判定
    collisions = checker.collision_check(
        ego_state=to_tensor([ego_state]),
        objects=to_tensor(objects),
        objects_width=to_tensor(widths),
        objects_length=to_tensor(lengths)
    )
    
    if collisions.any():
        print("衝突検出！緊急停止")
        break
    
    # 続行
    x, y, theta = simulate_one_step(ego_state, control)
```

### 画像処理パイプライン

```python
from src.utils.utils import crop_img_from_center, shift_and_rotate_img

# コスト地図を処理
cost_map = feature.data["cost_maps"]

# Ego の位置に応じて変形
transformed = shift_and_rotate_img(
    img=cost_map,
    shift=[current_state[1], -current_state[0], 0],
    angle=-current_state[2],
    resolution=0.2,
    cval=-200
)

# 中心からクロップ
final = crop_img_from_center(transformed, (500, 500))
```

---

## 📊 パフォーマンス

| 操作 | 実行時間 |
|------|--------|
| 衝突判定（1 batch） | ~1-2 ms |
| 画像クロップ | ~0.5 ms |
| 画像回転 | ~1 ms |
| 可視化 | ~100 ms |

---

## 📚 関連ファイル

- [data_augmentation/README.md](../data_augmentation/README.md) - 衝突チェックの使用
- [post_processing/README.md](../post_processing/README.md) - 軌跡評価
- [feature_builders/README.md](../feature_builders/README.md) - 特徴量計算
