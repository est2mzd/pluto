# optim モジュール解説

## 📋 概要

`optim` は、モデルの **学習率スケジューラー** を実装するモジュールです。

**主な役割：**
- 訓練中に学習率を動的に調整
- 最初は低い学習率でウォームアップ
- その後、コサイン減衰で徐々に学習率を低下

### 🎯 なぜ学習率スケジューラーが必要？

```
固定学習率での訓練:
  初期: 学習率が高 → 発散しやすい
  中盤: 最適化が不安定
  末期: 最小値に到達しにくい

  → 訓練が失敗しやすい

可変学習率での訓練:
  初期: 低い学習率でウォームアップ
  中盤: 高い学習率で高速最適化
  末期: 低い学習率で微調整
  
  → より高い精度に到達可能
```

---

## 📁 ファイル構成

| ファイル | 役割 |
|---------|------|
| `warmup_cos_lr.py` | メイン：WarmupCosLR クラス |

---

## 🔑 `WarmupCosLR` クラス

### 役割
PyTorch の `_LRScheduler` を拡張し、ウォームアップ + コサイン減衰スケジューラーを実装。

### シグネチャ

```python
class WarmupCosLR(_LRScheduler):
    def __init__(
        self,
        optimizer,        # 最適化器
        min_lr,          # 最小学習率
        lr,              # 最大学習率
        warmup_epochs,   # ウォームアップエポック数
        epochs,          # 総エポック数
        last_epoch=-1,   # 再開時のエポック番号
        verbose=False    # 詳細ログ
    )
```

### パラメータ詳細

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `min_lr` | 最小学習率（末期の学習率） | `1e-5` |
| `lr` | 最大学習率（ウォームアップ後） | `1e-3` |
| `warmup_epochs` | ウォームアップ期間 | `3` |
| `epochs` | 総訓練エポック | `25` |
| `last_epoch` | チェックポイント再開時 | `5` (5エポック目から再開) |

---

## 📈 学習率の時間変化

### グラフ

```
学習率
  ↑
  │        ╱╲
lr│       ╱  ╲___
  │      ╱       ╲____
  │     ╱             ╲___
  │    ╱                  ╲_____
min_lr├─────────────────────────────
  │   |====|  |================|
  └─────────────────────────────→ エポック
  0    ↑    ↑                  ↑
       3   5                  25
  warmup  最大点             終了
```

### 段階の説明

#### Phase 1: ウォームアップ（エポック 0～3）

```python
# warmup_epochs = 3

lr(epoch=0) = lr * (0+1) / 3 = 1e-3 * 1/3 ≈ 3.3e-4
lr(epoch=1) = lr * (1+1) / 3 = 1e-3 * 2/3 ≈ 6.7e-4
lr(epoch=2) = lr * (2+1) / 3 = 1e-3 * 3/3 = 1e-3
```

**目的**: 初期の不安定性を抑える

#### Phase 2: コサイン減衰（エポック 3～25）

```python
# warmup_epochs = 3, epochs = 25

t = (epoch - warmup_epochs) / (epochs - warmup_epochs)
# エポック 5: t = (5-3) / (25-3) = 2/22 ≈ 0.09

lr(epoch) = min_lr + 0.5 * (lr - min_lr) * (1 + cos(π * t))
          = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + cos(π * 0.09))
          = 1e-5 + 0.5 * 0.00099 * (1 + 0.988)
          ≈ 9.9e-4
```

**特徴:**
- 滑らかな減衰
- コサイン関数で物理的な自然さを実現

---

## 💡 実装のポイント

### 1. `get_lr()` メソッド

```python
def get_lr(self):
    if self.last_epoch < self.warmup_epochs:
        # ウォームアップ段階
        lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
    else:
        # コサイン減衰段階
        lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs)
                / (self.epochs - self.warmup_epochs)
            )
        )
    return [lr]  # リスト形式で返す（param_groups用）
```

### 2. State の保存/復元

```python
def state_dict(self):
    # optimizer を除いた全属性を保存
    return {key: value for key, value in self.__dict__.items() 
            if key != "optimizer"}

def load_state_dict(self, state_dict):
    # 保存された状態を復元
    self.__dict__.update(state_dict)
```

**用途**: チェックポイント保存

---

## 🚀 使用例

### PyTorch での使用

```python
import torch
from torch.optim import Adam
from src.optim.warmup_cos_lr import WarmupCosLR

# モデルと最適化器
model = MyModel()
optimizer = Adam(model.parameters(), lr=1e-3)

# スケジューラー
scheduler = WarmupCosLR(
    optimizer=optimizer,
    min_lr=1e-5,
    lr=1e-3,
    warmup_epochs=3,
    epochs=25
)

# 訓練ループ
for epoch in range(25):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # エポック終了時に学習率を更新
    scheduler.step()
    
    # 現在の学習率を表示
    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

### PyTorch Lightning での使用

```python
# model.py
class PLUTOLightningModule(pl.LightningModule):
    def __init__(self, model, lr, warmup_epochs, epochs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        
        scheduler = {
            "scheduler": WarmupCosLR(
                optimizer=optimizer,
                min_lr=1e-5,
                lr=self.lr,
                warmup_epochs=self.warmup_epochs,
                epochs=self.epochs
            ),
            "interval": "epoch"  # エポック単位で更新
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

### 設定ファイルでの使用

```yaml
# config/training/train_pluto.yaml
warm_up_lr_scheduler: warmup_cos  # スケジューラー指定
lr: 1e-3                          # 最大学習率
warmup_epochs: 3                  # ウォームアップ期間
epochs: 25                        # 総エポック
```

---

## 📊 パラメータ調整ガイド

### 学習がすぐに発散する

```yaml
# 現在の設定
warmup_epochs: 3
lr: 1e-3

# 修正: ウォームアップを長くする
warmup_epochs: 5  # 3 → 5
lr: 5e-4           # 1e-3 → 5e-4 (最大学習率を低く)
```

### 訓練が遅い

```yaml
# 現在の設定
warmup_epochs: 5
lr: 1e-4

# 修正: 学習率を上げる
warmup_epochs: 2  # 5 → 2
lr: 1e-3           # 1e-4 → 1e-3
min_lr: 1e-5      # デフォルト保持
```

### 精度が頭打ち

```yaml
# 現在の設定
min_lr: 1e-5
epochs: 25

# 修正: 訓練期間を延ばす
min_lr: 1e-6      # 最小学習率をさらに低く
epochs: 50        # 25 → 50 (倍延長)
```

---

## 🔄 チェックポイント再開の例

```python
# 訓練を中断して再開する場合

# 【保存】
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
}
torch.save(checkpoint, 'checkpoint.pt')

# 【復元】
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])

# エポック 10 から再開
last_epoch = 10
# scheduler は自動的に last_epoch に基づいて学習率を計算
```

---

## 📚 関連ファイル

- [custom_training/README.md](../custom_training/README.md) - 訓練ループ
- [../metrics/README.md](../metrics/README.md) - 性能評価
