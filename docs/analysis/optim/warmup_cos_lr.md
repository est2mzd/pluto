# WarmupCosLR 学習率スケジューラ 詳細ガイド

## 📋 概要

`WarmupCosLR` は、**ウォームアップ段階 + 余弦アニーリング段階** の2段階から構成される学習率スケジューラです。

---

## 🎯 学習率スケジュールの概念

### ステップ1: ウォームアップ段階 (0～steps)

```
学習率スケジュール (ウォームアップ)

LR
  │
  │  目標LR
  ├─────────────────────┐
  │                     │ ← 目標に到達
  │        /│           │
  │       / │           │
  │      /  │ ウォーム   │
  │     /   │  アップ   │
  │    /    │           │
  │   /     │           │
初期LR ├─────────────────────┴─────ケ─── 時刻
  0 steps  warmup_steps    predict_steps

目的: モデルが安定的に学習を始められるように、
     徐々に学習率を上げていく
```

**実装**:
```python
# 線形ウォームアップ
lr_t = initial_lr + (target_lr - initial_lr) * (t / warmup_steps)

# 例:
# initial_lr = 1e-5
# target_lr = 1e-3
# warmup_steps = 1000

# t=0:    lr = 1e-5
# t=500:  lr = 5e-4
# t=1000: lr = 1e-3 (到達)
```

### ステップ2: 余弦アニーリング段階 (warmup_steps～total_steps)

```
学習率スケジュール (余弦アニーリング)

LR
  │
  │  目標LR ┌─────────────┐
  │         │\            
  │         │ \   余弦    
  │         │  \  曲線    
  │         │   \        
最小LR ├─────────────┴────────────────→ 時刻
      warmup_steps   total_steps

目的: 学習率を徐々に下げることで、
     モデルを局所最適解に収束させる
```

**実装**:
```python
# 余弦アニーリング
progress = (t - warmup_steps) / (total_steps - warmup_steps)
lr_t = min_lr + (target_lr - min_lr) * (1 + cos(π * progress)) / 2

# 例:
# progress=0.0: cos(0) = 1  → lr = target_lr (最大)
# progress=0.5: cos(π/2) = 0 → lr = (target_lr + min_lr) / 2 (中間)
# progress=1.0: cos(π) = -1 → lr = min_lr (最小)
```

---

## 🔧 実装コード

### クラス定義

```python
class WarmupCosLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        learning_rate: float = 1e-3,
        min_lr: float = 0.0,
        warmup_factor: float = 0.1,
    ):
        """
        Args:
            optimizer: PyTorch オプティマイザー
            warmup_steps: ウォームアップ期間 [ステップ数]
            total_steps: 総訓練ステップ数
            learning_rate: 目標学習率（ウォームアップ後）
            min_lr: 最小学習率（アニーリング最小値）
            warmup_factor: ウォームアップ初期学習率の倍数
                          初期LR = learning_rate * warmup_factor
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_factor = warmup_factor
        
        # 初期学習率の計算
        self.initial_lr = learning_rate * warmup_factor
        
        # 現在のステップ
        self.current_step = 0
        
        super().__init__(optimizer)
    
    def step(self) -> None:
        """各ステップで学習率を更新"""
        
        if self.current_step < self.warmup_steps:
            # ========== ウォームアップ段階 ==========
            progress = self.current_step / self.warmup_steps
            lr = self.initial_lr + (self.learning_rate - self.initial_lr) * progress
            
        else:
            # ========== 余弦アニーリング段階 ==========
            annealing_steps = self.current_step - self.warmup_steps
            annealing_total = self.total_steps - self.warmup_steps
            
            progress = annealing_steps / annealing_total
            
            # 余弦関数: cos(π * progress)
            # progress=0: cos(0) = 1
            # progress=1: cos(π) = -1
            cos_factor = (1 + math.cos(math.pi * progress)) / 2
            
            lr = self.min_lr + (self.learning_rate - self.min_lr) * cos_factor
        
        # オプティマイザーの学習率を更新
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        self.current_step += 1
```

---

## 📊 パラメータの効果

### warmup_factor の選択

```python
# warmup_factor = 0.1 (推奨)
initial_lr = 1e-3 * 0.1 = 1e-4

# 訓練開始時:
# 学習率が低いので、大きな更新による発散を防ぐ
# → 安定した学習の開始

# warmup_factor = 1.0 (非推奨)
initial_lr = 1e-3

# 最初から目標学習率で開始
# → 不安定な学習、loss が発散する可能性

# warmup_factor = 0.01 (控え目)
initial_lr = 1e-5

# 非常に徐々に学習率を上げる
# → 収束は遅い、安定性は高い
```

### min_lr の選択

```python
# min_lr = 0.0 (推奨)
# 最終的に学習率を0まで低下させ、完全に収束

# min_lr = 1e-6 (推奨)
# 最終的な微調整のための最小学習率を維持
# → わずかな更新で過学習を防ぐ

# min_lr = 1e-4 (非推奨)
# 学習率の低下幅が小さく、収束が遅い
```

---

## 🚀 使用例

### 基本的な訓練ループ

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from src.optim.warmup_cos_lr import WarmupCosLR

# モデル定義
model = PlutoModel(...)

# オプティマイザー
optimizer = Adam(model.parameters(), lr=1e-3)

# スケジューラー
total_epochs = 50
epoch_steps = len(train_loader)
total_steps = total_epochs * epoch_steps

scheduler = WarmupCosLR(
    optimizer=optimizer,
    warmup_steps=1000,         # 最初の1000ステップでウォームアップ
    total_steps=total_steps,   # 全50エポック
    learning_rate=1e-3,        # 目標学習率
    min_lr=1e-6,              # 最小学習率
    warmup_factor=0.1         # 初期LR = 1e-3 * 0.1 = 1e-4
)

# 訓練ループ
for epoch in range(total_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # Forward pass
        outputs = model(batch)
        loss = compute_loss(outputs, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 学習率更新
        scheduler.step()
        
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Batch {batch_idx}, LR: {current_lr:.2e}, Loss: {loss:.4f}")
```

### Hydra 設定での使用

```yaml
# config/training/train_pluto.yaml

scheduler:
  _target_: src.optim.warmup_cos_lr.WarmupCosLR
  
  # ウォームアップ設定
  warmup_steps: 1000         # 約15分（バッチサイズ64, GPU1枚）
  warmup_factor: 0.1         # 初期LR = LR * 0.1
  
  # アニーリング設定
  learning_rate: 1e-3        # 目標学習率
  min_lr: 1e-6              # 最小学習率
  total_steps: 500000        # 50エポック分
```

Python での読み込み:

```python
from hydra.utils import instantiate
import yaml

with open("config/training/train_pluto.yaml") as f:
    config = yaml.safe_load(f)

scheduler = instantiate(
    config["scheduler"],
    optimizer=optimizer
)
```

---

## 📈 学習率の可視化

### スケジュール表示コード

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_schedule(warmup_steps, total_steps, learning_rate, min_lr):
    lrs = []
    steps = range(total_steps)
    
    for step in steps:
        if step < warmup_steps:
            # ウォームアップ
            progress = step / warmup_steps
            lr = learning_rate * 0.1 + (learning_rate - learning_rate * 0.1) * progress
        else:
            # アニーリング
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cos_factor = (1 + np.cos(np.pi * progress)) / 2
            lr = min_lr + (learning_rate - min_lr) * cos_factor
        
        lrs.append(lr)
    
    plt.figure(figsize=(12, 4))
    plt.plot(lrs)
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("WarmupCosLR Schedule")
    plt.grid(True, alpha=0.3)
    
    # ウォームアップ期間を明示
    plt.axvline(warmup_steps, color='r', linestyle='--', alpha=0.5, label='Warmup End')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("lr_schedule.png", dpi=150)
    plt.show()

# 実行例
visualize_schedule(
    warmup_steps=1000,
    total_steps=100000,
    learning_rate=1e-3,
    min_lr=1e-6
)
```

---

## 🔍 デバッグ・モニタリング

### 学習率のログ記録

```python
from tensorboard.compat.tensorflow_stub import io as tb_io
import torch

class LRLogger:
    def __init__(self, writer=None):
        self.writer = writer
        self.step = 0
    
    def log(self, optimizer, loss):
        current_lr = optimizer.param_groups[0]['lr']
        
        if self.writer:
            self.writer.add_scalar("lr", current_lr, self.step)
            self.writer.add_scalar("loss", loss, self.step)
        
        self.step += 1

# 使用
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/experiment_1")
lr_logger = LRLogger(writer)

# 訓練ループ内
for batch_idx, batch in enumerate(train_loader):
    loss = train_one_batch(batch)
    optimizer.step()
    scheduler.step()
    
    lr_logger.log(optimizer, loss)

writer.close()
```

### スケジューラー状態の検証

```python
def validate_scheduler(scheduler, total_steps=1000):
    """スケジューラーの動作を検証"""
    
    lrs = []
    for _ in range(total_steps):
        lrs.append(scheduler.optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    print(f"最小学習率: {min(lrs):.2e}")
    print(f"最大学習率: {max(lrs):.2e}")
    print(f"初期学習率: {lrs[0]:.2e}")
    print(f"最終学習率: {lrs[-1]:.2e}")
    
    # ウォームアップ段階の確認
    warmup_lrs = lrs[:1000]
    print(f"ウォームアップが上昇傾向か: {warmup_lrs[-1] > warmup_lrs[0]}")
    
    # アニーリング段階の確認
    anneal_lrs = lrs[1000:]
    print(f"アニーリングが下降傾向か: {anneal_lrs[-1] < anneal_lrs[0]}")

validate_scheduler(scheduler)
```

---

## 📚 関連ファイル

- [../custom_training/custom_training_builder.md](../custom_training/custom_training_builder.md) - 訓練パイプライン
- [../models/pluto_model.md](../models/pluto_model.md) - モデル実装
