# metrics モジュール解説

## 📋 概要

`metrics` は、PLUTOモデルの **性能を評価するための指標** を実装するモジュールです。

**主な役割：**
- 軌跡予測精度の計算（ADE, FDE など）
- ミス率（MR）の計算
- 訓練/検証時のリアルタイム計測

### 🎯 なぜ評価指標が必要？

```
訓練中: 「モデルが改善しているか？」を判定
検証時: 「一般化性能は良いか？」を判定
テスト: 「実世界で動作するか？」を判定

評価指標なしでは、意思決定ができない
```

---

## 📁 ファイル構成

| ファイル | 役割 |
|---------|------|
| `min_ade.py` | Minimum Average Displacement Error |
| `min_fde.py` | Minimum Final Displacement Error |
| `mr.py` | Miss Rate |
| `prediction_avg_ade.py` | Prediction Average ADE |
| `prediction_avg_fde.py` | Prediction Average FDE |
| `utils.py` | 補助関数（ソート、正規化など） |

---

## 🔑 主要な評価指標

### 1. **minADE（Minimum Average Displacement Error）**

#### 概念
```
複数の予測軌跡の中で、
最も誤差が小さい軌跡と
正解軌跡の平均L2距離
```

#### 計算式
```
minADE = min over k trajectories (
  mean distance from predicted to ground truth
)

例:
  k=1: モデルが1個の軌跡を予測
  k=6: モデルが複数予測し、最良を選択

k が大きいほど性能が良くなる傾向
```

#### 実装（`min_ade.py`）

```python
class minADE(Metric):
    def update(self, outputs, target):
        # 最良の k 個軌跡を取得
        pred, _ = sort_predictions(
            outputs["trajectory"],
            outputs["probability"],
            k=self.k
        )
        
        # 各軌跡の平均距離を計算
        ade = torch.norm(
            pred[..., :2] - target.unsqueeze(1)[..., :2],
            p=2, dim=-1
        ).mean(-1)
        
        # 最小値を取得
        min_ade = ade.min(-1)[0]
        self.sum += min_ade.sum()
        self.count += pred.size(0)
    
    def compute(self):
        return self.sum / self.count  # 平均minADE
```

**パラメータ:**
- `k=6`: 上位6個の軌跡から最良を選択

**評価基準:**
- 0.5～1.0 m: 優秀
- 1.0～2.0 m: 良好
- 2.0 m以上: 要改善

---

### 2. **minFDE（Minimum Final Displacement Error）**

#### 概念
```
複数の予測軌跡の中で、
終点（8秒後）の
最小距離誤差
```

#### 計算式
```
minFDE = min over k trajectories (
  distance at final time step
)

例:
  正解: (50.0, 40.0)
  予測1: (50.5, 40.2) → FDE = 0.54
  予測2: (51.0, 39.0) → FDE = 1.41
  予測3: (49.8, 40.1) → FDE = 0.20 ← 最小
  
  minFDE = 0.20
```

**パラメータ:**
- `k=6`: minADE と同様

**評価基準:**
- 0.5～1.0 m: 優秀
- 1.0～2.0 m: 良好
- 2.0 m以上: 要改善

---

### 3. **MR（Miss Rate）**

#### 概念
```
予測軌跡が正解軌跡から
一定距離以上離れている
確率
```

#### 計算式
```
MR = (誤差 > threshold の予測数) / (全予測数)

例:
  threshold = 2.0 m
  
  予測1: maxFDE = 1.5 m ✓（成功）
  予測2: maxFDE = 2.5 m ✗（失敗）
  予測3: maxFDE = 0.8 m ✓（成功）
  
  MR = 1/3 = 33.3%
```

**パラメータ:**
- `threshold`: デフォルト 2.0 m

**評価基準:**
- MR < 20%: 優秀
- MR < 50%: 良好
- MR > 50%: 要改善

---

### 4. **Prediction Average ADE/FDE**

#### 概念
```
全予測軌跡（加重平均）の
ADE / FDE

各軌跡の確率で加重
```

#### 計算式
```
Prediction Avg ADE = sum(ADE_i * probability_i)
```

**用途:**
- モデルの「平均的な予測精度」を評価
- minADE と異なり、最良軌跡ではなく、平均を見る

---

## 💡 実装のポイント

### 1. TorchMetrics の使用

```python
from torchmetrics import Metric

class minADE(Metric):
    full_state_update: bool = False
    higher_is_better: bool = False  # 小さいほど良い
    
    def __init__(self, k=6, ...):
        super().__init__(...)
        self.add_state("sum", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))
    
    def update(self, outputs, target):
        # バッチごとに更新
        ...
    
    def compute(self):
        # 全バッチの統計を返す
        return self.sum / self.count
```

**メリット:**
- 分散学習に対応
- 複数 GPU での自動集計
- PyTorch Lightning との統合

### 2. Torch.no_grad() による効率化

```python
def update(self, outputs, target):
    with torch.no_grad():  # 勾配計算をスキップ
        # 軽い計算のみ
        pred = sort_predictions(...)
        ade = torch.norm(...)
```

**メリット:**
- メモリ節約
- 高速化

### 3. Sort Predictions の役割

```python
def sort_predictions(trajectory, probability, k):
    # 確率が高い k 個の軌跡を選択
    # 例: k=6 なら、確率上位6個を返す
    return top_k_trajectories, top_k_probabilities
```

---

## 🚀 使用例

### 訓練時の使用

```python
from src.metrics import minADE, minFDE, MR

# 評価指標の定義
metrics = {
    "minADE": minADE(k=6),
    "minFDE": minFDE(k=6),
    "MR": MR(threshold=2.0)
}

# 訓練ループ
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        target = batch["target"]
        
        # 各指標を更新
        for name, metric in metrics.items():
            metric.update(outputs, target)
    
    # エポック末に統計を計算
    for name, metric in metrics.items():
        value = metric.compute()
        print(f"{name}: {value:.3f}")
        metric.reset()  # 次のエポックに向けてリセット
```

### WandB へのログ記録

```python
# PyTorch Lightning が自動でログ
logger.log({
    "minADE": minADE_value,
    "minFDE": minFDE_value,
    "MR": mr_value
})

# WandB ダッシュボードで確認
```

---

## 📊 性能評価表

| 指標 | 優秀 | 良好 | 要改善 |
|------|------|------|-------|
| minADE | < 0.5m | 0.5-1.0m | > 1.0m |
| minFDE | < 0.5m | 0.5-1.0m | > 1.0m |
| MR | < 20% | 20-50% | > 50% |
| Avg ADE | < 1.0m | 1.0-2.0m | > 2.0m |

---

## 🐛 よくあるエラー

### Error: `sort_predictions` の出力形状が異なる
```
原因: モデルの出力形式がメトリクス想定と異なる
解決: outputs["trajectory"] の形状を確認
      期待: (batch_size, k, future_steps, 2)
```

### Warning: メトリクスが正しく計測されない
```
原因: torch.no_grad() 内で勾配が必要な操作
解決: メトリクス計算で勾配が不要か確認
```

---

## 📚 関連ファイル

- [custom_training/README.md](../custom_training/README.md) - 訓練ループ
- [../models/README.md](../models/README.md) - モデル出力形式
