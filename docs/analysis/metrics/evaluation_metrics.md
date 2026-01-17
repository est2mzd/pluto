# 評価メトリクス実装 詳細ガイド

## 📋 概要

PLUTO の軌跡予測性能を評価する **複数のメトリクス** が実装されています。

---

## 🎯 メトリクスの一覧

### 1️⃣ minADE (Minimum Average Displacement Error)

**定義**: 予測軌跡と正解軌跡の **最小平均距離誤差**

$$\text{minADE} = \min_{m=1}^{M} \frac{1}{T} \sum_{t=1}^{T} \sqrt{(\hat{x}_{m,t} - x_t)^2 + (\hat{y}_{m,t} - y_t)^2}$$

- $M$: 予測モード数
- $T$: 予測ステップ数
- $(\hat{x}_{m,t}, \hat{y}_{m,t})$: モード $m$ の時刻 $t$ の予測位置
- $(x_t, y_t)$: 正解位置

**解釈**:
```
低いほど良い
minADE = 0.5 m  → 非常に良い
minADE = 2.0 m  → 良い
minADE = 5.0 m  → 許容範囲
```

### 2️⃣ minFDE (Minimum Final Displacement Error)

**定義**: 予測軌跡と正解軌跡の **最小最終距離誤差**

$$\text{minFDE} = \min_{m=1}^{M} \sqrt{(\hat{x}_{m,T} - x_T)^2 + (\hat{y}_{m,T} - y_T)^2}$$

**解釈**:
```
最終位置だけに着目（初期フェーズの誤差は無視）
minFDE = 1.0 m  → 非常に良い
minFDE = 3.0 m  → 良い
minFDE = 8.0 m  → 許容範囲
```

### 3️⃣ MR (Miss Rate)

**定義**: 誤差閾値を超える予測の割合

$$\text{MR} = \frac{\#(\text{predictions where } \text{minFDE} > \text{threshold})}{\text{total predictions}}$$

**一般的な閾値**:
```
閾値 = 2.0 m（自動運転の安全性基準）
MR = 0.0  → 100% の予測が 2.0 m 以内
MR = 0.1  → 90% の予測が 2.0 m 以内
MR = 0.5  → 50% の予測が 2.0 m 以内
```

---

## 🔧 実装の詳細

### MinADE クラス

```python
class MinADE(Metric):
    """Minimum Average Displacement Error"""
    
    def __init__(self):
        super().__init__()
        # 内部状態
        self.add_state("sum_ade", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))
    
    def update(
        self,
        predictions: torch.Tensor,  # (batch, agents, modes, T, 2)
        targets: torch.Tensor       # (batch, agents, T, 2)
    ) -> None:
        """
        1. 各エージェントの予測軌跡と正解軌跡の距離を計算
        2. モードごとに平均距離を計算
        3. モード間で最小値を取得
        """
        
        batch_size, num_agents, num_modes, T, _ = predictions.shape
        
        # Step 1: 距離行列の計算
        # predictions: (B, A, M, T, 2)
        # targets:     (B, A, T, 2)
        
        # targets を拡張: (B, A, 1, T, 2)
        targets_expanded = targets.unsqueeze(2)
        
        # 距離: (B, A, M, T)
        distances = torch.norm(
            predictions - targets_expanded,
            p=2,
            dim=-1
        )
        
        # Step 2: 平均距離 (B, A, M)
        avg_distances = distances.mean(dim=-1)
        
        # Step 3: モード間で最小値 (B, A)
        min_ade = avg_distances.min(dim=-1)[0]
        
        # Step 4: すべてのエージェントで平均
        ade = min_ade.mean()
        
        self.sum_ade += ade * batch_size
        self.count += batch_size
    
    def compute(self) -> torch.Tensor:
        """蓄積された ADE の平均を返す"""
        return self.sum_ade / self.count
```

### MinFDE クラス

```python
class MinFDE(Metric):
    """Minimum Final Displacement Error"""
    
    def update(
        self,
        predictions: torch.Tensor,  # (B, A, M, T, 2)
        targets: torch.Tensor       # (B, A, T, 2)
    ) -> None:
        # 最終位置のみを抽出
        pred_final = predictions[..., -1, :]  # (B, A, M, 2)
        target_final = targets[..., -1, :]    # (B, A, 2)
        
        # 最終距離: (B, A, M)
        fde = torch.norm(
            pred_final - target_final.unsqueeze(2),
            p=2,
            dim=-1
        )
        
        # モード間で最小値: (B, A)
        min_fde = fde.min(dim=-1)[0]
        
        # 平均
        metric = min_fde.mean()
        
        self.sum_fde += metric * predictions.shape[0]
        self.count += predictions.shape[0]
```

### MR クラス

```python
class MissRate(Metric):
    """Miss Rate - % of predictions exceeding error threshold"""
    
    def __init__(self, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold
        self.add_state("num_miss", default=torch.tensor(0))
        self.add_state("num_total", default=torch.tensor(0))
    
    def update(
        self,
        predictions: torch.Tensor,  # (B, A, M, T, 2)
        targets: torch.Tensor       # (B, A, T, 2)
    ) -> None:
        # MinFDE の計算と同じ
        pred_final = predictions[..., -1, :]
        target_final = targets[..., -1, :]
        
        fde = torch.norm(
            pred_final - target_final.unsqueeze(2),
            p=2,
            dim=-1
        )
        
        min_fde = fde.min(dim=-1)[0]  # (B, A)
        
        # 閾値超過をカウント
        num_miss = (min_fde > self.threshold).sum()
        num_total = min_fde.numel()
        
        self.num_miss += num_miss
        self.num_total += num_total
    
    def compute(self) -> torch.Tensor:
        return self.num_miss.float() / self.num_total.float()
```

---

## 📊 メトリクスの使用例

### 単一バッチでの計算

```python
from src.metrics import MinADE, MinFDE, MissRate

# モデル出力
batch_predictions = torch.randn(
    32,      # batch_size
    64,      # max_agents
    6,       # num_modes (multimodal prediction)
    80,      # future_steps
    2        # (x, y)
)

# 正解
batch_targets = torch.randn(
    32,      # batch_size
    64,      # max_agents
    80,      # future_steps
    2        # (x, y)
)

# メトリクス初期化
min_ade = MinADE()
min_fde = MinFDE()
mr = MissRate(threshold=2.0)

# 更新
min_ade.update(batch_predictions, batch_targets)
min_fde.update(batch_predictions, batch_targets)
mr.update(batch_predictions, batch_targets)

# 計算
print(f"minADE: {min_ade.compute():.3f} m")
print(f"minFDE: {min_fde.compute():.3f} m")
print(f"MR (>2.0m): {mr.compute():.3f}")
```

### 複数バッチでの集計（訓練ループ）

```python
# エポック開始
min_ade = MinADE()
min_fde = MinFDE()

for batch_idx, batch in enumerate(val_loader):
    features, targets = batch
    
    # 推論
    with torch.no_grad():
        outputs = model(features)
    
    predictions = outputs["prediction"]  # (B, A, M, T, 2)
    
    # メトリクス更新
    min_ade.update(predictions, targets)
    min_fde.update(predictions, targets)

# エポック終了 - 集計結果
epoch_ade = min_ade.compute()
epoch_fde = min_fde.compute()

print(f"Epoch ADE: {epoch_ade:.3f}, FDE: {epoch_fde:.3f}")
```

---

## 🔍 メトリクス間の関係

### ADE vs FDE の違い

```
軌跡例:
  予測軌跡: (0, 0) → (1, 0) → (2, 0) → (3, 0)
  正解軌跡: (0, 0) → (1, 0) → (1, 1) → (1, 2)

距離計算:
  時刻0: 0.0 m
  時刻1: 0.0 m
  時刻2: |2-1| + |0-1| = 1.414 m
  時刻3: |3-1| + |0-2| = 2.828 m

ADE = (0 + 0 + 1.414 + 2.828) / 4 = 1.06 m
FDE = 2.828 m  (最終位置のみ)
```

### minADE vs ADE

```
Multi-modal 予測:
  モード1: (0,0) → (1,1) → (2,2)  → 平均距離 = 1.5 m
  モード2: (0,0) → (0,1) → (0,2)  → 平均距離 = 0.5 m
  正解:    (0,0) → (0,1) → (0,2)

ADE (モード平均): (1.5 + 0.5) / 2 = 1.0 m
minADE (最小モード): min(1.5, 0.5) = 0.5 m
```

**多重モード予測では minADE を使用**（最も良いモードで評価）

---

## 📈 メトリクスの分析テクニック

### エージェントカテゴリー別の分析

```python
def analyze_by_category(predictions, targets, categories):
    """エージェント種類別にメトリクスを計算"""
    
    results = {}
    
    for category_id in [0, 1, 2]:  # {0: 車, 1: 歩行者, 2: 自転車}
        mask = categories == category_id
        
        if mask.sum() == 0:
            continue
        
        category_pred = predictions[mask]
        category_target = targets[mask]
        
        metric = MinFDE()
        metric.update(category_pred, category_target)
        
        results[f"category_{category_id}"] = metric.compute()
    
    return results
```

### 時間ウィンドウ別の分析

```python
def analyze_by_time_window(predictions, targets, window_size=20):
    """予測時間を複数ウィンドウに分割して評価"""
    
    T = predictions.shape[-2]
    num_windows = T // window_size
    
    results = {}
    
    for w in range(num_windows):
        start = w * window_size
        end = (w + 1) * window_size
        
        window_pred = predictions[..., start:end, :]
        window_target = targets[..., start:end, :]
        
        metric = MinADE()
        metric.update(window_pred, window_target)
        
        results[f"window_{w}"] = metric.compute()
    
    return results
```

---

## 🔗 評価パイプライン全体

```python
from src.metrics import MinADE, MinFDE, MissRate
from torch.utils.data import DataLoader

class EvaluationPipeline:
    def __init__(self, model, device="cuda:0"):
        self.model = model.to(device)
        self.device = device
        
        # メトリクス
        self.min_ade = MinADE()
        self.min_fde = MinFDE()
        self.mr = MissRate(threshold=2.0)
    
    def evaluate(self, val_loader):
        self.model.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                
                # 推論
                outputs = self.model(features)
                predictions = outputs["prediction"]
                
                # メトリクス更新
                self.min_ade.update(predictions, targets)
                self.min_fde.update(predictions, targets)
                self.mr.update(predictions, targets)
        
        # 結果
        return {
            "minADE": self.min_ade.compute().item(),
            "minFDE": self.min_fde.compute().item(),
            "MR": self.mr.compute().item(),
        }

# 使用
evaluator = EvaluationPipeline(model)
metrics = evaluator.evaluate(val_loader)
print(metrics)
# {'minADE': 0.523, 'minFDE': 1.245, 'MR': 0.082}
```

---

## 📊 メトリクスの解釈チャート

| メトリクス | 優秀 | 良好 | 平均 | 要改善 |
|-----------|------|------|------|--------|
| **minADE** | < 0.5 m | 0.5-1.0 | 1.0-2.0 | > 2.0 |
| **minFDE** | < 1.0 m | 1.0-2.0 | 2.0-4.0 | > 4.0 |
| **MR** (2m) | < 5% | 5-10% | 10-20% | > 20% |

---

## 📚 関連ファイル

- [../planners/pluto_planner.md](../planners/pluto_planner.md) - 推論エンジン
- [../post_processing/trajectory_evaluation.md](../post_processing/trajectory_evaluation.md) - 軌跡検証
