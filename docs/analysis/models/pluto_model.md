# PlutoModel ニューラルネットワークアーキテクチャ 詳細ガイド

## 📋 概要

`PlutoModel` は、**マルチモーダル軌跡予測** を行うエンドツーエンドニューラルネットワークです。

---

## 🏗️ アーキテクチャ全体

### 構成図

```
【入力層】
    ├─ Ego 状態: [x, y, yaw, vx, vy, ax, ay, ...]
    ├─ 周辺エージェント: (64, 101, 10) ← 位置、速度、形状など
    ├─ マップ: ポリゴン、セマンティックレイヤー
    └─ コスト地図: occupancy grid (500, 500)

            ↓ 正規化・特徴抽出

【エンコーダー】
    ├─ Ego Encoder
    │   ├─ MLP: 10 → 128 → 256
    │   └─ 出力: (1, 256)
    │
    ├─ Agent Encoder (64並列)
    │   ├─ LSTM: 軌跡時系列処理 (101, 10) → 256
    │   ├─ Self-Attention: エージェント間の相互作用
    │   └─ 出力: (64, 256)
    │
    ├─ Map Encoder
    │   ├─ GNN: ポリゴングラフの処理
    │   ├─ Graph Attention: マップ要素の相互作用
    │   └─ 出力: (num_nodes, 256)
    │
    └─ Fusion
        ├─ Ego + Agent + Map の特徴量統合
        ├─ Cross-Attention: モダリティ間の相互作用
        └─ 出力: (1, 512)

            ↓ Context Vector

【デコーダー】
    ├─ Trajectory Generation Mode 1
    │   ├─ MLP Decoder: 512 → 256 → 128
    │   ├─ 軌跡生成: (80, 3) = (x, y, yaw)
    │   └─ 出力: (80, 3)
    │
    ├─ Trajectory Generation Mode 2
    │   └─ 同上
    │
    ├─ ... Mode K
    │   └─ 同上
    │
    └─ Confidence Head
        ├─ MLP: 512 → 256 → K
        └─ Softmax: (K,) ← モード確率

【出力層】
    ├─ predictions: (num_agents, K, 80, 3)
    ├─ confidence: (num_agents, K)
    └─ auxiliary: {活性化マップなど}
```

---

## 🔧 各モジュールの詳細実装

### Ego Encoder

```python
class EgoEncoder(nn.Module):
    def __init__(self, state_dim: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        # 出力: (hidden_dim,)
    
    def forward(self, ego_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ego_state: (batch, 10) ← [x, y, yaw, vx, vy, ax, ay, steer, steer_rate, ...]
        
        Returns:
            ego_embedding: (batch, hidden_dim)
        """
        return self.mlp(ego_state)
```

### Agent Encoder

```python
class AgentEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # 時系列処理
        self.lstm = nn.LSTM(
            input_size=10,      # [x, y, vx, vy, yaw, ...] × 2 (相対・絶対)
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        # 自己注意機構（異なる時刻での関連性を学習）
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
    
    def forward(
        self,
        agent_trajectories: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            agent_trajectories: (batch, num_agents, time=101, 10)
            valid_mask: (batch, num_agents) ← エージェントの有効性
        
        Returns:
            agent_embeddings: (batch, num_agents, hidden_dim)
        """
        batch, num_agents, T, _ = agent_trajectories.shape
        
        # LSTM による時系列符号化
        lstm_out, (h_n, c_n) = self.lstm(agent_trajectories)
        # lstm_out: (batch, num_agents, T, 256)
        # 最終隠れ状態を使用: (batch, num_agents, 256)
        
        # Self-Attention で重要な時刻に焦点
        attention_out, _ = self.temporal_attention(
            lstm_out,      # Query
            lstm_out,      # Key
            lstm_out       # Value
        )
        
        # 平均プーリング
        agent_embeddings = attention_out.mean(dim=2)  # (batch, num_agents, 256)
        
        # 無効なエージェント（パディング）をゼロにする
        agent_embeddings = agent_embeddings * valid_mask.unsqueeze(-1)
        
        return agent_embeddings
```

### Map Encoder

```python
class MapEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # グラフニューラルネットワーク（ポリゴングラフ）
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=8)
            for _ in range(3)
        ])
    
    def forward(
        self,
        map_polygons: torch.Tensor,      # (batch, num_polygons, 8, 2)
        polygon_types: torch.Tensor      # (batch, num_polygons) ← タイプID
    ) -> torch.Tensor:
        """
        Args:
            map_polygons: 各ポリゴンの頂点座標
            polygon_types: レーン、停止線など のタイプ

        Returns:
            map_embeddings: (batch, num_polygons, hidden_dim)
        """
        
        # ポリゴン毎の埋め込み
        # 各ポリゴンの頂点を平均化
        polygon_embeddings = map_polygons.mean(dim=2)  # (batch, num_polygons, 2)
        
        # タイプ埋め込みと連結
        type_embeddings = self.type_embedding(polygon_types)
        embeddings = torch.cat([polygon_embeddings, type_embeddings], dim=-1)
        
        # GNN層を通す
        for gnn_layer in self.gnn_layers:
            embeddings = gnn_layer(embeddings)
        
        return embeddings  # (batch, num_polygons, hidden_dim)
```

### Multimodal Fusion

```python
class MultimodalFusion(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # 各モダリティから共通潜在空間へ
        self.ego_projection = nn.Linear(hidden_dim, hidden_dim)
        self.agent_projection = nn.Linear(hidden_dim, hidden_dim)
        self.map_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-Attention: モダリティ間の相互作用
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(
        self,
        ego_embedding: torch.Tensor,          # (batch, 256)
        agent_embeddings: torch.Tensor,       # (batch, 64, 256)
        map_embeddings: torch.Tensor          # (batch, num_polygons, 256)
    ) -> torch.Tensor:
        """複数モダリティの特徴量を統合"""
        
        # 投影
        ego_proj = self.ego_projection(ego_embedding)  # (batch, 256)
        agent_proj = self.agent_projection(agent_embeddings)  # (batch, 64, 256)
        map_proj = self.map_projection(map_embeddings)  # (batch, num_polygons, 256)
        
        # 全要素を統合
        # Ego を中心に、Agent と Map を Query として Cross-Attention
        fused = self.cross_attention(
            ego_proj.unsqueeze(1),      # Query: (batch, 1, 256)
            torch.cat([agent_proj, map_proj], dim=1),  # Key, Value
            torch.cat([agent_proj, map_proj], dim=1)
        )
        
        return fused[0].squeeze(1)  # (batch, 256)
```

### Trajectory Decoder

```python
class TrajectoryDecoder(nn.Module):
    def __init__(self, num_modes: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        
        # モード別デコーダー（重みは共有）
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 80 * 3)  # 80ステップ × 3 (x, y, yaw)
        )
    
    def forward(
        self,
        context: torch.Tensor,
        mode_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context: (batch, hidden_dim) ← エンコーダー出力
            mode_embeddings: (num_modes, hidden_dim) ← モード固有の埋め込み
        
        Returns:
            trajectories: (batch, num_modes, 80, 3)
        """
        
        batch_size = context.shape[0]
        trajectories = []
        
        # 各モードの軌跡を生成
        for mode_idx in range(self.num_modes):
            # Context とモード埋め込みを結合
            input_vec = context + mode_embeddings[mode_idx]
            
            # デコード
            traj = self.decoder(input_vec)  # (batch, 240)
            traj = traj.view(batch_size, 80, 3)  # (batch, 80, 3)
            
            trajectories.append(traj)
        
        return torch.stack(trajectories, dim=1)  # (batch, num_modes, 80, 3)
```

---

## 📊 モデルパラメータ数の詳細

### 層ごとのパラメータ数

```
Ego Encoder:
  Linear(10, 128):      1.3 K
  Linear(128, 256):    32.8 K
  Linear(256, 256):    65.5 K
  小計:               ~100 K

Agent Encoder:
  LSTM(10 → 256, 2層): 300 K
  MultiheadAttention:   200 K
  小計:               ~500 K

Map Encoder:
  GNN Layer × 3:       ~600 K

Fusion Module:
  Projections:         ~200 K
  Cross-Attention:     ~300 K

Decoder:
  Trajectory Decoder:  ~400 K
  Confidence Head:     ~50 K

全体:                ~2.2 M パラメータ
```

### GPU メモリ消費量

```
バッチサイズ = 32

Forward Pass:
  Activation 保存:      ~800 MB
  
Backward Pass:
  勾配計算:            ~400 MB
  Optimizer State:      ~100 MB

合計:               ~1.3 GB / GPU
```

---

## 🚀 使用例

### モデルの構築と推論

```python
from src.models.pluto.pluto_model import PlutoModel
import torch

# モデル作成
model = PlutoModel(
    num_modes=6,           # 予測モード数
    hidden_dim=256,
    num_agents=64,
    future_steps=80
)

# GPU に転送
model = model.to("cuda:0")

# 特徴量準備
feature = builder(scenario, iteration=0)
feature = feature.to_tensor(device="cuda:0")

# 推論
model.eval()
with torch.no_grad():
    outputs = model(feature)

# 出力確認
predictions = outputs["prediction"]       # (1, 64, 6, 80, 3)
confidence = outputs["confidence"]        # (1, 64, 6)

print(f"Predictions shape: {predictions.shape}")
print(f"Confidence shape: {confidence.shape}")

# Ego の予測軌跡（第0要素）
ego_predictions = predictions[0, 0, :, :, :]  # (6, 80, 3)
ego_confidence = confidence[0, 0, :]          # (6,)

for mode in range(6):
    print(f"Mode {mode}: confidence={ego_confidence[mode]:.3f}")
    print(f"  Final position: {ego_predictions[mode, -1, :2]}")
```

### 訓練

```python
from torch.optim import Adam
from src.optim.warmup_cos_lr import WarmupCosLR

# オプティマイザーとスケジューラー
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = WarmupCosLR(
    optimizer=optimizer,
    warmup_steps=1000,
    total_steps=100000,
    learning_rate=1e-3
)

# 訓練ループ
for epoch in range(50):
    for batch_idx, batch in enumerate(train_loader):
        # Forward
        outputs = model(batch)
        predictions = outputs["prediction"]
        confidence = outputs["confidence"]
        
        # Loss 計算
        loss = compute_loss(
            predictions,
            confidence,
            batch["target"],
            batch["valid_mask"]
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 100 == 0:
            print(f"Loss: {loss.item():.4f}")
```

---

## 🔍 デバッグ・可視化

### モデル出力の確認

```python
def validate_model_output(outputs):
    """モデル出力の妥当性検証"""
    
    predictions = outputs["prediction"]
    confidence = outputs["confidence"]
    
    # 形状チェック
    assert predictions.shape[-1] == 3, "軌跡は3次元（x, y, yaw）"
    assert predictions.shape[-2] == 80, "予測ステップは80"
    assert predictions.shape[-3] == 6, "モード数は6"
    
    # 確率チェック
    assert confidence.min() >= 0, "確率が負"
    assert confidence.max() <= 1, "確率が1超過"
    assert torch.allclose(confidence.sum(dim=-1), torch.ones(1)), "確率合計が1でない"
    
    print("✓ モデル出力が妥当")

outputs = model(feature)
validate_model_output(outputs)
```

### 予測軌跡の可視化

```python
import matplotlib.pyplot as plt

def visualize_model_predictions(predictions, confidence, colors=['r', 'g', 'b', 'orange', 'purple', 'brown']):
    """6つのモード予測を可視化"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 軌跡プロット
    for mode in range(6):
        traj = predictions[0, 0, mode, :, :2].cpu().numpy()
        conf = confidence[0, 0, mode].cpu().item()
        
        ax1.plot(traj[:, 0], traj[:, 1], 
                color=colors[mode], alpha=0.7,
                label=f"Mode {mode} (p={conf:.3f})")
    
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title("Multimodal Trajectory Predictions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")
    
    # 確率分布プロット
    confs = confidence[0, 0].cpu().numpy()
    ax2.bar(range(6), confs, color=colors)
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Probability")
    ax2.set_title("Mode Confidence Distribution")
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig("model_predictions.png")
    plt.show()

visualize_model_predictions(predictions, confidence)
```

---

## 📚 関連ファイル

- [../custom_training/custom_training_builder.md](../custom_training/custom_training_builder.md) - 訓練パイプライン
- [../metrics/evaluation_metrics.md](../metrics/evaluation_metrics.md) - 評価
