# contrastive_scenario_generator.py 詳細解説

## 📝 ファイル概要

このファイルは、対比学習用のデータ拡張エンジンを実装しています。

**主な責務：**
- Ego 車の状態に小さなノイズを加える（ポジティブサンプル）
- シナリオの要素を修正/削除/追加する（ネガティブサンプル）
- 物理的に妥当なシナリオのみを生成

---

## 🔧 主要クラス・関数

### 1. `ContrastiveScenarioGenerator` クラス

#### 役割
PyTorch Lightning の `AbstractAugmentor` を拡張し、データ拡張を実装。

#### クラス継承図
```
AbstractAugmentor（nuplan-devkit）
    ↑
    │ 継承
    │
ContrastiveScenarioGenerator
```

#### `__init__()` - 初期化

##### シグネチャ
```python
def __init__(
    self,
    history_steps=21,              # 過去何ステップを使用するか
    max_interaction_horizon=40,    # インタラクティブ判定の最大距離
    low: List[float] = [0.0, -1.5, -0.35, -1, -0.5, -0.2, -0.2],   # ノイズ下限
    high: List[float] = [2.0, 1.5, 0.35, 1, 0.5, 0.2, 0.2],        # ノイズ上限
    use_negative_sample: bool = True  # ネガティブサンプル生成するか
) -> None:
```

##### パラメータ詳細

| パラメータ | 意味 | 例 |
|-----------|------|-----|
| `history_steps` | 訓練に使用する過去ステップ数 | `21` = 2.1秒（dt=0.1s） |
| `max_interaction_horizon` | インタラクティブと判定する最大ステップ数 | `40` = 4.0秒 |
| `low` | ノイズの下限値リスト | `[Δx, Δy, Δyaw, ...]` |
| `high` | ノイズの上限値リスト | `[Δx, Δy, Δyaw, ...]` |
| `use_negative_sample` | ネガティブサンプル生成の有無 | `true` / `false` |

##### `low` と `high` の詳細
```python
# ノイズの種類（要素数: 7）
low =  [0.0,  -1.5, -0.35, -1,   -0.5,  -0.2,  -0.2]
high = [2.0,   1.5,  0.35,  1,    0.5,   0.2,   0.2]
        Δx    Δy    Δyaw  Δvx  Δacc Δster Δster_rate

# 例
Δx ∈ [0.0, 2.0]      # 前後方向: 0～2.0m
Δy ∈ [-1.5, 1.5]     # 左右方向: -1.5～1.5m
Δyaw ∈ [-0.35, 0.35] # 回転: -0.35～0.35 rad (-20～20度)
```

##### 初期化の内部処理
```python
# ランダムノイズジェネレータ
self._random_offset_generator = UniformNoise(low, high)

# 衝突判定エンジン
self._collision_checker = CollisionChecker()

# Pacifica パラメータ（ベース車両）
self._rear_to_cog = get_pacifica_parameters().rear_axle_to_center
# リア軸からコクピット中心までの距離
```

---

### 2. `augment()` メソッド（エントリーポイント）

#### 役割
PyTorch Lightning が各バッチで呼び出すメイン拡張メソッド。

#### シグネチャ
```python
def augment(
    self,
    features: FeaturesType,  # 入力特徴量（辞書）
    targets: TargetsType = None,  # 出力ターゲット（軌跡など）
    scenario: Optional[AbstractScenario] = None  # シナリオメタデータ
) -> Tuple[FeaturesType, TargetsType]:
```

#### 処理フロー

```python
# 【ステップ1】特徴量から PlutoFeature を抽出
feature: PlutoFeature = features["feature"]

# 【ステップ2】ポジティブサンプルを生成
feature.data_p = self.generate_positive_sample(feature.data)
      ↓
# コスト地図を中央からクロップ（500x500）
if "cost_maps" in feature.data:
    feature.data["cost_maps"] = crop_img_from_center(...)

# 【ステップ3】ネガティブサンプルを生成（オプション）
if self.use_negative_sample:
    feature.data_n, feature.data_n_info = self.generate_negative_sample(feature.data)
    # data_n: ネガティブサンプルデータ
    # data_n_info: メタデータ
    #   - valid_mask: 有効か
    #   - type: ネガティブの種類（MAP_CONTRAST_TYPE=0 or AGENT_CONTRAST_TYPE=1）

# 【ステップ4】特徴量を更新して返す
features["feature"] = feature
return features, targets
```

#### `PlutoFeature` の構造
```
PlutoFeature
├─ data: Dict[str, Any]       # 元のシナリオデータ
├─ data_p: Dict[str, Any]     # ポジティブサンプル
├─ data_n: Dict[str, Any]     # ネガティブサンプル
├─ data_n_info: Dict[str, Any]  # ネガティブのメタ情報
└─ 他のフィールド...
```

---

### 3. `generate_positive_sample()` メソッド

#### 役割
Ego 車に小さなノイズを加えて、似たシナリオを生成。

#### シグネチャ
```python
def generate_positive_sample(self, data: Dict) -> Dict:
```

#### 処理フロー（詳細版）

```python
【ステップ1】元のデータをコピー
new_data = deepcopy(data)

【ステップ2】現在の Ego 状態を取得
current_state = data["current_state"]
# [x, y, yaw, vel, acc, steer, steer_rate, angular_vel, angular_acc]

【ステップ3】周りのエージェント情報を取得
agents_position = data["agent"]["position"][1:11, history_steps-1]
# 他のエージェント 1～10 の現在位置
agents_heading = data["agent"]["heading"][1:11, history_steps-1]
agents_shape = data["agent"]["shape"][1:11, history_steps-1]

【ステップ4】ノイズを生成
noise = self._random_offset_generator.sample()
# [Δx, Δy, Δyaw, ...]

【ステップ5】衝突チェック付きでノイズを適用
num_tries, scale = 0, 1.0
while num_tries < 5:
    new_noise = noise * scale
    new_state = current_state + new_noise
    new_state[3] = max(0.0, new_state[3])  # 速度は非負

    if self.safety_check(...):  # 衝突なし？
        break

    num_tries += 1
    scale *= 0.5  # ノイズを半減

【ステップ6】新しい Ego 状態を保存
new_data["current_state"] = new_state
new_data["agent"]["position"][0, history_steps-1] = new_state[:2]
new_data["agent"]["heading"][0, history_steps-1] = new_state[2]

【ステップ7】コスト地図を回転・シフト
if "cost_maps" in data:
    new_data["cost_maps"] = crop_img_from_center(
        shift_and_rotate_img(
            img=...,
            shift=[new_noise[1], -new_noise[0], 0],  # (Δy, -Δx, 0)
            angle=-new_noise[2],  # -Δyaw
            resolution=0.2,
            cval=-200
        ),
        (500, 500)
    )

【ステップ8】無関係なエージェントをドロップ
non_interacting_agent_mask = data["causal"]["interaction_label"] <= 0

if non_interacting_agent_mask.sum() > 1 and random < 0.5:
    # 50% の確率で無関係エージェントをドロップ
    non_interacting_agent_mask[0] = False  # Ego を除外
    non_interacting_agent_mask[leading_agent] = False  # リード車を除外
    
    drop_portion = random(0.1, 1.0)  # 10%～100% をドロップ
    drop_mask = random(0, 1, N) <= drop_portion
    
    for key, value in new_data["agent"].items():
        new_data["agent"][key] = value[~drop_mask]  # ドロップ対象を削除

【ステップ9】正規化して返す
new_data = PlutoFeature.normalize(new_data).data
return new_data
```

#### 重要ポイント

**衝突チェックの段階的緩和:**
```
試行 1: noise * 1.0  (元のノイズ)
試行 2: noise * 0.5  (半分)
試行 3: noise * 0.25 (1/4)
試行 4: noise * 0.125 (1/8)
試行 5: noise * 0.0625 (1/16)

5回試しても衝突 → スキップ（このサンプルは生成しない）
```

**無関係エージェント削除:**
```
「無関係」 = interaction_label <= 0
(インタラクティブでないエージェント)

50% の確率で、これらをドロップすることで、
モデルに「重要なエージェント」を学ばせる
```

---

### 4. `generate_negative_sample()` メソッド

#### 役割
シナリオの要素を修正/削除/追加して、対照的なシナリオを生成。

#### シグネチャ
```python
def generate_negative_sample(
    self, data: Dict
) -> Tuple[Dict, Dict]:  # (ネガティブデータ, メタ情報)
```

#### 処理フロー

```python
【ステップ1】利用可能な生成方法をリストアップ
available_generators = []

【ステップ2】赤信号で停止中なら、信号反転を候補に追加
if not data["causal"]["is_waiting_for_red_light_without_lead"]:
    if leading_agent or interacting_agent:
        available_generators.append(self.neg_interacting_agent_dropout)
else:
    available_generators.append(self.neg_traffic_light_inversion)

【ステップ3】自由な経路があれば、エージェント挿入を候補に追加
if len(data["causal"]["free_path_points"]) > 0 and agent_num > 1:
    available_generators.append(self.neg_leading_agent_insertion)

【ステップ4】候補から1つをランダムに選択して実行
if len(available_generators) > 0:
    generator = np.random.choice(available_generators)
    data_n, contrast_type = generator(data)  # 実行
    data_n_valid_mask = True
else:
    # 候補がない場合は元データを返す
    data_n = data
    contrast_type = 0
    data_n_valid_mask = False

【ステップ5】メタ情報を返す
return data_n, {
    "valid_mask": data_n_valid_mask,   # 有効か
    "type": contrast_type              # 0=MAP, 1=AGENT
}
```

---

### 5. ネガティブ生成メソッド（詳細）

#### `neg_traffic_light_inversion()` - 信号反転

```python
def neg_traffic_light_inversion(self, data):
    """
    赤信号を GREEN に反転（矛盾シナリオ）
    """
    new_data = deepcopy(data)
    
    # Ego が気にしている赤信号を取得
    ego_care_red_light_mask = data["causal"]["ego_care_red_light_mask"]
    
    # 新しい信号をランダムに選択（GREEN or UNKNOWN）
    choices = [TrafficLightStatusType.GREEN, TrafficLightStatusType.UNKNOWN]
    new_status = np.random.choice(choices, size=ego_care_red_light_mask.sum())
    
    # 信号を更新
    new_data["map"]["polygon_tl_status"][ego_care_red_light_mask] = new_status
    
    return new_data, MAP_CONTRAST_TYPE  # MAP修正型
```

**例:**
```
【元】
  Ego: 停止中 (v=0)
  信号: RED
  行動: 停止 ✓ 合理的

【ネガティブ】
  Ego: 停止中 (v=0)
  信号: GREEN ← 反転
  行動: 停止 ✗ 矛盾！

モデル:「信号を無視する Ego？」
```

#### `neg_interacting_agent_dropout()` - エージェント削除

```python
def neg_interacting_agent_dropout(self, data):
    """
    インタラクティブなエージェント（脅威）を削除
    """
    new_data = deepcopy(data)
    
    # ドロップ対象: リード車 or インタラクティブ車
    dropout_mask = (
        data["causal"]["leading_agent_mask"]
        | data["causal"]["interacting_agent_mask"]
    )
    
    # 該当するエージェントを削除
    for key, value in new_data["agent"].items():
        new_data["agent"][key] = value[~dropout_mask]
    
    return new_data, AGENT_CONTRAST_TYPE  # AGENT修正型
```

**例:**
```
【元】
  Ego: 加速（v=2m/s）
  他車: ブレーキ中（v=3→0）
  行動: Ego も減速

【ネガティブ】
  Ego: 加速（v=2m/s）
  他車: 削除 ← ブレーキ中の車がない
  行動: Ego も減速 ✗ なぜ？

モデル:「見えない脅威を検出できるか？」
```

#### `neg_leading_agent_insertion()` - エージェント挿入

```python
def neg_leading_agent_insertion(self, data):
    """
    自由な経路上に新しいエージェントを挿入
    """
    new_data = deepcopy(data)
    
    # 【ステップA】自由な経路上にランダムに点を選択
    path_point = data["causal"]["free_path_points"][
        np.random.choice(len(data["causal"]["free_path_points"]))
    ]
    
    # 【ステップB】既存エージェントから、速度が最も似ているものを選ぶ
    agents_velocity = np.linalg.norm(
        data["agent"]["velocity"][:, self.history_steps-1], axis=-1
    )
    agents_velocity_diff = np.abs(agents_velocity[1:] - agents_velocity[0])
    similar_agent_idx = np.argmin(agents_velocity_diff)
    
    if agents_velocity_diff[similar_agent_idx] < 2:
        copy_agent_idx = similar_agent_idx + 1
    else:
        copy_agent_idx = 0  # Ego を参考に
    
    # 【ステップC】スケール係数を計算
    if agents_velocity[copy_agent_idx] < 0.1:
        scale_coeff = 1.0
    else:
        scale_coeff = agents_velocity[0] / agents_velocity[copy_agent_idx]
    
    # 【ステップD】新しいエージェントを生成
    generated_agent = self._generate_agent_from_idx(
        data["agent"], copy_agent_idx, scale_coeff, path_point
    )
    
    # 【ステップE】リストに追加
    for key, value in new_data["agent"].items():
        new_data["agent"][key] = np.concatenate(
            [value, generated_agent[key][None, ...]],
            axis=0
        )
    
    return new_data, AGENT_CONTRAST_TYPE  # AGENT修正型
```

**例:**
```
【元】
  Ego: 加速（v=2m/s）
  前方: 自由な経路
  行動: 直進加速

【ネガティブ】
  Ego: 加速（v=2m/s）
  前方: 新しい車が登場
  行動: 直進加速 ✗ なぜ衝突しない？

モデル:「新しい障害物に対応できるか？」
```

---

### 6. `_generate_agent_from_idx()` メソッド

#### 役割
既存エージェントをテンプレートとして、新しいエージェントを生成。

#### シグネチャ
```python
def _generate_agent_from_idx(
    self,
    agent: Dict,  # エージェント情報
    idx: int,  # テンプレート選択インデックス
    scale_coeff: float,  # 速度スケール係数
    path_point: np.ndarray,  # 挿入する経路上の点 [x, y, yaw]
    shape_scale: List[float] = [0.9, 1.1]  # サイズスケール範囲
) -> Dict:
```

#### 処理フロー

```python
# 【ステップ1】スケール係数にランダムノイズを加える
scale_coeff *= np.random.uniform(low=0.0, high=0.8)

# 【ステップ2】既存エージェント (idx) の位置情報を取得
current_position = agent["position"][idx][history_steps-1]
hist_position = agent["position"][idx][:history_steps]  # 過去
fut_position = agent["position"][idx][history_steps-1:]  # 未来

# 【ステップ3】位置の差分を計算＆スケール
hist_diff = np.concatenate(
    [scale_coeff * np.diff(hist_position, axis=0), np.zeros((1, 2))],
    axis=0
)
fut_diff = scale_coeff * np.diff(fut_position, axis=0)

# 【ステップ4】スケール後の軌跡を再構成
scaled_position = np.concatenate([
    -np.cumsum(hist_diff[::-1], axis=0)[::-1] + current_position,  # 過去
    np.cumsum(fut_diff, axis=0) + current_position                 # 未来
], axis=0)

# 【ステップ5】方向（heading）を path_point に合わせる
heading = agent["heading"][idx]
delta_angle = heading[history_steps-1] - path_point[2]
cos, sin = np.cos(delta_angle), np.sin(delta_angle)
rot_mat = np.array([[cos, -sin], [sin, cos]])

new_position = np.matmul(scaled_position - current_position, rot_mat) + path_point[:2]
new_heading = heading - heading[history_steps-1] + path_point[2]

# 【ステップ6】速度を回転
velocity = scale_coeff * agent["velocity"][idx]
new_velocity = np.matmul(velocity, rot_mat)

# 【ステップ7】サイズをランダムにスケール
shape = agent["shape"][idx]
new_shape = shape * np.random.uniform(*shape_scale, size=shape.shape)

# 【ステップ8】出力ターゲット（将来軌跡）を計算
new_target = np.concatenate([
    new_position[history_steps:] - new_position[history_steps-1],  # 位置差分
    (new_heading[history_steps:] - new_heading[history_steps-1])[:, None]  # yaw差分
], axis=-1)

return {
    "position": new_position,
    "heading": new_heading,
    "velocity": new_velocity,
    "shape": new_shape,
    "category": agent["category"][idx],
    "valid_mask": agent["valid_mask"][idx],
    "target": new_target
}
```

---

### 7. `safety_check()` メソッド

#### 役割
Ego 車が周りのエージェントと衝突していないかを判定。

#### シグネチャ
```python
def safety_check(
    self,
    ego_position: np.ndarray,      # (2,) = [x, y]
    ego_heading: np.ndarray,       # () = yaw角度
    agents_position: np.ndarray,   # (N, 2)
    agents_heading: np.ndarray,    # (N,)
    agents_shape: np.ndarray       # (N, 2) = [width, length]
) -> bool:  # True: 衝突なし, False: 衝突あり
```

#### 処理フロー

```python
# 衝突なし（他の車がない）
if len(agents_position) == 0:
    return True

# Ego の中心位置を計算
ego_center = (
    ego_position
    + np.stack([np.cos(ego_heading), np.sin(ego_heading)], axis=-1)
    * self._rear_to_cog
)

# PyTorch テンソルに変換
ego_state = torch.from_numpy(
    np.concatenate([ego_center, [ego_heading]], axis=-1)
).unsqueeze(0)  # (1, 3)

objects_state = torch.from_numpy(
    np.concatenate([agents_position, agents_heading[..., None]], axis=-1)
).unsqueeze(0)  # (1, N, 3)

# 衝突判定エンジンで判定
collisions = self._collision_checker.collision_check(
    ego_state=ego_state,
    objects=objects_state,
    objects_width=torch.from_numpy(agents_shape[:, 0]).unsqueeze(0),
    objects_length=torch.from_numpy(agents_shape[:, 1]).unsqueeze(0)
)

# 結果を返す
return not collisions.any()  # 衝突がなければ True
```

---

## 💡 実装の工夫

### 1. ディープコピーで安全性確保
```python
new_data = deepcopy(data)  # 元データの変更を防ぐ
```

### 2. 正規化の重要性
```python
new_data = PlutoFeature.normalize(new_data).data  # 統計的に正規化
```

### 3. メタデータの活用
```python
data_n_info = {
    "valid_mask": data_n_valid_mask,  # 有効なネガティブか
    "type": contrast_type              # 修正方法（MAP or AGENT）
}
```

このメタデータは、モデルの訓練時に、どの修正方法が最も効果的かを分析できます。

---

## 🎨 カスタマイズ例

### 例1: 強力なデータ拡張
```yaml
# config.yaml
data_augmentation:
  history_steps: 30           # さらに長い過去を見る
  max_interaction_horizon: 60 # 相互作用範囲を拡張
  low: [-2.0, -3.0, -0.7, -2, -1.0, -0.4, -0.4]
  high: [3.0, 3.0, 0.7, 2, 1.0, 0.4, 0.4]
  use_negative_sample: true
```

### 例2: ポジティブのみ（対比学習なし）
```yaml
data_augmentation:
  use_negative_sample: false
```

### 例3: カスタムネガティブ生成
ファイルを編集して、`available_generators` をフィルタ:
```python
# negative generators を限定
if some_condition:
    available_generators = [self.neg_traffic_light_inversion]
```

---

## 📚 関連ファイル

- [README.md](./README.md) - モジュール概要
- [../custom_training/README.md](../custom_training/README.md) - 訓練ループ
- [../../../config/data_augmentation/contrastive_scenario_generator.yaml](../../../config/data_augmentation/contrastive_scenario_generator.yaml) - 設定
