# Pluto Training Failure: q が NaN / Inf になる問題の整理と対策

## 1. 現状の整理（事実ベース）

### 1.1 発生しているエラー

学習中、以下の assert で停止する。

```python
assert torch.isfinite(q).all()
```

発生箇所：

```
PlanningDecoder.forward()
```

`q` は PlanningDecoder 内部で生成される **trajectory query tensor**。

---

### 1.2 地図（map.gpkg）について

確認結果：

* `/nuplan/dataset/maps/us-ma-boston/9.12.1817/map.gpkg` → **存在**
* `/nuplan/dataset/maps/us-nv-las-vegas-strip/9.15.1915/map.gpkg` → **存在**

よって：

❌ map.gpkg の **実体欠如が直接の原因ではない**

（※ map を一時的に動かしたことで別エラーは出たが、現在の q NaN 問題とは独立）

---

## 2. q がどこで作られているか（コード構造）

```python
r_emb = self.r_encoder(...)
m_emb = self.m_emb
q = self.q_proj(torch.cat([r_emb, m_emb], dim=-1))

for blk in self.decoder_blocks:
    q = blk(...)
    assert torch.isfinite(q).all()
```

NaN/Inf の混入ポイントは **2 系統のみ**。

---

## 3. NaN/Inf の確定的発生ルート

### 系統 A：PointsEncoder に「全 False mask」が入る

#### 問題箇所

```python
r_valid_mask = r_valid_mask.view(bs * R, P)
r_emb = self.r_encoder(r_feature, r_valid_mask)
```

`PointsEncoder.forward()` 内部では：

```python
x_valid = x[mask]          # mask が全 False → 空
BatchNorm1d(x_valid)      # 不定挙動 / NaN 発生しうる
```

#### 発生条件

* reference_line が **全 invalid** なサンプルが混入
* Boston データでも実際に起きうる（事実）

#### 結果

* `r_emb` に NaN
* `q = Linear([r_emb, m_emb])` で即 NaN

---

### 系統 B：MultiheadAttention に「全マスク系列」が入る

#### 問題箇所

```python
self.cross_attn(
    tgt, memory, memory,
    key_padding_mask=enc_key_padding_mask
)
```

#### PyTorch MHA の仕様

* key_padding_mask が **全 True**
* attention score = 全 -inf
* softmax → NaN

#### 発生条件

* encoder 側出力が全 invalid
* または scenario/map の不整合で encoder feature が全 mask

---

## 4. これまでの対策が効かなかった理由

| 試した対策           | 効果がなかった理由                                |
| --------------- | ---------------------------------------- |
| `x / sqrt(dim)` | planning_decoder では scale 不要。NaN の根源ではない |
| dropout 調整      | NaN 発生後の操作なので無意味                         |
| attn_norm 追加    | Decoder 側には存在しない                         |

👉 **NaN は「数値発散」ではなく「無効入力」由来**

---

## 5. 最小・正当な対策方針（推奨順）

### 対策①：PointsEncoder を「全 False mask 安全」にする（最重要）

**意図**：

* 無効 reference_line を「ゼロ特徴」として扱う
* NaN を作らない

#### 最小修正案（例）

```python
# PointsEncoder.forward の冒頭
if mask.sum() == 0:
    return torch.zeros(bs, self.encoder_channel, device=device)
```

※ 不要なロジック変更なし

---

### 対策②：Decoder 側で「全マスク系列」を1点だけ解除

**意図**：

* MultiheadAttention の softmax NaN を防ぐ

```python
all_e = enc_key_padding_mask.all(dim=1)
enc_key_padding_mask[all_e, 0] = False
```

※ これは **NaN 回避の安全弁**。根本解決は①。

---

## 6. 原因切り分け用の最小ログ（推奨）

```python
if not torch.isfinite(q).all():
    print(
        'r_emb finite=', torch.isfinite(r_emb).all().item(),
        'enc_mask all_true=', enc_key_padding_mask.all(dim=1).any().item(),
    )
    raise
```

これで：

* r_emb が NaN → **系統 A**
* block 後で NaN → **系統 B**

が即判別可能。

---

## 7. まとめ（重要）

* ❌ map.gpkg は原因ではない
* ❌ attention の数値安定化では解決しない
* ✅ 原因は「全 invalid 入力」
* ✅ 対策は mask / empty-input handling

---

## 8. 次にやるべきこと（最短）

1. PointsEncoder に **全 False mask 対策**を入れる
2. そのまま再実行
3. まだ落ちたら Decoder 側全マスク対策を追加

この順で、**最小変更・最大確度**で直ります。
