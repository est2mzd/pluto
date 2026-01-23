# train_boston 実行ログ

📌 **[← 戻る: README.md](README.md)** | このドキュメントは train_trial README から参照されています

---

## Step 1: キャッシュ作成 ✅ 成功

**コマンド**:
```bash
python run_training.py py_func=cache +training=train_boston cache.cache_path=/nuplan/exp/boston_cache_100_v2 cache.cleanup_cache=true worker=sequential
```

**結果**: ✅ 成功
- 処理時間: 約8分
- シナリオ: 100/100 完了
- 失敗: 0 out of 200
- キャッシュ作成位置: `/nuplan/exp/boston_cache_100_v2/`

---

## Step 2: 学習実行 ❌ 失敗

**コマンド**:
```bash
python run_training.py py_func=train +training=train_boston cache.cache_path=/nuplan/exp/boston_cache_100_v2 worker=sequential epochs=5
```

**エラー**:
```
AssertionError: Splitter returned no validation samples
```

**原因**: 
- Splitterが検証セット（val）を返していない
- 100シナリオ中、val:15% = 15サンプル のはずが 0 になっている

**ログ**: `/workspace/pluto/docs/logs/step2_training_attempt1.log`

---

## Step 3: デバッグと修正試行

**対策**:
1. Splitterの動作確認
2. データ分割の問題調査
3. 設定の見直し

**作成したファイル**:
- `/workspace/pluto/src/custom_training/ratio_splitter.py` - カスタム比率ベースSplitter
- `/workspace/pluto/config/splitter/ratio_splitter.yaml` - Splitter設定
- `/workspace/pluto/config/training/train_boston.yaml` - 更新（ratio_splitterを使用）

---

## Step 4: miniデータでの学習テスト ✅ 成功

**コマンド**:
```bash
python run_training.py py_func=train +training=train_pluto cache.cache_path=/nuplan/exp/sanity_check worker=sequential epochs=10
```

**結果**: ✅ 成功
- 訓練セット: 38サンプル
- 検証セット: 5サンプル
- 全10エポック完了
- ログ: `/workspace/pluto/docs/logs/step4_mini_training_v2.log`

---

## Step 5: train_bostonキャッシュ作成（500） ✅ 成功

**コマンド**:
```bash
python run_training.py py_func=cache +training=train_boston cache.cache_path=/nuplan/exp/boston_cache_500 cache.cleanup_cache=true worker=sequential
```

**結果**: ✅ 成功
- 500/500 シナリオ処理完了
- 失敗: 10 out of 1000 (99%成功)
- 実行時間: 約8分
- ログ: `/workspace/pluto/docs/logs/step5_boston_cache_500.log`

---

## Step 6: train_boston学習 ❌ 失敗（NaN問題）

**コマンド**:
```bash
python run_training.py py_func=train +training=train_boston cache.cache_path=/nuplan/exp/boston_cache_500 worker=sequential epochs=10
```

**エラー**:
```
AssertionError in planning_decoder.py line 175: assert torch.isfinite(q).all()
```

**原因**: 
- モデルforward中に NaN/Inf 値が発生
- planning_decoder のクエリ (q) に無限大値が含まれる
- 可能性：学習率が高すぎる、初期化の問題、特徴量の問題

**ログ**: `/workspace/pluto/docs/logs/step6_boston_training.log`

---

## まとめと次のステップ

### 成功した内容
1. ✅ train_boston用設定作成
2. ✅ カスタムRatioSplitter作成
3. ✅ train_bostonキャッシュ作成（500シナリオ）
4. ✅ mini（50シナリオ）での学習成功

### 残っている問題
1. ❌ train_boston学習でNaN/Inf発生

### 考えられる解決策
1. 学習率を下げる（例: 1e-4 → 1e-5）
2. Gradient clipping を有効化
3. Mixed precision (FP16) を試す
4. バッチサイズを調整
5. 特徴量の正規化を確認
6. より少ないシナリオ（100-200）で試す

### おやすみなさい！
ユーザーが戻られたら、上記の解決策を試して学習を成功させます。
