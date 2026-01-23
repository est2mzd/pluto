# train_boston 学習試行レコード

## 🎯 プロジェクト概要

このディレクトリは、**train_bostonデータセット**を使用した、Plutoモデルの学習セットアップと試行記録を保管しています。

### 目的
- train_bostonデータ（Boston市の1,647シナリオ）でのキャッシュ作成
- Plutoモデルの学習実行
- 各試行のログとレポート保存

---

## 📊 実行結果サマリー

| ステップ | 内容 | 結果 | ファイル |
|---------|------|------|--------|
| Step 1 | 環境構築・設定作成 | ✅ 成功 | [execution_summary.md](execution_summary.md) |
| Step 2 | mini(50)での学習 | ✅ 成功 | [step4_mini_training_v2.log](step4_mini_training_v2.log) |
| Step 3 | boston(500)キャッシュ | ✅ 成功 | [step5_boston_cache_500.log](step5_boston_cache_500.log) |
| Step 4 | boston(500)学習 | ⚠️ NaN問題 | [FINAL_REPORT.md](FINAL_REPORT.md) |

---

## 📁 ファイル構成

### レポート・実行記録

#### [📋 FINAL_REPORT.md](FINAL_REPORT.md) - 最終実行レポート
最終的な実行結果と課題分析。

**内容**：
- ✅ 成功した内容（設定作成、キャッシュ作成）
- ⚠️ 問題点（NaN値）
- 🔧 対策案（4つのアプローチ）
- 📊 成果物一覧

**推奨読順**: 最初に読むべき

#### [📊 execution_summary.md](execution_summary.md) - 実行サマリー
各ステップの実行内容と結果の詳細。

**内容**：
- Step 1: キャッシュ作成（✅ 成功）
- Step 2: 学習1回目（❌ 失敗: Splitter問題）
- Step 3: miniテスト（✅ 成功）
- Step 4: bostonキャッシュ（✅ 成功）
- Step 5-7: 学習試行（⚠️ NaN問題）

### ログファイル

| ファイル | 説明 | 結果 | 実行時間 |
|---------|------|------|--------|
| `step3_mini_training.log` | mini(50), epoch=3 | ❌ LR scheduler問題 | 1分 |
| `step4_mini_training_v2.log` | mini(50), epoch=10 | ✅ 成功 | 2分 |
| `step5_boston_cache_500.log` | boston(500) キャッシュ | ✅ 成功(99%) | 8分 |
| `step6_boston_training.log` | boston(500), epoch=10 | ❌ NaN assertion | 8分 |
| `step7_boston_training_lr0001.log` | boston(500), lr=1e-4 | ❌ 同じNaN | 8分 |
| `step8_boston_training_clamp.log` | boston(500), NaN clamp | ⚠️ 完了(メトリクスNaN) | 8分 |
| `archive_train_boston_execution_log.md` | 初期実行ログ（参考用） | ℹ️ 途中停止 | — |

---

## 🔑 キーポイント

### ✅ 成功したこと

1. **カスタムRatioSplitter作成**
   - 任意のデータセットに対応可能な比率ベース分割
   - コード: `/workspace/pluto/src/custom_training/ratio_splitter.py`

2. **train_boston設定完成**
   - config: `/workspace/pluto/config/training/train_boston.yaml`
   - filter: `/workspace/pluto/config/scenario_filter/training_scenarios_boston.yaml`

3. **キャッシュ作成成功**
   - 500シナリオ処理完了
   - 失敗率: 1% (10 out of 1000)
   - パス: `/nuplan/exp/boston_cache_500/`

4. **mini(50)での学習成功**
   - 10エポック完全実行
   - train: 38, val: 5, test: 7

### ⚠️ 課題: NaN問題

**症状**:
- planning_decoder で Non-finite値が大量発生
- 全メトリクスがNaN

**試行した対策**:
- ✗ 学習率低下（1e-3 → 1e-4）
- ✓ NaN clamp（プログラムは完了、ただしメトリクスNaN）

**考えられる原因**:
1. train_bostonの特徴量分布がminiと異なる
2. 特徴量の正規化が不適切
3. モデルの初期化がbostonに適応していない

---

## 🚀 次のステップ（推奨順）

### 1️⃣ 原因調査（優先度: 高）
```bash
# Boston vs Mini の特徴量統計比較
python analyze_features.py --cache_path /nuplan/exp/boston_cache_500
```

**確認項目**:
- 特徴量の min/max/mean/std
- 正規化パラメータの確認
- NaN/Inf の出現箇所

### 2️⃣ 少ないデータで段階的テスト（優先度: 中）
```bash
# 100シナリオで試行
python run_training.py py_func=cache +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_100_v2 \
  scenario_filter.training_scenarios_boston.limit_total_scenarios=100

# 学習
python run_training.py py_func=train +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_100_v2 \
  epochs=5
```

### 3️⃣ Fine-tuning アプローチ（優先度: 中）
```bash
# miniで学習済みモデルをロード
python run_training.py py_func=train +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_500 \
  checkpoint=/root/nuplan/exp/exp/training/pluto/*/checkpoints/best.ckpt \
  lr=0.00001
```

### 4️⃣ データ前処理改善（優先度: 低）
- 特徴量の正規化方式確認
- scaling パラメータの調整
- outlier 除外

---

## 📖 関連ドキュメント

### 基本ドキュメント
- [`/workspace/pluto/docs/analysis_devkit/command_settings_detailed.md`](../analysis_devkit/command_settings_detailed.md) - コマンド引数の詳細解説
- [`/workspace/pluto/docs/analysis_devkit/beginner_guide.md`](../analysis_devkit/beginner_guide.md) - システム概要

### 実装ファイル
- **Splitter**: `/workspace/pluto/src/custom_training/ratio_splitter.py`
- **Training**: `/workspace/pluto/run_training.py`
- **Config**: `/workspace/pluto/config/training/train_boston.yaml`

---

## 💾 キャッシュ・チェックポイント

| リソース | パス | 状態 | サイズ |
|---------|------|------|--------|
| boston_cache_500 | `/nuplan/exp/boston_cache_500/` | ✅ 利用可能 | ~1GB |
| mini_cache | `/nuplan/exp/sanity_check/` | ✅ 利用可能 | ~100MB |

---

## 🔍 トラブルシューティング

### Q: なぜNaNが出るのか？
**A**: 
- train_bostonの特徴量分布がminiと大きく異なる可能性
- モデルパラメータの初期化がbostonに適応していない
- → 詳細は [FINAL_REPORT.md](FINAL_REPORT.md) の "次のステップ" を参照

### Q: miniでは成功したのになぜboston失敗？
**A**:
- mini: 50シナリオ → 安定した学習
- boston: 500シナリオ → より複雑なデータ分布
- Plutoモデルが多様な環境に対応需要

### Q: 過去の試行を確認したい
**A**:
- 各ステップのログは `stepN_*.log` ファイル
- 詳細は [execution_summary.md](execution_summary.md) を参照

---

## 📝 使用コマンド

### キャッシュ作成
```bash
cd /workspace/pluto
python run_training.py \
  py_func=cache \
  +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_500 \
  cache.cleanup_cache=true \
  worker=sequential
```

### 学習実行
```bash
cd /workspace/pluto
python run_training.py \
  py_func=train \
  +training=train_boston \
  cache.cache_path=/nuplan/exp/boston_cache_500 \
  worker=sequential \
  epochs=10
```

---

**最終更新**: 2026-01-21  
**ステータス**: 🟡 進行中（NaN問題解決待ち）

