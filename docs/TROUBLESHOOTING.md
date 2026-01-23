# PLUTO トラブルシューティング

## 問題 1: ModuleNotFoundError: No module named 'nuplan'

### 症状
```bash
python run_training.py ...
Traceback (most recent call last):
  File "/workspace/pluto/run_training.py", line 7, in <module>
    from nuplan.planning.script.builders.folder_builder import (
ModuleNotFoundError: No module named 'nuplan'
```

### 原因
- Dockerfileを編集した後、コンテナを再起動した際に発生
- `nuplan-devkit` がインストールされていない状態になった
- Dockerfileの`CMD`で実行される`pip install -e /workspace/nuplan-devkit`がコンテナ起動時に実行されるはずだが、何らかの理由で実行されていない

### 確認方法
```bash
# nuplanモジュールがインストールされているか確認
pip list | grep nuplan

# 何も表示されなければインストールされていない
```

### 解決方法
```bash
cd /workspace/pluto
pip install -e ./nuplan-devkit
```

### 結果
```bash
Successfully installed nuplan-devkit-1.2.2
```

### 検証
```bash
python -c "import nuplan; print('nuplan imported successfully')"
# 出力: nuplan imported successfully
```

---

## 問題 2: TypeError - attn_drop パラメータエラー（未解決）

### 症状
```bash
Error in call to target 'src.models.pluto.pluto_model.PlanningModel':
TypeError("__init__() got an unexpected keyword argument 'attn_drop'")
full_key: model
```

### 状況
- `nuplan`モジュールのインポートは成功
- モデルの初期化時にパラメータエラーが発生
- `attn_drop`パラメータが予期しない引数として扱われている

### 次のステップ
1. `HYDRA_FULL_ERROR=1`でフルスタックトレースを確認
2. `src/models/pluto/pluto_model.py`のPlanningModelクラスを確認
3. `config/model/pluto_model.yaml`の設定を確認
4. timmライブラリのバージョンとの互換性を確認

---

## Docker環境での注意事項

### コンテナ再起動後の必須手順

Dockerfileを編集してコンテナを再ビルド・再起動した場合、以下を確認：

1. **nuplan-devkitのインストール確認**
   ```bash
   pip list | grep nuplan
   ```

2. **インストールされていない場合**
   ```bash
   cd /workspace/pluto
   pip install -e ./nuplan-devkit
   ```

3. **PYTHONPATH の確認**
   ```bash
   echo $PYTHONPATH
   # 期待値: /workspace: (またはそれを含む)
   ```

### Dockerfileの関連部分

現在のDockerfile (CMD部分):
```dockerfile
CMD ["/bin/bash", "-c", "\
    /opt/conda/bin/conda run -n pluto pip install -e /workspace/nuplan-devkit && \
    /opt/conda/bin/conda run -n pluto pip install -e /workspace && \
    /opt/conda/bin/conda run -n pluto bash -c 'cd /workspace && sh ./script/setup_env.sh' && \
    exec /bin/bash"]
```

**注意**: CMDは`docker run`時に実行されるが、`docker exec`などでシェルに入った場合は実行されない。

---

## 作業履歴 (2026-01-17)

### タイムライン

1. **21:45** - Sanity check実行試行
   - エラー: `ModuleNotFoundError: No module named 'nuplan'`
   
2. **21:46** - 調査と修正
   - Dockerfileの確認
   - `pip list`でnuplan未インストールを確認
   - `pip install -e ./nuplan-devkit`を実行して解決
   
3. **21:46** - 再実行
   - nuplanのインポート成功
   - 新しいエラー: `attn_drop`パラメータエラー発生
   - キャッシング処理は開始したが、モデル初期化で失敗

### 状態

- ✅ Docker環境構築完了
- ✅ nuplan-devkitインストール完了
- ✅ nuplanモジュールのインポート成功
- ❌ モデル初期化エラー（調査中）
