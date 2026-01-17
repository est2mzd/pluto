# PLUTO Docker環境

このディレクトリには、PLUTOプロジェクトをDocker環境で実行するためのファイルが含まれています。

## ファイル構成

- **Dockerfile**: PLUTOプロジェクト用のDockerイメージ定義
- **build.sh**: Dockerイメージをビルドするスクリプト
- **start.sh**: Dockerコンテナを起動するスクリプト
- **into.sh**: 実行中のDockerコンテナに入るスクリプト
- **stop.sh**: Dockerコンテナを停止するスクリプト
- **.dockerignore**: Dockerイメージビルド時に除外するファイルの定義

## 使用方法

### 1. 環境変数の設定

nuPlanデータセットのパスを環境変数として設定してください：

```bash
export NUPLAN_DATA_PATH=/path/to/nuplan/data
```

これらのパスは、コンテナ内の `/nuplan/data` と `/nuplan/exp` にマウントされます。

### 2. Dockerイメージのビルド

```bash
sh docker/build.sh
```

このスクリプトは `pluto:latest` という名前のDockerイメージを作成します。

### 3. Dockerコンテナの起動

```bash
sh docker/start.sh
```

このスクリプトは `pluto_container` という名前でコンテナを起動します。

### 4. コンテナに入る

```bash
sh docker/into.sh
```

実行中のコンテナにbashシェルで接続します。

### 5. コンテナの停止

```bash
sh docker/stop.sh
```

実行中のコンテナを安全に停止します。

## コンテナ内での作業

コンテナ内では、プロジェクトルートは `/workspace/pluto` にマウントされています。

### 特徴キャッシュの作成

```bash
python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan_mini \
    cache.cache_path=/nuplan/exp/sanity_check \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_tiny \
    worker=sequential
```

### 学習の実行

```bash
CUDA_VISIBLE_DEVICES=0 python run_training.py \
    py_func=train +training=train_pluto \
    worker=single_machine_thread_pool worker.max_workers=4 \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/sanity_check \
    cache.use_cache_without_dataset=true \
    data_loader.params.batch_size=4 \
    data_loader.params.num_workers=1
```

### シミュレーションの実行

```bash
sh ./script/run_pluto_planner.sh pluto_planner nuplan_mini mini_demo_scenario pluto_1M_aux_cil.ckpt /nuplan/exp/simulation_results
```

## 注意事項

- **GPU要件**: このDockerイメージはGPUサポートを前提としています。`--gpus all` フラグでコンテナを起動します。
- **共有メモリ**: PyTorchのDataLoaderで複数workerを使用する場合、十分な共有メモリが必要です（デフォルト: 16GB）。
- **ボリュームマウント**: プロジェクトのコードとデータはボリュームマウントされているため、ホスト側での変更がコンテナ内に即座に反映されます。

## トラブルシューティング

### GPU が認識されない場合

nvidia-docker がインストールされていることを確認してください：

```bash
docker run --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi
```

### 共有メモリ不足エラー

`start.sh` の `--shm-size` パラメータを調整してください：

```bash
--shm-size=32g  # 32GBに増やす
```

### パーミッションエラー

コンテナ内のユーザーとホストのユーザーのUID/GIDが異なる場合、ファイルのパーミッション問題が発生することがあります。必要に応じて Dockerfile でユーザーを作成してください。
