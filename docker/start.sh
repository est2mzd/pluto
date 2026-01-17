#!/bin/bash

# PLUTOプロジェクトのDockerコンテナを起動するスクリプト
# 使用方法: sh docker/start.sh

# エラー時にスクリプトを終了
set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# プロジェクトルートを取得（このスクリプトの親ディレクトリ）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 共通設定を読み込み（POSIXシェル互換のため . コマンドを使用）
. "$SCRIPT_DIR/common.sh"

echo "======================================"
echo "PLUTO Dockerコンテナを起動します"
echo "======================================"

# 既存のコンテナが実行中かチェック
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "コンテナ '$CONTAINER_NAME' は既に実行中です"
    echo "コンテナに入るには: sh docker/into.sh"
    exit 0
fi

# 停止中のコンテナが存在するかチェック
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "停止中のコンテナ '$CONTAINER_NAME' を再起動します..."
    docker start "$CONTAINER_NAME"
    echo "コンテナを再起動しました"
    exit 0
fi

echo "設定:"
echo "  - イメージ: $IMAGE_NAME:$IMAGE_TAG"
echo "  - コンテナ名: $CONTAINER_NAME"
echo "  - プロジェクトルート: $PROJECT_ROOT"
echo "  - nuPlanデータ: $NUPLAN_DATA_ROOT -> /nuplan"
echo ""

# nuPlanデータパスの存在確認（警告のみ）
if [ ! -d "$NUPLAN_DATA_ROOT" ]; then
    echo "警告: nuPlanデータパスが存在しません: $NUPLAN_DATA_ROOT"
    echo "環境変数NUPLAN_DATA_ROOTを設定してください"
    echo "例: export NUPLAN_DATA_ROOT=/media/takuya/Transcend/work/nuPlan"
fi

# Dockerコンテナを起動
# --gpus all: すべてのGPUを利用可能にする（学習・推論に必要）
# --name: コンテナに名前を付けて管理しやすくする
# -v: ボリュームマウント（ホストとコンテナ間でファイル共有）
#   - プロジェクトルート: コード変更をリアルタイムで反映
#   - nuPlanデータ: 大容量データセット（data, maps, zips）へのアクセス
#   - nuPlan実験結果: 学習結果やキャッシュの保存
# --shm-size: 共有メモリサイズ（PyTorchのDataLoaderで複数workerを使用する際に必要）
# -d: デタッチモード（バックグラウンドで実行）
# --rm: コンテナ停止時に自動削除（オプション、必要に応じて削除）
docker run \
    --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$PROJECT_ROOT:/workspace/pluto" \
    -v "$NUPLAN_DATA_ROOT:/nuplan" \
    --shm-size=$SHM_SIZE \
    -d \
    -it \
    "$IMAGE_NAME:$IMAGE_TAG" \
    /bin/bash

# 起動成功を確認
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "コンテナの起動に成功しました！"
    echo "======================================"
    echo "コンテナ名: $CONTAINER_NAME"
    echo ""
    echo "次のステップ:"
    echo "  - コンテナに入る: sh docker/into.sh"
    echo "  - コンテナを停止: sh docker/stop.sh"
    echo "  - ログを確認: docker logs $CONTAINER_NAME"
else
    echo ""
    echo "エラー: コンテナの起動に失敗しました"
    exit 1
fi
