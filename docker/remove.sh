#!/bin/bash

# PLUTOプロジェクトのDockerコンテナとイメージを削除するスクリプト
# 使用方法: sh docker/remove.sh
# 機能: コンテナが実行中なら停止してから削除、その後イメージも削除

# エラー時にスクリプトを終了
set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 共通設定を読み込み（POSIXシェル互換のため . コマンドを使用）
. "$SCRIPT_DIR/common.sh"

echo "======================================"
echo "PLUTO Dockerコンテナとイメージを削除します"
echo "======================================"

# コンテナが実行中かチェック
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "コンテナ '$CONTAINER_NAME' が実行中です。停止します..."
    docker stop -t 10 "$CONTAINER_NAME"
    echo "コンテナを停止しました"
fi

# コンテナが存在するかチェック
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "コンテナ '$CONTAINER_NAME' を削除中..."
    docker rm "$CONTAINER_NAME"
    echo "コンテナを削除しました"
else
    echo "コンテナ '$CONTAINER_NAME' は存在しません"
fi

echo ""
echo "======================================"
echo "削除完了！"
echo "======================================"
echo ""
echo "次のステップ:"
echo "  - 新しいコンテナを起動: sh docker/start.sh"
