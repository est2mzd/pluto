#!/bin/bash

# PLUTO Dockerコンテナを停止するスクリプト
# 使用方法: sh docker/stop.sh

# エラー時にスクリプトを終了
set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 共通設定を読み込み（POSIXシェル互換のため . コマンドを使用）
. "$SCRIPT_DIR/common.sh"

echo "======================================"
echo "PLUTO Dockerコンテナを停止します"
echo "======================================"

# コンテナが存在するかチェック
if [ ! "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "コンテナ '$CONTAINER_NAME' は存在しません"
    exit 0
fi

# コンテナが実行中かチェック
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "コンテナ '$CONTAINER_NAME' は既に停止しています"
    exit 0
fi

echo "コンテナ '$CONTAINER_NAME' を停止中..."

# コンテナを停止
# -t: 停止前に待機する秒数（デフォルトは10秒）
# 根拠: 実行中のプロセスが安全に終了できるように猶予時間を設定
docker stop -t 10 "$CONTAINER_NAME"

# 停止成功を確認
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "コンテナを停止しました"
    echo "======================================"
    echo ""
    echo "次のステップ:"
    echo "  - コンテナを再起動: sh docker/start.sh"
    echo "  - コンテナを削除: docker rm $CONTAINER_NAME"
else
    echo ""
    echo "エラー: コンテナの停止に失敗しました"
    exit 1
fi
