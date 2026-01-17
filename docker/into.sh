#!/bin/bash

# 実行中のPLUTO Dockerコンテナに入るスクリプト
# 使用方法: sh docker/into.sh

# エラー時にスクリプトを終了
set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 共通設定を読み込み（POSIXシェル互換のため . コマンドを使用）
. "$SCRIPT_DIR/common.sh"

echo "======================================"
echo "PLUTO Dockerコンテナに接続します"
echo "======================================"

# コンテナが実行中かチェック
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "エラー: コンテナ '$CONTAINER_NAME' が実行されていません"
    echo ""
    echo "コンテナを起動してください: sh docker/start.sh"
    exit 1
fi

echo "コンテナ '$CONTAINER_NAME' に接続中..."
echo ""

# コンテナに入る
# -it: インタラクティブモードでTTYを割り当て
# /bin/bash: bashシェルを起動
# 根拠: 開発作業やデバッグのためにシェルアクセスが必要
docker exec -it "$CONTAINER_NAME" /bin/bash

echo ""
echo "コンテナから退出しました"
