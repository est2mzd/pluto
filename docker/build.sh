#!/bin/bash

# PLUTOプロジェクトのDockerイメージをビルドするスクリプト
# 使用方法: sh docker/build.sh

# エラー時にスクリプトを終了
set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# プロジェクトルートを取得（このスクリプトの親ディレクトリ）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 共通設定を読み込み（POSIXシェル互換のため . コマンドを使用）
. "$SCRIPT_DIR/common.sh"

echo "======================================"
echo "PLUTO Dockerイメージのビルドを開始します"
echo "======================================"
echo "プロジェクトルート: $PROJECT_ROOT"
echo "イメージ名: $IMAGE_NAME:$IMAGE_TAG"
echo ""

# Dockerfileのパスを確認（プロジェクトルートに配置）
DOCKERFILE_PATH="$PROJECT_ROOT/Dockerfile"
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "エラー: Dockerfileが見つかりません: $DOCKERFILE_PATH"
    exit 1
fi

# Dockerイメージをビルド
# --build-arg: ビルド時の引数を指定（将来的な拡張用）
# --tag: イメージ名とタグを指定
# --file: Dockerfileのパスを明示的に指定
# ビルドコンテキスト: プロジェクトルート
echo "Dockerイメージをビルド中..."
docker build \
    --tag "$IMAGE_NAME:$IMAGE_TAG" \
    --file "$DOCKERFILE_PATH" \
    "$PROJECT_ROOT"

# ビルド成功を確認
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "ビルドが成功しました！"
    echo "======================================"
    echo "イメージ: $IMAGE_NAME:$IMAGE_TAG"
    echo ""
    echo "次のステップ:"
    echo "  - コンテナを起動: sh docker/start.sh"
    echo "  - コンテナに入る: sh docker/into.sh"
else
    echo ""
    echo "エラー: ビルドに失敗しました"
    exit 1
fi
