#!/bin/bash

# PLUTOプロジェクトのDocker環境で共通で使用する環境変数と設定
# 他のスクリプトから source コマンドで読み込んで使用します

# Dockerイメージの設定
# 根拠: すべてのスクリプトで同じイメージ名・タグを使用する必要がある
IMAGE_NAME="pluto"
IMAGE_TAG="latest"

# Dockerコンテナの設定
# 根拠: コンテナ名を統一することで、複数のスクリプトから同じコンテナを操作可能
CONTAINER_NAME="pluto_container"

# nuPlanデータセットのパス設定
# ホスト側のパスを指定（コンテナ内では /nuplan にマウントされる）
# 根拠: start.sh で -v "$NUPLAN_DATA_ROOT:/nuplan" でマウントされる
NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/mnt/usb-hdd-01/nuPlan}"

# 共有メモリサイズの設定
# 根拠: PyTorchのDataLoaderで複数workerを使用する際に必要
# README.mdで data_loader.params.num_workers=16 などが使用されている
SHM_SIZE="16g"

# スクリプトのディレクトリとプロジェクトルートを取得する関数
# 各スクリプトから呼び出して使用する
get_project_root() {
    # common.sh が呼ばれたスクリプト（docker/build.sh など）の親ディレクトリがプロジェクトルート
    # BASH_SOURCE[1] は common.sh を読み込んだスクリプト（build.sh など）
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    local project_root="$(cd "$script_dir/.." && pwd)"
    echo "$project_root"
}
