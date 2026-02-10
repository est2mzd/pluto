#!/usr/bin/env bash
# 目的:
#   checkpoint ディレクトリを定期監視し、
#   まだコピーされていない .ckpt ファイルだけを別ディレクトリに退避する
# 特徴:
#   - rsync --ignore-existing を使用（上書きなし）
#   - シンプルな SRC -> DST コピー（パス二重化バグを回避）

set -euo pipefail                               # エラー即終了・未定義変数検知

# ===== ユーザー設定 =====
SRC_DIR="/workspace/pluto/tensorboard/training/pluto/2026.02.06.02.45.42/checkpoints"
SRC_DIR="/workspace/pluto/tensorboard/training/pluto/2026.02.08.16.25.03/checkpoints"
DST_DIR="/workspace/pluto/ckpt/boston_1M"
INTERVAL_SEC=600                               # チェック間隔（秒）
# ========================

mkdir -p "${DST_DIR}"                          # 退避先が無ければ作成

# 二重起動防止（flock がある場合のみ）
if command -v flock >/dev/null 2>&1; then
  exec 9>"${DST_DIR}/.copy_ckpt.lock"
  flock -n 9 || exit 0
fi

echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] DST_DIR=${DST_DIR}"
echo "[INFO] INTERVAL_SEC=${INTERVAL_SEC}"

while true; do
  TS="$(date '+%Y-%m-%d %H:%M:%S')"

  # *.ckpt を「存在しないものだけ」コピー
  rsync -av --ignore-existing \
    "${SRC_DIR}/"*.ckpt \
    "${DST_DIR}/" \
    && echo "[${TS}] Copied new checkpoint(s)." \
    || echo "[${TS}] WARN: rsync failed (check path or permissions)." >&2

  sleep "${INTERVAL_SEC}"
done
