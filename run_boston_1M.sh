#!/usr/bin/env bash
set -euo pipefail

# ==============================
# CPU / worker 設定
# ==============================
CPU_COUNT=$(nproc)
MAX_WORKERS=$(( CPU_COUNT * 3 / 4 ))

echo "CPUコア数: ${CPU_COUNT}"
echo "割り当てワーカー数（3/4）: ${MAX_WORKERS}"

# ==============================
# GPU 設定
# ==============================
export CUDA_VISIBLE_DEVICES=0

# ==============================
# Python 実行環境
# ==============================
PYTHON_PATH="/opt/conda/envs/pluto/bin/python"
WORKDIR="/workspace/pluto"

# ==============================
# Training 実行
# ==============================
cd "${WORKDIR}"

"${PYTHON_PATH}" run_training.py \
  py_func=train \
  +training=train_pluto \
  group=/workspace/pluto/tensorboard \
  splitter=ratio_splitter \
  worker=single_machine_thread_pool \
  worker.max_workers="${MAX_WORKERS}" \
  scenario_builder=nuplan_boston \
  cache.cache_path=/nuplan/exp/pluto/cache_pluto_boston_1M \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=64 \
  data_loader.params.num_workers=16 \
  lr=1e-3 \
  epochs=300 \
  warmup_epochs=3 \
  weight_decay=0.0001 \
  wandb.mode=disable \
  wandb.log_model=False \
  wandb.project=nuplan_boston1M_from_v26 \
  wandb.name=pluto \
  checkpoint=/workspace/pluto/ckpt/boston_1M/model_v199.ckpt

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
exit "${EXIT_CODE}"
