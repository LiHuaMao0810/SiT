#!/usr/bin/env bash
set -euo pipefail

# -------- editable params --------
DEVICE="${DEVICE:-cuda:2}"
CKPT="${CKPT:-pretrained_models/SiT-XL-2-256x256.pt}"
RESULTS_DIR="${RESULTS_DIR:-results/distill_only}"
BATCH_SIZE="${BATCH_SIZE:-2}"
T_HQ="${T_HQ:-5}"
TOTAL_ITERS="${TOTAL_ITERS:-3000}"
LOG_EVERY="${LOG_EVERY:-10}"
SAMPLE_EVERY="${SAMPLE_EVERY:-100}"   # 0 disables sample image saving
CKPT_EVERY="${CKPT_EVERY:-1000}"       # 0 disables checkpoint saving
CFG_SCALE="${CFG_SCALE:-1.5}"
LR="${LR:-1e-4}"
SEED="${SEED:-0}"

python train_distill_only.py \
  --ckpt "${CKPT}" \
  --results-dir "${RESULTS_DIR}" \
  --device "${DEVICE}" \
  --T-hq "${T_HQ}" \
  --batch-size "${BATCH_SIZE}" \
  --cfg-scale "${CFG_SCALE}" \
  --lr "${LR}" \
  --total-iters "${TOTAL_ITERS}" \
  --log-every "${LOG_EVERY}" \
  --sample-every "${SAMPLE_EVERY}" \
  --ckpt-every "${CKPT_EVERY}" \
  --seed "${SEED}" \
  --amp
