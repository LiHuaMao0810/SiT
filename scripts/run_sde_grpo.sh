#!/usr/bin/env bash
set -euo pipefail

# -------- editable params --------
DEVICE="${DEVICE:-cuda:3}"
CKPT="${CKPT:-pretrained_models/SiT-XL-2-256x256.pt}"
RESULTS_DIR="${RESULTS_DIR:-results/sde_grpo}"
BATCH_SIZE="${BATCH_SIZE:-1}"
T_EXP="${T_EXP:-5}"
T_HQ="${T_HQ:-50}"
G="${G:-4}"
K="${K:-2}"
BG_CHUNK_SIZE="${BG_CHUNK_SIZE:-2}"
SDE_SIGMA="${SDE_SIGMA:-0.1}"
EPS_CLIP="${EPS_CLIP:-0.05}"
ADV_CLIP="${ADV_CLIP:-2.0}"
LAMBDA_MSE="${LAMBDA_MSE:-0.1}"
RATIO_STOP="${RATIO_STOP:-1.5}"
CFG_SCALE="${CFG_SCALE:-1.5}"
LR="${LR:-1e-4}"
TOTAL_ITERS="${TOTAL_ITERS:-3000}"
LOG_EVERY="${LOG_EVERY:-50}"
SAMPLE_EVERY="${SAMPLE_EVERY:-100}"   # 0 disables sample image saving
CKPT_EVERY="${CKPT_EVERY:-1000}"       # 0 disables checkpoint saving
SEED="${SEED:-0}"

python train_sde_grpo.py \
  --ckpt "${CKPT}" \
  --results-dir "${RESULTS_DIR}" \
  --device "${DEVICE}" \
  --T-exp "${T_EXP}" \
  --T-hq "${T_HQ}" \
  --batch-size "${BATCH_SIZE}" \
  --G "${G}" \
  --K "${K}" \
  --bg-chunk-size "${BG_CHUNK_SIZE}" \
  --sde-sigma "${SDE_SIGMA}" \
  --eps-clip "${EPS_CLIP}" \
  --adv-clip "${ADV_CLIP}" \
  --lambda-mse "${LAMBDA_MSE}" \
  --ratio-stop "${RATIO_STOP}" \
  --cfg-scale "${CFG_SCALE}" \
  --lr "${LR}" \
  --total-iters "${TOTAL_ITERS}" \
  --log-every "${LOG_EVERY}" \
  --sample-every "${SAMPLE_EVERY}" \
  --ckpt-every "${CKPT_EVERY}" \
  --seed "${SEED}" \
  --amp
