#!/usr/bin/env bash
set -euo pipefail

# Launch three experiments in parallel on three GPUs.
# Edit GPU IDs here if needed:
GPU_DISTILL="${GPU_DISTILL:-cuda:2}"
GPU_ELBO="${GPU_ELBO:-cuda:3}"
GPU_SDE="${GPU_SDE:-cuda:4}"

echo "[1/3] distill_only on ${GPU_DISTILL}"
DEVICE="${GPU_DISTILL}" RESULTS_DIR="results/distill_only" bash scripts/run_distill_only.sh \
  > results/distill_only.log 2>&1 &
PID1=$!

echo "[2/3] elbo_grpo on ${GPU_ELBO}"
DEVICE="${GPU_ELBO}" RESULTS_DIR="results/elbo_grpo" bash scripts/run_elbo_grpo.sh \
  > results/elbo_grpo.log 2>&1 &
PID2=$!

echo "[3/3] sde_grpo on ${GPU_SDE}"
DEVICE="${GPU_SDE}" RESULTS_DIR="results/sde_grpo" bash scripts/run_sde_grpo.sh \
  > results/sde_grpo.log 2>&1 &
PID3=$!

echo "Launched PIDs: ${PID1} ${PID2} ${PID3}"
echo "Logs:"
echo "  tail -f results/distill_only.log"
echo "  tail -f results/elbo_grpo.log"
echo "  tail -f results/sde_grpo.log"

wait "${PID1}" "${PID2}" "${PID3}"
echo "All experiments finished."
