#!/bin/bash
#SBATCH --job-name=train_write_controller_online_rl_efp_endo18
#SBATCH --output=shell/local/offline_write_controller/logs/train_write_controller_online_rl_efp_endo18_%j.out
#SBATCH --error=shell/local/offline_write_controller/logs/train_write_controller_online_rl_efp_endo18_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -e
set -x

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CONDA_BIN="${CONDA_BIN:-$(command -v conda)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sam2}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset/VOS-Endovis18/train}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_s.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam2.1_hiera_s_endo18.pth}"
OUTPUT_CKPT="${OUTPUT_CKPT:-${REPO_ROOT}/artifacts/offline_write_controller/online_rl_efp_endo18_skip_write_v1.pt}"
OUTPUT_METRICS_JSON="${OUTPUT_METRICS_JSON:-${REPO_ROOT}/artifacts/offline_write_controller/online_rl_efp_endo18_skip_write_v1_metrics.json}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIMS_STR="${HIDDEN_DIMS:-16 8}"
VAL_RATIO="${VAL_RATIO:-0.25}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.98}"
REWARD_HORIZON="${REWARD_HORIZON:-8}"
WRITE_COST="${WRITE_COST:-0.01}"
ENTROPY_COEF="${ENTROPY_COEF:-1e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_OBJECTS_PER_VIDEO="${MAX_OBJECTS_PER_VIDEO:-0}"
THRESHOLD="${THRESHOLD:-0.5}"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_OUT="${LOG_DIR}/train_write_controller_online_rl_efp_endo18.out"
LOG_ERR="${LOG_DIR}/train_write_controller_online_rl_efp_endo18.err"

mkdir -p "${LOG_DIR}"
if [ -z "${CONDA_BIN}" ]; then
  echo "conda command not found in PATH" >&2
  exit 1
fi
if [ ! -f "${SAM2_CHECKPOINT}" ]; then
  echo "sam2 checkpoint not found: ${SAM2_CHECKPOINT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
exec >> "${LOG_OUT}" 2>> "${LOG_ERR}"

read -r -a HIDDEN_DIMS <<< "${HIDDEN_DIMS_STR}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" python tools/train_write_controller_online_rl.py \
  --dataset_roots "${DATASET_ROOT}" \
  --sam2_cfg "${SAM2_CFG}" \
  --sam2_checkpoint "${SAM2_CHECKPOINT}" \
  --output_ckpt "${OUTPUT_CKPT}" \
  --output_metrics_json "${OUTPUT_METRICS_JSON}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --hidden_dims "${HIDDEN_DIMS[@]}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
  --gamma "${GAMMA}" \
  --reward_horizon "${REWARD_HORIZON}" \
  --write_cost "${WRITE_COST}" \
  --entropy_coef "${ENTROPY_COEF}" \
  --grad_clip "${GRAD_CLIP}" \
  --max_videos "${MAX_VIDEOS}" \
  --max_objects_per_video "${MAX_OBJECTS_PER_VIDEO}" \
  --threshold "${THRESHOLD}" \
  --normalize_advantages
