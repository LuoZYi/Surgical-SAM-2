#!/bin/bash
#SBATCH --job-name=train_write_controller_rl_skip_write_v1
#SBATCH --output=shell/local/offline_write_controller/logs/train_write_controller_rl_skip_write_v1_%j.out
#SBATCH --error=shell/local/offline_write_controller/logs/train_write_controller_rl_skip_write_v1_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
CSV_PATHS_STR="${CSV_PATHS:-${REPO_ROOT}/artifacts/offline_write_controller/oracle_train_1718.csv}"
OUTPUT_CKPT="${OUTPUT_CKPT:-${REPO_ROOT}/artifacts/offline_write_controller/rl_skip_write_v1.pt}"
OUTPUT_METRICS_JSON="${OUTPUT_METRICS_JSON:-${REPO_ROOT}/artifacts/offline_write_controller/rl_skip_write_v1_metrics.json}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIMS_STR="${HIDDEN_DIMS:-16 8}"
VAL_RATIO="${VAL_RATIO:-0.25}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.98}"
WRITE_COST="${WRITE_COST:-0.01}"
SKIP_GAIN_SCALE="${SKIP_GAIN_SCALE:-1.0}"
ENTROPY_COEF="${ENTROPY_COEF:-1e-3}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_OUT="${LOG_DIR}/train_write_controller_rl_skip_write_v1.out"
LOG_ERR="${LOG_DIR}/train_write_controller_rl_skip_write_v1.err"

mkdir -p "${LOG_DIR}"
if [ -z "${CONDA_BIN}" ]; then
  echo "conda command not found in PATH" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
exec >> "${LOG_OUT}" 2>> "${LOG_ERR}"

read -r -a CSV_PATHS <<< "${CSV_PATHS_STR}"
read -r -a HIDDEN_DIMS <<< "${HIDDEN_DIMS_STR}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" python tools/train_write_controller_rl.py \
  --csv_paths "${CSV_PATHS[@]}" \
  --output_ckpt "${OUTPUT_CKPT}" \
  --output_metrics_json "${OUTPUT_METRICS_JSON}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --hidden_dims "${HIDDEN_DIMS[@]}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
  --gamma "${GAMMA}" \
  --write_cost "${WRITE_COST}" \
  --skip_gain_scale "${SKIP_GAIN_SCALE}" \
  --entropy_coef "${ENTROPY_COEF}" \
  --grad_clip "${GRAD_CLIP}" \
  --normalize_returns
