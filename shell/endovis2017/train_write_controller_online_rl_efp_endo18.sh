#!/bin/bash
#SBATCH --job-name=train_rl_efp_endo18_2017
#SBATCH --output=shell/endovis2017/logs/train_write_controller_online_rl_efp_endo18_%j.out
#SBATCH --error=shell/endovis2017/logs/train_write_controller_online_rl_efp_endo18_%j.err
#SBATCH --time=24:00:00        
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G                
#SBATCH --gres=gpu:h100-47:1      
#SBATCH --constraint=xgpi  

set -e
set -x

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CONDA_BIN=$(command -v conda)
CONDA_ENV_NAME="sam2"

DATASET_ROOT="${REPO_ROOT}/dataset/VOS-Endovis17/train"
SAM2_CFG="configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT="${REPO_ROOT}/checkpoints/sam2.1_hiera_s_endo18.pth"
OUTPUT_CKPT="${REPO_ROOT}/artifacts/offline_write_controller/online_rl_efp_endo18_2017_skip_write_v1.pt"
OUTPUT_METRICS_JSON="${REPO_ROOT}/artifacts/offline_write_controller/online_rl_efp_endo18_2017_skip_write_v1_metrics.json"

EPOCHS=5
LR=5e-4
WEIGHT_DECAY=1e-4
HIDDEN_DIMS=(16 8)
VAL_RATIO=0.25
SEED=42
GAMMA=0.98
REWARD_HORIZON=8
WRITE_COST=0.01
ENTROPY_COEF=1e-3
GRAD_CLIP=1.0
MAX_VIDEOS=0
MAX_OBJECTS_PER_VIDEO=0
THRESHOLD=0.5

LOG_DIR="${SCRIPT_DIR}/logs"

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
