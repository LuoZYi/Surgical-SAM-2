#!/bin/bash
#SBATCH --job-name=train_rl_efp_endo18_2018
#SBATCH --output=logs/train_write_controller_online_rl_efp_endo18_%j.out
#SBATCH --error=logs/train_write_controller_online_rl_efp_endo18_%j.err
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
REPO_ROOT=""
for CANDIDATE in \
  "${SLURM_SUBMIT_DIR:-}" \
  "${SLURM_SUBMIT_DIR:-}/.." \
  "${SLURM_SUBMIT_DIR:-}/../.." \
  "$(pwd)" \
  "$(pwd)/.." \
  "$(pwd)/../.."
do
  if [ -n "${CANDIDATE}" ] && [ -f "${CANDIDATE}/tools/train_write_controller_online_rl.py" ]; then
    REPO_ROOT=$(cd "${CANDIDATE}" && pwd)
    break
  fi
done
if [ -z "${REPO_ROOT}" ]; then
  echo "failed to locate repo root; please submit the job from inside the Surgical-SAM-2 repo" >&2
  exit 1
fi

VENV_ACTIVATE="${VENV_ACTIVATE:-/home/e/e0968951/fyp/fyp_env/bin/activate}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset/VOS-Endovis18/train}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_s.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam2.1_hiera_s_endo18.pth}"
OUTPUT_CKPT="${OUTPUT_CKPT:-${REPO_ROOT}/artifacts/offline_write_controller/online_rl_efp_endo18_2018_skip_write_v5.pt}"
OUTPUT_METRICS_JSON="${OUTPUT_METRICS_JSON:-${REPO_ROOT}/artifacts/offline_write_controller/online_rl_efp_endo18_2018_skip_write_v5_metrics.json}"

EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIMS_STR="${HIDDEN_DIMS:-16 8}"
VAL_RATIO="${VAL_RATIO:-0.25}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.98}"
REWARD_HORIZON="${REWARD_HORIZON:-8}"
WRITE_COST="${WRITE_COST:-0.03}"
ENTROPY_COEF="${ENTROPY_COEF:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_OBJECTS_PER_VIDEO="${MAX_OBJECTS_PER_VIDEO:-0}"
THRESHOLD="${THRESHOLD:-0.5}"

LOG_DIR="${REPO_ROOT}/shell/endovis2018/logs"

mkdir -p "${LOG_DIR}"
if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "python venv activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
if [ ! -f "${SAM2_CHECKPOINT}" ]; then
  echo "sam2 checkpoint not found: ${SAM2_CHECKPOINT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
source "${VENV_ACTIVATE}"
read -r -a HIDDEN_DIMS <<< "${HIDDEN_DIMS_STR}"

python - <<'PY'
import importlib
import sys

required = [
    "numpy",
    "torch",
    "hydra",
    "omegaconf",
    "PIL",
    "cv2",
    "skimage",
    "tqdm",
    "sympy",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

if missing:
    print("Missing Python packages in venv:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)

print("Python dependency check passed.")
PY

python tools/train_write_controller_online_rl.py \
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
