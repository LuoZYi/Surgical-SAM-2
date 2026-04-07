#!/bin/bash
#SBATCH --job-name=vos_2018_sam2
#SBATCH --output=shell/local/endovis2018/logs/vos_2018_sam2_%j.out
#SBATCH --error=shell/local/endovis2018/logs/vos_2018_sam2_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
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
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_OUT="${LOG_DIR}/vos_2018_sam2.out"
LOG_ERR="${LOG_DIR}/vos_2018_sam2.err"

mkdir -p "${LOG_DIR}"
if [ -z "${CONDA_BIN}" ]; then
  echo "conda command not found in PATH" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
exec >> "${LOG_OUT}" 2>> "${LOG_ERR}"

mkdir -p "${RESULTS_ROOT}/vos_2018_sam2"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" python tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_small.pt \
  --output_mask_dir "${RESULTS_ROOT}/vos_2018_sam2" \
  --input_mask_dir "${DATASET_ROOT}/VOS-Endovis18/valid/VOS/Annotations_vos_instrument" \
  --base_video_dir "${DATASET_ROOT}/VOS-Endovis18/valid/JPEGImages" \
  --gt_root "${DATASET_ROOT}/VOS-Endovis18/valid/Annotations" \
  --gpu_id 0 \
  --video_list_file ./video_txt/2018_val.txt \
  --memory_prune_mode off \
  --num_frame_to_prune 0
