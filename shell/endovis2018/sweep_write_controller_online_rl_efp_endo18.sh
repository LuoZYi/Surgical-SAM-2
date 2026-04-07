#!/bin/bash
#SBATCH --job-name=sweep_online_rl_endo18
#SBATCH --output=shell/endovis2018/logs/sweep_online_rl_efp_endo18_%j.out
#SBATCH --error=shell/endovis2018/logs/sweep_online_rl_efp_endo18_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100-47:1
#SBATCH --constraint=xgpi

# Simple train+inference sweep for the Endovis2018 online RL write controller.

set -euo pipefail
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
  echo "failed to locate repo root; please run this script from inside the Surgical-SAM-2 repo" >&2
  exit 1
fi

TRAIN_SCRIPT="${REPO_ROOT}/shell/endovis2018/train_write_controller_online_rl_efp_endo18.sh"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/e/e0968951/fyp/fyp_env/bin/activate}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_s.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam2.1_hiera_s_endo18.pth}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-${REPO_ROOT}/dataset/VOS-Endovis18/train}"
VALID_BASE_VIDEO_DIR="${VALID_BASE_VIDEO_DIR:-${REPO_ROOT}/dataset/VOS-Endovis18/valid/JPEGImages}"
VALID_INPUT_MASK_DIR="${VALID_INPUT_MASK_DIR:-${REPO_ROOT}/dataset/VOS-Endovis18/valid/VOS/Annotations_vos_instrument}"
VALID_GT_ROOT="${VALID_GT_ROOT:-${REPO_ROOT}/dataset/VOS-Endovis18/valid/Annotations}"
VALID_VIDEO_LIST_FILE="${VALID_VIDEO_LIST_FILE:-${REPO_ROOT}/video_txt/2018_val.txt}"
SWEEP_TAG="${SWEEP_TAG:-v1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/offline_write_controller/sweeps/online_rl_efp_endo18_2018_${SWEEP_TAG}}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/sweeps/online_rl_efp_endo18_2018_${SWEEP_TAG}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/shell/endovis2018/logs/sweep_online_rl_efp_endo18_2018_${SWEEP_TAG}}"
SUMMARY_CSV="${SUMMARY_CSV:-${ARTIFACT_ROOT}/summary.csv}"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "training script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "python venv activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
if [ ! -f "${SAM2_CHECKPOINT}" ]; then
  echo "sam2 checkpoint not found: ${SAM2_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_ROOT}" "${RESULTS_ROOT}" "${LOG_ROOT}"

cat > "${SUMMARY_CSV}" <<'EOF'
tag,epochs,lr,write_cost,threshold,train_best_val_dice,train_best_val_return,train_best_val_write_rate,global_jf,global_j,global_f,global_dice,ckpt,metrics_json,results_csv
EOF

run_one() {
  local epochs="$1"
  local lr="$2"
  local write_cost="$3"
  local threshold="$4"

  local safe_lr="${lr//./p}"
  local safe_cost="${write_cost//./p}"
  local safe_threshold="${threshold//./p}"
  local tag="e${epochs}_lr${safe_lr}_wc${safe_cost}_th${safe_threshold}"

  local output_ckpt="${ARTIFACT_ROOT}/${tag}.pt"
  local output_metrics_json="${ARTIFACT_ROOT}/${tag}_metrics.json"
  local output_mask_dir="${RESULTS_ROOT}/${tag}"
  local train_log="${LOG_ROOT}/${tag}_train.out"
  local infer_log="${LOG_ROOT}/${tag}_infer.out"
  local results_csv="${output_mask_dir}/results.csv"

  echo "=== TRAIN ${tag} ===" | tee "${train_log}"
  (
    cd "${REPO_ROOT}"
    DATASET_ROOT="${TRAIN_DATASET_ROOT}" \
    SAM2_CFG="${SAM2_CFG}" \
    SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" \
    OUTPUT_CKPT="${output_ckpt}" \
    OUTPUT_METRICS_JSON="${output_metrics_json}" \
    EPOCHS="${epochs}" \
    LR="${lr}" \
    WRITE_COST="${write_cost}" \
    THRESHOLD="${threshold}" \
    VENV_ACTIVATE="${VENV_ACTIVATE}" \
    bash "${TRAIN_SCRIPT}"
  ) >> "${train_log}" 2>&1

  if [ ! -f "${output_ckpt}" ]; then
    echo "missing controller checkpoint: ${output_ckpt}" >&2
    exit 1
  fi
  if [ ! -f "${output_metrics_json}" ]; then
    echo "missing metrics json: ${output_metrics_json}" >&2
    exit 1
  fi

  mkdir -p "${output_mask_dir}"
  echo "=== INFER ${tag} ===" | tee "${infer_log}"
  (
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    source "${VENV_ACTIVATE}"
    python tools/hydra_infer.py \
      dataset=endovis2018 \
      memory_policy=efp_learned_skip_write_v1 \
      sam2_cfg="${SAM2_CFG}" \
      sam2_checkpoint="${SAM2_CHECKPOINT}" \
      model.memory_write_controller_path="${output_ckpt}" \
      output_mask_dir="${output_mask_dir}" \
      dataset.base_video_dir="${VALID_BASE_VIDEO_DIR}" \
      dataset.input_mask_dir="${VALID_INPUT_MASK_DIR}" \
      dataset.gt_root="${VALID_GT_ROOT}" \
      dataset.video_list_file="${VALID_VIDEO_LIST_FILE}"
  ) >> "${infer_log}" 2>&1

  if [ ! -f "${results_csv}" ]; then
    echo "missing inference results csv: ${results_csv}" >&2
    exit 1
  fi

  python - <<'PY' "${output_metrics_json}" "${results_csv}" "${SUMMARY_CSV}" "${tag}" "${epochs}" "${lr}" "${write_cost}" "${threshold}" "${output_ckpt}"
import csv
import json
import sys

metrics_json, results_csv, summary_csv, tag, epochs, lr, write_cost, threshold, ckpt = sys.argv[1:]

with open(metrics_json, "r", encoding="utf-8") as f:
    metrics = json.load(f)

global_jf = global_j = global_f = global_dice = ""
with open(results_csv, "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line.startswith("Global score"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            global_jf, global_j, global_f, global_dice = parts[2:6]

if global_dice == "":
    raise RuntimeError(f"Failed to parse Global score from {results_csv}")

row = [
    tag,
    epochs,
    lr,
    write_cost,
    threshold,
    metrics.get("best_val_dice", ""),
    metrics.get("best_val_return", ""),
    metrics.get("best_val_write_rate", ""),
    global_jf,
    global_j,
    global_f,
    global_dice,
    ckpt,
    metrics_json,
    results_csv,
]

with open(summary_csv, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(row)
PY
}

cd "${REPO_ROOT}"

# epochs lr write_cost threshold
CONFIGS=(
  "20 2e-4 0.01 0.50"
  "20 2e-4 0.02 0.50"
  "20 2e-4 0.02 0.55"
  "20 1e-4 0.02 0.50"
  "30 1e-4 0.02 0.50"
)

for config in "${CONFIGS[@]}"; do
  run_one ${config}
done

python - <<'PY' "${SUMMARY_CSV}"
import csv
import sys

summary_csv = sys.argv[1]
with open(summary_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda row: float(row["global_dice"]), reverse=True)
print("Top configs by global_dice:")
for row in rows[:5]:
    print(
        f"{row['tag']}: global_dice={row['global_dice']} "
        f"train_best_val_dice={row['train_best_val_dice']} "
        f"train_best_val_write_rate={row['train_best_val_write_rate']}"
    )
print(f"saved summary to {summary_csv}")
PY
