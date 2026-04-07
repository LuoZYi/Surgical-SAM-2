#!/bin/bash

set -euo pipefail

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
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/tuning/learned_adaptive_controller_v2_sweep}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/shell/local/tuning_logs/learned_adaptive_controller_v2_sweep}"
CONTROLLER_CKPT="${CONTROLLER_CKPT:-${REPO_ROOT}/artifacts/offline_write_controller/learned_adaptive_controller_v2.pt}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULTS_ROOT}/summary.csv}"

if [ -z "${CONDA_BIN}" ]; then
  echo "conda command not found in PATH" >&2
  exit 1
fi

if [ ! -f "${CONTROLLER_CKPT}" ]; then
  echo "controller checkpoint not found: ${CONTROLLER_CKPT}" >&2
  exit 1
fi

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

if [ ! -f "${SUMMARY_CSV}" ]; then
  cat > "${SUMMARY_CSV}" <<'EOF'
dataset,tag,threshold,low,high,jf,j,f,dice,results_csv
EOF
fi

run_one() {
  local dataset="$1"
  local low="$2"
  local high="$3"
  local threshold="$4"
  local tag="$5"

  local dataset_base_video_dir
  local dataset_input_mask_dir
  local dataset_gt_root
  local dataset_video_list_file

  case "${dataset}" in
    endovis2017)
      dataset_base_video_dir="${DATASET_ROOT}/VOS-Endovis17/valid/JPEGImages"
      dataset_input_mask_dir="${DATASET_ROOT}/VOS-Endovis17/valid/VOS/Annotations_vos_instrument"
      dataset_gt_root="${DATASET_ROOT}/VOS-Endovis17/valid/Annotations"
      dataset_video_list_file="./video_txt/2017_val_full.txt"
      ;;
    endovis2018)
      dataset_base_video_dir="${DATASET_ROOT}/VOS-Endovis18/valid/JPEGImages"
      dataset_input_mask_dir="${DATASET_ROOT}/VOS-Endovis18/valid/VOS/Annotations_vos_instrument"
      dataset_gt_root="${DATASET_ROOT}/VOS-Endovis18/valid/Annotations"
      dataset_video_list_file="./video_txt/2018_val.txt"
      ;;
    *)
      echo "unsupported dataset: ${dataset}" >&2
      exit 1
      ;;
  esac

  local output_dir="${RESULTS_ROOT}/${dataset}_${tag}"
  local log_out="${LOG_ROOT}/${dataset}_${tag}.out"
  local log_err="${LOG_ROOT}/${dataset}_${tag}.err"
  local results_csv="${output_dir}/results.csv"

  mkdir -p "${output_dir}"

  echo "=== RUN ${dataset} ${tag} threshold=${threshold} low=${low} high=${high} ===" | tee "${log_out}"

  (
    cd "${REPO_ROOT}"
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" python tools/hydra_infer.py \
      dataset="${dataset}" \
      memory_policy=efp_learned_adaptive_controller_v2 \
      model.memory_write_controller_path="${CONTROLLER_CKPT}" \
      model.memory_write_controller_threshold="${threshold}" \
      model.memory_write_controller_low_threshold="${low}" \
      model.memory_write_controller_high_threshold="${high}" \
      output_mask_dir="${output_dir}" \
      dataset.base_video_dir="${dataset_base_video_dir}" \
      dataset.input_mask_dir="${dataset_input_mask_dir}" \
      dataset.gt_root="${dataset_gt_root}" \
      dataset.video_list_file="${dataset_video_list_file}"
  ) >> "${log_out}" 2>> "${log_err}"

  if [ ! -f "${results_csv}" ]; then
    echo "missing results csv: ${results_csv}" >&2
    exit 1
  fi

  local jf
  local j
  local f
  local dice
  jf=$(awk -F',' '/^Global score/ {gsub(/ /,"",$3); print $3}' "${results_csv}")
  j=$(awk -F',' '/^Global score/ {gsub(/ /,"",$4); print $4}' "${results_csv}")
  f=$(awk -F',' '/^Global score/ {gsub(/ /,"",$5); print $5}' "${results_csv}")
  dice=$(awk -F',' '/^Global score/ {gsub(/ /,"",$6); print $6}' "${results_csv}")

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${dataset}" "${tag}" "${threshold}" "${low}" "${high}" "${jf}" "${j}" "${f}" "${dice}" "${results_csv}" \
    >> "${SUMMARY_CSV}"

  echo "=== DONE ${dataset} ${tag} J&F=${jf} ===" | tee -a "${log_out}"
}

cd "${REPO_ROOT}"

run_one "endovis2017" "0.40" "0.60" "0.500" "low040_high060"
run_one "endovis2018" "0.40" "0.60" "0.500" "low040_high060"

run_one "endovis2017" "0.40" "0.65" "0.525" "low040_high065"
run_one "endovis2018" "0.40" "0.65" "0.525" "low040_high065"

run_one "endovis2017" "0.40" "0.70" "0.550" "low040_high070"
run_one "endovis2018" "0.40" "0.70" "0.550" "low040_high070"

echo "saved summary to ${SUMMARY_CSV}"
