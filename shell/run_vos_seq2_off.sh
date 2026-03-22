#!/bin/bash
#SBATCH --job-name=vos_seq2_off
#SBATCH --output=/home/e/e0968951/fyp/Surgical-SAM-2/shell/logs/vos_seq2_off_%j.out
#SBATCH --error=/home/e/e0968951/fyp/Surgical-SAM-2/shell/logs/vos_seq2_off_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=xgpe

set -e
set -x

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/e/e0968951/fyp/Surgical-SAM-2
export PYTHONPATH=/home/e/e0968951/fyp/Surgical-SAM-2:$PYTHONPATH

mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/results/debug_seq2_off
mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/shell/logs

echo "===== SLURM DEBUG ====="
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "which python: $(which python)"
/usr/bin/nvidia-smi || true

/home/e/e0968951/fyp/fyp_env/bin/python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

/home/e/e0968951/fyp/fyp_env/bin/python tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_s_endo18.pth \
  --output_mask_dir ./results/debug_seq2_off \
  --input_mask_dir ./dataset/endovis18_debug/train/VOS/Annotations_vos_instrument \
  --base_video_dir ./dataset/endovis18_debug/train/JPEGImages \
  --gt_root ./dataset/endovis18_debug/train/Annotations \
  --video_list_file ./one_video.txt \
