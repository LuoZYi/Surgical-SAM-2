#!/bin/bash
#SBATCH --job-name=vos_seq2
#SBATCH --output=/home/e/e0968951/fyp/Surgical-SAM-2/shell/logs/vos_seq2_%j.out
#SBATCH --error=/home/e/e0968951/fyp/Surgical-SAM-2/shell/logs/vos_seq2_%j.err
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

mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/results/debug_seq2

/home/e/e0968951/fyp/fyp_env/bin/python tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_s_endo18.pth \
  --output_mask_dir ./results/debug_seq2_off \
  --input_mask_dir ./dataset/endovis18_debug/train/VOS/Annotations_vos_instrument \
  --base_video_dir ./dataset/endovis18_debug/train/JPEGImages \
  --gt_root ./dataset/endovis18_debug/train/Annotations \
  --gpu_id 0 \
  --video_list_file ./one_video.txt