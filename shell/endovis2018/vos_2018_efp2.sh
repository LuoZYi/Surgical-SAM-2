#!/bin/bash
#SBATCH --job-name=vos_2018_efp2
#SBATCH --output=/home/e/e0968951/fyp/Surgical-SAM-2/shell/endovis2018/logs/vos_2018_efp2_%j.out
#SBATCH --error=/home/e/e0968951/fyp/Surgical-SAM-2/shell/endovis2018/logs/vos_2018_efp2_%j.err
#SBATCH --time=03:00:00           
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

cd /home/e/e0968951/fyp/Surgical-SAM-2
export PYTHONPATH=/home/e/e0968951/fyp/Surgical-SAM-2:$PYTHONPATH

# 创建结果保存目录
mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/results/vos_2018_efp2

/home/e/e0968951/fyp/fyp_env/bin/python tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_s_endo18.pth \
  --output_mask_dir ./results/vos_2018_efp2 \
  --input_mask_dir ./dataset/VOS-Endovis18/valid/VOS/Annotations_vos_instrument \
  --base_video_dir ./dataset/VOS-Endovis18/valid/JPEGImages \
  --gt_root ./dataset/VOS-Endovis18/valid/Annotations \
  --gpu_id 0 \
  --video_list_file ./video_txt/2018_val.txt \
  --memory_prune_mode efp \
  --num_frame_to_prune 2
