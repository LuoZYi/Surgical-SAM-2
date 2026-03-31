#!/bin/bash
#SBATCH --job-name=vos_2017_rule
#SBATCH --output=/home/e/e0968951/fyp/Surgical-SAM-2/endovis2017/logs/vos_2017_rule_%j.out
#SBATCH --error=/home/e/e0968951/fyp/Surgical-SAM-2/endovis2017/logs/vos_2017_rule_%j.err
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
mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/results/vos_2017_rule

/home/e/e0968951/fyp/fyp_env/bin/python tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_s_endo18.pth \
  --output_mask_dir ./results/vos_2017_rule \
  --input_mask_dir ./dataset/VOS-Endovis17/valid/VOS/Annotations_vos_instrument \
  --base_video_dir ./dataset/VOS-Endovis17/valid/JPEGImages \
  --gt_root ./dataset/VOS-Endovis17/valid/Annotations \
  --gpu_id 0 \
  --video_list_file ./video_txt/2017_val.txt \
  --memory_prune_mode rule_based \
  --memory_score_mode cosine_only \
  --num_frame_to_prune 2 \
  --protect_conditioning_memories