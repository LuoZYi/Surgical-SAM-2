#!/bin/bash
#SBATCH --job-name=vos_seq2_rb_cosine
#SBATCH --output=/home/e/e0968951/fyp/Surgical-SAM-2/shell/logs/vos_seq2_rb_cosine_%j.out
#SBATCH --error=/home/e/e0968951/fyp/Surgical-SAM-2/shell/logs/vos_seq2_rb_cosine_%j.err
#SBATCH --time=03:00:00           # 建议和你salloc的时间保持一致
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G                # 对应你salloc中的256G
#SBATCH --gres=gpu:h100-47:1      # 关键点：指定GPU型号和数量
#SBATCH --constraint=xgpi         # 按照你的说法，这里改为 xgpi

set -e
set -x

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/e/e0968951/fyp/Surgical-SAM-2
export PYTHONPATH=/home/e/e0968951/fyp/Surgical-SAM-2:$PYTHONPATH

mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/results/debug_seq2_rb_cosine



#/home/e/e0968951/fyp/fyp_env/bin/python tools/vos_inference.py \
#  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
#  --sam2_checkpoint ./checkpoints/sam2.1_hiera_s_endo18.pth \
#  --output_mask_dir ./results/debug_seq2_off \
#  --input_mask_dir ./dataset/endovis18_debug/train/VOS/Annotations_vos_instrument \
#  --base_video_dir ./dataset/endovis18_debug/train/JPEGImages \
#  --gt_root ./dataset/endovis18_debug/train/Annotations \
#  --gpu_id 0 \
#  --video_list_file ./one_video.txt


/home/e/e0968951/fyp/fyp_env/bin/python tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_s_endo18.pth \
  --output_mask_dir ./results/debug_seq2_rb_cosine \
  --input_mask_dir ./dataset/endovis18_debug/train/VOS/Annotations_vos_instrument \
  --base_video_dir ./dataset/endovis18_debug/train/JPEGImages \
  --gt_root ./dataset/endovis18_debug/train/Annotations \
  --gpu_id 0 \
  --video_list_file ./one_video.txt \
  --memory_prune_mode rule_based \
  --memory_score_mode cosine_only \
  --num_frame_to_prune 2 \
  --protect_conditioning_memories