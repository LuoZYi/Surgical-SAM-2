#!/bin/bash
#SBATCH --job-name=vos_2018_off
#SBATCH --output=/home/e/e0968951/fyp/Surgical-SAM-2/shell/logs/vos_2018_off_%j.out
#SBATCH --error=/home/e/e0968951/fyp/Surgical-SAM-2/shell/logs/vos_2018_off_%j.err
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

mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/results/vos_2018_off
mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/dataset/endovis_2018/val/rgb_Annotations
mkdir -p /home/e/e0968951/fyp/Surgical-SAM-2/dataset/endovis_2018/val/rgb_VOS

/home/e/e0968951/fyp/fyp_env/bin/python tools/convert.py --src_root "/home/e/e0968951/fyp/Surgical-SAM-2/dataset/endovis_2018/val/Annotations" --dst_root "/home/e/e0968951/fyp/Surgical-SAM-2/dataset/endovis_2018/val/rgb_Annotations"


/home/e/e0968951/fyp/fyp_env/bin/python tools/convert.py --src_root "/home/e/e0968951/fyp/Surgical-SAM-2/dataset/endovis_2018/val/VOS" --dst_root "/home/e/e0968951/fyp/Surgical-SAM-2/dataset/endovis_2018/val/rgb_VOS"


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
  --output_mask_dir ./results/vos_2018_off \
  --input_mask_dir ./dataset/endovis_2018/val/VOS/Annotations_vos \
  --base_video_dir ./dataset/endovis_2018/val/JPEGImages \
  --gt_root ./dataset/endovis_2018/val/Annotations \
  --gpu_id 0 \
  --video_list_file ./2018.txt \
  --memory_prune_mode off \
  --num_frame_to_prune 0
