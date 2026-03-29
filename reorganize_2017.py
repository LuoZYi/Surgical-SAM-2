# import os
# import re
# from pathlib import Path
# from PIL import Image
# import numpy as np

# # ================= 配置区域 =================
# # 请确保这两个路径指向正确的目录
# src_root = Path("/home/e/e0968951/fyp/endovis2017")
# dst_root = Path("/home/e/e0968951/fyp/Surgical-SAM-2/dataset/endovis2017_reorganized")

# # 调色板定义 (保持不变)
# palette = [0] * (256 * 3)
# base_colors = {
#     0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
#     4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255),
#     7: (255, 128, 0), 8: (128, 0, 255), 9: (0, 128, 255),
# }
# for idx, (r, g, b) in base_colors.items():
#     palette[3 * idx : 3 * idx + 3] = [r, g, b]

# # ===========================================

# def get_frame_num(filename):
#     """从文件名中提取数字，例如 'seq_2_frame225.bmp' -> '225'"""
#     match = re.search(r'frame(\d+)', filename)
#     return match.group(1) if match else None

# def process_label_with_palette(src_p, dst_p):
#     """读取灰度bmp，应用彩色调色板，保存为png"""
#     img = Image.open(src_p)
#     if img.mode != "L":
#         img = img.convert("L")
    
#     out = Image.new("P", img.size)
#     out.putdata(list(img.getdata()))
#     out.putpalette(palette)
#     out.save(dst_p)

# # 创建输出根目录
# dst_root.mkdir(parents=True, exist_ok=True)

# # 遍历 src_root 下的所有子文件夹
# for sub in sorted(src_root.iterdir()):
#     if not sub.is_dir():
#         continue
    
#     # 【关键修改】：如果是 train 文件夹，直接跳过，不处理
#     if sub.name.lower() == "train":
#         print(f"Skipping directory: {sub.name}")
#         continue

#     print(f"Processing sequence: {sub.name}...")
    
#     # 1. 定义并创建目标子目录
#     img_dst_dir = dst_root / "JPEGImages" / sub.name
#     ann_dst_dir = dst_root / "Annotations" / sub.name
#     vos_dst_dir = dst_root / "VOS" / "Annotations_vos_instrument" / sub.name
    
#     for d in [img_dst_dir, ann_dst_dir, vos_dst_dir]:
#         d.mkdir(parents=True, exist_ok=True)

#     # 2. 处理图片 (JPEGImages) - BMP 转 PNG
#     src_img_dir = sub / "image"
#     if src_img_dir.exists():
#         for bmp_path in src_img_dir.glob("*.bmp"):
#             frame_num = get_frame_num(bmp_path.name)
#             if frame_num:
#                 with Image.open(bmp_path) as img:
#                     # 正常转为 RGB 并保存为 PNG
#                     img.convert("RGB").save(img_dst_dir / f"{frame_num}.png")

#     # 3. 处理标签 (Annotations)
#     src_lab_dir = sub / "label"
#     processed_labels = [] 
    
#     if src_lab_dir.exists():
#         # 获取该序列下所有 label 并按帧号排序
#         all_label_paths = sorted(list(src_lab_dir.glob("*.bmp")), 
#                                  key=lambda x: int(get_frame_num(x.name)))
        
#         seen_classes = set([0])  # 0 是背景，默认已见过
        
#         for bmp_path in all_label_paths:
#             frame_num = get_frame_num(bmp_path.name)
#             target_ann_path = ann_dst_dir / f"{frame_num}.png"
            
#             # 1. 执行转换（彩色调色板）并保存到 Annotations
#             process_label_with_palette(bmp_path, target_ann_path)
#             processed_labels.append((int(frame_num), target_ann_path))
            
#             # 2. 【核心逻辑】：检查是否有新物体出现
#             with Image.open(bmp_path) as img_check:
#                 img_array = np.array(img_check)
#                 unique_ids = set(np.unique(img_array))
                
#                 # 找出当前帧里有，但之前没见过的类别
#                 new_classes = unique_ids - seen_classes
                
#                 if new_classes:
#                     # 如果发现了新物体，把这一帧也存入 VOS 文件夹作为推理 Prompt
#                     vos_target_path = vos_dst_dir / f"{frame_num}.png"
#                     with Image.open(target_ann_path) as img_to_copy:
#                         img_to_copy.save(vos_target_path)
                    
#                     print(f"  - New object(s) {new_classes} found at frame {frame_num}, saving as VOS prompt.")
                    
#                     # 更新已见过的类别
#                     seen_classes.update(new_classes)

#     # 4. 处理 VOS (仅保留每组序号最小的第一帧)
#     if processed_labels:
#         # 按帧号数值排序
#         processed_labels.sort(key=lambda x: x[0])
#         first_frame_num, first_frame_path = processed_labels[0]
        
#         # 复制第一张彩色标签到 VOS 目录
#         vos_target_path = vos_dst_dir / f"{first_frame_num}.png"
#         with Image.open(first_frame_path) as img:
#             img.save(vos_target_path)
#         print(f"  - Done. VOS first frame: {first_frame_num}.png")

# print("\nTask finished! 🚀 'train' folder was untouched.")

import os
import re
import numpy as np
from pathlib import Path
from PIL import Image

# ================= 配置区域 =================
# 请根据实际路径修改以下两个变量
src_root = Path("/home/e/e0968951/fyp/endovis2017")
dst_root = Path("/home/e/e0968951/fyp/Surgical-SAM-2/dataset/endovis2017_reorganized")

# 调色板定义 (0:背景, 1:红, 2:绿, 3:蓝 ...)
palette = [0] * (256 * 3)
base_colors = {
    0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
    4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255),
    7: (255, 128, 0), 8: (128, 0, 255), 9: (0, 128, 255),
}
for idx, (r, g, b) in base_colors.items():
    palette[3 * idx : 3 * idx + 3] = [r, g, b]

# ===========================================

def get_frame_num(filename):
    """提取数字编号，支持 frame000 或 frame_000"""
    match = re.search(r'frame_?(\d+)', filename, re.IGNORECASE)
    return match.group(1) if match else None

def process_label_with_palette(src_p, dst_p):
    """将灰度图转换为带调色板的索引模式 PNG"""
    img = Image.open(src_p).convert("L")
    out = Image.new("P", img.size)
    out.putdata(list(img.getdata()))
    out.putpalette(palette)
    out.save(dst_p)

# 创建输出根目录
dst_root.mkdir(parents=True, exist_ok=True)

# 遍历源目录下的所有子文件夹 (seq_1, seq_2, val1, val2 等)
for sub in sorted(src_root.iterdir()):
    if not sub.is_dir():
        continue
    
    # 跳过 train 文件夹
    if sub.name.lower() == "train":
        print(f"Skipping: {sub.name}")
        continue

    print(f"Processing sequence: {sub.name}...")
    
    # 1. 定义并创建目标子目录
    img_dst_dir = dst_root / "JPEGImages" / sub.name
    ann_dst_dir = dst_root / "Annotations" / sub.name
    vos_dst_dir = dst_root / "VOS" / "Annotations_vos_instrument" / sub.name
    
    for d in [img_dst_dir, ann_dst_dir, vos_dst_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. 准备数据列表并按帧号数值排序
    src_lab_dir = sub / "label"
    src_img_dir = sub / "image"
    
    if not src_lab_dir.exists():
        print(f"  Warning: No label folder in {sub.name}, skipping.")
        continue

    # 获取所有 label 文件并排序
    all_label_files = sorted(
        [f for f in src_lab_dir.glob("*.bmp")],
        key=lambda x: int(get_frame_num(x.name))
    )

    seen_classes = {0} # 背景默认已见过
    processed_info = [] # 记录 (帧整数ID, 字符串ID, 标签目标路径)

    # 3. 循环处理每一帧
    for bmp_path in all_label_files:
        raw_num = get_frame_num(bmp_path.name)
        if raw_num is None: continue
        
        # 统一命名规范：去掉领先的零 (000 -> 0)
        f_idx = int(raw_num)
        f_str = str(f_idx)
        
        target_ann_path = ann_dst_dir / f"{f_str}.png"
        target_img_path = img_dst_dir / f"{f_str}.png"
        
        # A. 处理图片 (Image)
        # 尝试寻找对应的原图 (假设原图也是 .bmp，如果不是可以改后缀)
        potential_img = src_img_dir / bmp_path.name 
        if potential_img.exists():
            with Image.open(potential_img) as img:
                img.convert("RGB").save(target_img_path)

        # B. 处理标签 (Annotation)
        process_label_with_palette(bmp_path, target_ann_path)
        processed_info.append((f_idx, f_str, target_ann_path))

        # C. VOS 逻辑：检测新出现的物体类别
        with Image.open(bmp_path) as img_check:
            img_array = np.array(img_check)
            unique_ids = set(np.unique(img_array))
            new_classes = unique_ids - seen_classes
            
            if new_classes:
                # 发现新物体，将该彩色标签存入 VOS 目录作为推理提示
                vos_target_path = vos_dst_dir / f"{f_str}.png"
                with Image.open(target_ann_path) as img_palette:
                    img_palette.save(vos_target_path)
                
                print(f"  - [VOS] New object {new_classes} at frame {f_str}")
                seen_classes.update(new_classes)

    # 4. 强制确保第一帧出现在 VOS 中 (如果之前没被存入的话)
    if processed_info:
        # 已按数值排过序，取第一个
        first_idx, first_str, first_path = processed_info[0]
        vos_first_path = vos_dst_dir / f"{first_str}.png"
        
        if not vos_first_path.exists():
            with Image.open(first_path) as img:
                img.save(vos_first_path)
            print(f"  - [VOS] Initial frame {first_str}.png saved.")

print("\nDone! 🚀 所有 Val 数据已重构。")
print(f"输出路径: {dst_root}")