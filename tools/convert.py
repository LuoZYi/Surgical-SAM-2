from pathlib import Path
from PIL import Image
import shutil

# ===== 改这里 =====
src_root = Path("/home/e/e0968951/fyp/Surgical-SAM-2/dataset/VOS-Endovis17")
dst_root = Path("/home/e/e0968951/fyp/Surgical-SAM-2/dataset/VOS-Endovis17_new")
target_h = 1024
target_w = 1280
# =================

valid_exts = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}

def center_crop_image(img, target_h, target_w):
    w, h = img.size  # PIL: (W, H)
    if h < target_h or w < target_w:
        raise ValueError(f"image too small: {(h, w)} < {(target_h, target_w)}")

    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return img.crop((left, top, right, bottom))

count_ok = 0
count_skip = 0

for jpeg_dir in src_root.rglob("JPEGImages"):
    if not jpeg_dir.is_dir():
        continue

    rel_jpeg_dir = jpeg_dir.relative_to(src_root)
    out_jpeg_dir = dst_root / rel_jpeg_dir
    out_jpeg_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[PROCESS JPEGImages DIR] {jpeg_dir}")

    for img_path in jpeg_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix not in valid_exts:
            continue

        rel_img_path = img_path.relative_to(src_root)
        out_img_path = dst_root / rel_img_path
        out_img_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if img_path.stat().st_size == 0:
                raise ValueError("empty file")

            with Image.open(img_path) as im:
                im.verify()

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            if (h, w) == (target_h, target_w):
                shutil.copy2(img_path, out_img_path)
            else:
                cropped = center_crop_image(img, target_h, target_w)
                cropped.save(out_img_path)

            count_ok += 1

        except Exception as e:
            print(f"[SKIP] {img_path} -> {e}")
            count_skip += 1

print(f"\nDone. OK={count_ok}, SKIP={count_skip}")