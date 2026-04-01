from pathlib import Path
from PIL import Image

# ========= 改这里 =========
root_dir = Path(r"/home/e/e0968951/fyp/Surgical-SAM-2/dataset/VOS-Endovis17")
target_h = 1024
target_w = 1280
overwrite = True   # True = 直接覆盖原图；False = 另存为 *_cropped.png
# =========================

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

count = 0

for jpeg_dir in root_dir.rglob("JPEGImages"):
    if not jpeg_dir.is_dir():
        continue

    print(f"\n[PROCESS JPEGImages DIR] {jpeg_dir}")

    for img_path in jpeg_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix not in valid_exts:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            cropped = center_crop_image(img, target_h, target_w)

            if overwrite:
                cropped.save(img_path)
                print(f"overwritten: {img_path}")
            else:
                new_path = img_path.with_name(img_path.stem + "_cropped" + img_path.suffix)
                cropped.save(new_path)
                print(f"saved: {new_path}")

            count += 1

        except Exception as e:
            print(f"[SKIP] {img_path} -> {e}")

print(f"\nDone. Total processed images: {count}")