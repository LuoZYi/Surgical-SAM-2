from pathlib import Path

import numpy as np
from PIL import Image


# =========================
# Hardcoded label spec
# =========================
LABEL_SPEC = [
    {"name": "background-tissue",   "color": [0, 0, 0],       "classid": 0},
    {"name": "instrument-shaft",    "color": [0, 255, 0],     "classid": 1},
    {"name": "instrument-clasper",  "color": [0, 255, 255],   "classid": 2},
    {"name": "instrument-wrist",    "color": [125, 255, 12],  "classid": 3},
    {"name": "kidney-parenchyma",   "color": [255, 55, 0],    "classid": 4},
    {"name": "covered-kidney",      "color": [24, 55, 125],   "classid": 5},
    {"name": "thread",              "color": [187, 155, 25],  "classid": 6},
    {"name": "clamps",              "color": [0, 255, 125],   "classid": 7},
    {"name": "suturing-needle",     "color": [255, 255, 125], "classid": 8},
    {"name": "suction-instrument",  "color": [123, 15, 175],  "classid": 9},
    {"name": "small-intestine",     "color": [124, 155, 5],   "classid": 10},
    {"name": "ultrasound-probe",    "color": [12, 255, 141],  "classid": 11},
]


def build_mappings():
    color_to_id = {}
    palette = [0] * 768  # 256 * 3

    for item in LABEL_SPEC:
        color = tuple(int(x) for x in item["color"])
        classid = int(item["classid"])
        color_to_id[color] = classid

        if classid < 256:
            palette[classid * 3 + 0] = color[0]
            palette[classid * 3 + 1] = color[1]
            palette[classid * 3 + 2] = color[2]

    return color_to_id, palette


def rgba_or_rgb_to_label(arr, color_to_id, src_path):
    if arr.ndim == 2:
        return arr.astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]  # drop alpha

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unsupported image shape {arr.shape} in {src_path}")

    label = np.zeros(arr.shape[:2], dtype=np.uint8)

    unique_colors = np.unique(arr.reshape(-1, 3), axis=0)
    for color in unique_colors:
        color_t = tuple(int(x) for x in color.tolist())
        if color_t not in color_to_id:
            raise ValueError(
                f"Color {color_t} in {src_path} not found in hardcoded LABEL_SPEC."
            )
        classid = color_to_id[color_t]
        label[np.all(arr == color, axis=-1)] = classid

    return label


def convert_one_mask(src_path: Path, dst_path: Path, color_to_id: dict, palette: list):
    img = Image.open(src_path)
    arr = np.array(img).astype(np.uint8)

    label = rgba_or_rgb_to_label(arr, color_to_id, src_path)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    out = Image.fromarray(label, mode="P")
    out.putpalette(palette)
    out.save(dst_path)


def convert_tree(src_root: str, dst_root: str):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    color_to_id, palette = build_mappings()

    png_files = sorted(src_root.rglob("*.png"))
    print(f"Found {len(png_files)} PNG files under {src_root}")

    for i, src_path in enumerate(png_files, 1):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        convert_one_mask(src_path, dst_path, color_to_id, palette)

        if i % 100 == 0 or i == len(png_files):
            print(f"[{i}/{len(png_files)}] converted")

    print(f"Done. Converted masks saved to: {dst_root}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--dst_root", type=str, required=True)
    args = parser.parse_args()

    convert_tree(args.src_root, args.dst_root)
