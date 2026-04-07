#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.vos_inference import load_masks_from_dir


def sorted_mask_frame_names(video_mask_dir: str) -> List[str]:
    frame_names = [
        os.path.splitext(name)[0]
        for name in os.listdir(video_mask_dir)
        if os.path.splitext(name)[-1].lower() == ".png"
    ]
    frame_names.sort(key=lambda name: int(name))
    return frame_names


def choose_positive_points(mask: np.ndarray, num_points: int) -> List[List[float]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Cannot sample prompts from an empty mask")

    coords_xy = coords[:, [1, 0]].astype(np.float32)
    if len(coords_xy) > 5000:
        step = int(math.ceil(len(coords_xy) / 5000))
        coords_xy = coords_xy[::step]

    centroid = coords_xy.mean(axis=0, keepdims=True)
    sq_dist_to_centroid = np.sum((coords_xy - centroid) ** 2, axis=1)
    selected = [int(np.argmin(sq_dist_to_centroid))]

    while len(selected) < min(num_points, len(coords_xy)):
        chosen = coords_xy[selected]
        sq_dist = np.sum((coords_xy[:, None, :] - chosen[None, :, :]) ** 2, axis=2)
        min_sq_dist = np.min(sq_dist, axis=1)
        min_sq_dist[selected] = -1.0
        selected.append(int(np.argmax(min_sq_dist)))

    return [[float(coords_xy[idx, 0]), float(coords_xy[idx, 1])] for idx in selected]


def collect_prompt_payload(
    input_mask_dir: str,
    video_names: List[str],
    num_points: int,
) -> Dict[str, object]:
    videos_payload: Dict[str, object] = {}
    for video_name in video_names:
        video_mask_dir = os.path.join(input_mask_dir, video_name)
        frame_names = sorted_mask_frame_names(video_mask_dir)
        first_masks_per_object: Dict[int, Dict[str, object]] = defaultdict(dict)
        for frame_name in frame_names:
            per_obj_mask, _ = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_name,
                per_obj_png_file=False,
                allow_missing=False,
            )
            for object_id, object_mask in per_obj_mask.items():
                if object_id in first_masks_per_object:
                    continue
                if not np.any(object_mask):
                    continue
                first_masks_per_object[object_id] = {
                    "frame_name": frame_name,
                    "points": choose_positive_points(object_mask, num_points=num_points),
                    "labels": [1] * min(num_points, int(np.count_nonzero(object_mask))),
                }

        videos_payload[video_name] = {
            "objects": {
                str(object_id): first_masks_per_object[object_id]
                for object_id in sorted(first_masks_per_object)
            }
        }
    return {
        "metadata": {
            "num_points": int(num_points),
            "sampling_strategy": "centroid_plus_greedy_farthest_positive",
        },
        "videos": videos_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mask_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--num_points", type=int, required=True)
    parser.add_argument("--video_list_file", type=str, default=None)
    args = parser.parse_args()

    if args.video_list_file:
        with open(args.video_list_file, "r") as f:
            video_names = [line.strip() for line in f if line.strip()]
    else:
        video_names = sorted(
            [
                name
                for name in os.listdir(args.input_mask_dir)
                if os.path.isdir(os.path.join(args.input_mask_dir, name))
            ]
        )

    payload = collect_prompt_payload(
        input_mask_dir=os.path.abspath(args.input_mask_dir),
        video_names=video_names,
        num_points=int(args.num_points),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Saved {args.num_points}-point prompts for {len(video_names)} videos to "
        f"{os.path.abspath(args.output_json)}"
    )


if __name__ == "__main__":
    main()
