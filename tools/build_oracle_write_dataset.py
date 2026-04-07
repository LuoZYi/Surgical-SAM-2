#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.memory_write_controller import DEFAULT_WRITE_FEATURE_NAMES, build_feature_vector
from tools.vos_inference import load_masks_from_dir


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def sorted_frame_names(video_dir: str) -> List[str]:
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    frame_names.sort(key=lambda p: int(p))
    return frame_names


def collect_inputs_per_object(
    ann_root: str,
    video_name: str,
    frame_names: List[str],
) -> Dict[int, Dict[int, np.ndarray]]:
    inputs_per_object: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    for idx, frame_name in enumerate(frame_names):
        per_obj_mask, _ = load_masks_from_dir(
            input_mask_dir=ann_root,
            video_name=video_name,
            frame_name=frame_name,
            per_obj_png_file=False,
            allow_missing=False,
        )
        for object_id, object_mask in per_obj_mask.items():
            if object_id in inputs_per_object:
                continue
            if not np.any(object_mask):
                continue
            inputs_per_object[object_id][idx] = object_mask
    return inputs_per_object


def load_gt_masks(
    ann_root: str,
    video_name: str,
    frame_names: List[str],
) -> List[Dict[int, np.ndarray]]:
    gt_per_frame: List[Dict[int, np.ndarray]] = []
    for frame_name in frame_names:
        per_obj_mask, _ = load_masks_from_dir(
            input_mask_dir=ann_root,
            video_name=video_name,
            frame_name=frame_name,
            per_obj_png_file=False,
            allow_missing=False,
        )
        gt_per_frame.append(per_obj_mask)
    return gt_per_frame


def build_oracle_labels(
    records: List[Dict[str, object]],
    horizon: int,
    gain_threshold: float,
    novelty_threshold: float,
    pred_quality_threshold: float,
) -> None:
    if len(records) == 0:
        return

    gt_masks = [record["_gt_mask"] for record in records]
    num_records = len(records)
    iou_matrix = np.zeros((num_records, num_records), dtype=np.float32)
    for i in range(num_records):
        for j in range(i, num_records):
            value = mask_iou(gt_masks[i], gt_masks[j])
            iou_matrix[i, j] = value
            iou_matrix[j, i] = value

    last_written_idx = 0
    records[0]["oracle_write"] = 1
    records[0]["oracle_future_gain"] = 0.0
    records[0]["oracle_gt_novelty"] = 0.0
    for idx in range(1, num_records):
        record = records[idx]
        gt_area = float(record["gt_area_ratio"])
        pred_iou = float(record["pred_iou_to_gt"])
        novelty = 1.0 - float(iou_matrix[idx, last_written_idx])
        future_end = min(num_records, idx + 1 + horizon)
        future_indices = [
            future_idx
            for future_idx in range(idx + 1, future_end)
            if float(records[future_idx]["gt_area_ratio"]) > 0.0
        ]
        if len(future_indices) == 0:
            future_gain = 0.0
        else:
            future_gain = float(
                np.mean(
                    [
                        iou_matrix[idx, future_idx] - iou_matrix[last_written_idx, future_idx]
                        for future_idx in future_indices
                    ]
                )
            )

        should_write = (
            gt_area > 0.0
            and pred_iou >= pred_quality_threshold
            and novelty >= novelty_threshold
            and future_gain >= gain_threshold
        )
        record["oracle_write"] = int(should_write)
        record["oracle_future_gain"] = future_gain
        record["oracle_gt_novelty"] = novelty
        if should_write:
            last_written_idx = idx


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam2_checkpoint", type=str, default="./checkpoints/sam2.1_hiera_s_endo18.pth")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--oracle_gain_threshold", type=float, default=0.03)
    parser.add_argument("--oracle_novelty_threshold", type=float, default=0.08)
    parser.add_argument("--oracle_pred_quality_threshold", type=float, default=0.60)
    args = parser.parse_args()

    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=False,
        hydra_overrides_extra=[
            "++model._target_=sam2.sam2_video_predictor_new.SAM2VideoPredictorNew",
            "++model.memory_prune_mode=efp",
            "++model.memory_score_mode=cosine_only",
            "++model.num_frame_to_prune=2",
            "++model.memory_write_mode=adaptive_skip_write_v2",
            "++model.memory_write_similarity_threshold=2.0",
            "++model.memory_write_mask_change_threshold=-1.0",
            "++model.memory_write_quality_threshold=-1.0",
            "++model.memory_write_min_area=-1.0",
            "++model.memory_controller_low_quality_threshold=-1.0",
            "++model.memory_controller_skip_centroid_shift_threshold=-1.0",
            "++model.debug_memory_write=true",
        ],
    )

    fieldnames = [
        "dataset_name",
        "video_name",
        "object_id",
        "frame_idx",
        "frame_name",
        "is_prompt_frame",
        "pred_iou_to_gt",
        "gt_area_ratio",
        "oracle_write",
        "oracle_future_gain",
        "oracle_gt_novelty",
        *DEFAULT_WRITE_FEATURE_NAMES,
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_root in args.datasets:
            dataset_root = os.path.abspath(dataset_root)
            dataset_name = os.path.basename(os.path.normpath(os.path.dirname(dataset_root)))
            video_root = os.path.join(dataset_root, "JPEGImages")
            ann_root = os.path.join(dataset_root, "Annotations")
            video_names = sorted(
                [p for p in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, p))]
            )
            if args.max_videos > 0:
                video_names = video_names[: args.max_videos]
            for video_name in video_names:
                video_dir = os.path.join(video_root, video_name)
                frame_names = sorted_frame_names(video_dir)
                gt_per_frame = load_gt_masks(ann_root, video_name, frame_names)
                inputs_per_object = collect_inputs_per_object(ann_root, video_name, frame_names)
                inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=False)

                for object_id in sorted(inputs_per_object):
                    predictor.reset_state(inference_state)
                    input_frame_inds = sorted(inputs_per_object[object_id])
                    for input_frame_idx in input_frame_inds:
                        predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=input_frame_idx,
                            obj_id=object_id,
                            mask=inputs_per_object[object_id][input_frame_idx],
                        )

                    object_records: List[Dict[str, object]] = []
                    start_frame_idx = min(input_frame_inds)
                    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=start_frame_idx,
                        reverse=False,
                    ):
                        gt_mask = gt_per_frame[out_frame_idx].get(
                            object_id,
                            np.zeros_like(next(iter(gt_per_frame[out_frame_idx].values()))) if gt_per_frame[out_frame_idx] else np.zeros_like(inputs_per_object[object_id][input_frame_inds[0]], dtype=bool),
                        )
                        gt_mask = gt_mask.astype(bool)
                        pred_mask = (out_mask_logits[0] > 0).detach().cpu().numpy().astype(bool)
                        pred_iou_to_gt = mask_iou(pred_mask, gt_mask)
                        write_debug = dict(getattr(predictor, "_last_memory_write_debug", {}))
                        feature_values = build_feature_vector(write_debug, DEFAULT_WRITE_FEATURE_NAMES)
                        record = {
                            "dataset_name": dataset_name,
                            "video_name": video_name,
                            "object_id": int(object_id),
                            "frame_idx": int(out_frame_idx),
                            "frame_name": frame_names[out_frame_idx],
                            "is_prompt_frame": int(out_frame_idx in input_frame_inds),
                            "pred_iou_to_gt": float(pred_iou_to_gt),
                            "gt_area_ratio": float(gt_mask.mean()),
                            "oracle_write": 0,
                            "oracle_future_gain": 0.0,
                            "oracle_gt_novelty": 0.0,
                            "_gt_mask": gt_mask,
                        }
                        for name, value in zip(DEFAULT_WRITE_FEATURE_NAMES, feature_values.tolist()):
                            record[name] = float(value)
                        object_records.append(record)

                    build_oracle_labels(
                        object_records,
                        horizon=args.horizon,
                        gain_threshold=args.oracle_gain_threshold,
                        novelty_threshold=args.oracle_novelty_threshold,
                        pred_quality_threshold=args.oracle_pred_quality_threshold,
                    )
                    for record in object_records:
                        record.pop("_gt_mask", None)
                        writer.writerow(record)
                        f.flush()


if __name__ == "__main__":
    main()
