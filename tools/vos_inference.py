# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import json
from PIL import Image
from sympy import evaluate
import inspect

from sam2.build_sam import build_sam2_video_predictor
from sav_dataset.utils.endo_sav_benchmark import benchmark

# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _get_peak_cuda_memory_stats():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    bytes_per_gb = 1024 ** 3
    peak_allocated_gb = torch.cuda.max_memory_allocated() / bytes_per_gb
    peak_reserved_gb = torch.cuda.max_memory_reserved() / bytes_per_gb
    return peak_allocated_gb, peak_reserved_gb


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if not os.path.exists(input_mask_path):
            pass
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def load_first_frame_prompt_file(path: Optional[str]) -> Optional[Dict[str, Dict[int, Dict[str, Any]]]]:
    """Load per-video point prompts from a JSON file."""
    if path in (None, ""):
        return None

    with open(path, "r") as f:
        payload = json.load(f)

    videos_payload = payload.get("videos", payload)
    normalized: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for video_name, video_payload in videos_payload.items():
        objects_payload = video_payload.get("objects", video_payload)
        per_video: Dict[int, Dict[str, Any]] = {}
        for object_id_str, object_payload in objects_payload.items():
            points = np.asarray(object_payload["points"], dtype=np.float32)
            labels = np.asarray(
                object_payload.get("labels", [1] * len(points)),
                dtype=np.int32,
            )
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(
                    f"Invalid points for video={video_name}, object={object_id_str}: "
                    f"expected shape [N, 2], got {points.shape}"
                )
            if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
                raise ValueError(
                    f"Invalid labels for video={video_name}, object={object_id_str}: "
                    f"expected shape [{points.shape[0]}], got {labels.shape}"
                )
            per_video[int(object_id_str)] = {
                "frame_name": object_payload.get("frame_name"),
                "frame_idx": object_payload.get("frame_idx"),
                "points": points,
                "labels": labels,
            }
        normalized[str(video_name)] = per_video
    return normalized


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    output_per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not output_per_obj_png_file:
        os.makedirs(os.path.join(output_mask_dir, video_name, 'all'), exist_ok=True)
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, 'all', f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask[output_mask>0] = object_id
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    first_frame_prompts=None,
    score_thresh=0.0,
    use_all_masks=False,
    read_frame_interval=1,
    save_frame_interval=1,
):
    """
    Run VOS inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    all_frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
    ]
    # only process frames with frame_interval
    frame_names = [p for p in all_frame_names if int(os.path.splitext(p)[0]) % read_frame_interval == 0]
    # save_pred_freq = len(all_frame_names) // len(frame_names)
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None
    frame_name_to_idx = {frame_name: idx for idx, frame_name in enumerate(frame_names)}

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    if first_frame_prompts is not None:
        input_palette = DAVIS_PALETTE
        for object_id, prompt_spec in sorted(first_frame_prompts.items()):
            frame_name = prompt_spec.get("frame_name")
            frame_idx = prompt_spec.get("frame_idx")
            if frame_idx is None:
                if frame_name is None:
                    raise ValueError(
                        f"Prompt for video={video_name}, object_id={object_id} "
                        "must contain either frame_name or frame_idx"
                    )
                if frame_name not in frame_name_to_idx:
                    raise ValueError(
                        f"Prompt frame {frame_name} not found in video {video_name}"
                    )
                frame_idx = frame_name_to_idx[frame_name]
            else:
                frame_idx = int(frame_idx)
            print(
                f"adding {len(prompt_spec['points'])} point prompts from frame {frame_names[frame_idx]} "
                f"as input for object_id={object_id}"
            )
            inputs_per_object[object_id][frame_idx] = {
                "points": np.asarray(prompt_spec["points"], dtype=np.float32),
                "labels": np.asarray(prompt_spec["labels"], dtype=np.int32),
            }
    else:
        for idx, name in enumerate(frame_names):
            if os.path.exists(os.path.join(input_mask_dir, video_name, f"{name}.png")):
                per_obj_input_mask, input_palette = load_masks_from_dir(
                    input_mask_dir=input_mask_dir,
                    video_name=video_name,
                    frame_name=frame_names[idx],
                    per_obj_png_file=False,  # our dataset combines all object masks into a single PNG file
                    allow_missing=False,
                )
                for object_id, object_mask in per_obj_input_mask.items():
                    # skip empty masks
                    if not np.any(object_mask):
                        continue
                    # if `use_all_masks=False`, we only use the first mask for each object
                    if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                        continue
                    print(f"adding mask from frame {idx} as input for {object_id=}")
                    inputs_per_object[object_id][idx] = object_mask

    # # step 1: run inference together for each object appearing in the first frame
    # object_ids = sorted(inputs_per_object)
    # output_scores_per_object = defaultdict(dict)
    # # find the object appear in the first frame
    # first_frame_idx = 0
    # first_frame_object_ids = []
    # latter_frame_object_ids = []
    # for object_id in object_ids:
    #     if inputs_per_object[object_id].keys().__contains__(first_frame_idx):
    #         first_frame_object_ids.append(object_id)
    #     else:
    #         latter_frame_object_ids.append(object_id)
    #
    # for object_id in first_frame_object_ids:
    #     predictor.add_new_mask(
    #         inference_state=inference_state,
    #         frame_idx=first_frame_idx,
    #         obj_id=object_id,
    #         mask=inputs_per_object[object_id][first_frame_idx],
    #     )
    #
    # # run propagation throughout the video and collect the results in a dict
    # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    #     inference_state, start_frame_idx=first_frame_idx, reverse=False,
    # ):
    #     obj_scores = out_mask_logits.cpu().numpy()
    #     for i, out_obj_id in enumerate(out_obj_ids):
    #         output_scores_per_object[out_obj_id][out_frame_idx] = obj_scores[i: i + 1]

    # step 2: run inference separately for the object appearing in the latter frame
    # for object_id in latter_frame_object_ids:
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    pure_inference_time_sec = 0.0
    pure_inference_frames = 0

    _sync_cuda()
    _reset_peak_cuda_memory()

    for object_id in object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            input_value = inputs_per_object[object_id][input_frame_idx]
            if isinstance(input_value, dict) and "points" in input_value:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    points=input_value["points"],
                    labels=input_value["labels"],
                )
            else:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=input_value,
                )

        # run propagation throughout the video and collect the results in a dict
        _sync_cuda()
        start_time = time.time()
        propagated_frames_this_object = 0
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):
            # if out_frame_idx < 10:
            #     dbg = predictor._last_memory_prune_debug
            #     print(f"\n[DBG] object_id={object_id}, frame={out_frame_idx}")
            #     print("mode:", dbg.get("mode"))
            #     print("score_mode:", dbg.get("score_mode"))
            #     print("pruned_frame_idx:", dbg.get("pruned_frame_idx"))
            #     print("candidate_frame_idx:", dbg.get("candidate_frame_idx"))
            #     print("candidate_similarities_mean:", dbg.get("candidate_similarities_mean"))
            #     print("candidate_motion:", dbg.get("candidate_motion"))
            #     print("candidate_geometry_overlap:", dbg.get("candidate_geometry_overlap"))
            #     print("candidate_drop_scores:", dbg.get("candidate_drop_scores"))
            obj_scores = out_mask_logits.cpu().numpy()
            output_scores_per_object[object_id][out_frame_idx] = obj_scores
            propagated_frames_this_object += 1
        _sync_cuda()
        pure_inference_time_sec += time.time() - start_time
        pure_inference_frames += propagated_frames_this_object

    video_segments = {}
    # save_frame_interval，控制保存频率
    for out_frame_idx in range(len(frame_names)):
        frame_time = int(frame_names[out_frame_idx])
        if frame_time % save_frame_interval != 0:
            continue
        video_segments[out_frame_idx] = {}
        for object_id in object_ids:
            if output_scores_per_object[object_id].keys().__contains__(out_frame_idx):
                video_segments[out_frame_idx][object_id] = output_scores_per_object[object_id][out_frame_idx] > score_thresh

    # step 3: save the output masks as per-object PNG files
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            per_obj_output_mask=per_obj_output_mask,
            frame_name=frame_names[out_frame_idx],
            output_per_obj_png_file=True,
            height=height,
            width=width,
            output_palette=input_palette,
        )

    # step 4: save the output masks as a single PNG file
    # post-processing: consolidate the per-object scores into per-frame masks
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx in range(len(frame_names)):
        frame_time = int(frame_names[frame_idx])
        if frame_time % save_frame_interval != 0:
            continue
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        scores = predictor._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask

    # write the output masks as palette PNG files to output_mask_dir
    for frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            output_per_obj_png_file=False,
            output_palette=input_palette,
        )

    peak_allocated_gb, peak_reserved_gb = _get_peak_cuda_memory_stats()
    pure_inference_fps = (
        pure_inference_frames / pure_inference_time_sec
        if pure_inference_time_sec > 0
        else 0.0
    )
    stats = {
        "video_name": video_name,
        "num_objects": len(object_ids),
        "pure_inference_frames": pure_inference_frames,
        "pure_inference_time_sec": pure_inference_time_sec,
        "pure_inference_fps": pure_inference_fps,
        "peak_cuda_memory_allocated_gb": peak_allocated_gb,
        "peak_cuda_memory_reserved_gb": peak_reserved_gb,
    }
    print(
        f"[PureInference] video={video_name} "
        f"objects={stats['num_objects']} "
        f"frames={pure_inference_frames} "
        f"time={pure_inference_time_sec:.2f}s "
        f"fps={pure_inference_fps:.2f} "
        f"peak_alloc={peak_allocated_gb:.2f}GB "
        f"peak_reserved={peak_reserved_gb:.2f}GB"
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_b+.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=True,
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    # we track the object appear in the first frame and the object appear in the latter frame
    parser.add_argument(
        "--first_frame_prompt_file",
        type=str, default=None,
    )
    parser.add_argument(
        "--gt_root",
        required=True,
        help="Path to the GT folder. For SA-V, it's sav_val/Annotations_6fps or sav_test/Annotations_6fps",
    )
    parser.add_argument(
        "--gpu_id",
        type=int, default=0,
    )
    parser.add_argument(
        "--read_frame_interval",
        type=int, default=1,
    )
    parser.add_argument(
        "--save_frame_interval",
        type=int, default=1,
    )
    # ----
    parser.add_argument(
        "-n", "--num_processes", default=16, type=int, help="Number of concurrent processes"
    )
    parser.add_argument(
        "-s",
        "--strict",
        help="Make sure every video in the gt_root folder has a corresponding video in the prediction",
        action="store_true",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Quietly run evaluation without printing the information out",
        action="store_true",
    )
    parser.add_argument(
    "--memory_prune_mode",
    type=str,
    default="efp",
    choices=["off", "efp", "rule_based", "state_aware"],
    )
    parser.add_argument(
        "--memory_score_mode",
        type=str,
        default="cosine_only",
        choices=["cosine_only", "cosine_motion", "cosine_motion_geometry"],
    )
    parser.add_argument(
        "--num_frame_to_prune",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--protect_conditioning_memories",
        action="store_true",
    )
    parser.add_argument(
        "--debug_memory_pruning",
        action="store_true",
    )
    parser.add_argument(
        "--memory_min_temporal_gap",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_recent_memory_guard",
        action="store_true",
    )
    parser.add_argument(
        "--recent_memory_min_keep",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--recent_memory_max_keep",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--recent_similarity_threshold",
        type=float,
        default=0.985,
    )
    parser.add_argument(
        "--recent_stability_threshold",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--recent_confidence_threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--memory_write_mode",
        type=str,
        default="off",
        choices=[
            "off",
            "quality_novelty_gate",
            "adaptive_controller",
            "adaptive_controller_v2",
            "adaptive_skip_write_v2",
            "learned_skip_write_controller",
            "learned_adaptive_controller_v2",
            "sliding_window_challenging_controller_v1",
        ],
    )
    parser.add_argument(
        "--memory_write_similarity_threshold",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--memory_write_mask_change_threshold",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--memory_write_quality_threshold",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--memory_write_min_area",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--memory_write_controller_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--memory_write_controller_threshold",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--memory_write_controller_low_threshold",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--memory_write_controller_high_threshold",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--memory_write_controller_fallback_band",
        type=float,
        default=0.075,
    )
    parser.add_argument(
        "--challenging_mode_detector_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--challenging_mode_detector_threshold",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--challenging_mode_enter_threshold",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--challenging_mode_exit_threshold",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--challenging_mode_window_size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--memory_controller_low_quality_threshold",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--memory_controller_skip_centroid_shift_threshold",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--memory_controller_conservative_similarity_threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--memory_controller_conservative_mask_change_threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--memory_controller_conservative_centroid_shift_threshold",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--memory_controller_conservative_fill_ratio_threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--memory_controller_conservative_prune_delta",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--memory_controller_aggressive_quality_threshold",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--memory_controller_aggressive_similarity_threshold",
        type=float,
        default=0.96,
    )
    parser.add_argument(
        "--memory_controller_aggressive_mask_change_threshold",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--memory_controller_aggressive_centroid_shift_threshold",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--memory_controller_aggressive_fill_ratio_threshold",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--memory_controller_aggressive_prune_delta",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--debug_memory_write",
        action="store_true",
    )
    parser.add_argument(
        "--do_not_skip_first_and_last_frame",
        help="In SA-V val and test, we skip the first and the last annotated frames in evaluation. "
             "Set this to true for evaluation on settings that doen't skip first and last frames",
        action="store_true",
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    print(f"Using GPU {args.gpu_id}")
    print('Warning: only support evaluating one object per sequence and saving in object directories.')
    print('Testing with first frame prompt file:', args.first_frame_prompt_file)

    # predictor = build_sam2_video_predictor(
    #     config_file=args.sam2_cfg,
    #     ckpt_path=args.sam2_checkpoint,
    #     apply_postprocessing=args.apply_postprocessing,
    #     hydra_overrides_extra=[],
    # )

    # predictor = build_sam2_video_predictor(
    #     config_file=args.sam2_cfg,
    #     ckpt_path=args.sam2_checkpoint,
    #     apply_postprocessing=args.apply_postprocessing,
    #     hydra_overrides_extra=[
    #         "++model._target_=sam2.sam2_video_predictor_new.SAM2VideoPredictorNew",
    #         "++model.memory_prune_mode=off",
    #         "++model.num_frame_to_prune=0",
    #         "++model.protect_conditioning_memories=false",
    #     ],
    # )
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_new.SAM2VideoPredictorNew",
        f"++model.memory_prune_mode={args.memory_prune_mode}",
        f"++model.memory_score_mode={args.memory_score_mode}",
        f"++model.num_frame_to_prune={args.num_frame_to_prune}",
        f"++model.memory_min_temporal_gap={args.memory_min_temporal_gap}",
        f"++model.protect_conditioning_memories={'true' if args.protect_conditioning_memories else 'false'}",
        f"++model.debug_memory_pruning={'true' if args.debug_memory_pruning else 'false'}",
        f"++model.use_recent_memory_guard={'true' if args.use_recent_memory_guard else 'false'}",
        f"++model.recent_memory_min_keep={args.recent_memory_min_keep}",
        f"++model.recent_memory_max_keep={args.recent_memory_max_keep}",
        f"++model.recent_similarity_threshold={args.recent_similarity_threshold}",
        f"++model.recent_stability_threshold={args.recent_stability_threshold}",
        f"++model.recent_confidence_threshold={args.recent_confidence_threshold}",
        f"++model.memory_write_mode={args.memory_write_mode}",
        f"++model.memory_write_similarity_threshold={args.memory_write_similarity_threshold}",
        f"++model.memory_write_mask_change_threshold={args.memory_write_mask_change_threshold}",
        f"++model.memory_write_quality_threshold={args.memory_write_quality_threshold}",
        f"++model.memory_write_min_area={args.memory_write_min_area}",
        f"++model.memory_write_controller_path={args.memory_write_controller_path}",
        f"++model.memory_write_controller_threshold={args.memory_write_controller_threshold}",
        f"++model.memory_write_controller_low_threshold={args.memory_write_controller_low_threshold}",
        f"++model.memory_write_controller_high_threshold={args.memory_write_controller_high_threshold}",
        f"++model.memory_write_controller_fallback_band={args.memory_write_controller_fallback_band}",
        f"++model.challenging_mode_detector_path={args.challenging_mode_detector_path}",
        f"++model.challenging_mode_detector_threshold={args.challenging_mode_detector_threshold}",
        f"++model.challenging_mode_enter_threshold={args.challenging_mode_enter_threshold}",
        f"++model.challenging_mode_exit_threshold={args.challenging_mode_exit_threshold}",
        f"++model.challenging_mode_window_size={args.challenging_mode_window_size}",
        f"++model.memory_controller_low_quality_threshold={args.memory_controller_low_quality_threshold}",
        f"++model.memory_controller_skip_centroid_shift_threshold={args.memory_controller_skip_centroid_shift_threshold}",
        f"++model.memory_controller_conservative_similarity_threshold={args.memory_controller_conservative_similarity_threshold}",
        f"++model.memory_controller_conservative_mask_change_threshold={args.memory_controller_conservative_mask_change_threshold}",
        f"++model.memory_controller_conservative_centroid_shift_threshold={args.memory_controller_conservative_centroid_shift_threshold}",
        f"++model.memory_controller_conservative_fill_ratio_threshold={args.memory_controller_conservative_fill_ratio_threshold}",
        f"++model.memory_controller_conservative_prune_delta={args.memory_controller_conservative_prune_delta}",
        f"++model.memory_controller_aggressive_quality_threshold={args.memory_controller_aggressive_quality_threshold}",
        f"++model.memory_controller_aggressive_similarity_threshold={args.memory_controller_aggressive_similarity_threshold}",
        f"++model.memory_controller_aggressive_mask_change_threshold={args.memory_controller_aggressive_mask_change_threshold}",
        f"++model.memory_controller_aggressive_centroid_shift_threshold={args.memory_controller_aggressive_centroid_shift_threshold}",
        f"++model.memory_controller_aggressive_fill_ratio_threshold={args.memory_controller_aggressive_fill_ratio_threshold}",
        f"++model.memory_controller_aggressive_prune_delta={args.memory_controller_aggressive_prune_delta}",
        f"++model.debug_memory_write={'true' if args.debug_memory_write else 'false'}",
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides,
    )


    print("Predictor class:", type(predictor))
    print("Predictor file:", inspect.getfile(type(predictor)))
    print("memory_prune_mode:", getattr(predictor, "memory_prune_mode", None))
    print("memory_score_mode:", getattr(predictor, "memory_score_mode", None))
    print("num_frame_to_prune:", getattr(predictor, "num_frame_to_prune", None))
    print("use_recent_memory_guard:", getattr(predictor, "use_recent_memory_guard", None))
    print("memory_write_mode:", getattr(predictor, "memory_write_mode", None))

    first_frame_prompts = load_first_frame_prompt_file(args.first_frame_prompt_file)

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
               # and int(p.split('_')[1]) in [9,10]
        ]
    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    total_pure_inference_time_sec = 0.0
    total_pure_inference_frames = 0
    max_peak_allocated_gb = 0.0
    max_peak_reserved_gb = 0.0

    # we first run every object separately and then combine them
    for n_video, video_name in enumerate(video_names):
        # if n_video >= 5:
        #     continue
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        video_stats = vos_separate_inference_per_object(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            first_frame_prompts=(
                None if first_frame_prompts is None else first_frame_prompts.get(video_name)
            ),
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
            read_frame_interval=args.read_frame_interval,
            save_frame_interval=args.save_frame_interval,
        )
        total_pure_inference_time_sec += video_stats["pure_inference_time_sec"]
        total_pure_inference_frames += video_stats["pure_inference_frames"]
        max_peak_allocated_gb = max(
            max_peak_allocated_gb, video_stats["peak_cuda_memory_allocated_gb"]
        )
        max_peak_reserved_gb = max(
            max_peak_reserved_gb, video_stats["peak_cuda_memory_reserved_gb"]
        )

    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )
    total_pure_inference_fps = (
        total_pure_inference_frames / total_pure_inference_time_sec
        if total_pure_inference_time_sec > 0
        else 0.0
    )
    print(
        f"[PureInference][Summary] videos={len(video_names)} "
        f"frames={total_pure_inference_frames} "
        f"time={total_pure_inference_time_sec:.2f}s "
        f"fps={total_pure_inference_fps:.2f} "
        f"max_peak_alloc={max_peak_allocated_gb:.2f}GB "
        f"max_peak_reserved={max_peak_reserved_gb:.2f}GB"
    )

    epoch = args.sam2_checkpoint.split('/')[-1].split('.')[0].split('_')[-1]
    epoch = int(epoch) if epoch.isdigit() else 0

    benchmark(
        [args.gt_root],
        [args.output_mask_dir],
        args.strict,
        args.num_processes,
        verbose=not args.quiet,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
        epoch=epoch,
    )



if __name__ == "__main__":
    main()
