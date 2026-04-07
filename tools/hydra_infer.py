#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Hydra-style VOS inference entry point that leaves `tools/vos_inference.py`
untouched.

Examples
--------
python tools/hydra_infer.py dataset=endovis2017 memory_policy=off
python tools/hydra_infer.py dataset=endovis2018 memory_policy=rule_recent
python tools/hydra_infer.py dataset=endovis2017 memory_policy=rule_recent dry_run=true print_config=true
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Iterable, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from hydra import compose
from omegaconf import DictConfig, OmegaConf

from sam2.build_sam import build_sam2_video_predictor
from sav_dataset.utils.endo_sav_benchmark import benchmark
from tools.vos_inference import (
    load_first_frame_prompt_file,
    vos_separate_inference_per_object,
)


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path in (None, ""):
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def _bool_str(value: bool) -> str:
    return "true" if bool(value) else "false"


def _normalize_overrides(overrides: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for item in overrides:
        if item.startswith("dataset="):
            normalized.append("dataset_name=" + item.split("=", 1)[1])
        elif item.startswith("memory_policy="):
            normalized.append("memory_policy_name=" + item.split("=", 1)[1])
        else:
            normalized.append(item)
    return normalized


def _split_overrides(overrides: Iterable[str]) -> tuple[List[str], List[str]]:
    pre_merge: List[str] = []
    post_merge: List[str] = []
    for item in overrides:
        if (
            item.startswith("dataset.")
            or item.startswith("memory_policy.")
            or item.startswith("model.")
        ):
            post_merge.append(item)
        else:
            pre_merge.append(item)
    return pre_merge, post_merge


def _load_fragment(group: str, name: str) -> DictConfig:
    path = os.path.join(REPO_ROOT, "sam2", "configs", "vos_inference", group, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config fragment: {path}")
    return OmegaConf.load(path)


def compose_app_config(overrides: Optional[Iterable[str]] = None) -> DictConfig:
    normalized_overrides = _normalize_overrides(list(overrides or []))
    pre_merge_overrides, post_merge_overrides = _split_overrides(normalized_overrides)

    cfg = compose(
        config_name="configs/vos_inference/default",
        overrides=pre_merge_overrides,
    )
    OmegaConf.set_struct(cfg, False)
    dataset_cfg = _load_fragment("dataset", str(cfg.dataset_name))
    memory_policy_cfg = _load_fragment("memory_policy", str(cfg.memory_policy_name))
    cfg = OmegaConf.merge(
        cfg,
        {"dataset": dataset_cfg},
        {"memory_policy": memory_policy_cfg},
        {"model": memory_policy_cfg.get("model", {})},
    )
    if post_merge_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(post_merge_overrides))
    OmegaConf.resolve(cfg)
    return cfg


def build_model_overrides(cfg: DictConfig) -> List[str]:
    model_cfg = cfg.model
    overrides = [
        f"++model._target_={model_cfg.predictor_target}",
        f"++model.memory_prune_mode={model_cfg.memory_prune_mode}",
        f"++model.memory_score_mode={model_cfg.memory_score_mode}",
        f"++model.num_frame_to_prune={model_cfg.num_frame_to_prune}",
        f"++model.memory_min_temporal_gap={model_cfg.memory_min_temporal_gap}",
        f"++model.protect_conditioning_memories={_bool_str(model_cfg.protect_conditioning_memories)}",
        f"++model.debug_memory_pruning={_bool_str(model_cfg.debug_memory_pruning)}",
        f"++model.use_recent_memory_guard={_bool_str(model_cfg.use_recent_memory_guard)}",
        f"++model.recent_memory_min_keep={model_cfg.recent_memory_min_keep}",
        f"++model.recent_memory_max_keep={model_cfg.recent_memory_max_keep}",
        f"++model.recent_similarity_threshold={model_cfg.recent_similarity_threshold}",
        f"++model.recent_stability_threshold={model_cfg.recent_stability_threshold}",
        f"++model.recent_confidence_threshold={model_cfg.recent_confidence_threshold}",
        f"++model.memory_write_mode={model_cfg.memory_write_mode}",
        f"++model.memory_write_similarity_threshold={model_cfg.memory_write_similarity_threshold}",
        f"++model.memory_write_mask_change_threshold={model_cfg.memory_write_mask_change_threshold}",
        f"++model.memory_write_quality_threshold={model_cfg.memory_write_quality_threshold}",
        f"++model.memory_write_min_area={model_cfg.memory_write_min_area}",
        f"++model.memory_write_controller_path={model_cfg.memory_write_controller_path}",
        f"++model.memory_write_controller_threshold={model_cfg.memory_write_controller_threshold}",
        f"++model.memory_write_controller_low_threshold={model_cfg.memory_write_controller_low_threshold}",
        f"++model.memory_write_controller_high_threshold={model_cfg.memory_write_controller_high_threshold}",
        f"++model.memory_write_controller_fallback_band={model_cfg.memory_write_controller_fallback_band}",
        f"++model.challenging_mode_detector_path={model_cfg.challenging_mode_detector_path}",
        f"++model.challenging_mode_detector_threshold={model_cfg.challenging_mode_detector_threshold}",
        f"++model.challenging_mode_enter_threshold={model_cfg.challenging_mode_enter_threshold}",
        f"++model.challenging_mode_exit_threshold={model_cfg.challenging_mode_exit_threshold}",
        f"++model.challenging_mode_window_size={model_cfg.challenging_mode_window_size}",
        f"++model.memory_controller_low_quality_threshold={model_cfg.memory_controller_low_quality_threshold}",
        f"++model.memory_controller_skip_centroid_shift_threshold={model_cfg.memory_controller_skip_centroid_shift_threshold}",
        f"++model.memory_controller_conservative_similarity_threshold={model_cfg.memory_controller_conservative_similarity_threshold}",
        f"++model.memory_controller_conservative_mask_change_threshold={model_cfg.memory_controller_conservative_mask_change_threshold}",
        f"++model.memory_controller_conservative_centroid_shift_threshold={model_cfg.memory_controller_conservative_centroid_shift_threshold}",
        f"++model.memory_controller_conservative_fill_ratio_threshold={model_cfg.memory_controller_conservative_fill_ratio_threshold}",
        f"++model.memory_controller_conservative_prune_delta={model_cfg.memory_controller_conservative_prune_delta}",
        f"++model.memory_controller_aggressive_quality_threshold={model_cfg.memory_controller_aggressive_quality_threshold}",
        f"++model.memory_controller_aggressive_similarity_threshold={model_cfg.memory_controller_aggressive_similarity_threshold}",
        f"++model.memory_controller_aggressive_mask_change_threshold={model_cfg.memory_controller_aggressive_mask_change_threshold}",
        f"++model.memory_controller_aggressive_centroid_shift_threshold={model_cfg.memory_controller_aggressive_centroid_shift_threshold}",
        f"++model.memory_controller_aggressive_fill_ratio_threshold={model_cfg.memory_controller_aggressive_fill_ratio_threshold}",
        f"++model.memory_controller_aggressive_prune_delta={model_cfg.memory_controller_aggressive_prune_delta}",
        f"++model.debug_memory_write={_bool_str(model_cfg.debug_memory_write)}",
    ]
    overrides.extend(list(cfg.model_hydra_overrides_extra))
    return overrides


def main(argv: Optional[List[str]] = None) -> None:
    cfg = compose_app_config(argv if argv is not None else sys.argv[1:])

    if cfg.print_config or cfg.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.dry_run:
        return

    base_video_dir = _resolve_path(cfg.dataset.base_video_dir)
    input_mask_dir = _resolve_path(cfg.dataset.input_mask_dir)
    output_mask_dir = _resolve_path(cfg.output_mask_dir)
    gt_root = _resolve_path(cfg.dataset.gt_root)
    video_list_file = _resolve_path(cfg.dataset.video_list_file)
    sam2_checkpoint = _resolve_path(cfg.sam2_checkpoint)
    first_frame_prompt_file = _resolve_path(cfg.first_frame_prompt_file)

    os.makedirs(output_mask_dir, exist_ok=True)

    torch.cuda.set_device(int(cfg.gpu_id))
    print(f"Using GPU {cfg.gpu_id}")
    print("Warning: only support evaluating one object per sequence and saving in object directories.")
    print("Testing with first frame prompt file:", cfg.first_frame_prompt_file)

    predictor = build_sam2_video_predictor(
        config_file=str(cfg.sam2_cfg),
        ckpt_path=sam2_checkpoint,
        apply_postprocessing=bool(cfg.apply_postprocessing),
        hydra_overrides_extra=build_model_overrides(cfg),
    )

    print("Predictor class:", type(predictor))
    print("Predictor file:", inspect.getfile(type(predictor)))
    print("memory_prune_mode:", getattr(predictor, "memory_prune_mode", None))
    print("memory_score_mode:", getattr(predictor, "memory_score_mode", None))
    print("num_frame_to_prune:", getattr(predictor, "num_frame_to_prune", None))
    print("use_recent_memory_guard:", getattr(predictor, "use_recent_memory_guard", None))
    print("memory_write_mode:", getattr(predictor, "memory_write_mode", None))

    first_frame_prompts = load_first_frame_prompt_file(first_frame_prompt_file)

    if cfg.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print("using only the first frame's mask in input_mask_dir as input to the SAM 2 model")

    if video_list_file is not None:
        with open(video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(base_video_dir)
            if os.path.isdir(os.path.join(base_video_dir, p))
        ]
    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    total_pure_inference_time_sec = 0.0
    total_pure_inference_frames = 0
    max_peak_allocated_gb = 0.0
    max_peak_reserved_gb = 0.0

    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        video_stats = vos_separate_inference_per_object(
            predictor=predictor,
            base_video_dir=base_video_dir,
            input_mask_dir=input_mask_dir,
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            first_frame_prompts=(
                None if first_frame_prompts is None else first_frame_prompts.get(video_name)
            ),
            score_thresh=float(cfg.score_thresh),
            use_all_masks=bool(cfg.use_all_masks),
            read_frame_interval=int(cfg.read_frame_interval),
            save_frame_interval=int(cfg.save_frame_interval),
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
        f"output masks saved to {output_mask_dir}"
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

    epoch = os.path.basename(str(cfg.sam2_checkpoint)).split(".")[0].split("_")[-1]
    epoch = int(epoch) if epoch.isdigit() else 0

    benchmark(
        [gt_root],
        [output_mask_dir],
        bool(cfg.strict),
        int(cfg.num_processes),
        verbose=not bool(cfg.quiet),
        skip_first_and_last=not bool(cfg.do_not_skip_first_and_last_frame),
        epoch=epoch,
    )


if __name__ == "__main__":
    main()
