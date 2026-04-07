#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Bernoulli

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.memory_write_controller import (
    DEFAULT_WRITE_FEATURE_NAMES,
    OfflineWriteController,
    build_feature_vector,
)
from tools.vos_inference import load_masks_from_dir


@dataclass
class EpisodeSpec:
    dataset_name: str
    video_name: str
    video_dir: str
    ann_root: str
    frame_names: List[str]
    object_id: int
    input_frame_idx: int
    input_mask: np.ndarray
    group_key: str


def sorted_frame_names(video_dir: str) -> List[str]:
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    frame_names.sort(key=lambda p: int(p))
    return frame_names


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    denom = float(mask_a.sum() + mask_b.sum())
    if denom == 0.0:
        return 1.0
    inter = float(np.logical_and(mask_a, mask_b).sum())
    return float((2.0 * inter) / denom)


def load_gt_masks(
    ann_root: str,
    video_name: str,
    frame_names: Sequence[str],
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


def collect_episode_specs(
    dataset_roots: Sequence[str],
    max_videos: int,
    max_objects_per_video: int,
) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    for dataset_root in dataset_roots:
        dataset_root = os.path.abspath(dataset_root)
        dataset_name = os.path.basename(os.path.normpath(os.path.dirname(dataset_root)))
        video_root = os.path.join(dataset_root, "JPEGImages")
        ann_root = os.path.join(dataset_root, "Annotations")
        video_names = sorted(
            [p for p in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, p))]
        )
        if max_videos > 0:
            video_names = video_names[:max_videos]

        for video_name in video_names:
            video_dir = os.path.join(video_root, video_name)
            frame_names = sorted_frame_names(video_dir)
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

            object_ids = sorted(inputs_per_object)
            if max_objects_per_video > 0:
                object_ids = object_ids[:max_objects_per_video]
            for object_id in object_ids:
                input_frame_idx = min(inputs_per_object[object_id])
                specs.append(
                    EpisodeSpec(
                        dataset_name=dataset_name,
                        video_name=video_name,
                        video_dir=video_dir,
                        ann_root=ann_root,
                        frame_names=list(frame_names),
                        object_id=int(object_id),
                        input_frame_idx=int(input_frame_idx),
                        input_mask=inputs_per_object[object_id][input_frame_idx].astype(bool),
                        group_key=f"{dataset_name}::{video_name}",
                    )
                )
    return specs


def split_specs_by_group(
    specs: Sequence[EpisodeSpec],
    val_ratio: float,
    seed: int,
) -> Tuple[List[EpisodeSpec], List[EpisodeSpec]]:
    unique_groups = sorted({spec.group_key for spec in specs})
    if len(unique_groups) == 1:
        duplicated = list(specs)
        return duplicated, duplicated
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    num_val = max(1, int(math.ceil(len(unique_groups) * val_ratio)))
    num_val = min(num_val, len(unique_groups) - 1)
    val_groups = set(unique_groups[:num_val])
    train_specs = [spec for spec in specs if spec.group_key not in val_groups]
    val_specs = [spec for spec in specs if spec.group_key in val_groups]
    return train_specs, val_specs


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    return returns


def build_step_rewards(
    dice_scores: Sequence[float],
    actions: Sequence[float],
    reward_horizon: int,
    write_cost: float,
) -> np.ndarray:
    rewards = np.zeros(len(actions), dtype=np.float32)
    for idx, action in enumerate(actions):
        future_start = idx + 1
        future_end = min(len(dice_scores), future_start + max(1, reward_horizon))
        if future_start < future_end:
            quality_reward = float(np.mean(dice_scores[future_start:future_end]))
        elif idx < len(dice_scores):
            quality_reward = float(dice_scores[idx])
        else:
            quality_reward = 0.0
        rewards[idx] = quality_reward - float(write_cost) * float(action)
    return rewards


def evaluate_policy_on_specs(
    predictor,
    inference_state_cache: Dict[str, Mapping[str, object]],
    specs: Sequence[EpisodeSpec],
    policy: OfflineWriteController,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    reward_horizon: int,
    write_cost: float,
    gamma: float,
    threshold: float,
    use_bfloat16: bool,
) -> Dict[str, float]:
    episode_dice = []
    episode_returns = []
    episode_write_rates = []
    episode_controller_prob_means = []
    episode_prob_ge_threshold_rates = []
    for spec in specs:
        result = run_episode(
            predictor=predictor,
            inference_state_cache=inference_state_cache,
            spec=spec,
            policy=policy,
            feature_mean=feature_mean,
            feature_std=feature_std,
            reward_horizon=reward_horizon,
            write_cost=write_cost,
            gamma=gamma,
            threshold=threshold,
            train_mode=False,
            use_bfloat16=use_bfloat16,
        )
        episode_dice.append(result["mean_dice"])
        episode_returns.append(result["episode_return"])
        episode_write_rates.append(result["write_rate"])
        episode_controller_prob_means.append(result["mean_controller_prob"])
        episode_prob_ge_threshold_rates.append(result["prob_ge_threshold_rate"])
    return {
        "mean_dice": float(np.mean(episode_dice)) if episode_dice else 0.0,
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_write_rate": float(np.mean(episode_write_rates)) if episode_write_rates else 0.0,
        "mean_controller_prob": (
            float(np.mean(episode_controller_prob_means))
            if episode_controller_prob_means
            else 0.0
        ),
        "mean_prob_ge_threshold_rate": (
            float(np.mean(episode_prob_ge_threshold_rates))
            if episode_prob_ge_threshold_rates
            else 0.0
        ),
    }


def run_episode(
    predictor,
    inference_state_cache: Dict[str, Mapping[str, object]],
    spec: EpisodeSpec,
    policy: OfflineWriteController,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    reward_horizon: int,
    write_cost: float,
    gamma: float,
    threshold: float,
    train_mode: bool,
    use_bfloat16: bool,
) -> Dict[str, object]:
    inference_state = inference_state_cache.get(spec.video_dir)
    if inference_state is None:
        inference_state = predictor.init_state(
            video_path=spec.video_dir,
            async_loading_frames=False,
        )
        inference_state_cache[spec.video_dir] = inference_state

    predictor.reset_state(inference_state)
    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=spec.input_frame_idx,
        obj_id=spec.object_id,
        mask=spec.input_mask,
    )

    gt_per_frame = load_gt_masks(spec.ann_root, spec.video_name, spec.frame_names)
    zero_mask = np.zeros_like(spec.input_mask, dtype=bool)

    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    actions: List[float] = []
    dice_scores: List[float] = []
    controller_probs: List[float] = []

    def external_policy(debug: Dict[str, object]) -> Dict[str, object]:
        features = build_feature_vector(debug, DEFAULT_WRITE_FEATURE_NAMES)
        features = ((features - feature_mean) / feature_std).astype(np.float32)
        with torch.inference_mode(False), torch.enable_grad():
            x = torch.tensor(features.tolist(), dtype=torch.float32).unsqueeze(0)
            logits = policy(x).squeeze(0)
            prob = torch.sigmoid(logits)
            prob_value = float(prob.detach().cpu().item())
            if train_mode:
                dist = Bernoulli(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
            else:
                action = (prob >= threshold).to(dtype=torch.float32)
        action_value = float(action.item())
        actions.append(action_value)
        controller_probs.append(prob_value)
        return {
            "write_memory": bool(action_value >= 0.5),
            "action": "WRITE_NORMAL" if action_value >= 0.5 else "SKIP_WRITE",
            "reason": "online_rl_policy",
            "controller_prob": prob_value,
            "controller_threshold": float(threshold),
        }

    predictor.set_external_write_policy(external_policy)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_context = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if use_bfloat16 and device_type == "cuda"
        else contextlib.nullcontext()
    )
    try:
        with torch.inference_mode(), amp_context:
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=spec.input_frame_idx,
                reverse=False,
            ):
                if out_frame_idx == spec.input_frame_idx:
                    continue
                gt_mask = gt_per_frame[out_frame_idx].get(spec.object_id, zero_mask).astype(bool)
                pred_mask = (out_mask_logits[0] > 0).detach().cpu().numpy().astype(bool)
                dice_scores.append(mask_dice(pred_mask, gt_mask))
    finally:
        predictor.clear_external_write_policy()

    if len(actions) != len(dice_scores):
        raise RuntimeError(
            f"Action/reward length mismatch for {spec.video_name} obj={spec.object_id}: "
            f"{len(actions)} actions vs {len(dice_scores)} dice scores"
        )

    rewards = build_step_rewards(
        dice_scores=dice_scores,
        actions=actions,
        reward_horizon=reward_horizon,
        write_cost=write_cost,
    )
    returns = discounted_returns(rewards, gamma=gamma)
    episode_return = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    mean_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    write_rate = float(np.mean(actions)) if len(actions) > 0 else 0.0
    mean_controller_prob = float(np.mean(controller_probs)) if controller_probs else 0.0
    prob_ge_threshold_rate = (
        float(np.mean(np.asarray(controller_probs, dtype=np.float32) >= float(threshold)))
        if controller_probs
        else 0.0
    )

    return {
        "log_probs": log_probs,
        "entropies": entropies,
        "actions": actions,
        "controller_probs": controller_probs,
        "dice_scores": dice_scores,
        "rewards": rewards,
        "returns": returns,
        "episode_return": episode_return,
        "mean_dice": mean_dice,
        "write_rate": write_rate,
        "mean_controller_prob": mean_controller_prob,
        "prob_ge_threshold_rate": prob_ge_threshold_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_roots",
        type=str,
        nargs="+",
        default=["./dataset/VOS-Endovis18/train"],
    )
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_s_endo18.pth",
    )
    parser.add_argument("--output_ckpt", type=str, required=True)
    parser.add_argument("--output_metrics_json", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--reward_horizon", type=int, default=8)
    parser.add_argument("--write_cost", type=float, default=0.01)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--baseline_momentum", type=float, default=0.9)
    parser.add_argument("--normalize_advantages", action="store_true")
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--max_objects_per_video", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--disable_bfloat16", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs = collect_episode_specs(
        dataset_roots=args.dataset_roots,
        max_videos=args.max_videos,
        max_objects_per_video=args.max_objects_per_video,
    )
    if not specs:
        raise ValueError("No training episodes found in the provided dataset roots")

    train_specs, val_specs = split_specs_by_group(specs, args.val_ratio, args.seed)
    if not train_specs or not val_specs:
        raise ValueError("Online RL split failed; adjust val_ratio or provide more videos")

    feature_mean = np.zeros((len(DEFAULT_WRITE_FEATURE_NAMES),), dtype=np.float32)
    feature_std = np.ones((len(DEFAULT_WRITE_FEATURE_NAMES),), dtype=np.float32)
    policy = OfflineWriteController(
        input_dim=len(DEFAULT_WRITE_FEATURE_NAMES),
        hidden_dims=args.hidden_dims,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
    inference_state_cache: Dict[str, Mapping[str, object]] = {}
    baseline = 0.0
    best_state = None
    best_metrics = None
    use_bfloat16 = not args.disable_bfloat16

    for epoch in range(args.epochs):
        random.shuffle(train_specs)
        policy.train()
        train_returns = []
        train_dice = []
        train_write_rates = []
        train_controller_prob_means = []
        train_prob_ge_threshold_rates = []
        train_losses = []

        for spec in train_specs:
            episode = run_episode(
                predictor=predictor,
                inference_state_cache=inference_state_cache,
                spec=spec,
                policy=policy,
                feature_mean=feature_mean,
                feature_std=feature_std,
                reward_horizon=args.reward_horizon,
                write_cost=args.write_cost,
                gamma=args.gamma,
                threshold=args.threshold,
                train_mode=True,
                use_bfloat16=use_bfloat16,
            )
            if len(episode["log_probs"]) == 0:
                continue

            returns = torch.from_numpy(episode["returns"])
            baseline = args.baseline_momentum * baseline + (1.0 - args.baseline_momentum) * float(
                returns.mean().item()
            )
            advantages = returns - baseline
            if args.normalize_advantages and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-6
                )

            log_probs = torch.stack(episode["log_probs"])
            entropies = torch.stack(episode["entropies"])
            loss = -(log_probs * advantages.detach()).sum() - args.entropy_coef * entropies.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            train_returns.append(float(episode["episode_return"]))
            train_dice.append(float(episode["mean_dice"]))
            train_write_rates.append(float(episode["write_rate"]))
            train_controller_prob_means.append(float(episode["mean_controller_prob"]))
            train_prob_ge_threshold_rates.append(float(episode["prob_ge_threshold_rate"]))
            train_losses.append(float(loss.item()))

        policy.eval()
        with torch.no_grad():
            val_metrics = evaluate_policy_on_specs(
                predictor=predictor,
                inference_state_cache=inference_state_cache,
                specs=val_specs,
                policy=policy,
                feature_mean=feature_mean,
                feature_std=feature_std,
                reward_horizon=args.reward_horizon,
                write_cost=args.write_cost,
                gamma=args.gamma,
                threshold=args.threshold,
                use_bfloat16=use_bfloat16,
            )

        print(
            f"epoch={epoch+1}/{args.epochs} "
            f"train_dice={np.mean(train_dice) if train_dice else 0.0:.4f} "
            f"train_return={np.mean(train_returns) if train_returns else 0.0:.4f} "
            f"train_write_rate={np.mean(train_write_rates) if train_write_rates else 0.0:.4f} "
            f"train_ctrl_prob={np.mean(train_controller_prob_means) if train_controller_prob_means else 0.0:.4f} "
            f"train_prob_ge_threshold={np.mean(train_prob_ge_threshold_rates) if train_prob_ge_threshold_rates else 0.0:.4f} "
            f"train_loss={np.mean(train_losses) if train_losses else 0.0:.4f} "
            f"val_dice={val_metrics['mean_dice']:.4f} "
            f"val_return={val_metrics['mean_return']:.4f} "
            f"val_write_rate={val_metrics['mean_write_rate']:.4f} "
            f"val_ctrl_prob={val_metrics['mean_controller_prob']:.4f} "
            f"val_prob_ge_threshold={val_metrics['mean_prob_ge_threshold_rate']:.4f}"
        )

        if best_metrics is None or val_metrics["mean_dice"] > best_metrics["mean_dice"]:
            best_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    assert best_state is not None
    policy.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_ckpt)), exist_ok=True)
    ckpt = {
        "state_dict": policy.state_dict(),
        "feature_names": list(DEFAULT_WRITE_FEATURE_NAMES),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "threshold": float(args.threshold),
        "hidden_dims": [int(v) for v in args.hidden_dims],
        "training_mode": "online_reinforce_gt_dice_skip_write",
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "memory_prune_mode": "efp",
        "gamma": float(args.gamma),
        "reward_horizon": int(args.reward_horizon),
        "write_cost": float(args.write_cost),
        "best_val_dice": float(best_metrics["mean_dice"]) if best_metrics else 0.0,
        "best_val_return": float(best_metrics["mean_return"]) if best_metrics else 0.0,
        "best_val_write_rate": float(best_metrics["mean_write_rate"]) if best_metrics else 0.0,
        "best_val_controller_prob": (
            float(best_metrics["mean_controller_prob"]) if best_metrics else 0.0
        ),
        "best_val_prob_ge_threshold_rate": (
            float(best_metrics["mean_prob_ge_threshold_rate"]) if best_metrics else 0.0
        ),
        "num_train_episodes": int(len(train_specs)),
        "num_val_episodes": int(len(val_specs)),
    }
    torch.save(ckpt, args.output_ckpt)
    print(f"saved online RL controller checkpoint to {args.output_ckpt}")

    if args.output_metrics_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_metrics_json)), exist_ok=True)
        with open(args.output_metrics_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold": float(args.threshold),
                    "training_mode": "online_reinforce_gt_dice_skip_write",
                    "sam2_checkpoint": str(args.sam2_checkpoint),
                    "memory_prune_mode": "efp",
                    "gamma": float(args.gamma),
                    "reward_horizon": int(args.reward_horizon),
                    "write_cost": float(args.write_cost),
                    "best_val_dice": float(best_metrics["mean_dice"]) if best_metrics else 0.0,
                    "best_val_return": float(best_metrics["mean_return"]) if best_metrics else 0.0,
                    "best_val_write_rate": float(best_metrics["mean_write_rate"]) if best_metrics else 0.0,
                    "best_val_controller_prob": (
                        float(best_metrics["mean_controller_prob"]) if best_metrics else 0.0
                    ),
                    "best_val_prob_ge_threshold_rate": (
                        float(best_metrics["mean_prob_ge_threshold_rate"])
                        if best_metrics
                        else 0.0
                    ),
                    "num_train_episodes": int(len(train_specs)),
                    "num_val_episodes": int(len(val_specs)),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
