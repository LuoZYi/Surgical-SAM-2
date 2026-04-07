#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Bernoulli

from sam2.modeling.memory_write_controller import (
    DEFAULT_WRITE_FEATURE_NAMES,
    OfflineWriteController,
)


@dataclass
class Episode:
    episode_key: str
    group_key: str
    features: np.ndarray
    future_gain: np.ndarray
    oracle_write: np.ndarray


def load_rows(csv_paths: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for csv_path in csv_paths:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def build_episodes(
    rows: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
) -> List[Episode]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    group_keys: Dict[str, str] = {}
    for row in rows:
        if int(row.get("is_prompt_frame", 0)) == 1:
            continue
        dataset_name = str(row["dataset_name"])
        video_name = str(row["video_name"])
        object_id = int(row["object_id"])
        episode_key = f"{dataset_name}::{video_name}::{object_id}"
        group_key = f"{dataset_name}::{video_name}"
        buckets.setdefault(episode_key, []).append(row)
        group_keys[episode_key] = group_key

    episodes: List[Episode] = []
    for episode_key, episode_rows in buckets.items():
        episode_rows = sorted(episode_rows, key=lambda row: int(row["frame_idx"]))
        features = np.asarray(
            [[float(row[name]) for name in feature_names] for row in episode_rows],
            dtype=np.float32,
        )
        future_gain = np.asarray(
            [float(row["oracle_future_gain"]) for row in episode_rows],
            dtype=np.float32,
        )
        oracle_write = np.asarray(
            [float(row["oracle_write"]) for row in episode_rows],
            dtype=np.float32,
        )
        episodes.append(
            Episode(
                episode_key=episode_key,
                group_key=group_keys[episode_key],
                features=features,
                future_gain=future_gain,
                oracle_write=oracle_write,
            )
        )
    return episodes


def split_by_group(
    episodes: Sequence[Episode],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Episode], List[Episode]]:
    unique_groups = sorted({episode.group_key for episode in episodes})
    if len(unique_groups) == 1:
        # Smoke tests or tiny debugging CSVs may only contain one video.
        duplicated = list(episodes)
        return duplicated, duplicated
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    num_val = max(1, int(math.ceil(len(unique_groups) * val_ratio)))
    num_val = min(num_val, len(unique_groups) - 1)
    val_groups = set(unique_groups[:num_val])
    train_episodes = [episode for episode in episodes if episode.group_key not in val_groups]
    val_episodes = [episode for episode in episodes if episode.group_key in val_groups]
    return train_episodes, val_episodes


def compute_feature_stats(episodes: Sequence[Episode]) -> Tuple[np.ndarray, np.ndarray]:
    x_train = np.concatenate([episode.features for episode in episodes], axis=0)
    feature_mean = x_train.mean(axis=0, keepdims=True)
    feature_std = x_train.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    return feature_mean.astype(np.float32), feature_std.astype(np.float32)


def normalize_episodes(
    episodes: Sequence[Episode],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> List[Episode]:
    normalized = []
    for episode in episodes:
        normalized.append(
            Episode(
                episode_key=episode.episode_key,
                group_key=episode.group_key,
                features=((episode.features - feature_mean) / feature_std).astype(np.float32),
                future_gain=episode.future_gain.copy(),
                oracle_write=episode.oracle_write.copy(),
            )
        )
    return normalized


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def rewards_from_actions(
    future_gain: torch.Tensor,
    actions: torch.Tensor,
    write_cost: float,
    skip_gain_scale: float,
) -> torch.Tensor:
    write_reward = future_gain - write_cost
    skip_reward = -skip_gain_scale * future_gain
    return actions * write_reward + (1.0 - actions) * skip_reward


def rewards_from_probs_np(
    future_gain: np.ndarray,
    actions: np.ndarray,
    write_cost: float,
    skip_gain_scale: float,
) -> np.ndarray:
    write_reward = future_gain - write_cost
    skip_reward = -skip_gain_scale * future_gain
    return actions * write_reward + (1.0 - actions) * skip_reward


def evaluate_thresholds(
    model: OfflineWriteController,
    episodes: Sequence[Episode],
    gamma: float,
    write_cost: float,
    skip_gain_scale: float,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    model.eval()
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for episode in episodes:
            x = torch.from_numpy(episode.features)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_probs.append(probs.astype(np.float32))

    best_tau = 0.5
    best_reward = -float("inf")
    for tau in thresholds:
        episode_returns = []
        for episode, probs in zip(episodes, all_probs):
            actions = (probs >= float(tau)).astype(np.float32)
            rewards = rewards_from_probs_np(
                episode.future_gain,
                actions,
                write_cost=write_cost,
                skip_gain_scale=skip_gain_scale,
            )
            total_return = 0.0
            running = 0.0
            for reward in rewards[::-1]:
                running = float(reward) + gamma * running
                total_return = running
            episode_returns.append(total_return)
        mean_reward = float(np.mean(episode_returns)) if episode_returns else -float("inf")
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_tau = float(tau)
    return best_tau, best_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_ckpt", type=str, required=True)
    parser.add_argument("--output_metrics_json", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--write_cost", type=float, default=0.01)
    parser.add_argument("--skip_gain_scale", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--normalize_returns", action="store_true")
    parser.add_argument("--baseline_momentum", type=float, default=0.9)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.csv_paths)
    episodes = build_episodes(rows, DEFAULT_WRITE_FEATURE_NAMES)
    if not episodes:
        raise ValueError("No usable non-prompt rows found in the provided CSV files")

    train_episodes, val_episodes = split_by_group(episodes, args.val_ratio, args.seed)
    if not train_episodes:
        raise ValueError("Training split is empty; adjust val_ratio or input data")
    if not val_episodes:
        raise ValueError("Validation split is empty; adjust val_ratio or input data")

    feature_mean, feature_std = compute_feature_stats(train_episodes)
    train_episodes = normalize_episodes(train_episodes, feature_mean, feature_std)
    val_episodes = normalize_episodes(val_episodes, feature_mean, feature_std)

    model = OfflineWriteController(
        input_dim=len(DEFAULT_WRITE_FEATURE_NAMES),
        hidden_dims=args.hidden_dims,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    thresholds = np.linspace(0.1, 0.9, 33)
    best_state = None
    best_threshold = 0.5
    best_val_reward = -float("inf")
    baseline = 0.0

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_episodes)
        epoch_policy_loss = 0.0
        epoch_episode_return = 0.0

        for episode in train_episodes:
            x = torch.from_numpy(episode.features)
            future_gain = torch.from_numpy(episode.future_gain)

            logits = model(x)
            dist = Bernoulli(logits=logits)
            actions = dist.sample()
            rewards = rewards_from_actions(
                future_gain=future_gain,
                actions=actions,
                write_cost=args.write_cost,
                skip_gain_scale=args.skip_gain_scale,
            )
            returns = discounted_returns(rewards, gamma=args.gamma)
            baseline = args.baseline_momentum * baseline + (1.0 - args.baseline_momentum) * float(
                returns.mean().item()
            )
            advantages = returns - baseline
            if args.normalize_returns and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-6
                )

            policy_loss = -(dist.log_prob(actions) * advantages.detach()).sum()
            entropy_bonus = dist.entropy().sum()
            loss = policy_loss - args.entropy_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_policy_loss += float(policy_loss.item())
            epoch_episode_return += float(returns[0].item())

        threshold, val_reward = evaluate_thresholds(
            model=model,
            episodes=val_episodes,
            gamma=args.gamma,
            write_cost=args.write_cost,
            skip_gain_scale=args.skip_gain_scale,
            thresholds=thresholds,
        )
        mean_train_return = epoch_episode_return / max(len(train_episodes), 1)
        mean_policy_loss = epoch_policy_loss / max(len(train_episodes), 1)
        print(
            f"epoch={epoch+1}/{args.epochs} "
            f"train_return={mean_train_return:.4f} "
            f"policy_loss={mean_policy_loss:.4f} "
            f"val_reward={val_reward:.4f} "
            f"best_tau={threshold:.3f}"
        )
        if val_reward > best_val_reward:
            best_val_reward = float(val_reward)
            best_threshold = float(threshold)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_ckpt)), exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "feature_names": list(DEFAULT_WRITE_FEATURE_NAMES),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "threshold": float(best_threshold),
        "hidden_dims": [int(v) for v in args.hidden_dims],
        "training_mode": "reinforce_skip_write",
        "gamma": float(args.gamma),
        "write_cost": float(args.write_cost),
        "skip_gain_scale": float(args.skip_gain_scale),
        "best_val_reward": float(best_val_reward),
        "num_train_episodes": int(len(train_episodes)),
        "num_val_episodes": int(len(val_episodes)),
        "num_train_steps": int(sum(len(episode.features) for episode in train_episodes)),
        "num_val_steps": int(sum(len(episode.features) for episode in val_episodes)),
    }
    torch.save(ckpt, args.output_ckpt)
    print(f"saved RL controller checkpoint to {args.output_ckpt}")
    print(f"best_val_reward={best_val_reward:.4f} threshold={best_threshold:.3f}")

    if args.output_metrics_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_metrics_json)), exist_ok=True)
        with open(args.output_metrics_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_reward": float(best_val_reward),
                    "threshold": float(best_threshold),
                    "training_mode": "reinforce_skip_write",
                    "gamma": float(args.gamma),
                    "write_cost": float(args.write_cost),
                    "skip_gain_scale": float(args.skip_gain_scale),
                    "num_train_episodes": int(len(train_episodes)),
                    "num_val_episodes": int(len(val_episodes)),
                    "num_train_steps": int(sum(len(episode.features) for episode in train_episodes)),
                    "num_val_steps": int(sum(len(episode.features) for episode in val_episodes)),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
