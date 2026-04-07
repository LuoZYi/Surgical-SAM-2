#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict, deque
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sam2.modeling.memory_write_controller import (
    DEFAULT_MODE_WINDOW_FEATURE_NAMES,
    OfflineWriteController,
    build_feature_vector,
    build_mode_window_feature_source,
)


def load_rows(csv_paths: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for csv_path in csv_paths:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    return rows


def group_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = f"{row['dataset_name']}::{row['video_name']}::{row['object_id']}"
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: int(row["frame_idx"]))
    return groups


def build_arrays(
    rows: Sequence[Dict[str, object]],
    window_size: int,
    low_iou_threshold: float,
    gain_threshold: float,
    novelty_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    grouped = group_rows(rows)
    features = []
    labels = []
    groups = []

    for group_key, group_rows_list in grouped.items():
        history: deque[Dict[str, object]] = deque(maxlen=max(1, window_size))
        for row in group_rows_list:
            source = {
                "quality": float(row["quality"]),
                "mask_cosine": float(row["mask_cosine"]),
                "mask_change": float(row["mask_change"]),
                "centroid_shift": float(row["centroid_shift"]),
                "current_area_ratio": float(row["current_area_ratio"]),
                "area_change": float(row["area_change"]),
                "reference_temporal_gap": float(row["reference_temporal_gap"]),
                "bank_fill_ratio": float(row["bank_fill_ratio"]),
                "pred_iou_to_gt": float(row["pred_iou_to_gt"]),
                "oracle_future_gain": float(row["oracle_future_gain"]),
                "oracle_gt_novelty": float(row["oracle_gt_novelty"]),
                "is_prompt_frame": int(row.get("is_prompt_frame", 0)),
            }
            history.append(source)

            if int(source["is_prompt_frame"]) == 1:
                continue

            window = list(history)
            feature_source = build_mode_window_feature_source(window)
            window_min_iou = min(float(item["pred_iou_to_gt"]) for item in window)
            window_mean_gain = float(np.mean([float(item["oracle_future_gain"]) for item in window]))
            window_max_novelty = max(float(item["oracle_gt_novelty"]) for item in window)
            challenging = int(
                (window_min_iou < low_iou_threshold)
                or (
                    window_mean_gain >= gain_threshold
                    and window_max_novelty >= novelty_threshold
                )
            )

            features.append(
                build_feature_vector(feature_source, DEFAULT_MODE_WINDOW_FEATURE_NAMES)
            )
            labels.append(float(challenging))
            dataset_name, video_name, _ = group_key.split("::", 2)
            groups.append(f"{dataset_name}::{video_name}")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        groups,
    )


def split_by_group(groups: Sequence[str], val_ratio: float, seed: int) -> Tuple[set[str], set[str]]:
    unique_groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    num_val = max(1, int(math.ceil(len(unique_groups) * val_ratio)))
    val_groups = set(unique_groups[:num_val])
    train_groups = set(unique_groups[num_val:])
    return train_groups, val_groups


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    best_score = -1.0
    best_tau = 0.5
    for tau in np.linspace(0.1, 0.9, 33):
        preds = probs >= tau
        tp = float(np.logical_and(preds, labels > 0.5).sum())
        fp = float(np.logical_and(preds, labels <= 0.5).sum())
        fn = float(np.logical_and(~preds, labels > 0.5).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_score:
            best_score = f1
            best_tau = float(tau)
    return best_tau, best_score


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Tuple[float, np.ndarray]:
    model.eval()
    probs = []
    with torch.no_grad():
        for batch_start in range(0, len(x), batch_size):
            logits = model(x[batch_start : batch_start + batch_size])
            probs.append(torch.sigmoid(logits).cpu())
    probs_np = torch.cat(probs, dim=0).numpy()
    loss = F.binary_cross_entropy(
        torch.from_numpy(probs_np).float(),
        y.cpu(),
    ).item()
    return float(loss), probs_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_ckpt", type=str, required=True)
    parser.add_argument("--output_metrics_json", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--label_low_iou_threshold", type=float, default=0.82)
    parser.add_argument("--label_gain_threshold", type=float, default=0.01)
    parser.add_argument("--label_novelty_threshold", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.csv_paths)
    x_all, y_all, groups = build_arrays(
        rows=rows,
        window_size=max(1, args.window_size),
        low_iou_threshold=args.label_low_iou_threshold,
        gain_threshold=args.label_gain_threshold,
        novelty_threshold=args.label_novelty_threshold,
    )

    train_groups, val_groups = split_by_group(groups, args.val_ratio, args.seed)
    train_mask = np.asarray([group in train_groups for group in groups], dtype=bool)
    val_mask = ~train_mask

    x_train = x_all[train_mask]
    y_train = y_all[train_mask]
    x_val = x_all[val_mask]
    y_val = y_all[val_mask]

    feature_mean = x_train.mean(axis=0, keepdims=True)
    feature_std = x_train.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    x_train = (x_train - feature_mean) / feature_std
    x_val = (x_val - feature_mean) / feature_std

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32)

    model = OfflineWriteController(
        input_dim=len(DEFAULT_MODE_WINDOW_FEATURE_NAMES),
        hidden_dims=args.hidden_dims,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    dataset = TensorDataset(x_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_items = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch_y,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)
            num_items += len(batch_x)
        train_loss = running_loss / max(num_items, 1)
        val_loss, _ = evaluate(model, x_val_t, y_val_t, args.batch_size)
        print(
            f"epoch={epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    _, val_probs = evaluate(model, x_val_t, y_val_t, args.batch_size)
    threshold, best_f1 = best_threshold(val_probs, y_val)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_ckpt)), exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "feature_names": list(DEFAULT_MODE_WINDOW_FEATURE_NAMES),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "threshold": float(threshold),
        "hidden_dims": [int(v) for v in args.hidden_dims],
        "val_f1": float(best_f1),
        "num_train": int(len(x_train)),
        "num_val": int(len(x_val)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
        "window_size": int(args.window_size),
        "label_low_iou_threshold": float(args.label_low_iou_threshold),
        "label_gain_threshold": float(args.label_gain_threshold),
        "label_novelty_threshold": float(args.label_novelty_threshold),
    }
    torch.save(ckpt, args.output_ckpt)
    print(f"saved mode detector checkpoint to {args.output_ckpt}")
    print(f"best_val_f1={best_f1:.4f} threshold={threshold:.3f}")

    if args.output_metrics_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_metrics_json)), exist_ok=True)
        with open(args.output_metrics_json, "w") as f:
            json.dump(
                {
                    "best_val_f1": float(best_f1),
                    "threshold": float(threshold),
                    "num_train": int(len(x_train)),
                    "num_val": int(len(x_val)),
                    "positive_rate_train": float(y_train.mean()),
                    "positive_rate_val": float(y_val.mean()),
                    "window_size": int(args.window_size),
                    "label_low_iou_threshold": float(args.label_low_iou_threshold),
                    "label_gain_threshold": float(args.label_gain_threshold),
                    "label_novelty_threshold": float(args.label_novelty_threshold),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
