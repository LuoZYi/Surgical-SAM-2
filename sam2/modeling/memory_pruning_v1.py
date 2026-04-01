"""
Utilities for memory-bank pruning in SAM2NewBase.

This file keeps the pruning logic separate from the model plumbing so you can run
clean ablations:

- prune_mode:
    off | efp | rule_based | state_aware

- score_mode (used by rule_based / state_aware):
    cosine_only
    cosine_motion
    cosine_motion_geometry

Design notes
------------
1) `efp` preserves the released SurgSAM2 behavior as closely as possible:
   candidate memories are compared to the latest memory using cosine similarity,
   and the most similar frames are dropped.

2) `rule_based` uses an explicit redundancy score:
      redundancy = appearance similarity
                  - motion diversity
                  + geometry overlap
                  + optional temporal closeness bonus

3) `state_aware` keeps the same score family, but adapts pruning aggressiveness
   and score weights online from a light-weight state:
      s_t = [redundancy, motion_change, confidence]

All functions are inference-safe and gracefully fall back if some metadata
(e.g. masks / IoU / object score) is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# -------------------------
# Small generic helpers
# -------------------------

def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except Exception:
        return default


def _as_pair(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().flatten().tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None
    return None


def _as_box(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().flatten().tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return float(value[0]), float(value[1]), float(value[2]), float(value[3])
        except Exception:
            return None
    return None


def _clamp_tensor_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _bbox_iou_xyxy(
    box1: Optional[Tuple[float, float, float, float]],
    box2: Optional[Tuple[float, float, float, float]],
) -> float:
    if box1 is None or box2 is None:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 1e-8:
        return 0.0
    return inter / union


# -------------------------
# Appearance similarity
# -------------------------

def cosine_similarity_to_last_raw(
    candidate_frames: torch.Tensor,
    last_frame: torch.Tensor,
) -> torch.Tensor:
    """
    Exact EFP-style aggregation.

    candidate_frames: [N, HW, B, C]
    last_frame:       [HW, B, C]
    returns:          [N]

    This matches the inlined SurgSAM2/SAM2NewBase-style behavior:
      1) cosine similarity along channel dim
      2) sum over token dim
      3) mean over batch dim
    """
    if candidate_frames.numel() == 0:
        return candidate_frames.new_zeros((0,))
    last_frame_expanded = last_frame.unsqueeze(0).expand_as(candidate_frames)
    similarities = F.cosine_similarity(candidate_frames, last_frame_expanded, dim=-1)
    similarities = similarities.sum(dim=1)   # [N, B]
    similarities = similarities.mean(dim=1)  # [N]
    return similarities


def cosine_similarity_to_last_mean(
    candidate_frames: torch.Tensor,
    last_frame: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-normalized cosine similarity in [-1, 1], better for readable scores and
    thresholding in rule-based / state-aware modes.
    """
    if candidate_frames.numel() == 0:
        return candidate_frames.new_zeros((0,))
    last_frame_expanded = last_frame.unsqueeze(0).expand_as(candidate_frames)
    similarities = F.cosine_similarity(candidate_frames, last_frame_expanded, dim=-1)
    similarities = similarities.mean(dim=1)  # [N, B]
    similarities = similarities.mean(dim=1)  # [N]
    return similarities


# -------------------------
# Metadata-driven signals
# -------------------------

def temporal_closeness_to_last(
    candidate_meta: Sequence[Dict[str, Any]],
    last_meta: Dict[str, Any],
    device: torch.device,
    min_temporal_gap: int = 0,
) -> torch.Tensor:
    """
    Higher => temporally closer to the latest memory => more likely redundant.
    Range is roughly [0, 1], with a small extra boost if the gap is smaller than
    `min_temporal_gap`.
    """
    last_idx = last_meta.get("frame_idx", None)
    scores: List[float] = []

    for meta in candidate_meta:
        cand_idx = meta.get("frame_idx", None)
        if last_idx is None or cand_idx is None:
            scores.append(0.0)
            continue

        gap = abs(int(last_idx) - int(cand_idx))
        closeness = 1.0 / float(gap + 1)
        if min_temporal_gap > 0 and gap < min_temporal_gap:
            closeness += 0.5
        scores.append(closeness)

    if len(scores) == 0:
        return torch.zeros(0, device=device)
    return torch.tensor(scores, device=device, dtype=torch.float32)


def pairwise_motion_to_last(
    candidate_meta: Sequence[Dict[str, Any]],
    last_meta: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Higher => more motion / geometry difference from latest memory => *less* redundant.
    Uses normalized centroid distance and a small area-change term.
    """
    last_centroid = _as_pair(last_meta.get("centroid_xy"))
    last_area = _as_float(last_meta.get("area_ratio"), default=0.0)

    vals: List[float] = []
    for meta in candidate_meta:
        cand_centroid = _as_pair(meta.get("centroid_xy"))
        cand_area = _as_float(meta.get("area_ratio"), default=0.0)

        centroid_dist = 0.0
        if last_centroid is not None and cand_centroid is not None:
            dx = cand_centroid[0] - last_centroid[0]
            dy = cand_centroid[1] - last_centroid[1]
            centroid_dist = (dx * dx + dy * dy) ** 0.5
            centroid_dist = min(1.0, centroid_dist / (2.0 ** 0.5))  # normalize by diagonal

        area_change = min(1.0, abs(cand_area - last_area))
        motion = 0.8 * centroid_dist + 0.2 * area_change
        vals.append(float(motion))

    if len(vals) == 0:
        return torch.zeros(0, device=device)
    return torch.tensor(vals, device=device, dtype=torch.float32)


def pairwise_geometry_overlap_to_last(
    candidate_meta: Sequence[Dict[str, Any]],
    last_meta: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Higher => more geometry overlap with latest memory => more redundant.
    Uses bbox IoU and area similarity.
    """
    last_box = _as_box(last_meta.get("bbox_xyxy"))
    last_area = _as_float(last_meta.get("area_ratio"), default=0.0)

    vals: List[float] = []
    for meta in candidate_meta:
        cand_box = _as_box(meta.get("bbox_xyxy"))
        cand_area = _as_float(meta.get("area_ratio"), default=0.0)

        bbox_iou = _bbox_iou_xyxy(cand_box, last_box)
        area_similarity = max(0.0, 1.0 - abs(cand_area - last_area))
        overlap = 0.7 * bbox_iou + 0.3 * area_similarity
        vals.append(float(overlap))

    if len(vals) == 0:
        return torch.zeros(0, device=device)
    return torch.tensor(vals, device=device, dtype=torch.float32)


def estimate_online_state(
    sim_mean_to_last: torch.Tensor,
    memory_meta: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, float]:
    """
    Estimate a tiny online state for the adaptive controller.

    Returns:
        {
            "redundancy": float in [0, 1] approx,
            "motion":     float in [0, 1],
            "confidence": float in [0, 1],
        }
    """
    if len(memory_meta) == 0:
        return {"redundancy": 0.0, "motion": 0.0, "confidence": 0.5}

    last_meta = memory_meta[-1]
    prev_meta = memory_meta[-2] if len(memory_meta) >= 2 else None

    if sim_mean_to_last.numel() == 0:
        redundancy = 0.0
    else:
        redundancy = float(_clamp_tensor_01(sim_mean_to_last.max()).item())

    # Motion: prefer the latest explicit motion metadata; otherwise derive from
    # latest-vs-previous geometry.
    motion = _as_float(last_meta.get("motion_mag"), default=-1.0)
    if motion < 0.0 and prev_meta is not None:
        motion_tensor = pairwise_motion_to_last([prev_meta], last_meta, device=device)
        motion = float(motion_tensor[0].item()) if motion_tensor.numel() > 0 else 0.0
    if motion < 0.0:
        motion = 0.0
    motion = max(0.0, min(1.0, motion))

    pred_iou = _as_float(last_meta.get("pred_iou"), default=-1.0)
    if pred_iou >= 0.0:
        confidence = max(0.0, min(1.0, pred_iou))
    else:
        obj_score = last_meta.get("obj_score", None)
        if obj_score is None:
            confidence = 0.5
        else:
            obj_score = _as_float(obj_score, default=0.0)
            confidence = float(torch.sigmoid(torch.tensor(obj_score)).item())
            confidence = max(0.0, min(1.0, confidence))

    return {
        "redundancy": redundancy,
        "motion": motion,
        "confidence": confidence,
    }


# -------------------------
# Controller
# -------------------------

DEFAULT_CONTROLLER_CFG: Dict[str, float] = {
    "redundancy_high": 0.72,
    "motion_low": 0.08,
    "motion_high": 0.18,
    "confidence_high": 0.72,
    "confidence_low": 0.45,
}

POLICY_PRESETS: Dict[str, Dict[str, float]] = {
    "conservative": {
        "prune_delta": -1.0,
        "motion_weight": 0.60,
        "geometry_weight": 0.10,
        "temporal_bonus_weight": 0.00,
    },
    "normal": {
        "prune_delta": 0.0,
        "motion_weight": 0.35,
        "geometry_weight": 0.25,
        "temporal_bonus_weight": 0.00,
    },
    "aggressive": {
        "prune_delta": 1.0,
        "motion_weight": 0.15,
        "geometry_weight": 0.35,
        "temporal_bonus_weight": 0.05,
    },
}


def select_state_policy(
    state: Dict[str, float],
    controller_cfg: Optional[Dict[str, float]] = None,
) -> str:
    cfg = dict(DEFAULT_CONTROLLER_CFG)
    if controller_cfg is not None:
        cfg.update(controller_cfg)

    r = state["redundancy"]
    m = state["motion"]
    c = state["confidence"]

    if (c < cfg["confidence_low"]) or (m > cfg["motion_high"]):
        return "conservative"
    if (r > cfg["redundancy_high"]) and (m < cfg["motion_low"]) and (c > cfg["confidence_high"]):
        return "aggressive"
    return "normal"


# -------------------------
# Score computation
# -------------------------

def compute_rule_based_scores(
    score_mode: str,
    candidate_frames: torch.Tensor,
    last_frame: torch.Tensor,
    candidate_meta: Sequence[Dict[str, Any]],
    last_meta: Dict[str, Any],
    min_temporal_gap: int = 0,
    motion_weight: float = 0.35,
    geometry_weight: float = 0.25,
    temporal_bonus_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Larger score => more redundant => more likely to be pruned.
    """
    device = last_frame.device

    sim_mean = cosine_similarity_to_last_mean(candidate_frames, last_frame)
    sim_raw = cosine_similarity_to_last_raw(candidate_frames, last_frame)

    motion = pairwise_motion_to_last(candidate_meta, last_meta, device=device)
    geometry = pairwise_geometry_overlap_to_last(candidate_meta, last_meta, device=device)
    temporal = temporal_closeness_to_last(
        candidate_meta, last_meta, device=device, min_temporal_gap=min_temporal_gap
    )

    scores = sim_mean.clone()

    if score_mode == "cosine_only":
        scores = sim_mean
    elif score_mode == "cosine_motion":
        scores = sim_mean - motion_weight * motion
    elif score_mode == "cosine_motion_geometry":
        scores = sim_mean - motion_weight * motion + geometry_weight * geometry
    else:
        raise ValueError(
            f"Unknown score_mode={score_mode!r}. Expected one of: "
            "cosine_only | cosine_motion | cosine_motion_geometry."
        )

    if temporal_bonus_weight != 0.0:
        scores = scores + temporal_bonus_weight * temporal

    debug = {
        "score_mode": score_mode,
        "candidate_similarities_raw": sim_raw.detach().cpu().tolist(),
        "candidate_similarities_mean": sim_mean.detach().cpu().tolist(),
        "candidate_motion": motion.detach().cpu().tolist(),
        "candidate_geometry_overlap": geometry.detach().cpu().tolist(),
        "candidate_temporal_closeness": temporal.detach().cpu().tolist(),
        "candidate_drop_scores": scores.detach().cpu().tolist(),
        "motion_weight": motion_weight,
        "geometry_weight": geometry_weight,
        "temporal_bonus_weight": temporal_bonus_weight,
    }
    return scores, debug


# -------------------------
# Selection
# -------------------------

def select_delete_indices_by_scores(
    base_indices: List[int],
    scores: torch.Tensor,
    metas: Sequence[Dict[str, Any]],
    num_to_prune: int,
    prefer_non_cond: bool = False,
    similarity_threshold: Optional[float] = None,
) -> List[int]:
    """
    Return actual indices in `to_cat_memory` to delete.
    Larger score => more likely to be pruned.
    """
    if num_to_prune <= 0 or len(base_indices) == 0:
        return []

    if similarity_threshold is not None:
        eligible_local = [
            i for i, score in enumerate(scores.detach().cpu().tolist())
            if score >= similarity_threshold
        ]
    else:
        eligible_local = list(range(len(base_indices)))

    def rank_subset(local_ids: List[int], k: int) -> List[int]:
        if k <= 0 or len(local_ids) == 0:
            return []
        subset_scores = scores[local_ids]
        _, subset_order = torch.sort(subset_scores, descending=True)
        chosen_local_ids = [local_ids[j] for j in subset_order[:k].tolist()]
        return [base_indices[j] for j in chosen_local_ids]

    if not prefer_non_cond:
        chosen = rank_subset(eligible_local, min(num_to_prune, len(eligible_local)))
        if len(chosen) < num_to_prune:
            already = set(chosen)
            remaining_local = [
                i for i in range(len(base_indices))
                if base_indices[i] not in already
            ]
            chosen.extend(rank_subset(remaining_local, num_to_prune - len(chosen)))
        return chosen

    non_cond_local = [
        i for i in eligible_local if not metas[i].get("is_conditioning", False)
    ]
    cond_local = [
        i for i in eligible_local if metas[i].get("is_conditioning", False)
    ]

    chosen = rank_subset(non_cond_local, min(num_to_prune, len(non_cond_local)))
    still_need = num_to_prune - len(chosen)

    if still_need > 0:
        already = set(chosen)
        cond_remaining = [
            i for i in cond_local
            if base_indices[i] not in already
        ]
        chosen.extend(rank_subset(cond_remaining, still_need))

    if len(chosen) < num_to_prune:
        already = set(chosen)
        remaining_local = [
            i for i in range(len(base_indices))
            if base_indices[i] not in already
        ]
        chosen.extend(rank_subset(remaining_local, num_to_prune - len(chosen)))

    return chosen


# -------------------------
# Main entry point
# -------------------------

def plan_memory_pruning(
    *,
    prune_mode: str,
    score_mode: str,
    to_cat_memory: Sequence[torch.Tensor],
    memory_meta: Sequence[Dict[str, Any]],
    num_maskmem: int,
    num_frame_to_prune: int,
    protect_conditioning_memories: bool = False,
    memory_similarity_threshold: Optional[float] = None,
    memory_min_temporal_gap: int = 0,
    controller_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Compute which indices in `to_cat_memory` should be pruned.

    Returns:
        delete_indices: actual indices in `to_cat_memory`
        debug: detailed debug payload
    """
    n = len(to_cat_memory)
    base_debug: Dict[str, Any] = {
        "mode": prune_mode,
        "score_mode": score_mode,
        "num_before": n,
        "num_after": n,
        "pruned_indices": [],
        "pruned_frame_idx": [],
    }

    if n <= 2:
        return [], base_debug

    if (n + num_frame_to_prune) <= num_maskmem:
        return [], base_debug

    num_candidates = n - 2
    base_num_to_prune = min(n + num_frame_to_prune - num_maskmem, num_candidates)
    if base_num_to_prune <= 0:
        return [], base_debug

    last_frame = to_cat_memory[-1]
    candidate_frames = torch.stack(list(to_cat_memory[1:-1]), dim=0)
    candidate_meta = list(memory_meta[1:-1])
    last_meta = memory_meta[-1]
    base_indices = list(range(1, n - 1))

    if prune_mode == "efp":
        similarities_raw = cosine_similarity_to_last_raw(candidate_frames, last_frame)
        similarities_mean = cosine_similarity_to_last_mean(candidate_frames, last_frame)

        _, sorted_indices = torch.sort(similarities_raw, descending=True)
        delete_indices = (sorted_indices[:base_num_to_prune] + 1).tolist()
        delete_indices = sorted(delete_indices, reverse=True)

        debug = dict(base_debug)
        debug.update(
            {
                "mode": "efp",
                "num_to_prune": base_num_to_prune,
                "candidate_frame_idx": [m.get("frame_idx") for m in candidate_meta],
                "candidate_is_conditioning": [m.get("is_conditioning") for m in candidate_meta],
                "candidate_similarities_raw": similarities_raw.detach().cpu().tolist(),
                "candidate_similarities_mean": similarities_mean.detach().cpu().tolist(),
            }
        )
        return delete_indices, debug

    if prune_mode not in {"rule_based", "state_aware"}:
        raise ValueError(
            f"Unknown prune_mode={prune_mode!r}. Expected one of: "
            "efp | rule_based | state_aware."
        )

    # Rule-based baseline weights.
    policy_name = "normal"
    motion_weight = 0.35
    geometry_weight = 0.25
    temporal_bonus_weight = 0.0
    num_to_prune = base_num_to_prune

    if prune_mode == "state_aware":
        sim_mean = cosine_similarity_to_last_mean(candidate_frames, last_frame)
        state = estimate_online_state(sim_mean_to_last=sim_mean, memory_meta=memory_meta, device=last_frame.device)
        policy_name = select_state_policy(state, controller_cfg=controller_cfg)
        preset = POLICY_PRESETS[policy_name]

        num_to_prune = int(base_num_to_prune + preset["prune_delta"])
        num_to_prune = max(1, min(num_to_prune, num_candidates))

        motion_weight = float(preset["motion_weight"])
        geometry_weight = float(preset["geometry_weight"])
        temporal_bonus_weight = float(preset["temporal_bonus_weight"])
    else:
        state = None

    scores, score_debug = compute_rule_based_scores(
        score_mode=score_mode,
        candidate_frames=candidate_frames,
        last_frame=last_frame,
        candidate_meta=candidate_meta,
        last_meta=last_meta,
        min_temporal_gap=memory_min_temporal_gap,
        motion_weight=motion_weight,
        geometry_weight=geometry_weight,
        temporal_bonus_weight=temporal_bonus_weight,
    )

    delete_indices = select_delete_indices_by_scores(
        base_indices=base_indices,
        scores=scores,
        metas=candidate_meta,
        num_to_prune=num_to_prune,
        prefer_non_cond=protect_conditioning_memories,
        similarity_threshold=memory_similarity_threshold,
    )
    delete_indices = sorted(delete_indices, reverse=True)

    debug = dict(base_debug)
    debug.update(
        {
            "mode": prune_mode,
            "policy_name": policy_name,
            "num_to_prune": num_to_prune,
            "base_num_to_prune": base_num_to_prune,
            "candidate_frame_idx": [m.get("frame_idx") for m in candidate_meta],
            "candidate_is_conditioning": [m.get("is_conditioning") for m in candidate_meta],
            "state": state,
        }
    )
    debug.update(score_debug)
    return delete_indices, debug
