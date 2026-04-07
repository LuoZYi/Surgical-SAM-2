# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimally invasive extension of SAM2Base that keeps the upstream file untouched,
while making memory pruning modular and easy to ablate.

What this adds:
- prune mode switch:
    off | efp | rule_based | state_aware
- score mode switch for rule_based / state_aware:
    cosine_only | cosine_motion | cosine_motion_geometry
- write mode switch:
    off | quality_novelty_gate | adaptive_controller | adaptive_controller_v2 | adaptive_skip_write_v2 | learned_skip_write_controller | learned_adaptive_controller_v2 | sliding_window_challenging_controller_v1
- optional protection for conditioning memories
- lightweight debug bookkeeping via `self._last_memory_prune_debug`

Recommended usage:
    from sam2.modeling.sam2_newbase import SAM2NewBase

Then point your model builder / config at SAM2NewBase instead of SAM2Base.
"""

from __future__ import annotations

from collections import deque
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sam2.modeling.memory_pruning import plan_memory_pruning
from sam2.modeling.memory_write_controller import (
    DEFAULT_MODE_WINDOW_FEATURE_NAMES,
    DEFAULT_WRITE_FEATURE_NAMES,
    OfflineWriteController,
    build_feature_vector,
    build_mode_window_feature_source,
)
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames

if os.environ.get("SURGICAL_SAM2_USE_ORIGINAL_BASE") == "1":
    from sam2.modeling.sam2_base_original import SAM2Base
else:
    from sam2.modeling.sam2_base import SAM2Base


class SAM2NewBase(SAM2Base):
    def __init__(
        self,
        *args,
        memory_prune_mode: str = "efp",   # off | efp | rule_based | state_aware
        memory_score_mode: str = "cosine_only",  # cosine_only | cosine_motion | cosine_motion_geometry
        num_frame_to_prune: int = 2,
        protect_conditioning_memories: bool = False,
        memory_similarity_threshold: Optional[float] = None,
        memory_min_temporal_gap: int = 0,
        debug_memory_pruning: bool = False,
        state_controller_cfg: Optional[Dict[str, float]] = None,
        use_recent_memory_guard: bool = False,
        recent_memory_min_keep: int = 1,
        recent_memory_max_keep: int = 2,
        recent_similarity_threshold: float = 0.985,
        recent_stability_threshold: float = 0.72,
        recent_confidence_threshold: float = 0.60,
        memory_write_mode: str = "off",  # off | quality_novelty_gate | adaptive_controller | adaptive_controller_v2 | adaptive_skip_write_v2 | learned_skip_write_controller | learned_adaptive_controller_v2 | sliding_window_challenging_controller_v1
        memory_write_similarity_threshold: float = 0.995,
        memory_write_mask_change_threshold: float = 0.03,
        memory_write_quality_threshold: float = 0.55,
        memory_write_min_area: float = 0.0,
        memory_write_controller_path: str = "",
        memory_write_controller_threshold: float = -1.0,
        memory_write_controller_low_threshold: float = -1.0,
        memory_write_controller_high_threshold: float = -1.0,
        memory_write_controller_fallback_band: float = 0.075,
        challenging_mode_detector_path: str = "",
        challenging_mode_detector_threshold: float = -1.0,
        challenging_mode_enter_threshold: float = -1.0,
        challenging_mode_exit_threshold: float = -1.0,
        challenging_mode_window_size: int = 3,
        memory_controller_low_quality_threshold: float = 0.45,
        memory_controller_skip_centroid_shift_threshold: float = 0.02,
        memory_controller_conservative_similarity_threshold: float = 0.95,
        memory_controller_conservative_mask_change_threshold: float = 0.10,
        memory_controller_conservative_centroid_shift_threshold: float = 0.03,
        memory_controller_conservative_fill_ratio_threshold: float = 0.60,
        memory_controller_conservative_prune_delta: int = 1,
        memory_controller_aggressive_quality_threshold: float = 0.65,
        memory_controller_aggressive_similarity_threshold: float = 0.96,
        memory_controller_aggressive_mask_change_threshold: float = 0.12,
        memory_controller_aggressive_centroid_shift_threshold: float = 0.04,
        memory_controller_aggressive_fill_ratio_threshold: float = 0.75,
        memory_controller_aggressive_prune_delta: int = 1,
        debug_memory_write: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_frame_to_prune = num_frame_to_prune
        self.memory_prune_mode = memory_prune_mode
        self.memory_score_mode = memory_score_mode
        self.protect_conditioning_memories = protect_conditioning_memories
        self.memory_similarity_threshold = memory_similarity_threshold
        self.memory_min_temporal_gap = memory_min_temporal_gap
        self.debug_memory_pruning = debug_memory_pruning
        self.state_controller_cfg = dict(state_controller_cfg or {})
        self.use_recent_memory_guard = use_recent_memory_guard
        self.recent_memory_min_keep = max(1, int(recent_memory_min_keep))
        self.recent_memory_max_keep = max(
            self.recent_memory_min_keep,
            int(recent_memory_max_keep),
        )
        self.recent_similarity_threshold = float(recent_similarity_threshold)
        self.recent_stability_threshold = float(recent_stability_threshold)
        self.recent_confidence_threshold = float(recent_confidence_threshold)
        self.memory_write_mode = memory_write_mode
        self.memory_write_similarity_threshold = float(memory_write_similarity_threshold)
        self.memory_write_mask_change_threshold = float(memory_write_mask_change_threshold)
        self.memory_write_quality_threshold = float(memory_write_quality_threshold)
        self.memory_write_min_area = float(memory_write_min_area)
        self.memory_write_controller_path = str(memory_write_controller_path or "")
        self.memory_write_controller_threshold = float(memory_write_controller_threshold)
        self.memory_write_controller_low_threshold = float(
            memory_write_controller_low_threshold
        )
        self.memory_write_controller_high_threshold = float(
            memory_write_controller_high_threshold
        )
        self.memory_write_controller_fallback_band = float(
            memory_write_controller_fallback_band
        )
        self.challenging_mode_detector_path = str(challenging_mode_detector_path or "")
        self.challenging_mode_detector_threshold = float(challenging_mode_detector_threshold)
        self.challenging_mode_enter_threshold = float(challenging_mode_enter_threshold)
        self.challenging_mode_exit_threshold = float(challenging_mode_exit_threshold)
        self.challenging_mode_window_size = max(1, int(challenging_mode_window_size))
        self.memory_controller_low_quality_threshold = float(
            memory_controller_low_quality_threshold
        )
        self.memory_controller_skip_centroid_shift_threshold = float(
            memory_controller_skip_centroid_shift_threshold
        )
        self.memory_controller_conservative_similarity_threshold = float(
            memory_controller_conservative_similarity_threshold
        )
        self.memory_controller_conservative_mask_change_threshold = float(
            memory_controller_conservative_mask_change_threshold
        )
        self.memory_controller_conservative_centroid_shift_threshold = float(
            memory_controller_conservative_centroid_shift_threshold
        )
        self.memory_controller_conservative_fill_ratio_threshold = float(
            memory_controller_conservative_fill_ratio_threshold
        )
        self.memory_controller_conservative_prune_delta = int(
            memory_controller_conservative_prune_delta
        )
        self.memory_controller_aggressive_quality_threshold = float(
            memory_controller_aggressive_quality_threshold
        )
        self.memory_controller_aggressive_similarity_threshold = float(
            memory_controller_aggressive_similarity_threshold
        )
        self.memory_controller_aggressive_mask_change_threshold = float(
            memory_controller_aggressive_mask_change_threshold
        )
        self.memory_controller_aggressive_centroid_shift_threshold = float(
            memory_controller_aggressive_centroid_shift_threshold
        )
        self.memory_controller_aggressive_fill_ratio_threshold = float(
            memory_controller_aggressive_fill_ratio_threshold
        )
        self.memory_controller_aggressive_prune_delta = int(
            memory_controller_aggressive_prune_delta
        )
        self.debug_memory_write = debug_memory_write
        self._active_num_frame_to_prune = int(num_frame_to_prune)
        self._last_memory_prune_debug: Dict[str, Any] = {}
        self._last_memory_write_debug: Dict[str, Any] = {}
        self._memory_write_controller: Optional[OfflineWriteController] = None
        self._memory_write_controller_feature_names: Tuple[str, ...] = DEFAULT_WRITE_FEATURE_NAMES
        self._memory_write_controller_mean: Optional[np.ndarray] = None
        self._memory_write_controller_std: Optional[np.ndarray] = None
        self._memory_write_controller_default_threshold: float = 0.5
        self._external_write_policy_fn: Optional[
            Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
        ] = None
        self._challenging_mode_detector: Optional[OfflineWriteController] = None
        self._challenging_mode_detector_feature_names: Tuple[str, ...] = (
            DEFAULT_MODE_WINDOW_FEATURE_NAMES
        )
        self._challenging_mode_detector_mean: Optional[np.ndarray] = None
        self._challenging_mode_detector_std: Optional[np.ndarray] = None
        self._challenging_mode_detector_default_threshold: float = 0.5
        self._challenging_mode_history: deque[Dict[str, float]] = deque(
            maxlen=max(0, self.challenging_mode_window_size - 1)
        )
        self._challenging_mode_active: bool = False

    def set_external_write_policy(
        self,
        policy_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]],
    ) -> None:
        self._external_write_policy_fn = policy_fn

    def clear_external_write_policy(self) -> None:
        self._external_write_policy_fn = None

    def _load_memory_write_controller(self) -> None:
        if self.memory_write_controller_path == "":
            raise ValueError(
                "memory_write_mode=learned_skip_write_controller requires "
                "memory_write_controller_path to be set."
            )
        checkpoint = torch.load(
            self.memory_write_controller_path,
            map_location="cpu",
            weights_only=False,
        )
        feature_names = tuple(
            checkpoint.get("feature_names", list(DEFAULT_WRITE_FEATURE_NAMES))
        )
        hidden_dims = checkpoint.get("hidden_dims", [16, 8])
        controller = OfflineWriteController(
            input_dim=len(feature_names),
            hidden_dims=hidden_dims,
        )
        controller.load_state_dict(checkpoint["state_dict"])
        controller.eval()
        self._memory_write_controller = controller
        self._memory_write_controller_feature_names = feature_names
        self._memory_write_controller_mean = np.asarray(
            checkpoint["feature_mean"], dtype=np.float32
        ).reshape(1, -1)
        self._memory_write_controller_std = np.asarray(
            checkpoint["feature_std"], dtype=np.float32
        ).reshape(1, -1)
        self._memory_write_controller_default_threshold = float(
            checkpoint.get("threshold", 0.5)
        )

    def _predict_learned_write_probability(
        self,
        feature_source: Dict[str, Any],
    ) -> float:
        if self._memory_write_controller is None:
            self._load_memory_write_controller()
        assert self._memory_write_controller_mean is not None
        assert self._memory_write_controller_std is not None
        feature_array = build_feature_vector(
            feature_source,
            self._memory_write_controller_feature_names,
        ).reshape(1, -1)
        feature_array = (
            feature_array - self._memory_write_controller_mean
        ) / self._memory_write_controller_std
        with torch.no_grad():
            logits = self._memory_write_controller(torch.from_numpy(feature_array))
            probability = torch.sigmoid(logits).item()
        return float(probability)

    def _load_challenging_mode_detector(self) -> None:
        if self.challenging_mode_detector_path == "":
            raise ValueError(
                "memory_write_mode=sliding_window_challenging_controller_v1 requires "
                "challenging_mode_detector_path to be set."
            )
        checkpoint = torch.load(
            self.challenging_mode_detector_path,
            map_location="cpu",
            weights_only=False,
        )
        feature_names = tuple(
            checkpoint.get("feature_names", list(DEFAULT_MODE_WINDOW_FEATURE_NAMES))
        )
        hidden_dims = checkpoint.get("hidden_dims", [16, 8])
        detector = OfflineWriteController(
            input_dim=len(feature_names),
            hidden_dims=hidden_dims,
        )
        detector.load_state_dict(checkpoint["state_dict"])
        detector.eval()
        self._challenging_mode_detector = detector
        self._challenging_mode_detector_feature_names = feature_names
        self._challenging_mode_detector_mean = np.asarray(
            checkpoint["feature_mean"], dtype=np.float32
        ).reshape(1, -1)
        self._challenging_mode_detector_std = np.asarray(
            checkpoint["feature_std"], dtype=np.float32
        ).reshape(1, -1)
        self._challenging_mode_detector_default_threshold = float(
            checkpoint.get("threshold", 0.5)
        )

    def _extract_mode_feature_source(
        self,
        feature_source: Dict[str, Any],
    ) -> Dict[str, float]:
        return {
            "quality": float(feature_source.get("quality", 0.0) or 0.0),
            "mask_cosine": float(feature_source.get("mask_cosine", 0.0) or 0.0),
            "mask_change": float(feature_source.get("mask_change", 1.0) or 1.0),
            "centroid_shift": float(feature_source.get("centroid_shift", 1.0) or 1.0),
            "current_area_ratio": float(
                feature_source.get("current_area_ratio", 0.0) or 0.0
            ),
            "area_change": float(feature_source.get("area_change", 0.0) or 0.0),
            "reference_temporal_gap": float(
                feature_source.get("reference_temporal_gap", 0.0) or 0.0
            ),
            "bank_fill_ratio": float(feature_source.get("bank_fill_ratio", 0.0) or 0.0),
        }

    def _predict_challenging_mode_probability(
        self,
        window_sources: List[Dict[str, float]],
    ) -> float:
        if self._challenging_mode_detector is None:
            self._load_challenging_mode_detector()
        assert self._challenging_mode_detector_mean is not None
        assert self._challenging_mode_detector_std is not None
        aggregated = build_mode_window_feature_source(window_sources)
        feature_array = build_feature_vector(
            aggregated,
            self._challenging_mode_detector_feature_names,
        ).reshape(1, -1)
        feature_array = (
            feature_array - self._challenging_mode_detector_mean
        ) / self._challenging_mode_detector_std
        with torch.no_grad():
            logits = self._challenging_mode_detector(torch.from_numpy(feature_array))
            probability = torch.sigmoid(logits).item()
        return float(probability)

    def _evaluate_challenging_mode(
        self,
        feature_source: Dict[str, Any],
        is_init_cond_frame: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        if is_init_cond_frame:
            self._challenging_mode_history.clear()
            self._challenging_mode_active = False
            return False, {
                "challenging_mode_prob": 0.0,
                "challenging_mode_active": False,
                "challenging_mode_enter_threshold": None,
                "challenging_mode_exit_threshold": None,
                "challenging_mode_window_len": 1,
                "challenging_mode_reason": "init_cond_frame",
            }

        current_source = self._extract_mode_feature_source(feature_source)
        window_sources = list(self._challenging_mode_history)
        window_sources.append(current_source)
        if self.challenging_mode_window_size > 0:
            window_sources = window_sources[-self.challenging_mode_window_size :]

        probability = self._predict_challenging_mode_probability(window_sources)
        base_threshold = self.challenging_mode_detector_threshold
        if base_threshold < 0.0:
            base_threshold = self._challenging_mode_detector_default_threshold
        enter_threshold = self.challenging_mode_enter_threshold
        exit_threshold = self.challenging_mode_exit_threshold
        if enter_threshold < 0.0:
            enter_threshold = min(1.0, base_threshold + 0.05)
        if exit_threshold < 0.0:
            exit_threshold = max(0.0, base_threshold - 0.05)
        if enter_threshold < exit_threshold:
            enter_threshold, exit_threshold = exit_threshold, enter_threshold

        challenging_mode_active = self._challenging_mode_active
        if challenging_mode_active:
            if probability < exit_threshold:
                challenging_mode_active = False
        else:
            if probability >= enter_threshold:
                challenging_mode_active = True
        self._challenging_mode_active = challenging_mode_active

        return challenging_mode_active, {
            "challenging_mode_prob": probability,
            "challenging_mode_active": challenging_mode_active,
            "challenging_mode_base_threshold": base_threshold,
            "challenging_mode_enter_threshold": enter_threshold,
            "challenging_mode_exit_threshold": exit_threshold,
            "challenging_mode_window_len": len(window_sources),
            "challenging_mode_reason": (
                "challenging"
                if challenging_mode_active
                else "easy"
            ),
        }

    def _update_challenging_mode_history(
        self,
        feature_source: Dict[str, Any],
        is_init_cond_frame: bool,
    ) -> None:
        if self.memory_write_mode != "sliding_window_challenging_controller_v1":
            return
        if is_init_cond_frame:
            self._challenging_mode_history.clear()
            self._challenging_mode_active = False
        self._challenging_mode_history.append(
            self._extract_mode_feature_source(feature_source)
        )

    # ---------------------------------------------------------------------
    # Metadata extraction
    # ---------------------------------------------------------------------

    def _extract_first_available_tensor(
        self,
        container: Dict[str, Any],
        keys: List[str],
    ) -> Optional[torch.Tensor]:
        for key in keys:
            value = container.get(key, None)
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _extract_mask_tensor(self, frame_out: Dict[str, Any]) -> Optional[torch.Tensor]:
        return self._extract_first_available_tensor(
            frame_out,
            [
                "high_res_masks",
                "pred_masks_high_res",
                "high_res_pred_masks",
                "pred_masks",
                "low_res_masks",
                "mask_inputs",
            ],
        )

    def _extract_low_res_mask_tensor(self, frame_out: Dict[str, Any]) -> Optional[torch.Tensor]:
        return self._extract_first_available_tensor(
            frame_out,
            [
                "pred_masks",
                "low_res_masks",
                "mask_inputs",
                "pred_masks_high_res",
                "high_res_masks",
                "high_res_pred_masks",
            ],
        )

    def _mask_tensor_to_geometry(self, mask_tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        Convert a predicted mask tensor into a small geometry summary usable by
        memory pruning. This is lightweight and only used during inference.

        Returns:
            {
                "centroid_xy": (x, y) normalized to [0,1],
                "area_ratio": float in [0,1],
                "bbox_xyxy": (x1, y1, x2, y2) normalized to [0,1],
            }
        """
        if mask_tensor is None:
            return {
                "centroid_xy": None,
                "area_ratio": None,
                "bbox_xyxy": None,
            }

        mask = mask_tensor.detach().float()

        # Accept [B,1,H,W], [B,H,W], [1,H,W], or [H,W].
        if mask.ndim == 4:
            mask = mask[:, 0]
        elif mask.ndim == 3:
            pass
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
        else:
            return {
                "centroid_xy": None,
                "area_ratio": None,
                "bbox_xyxy": None,
            }

        # Convert logits -> binary mask, or probabilities -> binary mask.
        if mask.min().item() < 0.0 or mask.max().item() > 1.0:
            bin_mask = mask > 0.0
        else:
            bin_mask = mask > 0.5

        B, H, W = bin_mask.shape
        centroids_x: List[float] = []
        centroids_y: List[float] = []
        areas: List[float] = []
        boxes: List[List[float]] = []

        ys = torch.arange(H, device=bin_mask.device, dtype=torch.float32)
        xs = torch.arange(W, device=bin_mask.device, dtype=torch.float32)

        for b in range(B):
            m = bin_mask[b]
            area = float(m.float().mean().item())
            if area <= 0.0:
                continue

            proj_y = m.float().sum(dim=1)
            proj_x = m.float().sum(dim=0)

            y_idx = torch.where(proj_y > 0)[0]
            x_idx = torch.where(proj_x > 0)[0]
            if y_idx.numel() == 0 or x_idx.numel() == 0:
                continue

            mass = m.float().sum().clamp(min=1.0)
            cy = float((m.float().sum(dim=1) * ys).sum().item() / mass.item())
            cx = float((m.float().sum(dim=0) * xs).sum().item() / mass.item())

            y1 = float(y_idx[0].item())
            y2 = float(y_idx[-1].item())
            x1 = float(x_idx[0].item())
            x2 = float(x_idx[-1].item())

            centroids_x.append(cx / max(1.0, W - 1.0))
            centroids_y.append(cy / max(1.0, H - 1.0))
            areas.append(area)
            boxes.append(
                [
                    x1 / max(1.0, W - 1.0),
                    y1 / max(1.0, H - 1.0),
                    x2 / max(1.0, W - 1.0),
                    y2 / max(1.0, H - 1.0),
                ]
            )

        if len(areas) == 0:
            return {
                "centroid_xy": None,
                "area_ratio": 0.0,
                "bbox_xyxy": None,
            }

        mean_box = [sum(coords[i] for coords in boxes) / len(boxes) for i in range(4)]
        return {
            "centroid_xy": (
                sum(centroids_x) / len(centroids_x),
                sum(centroids_y) / len(centroids_y),
            ),
            "area_ratio": sum(areas) / len(areas),
            "bbox_xyxy": tuple(mean_box),
        }

    def _extract_confidence_metadata(self, frame_out: Dict[str, Any]) -> Dict[str, Any]:
        pred_iou = None
        obj_score = None

        # IoU prediction from SAM-style decoder output
        for key in ["best_iou", "pred_iou", "iou_scores"]:
            value = frame_out.get(key, None)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    pred_iou = float(value.detach().float().mean().item())
                else:
                    pred_iou = float(value)
                break

        # Common case: "ious" with shape [B, M]
        if pred_iou is None:
            ious = frame_out.get("ious", None)
            if isinstance(ious, torch.Tensor) and ious.numel() > 0:
                if ious.ndim >= 2:
                    ious = ious.max(dim=-1).values
                pred_iou = float(ious.detach().float().mean().item())

        # Objectness / presence confidence
        for key in ["object_score_logits", "object_scores", "obj_score"]:
            value = frame_out.get(key, None)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    obj_score = float(value.detach().float().mean().item())
                else:
                    obj_score = float(value)
                break

        return {
            "pred_iou": pred_iou,
            "obj_score": obj_score,
        }

    def _estimate_write_quality(self, frame_out: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        confidence_meta = self._extract_confidence_metadata(frame_out)
        pred_iou = confidence_meta.get("pred_iou", None)
        obj_score = confidence_meta.get("obj_score", None)

        if pred_iou is not None:
            quality = float(max(0.0, min(1.0, pred_iou)))
        elif obj_score is not None:
            quality = float(torch.sigmoid(torch.tensor(float(obj_score))).item())
        else:
            quality = 0.5

        return quality, confidence_meta

    def _mask_tensor_to_prob_map(self, mask_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask_tensor is None:
            return None

        mask = mask_tensor.detach().float()
        if mask.ndim == 4:
            mask = mask[:, 0]
        elif mask.ndim == 3:
            pass
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
        else:
            return None

        if mask.min().item() < 0.0 or mask.max().item() > 1.0:
            mask = torch.sigmoid(mask)
        return mask

    def _compute_mask_similarity_stats(
        self,
        current_mask: Optional[torch.Tensor],
        reference_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        current_prob = self._mask_tensor_to_prob_map(current_mask)
        reference_prob = self._mask_tensor_to_prob_map(reference_mask)

        if current_prob is None or reference_prob is None:
            return {
                "mask_cosine": 0.0,
                "mask_iou": 0.0,
                "mask_change": 1.0,
            }

        if reference_prob.shape[-2:] != current_prob.shape[-2:]:
            reference_prob = F.interpolate(
                reference_prob.unsqueeze(1),
                size=current_prob.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        reference_prob = reference_prob.to(current_prob.device, non_blocking=True)

        current_flat = current_prob.flatten(1)
        reference_flat = reference_prob.flatten(1)
        mask_cosine = float(
            F.cosine_similarity(current_flat, reference_flat, dim=1).mean().item()
        )

        current_bin = current_prob > 0.5
        reference_bin = reference_prob > 0.5
        inter = (current_bin & reference_bin).float().flatten(1).sum(dim=1)
        union = (current_bin | reference_bin).float().flatten(1).sum(dim=1)
        mask_iou = torch.where(
            union > 0,
            inter / union.clamp(min=1.0),
            torch.ones_like(union),
        )
        mask_iou = float(mask_iou.mean().item())

        return {
            "mask_cosine": mask_cosine,
            "mask_iou": mask_iou,
            "mask_change": 1.0 - mask_iou,
        }

    def _select_reference_memory_output(
        self,
        frame_idx: int,
        output_dict: Dict[str, Dict[int, Dict[str, Any]]],
        track_in_reverse: bool = False,
    ) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None

        for source_name, outputs in (
            ("cond", output_dict.get("cond_frame_outputs", {})),
            ("non_cond", output_dict.get("non_cond_frame_outputs", {})),
        ):
            for ref_frame_idx, out in outputs.items():
                if out is None or out.get("maskmem_features") is None:
                    continue

                if track_in_reverse:
                    if ref_frame_idx <= frame_idx:
                        continue
                    temporal_gap = ref_frame_idx - frame_idx
                else:
                    if ref_frame_idx >= frame_idx:
                        continue
                    temporal_gap = frame_idx - ref_frame_idx

                if best is None or temporal_gap < best["temporal_gap"]:
                    best = {
                        "frame_idx": ref_frame_idx,
                        "out": out,
                        "source": source_name,
                        "temporal_gap": temporal_gap,
                    }

        return best

    def _count_reference_memory_outputs(
        self,
        frame_idx: int,
        output_dict: Dict[str, Dict[int, Dict[str, Any]]],
        track_in_reverse: bool = False,
    ) -> int:
        count = 0
        for outputs in (
            output_dict.get("cond_frame_outputs", {}),
            output_dict.get("non_cond_frame_outputs", {}),
        ):
            for ref_frame_idx, out in outputs.items():
                if out is None or out.get("maskmem_features") is None:
                    continue

                if track_in_reverse:
                    if ref_frame_idx <= frame_idx:
                        continue
                else:
                    if ref_frame_idx >= frame_idx:
                        continue
                count += 1
        return count

    def _list_reference_memory_outputs(
        self,
        frame_idx: int,
        output_dict: Dict[str, Dict[int, Dict[str, Any]]],
        track_in_reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        references: List[Dict[str, Any]] = []

        for source_name, outputs in (
            ("cond", output_dict.get("cond_frame_outputs", {})),
            ("non_cond", output_dict.get("non_cond_frame_outputs", {})),
        ):
            for ref_frame_idx, out in outputs.items():
                if out is None or out.get("maskmem_features") is None:
                    continue

                if track_in_reverse:
                    if ref_frame_idx <= frame_idx:
                        continue
                    temporal_gap = ref_frame_idx - frame_idx
                else:
                    if ref_frame_idx >= frame_idx:
                        continue
                    temporal_gap = frame_idx - ref_frame_idx

                references.append(
                    {
                        "frame_idx": ref_frame_idx,
                        "out": out,
                        "source": source_name,
                        "temporal_gap": temporal_gap,
                    }
                )

        references.sort(key=lambda item: item["temporal_gap"])
        return references

    def _compute_centroid_shift(
        self,
        current_geom: Dict[str, Any],
        reference_geom: Dict[str, Any],
    ) -> float:
        current_centroid = current_geom.get("centroid_xy", None)
        reference_centroid = reference_geom.get("centroid_xy", None)
        if current_centroid is None or reference_centroid is None:
            return 1.0

        dx = float(current_centroid[0]) - float(reference_centroid[0])
        dy = float(current_centroid[1]) - float(reference_centroid[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _summarize_bank_similarity(
        self,
        current_mask: Optional[torch.Tensor],
        references: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if current_mask is None or len(references) == 0:
            return {
                "sim_mean_bank": 0.0,
                "sim_max_bank": 0.0,
                "sim_min_bank": 0.0,
                "mask_change_mean_bank": 1.0,
            }

        cosines: List[float] = []
        changes: List[float] = []
        for reference in references:
            reference_mask = self._extract_mask_tensor(reference["out"])
            stats = self._compute_mask_similarity_stats(
                current_mask=current_mask,
                reference_mask=reference_mask,
            )
            cosines.append(float(stats["mask_cosine"]))
            changes.append(float(stats["mask_change"]))

        return {
            "sim_mean_bank": sum(cosines) / len(cosines),
            "sim_max_bank": max(cosines),
            "sim_min_bank": min(cosines),
            "mask_change_mean_bank": sum(changes) / len(changes),
        }

    def _should_write_memory(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        point_inputs: Optional[Dict[str, Any]],
        mask_inputs: Optional[torch.Tensor],
        output_dict: Dict[str, Dict[int, Dict[str, Any]]],
        track_in_reverse: bool,
        current_out: Dict[str, Any],
        run_mem_encoder: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        base_prune_count = max(0, int(self.num_frame_to_prune))
        debug: Dict[str, Any] = {
            "mode": self.memory_write_mode,
            "frame_idx": frame_idx,
            "action": "WRITE_NORMAL",
            "write_memory": True,
            "reason": "enabled",
            "next_num_frame_to_prune": base_prune_count,
        }

        if self.training:
            debug["reason"] = "training"
            return True, debug

        if not run_mem_encoder:
            debug["action"] = "SKIP_WRITE"
            debug["write_memory"] = False
            debug["reason"] = "caller_disabled"
            return False, debug

        if self.num_maskmem <= 0:
            debug["action"] = "SKIP_WRITE"
            debug["write_memory"] = False
            debug["reason"] = "memory_disabled"
            return False, debug

        if self.memory_write_mode == "off":
            debug["reason"] = "off"
            return True, debug

        if self.memory_write_mode not in {
            "quality_novelty_gate",
            "adaptive_controller",
            "adaptive_controller_v2",
            "adaptive_skip_write_v2",
            "learned_skip_write_controller",
            "learned_adaptive_controller_v2",
            "sliding_window_challenging_controller_v1",
        }:
            raise ValueError(
                f"Unknown memory_write_mode={self.memory_write_mode!r}. "
                "Expected one of: off | quality_novelty_gate | adaptive_controller | adaptive_controller_v2 | adaptive_skip_write_v2 | learned_skip_write_controller | learned_adaptive_controller_v2 | sliding_window_challenging_controller_v1."
            )

        if is_init_cond_frame:
            debug["reason"] = "init_cond_frame"
            return True, debug

        if point_inputs is not None or mask_inputs is not None:
            debug["reason"] = "interactive_frame"
            return True, debug

        quality, confidence_meta = self._estimate_write_quality(current_out)
        use_low_res_controller = self.memory_write_mode in {
            "adaptive_controller_v2",
            "adaptive_skip_write_v2",
            "learned_skip_write_controller",
            "learned_adaptive_controller_v2",
            "sliding_window_challenging_controller_v1",
        }
        if use_low_res_controller:
            current_mask = current_out.get("pred_masks", None)
        else:
            current_mask = current_out.get("pred_masks_high_res", None)
        current_geom = self._mask_tensor_to_geometry(current_mask)
        current_area = current_geom.get("area_ratio", 0.0)
        current_area = 0.0 if current_area is None else float(current_area)

        debug.update(
            {
                "quality": quality,
                "pred_iou": confidence_meta.get("pred_iou", None),
                "obj_score": confidence_meta.get("obj_score", None),
                "current_area_ratio": current_area,
            }
        )

        low_quality_threshold = self.memory_write_quality_threshold
        if self.memory_write_mode in {
            "adaptive_controller",
            "adaptive_controller_v2",
            "adaptive_skip_write_v2",
            "learned_skip_write_controller",
            "learned_adaptive_controller_v2",
            "sliding_window_challenging_controller_v1",
        }:
            low_quality_threshold = self.memory_controller_low_quality_threshold

        if quality < low_quality_threshold:
            debug["action"] = "SKIP_WRITE"
            debug["write_memory"] = False
            debug["reason"] = "low_quality"
            return False, debug

        if current_area <= self.memory_write_min_area:
            debug["action"] = "SKIP_WRITE"
            debug["write_memory"] = False
            debug["reason"] = "tiny_mask"
            return False, debug

        reference = self._select_reference_memory_output(
            frame_idx=frame_idx,
            output_dict=output_dict,
            track_in_reverse=track_in_reverse,
        )
        if reference is None:
            debug["reason"] = "no_reference_memory"
            return True, debug

        if use_low_res_controller:
            reference_mask = self._extract_low_res_mask_tensor(reference["out"])
        else:
            reference_mask = self._extract_mask_tensor(reference["out"])

        reference_geom = self._mask_tensor_to_geometry(reference_mask)
        similarity_stats = self._compute_mask_similarity_stats(
            current_mask=current_mask,
            reference_mask=reference_mask,
        )
        centroid_shift = self._compute_centroid_shift(current_geom, reference_geom)
        reference_area = reference_geom.get("area_ratio", 0.0)
        reference_area = 0.0 if reference_area is None else float(reference_area)
        area_change = abs(current_area - reference_area)
        num_references = self._count_reference_memory_outputs(
            frame_idx=frame_idx,
            output_dict=output_dict,
            track_in_reverse=track_in_reverse,
        )
        effective_slots = max(1, self.num_maskmem - 1)
        bank_fill_ratio = min(num_references, effective_slots) / effective_slots

        debug.update(
            {
                "reference_frame_idx": reference["frame_idx"],
                "reference_source": reference["source"],
                "reference_temporal_gap": reference["temporal_gap"],
                "centroid_shift": centroid_shift,
                "area_change": area_change,
                "bank_fill_ratio": bank_fill_ratio,
                **similarity_stats,
            }
        )

        if not use_low_res_controller:
            bank_similarity = self._summarize_bank_similarity(
                current_mask=current_mask,
                references=self._list_reference_memory_outputs(
                    frame_idx=frame_idx,
                    output_dict=output_dict,
                    track_in_reverse=track_in_reverse,
                ),
            )
            debug.update(bank_similarity)
        else:
            bank_similarity = {
                "sim_mean_bank": float(similarity_stats["mask_cosine"]),
                "sim_max_bank": float(similarity_stats["mask_cosine"]),
                "sim_min_bank": float(similarity_stats["mask_cosine"]),
                "mask_change_mean_bank": float(similarity_stats["mask_change"]),
            }

        if self._external_write_policy_fn is not None:
            override = self._external_write_policy_fn(dict(debug))
            if override is not None:
                forced_write = bool(override.get("write_memory", True))
                debug["action"] = str(
                    override.get(
                        "action",
                        "WRITE_NORMAL" if forced_write else "SKIP_WRITE",
                    )
                )
                debug["write_memory"] = forced_write
                debug["reason"] = str(override.get("reason", "external_policy"))
                if "controller_prob" in override:
                    debug["controller_prob"] = float(override["controller_prob"])
                if "controller_threshold" in override:
                    debug["controller_threshold"] = float(override["controller_threshold"])
                if "controller_entropy" in override:
                    debug["controller_entropy"] = float(override["controller_entropy"])
                if "next_num_frame_to_prune" in override:
                    debug["next_num_frame_to_prune"] = int(
                        override["next_num_frame_to_prune"]
                    )
                return forced_write, debug

        effective_write_mode = self.memory_write_mode
        if self.memory_write_mode == "sliding_window_challenging_controller_v1":
            challenging_active, mode_debug = self._evaluate_challenging_mode(
                feature_source=debug,
                is_init_cond_frame=is_init_cond_frame,
            )
            debug.update(mode_debug)
            if not challenging_active:
                debug["action"] = "WRITE_NORMAL"
                debug["write_memory"] = True
                debug["reason"] = "easy_mode_baseline"
                return True, debug
            effective_write_mode = "learned_adaptive_controller_v2"

        if (
            similarity_stats["mask_cosine"] >= self.memory_write_similarity_threshold
            and similarity_stats["mask_change"] <= self.memory_write_mask_change_threshold
            and centroid_shift <= self.memory_controller_skip_centroid_shift_threshold
        ):
            debug["action"] = "SKIP_WRITE"
            debug["write_memory"] = False
            debug["reason"] = "redundant_stable"
            return False, debug

        if effective_write_mode in {
            "learned_skip_write_controller",
            "learned_adaptive_controller_v2",
        }:
            controller_prob = self._predict_learned_write_probability(debug)
            controller_threshold = self.memory_write_controller_threshold
            if controller_threshold < 0.0:
                controller_threshold = self._memory_write_controller_default_threshold
            debug["controller_prob"] = controller_prob
            debug["controller_threshold"] = controller_threshold
            if effective_write_mode == "learned_skip_write_controller":
                if controller_prob < controller_threshold:
                    debug["action"] = "SKIP_WRITE"
                    debug["write_memory"] = False
                    debug["reason"] = "learned_skip"
                    return False, debug
                debug["reason"] = "learned_write"
                return True, debug

            controller_low_threshold = self.memory_write_controller_low_threshold
            controller_high_threshold = self.memory_write_controller_high_threshold
            if controller_low_threshold < 0.0 or controller_high_threshold < 0.0:
                band = max(0.0, float(self.memory_write_controller_fallback_band))
                controller_low_threshold = max(0.0, controller_threshold - band)
                controller_high_threshold = min(1.0, controller_threshold + band)
            if controller_low_threshold > controller_high_threshold:
                controller_low_threshold, controller_high_threshold = (
                    controller_high_threshold,
                    controller_low_threshold,
                )

            debug["controller_low_threshold"] = controller_low_threshold
            debug["controller_high_threshold"] = controller_high_threshold

            if controller_prob < controller_low_threshold:
                debug["action"] = "SKIP_WRITE"
                debug["write_memory"] = False
                debug["reason"] = "learned_skip_low_conf"
                return False, debug

            if controller_prob <= controller_high_threshold:
                debug["action"] = "WRITE_NORMAL"
                debug["reason"] = "learned_fallback_band"
                return True, debug

            should_write_conservatively = (
                bank_fill_ratio >= self.memory_controller_conservative_fill_ratio_threshold
                and (
                    bank_similarity["sim_mean_bank"]
                    <= self.memory_controller_conservative_similarity_threshold
                    or similarity_stats["mask_change"]
                    >= self.memory_controller_conservative_mask_change_threshold
                    or centroid_shift
                    >= self.memory_controller_conservative_centroid_shift_threshold
                )
            )
            if should_write_conservatively:
                conservative_prune = max(
                    0,
                    base_prune_count - max(1, self.memory_controller_conservative_prune_delta),
                )
                debug["action"] = "WRITE_CONSERVATIVE"
                debug["reason"] = "learned_high_conf_hard_state"
                debug["next_num_frame_to_prune"] = conservative_prune
                return True, debug

            debug["action"] = "WRITE_NORMAL"
            debug["reason"] = "learned_high_conf_normal"
            return True, debug

        if effective_write_mode in {"adaptive_controller", "adaptive_controller_v2"}:
            should_write_aggressively = (
                quality >= self.memory_controller_aggressive_quality_threshold
                and (
                    bank_similarity["sim_mean_bank"]
                    <= self.memory_controller_aggressive_similarity_threshold
                    or similarity_stats["mask_change"]
                    >= self.memory_controller_aggressive_mask_change_threshold
                    or centroid_shift
                    >= self.memory_controller_aggressive_centroid_shift_threshold
                )
                and bank_fill_ratio >= self.memory_controller_aggressive_fill_ratio_threshold
            )
            if should_write_aggressively:
                aggressive_prune = min(
                    max(0, self.num_maskmem - 2),
                    base_prune_count + max(1, self.memory_controller_aggressive_prune_delta),
                )
                debug["action"] = "WRITE_AGGRESSIVE"
                debug["reason"] = "novel_or_hard_state"
                debug["next_num_frame_to_prune"] = aggressive_prune
                return True, debug

        debug["reason"] = "novel_enough"
        return True, debug

    def _make_memory_meta(
        self,
        frame_idx: Optional[int],
        t_pos: int,
        is_conditioning: bool,
        source: str,
        prev: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "t_pos": t_pos,
            "is_conditioning": is_conditioning,
            "source": source,
        }

        if prev is None:
            return meta

        # If another part of the pipeline already wrote explicit pruning metadata,
        # use it directly.
        cached_meta = prev.get("memory_prune_meta", None)
        if isinstance(cached_meta, dict):
            meta.update(cached_meta)
            return meta

        # Otherwise derive a lightweight summary from available mask / score outputs.
        mask_tensor = self._extract_mask_tensor(prev)
        meta.update(self._mask_tensor_to_geometry(mask_tensor))
        meta.update(self._extract_confidence_metadata(prev))

        # Optional pre-computed motion magnitude, if caller stored one.
        if "motion_mag" in prev:
            motion_mag = prev["motion_mag"]
            if isinstance(motion_mag, torch.Tensor):
                meta["motion_mag"] = float(motion_mag.detach().float().mean().item())
            else:
                meta["motion_mag"] = float(motion_mag)

        return meta

    # ---------------------------------------------------------------------
    # Pruning
    # ---------------------------------------------------------------------

    def _prune_memory_frames(
        self,
        to_cat_memory: List[torch.Tensor],
        to_cat_memory_pos_embed: List[torch.Tensor],
        memory_meta: List[Dict[str, Any]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
        active_num_frame_to_prune = max(
            0, int(getattr(self, "_active_num_frame_to_prune", self.num_frame_to_prune))
        )
        if self.training:
            self._last_memory_prune_debug = {
                "mode": self.memory_prune_mode,
                "skipped": "training",
                "num_before": len(to_cat_memory),
                "num_after": len(to_cat_memory),
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        if self.memory_prune_mode == "off":
            self._last_memory_prune_debug = {
                "mode": "off",
                "score_mode": self.memory_score_mode,
                "num_before": len(to_cat_memory),
                "num_after": len(to_cat_memory),
                "pruned_indices": [],
                "pruned_frame_idx": [],
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        delete_indices, debug = plan_memory_pruning(
            prune_mode=self.memory_prune_mode,
            score_mode=self.memory_score_mode,
            to_cat_memory=to_cat_memory,
            memory_meta=memory_meta,
            num_maskmem=self.num_maskmem,
            num_frame_to_prune=active_num_frame_to_prune,
            protect_conditioning_memories=self.protect_conditioning_memories,
            memory_similarity_threshold=self.memory_similarity_threshold,
            memory_min_temporal_gap=self.memory_min_temporal_gap,
            controller_cfg=self.state_controller_cfg,
            use_recent_memory_guard=self.use_recent_memory_guard,
            recent_memory_min_keep=self.recent_memory_min_keep,
            recent_memory_max_keep=self.recent_memory_max_keep,
            recent_similarity_threshold=self.recent_similarity_threshold,
            recent_stability_threshold=self.recent_stability_threshold,
            recent_confidence_threshold=self.recent_confidence_threshold,
        )

        pruned_frame_idx = [memory_meta[i].get("frame_idx") for i in delete_indices]
        for i in delete_indices:
            to_cat_memory.pop(i)
            to_cat_memory_pos_embed.pop(i)
            memory_meta.pop(i)

        debug["num_after"] = len(to_cat_memory)
        debug["active_num_frame_to_prune"] = active_num_frame_to_prune
        debug["pruned_indices"] = delete_indices
        debug["pruned_frame_idx"] = pruned_frame_idx

        if not self.debug_memory_pruning:
            # Keep debug compact unless explicitly requested.
            debug = {
                "mode": debug.get("mode"),
                "score_mode": debug.get("score_mode"),
                "policy_name": debug.get("policy_name", None),
                "state": debug.get("state", None),
                "num_before": debug.get("num_before"),
                "num_after": debug.get("num_after"),
                "num_to_prune": debug.get("num_to_prune", 0),
                "active_num_frame_to_prune": debug.get("active_num_frame_to_prune"),
                "pruned_indices": debug.get("pruned_indices", []),
                "pruned_frame_idx": debug.get("pruned_frame_idx", []),
                "protected_recent_frame_idx": debug.get("protected_recent_frame_idx", []),
                "released_recent_frame_idx": debug.get("released_recent_frame_idx", []),
            }

        self._last_memory_prune_debug = debug
        return to_cat_memory, to_cat_memory_pos_embed, memory_meta

    # ---------------------------------------------------------------------
    # Main override: only minimally changes the memory-collection section,
    # while keeping the object-pointer logic and attention call unchanged.
    # ---------------------------------------------------------------------

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        if not is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed, memory_meta = [], [], []

            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )

            # Track richer metadata so prune policies can be role-aware.
            t_pos_and_prevs: List[Tuple[int, Optional[int], Optional[Dict[str, Any]], bool, str]] = [
                (0, t, out, True, "cond")
                for t, out in selected_cond_outputs.items()
            ]

            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    if not track_in_reverse:
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        prev_frame_idx = frame_idx + t_rel
                else:
                    if not track_in_reverse:
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride

                prev = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                is_conditioning = False
                source = "non_cond"
                if prev is None:
                    prev = unselected_cond_outputs.get(prev_frame_idx, None)
                    if prev is not None:
                        is_conditioning = True
                        source = "cond_unselected"

                t_pos_and_prevs.append(
                    (t_pos, prev_frame_idx, prev, is_conditioning, source)
                )

            for t_pos, prev_frame_idx, prev, is_conditioning, source in t_pos_and_prevs:
                if prev is None:
                    continue
                if prev.get("maskmem_features", None) is None:
                    continue
                if prev.get("maskmem_pos_enc", None) is None:
                    continue

                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))

                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

                memory_meta.append(
                    self._make_memory_meta(
                        frame_idx=prev_frame_idx,
                        t_pos=t_pos,
                        is_conditioning=is_conditioning,
                        source=source,
                        prev=prev,
                    )
                )

            # Prune only frame memories. Keep object-pointer logic below unchanged.
            to_cat_memory, to_cat_memory_pos_embed, memory_meta = self._prune_memory_frames(
                to_cat_memory=to_cat_memory,
                to_cat_memory_pos_embed=to_cat_memory_pos_embed,
                memory_meta=memory_meta,
            )

            # Object pointers: keep the upstream logic unchanged.
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs

                pos_and_ptrs = [
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]

                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))

                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)

                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(device=device, non_blocking=True)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)

                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0

        else:
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["ious"] = ious
        if not self.training:
            current_out["object_score_logits"] = object_score_logits

        should_write_memory, write_debug = self._should_write_memory(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            track_in_reverse=track_in_reverse,
            current_out=current_out,
            run_mem_encoder=run_mem_encoder,
        )
        self._active_num_frame_to_prune = int(
            write_debug.get("next_num_frame_to_prune", self.num_frame_to_prune)
        )
        self._update_challenging_mode_history(
            feature_source=write_debug,
            is_init_cond_frame=is_init_cond_frame,
        )

        if not self.debug_memory_write:
            write_debug = {
                "mode": write_debug.get("mode"),
                "frame_idx": write_debug.get("frame_idx"),
                "action": write_debug.get("action"),
                "write_memory": write_debug.get("write_memory"),
                "reason": write_debug.get("reason"),
                "next_num_frame_to_prune": write_debug.get("next_num_frame_to_prune"),
                "quality": write_debug.get("quality", None),
                "reference_frame_idx": write_debug.get("reference_frame_idx", None),
                "mask_cosine": write_debug.get("mask_cosine", None),
                "mask_change": write_debug.get("mask_change", None),
                "sim_mean_bank": write_debug.get("sim_mean_bank", None),
                "centroid_shift": write_debug.get("centroid_shift", None),
                "challenging_mode_prob": write_debug.get("challenging_mode_prob", None),
                "challenging_mode_active": write_debug.get("challenging_mode_active", None),
                "controller_prob": write_debug.get("controller_prob", None),
            }
        self._last_memory_write_debug = write_debug

        self._encode_memory_in_output(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            run_mem_encoder=bool(run_mem_encoder and should_write_memory),
            high_res_masks=high_res_masks,
            object_score_logits=object_score_logits,
            current_out=current_out,
        )

        return current_out
