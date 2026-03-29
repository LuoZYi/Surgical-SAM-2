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
- optional protection for conditioning memories
- lightweight debug bookkeeping via `self._last_memory_prune_debug`

Recommended usage:
    from sam2.modeling.sam2_newbase import SAM2NewBase

Then point your model builder / config at SAM2NewBase instead of SAM2Base.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from sam2.modeling.memory_pruning import plan_memory_pruning
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames


class SAM2NewBase(SAM2Base):
    def __init__(
        self,
        *args,
        memory_prune_mode: str = "efp",   # off | efp | rule_based | state_aware
        memory_score_mode: str = "cosine_only",  # cosine_only | cosine_motion | cosine_motion_geometry
        protect_conditioning_memories: bool = False,
        memory_similarity_threshold: Optional[float] = None,
        memory_min_temporal_gap: int = 0,
        debug_memory_pruning: bool = False,
        state_controller_cfg: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_prune_mode = memory_prune_mode
        self.memory_score_mode = memory_score_mode
        self.protect_conditioning_memories = protect_conditioning_memories
        self.memory_similarity_threshold = memory_similarity_threshold
        self.memory_min_temporal_gap = memory_min_temporal_gap
        self.debug_memory_pruning = debug_memory_pruning
        self.state_controller_cfg = dict(state_controller_cfg or {})
        self._last_memory_prune_debug: Dict[str, Any] = {}

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
            num_frame_to_prune=self.num_frame_to_prune,
            protect_conditioning_memories=self.protect_conditioning_memories,
            memory_similarity_threshold=self.memory_similarity_threshold,
            memory_min_temporal_gap=self.memory_min_temporal_gap,
            controller_cfg=self.state_controller_cfg,
        )

        pruned_frame_idx = [memory_meta[i].get("frame_idx") for i in delete_indices]
        for i in delete_indices:
            to_cat_memory.pop(i)
            to_cat_memory_pos_embed.pop(i)
            memory_meta.pop(i)

        debug["num_after"] = len(to_cat_memory)
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
                "pruned_indices": debug.get("pruned_indices", []),
                "pruned_frame_idx": debug.get("pruned_frame_idx", []),
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

