# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimally invasive extension of SAM2Base that keeps the upstream file untouched.

What this adds on top of the current custom sam2_base.py:
- a real prune mode switch: off | efp | rule_based
- optional protection for conditioning memories
- a small rule-based controller that is aware of memory role and temporal closeness
- lightweight debug bookkeeping via `self._last_memory_prune_debug`

Recommended usage:
    from sam2.modeling.sam2_newbase import SAM2NewBase

Then point your model builder / config at SAM2NewBase instead of SAM2Base.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames


class SAM2NewBase(SAM2Base):
    def __init__(
        self,
        *args,
        memory_prune_mode: str = "efp",  # off | efp | rule_based
        protect_conditioning_memories: bool = False,
        memory_similarity_threshold: Optional[float] = None,
        memory_min_temporal_gap: int = 0,
        debug_memory_pruning: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_prune_mode = memory_prune_mode
        self.protect_conditioning_memories = protect_conditioning_memories
        self.memory_similarity_threshold = memory_similarity_threshold
        self.memory_min_temporal_gap = memory_min_temporal_gap
        self.debug_memory_pruning = debug_memory_pruning
        self._last_memory_prune_debug: Dict[str, Any] = {}

    def _make_memory_meta(
        self,
        frame_idx: Optional[int],
        t_pos: int,
        is_conditioning: bool,
        source: str,
    ) -> Dict[str, Any]:
        return {
            "frame_idx": frame_idx,
            "t_pos": t_pos,
            "is_conditioning": is_conditioning,
            "source": source,
        }

    def _compute_similarity_to_last(
        self,
        candidate_frames: torch.Tensor,
        last_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        candidate_frames: [N, HW, B, C]
        last_frame:       [HW, B, C]
        returns:          [N]
        """
        if candidate_frames.numel() == 0:
            return candidate_frames.new_zeros((0,))
        last_frame_expanded = last_frame.unsqueeze(0).expand_as(candidate_frames)
        similarities = F.cosine_similarity(candidate_frames, last_frame_expanded, dim=-1)
        similarities = similarities.sum(dim=1)   # [N, B]
        similarities = similarities.mean(dim=1)  # [N]
        return similarities

    def _estimate_temporal_closeness_to_last(
        self,
        candidate_meta: List[Dict[str, Any]],
        last_meta: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Higher score => temporally closer to the last memory => more redundant.
        Returns a tensor of shape [N].
        """
        last_idx = last_meta.get("frame_idx", None)
        if last_idx is None or len(candidate_meta) == 0:
            return torch.zeros(len(candidate_meta), device=device)

        closeness_scores = []
        for meta in candidate_meta:
            cand_idx = meta.get("frame_idx", None)
            if cand_idx is None:
                closeness_scores.append(0.0)
                continue
            gap = abs(last_idx - cand_idx)
            # closer frames receive larger score and are thus more likely to be pruned
            closeness = 1.0 / float(gap + 1)
            if self.memory_min_temporal_gap > 0 and gap < self.memory_min_temporal_gap:
                closeness += 1.0
            closeness_scores.append(closeness)
        return torch.tensor(closeness_scores, device=device)

    def _select_delete_indices_by_scores(
        self,
        base_indices: List[int],
        scores: torch.Tensor,
        metas: List[Dict[str, Any]],
        num_to_prune: int,
        prefer_non_cond: bool,
    ) -> List[int]:
        """
        Returns actual indices in `to_cat_memory` to delete.
        Larger score => more likely to be pruned.
        """
        if num_to_prune <= 0 or len(base_indices) == 0:
            return []

        if self.memory_similarity_threshold is not None:
            eligible = [
                i for i, score in enumerate(scores.tolist())
                if score >= self.memory_similarity_threshold
            ]
        else:
            eligible = list(range(len(base_indices)))

        def rank_subset(local_ids: List[int], k: int) -> List[int]:
            if k <= 0 or len(local_ids) == 0:
                return []
            subset_scores = scores[local_ids]
            _, subset_order = torch.sort(subset_scores, descending=True)
            chosen_local_ids = [local_ids[j] for j in subset_order[:k].tolist()]
            return [base_indices[j] for j in chosen_local_ids]

        if not prefer_non_cond:
            chosen = rank_subset(eligible, min(num_to_prune, len(eligible)))
            if len(chosen) < num_to_prune:
                remaining = [i for i in range(len(base_indices)) if base_indices[i] not in chosen]
                chosen.extend(rank_subset(remaining, num_to_prune - len(chosen)))
            return chosen

        non_cond_ids = [
            i for i in eligible if not metas[i].get("is_conditioning", False)
        ]
        cond_ids = [
            i for i in eligible if metas[i].get("is_conditioning", False)
        ]

        chosen = rank_subset(non_cond_ids, min(num_to_prune, len(non_cond_ids)))
        still_need = num_to_prune - len(chosen)
        if still_need > 0:
            already = set(chosen)
            cond_candidates = [i for i in cond_ids if base_indices[i] not in already]
            chosen.extend(rank_subset(cond_candidates, still_need))

        # If threshold filtered out too much, fall back to all candidates.
        if len(chosen) < num_to_prune:
            already = set(chosen)
            remaining_ids = [i for i in range(len(base_indices)) if base_indices[i] not in already]
            chosen.extend(rank_subset(remaining_ids, num_to_prune - len(chosen)))

        return chosen

    def _efp_prune_memory_frames(
        self,
        to_cat_memory: List[torch.Tensor],
        to_cat_memory_pos_embed: List[torch.Tensor],
        memory_meta: List[Dict[str, Any]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
        """Behavior-preserving refactor of the current inlined EFP logic."""
        n = len(to_cat_memory)
        if n <= 2:
            self._last_memory_prune_debug = {
                "mode": "efp",
                "num_before": n,
                "num_after": n,
                "pruned_indices": [],
                "pruned_frame_idx": [],
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        if (n + self.num_frame_to_prune) <= self.num_maskmem:
            self._last_memory_prune_debug = {
                "mode": "efp",
                "num_before": n,
                "num_after": n,
                "pruned_indices": [],
                "pruned_frame_idx": [],
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        num_to_prune = n + self.num_frame_to_prune - self.num_maskmem
        num_candidates = n - 2
        num_to_prune = min(num_to_prune, num_candidates)
        if num_to_prune <= 0:
            self._last_memory_prune_debug = {
                "mode": "efp",
                "num_before": n,
                "num_after": n,
                "pruned_indices": [],
                "pruned_frame_idx": [],
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        last_frame = to_cat_memory[-1]
        candidate_frames = torch.stack(to_cat_memory[1:-1], dim=0)
        candidate_meta = memory_meta[1:-1]
        similarities = self._compute_similarity_to_last(candidate_frames, last_frame)

        # Exact current behavior: sort descending by similarity and drop the top-k.
        _, sorted_indices = torch.sort(similarities, descending=True)
        delete_indices = (sorted_indices[:num_to_prune] + 1).tolist()
        delete_indices = sorted(delete_indices, reverse=True)

        pruned_frame_idx = [memory_meta[i].get("frame_idx") for i in delete_indices]
        for i in delete_indices:
            to_cat_memory.pop(i)
            to_cat_memory_pos_embed.pop(i)
            memory_meta.pop(i)

        self._last_memory_prune_debug = {
            "mode": "efp",
            "num_before": n,
            "num_after": len(to_cat_memory),
            "num_to_prune": num_to_prune,
            "candidate_frame_idx": [m.get("frame_idx") for m in candidate_meta],
            "candidate_similarities": similarities.detach().cpu().tolist(),
            "pruned_indices": delete_indices,
            "pruned_frame_idx": pruned_frame_idx,
        }
        return to_cat_memory, to_cat_memory_pos_embed, memory_meta

    def _rule_based_prune_memory_frames(
        self,
        to_cat_memory: List[torch.Tensor],
        to_cat_memory_pos_embed: List[torch.Tensor],
        memory_meta: List[Dict[str, Any]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
        """
        A lightweight, FYP-safe controller:
        - keeps first and last memory entries fixed
        - scores middle memories by redundancy to latest memory
        - adds a temporal-closeness term
        - can prefer pruning non-conditioning memories first
        """
        n = len(to_cat_memory)
        if n <= 2:
            self._last_memory_prune_debug = {
                "mode": "rule_based",
                "num_before": n,
                "num_after": n,
                "pruned_indices": [],
                "pruned_frame_idx": [],
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        if (n + self.num_frame_to_prune) <= self.num_maskmem:
            self._last_memory_prune_debug = {
                "mode": "rule_based",
                "num_before": n,
                "num_after": n,
                "pruned_indices": [],
                "pruned_frame_idx": [],
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        num_to_prune = n + self.num_frame_to_prune - self.num_maskmem
        num_candidates = n - 2
        num_to_prune = min(num_to_prune, num_candidates)
        if num_to_prune <= 0:
            self._last_memory_prune_debug = {
                "mode": "rule_based",
                "num_before": n,
                "num_after": n,
                "pruned_indices": [],
                "pruned_frame_idx": [],
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        last_frame = to_cat_memory[-1]
        last_meta = memory_meta[-1]
        candidate_frames = torch.stack(to_cat_memory[1:-1], dim=0)
        candidate_meta = memory_meta[1:-1]

        sim_to_last = self._compute_similarity_to_last(candidate_frames, last_frame)
        temporal_closeness = self._estimate_temporal_closeness_to_last(
            candidate_meta, last_meta, device=sim_to_last.device
        )
        # Simple score: high similarity + temporal closeness => more redundant.
        drop_scores = sim_to_last + 0.25 * temporal_closeness

        base_indices = list(range(1, n - 1))
        delete_indices = self._select_delete_indices_by_scores(
            base_indices=base_indices,
            scores=drop_scores,
            metas=candidate_meta,
            num_to_prune=num_to_prune,
            prefer_non_cond=self.protect_conditioning_memories,
        )
        delete_indices = sorted(delete_indices, reverse=True)

        pruned_frame_idx = [memory_meta[i].get("frame_idx") for i in delete_indices]
        for i in delete_indices:
            to_cat_memory.pop(i)
            to_cat_memory_pos_embed.pop(i)
            memory_meta.pop(i)

        self._last_memory_prune_debug = {
            "mode": "rule_based",
            "num_before": n,
            "num_after": len(to_cat_memory),
            "num_to_prune": num_to_prune,
            "candidate_frame_idx": [m.get("frame_idx") for m in candidate_meta],
            "candidate_is_conditioning": [m.get("is_conditioning") for m in candidate_meta],
            "candidate_similarities": sim_to_last.detach().cpu().tolist(),
            "candidate_temporal_closeness": temporal_closeness.detach().cpu().tolist(),
            "candidate_drop_scores": drop_scores.detach().cpu().tolist(),
            "pruned_indices": delete_indices,
            "pruned_frame_idx": pruned_frame_idx,
        }
        return to_cat_memory, to_cat_memory_pos_embed, memory_meta

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
                "num_before": len(to_cat_memory),
                "num_after": len(to_cat_memory),
                "pruned_indices": [],
                "pruned_frame_idx": [],
            }
            return to_cat_memory, to_cat_memory_pos_embed, memory_meta

        if self.memory_prune_mode == "efp":
            return self._efp_prune_memory_frames(
                to_cat_memory, to_cat_memory_pos_embed, memory_meta
            )

        if self.memory_prune_mode == "rule_based":
            return self._rule_based_prune_memory_frames(
                to_cat_memory, to_cat_memory_pos_embed, memory_meta
            )

        raise ValueError(
            f"Unknown memory_prune_mode={self.memory_prune_mode!r}. "
            "Expected one of: off | efp | rule_based."
        )

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

            # Keep richer metadata so pruning policies can be role-aware later.
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
                    )
                )

            # Prune only the frame memories. Keep the object-pointer logic below unchanged.
            to_cat_memory, to_cat_memory_pos_embed, memory_meta = self._prune_memory_frames(
                to_cat_memory,
                to_cat_memory_pos_embed,
                memory_meta,
            )

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
