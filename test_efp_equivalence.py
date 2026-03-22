#!/usr/bin/env python3
"""
Unit test for checking whether SAM2NewBase(memory_prune_mode='efp')
reproduces the current inline EFP pruning logic.

Run from repo root, for example:
    python test_efp_equivalence.py

What it tests:
1) Original inline EFP logic (implemented here as oracle_original_efp)
2) Refactored EFP helper in sam2_newbase.py

It compares:
- pruned indices
- pruned frame indices
- remaining memory tensors
- remaining pos-embed tensors
- remaining metadata order

This is the fastest and cleanest way to verify "behavior-preserving refactor"
without touching the old sam2_base.py or instantiating the full SurgSAM2 model.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Make sure we can import the local repo package when running from repo root.
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam2.modeling.sam2_newbase import SAM2NewBase  # noqa: E402


@dataclass
class TrialConfig:
    n_memories: int
    num_maskmem: int
    num_frame_to_prune: int
    hw: int
    batch: int
    channels: int
    seed: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_tensor_list(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x.clone() for x in xs]


def clone_meta_list(xs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(x) for x in xs]


def build_random_case(cfg: TrialConfig, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
    """
    Construct synthetic frame-memory tensors of shape [HW, B, C], which is exactly
    what the pruning helper expects.
    """
    set_seed(cfg.seed)
    to_cat_memory: List[torch.Tensor] = []
    to_cat_memory_pos_embed: List[torch.Tensor] = []
    memory_meta: List[Dict[str, Any]] = []

    # Use deterministic-but-varied frame indices so metadata alignment is testable.
    # First few can be "conditioning" and later ones "non-conditioning".
    frame_indices = list(range(100, 100 + cfg.n_memories))

    for i in range(cfg.n_memories):
        mem = torch.randn(cfg.hw, cfg.batch, cfg.channels, device=device)
        pos = torch.randn(cfg.hw, cfg.batch, cfg.channels, device=device)
        to_cat_memory.append(mem)
        to_cat_memory_pos_embed.append(pos)
        memory_meta.append(
            {
                "frame_idx": frame_indices[i],
                "t_pos": i,
                "is_conditioning": i < max(1, cfg.n_memories // 3),
                "source": "cond" if i < max(1, cfg.n_memories // 3) else "non_cond",
            }
        )

    return to_cat_memory, to_cat_memory_pos_embed, memory_meta


def oracle_original_efp(
    to_cat_memory: List[torch.Tensor],
    to_cat_memory_pos_embed: List[torch.Tensor],
    memory_meta: List[Dict[str, Any]],
    num_maskmem: int,
    num_frame_to_prune: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Exact oracle of the inline EFP block from the user's current custom sam2_base.py.

    Important:
    - protects index 0 and index -1
    - candidates are [1:-1]
    - uses cosine similarity to last memory
    - sums over HW, averages over batch
    - removes top-k most similar
    """
    n = len(to_cat_memory)
    debug: Dict[str, Any] = {
        "mode": "oracle_original_inline_efp",
        "num_before": n,
        "num_after": n,
        "num_to_prune": 0,
        "candidate_frame_idx": [],
        "candidate_similarities": [],
        "pruned_indices": [],
        "pruned_frame_idx": [],
    }

    if n <= 2:
        return to_cat_memory, to_cat_memory_pos_embed, memory_meta, debug

    if (n + num_frame_to_prune) <= num_maskmem:
        return to_cat_memory, to_cat_memory_pos_embed, memory_meta, debug

    this_num_frame_to_prune = n + num_frame_to_prune - num_maskmem
    num_candidates = n - 2
    this_num_frame_to_prune = min(this_num_frame_to_prune, num_candidates)
    if this_num_frame_to_prune <= 0:
        return to_cat_memory, to_cat_memory_pos_embed, memory_meta, debug

    last_vision_feature = to_cat_memory[-1]
    candidate_frames = torch.stack(to_cat_memory[1:-1], dim=0)
    last_frame_expanded = last_vision_feature.unsqueeze(0).expand_as(candidate_frames)

    similarities = F.cosine_similarity(candidate_frames, last_frame_expanded, dim=-1)
    similarities = torch.sum(similarities, dim=1)
    similarities = similarities.mean(dim=1)

    _, sorted_indices = torch.sort(similarities, descending=True)
    delete_indices = (sorted_indices[:this_num_frame_to_prune] + 1).tolist()
    delete_indices = sorted(delete_indices, reverse=True)

    pruned_frame_idx = [memory_meta[i]["frame_idx"] for i in delete_indices]

    for i in delete_indices:
        to_cat_memory.pop(i)
        to_cat_memory_pos_embed.pop(i)
        memory_meta.pop(i)

    debug.update(
        {
            "num_after": len(to_cat_memory),
            "num_to_prune": this_num_frame_to_prune,
            "candidate_frame_idx": [m["frame_idx"] for m in memory_meta[:-1][1:]] if len(memory_meta) > 2 else [],
            "candidate_similarities": similarities.detach().cpu().tolist(),
            "pruned_indices": delete_indices,
            "pruned_frame_idx": pruned_frame_idx,
        }
    )
    return to_cat_memory, to_cat_memory_pos_embed, memory_meta, debug


def make_newbase_harness(num_maskmem: int, num_frame_to_prune: int) -> SAM2NewBase:
    """
    Create a lightweight SAM2NewBase instance *without* running full __init__, so we
    can directly test the helper methods without building the full SurgSAM2 model.
    """
    model = SAM2NewBase.__new__(SAM2NewBase)
    model.num_maskmem = num_maskmem
    model.num_frame_to_prune = num_frame_to_prune
    model.memory_prune_mode = "efp"
    model.protect_conditioning_memories = False
    model.memory_similarity_threshold = None
    model.memory_min_temporal_gap = 0
    model.debug_memory_pruning = True
    model._last_memory_prune_debug = {}
    return model


def compare_tensor_lists(a: List[torch.Tensor], b: List[torch.Tensor], atol: float = 0.0, rtol: float = 0.0) -> Tuple[bool, str]:
    if len(a) != len(b):
        return False, f"length mismatch: {len(a)} vs {len(b)}"
    for i, (xa, xb) in enumerate(zip(a, b)):
        if xa.shape != xb.shape:
            return False, f"tensor[{i}] shape mismatch: {tuple(xa.shape)} vs {tuple(xb.shape)}"
        if not torch.allclose(xa, xb, atol=atol, rtol=rtol):
            max_abs = (xa - xb).abs().max().item()
            return False, f"tensor[{i}] values differ (max_abs={max_abs})"
    return True, "ok"


def compare_meta_lists(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if len(a) != len(b):
        return False, f"meta length mismatch: {len(a)} vs {len(b)}"
    for i, (ma, mb) in enumerate(zip(a, b)):
        if ma != mb:
            return False, f"meta[{i}] mismatch: {ma} vs {mb}"
    return True, "ok"


def run_single_trial(cfg: TrialConfig, device: torch.device, verbose: bool = False) -> bool:
    mem0, pos0, meta0 = build_random_case(cfg, device=device)

    # Oracle copies
    mem_oracle = clone_tensor_list(mem0)
    pos_oracle = clone_tensor_list(pos0)
    meta_oracle = clone_meta_list(meta0)

    # Newbase copies
    mem_new = clone_tensor_list(mem0)
    pos_new = clone_tensor_list(pos0)
    meta_new = clone_meta_list(meta0)

    # Run oracle
    mem_oracle, pos_oracle, meta_oracle, oracle_dbg = oracle_original_efp(
        mem_oracle,
        pos_oracle,
        meta_oracle,
        num_maskmem=cfg.num_maskmem,
        num_frame_to_prune=cfg.num_frame_to_prune,
    )

    # Run newbase EFP helper
    model = make_newbase_harness(
        num_maskmem=cfg.num_maskmem,
        num_frame_to_prune=cfg.num_frame_to_prune,
    )
    mem_new, pos_new, meta_new = model._efp_prune_memory_frames(
        mem_new,
        pos_new,
        meta_new,
    )
    new_dbg = copy.deepcopy(model._last_memory_prune_debug)

    ok = True
    reasons = []

    if oracle_dbg["pruned_indices"] != new_dbg.get("pruned_indices", []):
        ok = False
        reasons.append(
            f"pruned_indices mismatch: oracle={oracle_dbg['pruned_indices']} new={new_dbg.get('pruned_indices', [])}"
        )

    if oracle_dbg["pruned_frame_idx"] != new_dbg.get("pruned_frame_idx", []):
        ok = False
        reasons.append(
            f"pruned_frame_idx mismatch: oracle={oracle_dbg['pruned_frame_idx']} new={new_dbg.get('pruned_frame_idx', [])}"
        )

    same_mem, mem_msg = compare_tensor_lists(mem_oracle, mem_new)
    if not same_mem:
        ok = False
        reasons.append(f"remaining memory mismatch: {mem_msg}")

    same_pos, pos_msg = compare_tensor_lists(pos_oracle, pos_new)
    if not same_pos:
        ok = False
        reasons.append(f"remaining pos_embed mismatch: {pos_msg}")

    same_meta, meta_msg = compare_meta_lists(meta_oracle, meta_new)
    if not same_meta:
        ok = False
        reasons.append(f"remaining meta mismatch: {meta_msg}")

    if verbose or (not ok):
        print("=" * 88)
        print("TRIAL CONFIG:", cfg)
        print("ORACLE DEBUG:", oracle_dbg)
        print("NEWBASE DEBUG:", new_dbg)
        print("RESULT:", "PASS" if ok else "FAIL")
        if reasons:
            for r in reasons:
                print(" -", r)

    return ok


def generate_trial_grid(num_trials: int) -> List[TrialConfig]:
    """
    Generate a varied set of test configs.
    We intentionally include both no-prune and prune cases.
    """
    trials: List[TrialConfig] = []
    seeds = list(range(num_trials))
    n_options = [2, 3, 4, 5, 6, 7, 8, 9]
    maskmem_options = [4, 5, 6, 7]
    prune_options = [0, 1, 2, 3]
    hw_options = [16, 32, 64]
    batch_options = [1, 2]
    ch_options = [8, 16, 32]

    idx = 0
    for seed in seeds:
        cfg = TrialConfig(
            n_memories=n_options[idx % len(n_options)],
            num_maskmem=maskmem_options[(idx // 1) % len(maskmem_options)],
            num_frame_to_prune=prune_options[(idx // 2) % len(prune_options)],
            hw=hw_options[(idx // 3) % len(hw_options)],
            batch=batch_options[(idx // 4) % len(batch_options)],
            channels=ch_options[(idx // 5) % len(ch_options)],
            seed=seed,
        )
        trials.append(cfg)
        idx += 1
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SAM2NewBase EFP equivalence against original inline EFP logic.")
    parser.add_argument("--num-trials", type=int, default=50, help="Number of randomized trials to run.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for synthetic tensors.")
    parser.add_argument("--verbose", action="store_true", help="Print every trial, not only failures.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    device = torch.device(args.device)

    trials = generate_trial_grid(args.num_trials)
    passed = 0
    failed = 0

    for cfg in trials:
        ok = run_single_trial(cfg, device=device, verbose=args.verbose)
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "#" * 88)
    print(f"EFP equivalence summary: passed={passed}, failed={failed}, total={len(trials)}")
    if failed == 0:
        print("SUCCESS: refactored EFP matches the original inline EFP on all tested cases.")
    else:
        print("FAILURE: some trials differ. Re-run with --verbose to inspect details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
