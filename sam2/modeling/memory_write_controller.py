from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


DEFAULT_WRITE_FEATURE_NAMES = (
    "quality",
    "mask_cosine",
    "mask_change",
    "centroid_shift",
    "current_area_ratio",
    "area_change",
    "reference_temporal_gap",
    "bank_fill_ratio",
)

DEFAULT_MODE_WINDOW_FEATURE_NAMES = (
    "quality_curr",
    "quality_mean",
    "quality_min",
    "mask_cosine_curr",
    "mask_cosine_mean",
    "mask_cosine_min",
    "mask_change_curr",
    "mask_change_mean",
    "mask_change_max",
    "centroid_shift_curr",
    "centroid_shift_mean",
    "centroid_shift_max",
    "bank_fill_ratio_curr",
    "bank_fill_ratio_mean",
    "area_change_mean",
    "reference_gap_mean",
)


def build_feature_vector(
    source: Mapping[str, object],
    feature_names: Sequence[str] = DEFAULT_WRITE_FEATURE_NAMES,
) -> np.ndarray:
    values = []
    for name in feature_names:
        value = source.get(name, 0.0)
        if value is None:
            value = 0.0
        values.append(float(value))
    return np.asarray(values, dtype=np.float32)


def build_mode_window_feature_source(
    window_sources: Sequence[Mapping[str, object]],
) -> Mapping[str, float]:
    if len(window_sources) == 0:
        return {name: 0.0 for name in DEFAULT_MODE_WINDOW_FEATURE_NAMES}

    def values(name: str) -> list[float]:
        out = []
        for source in window_sources:
            value = source.get(name, 0.0)
            if value is None:
                value = 0.0
            out.append(float(value))
        return out

    quality = values("quality")
    mask_cosine = values("mask_cosine")
    mask_change = values("mask_change")
    centroid_shift = values("centroid_shift")
    bank_fill_ratio = values("bank_fill_ratio")
    area_change = values("area_change")
    reference_gap = values("reference_temporal_gap")

    return {
        "quality_curr": quality[-1],
        "quality_mean": float(np.mean(quality)),
        "quality_min": float(np.min(quality)),
        "mask_cosine_curr": mask_cosine[-1],
        "mask_cosine_mean": float(np.mean(mask_cosine)),
        "mask_cosine_min": float(np.min(mask_cosine)),
        "mask_change_curr": mask_change[-1],
        "mask_change_mean": float(np.mean(mask_change)),
        "mask_change_max": float(np.max(mask_change)),
        "centroid_shift_curr": centroid_shift[-1],
        "centroid_shift_mean": float(np.mean(centroid_shift)),
        "centroid_shift_max": float(np.max(centroid_shift)),
        "bank_fill_ratio_curr": bank_fill_ratio[-1],
        "bank_fill_ratio_mean": float(np.mean(bank_fill_ratio)),
        "area_change_mean": float(np.mean(area_change)),
        "reference_gap_mean": float(np.mean(reference_gap)),
    }


class OfflineWriteController(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (16, 8),
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                continue
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
