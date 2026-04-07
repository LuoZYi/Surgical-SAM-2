from __future__ import annotations

import importlib.util
from pathlib import Path


_ORIGINAL_SAM2_BASE_PATH = Path("/mnt/e/sam_2/sam2/sam2/modeling/sam2_base.py")

if not _ORIGINAL_SAM2_BASE_PATH.is_file():
    raise FileNotFoundError(
        f"Original SAM2 base file not found: {_ORIGINAL_SAM2_BASE_PATH}"
    )

_spec = importlib.util.spec_from_file_location(
    "sam2_original_modeling_sam2_base", _ORIGINAL_SAM2_BASE_PATH
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load module spec from {_ORIGINAL_SAM2_BASE_PATH}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

NO_OBJ_SCORE = _module.NO_OBJ_SCORE
SAM2Base = _module.SAM2Base
