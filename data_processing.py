"""Deprecated compatibility wrapper.

This repository uses the files in `modules/` as the active implementation.
The root copies are kept only to avoid import errors from older scripts.
"""

from modules.data_processing import *  # noqa: F401,F403

__all__ = [
    "CAT_FEATURES",
    "NUM_FEATURES",
    "FEATURES",
    "FORBIDDEN",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "COLUMN_ALIASES",
    "CHOICES",
    "normalize_columns",
    "validate_schema",
    "quality_summary",
    "prepare_data",
    "labeled_subset",
    "make_X",
]
