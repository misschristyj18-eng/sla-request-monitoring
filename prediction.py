"""Deprecated compatibility wrapper.

This repository uses the files in `modules/` as the active implementation.
The root copies are kept only to avoid import errors from older scripts.
"""

from modules.prediction import *  # noqa: F401,F403

__all__ = [
    "MODEL_PATH",
    "INFO_PATH",
    "load_model",
    "load_info",
    "risk_label",
    "predict_frame",
    "top_reasons",
    "similar_rate",
    "show",
]
