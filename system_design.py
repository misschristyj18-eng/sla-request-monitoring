"""Deprecated compatibility wrapper.

This repository uses the files in `modules/` as the active implementation.
The root copies are kept only to avoid import errors from older scripts.
"""

from modules.system_design import *  # noqa: F401,F403

__all__ = [
    "show",
]
