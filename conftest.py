"""Pytest bootstrap.

Makes the project importable as a package root regardless of how pytest is
invoked (``pytest``, ``python -m pytest``, from another directory...).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
