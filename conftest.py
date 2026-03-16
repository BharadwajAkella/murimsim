"""Root conftest: adds the project root to sys.path so `murimsim` is importable
without requiring `pip install -e .` during development."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
