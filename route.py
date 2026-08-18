"""Generate four diverse EcoRoute alternatives from a prepared road graph."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecoroute.routing import main  # noqa: E402


if __name__ == "__main__":
    main()
