#!/usr/bin/env python3
"""README CLI shim — implementation lives in geogenie.eval.benchmark."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geogenie.eval.benchmark import main

if __name__ == "__main__":
    main()
