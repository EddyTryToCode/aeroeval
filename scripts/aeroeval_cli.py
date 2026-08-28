#!/usr/bin/env python
"""
Standalone CLI script for AeroEval Evaluation Platform.
"""

import sys
from pathlib import Path

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from aeroeval.cli import main

if __name__ == "__main__":
    main()
