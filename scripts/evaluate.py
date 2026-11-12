"""Evaluate a trained valence-arousal model."""

import sys
import _common
from unified_va.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
