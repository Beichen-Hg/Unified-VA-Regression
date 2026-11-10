"""Use the local checkout in script subprocesses."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
SOURCE = str(ROOT / "src")
sys.path.insert(0, SOURCE)
os.environ["PYTHONPATH"] = SOURCE + os.pathsep + os.environ.get("PYTHONPATH", "")
