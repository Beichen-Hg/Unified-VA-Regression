"""Logging helpers."""
import logging
import sys
from pathlib import Path


def get_logger(name: str = "dimabsa", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    return logger


def add_file_handler(logger: logging.Logger, path: Path):
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
