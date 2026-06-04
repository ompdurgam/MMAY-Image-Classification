"""
app/core/logging.py
Configure root logger once; every module gets a child logger.
"""
import logging


def setup_logging(level: str = "info") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)