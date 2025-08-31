"""
Utilities for structured JSON logging.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from pythonjsonlogger import jsonlogger

def setup_logger(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """
    Configures a JSON logger with a rotating file handler.

    Args:
        log_dir: The directory to store log files.
        level: The minimum logging level (e.g., "INFO", "DEBUG").

    Returns:
        A configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pawikan.json.log"

    logger = logging.getLogger("pawikan_sentinel")
    logger.propagate = False # Prevent duplicate logs in parent loggers

    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the level after clearing handlers
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Formatter
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s %(extra)s"
    )

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB x 5 files
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (for debugging)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
