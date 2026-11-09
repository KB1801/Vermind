"""
Structured Colorized Logger for VeriMind Enterprise Platform.
Outputs clean console logs and JSON formatted logs for production observability.
"""
import logging
import sys
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "filename": record.filename,
            "line": record.lineno,
            "message": record.getMessage()
        }
        if hasattr(record, "extra_data"):
            log_obj["extra"] = record.extra_data
        return json.dumps(log_obj)

class ColoredFormatter(logging.Formatter):
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    FORMAT = "%(asctime)s - [%(levelname)s] - %(name)s (%(filename)s:%(lineno)d) - %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logger(name: str = "VeriMind", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()