import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(name: str = "root") -> logging.Logger:
    """
    Sets up a logger that writes to both stderr and a daily log file.
    Log file location: ./logs/MM_DD_YYYY_debug.log
    """
    # Create logs directory if it doesn't exist
    log_dir_path = os.getenv("LOG_DIR", "logs")
    log_dir = Path(log_dir_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename based on current date
    date_str = datetime.now().strftime("%m_%d_%Y")
    log_file = log_dir / f"{date_str}_debug.log"

    # Configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if setup is called multiple times for the same logger
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
    
    return logger
