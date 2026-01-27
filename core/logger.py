"""
Logging configuration for the Face Recognition System.
Provides structured logging with file rotation and console output.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "face_recognition",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, only console logging)
        level: Logging level (default: INFO)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always add)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger instance
_log_file = Path(__file__).parent.parent / "logs" / "app.log"
logger = setup_logger(
    name="face_recognition",
    log_file=_log_file,
    level=logging.INFO
)
