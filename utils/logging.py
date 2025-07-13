"""Logging configuration for MLX distributed inference."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    rank: int = 0,
    size: int = 1,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    name: str = "mlx_distributed"
) -> logging.Logger:
    """
    Set up logging configuration for distributed inference.
    
    Args:
        rank: Current process rank
        size: Total number of processes
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter with rank information
    formatter = logging.Formatter(
        f'[%(asctime)s] [Rank {rank}/{size}] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add rank to filename if multiple processes
        if size > 1:
            log_path = log_path.parent / f"{log_path.stem}_rank{rank}{log_path.suffix}"
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "mlx_distributed") -> logging.Logger:
    """Get the configured logger instance."""
    return logging.getLogger(name)
