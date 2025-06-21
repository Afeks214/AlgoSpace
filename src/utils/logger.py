import structlog
import logging
from pathlib import Path
from datetime import datetime
import os
from typing import Optional

# Global flag to track if logging has been set up
_logging_configured = False


def setup_logging(log_level: Optional[str] = None, log_dir: Optional[str] = None) -> None:
    """
    Initialize structured logging for the AlgoSpace project
    
    Args:
        log_level: Logging level (defaults to environment variable or INFO)
        log_dir: Directory for log files (defaults to ./logs)
    """
    global _logging_configured
    
    # Only configure once
    if _logging_configured:
        return
    
    # Get log level from environment or parameter
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Configure Python logging
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters and handlers
    log_file = log_dir / f"algospace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Clear any existing handlers
    logging.root.handlers = []
    
    logging.basicConfig(
        level=log_level_num,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if os.getenv("ENVIRONMENT") == "development" 
            else structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    _logging_configured = True
    
    logger = structlog.get_logger()
    logger.info("Logging system initialized", 
                log_level=log_level, 
                log_file=str(log_file),
                environment=os.getenv("ENVIRONMENT", "production"))


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (typically module or class name)
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is set up
    if not _logging_configured:
        setup_logging()
    
    if name:
        return structlog.get_logger(name)
    else:
        return structlog.get_logger()


# Legacy function for backward compatibility
def setup_logger(log_level: str = "INFO", log_dir: Optional[str] = None) -> structlog.BoundLogger:
    """
    Initialize structured logging for the AlgoSpace project
    (Legacy function - use setup_logging() instead)
    """
    setup_logging(log_level, log_dir)
    return get_logger()