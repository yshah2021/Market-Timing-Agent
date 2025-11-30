"""
Logging configuration for the Market Timing Agents system
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import time
import uuid

class EnhancedLogger:
    """Enhanced logger wrapper with session and timing capabilities"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.sessions = {}
        self.timers = {}
    
    def start_session(self, config: dict) -> str:
        """Start a new session and return session ID"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'config': config,
            'start_time': datetime.now(),
            'events': []
        }
        return session_id
    
    def start_timer(self, name: str):
        """Start a timer"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a timer and return elapsed time"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            return elapsed
        return 0.0
    
    def log_structured(self, event_type: str, **kwargs):
        """Log structured data"""
        log_data = {'event_type': event_type, **kwargs}
        self.logger.info(str(log_data))
    
    def log_metric(self, metric_name: str, metric_value: float, metric_unit: str):
        """Log a metric"""
        metric_data = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'metric_unit': metric_unit,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"METRIC: {metric_data}")
    
    # Delegate standard logging methods
    def info(self, msg):
        if isinstance(msg, dict):
            self.logger.info(str(msg))
        else:
            self.logger.info(msg)
    
    def warning(self, msg):
        if isinstance(msg, dict):
            self.logger.warning(str(msg))
        else:
            self.logger.warning(msg)
    
    def error(self, msg):
        if isinstance(msg, dict):
            self.logger.error(str(msg))
        else:
            self.logger.error(msg)
    
    def debug(self, msg):
        if isinstance(msg, dict):
            self.logger.debug(str(msg))
        else:
            self.logger.debug(msg)

def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Set up logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler with rotation
    log_file = log_path / f"market_timing_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    if not root_logger.handlers:  # Avoid duplicate handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    return root_logger

def get_logger(name: str) -> EnhancedLogger:
    """Get an enhanced logger instance"""
    # Ensure logging is set up
    setup_logging()
    return EnhancedLogger(name)