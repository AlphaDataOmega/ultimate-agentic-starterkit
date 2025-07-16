"""
Logging system for the Ultimate Agentic StarterKit.

This module provides structured logging with JSON format, agent ID tracking,
performance metrics, and comprehensive error handling.
"""

import logging
import logging.handlers
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
import time
import uuid
import traceback

from .config import get_config


class AgentContextFilter(logging.Filter):
    """
    Filter to add agent context to log records.
    """
    
    def __init__(self):
        super().__init__()
        self.local = threading.local()
    
    def filter(self, record):
        """Add agent context to log record."""
        # Get agent ID from thread local storage
        agent_id = getattr(self.local, 'agent_id', None)
        task_id = getattr(self.local, 'task_id', None)
        
        record.agent_id = agent_id or 'system'
        record.task_id = task_id or 'none'
        
        return True
    
    def set_agent_context(self, agent_id: str, task_id: Optional[str] = None):
        """Set agent context for current thread."""
        self.local.agent_id = agent_id
        self.local.task_id = task_id
    
    def clear_agent_context(self):
        """Clear agent context for current thread."""
        self.local.agent_id = None
        self.local.task_id = None


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs.
    """
    
    def format(self, record):
        """Format log record as structured JSON."""
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'agent_id': getattr(record, 'agent_id', 'system'),
            'task_id': getattr(record, 'task_id', 'none'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message',
                          'agent_id', 'task_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Custom formatter for console output with colors and agent tracking.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record for console output."""
        # Get color for log level
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get agent and task info
        agent_id = getattr(record, 'agent_id', 'system')
        task_id = getattr(record, 'task_id', 'none')
        
        # Format message
        message = record.getMessage()
        
        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        # Construct formatted message
        formatted = f"{color}[{timestamp}] {record.levelname:8s}{reset} " \
                   f"[{agent_id}:{task_id}] {record.name} - {message}"
        
        return formatted


class PerformanceTracker:
    """
    Track performance metrics for logging.
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start a performance timer."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a performance timer and return duration."""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics[name] = duration
            del self.start_times[name]
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    def clear_metrics(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.start_times.clear()


class StarterKitLogger:
    """
    Main logger class with enhanced functionality.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.performance_tracker = PerformanceTracker()
        self._setup_complete = False
    
    def _setup_logger(self):
        """Setup logger with configuration."""
        if self._setup_complete:
            return
        
        config = get_config()
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, config.logging.level))
        
        # Create agent context filter
        agent_filter = AgentContextFilter()
        
        # Setup file handler with structured logging
        if config.logging.file_path:
            file_path = Path(config.logging.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config.logging.rotation:
                # Use rotating file handler
                file_handler = logging.handlers.RotatingFileHandler(
                    file_path,
                    maxBytes=self._parse_size(config.logging.max_size),
                    backupCount=config.logging.backup_count
                )
            else:
                file_handler = logging.FileHandler(file_path)
            
            if config.logging.structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [%(agent_id)s:%(task_id)s] - %(message)s'
                ))
            
            file_handler.addFilter(agent_filter)
            self.logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConsoleFormatter())
        console_handler.addFilter(agent_filter)
        self.logger.addHandler(console_handler)
        
        # Store filter reference for context management
        self.agent_filter = agent_filter
        self._setup_complete = True
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '10MB') to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    @contextmanager
    def agent_context(self, agent_id: str, task_id: Optional[str] = None):
        """Context manager for agent-specific logging."""
        self._setup_logger()
        
        # Set agent context
        self.agent_filter.set_agent_context(agent_id, task_id)
        
        try:
            yield self
        finally:
            # Clear agent context
            self.agent_filter.clear_agent_context()
    
    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance tracking."""
        self._setup_logger()
        
        # Start timer
        self.performance_tracker.start_timer(operation_name)
        start_time = time.time()
        
        try:
            yield self
        finally:
            # End timer and log performance
            duration = self.performance_tracker.end_timer(operation_name)
            self.logger.info(f"Performance: {operation_name} completed in {duration:.3f}s",
                           extra={'operation': operation_name, 'duration': duration})
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._setup_logger()
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._setup_logger()
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._setup_logger()
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._setup_logger()
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._setup_logger()
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._setup_logger()
        self.logger.exception(message, extra=kwargs)
    
    def log_agent_start(self, agent_id: str, task_id: Optional[str] = None, **kwargs):
        """Log agent start event."""
        with self.agent_context(agent_id, task_id):
            self.info(f"Agent {agent_id} started", 
                     extra={'event': 'agent_start', 'agent_id': agent_id, 'task_id': task_id, **kwargs})
    
    def log_agent_complete(self, agent_id: str, task_id: Optional[str] = None, 
                          success: bool = True, **kwargs):
        """Log agent completion event."""
        with self.agent_context(agent_id, task_id):
            level = 'info' if success else 'error'
            status = 'completed' if success else 'failed'
            getattr(self, level)(f"Agent {agent_id} {status}", 
                               extra={'event': 'agent_complete', 'agent_id': agent_id, 
                                    'task_id': task_id, 'success': success, **kwargs})
    
    def log_task_progress(self, task_id: str, progress: float, message: str, **kwargs):
        """Log task progress."""
        self.info(f"Task {task_id} progress: {progress:.1%} - {message}",
                 extra={'event': 'task_progress', 'task_id': task_id, 
                       'progress': progress, **kwargs})
    
    def log_api_call(self, provider: str, model: str, tokens_used: int, 
                    duration: float, success: bool = True, **kwargs):
        """Log API call metrics."""
        level = 'info' if success else 'error'
        getattr(self, level)(f"API call to {provider}:{model} - "
                           f"tokens: {tokens_used}, duration: {duration:.3f}s",
                           extra={'event': 'api_call', 'provider': provider, 
                                 'model': model, 'tokens_used': tokens_used,
                                 'duration': duration, 'success': success, **kwargs})
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_tracker.get_metrics()
    
    def clear_performance_metrics(self):
        """Clear performance metrics."""
        self.performance_tracker.clear_metrics()


# Global logger instances
_loggers: Dict[str, StarterKitLogger] = {}


def get_logger(name: str) -> StarterKitLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        StarterKitLogger: Logger instance
    """
    if name not in _loggers:
        _loggers[name] = StarterKitLogger(name)
    return _loggers[name]


def setup_global_logging():
    """Setup global logging configuration."""
    config = get_config()
    
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, config.logging.level))
    
    # Setup default logger
    default_logger = get_logger('starterkit')
    default_logger._setup_logger()
    
    # Log startup message
    default_logger.info("StarterKit logging system initialized",
                       extra={'event': 'logging_init', 'config': config.logging.dict()})


def log_system_info():
    """Log system information for debugging."""
    import platform
    import psutil
    
    logger = get_logger('starterkit.system')
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    logger.info("System information collected",
               extra={'event': 'system_info', **system_info})


# Context managers for common logging patterns
@contextmanager
def agent_execution_context(agent_id: str, task_id: Optional[str] = None):
    """Context manager for agent execution logging."""
    logger = get_logger('starterkit.agent')
    
    with logger.agent_context(agent_id, task_id):
        logger.log_agent_start(agent_id, task_id)
        
        try:
            yield logger
            logger.log_agent_complete(agent_id, task_id, success=True)
        except Exception as e:
            logger.log_agent_complete(agent_id, task_id, success=False, error=str(e))
            logger.exception(f"Agent {agent_id} execution failed")
            raise


@contextmanager
def performance_monitoring(operation_name: str):
    """Context manager for performance monitoring."""
    logger = get_logger('starterkit.performance')
    
    with logger.performance_context(operation_name):
        yield logger