"""
Test logging system.
"""

import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from StarterKit.core.logger import (
    AgentContextFilter,
    StructuredFormatter,
    ConsoleFormatter,
    PerformanceTracker,
    StarterKitLogger,
    get_logger,
    setup_global_logging,
    log_system_info,
    agent_execution_context,
    performance_monitoring
)


class TestAgentContextFilter:
    """Test agent context filter."""
    
    def test_filter_initialization(self):
        """Test filter initialization."""
        filter_obj = AgentContextFilter()
        assert hasattr(filter_obj, 'local')
    
    def test_filter_default_values(self):
        """Test filter with default values."""
        filter_obj = AgentContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        assert result is True
        assert record.agent_id == 'system'
        assert record.task_id == 'none'
    
    def test_set_agent_context(self):
        """Test setting agent context."""
        filter_obj = AgentContextFilter()
        filter_obj.set_agent_context("agent_001", "task_123")
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        assert record.agent_id == 'agent_001'
        assert record.task_id == 'task_123'
    
    def test_clear_agent_context(self):
        """Test clearing agent context."""
        filter_obj = AgentContextFilter()
        filter_obj.set_agent_context("agent_001", "task_123")
        filter_obj.clear_agent_context()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        assert record.agent_id == 'system'
        assert record.task_id == 'none'
    
    def test_thread_local_context(self):
        """Test thread-local context isolation."""
        filter_obj = AgentContextFilter()
        
        results = {}
        
        def set_context_thread_1():
            filter_obj.set_agent_context("agent_001", "task_001")
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="Test", args=(), exc_info=None
            )
            filter_obj.filter(record)
            results["thread_1"] = (record.agent_id, record.task_id)
        
        def set_context_thread_2():
            filter_obj.set_agent_context("agent_002", "task_002")
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="Test", args=(), exc_info=None
            )
            filter_obj.filter(record)
            results["thread_2"] = (record.agent_id, record.task_id)
        
        thread1 = threading.Thread(target=set_context_thread_1)
        thread2 = threading.Thread(target=set_context_thread_2)
        
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        
        assert results["thread_1"] == ("agent_001", "task_001")
        assert results["thread_2"] == ("agent_002", "task_002")


class TestStructuredFormatter:
    """Test structured JSON formatter."""
    
    def test_format_basic_record(self):
        """Test formatting basic log record."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add agent context
        record.agent_id = "agent_001"
        record.task_id = "task_123"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.module"
        assert log_data["message"] == "Test message"
        assert log_data["agent_id"] == "agent_001"
        assert log_data["task_id"] == "task_123"
        assert log_data["module"] == "test"
        assert log_data["function"] == "<module>"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
    
    def test_format_with_exception(self):
        """Test formatting record with exception."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
            record.agent_id = "system"
            record.task_id = "none"
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            assert "exception" in log_data
            assert log_data["exception"]["type"] == "ValueError"
            assert log_data["exception"]["message"] == "Test exception"
            assert isinstance(log_data["exception"]["traceback"], list)
    
    def test_format_with_extra_fields(self):
        """Test formatting record with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.agent_id = "system"
        record.task_id = "none"
        record.custom_field = "custom_value"
        record.numeric_field = 42
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["custom_field"] == "custom_value"
        assert log_data["numeric_field"] == 42


class TestConsoleFormatter:
    """Test console formatter."""
    
    def test_format_basic_record(self):
        """Test formatting basic record for console."""
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add agent context
        record.agent_id = "agent_001"
        record.task_id = "task_123"
        
        formatted = formatter.format(record)
        
        assert "INFO" in formatted
        assert "agent_001:task_123" in formatted
        assert "test.module" in formatted
        assert "Test message" in formatted
    
    def test_format_with_colors(self):
        """Test that color codes are included."""
        formatter = ConsoleFormatter()
        
        # Test different log levels
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL")
        ]
        
        for level, level_name in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None
            )
            record.agent_id = "system"
            record.task_id = "none"
            
            formatted = formatter.format(record)
            assert level_name in formatted
            # Check that color codes are present
            assert "\033[" in formatted  # ANSI color codes
    
    def test_format_with_exception(self):
        """Test formatting with exception."""
        formatter = ConsoleFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
            record.agent_id = "system"
            record.task_id = "none"
            
            formatted = formatter.format(record)
            assert "Error occurred" in formatted
            assert "ValueError" in formatted
            assert "Test exception" in formatted


class TestPerformanceTracker:
    """Test performance tracker."""
    
    def test_start_end_timer(self):
        """Test starting and ending a timer."""
        tracker = PerformanceTracker()
        
        tracker.start_timer("test_operation")
        time.sleep(0.1)  # Small delay
        duration = tracker.end_timer("test_operation")
        
        assert duration >= 0.1
        assert duration < 0.2  # Should be around 0.1 seconds
        assert "test_operation" in tracker.get_metrics()
    
    def test_multiple_timers(self):
        """Test multiple concurrent timers."""
        tracker = PerformanceTracker()
        
        tracker.start_timer("operation_1")
        time.sleep(0.05)
        tracker.start_timer("operation_2")
        time.sleep(0.05)
        
        duration_1 = tracker.end_timer("operation_1")
        duration_2 = tracker.end_timer("operation_2")
        
        assert duration_1 >= 0.1
        assert duration_2 >= 0.05
        assert duration_1 > duration_2
        
        metrics = tracker.get_metrics()
        assert "operation_1" in metrics
        assert "operation_2" in metrics
    
    def test_end_nonexistent_timer(self):
        """Test ending a timer that doesn't exist."""
        tracker = PerformanceTracker()
        duration = tracker.end_timer("nonexistent")
        assert duration == 0.0
    
    def test_clear_metrics(self):
        """Test clearing metrics."""
        tracker = PerformanceTracker()
        tracker.start_timer("test")
        tracker.end_timer("test")
        
        assert len(tracker.get_metrics()) == 1
        tracker.clear_metrics()
        assert len(tracker.get_metrics()) == 0


class TestStarterKitLogger:
    """Test main logger class."""
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = StarterKitLogger("test_logger")
        assert logger.name == "test_logger"
        assert isinstance(logger.performance_tracker, PerformanceTracker)
        assert not logger._setup_complete
    
    @patch('StarterKit.core.logger.get_config')
    def test_logger_setup(self, mock_get_config):
        """Test logger setup."""
        # Mock configuration
        mock_config = Mock()
        mock_config.logging.level = "INFO"
        mock_config.logging.file_path = None
        mock_config.logging.structured = True
        mock_get_config.return_value = mock_config
        
        logger = StarterKitLogger("test_logger")
        logger._setup_logger()
        
        assert logger._setup_complete
        assert len(logger.logger.handlers) > 0
    
    @patch('StarterKit.core.logger.get_config')
    def test_logger_with_file_handler(self, mock_get_config):
        """Test logger with file handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Mock configuration
            mock_config = Mock()
            mock_config.logging.level = "INFO"
            mock_config.logging.file_path = str(log_file)
            mock_config.logging.structured = True
            mock_config.logging.rotation = False
            mock_get_config.return_value = mock_config
            
            logger = StarterKitLogger("test_logger")
            logger._setup_logger()
            
            # Test logging
            logger.info("Test message")
            
            # Check if file was created and has content
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content
    
    @patch('StarterKit.core.logger.get_config')
    def test_agent_context_manager(self, mock_get_config):
        """Test agent context manager."""
        mock_config = Mock()
        mock_config.logging.level = "INFO"
        mock_config.logging.file_path = None
        mock_config.logging.structured = True
        mock_get_config.return_value = mock_config
        
        logger = StarterKitLogger("test_logger")
        
        with logger.agent_context("agent_001", "task_123"):
            # Context should be set
            assert logger.agent_filter.local.agent_id == "agent_001"
            assert logger.agent_filter.local.task_id == "task_123"
        
        # Context should be cleared after exiting
        assert logger.agent_filter.local.agent_id is None
        assert logger.agent_filter.local.task_id is None
    
    @patch('StarterKit.core.logger.get_config')
    def test_performance_context_manager(self, mock_get_config):
        """Test performance context manager."""
        mock_config = Mock()
        mock_config.logging.level = "INFO"
        mock_config.logging.file_path = None
        mock_config.logging.structured = True
        mock_get_config.return_value = mock_config
        
        logger = StarterKitLogger("test_logger")
        
        with logger.performance_context("test_operation"):
            time.sleep(0.05)
        
        metrics = logger.get_performance_metrics()
        assert "test_operation" in metrics
        assert metrics["test_operation"] >= 0.05
    
    @patch('StarterKit.core.logger.get_config')
    def test_logging_methods(self, mock_get_config):
        """Test various logging methods."""
        mock_config = Mock()
        mock_config.logging.level = "DEBUG"
        mock_config.logging.file_path = None
        mock_config.logging.structured = True
        mock_get_config.return_value = mock_config
        
        logger = StarterKitLogger("test_logger")
        
        # Test all logging methods
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Test with extra kwargs
        logger.info("Test with extra", extra_field="extra_value")
    
    @patch('StarterKit.core.logger.get_config')
    def test_agent_logging_methods(self, mock_get_config):
        """Test agent-specific logging methods."""
        mock_config = Mock()
        mock_config.logging.level = "INFO"
        mock_config.logging.file_path = None
        mock_config.logging.structured = True
        mock_get_config.return_value = mock_config
        
        logger = StarterKitLogger("test_logger")
        
        # Test agent lifecycle logging
        logger.log_agent_start("agent_001", "task_123")
        logger.log_agent_complete("agent_001", "task_123", success=True)
        logger.log_agent_complete("agent_002", "task_456", success=False)
        
        # Test task progress logging
        logger.log_task_progress("task_123", 0.5, "Halfway complete")
        
        # Test API call logging
        logger.log_api_call("openai", "gpt-4", 1000, 2.5, success=True)
        logger.log_api_call("anthropic", "claude-3", 800, 1.8, success=False)
    
    def test_parse_size(self):
        """Test size string parsing."""
        logger = StarterKitLogger("test_logger")
        
        assert logger._parse_size("1024") == 1024
        assert logger._parse_size("1KB") == 1024
        assert logger._parse_size("1MB") == 1024 * 1024
        assert logger._parse_size("1GB") == 1024 * 1024 * 1024
        assert logger._parse_size("10mb") == 10 * 1024 * 1024


class TestGlobalLoggerFunctions:
    """Test global logger functions."""
    
    def test_get_logger_singleton(self):
        """Test that get_logger returns the same instance for same name."""
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        
        assert logger1 is logger2
        assert isinstance(logger1, StarterKitLogger)
    
    def test_get_logger_different_names(self):
        """Test that different names return different instances."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        
        assert logger1 is not logger2
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
    
    @patch('StarterKit.core.logger.get_config')
    def test_setup_global_logging(self, mock_get_config):
        """Test global logging setup."""
        mock_config = Mock()
        mock_config.logging.level = "INFO"
        mock_config.logging.dict.return_value = {"level": "INFO"}
        mock_get_config.return_value = mock_config
        
        setup_global_logging()
        
        # Check that root logger level is set
        assert logging.getLogger().level == logging.INFO
    
    @patch('StarterKit.core.logger.get_logger')
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_log_system_info(self, mock_disk_usage, mock_virtual_memory, 
                            mock_cpu_count, mock_get_logger):
        """Test system information logging."""
        # Mock system info
        mock_cpu_count.return_value = 4
        mock_virtual_memory.return_value = Mock(total=8192, available=4096)
        mock_disk_usage.return_value = Mock(percent=50.0)
        
        # Mock logger
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        log_system_info()
        
        # Check that logger was called
        mock_get_logger.assert_called_with('starterkit.system')
        mock_logger.info.assert_called_once()
        
        # Check that system info was included
        call_args = mock_logger.info.call_args
        assert 'extra' in call_args[1]
        extra_data = call_args[1]['extra']
        assert 'cpu_count' in extra_data
        assert 'memory_total' in extra_data
        assert 'disk_usage' in extra_data


class TestContextManagers:
    """Test context manager functions."""
    
    @patch('StarterKit.core.logger.get_logger')
    def test_agent_execution_context_success(self, mock_get_logger):
        """Test successful agent execution context."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        with agent_execution_context("agent_001", "task_123") as logger:
            assert logger is mock_logger
            # Simulate some work
            pass
        
        # Check that start and complete were logged
        mock_logger.log_agent_start.assert_called_once_with("agent_001", "task_123")
        mock_logger.log_agent_complete.assert_called_once_with("agent_001", "task_123", success=True)
    
    @patch('StarterKit.core.logger.get_logger')
    def test_agent_execution_context_failure(self, mock_get_logger):
        """Test failed agent execution context."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        with pytest.raises(ValueError):
            with agent_execution_context("agent_001", "task_123"):
                raise ValueError("Test error")
        
        # Check that start and failure were logged
        mock_logger.log_agent_start.assert_called_once_with("agent_001", "task_123")
        mock_logger.log_agent_complete.assert_called_once_with("agent_001", "task_123", success=False, error="Test error")
        mock_logger.exception.assert_called_once()
    
    @patch('StarterKit.core.logger.get_logger')
    def test_performance_monitoring_context(self, mock_get_logger):
        """Test performance monitoring context."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        with performance_monitoring("test_operation") as logger:
            assert logger is mock_logger
            time.sleep(0.01)  # Small delay
        
        # Check that performance context was used
        mock_logger.performance_context.assert_called_once_with("test_operation")


class TestLoggerErrorHandling:
    """Test error handling in logger."""
    
    @patch('StarterKit.core.logger.get_config')
    def test_logger_setup_with_invalid_config(self, mock_get_config):
        """Test logger setup with invalid configuration."""
        # Mock config that might cause issues
        mock_config = Mock()
        mock_config.logging.level = "INVALID_LEVEL"
        mock_config.logging.file_path = None
        mock_config.logging.structured = True
        mock_get_config.return_value = mock_config
        
        logger = StarterKitLogger("test_logger")
        
        # Should not raise exception, but handle gracefully
        logger._setup_logger()
        
        # Logger should still be functional
        assert logger.logger is not None
    
    @patch('StarterKit.core.logger.get_config')
    def test_logger_with_permission_error(self, mock_get_config):
        """Test logger with permission error on file creation."""
        # Mock config with inaccessible file path
        mock_config = Mock()
        mock_config.logging.level = "INFO"
        mock_config.logging.file_path = "/root/no_permission.log"
        mock_config.logging.structured = True
        mock_config.logging.rotation = False
        mock_get_config.return_value = mock_config
        
        logger = StarterKitLogger("test_logger")
        
        # Should handle permission error gracefully
        with pytest.raises(PermissionError):
            logger._setup_logger()
    
    @patch('StarterKit.core.logger.get_config')
    def test_exception_logging(self, mock_get_config):
        """Test exception logging."""
        mock_config = Mock()
        mock_config.logging.level = "INFO"
        mock_config.logging.file_path = None
        mock_config.logging.structured = True
        mock_get_config.return_value = mock_config
        
        logger = StarterKitLogger("test_logger")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("An error occurred")
        
        # Should not raise any additional exceptions