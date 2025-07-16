"""
Unit tests for Error Handling framework.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from StarterKit.integrations.error_handling import (
    ErrorHandler,
    ErrorRecord,
    ErrorSeverity,
    RecoveryAction,
    FallbackConfig,
    CircuitBreaker,
    get_error_handler,
    handle_errors
)


class TestErrorRecord:
    """Test suite for Error Record."""
    
    def test_initialization(self):
        """Test error record initialization."""
        record = ErrorRecord(
            error_type="ConnectionError",
            message="Connection failed",
            severity=ErrorSeverity.HIGH,
            source="test_service",
            details={"url": "http://test.com"},
            recovery_action=RecoveryAction.RETRY
        )
        
        assert record.error_type == "ConnectionError"
        assert record.message == "Connection failed"
        assert record.severity == ErrorSeverity.HIGH
        assert record.source == "test_service"
        assert record.details == {"url": "http://test.com"}
        assert record.recovery_action == RecoveryAction.RETRY
        assert record.resolved is False
        assert record.attempts == 0
        assert record.max_attempts == 3
    
    def test_mark_resolved(self):
        """Test marking error as resolved."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test"
        )
        
        assert record.resolved is False
        assert record.resolution_time is None
        
        record.mark_resolved()
        
        assert record.resolved is True
        assert record.resolution_time is not None
    
    def test_can_retry(self):
        """Test retry capability check."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test"
        )
        
        assert record.can_retry() is True
        
        record.attempts = 3
        assert record.can_retry() is False
    
    def test_increment_attempts(self):
        """Test incrementing attempts."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test"
        )
        
        assert record.attempts == 0
        
        record.increment_attempts()
        
        assert record.attempts == 1
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test",
            details={"key": "value"},
            recovery_action=RecoveryAction.RETRY
        )
        
        record_dict = record.to_dict()
        
        assert record_dict["error_type"] == "TestError"
        assert record_dict["message"] == "Test message"
        assert record_dict["severity"] == ErrorSeverity.LOW
        assert record_dict["source"] == "test"
        assert record_dict["details"] == {"key": "value"}
        assert record_dict["recovery_action"] == RecoveryAction.RETRY
        assert record_dict["resolved"] is False
        assert record_dict["attempts"] == 0


class TestCircuitBreaker:
    """Test suite for Circuit Breaker."""
    
    def test_initialization(self):
        """Test circuit breaker initialization."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=ValueError
        )
        
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.recovery_timeout == 30
        assert circuit_breaker.expected_exception == ValueError
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful function."""
        circuit_breaker = CircuitBreaker(failure_threshold=2)
        
        @circuit_breaker
        async def test_function():
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self):
        """Test circuit breaker with failures."""
        circuit_breaker = CircuitBreaker(failure_threshold=2)
        
        @circuit_breaker
        async def test_function():
            raise Exception("Test error")
        
        # First failure
        with pytest.raises(Exception):
            await test_function()
        
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await test_function()
        
        assert circuit_breaker.state == "open"
        assert circuit_breaker.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state."""
        circuit_breaker = CircuitBreaker(failure_threshold=1)
        
        @circuit_breaker
        async def test_function():
            raise Exception("Test error")
        
        # Trigger circuit open
        with pytest.raises(Exception):
            await test_function()
        
        assert circuit_breaker.state == "open"
        
        # Should raise circuit breaker exception
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await test_function()


class TestErrorHandler:
    """Test suite for Error Handler."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        return ErrorHandler()
    
    def test_initialization(self, error_handler):
        """Test error handler initialization."""
        assert error_handler.error_history == []
        assert error_handler.error_counts == {}
        assert error_handler.fallback_configs == {}
        assert error_handler.circuit_breakers == {}
        assert error_handler.recovery_handlers == {}
        assert len(error_handler.error_patterns) > 0
    
    def test_register_fallback_config(self, error_handler):
        """Test registering fallback configuration."""
        config = FallbackConfig(
            primary_handler=Mock(),
            fallback_handlers=[Mock(), Mock()]
        )
        
        error_handler.register_fallback_config("test_service", config)
        
        assert "test_service" in error_handler.fallback_configs
        assert error_handler.fallback_configs["test_service"] is config
    
    def test_register_circuit_breaker(self, error_handler):
        """Test registering circuit breaker."""
        circuit_breaker = CircuitBreaker()
        
        error_handler.register_circuit_breaker("test_service", circuit_breaker)
        
        assert "test_service" in error_handler.circuit_breakers
        assert error_handler.circuit_breakers["test_service"] is circuit_breaker
    
    def test_register_recovery_handler(self, error_handler):
        """Test registering recovery handler."""
        handler = Mock()
        
        error_handler.register_recovery_handler("TestError", handler)
        
        assert "TestError" in error_handler.recovery_handlers
        assert error_handler.recovery_handlers["TestError"] is handler
    
    @pytest.mark.asyncio
    async def test_handle_error_success(self, error_handler):
        """Test successful error handling."""
        # Mock recovery handler
        recovery_handler = AsyncMock(return_value=True)
        error_handler.register_recovery_handler("ValueError", recovery_handler)
        
        error = ValueError("Test error")
        
        result = await error_handler.handle_error(error, "test_service")
        
        assert result["success"] is True
        assert result["error_record"]["error_type"] == "ValueError"
        assert result["error_record"]["resolved"] is True
        assert result["recovery_action"] == RecoveryAction.RETRY
    
    @pytest.mark.asyncio
    async def test_handle_error_failure(self, error_handler):
        """Test failed error handling."""
        # Mock recovery handler that fails
        recovery_handler = AsyncMock(return_value=False)
        error_handler.register_recovery_handler("ValueError", recovery_handler)
        
        error = ValueError("Test error")
        
        result = await error_handler.handle_error(error, "test_service")
        
        assert result["success"] is False
        assert result["error_record"]["error_type"] == "ValueError"
        assert result["error_record"]["resolved"] is False
    
    def test_detect_error_pattern_connection(self, error_handler):
        """Test detecting connection error pattern."""
        error_message = "Connection timeout occurred"
        
        pattern = error_handler._detect_error_pattern(error_message)
        
        assert pattern == "connection_error"
    
    def test_detect_error_pattern_authentication(self, error_handler):
        """Test detecting authentication error pattern."""
        error_message = "Authentication failed - unauthorized access"
        
        pattern = error_handler._detect_error_pattern(error_message)
        
        assert pattern == "authentication_error"
    
    def test_detect_error_pattern_none(self, error_handler):
        """Test detecting no error pattern."""
        error_message = "Unknown error occurred"
        
        pattern = error_handler._detect_error_pattern(error_message)
        
        assert pattern is None
    
    def test_create_error_record(self, error_handler):
        """Test creating error record."""
        error = ValueError("Test validation error")
        
        record = error_handler._create_error_record(error, "test_service", {"key": "value"})
        
        assert record.error_type == "ValueError"
        assert record.message == "Test validation error"
        assert record.source == "test_service"
        assert record.details["key"] == "value"
        assert record.details["exception_type"] == "ValueError"
    
    def test_add_to_history(self, error_handler):
        """Test adding error to history."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test"
        )
        
        error_handler._add_to_history(record)
        
        assert len(error_handler.error_history) == 1
        assert error_handler.error_history[0] is record
        assert error_handler.error_counts["TestError"] == 1
    
    def test_add_to_history_limit(self, error_handler):
        """Test error history size limit."""
        error_handler.max_history_size = 2
        
        # Add 3 records
        for i in range(3):
            record = ErrorRecord(
                error_type=f"TestError{i}",
                message=f"Test message {i}",
                severity=ErrorSeverity.LOW,
                source="test"
            )
            error_handler._add_to_history(record)
        
        # Should only keep last 2
        assert len(error_handler.error_history) == 2
        assert error_handler.error_history[0].error_type == "TestError1"
        assert error_handler.error_history[1].error_type == "TestError2"
    
    @pytest.mark.asyncio
    async def test_execute_retry_success(self, error_handler):
        """Test successful retry execution."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test"
        )
        
        # Mock recovery handler
        recovery_handler = AsyncMock(return_value=True)
        error_handler.register_recovery_handler("TestError", recovery_handler)
        
        result = await error_handler._execute_retry(record)
        
        assert result["success"] is True
        assert result["recovery_action"] == RecoveryAction.RETRY
        assert record.attempts == 1
    
    @pytest.mark.asyncio
    async def test_execute_retry_max_attempts(self, error_handler):
        """Test retry with max attempts exceeded."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test"
        )
        record.attempts = 3  # Max attempts
        
        result = await error_handler._execute_retry(record)
        
        assert result["success"] is False
        assert "Max retry attempts exceeded" in result["details"]
    
    @pytest.mark.asyncio
    async def test_execute_retry_no_handler(self, error_handler):
        """Test retry without recovery handler."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test"
        )
        
        result = await error_handler._execute_retry(record)
        
        assert result["success"] is False
        assert "No recovery handler available" in result["details"]
    
    @pytest.mark.asyncio
    async def test_execute_fallback_success(self, error_handler):
        """Test successful fallback execution."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test_service"
        )
        
        # Mock fallback handlers
        fallback1 = AsyncMock(return_value=False)
        fallback2 = AsyncMock(return_value=True)
        
        config = FallbackConfig(
            primary_handler=Mock(),
            fallback_handlers=[fallback1, fallback2]
        )
        
        error_handler.register_fallback_config("test_service", config)
        
        result = await error_handler._execute_fallback(record)
        
        assert result["success"] is True
        assert "Fallback handler 2 succeeded" in result["details"]
        
        fallback1.assert_called_once_with(record)
        fallback2.assert_called_once_with(record)
    
    @pytest.mark.asyncio
    async def test_execute_fallback_all_fail(self, error_handler):
        """Test fallback when all handlers fail."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test_service"
        )
        
        # Mock fallback handlers that all fail
        fallback1 = AsyncMock(return_value=False)
        fallback2 = AsyncMock(return_value=False)
        
        config = FallbackConfig(
            primary_handler=Mock(),
            fallback_handlers=[fallback1, fallback2]
        )
        
        error_handler.register_fallback_config("test_service", config)
        
        result = await error_handler._execute_fallback(record)
        
        assert result["success"] is False
        assert "All fallback handlers failed" in result["details"]
    
    @pytest.mark.asyncio
    async def test_execute_fallback_no_config(self, error_handler):
        """Test fallback without configuration."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test_service"
        )
        
        result = await error_handler._execute_fallback(record)
        
        assert result["success"] is False
        assert "No fallback configuration available" in result["details"]
    
    @pytest.mark.asyncio
    async def test_execute_degradation(self, error_handler):
        """Test degradation execution."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test_service"
        )
        
        result = await error_handler._execute_degradation(record)
        
        assert result["success"] is True
        assert result["recovery_action"] == RecoveryAction.DEGRADE
        assert "degraded successfully" in result["details"]
    
    @pytest.mark.asyncio
    async def test_execute_disable(self, error_handler):
        """Test disable execution."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test_service"
        )
        
        result = await error_handler._execute_disable(record)
        
        assert result["success"] is True
        assert result["recovery_action"] == RecoveryAction.DISABLE
        assert "disabled successfully" in result["details"]
    
    @pytest.mark.asyncio
    async def test_execute_escalation(self, error_handler):
        """Test escalation execution."""
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test_service"
        )
        
        result = await error_handler._execute_escalation(record)
        
        assert result["success"] is True
        assert result["recovery_action"] == RecoveryAction.ESCALATE
        assert "escalated for manual intervention" in result["details"]
    
    def test_get_error_statistics(self, error_handler):
        """Test getting error statistics."""
        # Add some test records
        for i in range(5):
            record = ErrorRecord(
                error_type=f"TestError{i}",
                message=f"Test message {i}",
                severity=ErrorSeverity.LOW,
                source="test"
            )
            if i < 3:
                record.mark_resolved()
            error_handler._add_to_history(record)
        
        stats = error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 5
        assert stats["resolved_errors"] == 3
        assert stats["resolution_rate"] == 0.6
        assert stats["severity_counts"][ErrorSeverity.LOW] == 5
    
    def test_get_recent_errors(self, error_handler):
        """Test getting recent errors."""
        # Add some test records
        for i in range(5):
            record = ErrorRecord(
                error_type=f"TestError{i}",
                message=f"Test message {i}",
                severity=ErrorSeverity.LOW,
                source="test"
            )
            error_handler._add_to_history(record)
        
        recent = error_handler.get_recent_errors(3)
        
        assert len(recent) == 3
        # Should be in reverse chronological order
        assert recent[0]["error_type"] == "TestError4"
        assert recent[1]["error_type"] == "TestError3"
        assert recent[2]["error_type"] == "TestError2"
    
    def test_clear_error_history(self, error_handler):
        """Test clearing error history."""
        # Add test record
        record = ErrorRecord(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.LOW,
            source="test"
        )
        error_handler._add_to_history(record)
        
        assert len(error_handler.error_history) == 1
        assert len(error_handler.error_counts) == 1
        
        error_handler.clear_error_history()
        
        assert len(error_handler.error_history) == 0
        assert len(error_handler.error_counts) == 0


class TestGlobalErrorHandler:
    """Test suite for global error handler."""
    
    def test_get_error_handler(self):
        """Test getting global error handler."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        assert handler1 is handler2  # Should be singleton
    
    @pytest.mark.asyncio
    async def test_handle_errors_decorator_success(self):
        """Test error handling decorator with success."""
        @handle_errors(source="test_function")
        async def test_function():
            return "success"
        
        result = await test_function()
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_handle_errors_decorator_failure(self):
        """Test error handling decorator with failure."""
        # Mock error handler
        with patch('StarterKit.integrations.error_handling.get_error_handler') as mock_get_handler:
            mock_handler = Mock()
            mock_handler.handle_error = AsyncMock(return_value={"success": False})
            mock_get_handler.return_value = mock_handler
            
            @handle_errors(source="test_function")
            async def test_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                await test_function()
            
            mock_handler.handle_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_errors_decorator_recovery(self):
        """Test error handling decorator with successful recovery."""
        # Mock error handler
        with patch('StarterKit.integrations.error_handling.get_error_handler') as mock_get_handler:
            mock_handler = Mock()
            mock_handler.handle_error = AsyncMock(return_value={"success": True})
            mock_get_handler.return_value = mock_handler
            
            @handle_errors(source="test_function")
            async def test_function():
                raise ValueError("Test error")
            
            result = await test_function()
            
            assert result == {"success": True}
            mock_handler.handle_error.assert_called_once()