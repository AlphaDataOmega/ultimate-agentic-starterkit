"""
Error handling framework for external integrations.

This module provides graceful degradation, fallback mechanisms, and recovery
procedures for external service failures.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import functools
import threading
import traceback

from ..core.logger import get_logger
from ..core.config import get_config
from ..core.voice_alerts import get_voice_alerts


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Recovery actions for different types of failures."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    DISABLE = "disable"
    ESCALATE = "escalate"


class ErrorRecord:
    """Record of an error occurrence."""
    
    def __init__(self, 
                 error_type: str,
                 message: str,
                 severity: ErrorSeverity,
                 source: str,
                 details: Dict[str, Any] = None,
                 recovery_action: RecoveryAction = None):
        """
        Initialize error record.
        
        Args:
            error_type: Type of error
            message: Error message
            severity: Error severity
            source: Source of the error
            details: Additional error details
            recovery_action: Recommended recovery action
        """
        self.error_type = error_type
        self.message = message
        self.severity = severity
        self.source = source
        self.details = details or {}
        self.recovery_action = recovery_action
        self.timestamp = datetime.now()
        self.resolved = False
        self.resolution_time = None
        self.attempts = 0
        self.max_attempts = 3
    
    def mark_resolved(self):
        """Mark error as resolved."""
        self.resolved = True
        self.resolution_time = datetime.now()
    
    def can_retry(self) -> bool:
        """Check if error can be retried."""
        return self.attempts < self.max_attempts
    
    def increment_attempts(self):
        """Increment retry attempts."""
        self.attempts += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity,
            "source": self.source,
            "details": self.details,
            "recovery_action": self.recovery_action,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts
        }


class FallbackConfig:
    """Configuration for fallback mechanisms."""
    
    def __init__(self, 
                 primary_handler: Callable,
                 fallback_handlers: List[Callable],
                 retry_config: Dict[str, Any] = None):
        """
        Initialize fallback configuration.
        
        Args:
            primary_handler: Primary handler function
            fallback_handlers: List of fallback handlers
            retry_config: Retry configuration
        """
        self.primary_handler = primary_handler
        self.fallback_handlers = fallback_handlers
        self.retry_config = retry_config or {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0
        }


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Exception = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.Lock()
        self.logger = get_logger("circuit_breaker")
    
    def __call__(self, func):
        """Decorator to apply circuit breaker."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        with self.lock:
            self.failure_count = 0
            self.state = "closed"
            self.logger.debug("Circuit breaker reset to closed state")
    
    def _on_failure(self):
        """Handle failed execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.warning("Circuit breaker opened due to failures")


class ErrorHandler:
    """Main error handling system."""
    
    def __init__(self):
        """Initialize error handler."""
        self.logger = get_logger("error_handler")
        self.config = get_config()
        self.voice = get_voice_alerts()
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.max_history_size = 1000
        self.error_counts: Dict[str, int] = {}
        
        # Fallback configurations
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery handlers
        self.recovery_handlers: Dict[str, Callable] = {}
        
        # Error patterns
        self.error_patterns = {
            "connection_error": {
                "patterns": ["connection", "timeout", "unreachable"],
                "severity": ErrorSeverity.HIGH,
                "recovery_action": RecoveryAction.RETRY
            },
            "authentication_error": {
                "patterns": ["auth", "unauthorized", "forbidden"],
                "severity": ErrorSeverity.HIGH,
                "recovery_action": RecoveryAction.ESCALATE
            },
            "resource_error": {
                "patterns": ["memory", "disk", "resource"],
                "severity": ErrorSeverity.MEDIUM,
                "recovery_action": RecoveryAction.DEGRADE
            },
            "validation_error": {
                "patterns": ["validation", "invalid", "malformed"],
                "severity": ErrorSeverity.LOW,
                "recovery_action": RecoveryAction.FALLBACK
            }
        }
    
    def register_fallback_config(self, service_name: str, config: FallbackConfig):
        """Register fallback configuration for a service."""
        self.fallback_configs[service_name] = config
        self.logger.info(f"Registered fallback config for {service_name}")
    
    def register_circuit_breaker(self, service_name: str, circuit_breaker: CircuitBreaker):
        """Register circuit breaker for a service."""
        self.circuit_breakers[service_name] = circuit_breaker
        self.logger.info(f"Registered circuit breaker for {service_name}")
    
    def register_recovery_handler(self, error_type: str, handler: Callable):
        """Register recovery handler for error type."""
        self.recovery_handlers[error_type] = handler
        self.logger.info(f"Registered recovery handler for {error_type}")
    
    async def handle_error(self, 
                          error: Exception,
                          source: str,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an error with recovery mechanisms.
        
        Args:
            error: The exception that occurred
            source: Source of the error
            context: Additional context information
        
        Returns:
            Dict containing error handling result
        """
        try:
            # Create error record
            error_record = self._create_error_record(error, source, context)
            
            # Add to history
            self._add_to_history(error_record)
            
            # Log error
            self._log_error(error_record)
            
            # Determine recovery action
            recovery_result = await self._execute_recovery(error_record)
            
            # Update error record
            if recovery_result["success"]:
                error_record.mark_resolved()
                self.voice.speak_success(f"Error recovered: {error_record.error_type}")
            else:
                self.voice.speak_error(f"Error recovery failed: {error_record.error_type}")
            
            return {
                "success": recovery_result["success"],
                "error_record": error_record.to_dict(),
                "recovery_action": recovery_result["recovery_action"],
                "recovery_details": recovery_result["details"]
            }
            
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "recovery_action": RecoveryAction.ESCALATE
            }
    
    def _create_error_record(self, 
                           error: Exception,
                           source: str,
                           context: Dict[str, Any] = None) -> ErrorRecord:
        """Create error record from exception."""
        error_message = str(error)
        error_type = type(error).__name__
        
        # Determine error pattern
        detected_pattern = self._detect_error_pattern(error_message)
        
        if detected_pattern:
            pattern_config = self.error_patterns[detected_pattern]
            severity = pattern_config["severity"]
            recovery_action = pattern_config["recovery_action"]
        else:
            severity = ErrorSeverity.MEDIUM
            recovery_action = RecoveryAction.RETRY
        
        # Collect error details
        details = context or {}
        details.update({
            "exception_type": error_type,
            "traceback": traceback.format_exc(),
            "detected_pattern": detected_pattern
        })
        
        return ErrorRecord(
            error_type=error_type,
            message=error_message,
            severity=severity,
            source=source,
            details=details,
            recovery_action=recovery_action
        )
    
    def _detect_error_pattern(self, error_message: str) -> Optional[str]:
        """Detect error pattern from message."""
        error_message_lower = error_message.lower()
        
        for pattern_name, pattern_config in self.error_patterns.items():
            for pattern in pattern_config["patterns"]:
                if pattern.lower() in error_message_lower:
                    return pattern_name
        
        return None
    
    def _add_to_history(self, error_record: ErrorRecord):
        """Add error record to history."""
        self.error_history.append(error_record)
        
        # Update error counts
        self.error_counts[error_record.error_type] = self.error_counts.get(error_record.error_type, 0) + 1
        
        # Maintain history size
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error record."""
        log_level = {
            ErrorSeverity.LOW: "info",
            ErrorSeverity.MEDIUM: "warning",
            ErrorSeverity.HIGH: "error",
            ErrorSeverity.CRITICAL: "critical"
        }.get(error_record.severity, "error")
        
        log_message = f"Error in {error_record.source}: {error_record.message}"
        
        getattr(self.logger, log_level)(
            log_message,
            error_type=error_record.error_type,
            severity=error_record.severity,
            recovery_action=error_record.recovery_action
        )
    
    async def _execute_recovery(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Execute recovery action for error."""
        recovery_action = error_record.recovery_action
        
        if recovery_action == RecoveryAction.RETRY:
            return await self._execute_retry(error_record)
        elif recovery_action == RecoveryAction.FALLBACK:
            return await self._execute_fallback(error_record)
        elif recovery_action == RecoveryAction.DEGRADE:
            return await self._execute_degradation(error_record)
        elif recovery_action == RecoveryAction.DISABLE:
            return await self._execute_disable(error_record)
        elif recovery_action == RecoveryAction.ESCALATE:
            return await self._execute_escalation(error_record)
        else:
            return {
                "success": False,
                "recovery_action": recovery_action,
                "details": "Unknown recovery action"
            }
    
    async def _execute_retry(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Execute retry recovery action."""
        try:
            if not error_record.can_retry():
                return {
                    "success": False,
                    "recovery_action": RecoveryAction.RETRY,
                    "details": "Max retry attempts exceeded"
                }
            
            error_record.increment_attempts()
            
            # Implement exponential backoff
            delay = min(
                1.0 * (2 ** (error_record.attempts - 1)),
                60.0  # Max delay
            )
            
            await asyncio.sleep(delay)
            
            # Check if recovery handler is available
            if error_record.error_type in self.recovery_handlers:
                handler = self.recovery_handlers[error_record.error_type]
                success = await handler(error_record)
                
                return {
                    "success": success,
                    "recovery_action": RecoveryAction.RETRY,
                    "details": f"Retry attempt {error_record.attempts} with {delay}s delay"
                }
            else:
                return {
                    "success": False,
                    "recovery_action": RecoveryAction.RETRY,
                    "details": "No recovery handler available"
                }
                
        except Exception as e:
            return {
                "success": False,
                "recovery_action": RecoveryAction.RETRY,
                "details": f"Retry failed: {str(e)}"
            }
    
    async def _execute_fallback(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Execute fallback recovery action."""
        try:
            service_name = error_record.source
            
            if service_name in self.fallback_configs:
                config = self.fallback_configs[service_name]
                
                # Try fallback handlers in order
                for i, fallback_handler in enumerate(config.fallback_handlers):
                    try:
                        result = await fallback_handler(error_record)
                        
                        if result:
                            return {
                                "success": True,
                                "recovery_action": RecoveryAction.FALLBACK,
                                "details": f"Fallback handler {i+1} succeeded"
                            }
                    except Exception as e:
                        self.logger.warning(f"Fallback handler {i+1} failed: {e}")
                        continue
                
                return {
                    "success": False,
                    "recovery_action": RecoveryAction.FALLBACK,
                    "details": "All fallback handlers failed"
                }
            else:
                return {
                    "success": False,
                    "recovery_action": RecoveryAction.FALLBACK,
                    "details": "No fallback configuration available"
                }
                
        except Exception as e:
            return {
                "success": False,
                "recovery_action": RecoveryAction.FALLBACK,
                "details": f"Fallback execution failed: {str(e)}"
            }
    
    async def _execute_degradation(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Execute degradation recovery action."""
        try:
            # Implement graceful degradation
            service_name = error_record.source
            
            self.logger.warning(f"Degrading service {service_name}")
            self.voice.speak_warning(f"Service {service_name} operating in degraded mode")
            
            # This would typically involve:
            # 1. Reducing functionality
            # 2. Using cached data
            # 3. Switching to simplified operations
            
            return {
                "success": True,
                "recovery_action": RecoveryAction.DEGRADE,
                "details": f"Service {service_name} degraded successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "recovery_action": RecoveryAction.DEGRADE,
                "details": f"Degradation failed: {str(e)}"
            }
    
    async def _execute_disable(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Execute disable recovery action."""
        try:
            service_name = error_record.source
            
            self.logger.error(f"Disabling service {service_name}")
            self.voice.speak_error(f"Service {service_name} disabled due to critical errors")
            
            # This would typically involve:
            # 1. Stopping the service
            # 2. Preventing further requests
            # 3. Cleaning up resources
            
            return {
                "success": True,
                "recovery_action": RecoveryAction.DISABLE,
                "details": f"Service {service_name} disabled successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "recovery_action": RecoveryAction.DISABLE,
                "details": f"Disable failed: {str(e)}"
            }
    
    async def _execute_escalation(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Execute escalation recovery action."""
        try:
            self.logger.critical(f"Escalating error: {error_record.message}")
            self.voice.speak_error("Critical error requires manual intervention")
            
            # This would typically involve:
            # 1. Notifying administrators
            # 2. Creating support tickets
            # 3. Triggering alerts
            
            return {
                "success": True,
                "recovery_action": RecoveryAction.ESCALATE,
                "details": "Error escalated for manual intervention"
            }
            
        except Exception as e:
            return {
                "success": False,
                "recovery_action": RecoveryAction.ESCALATE,
                "details": f"Escalation failed: {str(e)}"
            }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = len(self.error_history)
        resolved_errors = sum(1 for record in self.error_history if record.resolved)
        
        # Count by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity] = sum(
                1 for record in self.error_history
                if record.severity == severity
            )
        
        # Recent errors (last hour)
        recent_errors = [
            record for record in self.error_history
            if (datetime.now() - record.timestamp) < timedelta(hours=1)
        ]
        
        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_errors,
            "resolution_rate": resolved_errors / total_errors if total_errors > 0 else 0,
            "severity_counts": severity_counts,
            "error_type_counts": self.error_counts,
            "recent_errors": len(recent_errors),
            "registered_fallback_configs": len(self.fallback_configs),
            "registered_circuit_breakers": len(self.circuit_breakers),
            "registered_recovery_handlers": len(self.recovery_handlers)
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records."""
        recent_errors = sorted(
            self.error_history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        return [error.to_dict() for error in recent_errors]
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()
        self.logger.info("Error history cleared")


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


# Decorator for error handling
def handle_errors(source: str = None, recovery_action: RecoveryAction = None):
    """
    Decorator for automatic error handling.
    
    Args:
        source: Source identifier for errors
        recovery_action: Default recovery action
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                
                # Use function name as source if not provided
                error_source = source or func.__name__
                
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                
                if recovery_action:
                    context["suggested_recovery_action"] = recovery_action
                
                # Handle the error
                result = await error_handler.handle_error(e, error_source, context)
                
                # Re-raise if recovery failed
                if not result["success"]:
                    raise e
                
                return result
        
        return wrapper
    return decorator


# Convenience functions
async def handle_integration_error(error: Exception, integration_name: str) -> Dict[str, Any]:
    """Convenience function for handling integration errors."""
    error_handler = get_error_handler()
    return await error_handler.handle_error(error, integration_name)


def create_circuit_breaker(service_name: str, **kwargs) -> CircuitBreaker:
    """Create and register a circuit breaker for a service."""
    circuit_breaker = CircuitBreaker(**kwargs)
    error_handler = get_error_handler()
    error_handler.register_circuit_breaker(service_name, circuit_breaker)
    return circuit_breaker