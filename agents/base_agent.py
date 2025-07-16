"""
Base Agent Class for the Ultimate Agentic StarterKit.

This module provides the abstract base class for all AI agents with common
functionality including retry logic, error handling, and confidence scoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import uuid
import asyncio
from datetime import datetime

from core.models import AgentResult, ProjectTask, TaskStatus, AgentType
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    
    Provides common functionality including:
    - Unique agent identification
    - Retry logic with exponential backoff
    - Error handling and logging
    - Confidence threshold management
    - Voice alert integration
    - Performance tracking
    """
    
    def __init__(self, agent_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_type: Type identifier for the agent
            config: Optional configuration dictionary
        """
        self.agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        self.agent_type = agent_type
        self.logger = get_logger(self.agent_id)
        self.voice = get_voice_alerts()
        
        # Configuration with defaults
        self.config = config or {}
        self.max_retries = self.config.get('max_retries', 3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.max_retry_delay = self.config.get('max_retry_delay', 60.0)
        self.timeout = self.config.get('timeout', 300)
        
        # State tracking
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = None
        self.created_at = datetime.now()
        
        self.logger.info(f"Agent {self.agent_id} initialized with config: {self.config}")
    
    @abstractmethod
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute the agent's primary function.
        
        Args:
            task: The project task to execute
            
        Returns:
            AgentResult: Result of the execution
        """
        pass
    
    async def execute_with_retry(self, task: ProjectTask) -> AgentResult:
        """
        Execute the agent with retry logic and error handling.
        
        Args:
            task: The project task to execute
            
        Returns:
            AgentResult: Result of the execution with retry handling
        """
        self.execution_count += 1
        start_time = time.time()
        
        self.logger.info(f"Starting execution for task {task.id}")
        self.voice.speak_agent_start(self.agent_id, f"processing task {task.title}")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Calculate retry delay with exponential backoff
                if attempt > 0:
                    delay = min(self.retry_delay * (2 ** (attempt - 1)), self.max_retry_delay)
                    self.logger.info(f"Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                
                # Execute the task with timeout
                result = await asyncio.wait_for(
                    self.execute(task),
                    timeout=self.timeout
                )
                
                # Check if result meets confidence threshold
                if result.success and result.confidence >= self.confidence_threshold:
                    self.success_count += 1
                    execution_time = time.time() - start_time
                    self.total_execution_time += execution_time
                    self.last_execution_time = execution_time
                    
                    self.logger.info(f"Task {task.id} completed successfully with confidence {result.confidence:.2f}")
                    self.voice.speak_agent_complete(self.agent_id, f"task {task.title} completed successfully")
                    
                    return result
                
                # Low confidence - treat as failure for retry logic
                if result.success:
                    self.logger.warning(f"Low confidence result: {result.confidence:.2f} < {self.confidence_threshold}")
                    last_error = f"Low confidence: {result.confidence:.2f}"
                else:
                    self.logger.error(f"Task execution failed: {result.error}")
                    last_error = result.error
                
            except asyncio.TimeoutError:
                error_msg = f"Task execution timed out after {self.timeout} seconds"
                self.logger.error(error_msg)
                last_error = error_msg
                
            except Exception as e:
                error_msg = f"Unexpected error during execution: {str(e)}"
                self.logger.exception(error_msg)
                last_error = error_msg
        
        # All retries exhausted
        self.failure_count += 1
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time
        self.last_execution_time = execution_time
        
        self.logger.error(f"Task {task.id} failed after {self.max_retries} attempts")
        self.voice.speak_error(f"Task {task.title} failed after {self.max_retries} attempts")
        
        return AgentResult(
            success=False,
            confidence=0.0,
            output=None,
            error=last_error or "Unknown error",
            execution_time=execution_time,
            agent_id=self.agent_id,
            timestamp=datetime.now()
        )
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics.
        
        Returns:
            Dict with agent statistics
        """
        total_executions = self.execution_count
        success_rate = (self.success_count / total_executions) if total_executions > 0 else 0.0
        avg_execution_time = (self.total_execution_time / total_executions) if total_executions > 0 else 0.0
        
        uptime = (datetime.now() - self.created_at).total_seconds()
        
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'created_at': self.created_at.isoformat(),
            'uptime_seconds': uptime,
            'total_executions': total_executions,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_execution_time,
            'last_execution_time': self.last_execution_time,
            'confidence_threshold': self.confidence_threshold,
            'max_retries': self.max_retries
        }
    
    def reset_stats(self):
        """Reset agent statistics."""
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = None
        self.created_at = datetime.now()
        
        self.logger.info(f"Agent {self.agent_id} statistics reset")
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update agent configuration.
        
        Args:
            config: New configuration dictionary
        """
        old_config = self.config.copy()
        self.config.update(config)
        
        # Update derived properties
        self.max_retries = self.config.get('max_retries', 3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.max_retry_delay = self.config.get('max_retry_delay', 60.0)
        self.timeout = self.config.get('timeout', 300)
        
        self.logger.info(f"Agent {self.agent_id} configuration updated: {old_config} -> {self.config}")
    
    def _calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on various indicators.
        
        Args:
            indicators: Dictionary of confidence indicators
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        # Base implementation - can be overridden by subclasses
        base_confidence = 0.5
        
        # Adjust based on indicators
        if 'error_count' in indicators:
            error_penalty = min(indicators['error_count'] * 0.1, 0.3)
            base_confidence -= error_penalty
        
        if 'completion_status' in indicators:
            if indicators['completion_status'] == 'complete':
                base_confidence += 0.3
            elif indicators['completion_status'] == 'partial':
                base_confidence += 0.1
        
        if 'validation_passed' in indicators:
            if indicators['validation_passed']:
                base_confidence += 0.2
            else:
                base_confidence -= 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def _validate_task(self, task: ProjectTask) -> bool:
        """
        Validate that the task is appropriate for this agent.
        
        Args:
            task: The task to validate
            
        Returns:
            bool: True if task is valid for this agent
        """
        if not task:
            return False
        
        if not task.title or not task.description:
            return False
        
        if task.status not in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
            return False
        
        return True
    
    def __str__(self):
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type})"
    
    def __repr__(self):
        return self.__str__()