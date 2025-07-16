"""
Task Execution Engine for the Ultimate Agentic StarterKit.

This module provides sophisticated task execution with agent coordination,
parallel execution, retry logic, and comprehensive progress tracking.
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import uuid
import json

from core.models import ProjectTask, TaskStatus, AgentResult, AgentType
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts
from agents.factory import get_or_create_agent, AgentInstance


class ExecutionMode(str, Enum):
    """Task execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskExecutionResult:
    """Result of task execution with detailed information."""
    task_id: str
    agent_id: str
    agent_type: AgentType
    success: bool
    confidence: float
    execution_time: float
    start_time: datetime
    end_time: datetime
    output: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'success': self.success,
            'confidence': self.confidence,
            'execution_time': self.execution_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'output': self.output,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'metadata': self.metadata
        }


@dataclass
class ParallelGroup:
    """Group of tasks that can execute in parallel."""
    group_id: str
    tasks: List[ProjectTask]
    priority: TaskPriority = TaskPriority.MEDIUM
    max_concurrent: int = 3
    timeout: float = 600.0
    
    def __post_init__(self):
        if not self.group_id:
            self.group_id = str(uuid.uuid4())


class TaskExecutionEngine:
    """
    Advanced task execution engine with parallel processing and retry logic.
    
    This class manages the execution of individual tasks and groups of tasks
    with sophisticated coordination, monitoring, and error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the task execution engine.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.logger = get_logger("task_executor")
        self.voice = get_voice_alerts()
        
        # Execution configuration
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)
        self.default_task_timeout = self.config.get('default_task_timeout', 600)
        self.retry_delay_base = self.config.get('retry_delay_base', 2)
        self.retry_delay_max = self.config.get('retry_delay_max', 30)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_tasks,
            thread_name_prefix="task_executor"
        )
        
        # State tracking
        self.running_tasks: Dict[str, Future] = {}
        self.completed_tasks: Dict[str, TaskExecutionResult] = {}
        self.failed_tasks: Dict[str, TaskExecutionResult] = {}
        self.task_history: List[TaskExecutionResult] = []
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        # Coordination
        self.execution_lock = threading.Lock()
        self.is_shutdown = False
        
        # Performance metrics
        self.metrics = {
            'total_tasks_executed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'parallel_efficiency': 0.0
        }
        
        self.logger.info("Task Execution Engine initialized")
    
    async def execute_task(self, task: ProjectTask, 
                          agent_instance: Optional[AgentInstance] = None,
                          retry_count: int = 0) -> TaskExecutionResult:
        """
        Execute a single task with comprehensive error handling.
        
        Args:
            task: The task to execute
            agent_instance: Optional specific agent instance to use
            retry_count: Current retry attempt
            
        Returns:
            TaskExecutionResult with execution details
        """
        start_time = datetime.now()
        task_id = task.id
        
        # Get agent if not provided
        if agent_instance is None:
            agent_instance = get_or_create_agent(task.agent_type)
        
        agent_id = agent_instance.agent.agent_id
        
        self.logger.info(f"Executing task {task_id} with agent {agent_id} (attempt {retry_count + 1})")
        
        # Voice notification
        self.voice.speak_agent_start(agent_id, f"starting task {task.title}")
        
        try:
            # Execute task with timeout
            result = await asyncio.wait_for(
                agent_instance.execute_task(task),
                timeout=self.default_task_timeout
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Create execution result
            execution_result = TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                agent_type=task.agent_type,
                success=result.success,
                confidence=result.confidence,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                output=result.output,
                error=result.error,
                retry_count=retry_count,
                max_retries=task.max_attempts,
                metadata={
                    'task_title': task.title,
                    'task_type': task.type,
                    'agent_execution_time': result.execution_time
                }
            )
            
            # Update metrics
            self._update_metrics(execution_result)
            
            # Track result
            if result.success and result.confidence >= self.confidence_threshold:
                self.completed_tasks[task_id] = execution_result
                self.voice.speak_agent_complete(agent_id, f"completed task {task.title}")
                self.logger.info(f"Task {task_id} completed successfully (confidence: {result.confidence:.2f})")
            else:
                # Check if retry is needed
                if retry_count < task.max_attempts - 1:
                    self.logger.warning(f"Task {task_id} needs retry (confidence: {result.confidence:.2f})")
                    return await self._retry_task(task, agent_instance, retry_count + 1)
                else:
                    self.failed_tasks[task_id] = execution_result
                    self.voice.speak_error(f"Task {task.title} failed after {retry_count + 1} attempts")
                    self.logger.error(f"Task {task_id} failed after {retry_count + 1} attempts")
            
            # Add to history
            self.task_history.append(execution_result)
            
            # Notify progress callbacks
            await self._notify_progress_callbacks(execution_result)
            
            return execution_result
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            error_msg = f"Task {task_id} timed out after {self.default_task_timeout} seconds"
            self.logger.error(error_msg)
            
            execution_result = TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                agent_type=task.agent_type,
                success=False,
                confidence=0.0,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error=error_msg,
                retry_count=retry_count,
                max_retries=task.max_attempts,
                metadata={'timeout': True}
            )
            
            # Check if retry is needed
            if retry_count < task.max_attempts - 1:
                return await self._retry_task(task, agent_instance, retry_count + 1)
            else:
                self.failed_tasks[task_id] = execution_result
                self.voice.speak_error(f"Task {task.title} timed out")
            
            return execution_result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            error_msg = f"Task {task_id} execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            execution_result = TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                agent_type=task.agent_type,
                success=False,
                confidence=0.0,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error=error_msg,
                retry_count=retry_count,
                max_retries=task.max_attempts,
                metadata={'exception': str(e)}
            )
            
            # Check if retry is needed
            if retry_count < task.max_attempts - 1:
                return await self._retry_task(task, agent_instance, retry_count + 1)
            else:
                self.failed_tasks[task_id] = execution_result
                self.voice.speak_error(f"Task {task.title} failed with error")
            
            return execution_result
    
    async def _retry_task(self, task: ProjectTask, 
                         agent_instance: AgentInstance,
                         retry_count: int) -> TaskExecutionResult:
        """
        Retry a failed task with exponential backoff.
        
        Args:
            task: The task to retry
            agent_instance: Agent instance to use
            retry_count: Current retry count
            
        Returns:
            TaskExecutionResult from retry attempt
        """
        # Calculate delay with exponential backoff
        delay = min(
            self.retry_delay_base ** retry_count,
            self.retry_delay_max
        )
        
        self.logger.info(f"Retrying task {task.id} in {delay} seconds (attempt {retry_count + 1})")
        
        # Wait before retry
        await asyncio.sleep(delay)
        
        # Reset agent instance if needed
        if hasattr(agent_instance, 'reset'):
            agent_instance.reset()
        
        # Retry the task
        return await self.execute_task(task, agent_instance, retry_count)
    
    async def execute_parallel_group(self, parallel_group: ParallelGroup) -> List[TaskExecutionResult]:
        """
        Execute a group of tasks in parallel.
        
        Args:
            parallel_group: Group of tasks to execute in parallel
            
        Returns:
            List of TaskExecutionResult objects
        """
        group_id = parallel_group.group_id
        tasks = parallel_group.tasks
        max_concurrent = min(parallel_group.max_concurrent, len(tasks))
        
        self.logger.info(f"Executing parallel group {group_id} with {len(tasks)} tasks (max concurrent: {max_concurrent})")
        
        # Voice notification
        self.voice.speak_milestone(f"Starting parallel execution of {len(tasks)} tasks")
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(task: ProjectTask):
            async with semaphore:
                return await self.execute_task(task)
        
        # Create tasks for parallel execution
        parallel_tasks = [
            asyncio.create_task(execute_with_semaphore(task))
            for task in tasks
        ]
        
        # Wait for all tasks to complete
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*parallel_tasks, return_exceptions=True),
                timeout=parallel_group.timeout
            )
            
            # Process results
            execution_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle exceptions
                    task = tasks[i]
                    error_result = TaskExecutionResult(
                        task_id=task.id,
                        agent_id="unknown",
                        agent_type=task.agent_type,
                        success=False,
                        confidence=0.0,
                        execution_time=0.0,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error=str(result),
                        retry_count=0,
                        max_retries=task.max_attempts,
                        metadata={'parallel_group': group_id, 'exception': True}
                    )
                    execution_results.append(error_result)
                else:
                    result.metadata['parallel_group'] = group_id
                    execution_results.append(result)
            
            # Voice notification
            successful_count = sum(1 for r in execution_results if r.success)
            self.voice.speak_milestone(f"Parallel group completed: {successful_count}/{len(tasks)} tasks successful")
            
            self.logger.info(f"Parallel group {group_id} completed: {successful_count}/{len(tasks)} successful")
            
            return execution_results
            
        except asyncio.TimeoutError:
            self.logger.error(f"Parallel group {group_id} timed out after {parallel_group.timeout} seconds")
            
            # Cancel remaining tasks
            for task in parallel_tasks:
                if not task.done():
                    task.cancel()
            
            # Create timeout results for unfinished tasks
            timeout_results = []
            for task in tasks:
                if task.id not in self.completed_tasks and task.id not in self.failed_tasks:
                    timeout_result = TaskExecutionResult(
                        task_id=task.id,
                        agent_id="unknown",
                        agent_type=task.agent_type,
                        success=False,
                        confidence=0.0,
                        execution_time=parallel_group.timeout,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error=f"Parallel group timeout after {parallel_group.timeout} seconds",
                        retry_count=0,
                        max_retries=task.max_attempts,
                        metadata={'parallel_group': group_id, 'timeout': True}
                    )
                    timeout_results.append(timeout_result)
            
            return timeout_results
    
    async def execute_tasks_with_dependencies(self, tasks: List[ProjectTask]) -> List[TaskExecutionResult]:
        """
        Execute tasks respecting their dependencies.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            List of TaskExecutionResult objects
        """
        self.logger.info(f"Executing {len(tasks)} tasks with dependency management")
        
        # Build dependency graph
        task_map = {task.id: task for task in tasks}
        completed_task_ids = set()
        results = []
        
        # Track remaining tasks
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks that can be executed (dependencies met)
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep_id in completed_task_ids for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # No tasks ready - check for circular dependencies
                remaining_ids = [task.id for task in remaining_tasks]
                self.logger.error(f"No tasks ready for execution. Remaining: {remaining_ids}")
                
                # Create failure results for remaining tasks
                for task in remaining_tasks:
                    failure_result = TaskExecutionResult(
                        task_id=task.id,
                        agent_id="unknown",
                        agent_type=task.agent_type,
                        success=False,
                        confidence=0.0,
                        execution_time=0.0,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error="Circular dependency or unmet dependencies",
                        retry_count=0,
                        max_retries=task.max_attempts,
                        metadata={'dependency_issue': True}
                    )
                    results.append(failure_result)
                
                break
            
            # Execute ready tasks
            if len(ready_tasks) == 1:
                # Single task - execute directly
                result = await self.execute_task(ready_tasks[0])
                results.append(result)
                
                if result.success:
                    completed_task_ids.add(result.task_id)
                
                remaining_tasks.remove(ready_tasks[0])
            else:
                # Multiple tasks - execute in parallel
                parallel_group = ParallelGroup(
                    group_id=str(uuid.uuid4()),
                    tasks=ready_tasks,
                    max_concurrent=min(self.max_concurrent_tasks, len(ready_tasks))
                )
                
                parallel_results = await self.execute_parallel_group(parallel_group)
                results.extend(parallel_results)
                
                # Update completed tasks
                for result in parallel_results:
                    if result.success:
                        completed_task_ids.add(result.task_id)
                
                # Remove completed tasks
                for task in ready_tasks:
                    if task in remaining_tasks:
                        remaining_tasks.remove(task)
        
        self.logger.info(f"Dependency-based execution completed: {len(results)} tasks processed")
        return results
    
    def _update_metrics(self, result: TaskExecutionResult):
        """Update performance metrics."""
        with self.execution_lock:
            self.metrics['total_tasks_executed'] += 1
            self.metrics['total_execution_time'] += result.execution_time
            
            if result.success:
                self.metrics['successful_tasks'] += 1
            else:
                self.metrics['failed_tasks'] += 1
            
            # Update averages
            if self.metrics['total_tasks_executed'] > 0:
                self.metrics['average_execution_time'] = (
                    self.metrics['total_execution_time'] / self.metrics['total_tasks_executed']
                )
            
            # Calculate parallel efficiency (placeholder)
            if self.metrics['total_tasks_executed'] > 1:
                self.metrics['parallel_efficiency'] = min(
                    1.0,
                    self.metrics['successful_tasks'] / self.metrics['total_tasks_executed']
                )
    
    async def _notify_progress_callbacks(self, result: TaskExecutionResult):
        """Notify progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    def add_progress_callback(self, callback: Callable):
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """Add a completion callback function."""
        self.completion_callbacks.append(callback)
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        with self.execution_lock:
            return {
                **self.metrics,
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'total_tasks_in_history': len(self.task_history)
            }
    
    def get_task_results(self) -> Dict[str, Any]:
        """Get task execution results."""
        return {
            'completed': {k: v.to_dict() for k, v in self.completed_tasks.items()},
            'failed': {k: v.to_dict() for k, v in self.failed_tasks.items()},
            'running': list(self.running_tasks.keys())
        }
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get task execution history."""
        return [result.to_dict() for result in self.task_history]
    
    def clear_history(self):
        """Clear task execution history."""
        self.task_history.clear()
        self.logger.info("Task execution history cleared")
    
    def reset_metrics(self):
        """Reset execution metrics."""
        with self.execution_lock:
            self.metrics = {
                'total_tasks_executed': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'total_execution_time': 0.0,
                'average_execution_time': 0.0,
                'parallel_efficiency': 0.0
            }
        self.logger.info("Execution metrics reset")
    
    async def shutdown(self):
        """Shutdown the task execution engine."""
        self.logger.info("Shutting down task execution engine")
        
        self.is_shutdown = True
        
        # Cancel running tasks
        for task_id, future in self.running_tasks.items():
            if not future.done():
                future.cancel()
                self.logger.info(f"Cancelled running task: {task_id}")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Clear state
        self.running_tasks.clear()
        
        self.logger.info("Task execution engine shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.shutdown())


# Convenience functions for common use cases
async def execute_single_task(task: ProjectTask, 
                            config: Optional[Dict[str, Any]] = None) -> TaskExecutionResult:
    """
    Execute a single task with default configuration.
    
    Args:
        task: Task to execute
        config: Optional configuration
        
    Returns:
        TaskExecutionResult
    """
    async with TaskExecutionEngine(config) as executor:
        return await executor.execute_task(task)


async def execute_task_batch(tasks: List[ProjectTask],
                           parallel: bool = True,
                           config: Optional[Dict[str, Any]] = None) -> List[TaskExecutionResult]:
    """
    Execute a batch of tasks.
    
    Args:
        tasks: List of tasks to execute
        parallel: Whether to execute in parallel
        config: Optional configuration
        
    Returns:
        List of TaskExecutionResult objects
    """
    async with TaskExecutionEngine(config) as executor:
        if parallel:
            parallel_group = ParallelGroup(
                group_id=str(uuid.uuid4()),
                tasks=tasks
            )
            return await executor.execute_parallel_group(parallel_group)
        else:
            return await executor.execute_tasks_with_dependencies(tasks)


def create_parallel_group(tasks: List[ProjectTask],
                         group_id: Optional[str] = None,
                         max_concurrent: int = 3,
                         timeout: float = 600.0) -> ParallelGroup:
    """
    Create a parallel execution group.
    
    Args:
        tasks: Tasks to include in the group
        group_id: Optional group identifier
        max_concurrent: Maximum concurrent tasks
        timeout: Group timeout in seconds
        
    Returns:
        ParallelGroup instance
    """
    return ParallelGroup(
        group_id=group_id or str(uuid.uuid4()),
        tasks=tasks,
        max_concurrent=max_concurrent,
        timeout=timeout
    )