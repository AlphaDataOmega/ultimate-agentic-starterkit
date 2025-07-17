"""
Core data models for the Ultimate Agentic StarterKit.

This module defines Pydantic models for type safety, validation, and consistent
data structures across all system components.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(str, Enum):
    """Status enumeration for project tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration for agent results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentType(str, Enum):
    """Type enumeration for different agent roles."""
    PARSER = "parser"
    CODER = "coder"
    TESTER = "tester"
    ADVISOR = "advisor"
    ORCHESTRATOR = "orchestrator"
    DOCUMENTATION = "documentation"
    TESTING_VALIDATION = "testing_validation"
    VISUAL_TESTING = "visual_testing"
    BUG_BOUNTY = "bug_bounty"


class ProjectTask(BaseModel):
    """
    Model representing a single project task.
    
    Attributes:
        id: Unique identifier for the task
        title: Short descriptive title
        description: Detailed description of the task
        type: Task type (CREATE, MODIFY, TEST, VALIDATE)
        status: Current status of the task
        confidence: Confidence score (0.0 to 1.0)
        agent_type: Type of agent responsible for the task
        dependencies: List of task IDs that must complete first
        created_at: Timestamp when task was created
        updated_at: Timestamp when task was last updated
        attempts: Number of execution attempts
        max_attempts: Maximum allowed attempts
        error_message: Error message if task failed
        output_files: List of files created or modified by the task
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=10000)
    type: str = Field(..., pattern=r"^(CREATE|MODIFY|TEST|VALIDATE|PARSE)$")
    status: TaskStatus = TaskStatus.PENDING
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    agent_type: AgentType
    dependencies: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    error_message: Optional[str] = None
    output_files: List[str] = Field(default_factory=list)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        """Always set updated_at to current time when model is created/updated."""
        return datetime.now()
    
    def mark_in_progress(self) -> None:
        """Mark task as in progress and increment attempts."""
        self.status = TaskStatus.IN_PROGRESS
        self.attempts += 1
        self.updated_at = datetime.now()
    
    def mark_completed(self, confidence: float = 1.0) -> None:
        """Mark task as completed with confidence score."""
        self.status = TaskStatus.COMPLETED
        self.confidence = confidence
        self.updated_at = datetime.now()
        self.error_message = None
    
    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.now()
    
    def can_retry(self) -> bool:
        """Check if task can be retried based on attempts."""
        return self.attempts < self.max_attempts
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


class ProjectSpecification(BaseModel):
    """
    Model representing a complete project specification.
    
    Attributes:
        title: Project title
        description: Project description
        project_type: Type of project (web, blockchain, ai, general)
        tasks: List of tasks to be completed
        requirements: Project requirements dictionary
        validation_criteria: Validation criteria dictionary
        created_at: Timestamp when project was created
    """
    
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=5000)
    project_type: str = Field(..., pattern=r"^(web|blockchain|ai|general)$")
    tasks: List[ProjectTask] = Field(default_factory=list)
    requirements: Dict[str, Any] = Field(default_factory=dict)
    validation_criteria: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_task(self, task: ProjectTask) -> None:
        """Add a task to the project."""
        self.tasks.append(task)
    
    def get_task_by_id(self, task_id: str) -> Optional[ProjectTask]:
        """Get task by ID."""
        return next((task for task in self.tasks if task.id == task_id), None)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[ProjectTask]:
        """Get all tasks with specified status."""
        return [task for task in self.tasks if task.status == status]
    
    def get_ready_tasks(self) -> List[ProjectTask]:
        """Get tasks that are ready to be executed (no pending dependencies)."""
        completed_task_ids = {task.id for task in self.tasks if task.status == TaskStatus.COMPLETED}
        ready_tasks = []
        
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                dependencies_met = all(dep_id in completed_task_ids for dep_id in task.dependencies)
                if dependencies_met:
                    ready_tasks.append(task)
        
        return ready_tasks
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


class AgentResult(BaseModel):
    """
    Model representing the result of an agent's execution.
    
    Attributes:
        success: Whether the agent execution was successful
        confidence: Confidence score of the result
        output: The output data from the agent
        error: Error message if execution failed
        execution_time: Time taken for execution in seconds
        agent_id: Unique identifier for the agent
        timestamp: When the result was generated
    """
    
    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    output: Any
    error: Optional[str] = None
    execution_time: float = Field(ge=0.0)
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level enum based on confidence score."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


class WorkflowState(BaseModel):
    """
    Model representing the current state of a workflow execution.
    
    Attributes:
        project_spec: The project specification being executed
        current_task: The currently executing task
        completed_tasks: List of completed task IDs
        failed_tasks: List of failed task IDs
        agent_results: List of agent execution results
        overall_confidence: Overall confidence score for the workflow
        workflow_status: Current status of the workflow
    """
    
    project_spec: ProjectSpecification
    current_task: Optional[ProjectTask] = None
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    agent_results: List[AgentResult] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    workflow_status: TaskStatus = TaskStatus.PENDING
    
    def add_result(self, result: AgentResult) -> None:
        """Add an agent result to the workflow."""
        self.agent_results.append(result)
        self._update_overall_confidence()
    
    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        
        task = self.project_spec.get_task_by_id(task_id)
        if task:
            task.mark_completed()
        
        self._check_workflow_completion()
    
    def fail_task(self, task_id: str, error_message: str) -> None:
        """Mark a task as failed."""
        if task_id not in self.failed_tasks:
            self.failed_tasks.append(task_id)
        
        task = self.project_spec.get_task_by_id(task_id)
        if task:
            task.mark_failed(error_message)
    
    def get_next_task(self) -> Optional[ProjectTask]:
        """Get the next task that's ready to be executed."""
        ready_tasks = self.project_spec.get_ready_tasks()
        return ready_tasks[0] if ready_tasks else None
    
    def _update_overall_confidence(self) -> None:
        """Update overall confidence based on recent results."""
        if not self.agent_results:
            self.overall_confidence = 0.0
            return
        
        # Calculate weighted average of recent results
        recent_results = self.agent_results[-10:]  # Last 10 results
        if recent_results:
            total_confidence = sum(result.confidence for result in recent_results)
            self.overall_confidence = total_confidence / len(recent_results)
    
    def _check_workflow_completion(self) -> None:
        """Check if workflow is complete and update status."""
        all_tasks = self.project_spec.tasks
        if not all_tasks:
            return
        
        completed_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        total_count = len(all_tasks)
        
        if completed_count == total_count:
            self.workflow_status = TaskStatus.COMPLETED
        elif failed_count > 0 and (completed_count + failed_count) == total_count:
            self.workflow_status = TaskStatus.FAILED
        elif completed_count + failed_count < total_count:
            self.workflow_status = TaskStatus.IN_PROGRESS
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


# Factory functions for common use cases
def create_project_task(
    title: str,
    description: str,
    task_type: str,
    agent_type: AgentType,
    dependencies: Optional[List[str]] = None
) -> ProjectTask:
    """
    Factory function to create a project task with common defaults.
    
    Args:
        title: Task title
        description: Task description
        task_type: Task type (CREATE, MODIFY, TEST, VALIDATE)
        agent_type: Agent type responsible for the task
        dependencies: List of dependency task IDs
    
    Returns:
        ProjectTask: New project task instance
    """
    return ProjectTask(
        title=title,
        description=description,
        type=task_type,
        agent_type=agent_type,
        dependencies=dependencies or []
    )


def create_project_specification(
    title: str,
    description: str,
    project_type: str,
    tasks: Optional[List[ProjectTask]] = None
) -> ProjectSpecification:
    """
    Factory function to create a project specification.
    
    Args:
        title: Project title
        description: Project description
        project_type: Project type (web, blockchain, ai, general)
        tasks: List of project tasks
    
    Returns:
        ProjectSpecification: New project specification instance
    """
    return ProjectSpecification(
        title=title,
        description=description,
        project_type=project_type,
        tasks=tasks or []
    )


def create_agent_result(
    success: bool,
    confidence: float,
    output: Any,
    agent_id: str,
    execution_time: float,
    error: Optional[str] = None
) -> AgentResult:
    """
    Factory function to create an agent result.
    
    Args:
        success: Whether execution was successful
        confidence: Confidence score (0.0 to 1.0)
        output: Output data from the agent
        agent_id: Unique identifier for the agent
        execution_time: Execution time in seconds
        error: Error message if execution failed
    
    Returns:
        AgentResult: New agent result instance
    """
    return AgentResult(
        success=success,
        confidence=confidence,
        output=output,
        agent_id=agent_id,
        execution_time=execution_time,
        error=error
    )