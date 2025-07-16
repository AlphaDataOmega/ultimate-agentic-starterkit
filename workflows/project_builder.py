"""
LangGraph Workflow Manager for the Ultimate Agentic StarterKit.

This module implements the workflow execution system using LangGraph for state management
and workflow coordination with sophisticated error handling and recovery.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

# LangGraph imports
try:
    from langgraph import StateGraph, END
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for development without LangGraph
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    TypedDict = dict

from core.models import ProjectSpecification, ProjectTask, TaskStatus, AgentResult, AgentType
from core.orchestrator import O3Orchestrator
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts
from agents.factory import get_or_create_agent, AgentInstance


class WorkflowError(Exception):
    """Base exception for workflow errors."""
    pass


class WorkflowTimeoutError(WorkflowError):
    """Exception raised when workflow times out."""
    pass


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowMetrics:
    """Workflow execution metrics."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    retried_tasks: int = 0
    total_execution_time: float = 0.0
    average_confidence: float = 0.0
    
    @property
    def duration(self) -> float:
        """Get total workflow duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Get task success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'retried_tasks': self.retried_tasks,
            'total_execution_time': self.total_execution_time,
            'average_confidence': self.average_confidence,
            'success_rate': self.success_rate
        }


if LANGGRAPH_AVAILABLE:
    class ProjectBuilderState(TypedDict):
        """State schema for project builder workflow."""
        project_spec: Dict[str, Any]
        execution_plan: Dict[str, Any]
        current_task: Optional[Dict[str, Any]]
        completed_tasks: List[str]
        failed_tasks: List[str]
        agent_results: List[Dict[str, Any]]
        overall_confidence: float
        workflow_status: str
        retry_count: int
        max_retries: int
        error_message: Optional[str]
        metrics: Dict[str, Any]
        checkpoint_data: Dict[str, Any]
else:
    # Fallback for development without LangGraph
    ProjectBuilderState = dict


class LangGraphWorkflowManager:
    """
    LangGraph-based workflow execution manager.
    
    This class manages the complete workflow execution lifecycle using LangGraph
    for state management and coordination between different agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow manager.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.logger = get_logger("workflow_manager")
        self.voice = get_voice_alerts()
        
        # Initialize components
        self.orchestrator = O3Orchestrator()
        
        # Workflow configuration
        self.max_retries = self.config.get('max_retries', 3)
        self.task_timeout = self.config.get('task_timeout', 600)  # 10 minutes
        self.workflow_timeout = self.config.get('workflow_timeout', 7200)  # 2 hours
        
        # State management
        self.current_state: Optional[ProjectBuilderState] = None
        self.state_history: List[ProjectBuilderState] = []
        self.max_history_size = 100
        
        # Build workflow graph
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_workflow_graph()
        else:
            self.graph = None
            self.logger.warning("LangGraph not available, using fallback implementation")
        
        # Workflow control
        self.is_running = False
        self.should_stop = False
        self.pause_requested = False
        
        self.logger.info("LangGraph Workflow Manager initialized")
    
    def _build_workflow_graph(self) -> Optional[Any]:
        """Build LangGraph workflow graph."""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        try:
            graph = StateGraph(ProjectBuilderState)
            
            # Define workflow nodes
            graph.add_node("initialize_workflow", self._initialize_workflow)
            graph.add_node("select_next_task", self._select_next_task)
            graph.add_node("execute_task", self._execute_task)
            graph.add_node("validate_result", self._validate_result)
            graph.add_node("update_state", self._update_state)
            graph.add_node("check_completion", self._check_completion)
            graph.add_node("handle_failure", self._handle_failure)
            graph.add_node("adapt_plan", self._adapt_plan)
            graph.add_node("create_checkpoint", self._create_checkpoint)
            graph.add_node("finalize_workflow", self._finalize_workflow)
            
            # Define workflow edges
            graph.add_edge("initialize_workflow", "select_next_task")
            graph.add_edge("select_next_task", "execute_task")
            graph.add_edge("execute_task", "validate_result")
            
            # Conditional edges based on validation results
            graph.add_conditional_edges(
                "validate_result",
                self._should_retry,
                {
                    "retry": "execute_task",
                    "success": "update_state",
                    "failure": "handle_failure",
                    "adapt": "adapt_plan"
                }
            )
            
            graph.add_edge("update_state", "create_checkpoint")
            graph.add_edge("create_checkpoint", "check_completion")
            
            # Conditional edges for completion check
            graph.add_conditional_edges(
                "check_completion",
                self._is_workflow_complete,
                {
                    "continue": "select_next_task",
                    "complete": "finalize_workflow",
                    "failed": "handle_failure"
                }
            )
            
            graph.add_edge("handle_failure", "finalize_workflow")
            graph.add_edge("adapt_plan", "select_next_task")
            graph.add_edge("finalize_workflow", END)
            
            # Set entry point
            graph.set_entry_point("initialize_workflow")
            
            return graph.compile()
            
        except Exception as e:
            self.logger.error(f"Failed to build workflow graph: {e}")
            return None
    
    async def _initialize_workflow(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Initialize workflow state and create execution plan."""
        self.logger.info("Initializing workflow execution")
        
        try:
            # Initialize metrics
            metrics = WorkflowMetrics()
            metrics.total_tasks = len(state.get("project_spec", {}).get("tasks", []))
            
            # Create project specification object
            project_spec = ProjectSpecification(**state["project_spec"])
            
            # Create execution plan using orchestrator
            execution_plan = await self.orchestrator.create_execution_plan(project_spec)
            
            # Update state
            state.update({
                "execution_plan": execution_plan,
                "workflow_status": WorkflowStatus.RUNNING.value,
                "retry_count": 0,
                "max_retries": self.max_retries,
                "error_message": None,
                "metrics": metrics.to_dict(),
                "checkpoint_data": {
                    "last_checkpoint": datetime.now().isoformat(),
                    "checkpoint_count": 0
                }
            })
            
            self.voice.speak_success("Workflow initialized successfully")
            self.logger.info("Workflow initialization completed")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Workflow initialization failed: {e}")
            state["workflow_status"] = WorkflowStatus.FAILED.value
            state["error_message"] = str(e)
            return state
    
    async def _select_next_task(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Select next task for execution based on dependencies."""
        execution_plan = state.get("execution_plan", {})
        completed_tasks = set(state.get("completed_tasks", []))
        failed_tasks = set(state.get("failed_tasks", []))
        
        # Check if workflow should be paused
        if self.pause_requested:
            state["workflow_status"] = WorkflowStatus.PAUSED.value
            self.logger.info("Workflow paused by request")
            return state
        
        # Check for stop signal
        if self.should_stop:
            state["workflow_status"] = WorkflowStatus.CANCELLED.value
            self.logger.info("Workflow cancelled by request")
            return state
        
        # Find next available task
        execution_order = execution_plan.get("execution_order", [])
        
        for task_order in execution_order:
            task_id = task_order["task_id"]
            
            # Skip completed or failed tasks
            if task_id in completed_tasks or task_id in failed_tasks:
                continue
            
            # Check dependencies
            dependencies = task_order.get("dependencies", [])
            if all(dep in completed_tasks for dep in dependencies):
                # Find task details from project spec
                project_tasks = state["project_spec"]["tasks"]
                task_details = next((t for t in project_tasks if t["id"] == task_id), None)
                
                if task_details:
                    # Merge task details with execution info
                    current_task = {
                        **task_details,
                        **task_order
                    }
                    
                    state["current_task"] = current_task
                    self.logger.info(f"Selected task for execution: {task_id}")
                    return state
        
        # No available tasks
        state["current_task"] = None
        self.logger.info("No available tasks for execution")
        return state
    
    async def _execute_task(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Execute current task with appropriate agent."""
        current_task = state.get("current_task")
        if not current_task:
            return state
        
        task_id = current_task["id"]
        agent_type = current_task["agent_type"]
        
        self.logger.info(f"Executing task {task_id} with agent type {agent_type}")
        
        try:
            # Get agent instance
            agent_instance = get_or_create_agent(AgentType(agent_type))
            
            # Create task object
            task_obj = ProjectTask(
                id=current_task["id"],
                title=current_task["title"],
                description=current_task["description"],
                type=current_task["type"],
                agent_type=AgentType(agent_type),
                dependencies=current_task.get("dependencies", [])
            )
            
            # Execute task with timeout
            start_time = time.time()
            
            result = await asyncio.wait_for(
                agent_instance.execute_task(task_obj),
                timeout=self.task_timeout
            )
            
            execution_time = time.time() - start_time
            
            # Store result in state
            result_data = {
                "task_id": task_id,
                "agent_type": agent_type,
                "success": result.success,
                "confidence": result.confidence,
                "error": result.error,
                "execution_time": execution_time,
                "timestamp": result.timestamp.isoformat(),
                "output": result.output
            }
            
            state.setdefault("agent_results", []).append(result_data)
            
            self.logger.info(f"Task {task_id} executed with success={result.success}, confidence={result.confidence}")
            
            return state
            
        except asyncio.TimeoutError:
            error_msg = f"Task {task_id} timed out after {self.task_timeout} seconds"
            self.logger.error(error_msg)
            
            # Add timeout result
            result_data = {
                "task_id": task_id,
                "agent_type": agent_type,
                "success": False,
                "confidence": 0.0,
                "error": error_msg,
                "execution_time": self.task_timeout,
                "timestamp": datetime.now().isoformat(),
                "output": None
            }
            
            state.setdefault("agent_results", []).append(result_data)
            return state
            
        except Exception as e:
            error_msg = f"Task {task_id} execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Add error result
            result_data = {
                "task_id": task_id,
                "agent_type": agent_type,
                "success": False,
                "confidence": 0.0,
                "error": error_msg,
                "execution_time": 0.0,
                "timestamp": datetime.now().isoformat(),
                "output": None
            }
            
            state.setdefault("agent_results", []).append(result_data)
            return state
    
    async def _validate_result(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Validate task execution result."""
        agent_results = state.get("agent_results", [])
        if not agent_results:
            return state
        
        latest_result = agent_results[-1]
        current_task = state.get("current_task", {})
        
        # Get confidence threshold
        confidence_threshold = current_task.get("confidence_threshold", 0.8)
        
        # Determine validation status
        if latest_result["success"] and latest_result["confidence"] >= confidence_threshold:
            self.logger.info(f"Task {current_task['id']} validation passed")
            state["validation_status"] = "success"
            state["retry_count"] = 0  # Reset retry count on success
            
        elif state.get("retry_count", 0) < state.get("max_retries", 3):
            self.logger.warning(f"Task {current_task['id']} validation failed, retrying")
            state["validation_status"] = "retry"
            state["retry_count"] = state.get("retry_count", 0) + 1
            
        elif latest_result["confidence"] < 0.3:  # Very low confidence, try adaptation
            self.logger.info(f"Task {current_task['id']} has very low confidence, attempting adaptation")
            state["validation_status"] = "adapt"
            
        else:
            self.logger.error(f"Task {current_task['id']} validation failed after max retries")
            state["validation_status"] = "failure"
        
        return state
    
    def _should_retry(self, state: ProjectBuilderState) -> str:
        """Determine if task should be retried based on validation."""
        return state.get("validation_status", "failure")
    
    async def _update_state(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Update workflow state after successful task completion."""
        current_task = state.get("current_task")
        if not current_task:
            return state
        
        task_id = current_task["id"]
        
        # Add to completed tasks
        completed_tasks = state.get("completed_tasks", [])
        if task_id not in completed_tasks:
            completed_tasks.append(task_id)
            state["completed_tasks"] = completed_tasks
        
        # Update overall confidence
        agent_results = state.get("agent_results", [])
        successful_results = [r for r in agent_results if r["success"]]
        if successful_results:
            total_confidence = sum(r["confidence"] for r in successful_results)
            state["overall_confidence"] = total_confidence / len(successful_results)
        
        # Update metrics
        metrics_data = state.get("metrics", {})
        metrics_data["completed_tasks"] = len(completed_tasks)
        metrics_data["average_confidence"] = state.get("overall_confidence", 0.0)
        state["metrics"] = metrics_data
        
        # Voice notification
        self.voice.speak_success(f"Task {current_task['title']} completed successfully")
        
        self.logger.info(f"Task {task_id} completed successfully, {len(completed_tasks)} total completed")
        
        return state
    
    async def _handle_failure(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Handle task or workflow failure."""
        current_task = state.get("current_task")
        if current_task:
            task_id = current_task["id"]
            
            # Add to failed tasks
            failed_tasks = state.get("failed_tasks", [])
            if task_id not in failed_tasks:
                failed_tasks.append(task_id)
                state["failed_tasks"] = failed_tasks
            
            # Update metrics
            metrics_data = state.get("metrics", {})
            metrics_data["failed_tasks"] = len(failed_tasks)
            state["metrics"] = metrics_data
            
            # Voice notification
            self.voice.speak_error(f"Task {current_task['title']} failed")
            
            self.logger.error(f"Task {task_id} failed permanently")
        
        # Check if workflow should continue or fail
        total_tasks = len(state.get("project_spec", {}).get("tasks", []))
        failed_count = len(state.get("failed_tasks", []))
        completed_count = len(state.get("completed_tasks", []))
        
        # If more than 50% of tasks failed, fail the workflow
        if failed_count > total_tasks * 0.5:
            state["workflow_status"] = WorkflowStatus.FAILED.value
            state["error_message"] = f"Too many failed tasks: {failed_count}/{total_tasks}"
        
        return state
    
    async def _adapt_plan(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Adapt execution plan based on current results."""
        self.logger.info("Adapting execution plan based on current results")
        
        try:
            current_plan = state.get("execution_plan", {})
            failed_tasks = state.get("failed_tasks", [])
            agent_results = state.get("agent_results", [])
            
            # Use orchestrator to adapt the plan
            adapted_plan = await self.orchestrator.adapt_plan_during_execution(
                current_plan, failed_tasks, agent_results
            )
            
            state["execution_plan"] = adapted_plan
            
            # Voice notification
            self.voice.speak_milestone("Execution plan adapted")
            
            self.logger.info("Execution plan adapted successfully")
            
        except Exception as e:
            self.logger.error(f"Plan adaptation failed: {e}")
            # Continue with original plan
        
        return state
    
    async def _create_checkpoint(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Create checkpoint for workflow recovery."""
        checkpoint_data = state.get("checkpoint_data", {})
        checkpoint_data.update({
            "last_checkpoint": datetime.now().isoformat(),
            "checkpoint_count": checkpoint_data.get("checkpoint_count", 0) + 1,
            "completed_tasks": state.get("completed_tasks", []).copy(),
            "failed_tasks": state.get("failed_tasks", []).copy(),
            "overall_confidence": state.get("overall_confidence", 0.0)
        })
        
        state["checkpoint_data"] = checkpoint_data
        
        # Save state to history
        self._save_state_to_history(state)
        
        self.logger.debug(f"Checkpoint created: {checkpoint_data['checkpoint_count']}")
        
        return state
    
    def _save_state_to_history(self, state: ProjectBuilderState):
        """Save state to history with size limit."""
        # Create deep copy of state
        state_copy = json.loads(json.dumps(state, default=str))
        
        self.state_history.append(state_copy)
        
        # Maintain history size limit
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)
    
    def _is_workflow_complete(self, state: ProjectBuilderState) -> str:
        """Check if workflow is complete."""
        project_spec = state.get("project_spec", {})
        total_tasks = len(project_spec.get("tasks", []))
        completed_tasks = len(state.get("completed_tasks", []))
        failed_tasks = len(state.get("failed_tasks", []))
        
        if completed_tasks == total_tasks:
            return "complete"
        elif failed_tasks > 0 and (completed_tasks + failed_tasks) >= total_tasks:
            return "failed"
        else:
            return "continue"
    
    async def _finalize_workflow(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Finalize workflow execution."""
        # Update final status
        if state.get("workflow_status") != WorkflowStatus.FAILED.value:
            state["workflow_status"] = WorkflowStatus.COMPLETED.value
        
        # Update final metrics
        metrics_data = state.get("metrics", {})
        metrics_data["end_time"] = datetime.now().isoformat()
        
        # Calculate final statistics
        agent_results = state.get("agent_results", [])
        if agent_results:
            total_execution_time = sum(r.get("execution_time", 0) for r in agent_results)
            metrics_data["total_execution_time"] = total_execution_time
        
        state["metrics"] = metrics_data
        
        # Voice notification
        if state["workflow_status"] == WorkflowStatus.COMPLETED.value:
            self.voice.speak_milestone("Workflow completed successfully")
        else:
            self.voice.speak_error("Workflow failed")
        
        self.logger.info(f"Workflow finalized with status: {state['workflow_status']}")
        
        return state
    
    async def execute_workflow(self, project_spec: Dict[str, Any],
                             resume_from_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute complete workflow.
        
        Args:
            project_spec: Project specification dictionary
            resume_from_state: Optional state to resume from
            
        Returns:
            Final workflow state
        """
        self.logger.info("Starting workflow execution")
        self.is_running = True
        
        try:
            # Initialize or resume state
            if resume_from_state:
                initial_state = resume_from_state
                self.logger.info("Resuming workflow from saved state")
            else:
                initial_state = {
                    "project_spec": project_spec,
                    "execution_plan": {},
                    "current_task": None,
                    "completed_tasks": [],
                    "failed_tasks": [],
                    "agent_results": [],
                    "overall_confidence": 0.0,
                    "workflow_status": WorkflowStatus.INITIALIZING.value,
                    "retry_count": 0,
                    "max_retries": self.max_retries,
                    "error_message": None,
                    "metrics": {},
                    "checkpoint_data": {}
                }
            
            self.current_state = initial_state
            
            # Execute workflow
            if self.graph and LANGGRAPH_AVAILABLE:
                # Use LangGraph for execution
                final_state = await asyncio.wait_for(
                    self.graph.ainvoke(initial_state),
                    timeout=self.workflow_timeout
                )
            else:
                # Use fallback implementation
                final_state = await self._execute_workflow_fallback(initial_state)
            
            self.current_state = final_state
            
            return final_state
            
        except asyncio.TimeoutError:
            self.logger.error(f"Workflow timed out after {self.workflow_timeout} seconds")
            raise WorkflowTimeoutError(f"Workflow timed out after {self.workflow_timeout} seconds")
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise WorkflowError(f"Workflow execution failed: {e}")
            
        finally:
            self.is_running = False
    
    async def _execute_workflow_fallback(self, state: ProjectBuilderState) -> ProjectBuilderState:
        """Fallback workflow execution without LangGraph."""
        self.logger.info("Using fallback workflow execution")
        
        # Simple sequential execution
        state = await self._initialize_workflow(state)
        
        while True:
            if state.get("workflow_status") in [WorkflowStatus.FAILED.value, WorkflowStatus.COMPLETED.value]:
                break
            
            state = await self._select_next_task(state)
            
            if not state.get("current_task"):
                break
            
            state = await self._execute_task(state)
            state = await self._validate_result(state)
            
            if state.get("validation_status") == "success":
                state = await self._update_state(state)
                state = await self._create_checkpoint(state)
            elif state.get("validation_status") == "failure":
                state = await self._handle_failure(state)
            elif state.get("validation_status") == "adapt":
                state = await self._adapt_plan(state)
            
            # Check completion
            completion_status = self._is_workflow_complete(state)
            if completion_status == "complete":
                state["workflow_status"] = WorkflowStatus.COMPLETED.value
                break
            elif completion_status == "failed":
                state["workflow_status"] = WorkflowStatus.FAILED.value
                break
        
        return await self._finalize_workflow(state)
    
    def pause_workflow(self):
        """Pause workflow execution."""
        self.pause_requested = True
        self.logger.info("Workflow pause requested")
    
    def resume_workflow(self):
        """Resume paused workflow."""
        self.pause_requested = False
        self.logger.info("Workflow resume requested")
    
    def stop_workflow(self):
        """Stop workflow execution."""
        self.should_stop = True
        self.logger.info("Workflow stop requested")
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current workflow state."""
        return self.current_state
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get workflow state history."""
        return self.state_history.copy()
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics."""
        if not self.current_state:
            return {}
        
        return self.current_state.get("metrics", {})
    
    def clear_history(self):
        """Clear state history."""
        self.state_history.clear()
        self.logger.info("Workflow state history cleared")