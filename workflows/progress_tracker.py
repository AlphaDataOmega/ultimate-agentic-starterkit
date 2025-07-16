"""
Progress Tracking System for the Ultimate Agentic StarterKit.

This module provides real-time monitoring, visualization, milestone tracking,
and comprehensive progress reporting for workflow execution.
"""

import asyncio
import threading
import time
import json
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque

from core.logger import get_logger
from core.voice_alerts import get_voice_alerts
from core.config import get_config


class ProgressEventType(str, Enum):
    """Types of progress events."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRIED = "task_retried"
    MILESTONE_REACHED = "milestone_reached"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    CHECKPOINT_CREATED = "checkpoint_created"
    PLAN_ADAPTED = "plan_adapted"


class MilestoneType(str, Enum):
    """Types of milestones."""
    PERCENTAGE = "percentage"
    TASK_COUNT = "task_count"
    TIME_BASED = "time_based"
    CUSTOM = "custom"


@dataclass
class ProgressEvent:
    """Individual progress event."""
    event_id: str
    event_type: ProgressEventType
    timestamp: datetime
    workflow_id: str
    project_id: str
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'workflow_id': self.workflow_id,
            'project_id': self.project_id,
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'message': self.message,
            'data': self.data
        }


@dataclass
class Milestone:
    """Milestone definition."""
    milestone_id: str
    name: str
    milestone_type: MilestoneType
    target_value: Any
    description: str = ""
    voice_alert: bool = True
    reached: bool = False
    reached_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'milestone_id': self.milestone_id,
            'name': self.name,
            'milestone_type': self.milestone_type.value,
            'target_value': self.target_value,
            'description': self.description,
            'voice_alert': self.voice_alert,
            'reached': self.reached,
            'reached_at': self.reached_at.isoformat() if self.reached_at else None
        }


@dataclass
class ProgressSnapshot:
    """Snapshot of workflow progress at a point in time."""
    timestamp: datetime
    workflow_id: str
    project_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    running_tasks: int
    pending_tasks: int
    overall_progress: float
    overall_confidence: float
    estimated_completion: Optional[datetime] = None
    current_phase: str = "unknown"
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def failure_rate(self) -> float:
        """Get failure rate."""
        if self.total_tasks == 0:
            return 0.0
        return (self.failed_tasks / self.total_tasks) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'workflow_id': self.workflow_id,
            'project_id': self.project_id,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'running_tasks': self.running_tasks,
            'pending_tasks': self.pending_tasks,
            'overall_progress': self.overall_progress,
            'overall_confidence': self.overall_confidence,
            'completion_percentage': self.completion_percentage,
            'failure_rate': self.failure_rate,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'current_phase': self.current_phase
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for workflow execution."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_execution_time: float = 0.0
    task_execution_times: List[float] = field(default_factory=list)
    agent_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    throughput_tasks_per_minute: float = 0.0
    average_task_time: float = 0.0
    peak_concurrent_tasks: int = 0
    
    def update_metrics(self, completed_tasks: int, failed_tasks: int, 
                      current_time: Optional[datetime] = None):
        """Update performance metrics."""
        if current_time is None:
            current_time = datetime.now()
        
        # Calculate execution time
        if self.end_time:
            self.total_execution_time = (self.end_time - self.start_time).total_seconds()
        else:
            self.total_execution_time = (current_time - self.start_time).total_seconds()
        
        # Calculate throughput
        if self.total_execution_time > 0:
            total_processed = completed_tasks + failed_tasks
            self.throughput_tasks_per_minute = (total_processed / self.total_execution_time) * 60
        
        # Calculate average task time
        if self.task_execution_times:
            self.average_task_time = sum(self.task_execution_times) / len(self.task_execution_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_execution_time': self.total_execution_time,
            'task_execution_times': self.task_execution_times.copy(),
            'agent_performance': self.agent_performance.copy(),
            'throughput_tasks_per_minute': self.throughput_tasks_per_minute,
            'average_task_time': self.average_task_time,
            'peak_concurrent_tasks': self.peak_concurrent_tasks
        }


class ProgressTracker:
    """
    Comprehensive progress tracking system for workflow execution.
    
    This class provides real-time monitoring, milestone tracking, performance
    metrics, and visual progress reporting with voice alerts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the progress tracker.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.logger = get_logger("progress_tracker")
        self.voice = get_voice_alerts()
        
        # Configuration
        self.max_event_history = self.config.get('max_event_history', 1000)
        self.snapshot_interval = self.config.get('snapshot_interval', 30)  # seconds
        self.milestone_check_interval = self.config.get('milestone_check_interval', 10)  # seconds
        self.enable_voice_alerts = self.config.get('enable_voice_alerts', True)
        
        # Event tracking
        self.events: deque[ProgressEvent] = deque(maxlen=self.max_event_history)
        self.event_callbacks: List[Callable] = []
        self.event_lock = threading.Lock()
        
        # Progress snapshots
        self.snapshots: deque[ProgressSnapshot] = deque(maxlen=100)
        self.snapshot_timer: Optional[threading.Timer] = None
        
        # Milestones
        self.milestones: Dict[str, Milestone] = {}
        self.milestone_callbacks: List[Callable] = []
        
        # Performance metrics
        self.metrics: Dict[str, PerformanceMetrics] = {}
        
        # Current state tracking
        self.workflow_states: Dict[str, Dict[str, Any]] = {}
        
        # Visualization data
        self.visualization_data: Dict[str, Any] = {
            'progress_over_time': [],
            'task_completion_timeline': [],
            'agent_performance': {},
            'milestone_progress': []
        }
        
        self.logger.info("Progress Tracker initialized")
    
    def start_workflow_tracking(self, workflow_id: str, project_id: str,
                              total_tasks: int, workflow_data: Dict[str, Any] = None):
        """
        Start tracking a new workflow.
        
        Args:
            workflow_id: Workflow identifier
            project_id: Project identifier
            total_tasks: Total number of tasks
            workflow_data: Additional workflow data
        """
        self.logger.info(f"Starting workflow tracking: {workflow_id}")
        
        # Initialize workflow state
        self.workflow_states[workflow_id] = {
            'project_id': project_id,
            'total_tasks': total_tasks,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'running_tasks': 0,
            'pending_tasks': total_tasks,
            'overall_progress': 0.0,
            'overall_confidence': 0.0,
            'start_time': datetime.now(),
            'end_time': None,
            'current_phase': 'initialization',
            'data': workflow_data or {}
        }
        
        # Initialize metrics
        self.metrics[workflow_id] = PerformanceMetrics(start_time=datetime.now())
        
        # Create start event
        self._create_event(
            event_type=ProgressEventType.WORKFLOW_STARTED,
            workflow_id=workflow_id,
            project_id=project_id,
            message=f"Workflow {workflow_id} started with {total_tasks} tasks"
        )
        
        # Start snapshot timer
        if not self.snapshot_timer:
            self._start_snapshot_timer()
        
        # Voice alert
        if self.enable_voice_alerts:
            self.voice.speak_milestone(f"Workflow {workflow_id} started")
    
    def update_task_progress(self, workflow_id: str, task_id: str,
                           status: str, confidence: float = 0.0,
                           agent_id: str = None, execution_time: float = None,
                           task_data: Dict[str, Any] = None):
        """
        Update progress for a specific task.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Task identifier
            status: Task status (started, completed, failed, retried)
            confidence: Task confidence score
            agent_id: Agent identifier
            execution_time: Task execution time
            task_data: Additional task data
        """
        if workflow_id not in self.workflow_states:
            self.logger.warning(f"Unknown workflow: {workflow_id}")
            return
        
        state = self.workflow_states[workflow_id]
        
        # Update counters based on status
        if status == 'started':
            state['running_tasks'] += 1
            state['pending_tasks'] -= 1
            event_type = ProgressEventType.TASK_STARTED
            
        elif status == 'completed':
            state['completed_tasks'] += 1
            state['running_tasks'] -= 1
            event_type = ProgressEventType.TASK_COMPLETED
            
            # Update metrics
            if execution_time:
                self.metrics[workflow_id].task_execution_times.append(execution_time)
            
            # Voice alert for important tasks
            if self.enable_voice_alerts and confidence > 0.9:
                self.voice.speak_success(f"High-confidence task completed")
                
        elif status == 'failed':
            state['failed_tasks'] += 1
            state['running_tasks'] -= 1
            event_type = ProgressEventType.TASK_FAILED
            
            # Voice alert for failures
            if self.enable_voice_alerts:
                self.voice.speak_error(f"Task failed")
                
        elif status == 'retried':
            event_type = ProgressEventType.TASK_RETRIED
        else:
            self.logger.warning(f"Unknown task status: {status}")
            return
        
        # Update overall progress
        completed = state['completed_tasks']
        total = state['total_tasks']
        if total > 0:
            state['overall_progress'] = completed / total
        
        # Update agent performance
        if agent_id and execution_time:
            metrics = self.metrics[workflow_id]
            if agent_id not in metrics.agent_performance:
                metrics.agent_performance[agent_id] = {
                    'tasks_completed': 0,
                    'total_execution_time': 0.0,
                    'average_execution_time': 0.0,
                    'confidence_scores': []
                }
            
            agent_metrics = metrics.agent_performance[agent_id]
            agent_metrics['tasks_completed'] += 1
            agent_metrics['total_execution_time'] += execution_time
            agent_metrics['average_execution_time'] = (
                agent_metrics['total_execution_time'] / agent_metrics['tasks_completed']
            )
            
            if confidence > 0:
                agent_metrics['confidence_scores'].append(confidence)
        
        # Create event
        self._create_event(
            event_type=event_type,
            workflow_id=workflow_id,
            project_id=state['project_id'],
            task_id=task_id,
            agent_id=agent_id,
            message=f"Task {task_id} {status}",
            data={
                'confidence': confidence,
                'execution_time': execution_time,
                **(task_data or {})
            }
        )
        
        # Check milestones
        self._check_milestones(workflow_id)
        
        self.logger.debug(f"Task {task_id} progress updated: {status}")
    
    def complete_workflow(self, workflow_id: str, success: bool = True,
                         final_confidence: float = 0.0):
        """
        Mark workflow as completed.
        
        Args:
            workflow_id: Workflow identifier
            success: Whether workflow completed successfully
            final_confidence: Final confidence score
        """
        if workflow_id not in self.workflow_states:
            self.logger.warning(f"Unknown workflow: {workflow_id}")
            return
        
        state = self.workflow_states[workflow_id]
        state['end_time'] = datetime.now()
        state['overall_confidence'] = final_confidence
        
        # Update metrics
        metrics = self.metrics[workflow_id]
        metrics.end_time = datetime.now()
        metrics.update_metrics(
            completed_tasks=state['completed_tasks'],
            failed_tasks=state['failed_tasks']
        )
        
        # Create event
        event_type = ProgressEventType.WORKFLOW_COMPLETED if success else ProgressEventType.WORKFLOW_FAILED
        self._create_event(
            event_type=event_type,
            workflow_id=workflow_id,
            project_id=state['project_id'],
            message=f"Workflow {workflow_id} {'completed' if success else 'failed'}",
            data={
                'success': success,
                'final_confidence': final_confidence,
                'total_execution_time': metrics.total_execution_time
            }
        )
        
        # Voice alert
        if self.enable_voice_alerts:
            if success:
                self.voice.speak_milestone(f"Workflow completed successfully")
            else:
                self.voice.speak_error(f"Workflow failed")
        
        self.logger.info(f"Workflow {workflow_id} {'completed' if success else 'failed'}")
    
    def add_milestone(self, milestone: Milestone):
        """
        Add a milestone to track.
        
        Args:
            milestone: Milestone to add
        """
        self.milestones[milestone.milestone_id] = milestone
        self.logger.info(f"Added milestone: {milestone.name}")
    
    def create_percentage_milestone(self, name: str, percentage: float,
                                  description: str = "") -> str:
        """
        Create a percentage-based milestone.
        
        Args:
            name: Milestone name
            percentage: Target percentage (0-100)
            description: Optional description
            
        Returns:
            Milestone ID
        """
        milestone_id = str(uuid.uuid4())
        milestone = Milestone(
            milestone_id=milestone_id,
            name=name,
            milestone_type=MilestoneType.PERCENTAGE,
            target_value=percentage,
            description=description
        )
        self.add_milestone(milestone)
        return milestone_id
    
    def create_task_count_milestone(self, name: str, task_count: int,
                                  description: str = "") -> str:
        """
        Create a task count-based milestone.
        
        Args:
            name: Milestone name
            task_count: Target task count
            description: Optional description
            
        Returns:
            Milestone ID
        """
        milestone_id = str(uuid.uuid4())
        milestone = Milestone(
            milestone_id=milestone_id,
            name=name,
            milestone_type=MilestoneType.TASK_COUNT,
            target_value=task_count,
            description=description
        )
        self.add_milestone(milestone)
        return milestone_id
    
    def _check_milestones(self, workflow_id: str):
        """Check if any milestones have been reached."""
        if workflow_id not in self.workflow_states:
            return
        
        state = self.workflow_states[workflow_id]
        
        for milestone in self.milestones.values():
            if milestone.reached:
                continue
            
            reached = False
            
            if milestone.milestone_type == MilestoneType.PERCENTAGE:
                current_percentage = (state['completed_tasks'] / state['total_tasks']) * 100
                reached = current_percentage >= milestone.target_value
                
            elif milestone.milestone_type == MilestoneType.TASK_COUNT:
                reached = state['completed_tasks'] >= milestone.target_value
                
            elif milestone.milestone_type == MilestoneType.TIME_BASED:
                elapsed = (datetime.now() - state['start_time']).total_seconds()
                reached = elapsed >= milestone.target_value
            
            if reached:
                milestone.reached = True
                milestone.reached_at = datetime.now()
                
                # Create event
                self._create_event(
                    event_type=ProgressEventType.MILESTONE_REACHED,
                    workflow_id=workflow_id,
                    project_id=state['project_id'],
                    message=f"Milestone reached: {milestone.name}",
                    data={'milestone_id': milestone.milestone_id}
                )
                
                # Voice alert
                if self.enable_voice_alerts and milestone.voice_alert:
                    self.voice.speak_milestone(f"Milestone reached: {milestone.name}")
                
                # Notify callbacks
                for callback in self.milestone_callbacks:
                    try:
                        callback(milestone)
                    except Exception as e:
                        self.logger.error(f"Error in milestone callback: {e}")
                
                self.logger.info(f"Milestone reached: {milestone.name}")
    
    def _create_event(self, event_type: ProgressEventType, workflow_id: str,
                     project_id: str, task_id: str = None, agent_id: str = None,
                     message: str = "", data: Dict[str, Any] = None):
        """Create a progress event."""
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            workflow_id=workflow_id,
            project_id=project_id,
            task_id=task_id,
            agent_id=agent_id,
            message=message,
            data=data or {}
        )
        
        with self.event_lock:
            self.events.append(event)
        
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error in event callback: {e}")
    
    def _start_snapshot_timer(self):
        """Start the snapshot timer."""
        if self.snapshot_timer:
            self.snapshot_timer.cancel()
        
        self.snapshot_timer = threading.Timer(
            self.snapshot_interval,
            self._take_snapshots
        )
        self.snapshot_timer.daemon = True
        self.snapshot_timer.start()
    
    def _take_snapshots(self):
        """Take progress snapshots for all active workflows."""
        try:
            for workflow_id, state in self.workflow_states.items():
                if state['end_time'] is None:  # Only active workflows
                    snapshot = ProgressSnapshot(
                        timestamp=datetime.now(),
                        workflow_id=workflow_id,
                        project_id=state['project_id'],
                        total_tasks=state['total_tasks'],
                        completed_tasks=state['completed_tasks'],
                        failed_tasks=state['failed_tasks'],
                        running_tasks=state['running_tasks'],
                        pending_tasks=state['pending_tasks'],
                        overall_progress=state['overall_progress'],
                        overall_confidence=state['overall_confidence'],
                        current_phase=state['current_phase']
                    )
                    
                    self.snapshots.append(snapshot)
        
        except Exception as e:
            self.logger.error(f"Error taking snapshots: {e}")
        
        # Schedule next snapshot
        self._start_snapshot_timer()
    
    def get_current_progress(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current progress for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Progress data or None if not found
        """
        if workflow_id not in self.workflow_states:
            return None
        
        state = self.workflow_states[workflow_id]
        return {
            'workflow_id': workflow_id,
            'project_id': state['project_id'],
            'total_tasks': state['total_tasks'],
            'completed_tasks': state['completed_tasks'],
            'failed_tasks': state['failed_tasks'],
            'running_tasks': state['running_tasks'],
            'pending_tasks': state['pending_tasks'],
            'overall_progress': state['overall_progress'],
            'overall_confidence': state['overall_confidence'],
            'completion_percentage': (state['completed_tasks'] / state['total_tasks']) * 100 if state['total_tasks'] > 0 else 0,
            'current_phase': state['current_phase'],
            'start_time': state['start_time'].isoformat(),
            'end_time': state['end_time'].isoformat() if state['end_time'] else None
        }
    
    def get_progress_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        Get progress history for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            List of progress snapshots
        """
        return [
            snapshot.to_dict()
            for snapshot in self.snapshots
            if snapshot.workflow_id == workflow_id
        ]
    
    def get_events(self, workflow_id: str = None, event_type: ProgressEventType = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get progress events with optional filtering.
        
        Args:
            workflow_id: Optional workflow ID filter
            event_type: Optional event type filter
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        with self.event_lock:
            events = list(self.events)
        
        # Apply filters
        if workflow_id:
            events = [e for e in events if e.workflow_id == workflow_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        events = events[:limit]
        
        return [event.to_dict() for event in events]
    
    def get_milestones(self, workflow_id: str = None) -> List[Dict[str, Any]]:
        """
        Get milestones with optional filtering.
        
        Args:
            workflow_id: Optional workflow ID filter
            
        Returns:
            List of milestone dictionaries
        """
        return [milestone.to_dict() for milestone in self.milestones.values()]
    
    def get_performance_metrics(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Performance metrics or None if not found
        """
        if workflow_id not in self.metrics:
            return None
        
        return self.metrics[workflow_id].to_dict()
    
    def add_event_callback(self, callback: Callable):
        """Add an event callback function."""
        self.event_callbacks.append(callback)
    
    def add_milestone_callback(self, callback: Callable):
        """Add a milestone callback function."""
        self.milestone_callbacks.append(callback)
    
    def generate_progress_report(self, workflow_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive progress report.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Progress report dictionary
        """
        if workflow_id not in self.workflow_states:
            return {}
        
        state = self.workflow_states[workflow_id]
        metrics = self.metrics.get(workflow_id)
        
        # Get milestones
        milestones = [m for m in self.milestones.values() if m.reached]
        
        # Get recent events
        recent_events = self.get_events(workflow_id, limit=10)
        
        report = {
            'workflow_id': workflow_id,
            'project_id': state['project_id'],
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_tasks': state['total_tasks'],
                'completed_tasks': state['completed_tasks'],
                'failed_tasks': state['failed_tasks'],
                'running_tasks': state['running_tasks'],
                'pending_tasks': state['pending_tasks'],
                'completion_percentage': (state['completed_tasks'] / state['total_tasks']) * 100 if state['total_tasks'] > 0 else 0,
                'overall_confidence': state['overall_confidence'],
                'current_phase': state['current_phase']
            },
            'timing': {
                'start_time': state['start_time'].isoformat(),
                'end_time': state['end_time'].isoformat() if state['end_time'] else None,
                'total_execution_time': metrics.total_execution_time if metrics else 0.0
            },
            'performance': metrics.to_dict() if metrics else {},
            'milestones': [m.to_dict() for m in milestones],
            'recent_events': recent_events
        }
        
        return report
    
    def clear_history(self, workflow_id: str = None):
        """
        Clear progress history.
        
        Args:
            workflow_id: Optional workflow ID to clear specific history
        """
        if workflow_id:
            # Clear specific workflow data
            if workflow_id in self.workflow_states:
                del self.workflow_states[workflow_id]
            
            if workflow_id in self.metrics:
                del self.metrics[workflow_id]
            
            # Remove events for this workflow
            with self.event_lock:
                self.events = deque([
                    e for e in self.events if e.workflow_id != workflow_id
                ], maxlen=self.max_event_history)
            
            # Remove snapshots for this workflow
            self.snapshots = deque([
                s for s in self.snapshots if s.workflow_id != workflow_id
            ], maxlen=100)
            
        else:
            # Clear all history
            self.workflow_states.clear()
            self.metrics.clear()
            
            with self.event_lock:
                self.events.clear()
            
            self.snapshots.clear()
            self.milestones.clear()
        
        self.logger.info(f"Progress history cleared for {workflow_id or 'all workflows'}")
    
    def shutdown(self):
        """Shutdown the progress tracker."""
        self.logger.info("Shutting down progress tracker")
        
        # Cancel snapshot timer
        if self.snapshot_timer:
            self.snapshot_timer.cancel()
        
        # Clear data
        self.clear_history()
        
        self.logger.info("Progress tracker shutdown complete")


# Global progress tracker instance
_progress_tracker_instance: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """
    Get the global progress tracker instance.
    
    Returns:
        ProgressTracker instance
    """
    global _progress_tracker_instance
    if _progress_tracker_instance is None:
        _progress_tracker_instance = ProgressTracker()
    return _progress_tracker_instance


def start_workflow_tracking(workflow_id: str, project_id: str, total_tasks: int):
    """
    Convenience function to start workflow tracking.
    
    Args:
        workflow_id: Workflow identifier
        project_id: Project identifier
        total_tasks: Total number of tasks
    """
    get_progress_tracker().start_workflow_tracking(workflow_id, project_id, total_tasks)


def update_task_progress(workflow_id: str, task_id: str, status: str,
                        confidence: float = 0.0, agent_id: str = None):
    """
    Convenience function to update task progress.
    
    Args:
        workflow_id: Workflow identifier
        task_id: Task identifier
        status: Task status
        confidence: Task confidence
        agent_id: Agent identifier
    """
    get_progress_tracker().update_task_progress(
        workflow_id, task_id, status, confidence, agent_id
    )


def create_percentage_milestone(name: str, percentage: float) -> str:
    """
    Convenience function to create a percentage milestone.
    
    Args:
        name: Milestone name
        percentage: Target percentage
        
    Returns:
        Milestone ID
    """
    return get_progress_tracker().create_percentage_milestone(name, percentage)