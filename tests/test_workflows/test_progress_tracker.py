"""
Tests for Progress Tracking System.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import uuid

from workflows.progress_tracker import (
    ProgressTracker,
    ProgressEvent,
    ProgressSnapshot,
    Milestone,
    PerformanceMetrics,
    ProgressEventType,
    MilestoneType,
    get_progress_tracker,
    start_workflow_tracking,
    update_task_progress,
    create_percentage_milestone
)


class TestProgressEvent:
    """Test cases for ProgressEvent."""
    
    def test_progress_event_initialization(self):
        """Test progress event initialization."""
        event = ProgressEvent(
            event_id="test-event",
            event_type=ProgressEventType.TASK_COMPLETED,
            timestamp=datetime.now(),
            workflow_id="test-workflow",
            project_id="test-project",
            task_id="test-task",
            agent_id="test-agent",
            message="Task completed successfully",
            data={"confidence": 0.9}
        )
        
        assert event.event_id == "test-event"
        assert event.event_type == ProgressEventType.TASK_COMPLETED
        assert event.workflow_id == "test-workflow"
        assert event.project_id == "test-project"
        assert event.task_id == "test-task"
        assert event.agent_id == "test-agent"
        assert event.message == "Task completed successfully"
        assert event.data["confidence"] == 0.9
    
    def test_progress_event_to_dict(self):
        """Test progress event serialization."""
        event = ProgressEvent(
            event_id="test-event",
            event_type=ProgressEventType.TASK_COMPLETED,
            timestamp=datetime.now(),
            workflow_id="test-workflow",
            project_id="test-project",
            task_id="test-task",
            message="Task completed"
        )
        
        data = event.to_dict()
        
        assert data['event_id'] == "test-event"
        assert data['event_type'] == "task_completed"
        assert data['workflow_id'] == "test-workflow"
        assert data['project_id'] == "test-project"
        assert data['task_id'] == "test-task"
        assert data['message'] == "Task completed"
        assert 'timestamp' in data


class TestMilestone:
    """Test cases for Milestone."""
    
    def test_milestone_initialization(self):
        """Test milestone initialization."""
        milestone = Milestone(
            milestone_id="test-milestone",
            name="50% Complete",
            milestone_type=MilestoneType.PERCENTAGE,
            target_value=50.0,
            description="Half way there",
            voice_alert=True
        )
        
        assert milestone.milestone_id == "test-milestone"
        assert milestone.name == "50% Complete"
        assert milestone.milestone_type == MilestoneType.PERCENTAGE
        assert milestone.target_value == 50.0
        assert milestone.description == "Half way there"
        assert milestone.voice_alert is True
        assert milestone.reached is False
        assert milestone.reached_at is None
    
    def test_milestone_to_dict(self):
        """Test milestone serialization."""
        milestone = Milestone(
            milestone_id="test-milestone",
            name="50% Complete",
            milestone_type=MilestoneType.PERCENTAGE,
            target_value=50.0
        )
        
        data = milestone.to_dict()
        
        assert data['milestone_id'] == "test-milestone"
        assert data['name'] == "50% Complete"
        assert data['milestone_type'] == "percentage"
        assert data['target_value'] == 50.0
        assert data['reached'] is False
        assert data['reached_at'] is None


class TestProgressSnapshot:
    """Test cases for ProgressSnapshot."""
    
    def test_progress_snapshot_initialization(self):
        """Test progress snapshot initialization."""
        snapshot = ProgressSnapshot(
            timestamp=datetime.now(),
            workflow_id="test-workflow",
            project_id="test-project",
            total_tasks=10,
            completed_tasks=5,
            failed_tasks=1,
            running_tasks=2,
            pending_tasks=2,
            overall_progress=0.5,
            overall_confidence=0.85
        )
        
        assert snapshot.workflow_id == "test-workflow"
        assert snapshot.project_id == "test-project"
        assert snapshot.total_tasks == 10
        assert snapshot.completed_tasks == 5
        assert snapshot.failed_tasks == 1
        assert snapshot.running_tasks == 2
        assert snapshot.pending_tasks == 2
        assert snapshot.overall_progress == 0.5
        assert snapshot.overall_confidence == 0.85
    
    def test_progress_snapshot_completion_percentage(self):
        """Test progress snapshot completion percentage calculation."""
        snapshot = ProgressSnapshot(
            timestamp=datetime.now(),
            workflow_id="test-workflow",
            project_id="test-project",
            total_tasks=10,
            completed_tasks=7,
            failed_tasks=0,
            running_tasks=1,
            pending_tasks=2,
            overall_progress=0.7,
            overall_confidence=0.8
        )
        
        assert snapshot.completion_percentage == 70.0
    
    def test_progress_snapshot_failure_rate(self):
        """Test progress snapshot failure rate calculation."""
        snapshot = ProgressSnapshot(
            timestamp=datetime.now(),
            workflow_id="test-workflow",
            project_id="test-project",
            total_tasks=10,
            completed_tasks=7,
            failed_tasks=2,
            running_tasks=1,
            pending_tasks=0,
            overall_progress=0.7,
            overall_confidence=0.8
        )
        
        assert snapshot.failure_rate == 20.0
    
    def test_progress_snapshot_to_dict(self):
        """Test progress snapshot serialization."""
        snapshot = ProgressSnapshot(
            timestamp=datetime.now(),
            workflow_id="test-workflow",
            project_id="test-project",
            total_tasks=10,
            completed_tasks=5,
            failed_tasks=1,
            running_tasks=2,
            pending_tasks=2,
            overall_progress=0.5,
            overall_confidence=0.85
        )
        
        data = snapshot.to_dict()
        
        assert data['workflow_id'] == "test-workflow"
        assert data['project_id'] == "test-project"
        assert data['total_tasks'] == 10
        assert data['completed_tasks'] == 5
        assert data['completion_percentage'] == 50.0
        assert data['failure_rate'] == 10.0
        assert 'timestamp' in data


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics."""
    
    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization."""
        start_time = datetime.now()
        metrics = PerformanceMetrics(start_time=start_time)
        
        assert metrics.start_time == start_time
        assert metrics.end_time is None
        assert metrics.total_execution_time == 0.0
        assert metrics.throughput_tasks_per_minute == 0.0
        assert metrics.average_task_time == 0.0
        assert metrics.peak_concurrent_tasks == 0
    
    def test_performance_metrics_update(self):
        """Test performance metrics update."""
        start_time = datetime.now()
        metrics = PerformanceMetrics(start_time=start_time)
        
        # Add some task execution times
        metrics.task_execution_times = [10.0, 15.0, 20.0]
        
        # Update metrics
        metrics.update_metrics(completed_tasks=3, failed_tasks=1)
        
        assert metrics.average_task_time == 15.0
        assert metrics.throughput_tasks_per_minute > 0
    
    def test_performance_metrics_to_dict(self):
        """Test performance metrics serialization."""
        start_time = datetime.now()
        metrics = PerformanceMetrics(start_time=start_time)
        metrics.task_execution_times = [10.0, 15.0, 20.0]
        
        data = metrics.to_dict()
        
        assert 'start_time' in data
        assert data['total_execution_time'] == 0.0
        assert data['task_execution_times'] == [10.0, 15.0, 20.0]
        assert data['throughput_tasks_per_minute'] == 0.0
        assert data['average_task_time'] == 0.0


class TestProgressTracker:
    """Test cases for ProgressTracker."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            'max_event_history': 100,
            'snapshot_interval': 10,
            'milestone_check_interval': 5,
            'enable_voice_alerts': True
        }
    
    @pytest.fixture
    def progress_tracker(self, mock_config):
        """Create progress tracker instance."""
        with patch('workflows.progress_tracker.get_logger'):
            with patch('workflows.progress_tracker.get_voice_alerts'):
                return ProgressTracker(mock_config)
    
    def test_progress_tracker_initialization(self, progress_tracker):
        """Test progress tracker initialization."""
        assert progress_tracker.max_event_history == 100
        assert progress_tracker.snapshot_interval == 10
        assert progress_tracker.milestone_check_interval == 5
        assert progress_tracker.enable_voice_alerts is True
        assert len(progress_tracker.events) == 0
        assert len(progress_tracker.snapshots) == 0
        assert len(progress_tracker.milestones) == 0
    
    def test_start_workflow_tracking(self, progress_tracker):
        """Test starting workflow tracking."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        total_tasks = 10
        
        progress_tracker.start_workflow_tracking(workflow_id, project_id, total_tasks)
        
        assert workflow_id in progress_tracker.workflow_states
        state = progress_tracker.workflow_states[workflow_id]
        assert state['project_id'] == project_id
        assert state['total_tasks'] == total_tasks
        assert state['completed_tasks'] == 0
        assert state['failed_tasks'] == 0
        assert state['pending_tasks'] == total_tasks
        assert state['current_phase'] == 'initialization'
    
    def test_update_task_progress_started(self, progress_tracker):
        """Test updating task progress - started."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Update task progress
        progress_tracker.update_task_progress(
            workflow_id=workflow_id,
            task_id="task-1",
            status="started",
            agent_id="agent-1"
        )
        
        state = progress_tracker.workflow_states[workflow_id]
        assert state['running_tasks'] == 1
        assert state['pending_tasks'] == 4
    
    def test_update_task_progress_completed(self, progress_tracker):
        """Test updating task progress - completed."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Start task first
        progress_tracker.update_task_progress(
            workflow_id=workflow_id,
            task_id="task-1",
            status="started",
            agent_id="agent-1"
        )
        
        # Complete task
        progress_tracker.update_task_progress(
            workflow_id=workflow_id,
            task_id="task-1",
            status="completed",
            confidence=0.9,
            agent_id="agent-1",
            execution_time=10.0
        )
        
        state = progress_tracker.workflow_states[workflow_id]
        assert state['completed_tasks'] == 1
        assert state['running_tasks'] == 0
        assert state['overall_progress'] == 0.2  # 1/5
    
    def test_update_task_progress_failed(self, progress_tracker):
        """Test updating task progress - failed."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Start task first
        progress_tracker.update_task_progress(
            workflow_id=workflow_id,
            task_id="task-1",
            status="started",
            agent_id="agent-1"
        )
        
        # Fail task
        progress_tracker.update_task_progress(
            workflow_id=workflow_id,
            task_id="task-1",
            status="failed",
            confidence=0.3,
            agent_id="agent-1"
        )
        
        state = progress_tracker.workflow_states[workflow_id]
        assert state['failed_tasks'] == 1
        assert state['running_tasks'] == 0
    
    def test_complete_workflow_success(self, progress_tracker):
        """Test completing workflow - success."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Complete workflow
        progress_tracker.complete_workflow(
            workflow_id=workflow_id,
            success=True,
            final_confidence=0.9
        )
        
        state = progress_tracker.workflow_states[workflow_id]
        assert state['end_time'] is not None
        assert state['overall_confidence'] == 0.9
        
        # Check for workflow completion event
        workflow_events = [e for e in progress_tracker.events if e.event_type == ProgressEventType.WORKFLOW_COMPLETED]
        assert len(workflow_events) == 1
    
    def test_complete_workflow_failure(self, progress_tracker):
        """Test completing workflow - failure."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Complete workflow with failure
        progress_tracker.complete_workflow(
            workflow_id=workflow_id,
            success=False,
            final_confidence=0.3
        )
        
        # Check for workflow failure event
        workflow_events = [e for e in progress_tracker.events if e.event_type == ProgressEventType.WORKFLOW_FAILED]
        assert len(workflow_events) == 1
    
    def test_add_milestone(self, progress_tracker):
        """Test adding a milestone."""
        milestone = Milestone(
            milestone_id="test-milestone",
            name="50% Complete",
            milestone_type=MilestoneType.PERCENTAGE,
            target_value=50.0
        )
        
        progress_tracker.add_milestone(milestone)
        
        assert "test-milestone" in progress_tracker.milestones
        assert progress_tracker.milestones["test-milestone"] == milestone
    
    def test_create_percentage_milestone(self, progress_tracker):
        """Test creating a percentage milestone."""
        milestone_id = progress_tracker.create_percentage_milestone(
            name="75% Complete",
            percentage=75.0,
            description="Three quarters done"
        )
        
        assert milestone_id in progress_tracker.milestones
        milestone = progress_tracker.milestones[milestone_id]
        assert milestone.name == "75% Complete"
        assert milestone.milestone_type == MilestoneType.PERCENTAGE
        assert milestone.target_value == 75.0
        assert milestone.description == "Three quarters done"
    
    def test_create_task_count_milestone(self, progress_tracker):
        """Test creating a task count milestone."""
        milestone_id = progress_tracker.create_task_count_milestone(
            name="5 Tasks Complete",
            task_count=5,
            description="Five tasks done"
        )
        
        assert milestone_id in progress_tracker.milestones
        milestone = progress_tracker.milestones[milestone_id]
        assert milestone.name == "5 Tasks Complete"
        assert milestone.milestone_type == MilestoneType.TASK_COUNT
        assert milestone.target_value == 5
        assert milestone.description == "Five tasks done"
    
    def test_check_milestones_percentage(self, progress_tracker):
        """Test milestone checking - percentage based."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 10)
        
        # Add percentage milestone
        progress_tracker.create_percentage_milestone("50% Complete", 50.0)
        
        # Complete 5 tasks (50%)
        for i in range(5):
            progress_tracker.update_task_progress(
                workflow_id=workflow_id,
                task_id=f"task-{i}",
                status="completed",
                confidence=0.9
            )
        
        # Check if milestone was reached
        milestone_events = [e for e in progress_tracker.events if e.event_type == ProgressEventType.MILESTONE_REACHED]
        assert len(milestone_events) == 1
        assert milestone_events[0].data['milestone_id'] is not None
    
    def test_check_milestones_task_count(self, progress_tracker):
        """Test milestone checking - task count based."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 10)
        
        # Add task count milestone
        progress_tracker.create_task_count_milestone("3 Tasks Complete", 3)
        
        # Complete 3 tasks
        for i in range(3):
            progress_tracker.update_task_progress(
                workflow_id=workflow_id,
                task_id=f"task-{i}",
                status="completed",
                confidence=0.9
            )
        
        # Check if milestone was reached
        milestone_events = [e for e in progress_tracker.events if e.event_type == ProgressEventType.MILESTONE_REACHED]
        assert len(milestone_events) == 1
    
    def test_get_current_progress(self, progress_tracker):
        """Test getting current progress."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 10)
        
        # Complete some tasks
        for i in range(3):
            progress_tracker.update_task_progress(
                workflow_id=workflow_id,
                task_id=f"task-{i}",
                status="completed",
                confidence=0.9
            )
        
        progress = progress_tracker.get_current_progress(workflow_id)
        
        assert progress is not None
        assert progress['workflow_id'] == workflow_id
        assert progress['project_id'] == project_id
        assert progress['total_tasks'] == 10
        assert progress['completed_tasks'] == 3
        assert progress['completion_percentage'] == 30.0
    
    def test_get_current_progress_unknown_workflow(self, progress_tracker):
        """Test getting current progress for unknown workflow."""
        progress = progress_tracker.get_current_progress("unknown-workflow")
        assert progress is None
    
    def test_get_events_no_filter(self, progress_tracker):
        """Test getting events without filters."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Update task progress
        progress_tracker.update_task_progress(
            workflow_id=workflow_id,
            task_id="task-1",
            status="completed",
            confidence=0.9
        )
        
        events = progress_tracker.get_events()
        
        assert len(events) >= 2  # At least workflow start and task complete
        assert all('event_type' in event for event in events)
        assert all('timestamp' in event for event in events)
    
    def test_get_events_with_filter(self, progress_tracker):
        """Test getting events with filters."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Update task progress
        progress_tracker.update_task_progress(
            workflow_id=workflow_id,
            task_id="task-1",
            status="completed",
            confidence=0.9
        )
        
        # Get only task completion events
        events = progress_tracker.get_events(
            workflow_id=workflow_id,
            event_type=ProgressEventType.TASK_COMPLETED
        )
        
        assert len(events) == 1
        assert events[0]['event_type'] == 'task_completed'
        assert events[0]['workflow_id'] == workflow_id
    
    def test_get_performance_metrics(self, progress_tracker):
        """Test getting performance metrics."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        metrics = progress_tracker.get_performance_metrics(workflow_id)
        
        assert metrics is not None
        assert 'start_time' in metrics
        assert 'total_execution_time' in metrics
        assert 'throughput_tasks_per_minute' in metrics
    
    def test_get_performance_metrics_unknown_workflow(self, progress_tracker):
        """Test getting performance metrics for unknown workflow."""
        metrics = progress_tracker.get_performance_metrics("unknown-workflow")
        assert metrics is None
    
    def test_generate_progress_report(self, progress_tracker):
        """Test generating progress report."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 10)
        
        # Complete some tasks
        for i in range(5):
            progress_tracker.update_task_progress(
                workflow_id=workflow_id,
                task_id=f"task-{i}",
                status="completed",
                confidence=0.9
            )
        
        # Add milestone
        progress_tracker.create_percentage_milestone("50% Complete", 50.0)
        
        report = progress_tracker.generate_progress_report(workflow_id)
        
        assert report['workflow_id'] == workflow_id
        assert report['project_id'] == project_id
        assert 'generated_at' in report
        assert 'summary' in report
        assert 'timing' in report
        assert 'performance' in report
        assert 'milestones' in report
        assert 'recent_events' in report
        
        # Check summary
        summary = report['summary']
        assert summary['total_tasks'] == 10
        assert summary['completed_tasks'] == 5
        assert summary['completion_percentage'] == 50.0
    
    def test_clear_history_specific_workflow(self, progress_tracker):
        """Test clearing history for specific workflow."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Clear history for this workflow
        progress_tracker.clear_history(workflow_id)
        
        assert workflow_id not in progress_tracker.workflow_states
        assert workflow_id not in progress_tracker.metrics
    
    def test_clear_history_all(self, progress_tracker):
        """Test clearing all history."""
        workflow_id = "test-workflow"
        project_id = "test-project"
        
        # Start workflow tracking
        progress_tracker.start_workflow_tracking(workflow_id, project_id, 5)
        
        # Clear all history
        progress_tracker.clear_history()
        
        assert len(progress_tracker.workflow_states) == 0
        assert len(progress_tracker.metrics) == 0
        assert len(progress_tracker.events) == 0
        assert len(progress_tracker.snapshots) == 0
        assert len(progress_tracker.milestones) == 0
    
    def test_shutdown(self, progress_tracker):
        """Test progress tracker shutdown."""
        progress_tracker.shutdown()
        
        # Should clear all data
        assert len(progress_tracker.workflow_states) == 0
        assert len(progress_tracker.metrics) == 0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_get_progress_tracker(self):
        """Test getting global progress tracker."""
        with patch('workflows.progress_tracker.ProgressTracker') as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            tracker = get_progress_tracker()
            
            assert tracker is not None
            mock_tracker_class.assert_called_once()
    
    def test_start_workflow_tracking(self):
        """Test convenience function for starting workflow tracking."""
        with patch('workflows.progress_tracker.get_progress_tracker') as mock_get_tracker:
            mock_tracker = Mock()
            mock_get_tracker.return_value = mock_tracker
            
            start_workflow_tracking("test-workflow", "test-project", 10)
            
            mock_tracker.start_workflow_tracking.assert_called_once_with(
                "test-workflow", "test-project", 10
            )
    
    def test_update_task_progress(self):
        """Test convenience function for updating task progress."""
        with patch('workflows.progress_tracker.get_progress_tracker') as mock_get_tracker:
            mock_tracker = Mock()
            mock_get_tracker.return_value = mock_tracker
            
            update_task_progress("test-workflow", "test-task", "completed", 0.9, "test-agent")
            
            mock_tracker.update_task_progress.assert_called_once_with(
                "test-workflow", "test-task", "completed", 0.9, "test-agent"
            )
    
    def test_create_percentage_milestone(self):
        """Test convenience function for creating percentage milestone."""
        with patch('workflows.progress_tracker.get_progress_tracker') as mock_get_tracker:
            mock_tracker = Mock()
            mock_tracker.create_percentage_milestone.return_value = "milestone-id"
            mock_get_tracker.return_value = mock_tracker
            
            milestone_id = create_percentage_milestone("50% Complete", 50.0)
            
            assert milestone_id == "milestone-id"
            mock_tracker.create_percentage_milestone.assert_called_once_with(
                "50% Complete", 50.0
            )