"""
Test data models system.
"""

import pytest
from datetime import datetime, timedelta
from uuid import UUID

from StarterKit.core.models import (
    TaskStatus,
    ConfidenceLevel,
    AgentType,
    ProjectTask,
    ProjectSpecification,
    AgentResult,
    WorkflowState,
    create_project_task,
    create_project_specification,
    create_agent_result
)


class TestEnums:
    """Test enumeration types."""
    
    def test_task_status_enum(self):
        """Test TaskStatus enumeration."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.BLOCKED == "blocked"
    
    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enumeration."""
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.HIGH == "high"
    
    def test_agent_type_enum(self):
        """Test AgentType enumeration."""
        assert AgentType.PARSER == "parser"
        assert AgentType.CODER == "coder"
        assert AgentType.TESTER == "tester"
        assert AgentType.ADVISOR == "advisor"
        assert AgentType.ORCHESTRATOR == "orchestrator"


class TestProjectTask:
    """Test ProjectTask model."""
    
    def test_create_valid_task(self):
        """Test creating a valid project task."""
        task = ProjectTask(
            title="Test Task",
            description="A test task description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        assert task.title == "Test Task"
        assert task.description == "A test task description"
        assert task.type == "CREATE"
        assert task.agent_type == AgentType.PARSER
        assert task.status == TaskStatus.PENDING
        assert task.confidence == 0.0
        assert task.attempts == 0
        assert task.max_attempts == 3
        assert isinstance(task.id, str)
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
    
    def test_task_id_generation(self):
        """Test that task IDs are unique."""
        task1 = ProjectTask(
            title="Task 1",
            description="Description 1",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        task2 = ProjectTask(
            title="Task 2",
            description="Description 2",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        assert task1.id != task2.id
        # Verify UUID format
        UUID(task1.id)
        UUID(task2.id)
    
    def test_invalid_task_type(self):
        """Test invalid task type validation."""
        with pytest.raises(ValueError):
            ProjectTask(
                title="Test Task",
                description="Description",
                type="INVALID",
                agent_type=AgentType.PARSER
            )
    
    def test_invalid_confidence(self):
        """Test invalid confidence value."""
        with pytest.raises(ValueError):
            ProjectTask(
                title="Test Task",
                description="Description",
                type="CREATE",
                agent_type=AgentType.PARSER,
                confidence=1.5  # Invalid: > 1.0
            )
    
    def test_task_status_methods(self):
        """Test task status management methods."""
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Test mark_in_progress
        old_updated_at = task.updated_at
        task.mark_in_progress()
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.attempts == 1
        assert task.updated_at > old_updated_at
        
        # Test mark_completed
        task.mark_completed(confidence=0.95)
        assert task.status == TaskStatus.COMPLETED
        assert task.confidence == 0.95
        assert task.error_message is None
        
        # Test mark_failed
        task.mark_failed("Test error message")
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Test error message"
    
    def test_can_retry(self):
        """Test retry logic."""
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER,
            max_attempts=3
        )
        
        # Initially can retry
        assert task.can_retry()
        
        # After 3 attempts, cannot retry
        task.attempts = 3
        assert not task.can_retry()
        
        # After 2 attempts, can still retry
        task.attempts = 2
        assert task.can_retry()
    
    def test_task_validation(self):
        """Test task field validation."""
        # Test empty title
        with pytest.raises(ValueError):
            ProjectTask(
                title="",
                description="Description",
                type="CREATE",
                agent_type=AgentType.PARSER
            )
        
        # Test long title
        with pytest.raises(ValueError):
            ProjectTask(
                title="x" * 201,  # Too long
                description="Description",
                type="CREATE",
                agent_type=AgentType.PARSER
            )
        
        # Test empty description
        with pytest.raises(ValueError):
            ProjectTask(
                title="Test Task",
                description="",
                type="CREATE",
                agent_type=AgentType.PARSER
            )
    
    def test_task_dependencies(self):
        """Test task dependencies."""
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=["task1", "task2"]
        )
        
        assert task.dependencies == ["task1", "task2"]
        
        # Test default empty dependencies
        task2 = ProjectTask(
            title="Test Task 2",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        assert task2.dependencies == []


class TestProjectSpecification:
    """Test ProjectSpecification model."""
    
    def test_create_valid_project_spec(self):
        """Test creating a valid project specification."""
        spec = ProjectSpecification(
            title="Test Project",
            description="A test project description",
            project_type="web"
        )
        
        assert spec.title == "Test Project"
        assert spec.description == "A test project description"
        assert spec.project_type == "web"
        assert spec.tasks == []
        assert spec.requirements == {}
        assert spec.validation_criteria == {}
        assert isinstance(spec.created_at, datetime)
    
    def test_invalid_project_type(self):
        """Test invalid project type validation."""
        with pytest.raises(ValueError):
            ProjectSpecification(
                title="Test Project",
                description="Description",
                project_type="invalid"
            )
    
    def test_add_task(self):
        """Test adding tasks to project specification."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        spec.add_task(task)
        assert len(spec.tasks) == 1
        assert spec.tasks[0] == task
    
    def test_get_task_by_id(self):
        """Test getting task by ID."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        spec.add_task(task)
        
        # Test finding existing task
        found_task = spec.get_task_by_id(task.id)
        assert found_task == task
        
        # Test finding non-existent task
        not_found = spec.get_task_by_id("nonexistent")
        assert not_found is None
    
    def test_get_tasks_by_status(self):
        """Test getting tasks by status."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        # Add tasks with different statuses
        task1 = ProjectTask(
            title="Task 1",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        task2 = ProjectTask(
            title="Task 2",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        task2.mark_completed()
        
        task3 = ProjectTask(
            title="Task 3",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        task3.mark_in_progress()
        
        spec.add_task(task1)
        spec.add_task(task2)
        spec.add_task(task3)
        
        # Test getting tasks by status
        pending_tasks = spec.get_tasks_by_status(TaskStatus.PENDING)
        assert len(pending_tasks) == 1
        assert pending_tasks[0] == task1
        
        completed_tasks = spec.get_tasks_by_status(TaskStatus.COMPLETED)
        assert len(completed_tasks) == 1
        assert completed_tasks[0] == task2
        
        in_progress_tasks = spec.get_tasks_by_status(TaskStatus.IN_PROGRESS)
        assert len(in_progress_tasks) == 1
        assert in_progress_tasks[0] == task3
    
    def test_get_ready_tasks(self):
        """Test getting ready tasks (no pending dependencies)."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        # Create tasks with dependencies
        task1 = ProjectTask(
            title="Task 1",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        task2 = ProjectTask(
            title="Task 2",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=[task1.id]
        )
        
        task3 = ProjectTask(
            title="Task 3",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        spec.add_task(task1)
        spec.add_task(task2)
        spec.add_task(task3)
        
        # Initially, task1 and task3 should be ready (no dependencies)
        ready_tasks = spec.get_ready_tasks()
        assert len(ready_tasks) == 2
        assert task1 in ready_tasks
        assert task3 in ready_tasks
        assert task2 not in ready_tasks
        
        # After task1 is completed, task2 should be ready
        task1.mark_completed()
        ready_tasks = spec.get_ready_tasks()
        assert len(ready_tasks) == 2
        assert task2 in ready_tasks
        assert task3 in ready_tasks


class TestAgentResult:
    """Test AgentResult model."""
    
    def test_create_valid_agent_result(self):
        """Test creating a valid agent result."""
        result = AgentResult(
            success=True,
            confidence=0.85,
            output={"message": "Task completed successfully"},
            agent_id="agent_001",
            execution_time=1.5
        )
        
        assert result.success is True
        assert result.confidence == 0.85
        assert result.output == {"message": "Task completed successfully"}
        assert result.agent_id == "agent_001"
        assert result.execution_time == 1.5
        assert result.error is None
        assert isinstance(result.timestamp, datetime)
    
    def test_invalid_confidence(self):
        """Test invalid confidence value."""
        with pytest.raises(ValueError):
            AgentResult(
                success=True,
                confidence=1.5,  # Invalid: > 1.0
                output="Test output",
                agent_id="agent_001",
                execution_time=1.0
            )
    
    def test_negative_execution_time(self):
        """Test negative execution time."""
        with pytest.raises(ValueError):
            AgentResult(
                success=True,
                confidence=0.5,
                output="Test output",
                agent_id="agent_001",
                execution_time=-1.0  # Invalid: negative
            )
    
    def test_get_confidence_level(self):
        """Test confidence level categorization."""
        # High confidence
        result_high = AgentResult(
            success=True,
            confidence=0.9,
            output="Test",
            agent_id="agent_001",
            execution_time=1.0
        )
        assert result_high.get_confidence_level() == ConfidenceLevel.HIGH
        
        # Medium confidence
        result_medium = AgentResult(
            success=True,
            confidence=0.7,
            output="Test",
            agent_id="agent_001",
            execution_time=1.0
        )
        assert result_medium.get_confidence_level() == ConfidenceLevel.MEDIUM
        
        # Low confidence
        result_low = AgentResult(
            success=True,
            confidence=0.3,
            output="Test",
            agent_id="agent_001",
            execution_time=1.0
        )
        assert result_low.get_confidence_level() == ConfidenceLevel.LOW
    
    def test_failed_result_with_error(self):
        """Test failed result with error message."""
        result = AgentResult(
            success=False,
            confidence=0.0,
            output=None,
            agent_id="agent_001",
            execution_time=2.0,
            error="Task execution failed"
        )
        
        assert result.success is False
        assert result.error == "Task execution failed"
        assert result.confidence == 0.0


class TestWorkflowState:
    """Test WorkflowState model."""
    
    def test_create_valid_workflow_state(self):
        """Test creating a valid workflow state."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        workflow = WorkflowState(
            project_spec=spec
        )
        
        assert workflow.project_spec == spec
        assert workflow.current_task is None
        assert workflow.completed_tasks == []
        assert workflow.failed_tasks == []
        assert workflow.agent_results == []
        assert workflow.overall_confidence == 0.0
        assert workflow.workflow_status == TaskStatus.PENDING
    
    def test_add_result(self):
        """Test adding agent results."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        workflow = WorkflowState(project_spec=spec)
        
        result = AgentResult(
            success=True,
            confidence=0.8,
            output="Test output",
            agent_id="agent_001",
            execution_time=1.0
        )
        
        workflow.add_result(result)
        
        assert len(workflow.agent_results) == 1
        assert workflow.agent_results[0] == result
        assert workflow.overall_confidence == 0.8
    
    def test_complete_task(self):
        """Test completing a task."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        spec.add_task(task)
        workflow = WorkflowState(project_spec=spec)
        
        workflow.complete_task(task.id)
        
        assert task.id in workflow.completed_tasks
        assert task.status == TaskStatus.COMPLETED
        assert workflow.workflow_status == TaskStatus.COMPLETED
    
    def test_fail_task(self):
        """Test failing a task."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        spec.add_task(task)
        workflow = WorkflowState(project_spec=spec)
        
        workflow.fail_task(task.id, "Test error")
        
        assert task.id in workflow.failed_tasks
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Test error"
        assert workflow.workflow_status == TaskStatus.FAILED
    
    def test_get_next_task(self):
        """Test getting next ready task."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        task1 = ProjectTask(
            title="Task 1",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        task2 = ProjectTask(
            title="Task 2",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=[task1.id]
        )
        
        spec.add_task(task1)
        spec.add_task(task2)
        
        workflow = WorkflowState(project_spec=spec)
        
        # First task should be ready
        next_task = workflow.get_next_task()
        assert next_task == task1
        
        # After completing task1, task2 should be ready
        workflow.complete_task(task1.id)
        next_task = workflow.get_next_task()
        assert next_task == task2
        
        # After completing task2, no more tasks
        workflow.complete_task(task2.id)
        next_task = workflow.get_next_task()
        assert next_task is None
    
    def test_overall_confidence_calculation(self):
        """Test overall confidence calculation."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        workflow = WorkflowState(project_spec=spec)
        
        # Add multiple results
        results = [
            AgentResult(
                success=True,
                confidence=0.8,
                output="Test",
                agent_id="agent_001",
                execution_time=1.0
            ),
            AgentResult(
                success=True,
                confidence=0.9,
                output="Test",
                agent_id="agent_002",
                execution_time=1.0
            ),
            AgentResult(
                success=True,
                confidence=0.7,
                output="Test",
                agent_id="agent_003",
                execution_time=1.0
            )
        ]
        
        for result in results:
            workflow.add_result(result)
        
        # Overall confidence should be average of recent results
        expected_confidence = (0.8 + 0.9 + 0.7) / 3
        assert abs(workflow.overall_confidence - expected_confidence) < 0.01


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_project_task(self):
        """Test project task factory function."""
        task = create_project_task(
            title="Test Task",
            description="Description",
            task_type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=["dep1", "dep2"]
        )
        
        assert task.title == "Test Task"
        assert task.description == "Description"
        assert task.type == "CREATE"
        assert task.agent_type == AgentType.PARSER
        assert task.dependencies == ["dep1", "dep2"]
    
    def test_create_project_specification(self):
        """Test project specification factory function."""
        task = create_project_task(
            title="Test Task",
            description="Description",
            task_type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        spec = create_project_specification(
            title="Test Project",
            description="Description",
            project_type="web",
            tasks=[task]
        )
        
        assert spec.title == "Test Project"
        assert spec.description == "Description"
        assert spec.project_type == "web"
        assert len(spec.tasks) == 1
        assert spec.tasks[0] == task
    
    def test_create_agent_result(self):
        """Test agent result factory function."""
        result = create_agent_result(
            success=True,
            confidence=0.8,
            output={"message": "Success"},
            agent_id="agent_001",
            execution_time=1.5,
            error=None
        )
        
        assert result.success is True
        assert result.confidence == 0.8
        assert result.output == {"message": "Success"}
        assert result.agent_id == "agent_001"
        assert result.execution_time == 1.5
        assert result.error is None


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_task_serialization(self):
        """Test task JSON serialization."""
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Test dict conversion
        task_dict = task.dict()
        assert task_dict["title"] == "Test Task"
        assert task_dict["type"] == "CREATE"
        assert task_dict["agent_type"] == "parser"
        assert task_dict["status"] == "pending"
        
        # Test JSON conversion
        task_json = task.json()
        assert isinstance(task_json, str)
        assert "Test Task" in task_json
    
    def test_task_deserialization(self):
        """Test task deserialization from dict."""
        task_data = {
            "title": "Test Task",
            "description": "Description",
            "type": "CREATE",
            "agent_type": "parser"
        }
        
        task = ProjectTask(**task_data)
        assert task.title == "Test Task"
        assert task.agent_type == AgentType.PARSER
        assert task.status == TaskStatus.PENDING
    
    def test_project_spec_serialization(self):
        """Test project specification serialization."""
        spec = ProjectSpecification(
            title="Test Project",
            description="Description",
            project_type="web"
        )
        
        spec_dict = spec.dict()
        assert spec_dict["title"] == "Test Project"
        assert spec_dict["project_type"] == "web"
        assert spec_dict["tasks"] == []
        
        # Test with tasks
        task = ProjectTask(
            title="Test Task",
            description="Description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        spec.add_task(task)
        
        spec_dict = spec.dict()
        assert len(spec_dict["tasks"]) == 1
        assert spec_dict["tasks"][0]["title"] == "Test Task"