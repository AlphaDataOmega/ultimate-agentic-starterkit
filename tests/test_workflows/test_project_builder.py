"""
Tests for LangGraph Workflow Manager.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from workflows.project_builder import (
    LangGraphWorkflowManager,
    ProjectBuilderState,
    WorkflowStatus,
    WorkflowMetrics,
    WorkflowError,
    WorkflowTimeoutError
)
from core.models import ProjectSpecification, ProjectTask, AgentType


class TestWorkflowMetrics:
    """Test cases for WorkflowMetrics."""
    
    def test_workflow_metrics_initialization(self):
        """Test workflow metrics initialization."""
        metrics = WorkflowMetrics()
        
        assert metrics.total_tasks == 0
        assert metrics.completed_tasks == 0
        assert metrics.failed_tasks == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.start_time is not None
        assert metrics.end_time is None
    
    def test_workflow_metrics_duration(self):
        """Test workflow metrics duration calculation."""
        metrics = WorkflowMetrics()
        metrics.end_time = datetime.now()
        
        duration = metrics.duration
        assert duration >= 0
    
    def test_workflow_metrics_success_rate(self):
        """Test workflow metrics success rate calculation."""
        metrics = WorkflowMetrics()
        
        # No tasks
        assert metrics.success_rate == 0.0
        
        # Some tasks
        metrics.total_tasks = 10
        metrics.completed_tasks = 8
        assert metrics.success_rate == 0.8
    
    def test_workflow_metrics_to_dict(self):
        """Test workflow metrics serialization."""
        metrics = WorkflowMetrics()
        metrics.total_tasks = 5
        metrics.completed_tasks = 3
        
        data = metrics.to_dict()
        
        assert data['total_tasks'] == 5
        assert data['completed_tasks'] == 3
        assert data['success_rate'] == 0.6
        assert 'start_time' in data
        assert 'duration' in data


class TestLangGraphWorkflowManager:
    """Test cases for LangGraphWorkflowManager."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            'max_retries': 3,
            'task_timeout': 600,
            'workflow_timeout': 7200
        }
    
    @pytest.fixture
    def sample_project_spec(self):
        """Sample project specification."""
        task1 = ProjectTask(
            id="task-1",
            title="Parse Requirements",
            description="Parse project requirements",
            type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=[]
        )
        
        task2 = ProjectTask(
            id="task-2",
            title="Generate Code",
            description="Generate application code",
            type="CREATE",
            agent_type=AgentType.CODER,
            dependencies=["task-1"]
        )
        
        return {
            "title": "Test Project",
            "description": "A test project",
            "project_type": "web",
            "tasks": [task1.dict(), task2.dict()],
            "requirements": {"framework": "react"},
            "validation_criteria": {"coverage": 0.8}
        }
    
    @pytest.fixture
    def workflow_manager(self, mock_config):
        """Create workflow manager instance."""
        with patch('workflows.project_builder.get_logger'):
            with patch('workflows.project_builder.get_voice_alerts'):
                with patch('workflows.project_builder.O3Orchestrator'):
                    return LangGraphWorkflowManager(mock_config)
    
    @pytest.mark.asyncio
    async def test_initialize_workflow(self, workflow_manager, sample_project_spec):
        """Test workflow initialization."""
        initial_state = {
            "project_spec": sample_project_spec,
            "execution_plan": {},
            "current_task": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "agent_results": [],
            "overall_confidence": 0.0,
            "workflow_status": WorkflowStatus.INITIALIZING.value,
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.create_execution_plan = AsyncMock(return_value={
            "execution_order": [
                {"task_id": "task-1", "dependencies": []},
                {"task_id": "task-2", "dependencies": ["task-1"]}
            ],
            "parallel_groups": {},
            "critical_path": ["task-1", "task-2"]
        })
        
        workflow_manager.orchestrator = mock_orchestrator
        
        result_state = await workflow_manager._initialize_workflow(initial_state)
        
        assert result_state["workflow_status"] == WorkflowStatus.RUNNING.value
        assert "execution_plan" in result_state
        assert "metrics" in result_state
        assert result_state["retry_count"] == 0
    
    @pytest.mark.asyncio
    async def test_select_next_task(self, workflow_manager, sample_project_spec):
        """Test next task selection."""
        state = {
            "project_spec": sample_project_spec,
            "execution_plan": {
                "execution_order": [
                    {"task_id": "task-1", "dependencies": []},
                    {"task_id": "task-2", "dependencies": ["task-1"]}
                ]
            },
            "completed_tasks": [],
            "failed_tasks": []
        }
        
        result_state = await workflow_manager._select_next_task(state)
        
        assert result_state["current_task"] is not None
        assert result_state["current_task"]["id"] == "task-1"
    
    @pytest.mark.asyncio
    async def test_select_next_task_with_dependencies(self, workflow_manager, sample_project_spec):
        """Test next task selection with completed dependencies."""
        state = {
            "project_spec": sample_project_spec,
            "execution_plan": {
                "execution_order": [
                    {"task_id": "task-1", "dependencies": []},
                    {"task_id": "task-2", "dependencies": ["task-1"]}
                ]
            },
            "completed_tasks": ["task-1"],
            "failed_tasks": []
        }
        
        result_state = await workflow_manager._select_next_task(state)
        
        assert result_state["current_task"] is not None
        assert result_state["current_task"]["id"] == "task-2"
    
    @pytest.mark.asyncio
    async def test_select_next_task_no_ready_tasks(self, workflow_manager, sample_project_spec):
        """Test next task selection when no tasks are ready."""
        state = {
            "project_spec": sample_project_spec,
            "execution_plan": {
                "execution_order": [
                    {"task_id": "task-1", "dependencies": []},
                    {"task_id": "task-2", "dependencies": ["task-1"]}
                ]
            },
            "completed_tasks": ["task-1"],
            "failed_tasks": ["task-2"]
        }
        
        result_state = await workflow_manager._select_next_task(state)
        
        assert result_state["current_task"] is None
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, workflow_manager):
        """Test successful task execution."""
        current_task = {
            "id": "task-1",
            "title": "Test Task",
            "description": "A test task",
            "type": "CREATE",
            "agent_type": "parser",
            "dependencies": []
        }
        
        state = {
            "current_task": current_task,
            "agent_results": []
        }
        
        # Mock agent instance
        mock_agent = Mock()
        mock_agent.agent.agent_id = "test-agent"
        mock_agent.execute_task = AsyncMock(return_value=Mock(
            success=True,
            confidence=0.9,
            error=None,
            execution_time=10.0,
            timestamp=datetime.now(),
            output="Test output"
        ))
        
        with patch('workflows.project_builder.get_or_create_agent', return_value=mock_agent):
            result_state = await workflow_manager._execute_task(state)
            
            assert len(result_state["agent_results"]) == 1
            assert result_state["agent_results"][0]["success"] is True
            assert result_state["agent_results"][0]["confidence"] == 0.9
            assert result_state["agent_results"][0]["task_id"] == "task-1"
    
    @pytest.mark.asyncio
    async def test_execute_task_timeout(self, workflow_manager):
        """Test task execution timeout."""
        current_task = {
            "id": "task-1",
            "title": "Test Task",
            "description": "A test task",
            "type": "CREATE",
            "agent_type": "parser",
            "dependencies": []
        }
        
        state = {
            "current_task": current_task,
            "agent_results": []
        }
        
        # Mock agent instance with timeout
        mock_agent = Mock()
        mock_agent.agent.agent_id = "test-agent"
        mock_agent.execute_task = AsyncMock(side_effect=asyncio.TimeoutError())
        
        with patch('workflows.project_builder.get_or_create_agent', return_value=mock_agent):
            result_state = await workflow_manager._execute_task(state)
            
            assert len(result_state["agent_results"]) == 1
            assert result_state["agent_results"][0]["success"] is False
            assert "timed out" in result_state["agent_results"][0]["error"]
    
    @pytest.mark.asyncio
    async def test_validate_result_success(self, workflow_manager):
        """Test successful result validation."""
        state = {
            "current_task": {"id": "task-1", "confidence_threshold": 0.8},
            "agent_results": [
                {"success": True, "confidence": 0.9}
            ],
            "retry_count": 0,
            "max_retries": 3
        }
        
        result_state = await workflow_manager._validate_result(state)
        
        assert result_state["validation_status"] == "success"
    
    @pytest.mark.asyncio
    async def test_validate_result_retry(self, workflow_manager):
        """Test result validation requiring retry."""
        state = {
            "current_task": {"id": "task-1", "confidence_threshold": 0.8},
            "agent_results": [
                {"success": True, "confidence": 0.5}
            ],
            "retry_count": 1,
            "max_retries": 3
        }
        
        result_state = await workflow_manager._validate_result(state)
        
        assert result_state["validation_status"] == "retry"
        assert result_state["retry_count"] == 2
    
    @pytest.mark.asyncio
    async def test_validate_result_failure(self, workflow_manager):
        """Test result validation failure."""
        state = {
            "current_task": {"id": "task-1", "confidence_threshold": 0.8},
            "agent_results": [
                {"success": False, "confidence": 0.5}
            ],
            "retry_count": 3,
            "max_retries": 3
        }
        
        result_state = await workflow_manager._validate_result(state)
        
        assert result_state["validation_status"] == "failure"
    
    @pytest.mark.asyncio
    async def test_update_state(self, workflow_manager):
        """Test state update after task completion."""
        state = {
            "current_task": {"id": "task-1", "title": "Test Task"},
            "completed_tasks": [],
            "agent_results": [
                {"success": True, "confidence": 0.9}
            ],
            "overall_confidence": 0.0,
            "metrics": {"completed_tasks": 0}
        }
        
        result_state = await workflow_manager._update_state(state)
        
        assert "task-1" in result_state["completed_tasks"]
        assert result_state["overall_confidence"] == 0.9
        assert result_state["metrics"]["completed_tasks"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_failure(self, workflow_manager):
        """Test failure handling."""
        state = {
            "current_task": {"id": "task-1", "title": "Test Task"},
            "failed_tasks": [],
            "metrics": {"failed_tasks": 0}
        }
        
        result_state = await workflow_manager._handle_failure(state)
        
        assert "task-1" in result_state["failed_tasks"]
        assert result_state["metrics"]["failed_tasks"] == 1
    
    def test_is_workflow_complete_success(self, workflow_manager):
        """Test workflow completion check - success case."""
        state = {
            "project_spec": {"tasks": [{"id": "task-1"}, {"id": "task-2"}]},
            "completed_tasks": ["task-1", "task-2"],
            "failed_tasks": []
        }
        
        result = workflow_manager._is_workflow_complete(state)
        
        assert result == "complete"
    
    def test_is_workflow_complete_failure(self, workflow_manager):
        """Test workflow completion check - failure case."""
        state = {
            "project_spec": {"tasks": [{"id": "task-1"}, {"id": "task-2"}]},
            "completed_tasks": ["task-1"],
            "failed_tasks": ["task-2"]
        }
        
        result = workflow_manager._is_workflow_complete(state)
        
        assert result == "failed"
    
    def test_is_workflow_complete_continue(self, workflow_manager):
        """Test workflow completion check - continue case."""
        state = {
            "project_spec": {"tasks": [{"id": "task-1"}, {"id": "task-2"}]},
            "completed_tasks": ["task-1"],
            "failed_tasks": []
        }
        
        result = workflow_manager._is_workflow_complete(state)
        
        assert result == "continue"
    
    @pytest.mark.asyncio
    async def test_finalize_workflow_success(self, workflow_manager):
        """Test workflow finalization - success case."""
        state = {
            "workflow_status": WorkflowStatus.RUNNING.value,
            "metrics": {}
        }
        
        result_state = await workflow_manager._finalize_workflow(state)
        
        assert result_state["workflow_status"] == WorkflowStatus.COMPLETED.value
        assert "end_time" in result_state["metrics"]
    
    @pytest.mark.asyncio
    async def test_finalize_workflow_failure(self, workflow_manager):
        """Test workflow finalization - failure case."""
        state = {
            "workflow_status": WorkflowStatus.FAILED.value,
            "metrics": {}
        }
        
        result_state = await workflow_manager._finalize_workflow(state)
        
        assert result_state["workflow_status"] == WorkflowStatus.FAILED.value
    
    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, workflow_manager, sample_project_spec):
        """Test full workflow execution - success case."""
        # Mock the workflow execution
        workflow_manager.graph = None  # Force fallback implementation
        
        with patch.object(workflow_manager, '_execute_workflow_fallback') as mock_fallback:
            mock_fallback.return_value = {
                "workflow_status": WorkflowStatus.COMPLETED.value,
                "completed_tasks": ["task-1", "task-2"],
                "failed_tasks": [],
                "overall_confidence": 0.85
            }
            
            result = await workflow_manager.execute_workflow(sample_project_spec)
            
            assert result["workflow_status"] == WorkflowStatus.COMPLETED.value
            assert len(result["completed_tasks"]) == 2
            assert result["overall_confidence"] == 0.85
    
    @pytest.mark.asyncio
    async def test_execute_workflow_timeout(self, workflow_manager, sample_project_spec):
        """Test workflow execution timeout."""
        workflow_manager.workflow_timeout = 0.1  # Very short timeout
        
        with patch.object(workflow_manager, '_execute_workflow_fallback') as mock_fallback:
            mock_fallback.side_effect = asyncio.sleep(1)  # Longer than timeout
            
            with pytest.raises(WorkflowTimeoutError):
                await workflow_manager.execute_workflow(sample_project_spec)
    
    def test_pause_workflow(self, workflow_manager):
        """Test workflow pause."""
        workflow_manager.pause_workflow()
        
        assert workflow_manager.pause_requested is True
    
    def test_resume_workflow(self, workflow_manager):
        """Test workflow resume."""
        workflow_manager.pause_requested = True
        workflow_manager.resume_workflow()
        
        assert workflow_manager.pause_requested is False
    
    def test_stop_workflow(self, workflow_manager):
        """Test workflow stop."""
        workflow_manager.stop_workflow()
        
        assert workflow_manager.should_stop is True
    
    def test_get_current_state(self, workflow_manager):
        """Test getting current state."""
        test_state = {"test": "state"}
        workflow_manager.current_state = test_state
        
        assert workflow_manager.get_current_state() == test_state
    
    def test_get_state_history(self, workflow_manager):
        """Test getting state history."""
        test_history = [{"state": 1}, {"state": 2}]
        workflow_manager.state_history = test_history
        
        assert workflow_manager.get_state_history() == test_history
    
    def test_get_workflow_metrics(self, workflow_manager):
        """Test getting workflow metrics."""
        test_metrics = {"completed_tasks": 5, "failed_tasks": 1}
        workflow_manager.current_state = {"metrics": test_metrics}
        
        assert workflow_manager.get_workflow_metrics() == test_metrics
    
    def test_clear_history(self, workflow_manager):
        """Test clearing state history."""
        workflow_manager.state_history = [{"state": 1}, {"state": 2}]
        workflow_manager.clear_history()
        
        assert len(workflow_manager.state_history) == 0