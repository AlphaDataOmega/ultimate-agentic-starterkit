"""
Tests for Task Execution Engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import uuid

from workflows.task_executor import (
    TaskExecutionEngine,
    TaskExecutionResult,
    ParallelGroup,
    ExecutionMode,
    TaskPriority,
    execute_single_task,
    execute_task_batch,
    create_parallel_group
)
from core.models import ProjectTask, AgentType


class TestTaskExecutionResult:
    """Test cases for TaskExecutionResult."""
    
    def test_task_execution_result_initialization(self):
        """Test task execution result initialization."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)
        
        result = TaskExecutionResult(
            task_id="test-task",
            agent_id="test-agent",
            agent_type=AgentType.PARSER,
            success=True,
            confidence=0.9,
            execution_time=10.0,
            start_time=start_time,
            end_time=end_time,
            output="Test output",
            retry_count=1,
            max_retries=3
        )
        
        assert result.task_id == "test-task"
        assert result.agent_id == "test-agent"
        assert result.agent_type == AgentType.PARSER
        assert result.success is True
        assert result.confidence == 0.9
        assert result.execution_time == 10.0
        assert result.output == "Test output"
        assert result.retry_count == 1
        assert result.max_retries == 3
    
    def test_task_execution_result_to_dict(self):
        """Test task execution result serialization."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)
        
        result = TaskExecutionResult(
            task_id="test-task",
            agent_id="test-agent",
            agent_type=AgentType.PARSER,
            success=True,
            confidence=0.9,
            execution_time=10.0,
            start_time=start_time,
            end_time=end_time
        )
        
        data = result.to_dict()
        
        assert data['task_id'] == "test-task"
        assert data['agent_id'] == "test-agent"
        assert data['agent_type'] == "parser"
        assert data['success'] is True
        assert data['confidence'] == 0.9
        assert data['execution_time'] == 10.0
        assert 'start_time' in data
        assert 'end_time' in data


class TestParallelGroup:
    """Test cases for ParallelGroup."""
    
    def test_parallel_group_initialization(self):
        """Test parallel group initialization."""
        task1 = ProjectTask(
            id="task-1",
            title="Task 1",
            description="First task",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        task2 = ProjectTask(
            id="task-2",
            title="Task 2",
            description="Second task",
            type="CREATE",
            agent_type=AgentType.CODER
        )
        
        group = ParallelGroup(
            group_id="test-group",
            tasks=[task1, task2],
            priority=TaskPriority.HIGH,
            max_concurrent=2,
            timeout=300.0
        )
        
        assert group.group_id == "test-group"
        assert len(group.tasks) == 2
        assert group.priority == TaskPriority.HIGH
        assert group.max_concurrent == 2
        assert group.timeout == 300.0
    
    def test_parallel_group_auto_id(self):
        """Test parallel group automatic ID generation."""
        task1 = ProjectTask(
            id="task-1",
            title="Task 1",
            description="First task",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        group = ParallelGroup(
            group_id="",
            tasks=[task1]
        )
        
        assert group.group_id is not None
        assert len(group.group_id) > 0


class TestTaskExecutionEngine:
    """Test cases for TaskExecutionEngine."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            'max_concurrent_tasks': 3,
            'default_task_timeout': 300,
            'retry_delay_base': 1,
            'retry_delay_max': 5,
            'confidence_threshold': 0.8
        }
    
    @pytest.fixture
    def sample_task(self):
        """Sample task for testing."""
        return ProjectTask(
            id="test-task",
            title="Test Task",
            description="A test task",
            type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=[]
        )
    
    @pytest.fixture
    def execution_engine(self, mock_config):
        """Create task execution engine instance."""
        with patch('workflows.task_executor.get_logger'):
            with patch('workflows.task_executor.get_voice_alerts'):
                return TaskExecutionEngine(mock_config)
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, execution_engine, sample_task):
        """Test successful task execution."""
        # Mock agent instance
        mock_agent = Mock()
        mock_agent.agent.agent_id = "test-agent"
        mock_agent.execute_task = AsyncMock(return_value=Mock(
            success=True,
            confidence=0.9,
            output="Test output",
            error=None,
            execution_time=10.0,
            timestamp=datetime.now()
        ))
        
        result = await execution_engine.execute_task(sample_task, mock_agent)
        
        assert result.success is True
        assert result.confidence == 0.9
        assert result.task_id == "test-task"
        assert result.agent_id == "test-agent"
        assert result.output == "Test output"
        assert result.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_execute_task_low_confidence_retry(self, execution_engine, sample_task):
        """Test task execution with low confidence retry."""
        # Mock agent instance
        mock_agent = Mock()
        mock_agent.agent.agent_id = "test-agent"
        mock_agent.reset = Mock()
        
        # First call: low confidence, second call: high confidence
        mock_agent.execute_task = AsyncMock(side_effect=[
            Mock(
                success=True,
                confidence=0.5,  # Below threshold
                output="Test output",
                error=None,
                execution_time=10.0,
                timestamp=datetime.now()
            ),
            Mock(
                success=True,
                confidence=0.9,  # Above threshold
                output="Test output retry",
                error=None,
                execution_time=12.0,
                timestamp=datetime.now()
            )
        ])
        
        result = await execution_engine.execute_task(sample_task, mock_agent)
        
        assert result.success is True
        assert result.confidence == 0.9
        assert result.retry_count == 1
        assert mock_agent.execute_task.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_task_timeout(self, execution_engine, sample_task):
        """Test task execution timeout."""
        # Mock agent instance
        mock_agent = Mock()
        mock_agent.agent.agent_id = "test-agent"
        mock_agent.execute_task = AsyncMock(side_effect=asyncio.TimeoutError())
        
        result = await execution_engine.execute_task(sample_task, mock_agent)
        
        assert result.success is False
        assert "timed out" in result.error
        assert result.task_id == "test-task"
    
    @pytest.mark.asyncio
    async def test_execute_task_exception(self, execution_engine, sample_task):
        """Test task execution with exception."""
        # Mock agent instance
        mock_agent = Mock()
        mock_agent.agent.agent_id = "test-agent"
        mock_agent.execute_task = AsyncMock(side_effect=Exception("Test error"))
        
        result = await execution_engine.execute_task(sample_task, mock_agent)
        
        assert result.success is False
        assert "Test error" in result.error
        assert result.task_id == "test-task"
    
    @pytest.mark.asyncio
    async def test_execute_parallel_group_success(self, execution_engine):
        """Test successful parallel group execution."""
        task1 = ProjectTask(
            id="task-1",
            title="Task 1",
            description="First task",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        task2 = ProjectTask(
            id="task-2",
            title="Task 2",
            description="Second task",
            type="CREATE",
            agent_type=AgentType.CODER
        )
        
        group = ParallelGroup(
            group_id="test-group",
            tasks=[task1, task2],
            max_concurrent=2,
            timeout=30.0
        )
        
        # Mock successful execution
        with patch.object(execution_engine, 'execute_task') as mock_execute:
            mock_execute.side_effect = [
                TaskExecutionResult(
                    task_id="task-1",
                    agent_id="agent-1",
                    agent_type=AgentType.PARSER,
                    success=True,
                    confidence=0.9,
                    execution_time=5.0,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                ),
                TaskExecutionResult(
                    task_id="task-2",
                    agent_id="agent-2",
                    agent_type=AgentType.CODER,
                    success=True,
                    confidence=0.8,
                    execution_time=7.0,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
            ]
            
            results = await execution_engine.execute_parallel_group(group)
            
            assert len(results) == 2
            assert all(result.success for result in results)
            assert results[0].task_id == "task-1"
            assert results[1].task_id == "task-2"
    
    @pytest.mark.asyncio
    async def test_execute_parallel_group_timeout(self, execution_engine):
        """Test parallel group execution timeout."""
        task1 = ProjectTask(
            id="task-1",
            title="Task 1",
            description="First task",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        group = ParallelGroup(
            group_id="test-group",
            tasks=[task1],
            timeout=0.1  # Very short timeout
        )
        
        # Mock slow execution
        with patch.object(execution_engine, 'execute_task') as mock_execute:
            mock_execute.side_effect = lambda task: asyncio.sleep(1)  # Longer than timeout
            
            results = await execution_engine.execute_parallel_group(group)
            
            assert len(results) == 1
            assert results[0].success is False
            assert "timeout" in results[0].error
    
    @pytest.mark.asyncio
    async def test_execute_tasks_with_dependencies(self, execution_engine):
        """Test task execution with dependencies."""
        task1 = ProjectTask(
            id="task-1",
            title="Task 1",
            description="First task",
            type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=[]
        )
        
        task2 = ProjectTask(
            id="task-2",
            title="Task 2",
            description="Second task",
            type="CREATE",
            agent_type=AgentType.CODER,
            dependencies=["task-1"]
        )
        
        task3 = ProjectTask(
            id="task-3",
            title="Task 3",
            description="Third task",
            type="VALIDATE",
            agent_type=AgentType.TESTER,
            dependencies=["task-2"]
        )
        
        tasks = [task1, task2, task3]
        
        # Mock successful execution
        with patch.object(execution_engine, 'execute_task') as mock_execute:
            mock_execute.side_effect = [
                TaskExecutionResult(
                    task_id="task-1",
                    agent_id="agent-1",
                    agent_type=AgentType.PARSER,
                    success=True,
                    confidence=0.9,
                    execution_time=5.0,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                ),
                TaskExecutionResult(
                    task_id="task-2",
                    agent_id="agent-2",
                    agent_type=AgentType.CODER,
                    success=True,
                    confidence=0.8,
                    execution_time=7.0,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                ),
                TaskExecutionResult(
                    task_id="task-3",
                    agent_id="agent-3",
                    agent_type=AgentType.TESTER,
                    success=True,
                    confidence=0.95,
                    execution_time=3.0,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
            ]
            
            results = await execution_engine.execute_tasks_with_dependencies(tasks)
            
            assert len(results) == 3
            assert all(result.success for result in results)
            
            # Verify execution order
            task_ids = [result.task_id for result in results]
            assert task_ids.index("task-1") < task_ids.index("task-2")
            assert task_ids.index("task-2") < task_ids.index("task-3")
    
    @pytest.mark.asyncio
    async def test_execute_tasks_circular_dependencies(self, execution_engine):
        """Test task execution with circular dependencies."""
        task1 = ProjectTask(
            id="task-1",
            title="Task 1",
            description="First task",
            type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=["task-2"]
        )
        
        task2 = ProjectTask(
            id="task-2",
            title="Task 2",
            description="Second task",
            type="CREATE",
            agent_type=AgentType.CODER,
            dependencies=["task-1"]
        )
        
        tasks = [task1, task2]
        
        results = await execution_engine.execute_tasks_with_dependencies(tasks)
        
        # Should create failure results for circular dependencies
        assert len(results) == 2
        assert all(not result.success for result in results)
        assert all("dependency" in result.error for result in results)
    
    def test_update_metrics(self, execution_engine):
        """Test metrics update."""
        result = TaskExecutionResult(
            task_id="test-task",
            agent_id="test-agent",
            agent_type=AgentType.PARSER,
            success=True,
            confidence=0.9,
            execution_time=10.0,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        initial_count = execution_engine.metrics['total_tasks_executed']
        execution_engine._update_metrics(result)
        
        assert execution_engine.metrics['total_tasks_executed'] == initial_count + 1
        assert execution_engine.metrics['successful_tasks'] == 1
        assert execution_engine.metrics['total_execution_time'] == 10.0
    
    def test_get_execution_metrics(self, execution_engine):
        """Test getting execution metrics."""
        metrics = execution_engine.get_execution_metrics()
        
        assert 'total_tasks_executed' in metrics
        assert 'successful_tasks' in metrics
        assert 'failed_tasks' in metrics
        assert 'running_tasks' in metrics
        assert 'completed_tasks' in metrics
        assert 'total_tasks_in_history' in metrics
    
    def test_get_task_results(self, execution_engine):
        """Test getting task results."""
        # Add some test results
        result = TaskExecutionResult(
            task_id="test-task",
            agent_id="test-agent",
            agent_type=AgentType.PARSER,
            success=True,
            confidence=0.9,
            execution_time=10.0,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        execution_engine.completed_tasks["test-task"] = result
        
        task_results = execution_engine.get_task_results()
        
        assert 'completed' in task_results
        assert 'failed' in task_results
        assert 'running' in task_results
        assert 'test-task' in task_results['completed']
    
    def test_get_task_history(self, execution_engine):
        """Test getting task history."""
        # Add some test results to history
        result = TaskExecutionResult(
            task_id="test-task",
            agent_id="test-agent",
            agent_type=AgentType.PARSER,
            success=True,
            confidence=0.9,
            execution_time=10.0,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        execution_engine.task_history.append(result)
        
        history = execution_engine.get_task_history()
        
        assert len(history) == 1
        assert history[0]['task_id'] == "test-task"
    
    def test_clear_history(self, execution_engine):
        """Test clearing task history."""
        # Add some test results to history
        result = TaskExecutionResult(
            task_id="test-task",
            agent_id="test-agent",
            agent_type=AgentType.PARSER,
            success=True,
            confidence=0.9,
            execution_time=10.0,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        execution_engine.task_history.append(result)
        execution_engine.clear_history()
        
        assert len(execution_engine.task_history) == 0
    
    def test_reset_metrics(self, execution_engine):
        """Test resetting metrics."""
        # Set some metrics
        execution_engine.metrics['total_tasks_executed'] = 5
        execution_engine.metrics['successful_tasks'] = 3
        execution_engine.metrics['failed_tasks'] = 2
        
        execution_engine.reset_metrics()
        
        assert execution_engine.metrics['total_tasks_executed'] == 0
        assert execution_engine.metrics['successful_tasks'] == 0
        assert execution_engine.metrics['failed_tasks'] == 0
    
    @pytest.mark.asyncio
    async def test_shutdown(self, execution_engine):
        """Test engine shutdown."""
        await execution_engine.shutdown()
        
        assert execution_engine.is_shutdown is True
        assert len(execution_engine.running_tasks) == 0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_execute_single_task(self):
        """Test execute_single_task convenience function."""
        task = ProjectTask(
            id="test-task",
            title="Test Task",
            description="A test task",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        with patch('workflows.task_executor.TaskExecutionEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
            mock_engine.__aexit__ = AsyncMock(return_value=None)
            mock_engine.execute_task = AsyncMock(return_value=TaskExecutionResult(
                task_id="test-task",
                agent_id="test-agent",
                agent_type=AgentType.PARSER,
                success=True,
                confidence=0.9,
                execution_time=10.0,
                start_time=datetime.now(),
                end_time=datetime.now()
            ))
            
            mock_engine_class.return_value = mock_engine
            
            result = await execute_single_task(task)
            
            assert result.task_id == "test-task"
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_task_batch_parallel(self):
        """Test execute_task_batch convenience function with parallel execution."""
        task1 = ProjectTask(
            id="task-1",
            title="Task 1",
            description="First task",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        task2 = ProjectTask(
            id="task-2",
            title="Task 2",
            description="Second task",
            type="CREATE",
            agent_type=AgentType.CODER
        )
        
        tasks = [task1, task2]
        
        with patch('workflows.task_executor.TaskExecutionEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
            mock_engine.__aexit__ = AsyncMock(return_value=None)
            mock_engine.execute_parallel_group = AsyncMock(return_value=[
                TaskExecutionResult(
                    task_id="task-1",
                    agent_id="agent-1",
                    agent_type=AgentType.PARSER,
                    success=True,
                    confidence=0.9,
                    execution_time=5.0,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                ),
                TaskExecutionResult(
                    task_id="task-2",
                    agent_id="agent-2",
                    agent_type=AgentType.CODER,
                    success=True,
                    confidence=0.8,
                    execution_time=7.0,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
            ])
            
            mock_engine_class.return_value = mock_engine
            
            results = await execute_task_batch(tasks, parallel=True)
            
            assert len(results) == 2
            assert all(result.success for result in results)
    
    def test_create_parallel_group(self):
        """Test create_parallel_group convenience function."""
        task1 = ProjectTask(
            id="task-1",
            title="Task 1",
            description="First task",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        task2 = ProjectTask(
            id="task-2",
            title="Task 2",
            description="Second task",
            type="CREATE",
            agent_type=AgentType.CODER
        )
        
        tasks = [task1, task2]
        
        group = create_parallel_group(
            tasks=tasks,
            group_id="test-group",
            max_concurrent=2,
            timeout=300.0
        )
        
        assert group.group_id == "test-group"
        assert len(group.tasks) == 2
        assert group.max_concurrent == 2
        assert group.timeout == 300.0