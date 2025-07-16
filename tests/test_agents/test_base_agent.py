"""
Tests for BaseAgent class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from StarterKit.agents.base_agent import BaseAgent
from StarterKit.core.models import ProjectTask, AgentResult, AgentType, TaskStatus


class TestBaseAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def __init__(self, config=None):
        super().__init__("test", config)
        self.execute_mock = AsyncMock()
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """Mock execute method for testing."""
        return await self.execute_mock(task)


class TestBaseAgentClass:
    """Test cases for BaseAgent class."""
    
    def test_init_default_config(self):
        """Test BaseAgent initialization with default config."""
        agent = TestBaseAgent()
        
        assert agent.agent_type == "test"
        assert agent.agent_id.startswith("test_")
        assert agent.max_retries == 3
        assert agent.confidence_threshold == 0.7
        assert agent.retry_delay == 1.0
        assert agent.timeout == 300
        assert agent.execution_count == 0
        assert agent.success_count == 0
        assert agent.failure_count == 0
    
    def test_init_custom_config(self):
        """Test BaseAgent initialization with custom config."""
        config = {
            'max_retries': 5,
            'confidence_threshold': 0.8,
            'retry_delay': 2.0,
            'timeout': 600
        }
        agent = TestBaseAgent(config)
        
        assert agent.max_retries == 5
        assert agent.confidence_threshold == 0.8
        assert agent.retry_delay == 2.0
        assert agent.timeout == 600
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Test successful task execution with retry logic."""
        agent = TestBaseAgent()
        
        # Mock successful execution
        result = AgentResult(
            success=True,
            confidence=0.8,
            output="test output",
            execution_time=1.0,
            agent_id=agent.agent_id
        )
        agent.execute_mock.return_value = result
        
        # Create test task
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Execute task
        final_result = await agent.execute_with_retry(task)
        
        assert final_result.success is True
        assert final_result.confidence == 0.8
        assert agent.success_count == 1
        assert agent.failure_count == 0
        assert agent.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_low_confidence(self):
        """Test task execution with low confidence requiring retry."""
        agent = TestBaseAgent()
        
        # Mock low confidence result, then successful result
        low_confidence_result = AgentResult(
            success=True,
            confidence=0.5,  # Below threshold
            output="test output",
            execution_time=1.0,
            agent_id=agent.agent_id
        )
        
        high_confidence_result = AgentResult(
            success=True,
            confidence=0.8,
            output="test output",
            execution_time=1.0,
            agent_id=agent.agent_id
        )
        
        agent.execute_mock.side_effect = [low_confidence_result, high_confidence_result]
        
        # Create test task
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Execute task
        final_result = await agent.execute_with_retry(task)
        
        assert final_result.success is True
        assert final_result.confidence == 0.8
        assert agent.success_count == 1
        assert agent.failure_count == 0
        assert agent.execute_mock.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure(self):
        """Test task execution that fails all retries."""
        agent = TestBaseAgent()
        
        # Mock failure result
        failure_result = AgentResult(
            success=False,
            confidence=0.0,
            output=None,
            error="Test error",
            execution_time=1.0,
            agent_id=agent.agent_id
        )
        agent.execute_mock.return_value = failure_result
        
        # Create test task
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Execute task
        final_result = await agent.execute_with_retry(task)
        
        assert final_result.success is False
        assert final_result.confidence == 0.0
        assert agent.success_count == 0
        assert agent.failure_count == 1
        assert agent.execute_mock.call_count == 3  # All retries
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_timeout(self):
        """Test task execution that times out."""
        agent = TestBaseAgent({'timeout': 0.1})  # Very short timeout
        
        # Mock slow execution
        async def slow_execute(task):
            await asyncio.sleep(0.2)  # Longer than timeout
            return AgentResult(
                success=True,
                confidence=0.8,
                output="test output",
                execution_time=0.2,
                agent_id=agent.agent_id
            )
        
        agent.execute_mock.side_effect = slow_execute
        
        # Create test task
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Execute task
        final_result = await agent.execute_with_retry(task)
        
        assert final_result.success is False
        assert "timed out" in final_result.error
        assert agent.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_exception(self):
        """Test task execution that raises exception."""
        agent = TestBaseAgent()
        
        # Mock exception
        agent.execute_mock.side_effect = Exception("Test exception")
        
        # Create test task
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Execute task
        final_result = await agent.execute_with_retry(task)
        
        assert final_result.success is False
        assert "Test exception" in final_result.error
        assert agent.failure_count == 1
    
    def test_get_agent_stats(self):
        """Test agent statistics retrieval."""
        agent = TestBaseAgent()
        
        # Simulate some execution history
        agent.execution_count = 5
        agent.success_count = 3
        agent.failure_count = 2
        agent.total_execution_time = 10.0
        agent.last_execution_time = 2.0
        
        stats = agent.get_agent_stats()
        
        assert stats['agent_id'] == agent.agent_id
        assert stats['agent_type'] == "test"
        assert stats['total_executions'] == 5
        assert stats['success_count'] == 3
        assert stats['failure_count'] == 2
        assert stats['success_rate'] == 0.6
        assert stats['total_execution_time'] == 10.0
        assert stats['average_execution_time'] == 2.0
        assert stats['last_execution_time'] == 2.0
    
    def test_reset_stats(self):
        """Test agent statistics reset."""
        agent = TestBaseAgent()
        
        # Set some stats
        agent.execution_count = 5
        agent.success_count = 3
        agent.failure_count = 2
        agent.total_execution_time = 10.0
        agent.last_execution_time = 2.0
        
        # Reset
        agent.reset_stats()
        
        assert agent.execution_count == 0
        assert agent.success_count == 0
        assert agent.failure_count == 0
        assert agent.total_execution_time == 0.0
        assert agent.last_execution_time is None
    
    def test_update_config(self):
        """Test agent configuration update."""
        agent = TestBaseAgent()
        
        original_retries = agent.max_retries
        original_threshold = agent.confidence_threshold
        
        # Update config
        new_config = {
            'max_retries': 10,
            'confidence_threshold': 0.9,
            'new_setting': 'test_value'
        }
        agent.update_config(new_config)
        
        assert agent.max_retries == 10
        assert agent.confidence_threshold == 0.9
        assert agent.config['new_setting'] == 'test_value'
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        agent = TestBaseAgent()
        
        # Test with no indicators
        confidence = agent._calculate_confidence({})
        assert 0.0 <= confidence <= 1.0
        
        # Test with error indicators
        confidence = agent._calculate_confidence({'error_count': 2})
        assert confidence < 0.5
        
        # Test with completion status
        confidence = agent._calculate_confidence({'completion_status': 'complete'})
        assert confidence > 0.5
        
        # Test with validation
        confidence = agent._calculate_confidence({'validation_passed': True})
        assert confidence > 0.5
    
    def test_validate_task_valid(self):
        """Test task validation with valid task."""
        agent = TestBaseAgent()
        
        task = ProjectTask(
            title="Valid Task",
            description="Valid description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        assert agent._validate_task(task) is True
    
    def test_validate_task_invalid(self):
        """Test task validation with invalid tasks."""
        agent = TestBaseAgent()
        
        # Test with None task
        assert agent._validate_task(None) is False
        
        # Test with empty title
        task = ProjectTask(
            title="",
            description="Valid description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        assert agent._validate_task(task) is False
        
        # Test with empty description
        task = ProjectTask(
            title="Valid Title",
            description="",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        assert agent._validate_task(task) is False
        
        # Test with completed status
        task = ProjectTask(
            title="Valid Title",
            description="Valid description",
            type="CREATE",
            agent_type=AgentType.PARSER,
            status=TaskStatus.COMPLETED
        )
        assert agent._validate_task(task) is False
    
    def test_string_representation(self):
        """Test string representation of agent."""
        agent = TestBaseAgent()
        
        str_repr = str(agent)
        assert "TestBaseAgent" in str_repr
        assert agent.agent_id in str_repr
        assert "test" in str_repr
        
        repr_str = repr(agent)
        assert str_repr == repr_str