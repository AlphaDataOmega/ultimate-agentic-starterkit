"""
Tests for AgentFactory and AgentRegistry classes.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from StarterKit.agents.factory import (
    AgentFactory, AgentRegistry, AgentInstance, AgentStatus,
    create_agent, get_or_create_agent, get_agent_factory
)
from StarterKit.agents.base_agent import BaseAgent
from StarterKit.core.models import ProjectTask, AgentResult, AgentType


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, config=None):
        super().__init__("mock", config)
        self.execute_mock = Mock()
    
    async def execute(self, task):
        return await self.execute_mock(task)


class TestAgentInstance:
    """Test cases for AgentInstance class."""
    
    def test_init(self):
        """Test AgentInstance initialization."""
        agent = MockAgent()
        config = {'test': 'value'}
        
        instance = AgentInstance(agent, AgentType.PARSER, config)
        
        assert instance.agent == agent
        assert instance.agent_type == AgentType.PARSER
        assert instance.config == config
        assert instance.status == AgentStatus.READY
        assert instance.created_at is not None
        assert instance.last_used is None
        assert instance.error_message is None
        assert instance.is_processing is False
        assert instance.stats['total_tasks'] == 0
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self):
        """Test successful task execution."""
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        # Mock successful result
        result = AgentResult(
            success=True,
            confidence=0.8,
            output="test output",
            execution_time=1.0,
            agent_id=agent.agent_id
        )
        agent.execute_mock.return_value = result
        
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Execute task
        actual_result = await instance.execute_task(task)
        
        assert actual_result.success is True
        assert actual_result.confidence == 0.8
        assert instance.status == AgentStatus.READY
        assert instance.stats['total_tasks'] == 1
        assert instance.stats['successful_tasks'] == 1
        assert instance.stats['failed_tasks'] == 0
        assert instance.last_used is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_not_ready(self):
        """Test task execution when agent is not ready."""
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        instance.status = AgentStatus.BUSY
        
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        with pytest.raises(RuntimeError, match="not ready"):
            await instance.execute_task(task)
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self):
        """Test task execution failure."""
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        # Mock failure result
        result = AgentResult(
            success=False,
            confidence=0.0,
            output=None,
            error="Test error",
            execution_time=1.0,
            agent_id=agent.agent_id
        )
        agent.execute_mock.return_value = result
        
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Execute task
        actual_result = await instance.execute_task(task)
        
        assert actual_result.success is False
        assert instance.status == AgentStatus.READY
        assert instance.stats['total_tasks'] == 1
        assert instance.stats['successful_tasks'] == 0
        assert instance.stats['failed_tasks'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_task_exception(self):
        """Test task execution with exception."""
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        # Mock exception
        agent.execute_mock.side_effect = Exception("Test exception")
        
        task = ProjectTask(
            title="Test Task",
            description="Test description",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        # Execute task
        actual_result = await instance.execute_task(task)
        
        assert actual_result.success is False
        assert "Test exception" in actual_result.error
        assert instance.status == AgentStatus.ERROR
        assert instance.error_message == "Test exception"
        assert instance.stats['failed_tasks'] == 1
    
    def test_get_info(self):
        """Test getting instance information."""
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {'test': 'value'})
        
        info = instance.get_info()
        
        assert info['agent_id'] == agent.agent_id
        assert info['agent_type'] == AgentType.PARSER.value
        assert info['status'] == AgentStatus.READY.value
        assert info['created_at'] is not None
        assert info['last_used'] is None
        assert info['error_message'] is None
        assert info['is_processing'] is False
        assert info['stats']['total_tasks'] == 0
        assert info['config']['test'] == 'value'
    
    def test_reset(self):
        """Test instance reset."""
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        # Set some error state
        instance.status = AgentStatus.ERROR
        instance.error_message = "Test error"
        instance.is_processing = True
        
        # Reset
        instance.reset()
        
        assert instance.status == AgentStatus.READY
        assert instance.error_message is None
        assert instance.is_processing is False
    
    def test_stop(self):
        """Test instance stop."""
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        instance.stop()
        
        assert instance.status == AgentStatus.STOPPED
        assert instance.is_processing is False


class TestAgentRegistry:
    """Test cases for AgentRegistry class."""
    
    def test_init(self):
        """Test AgentRegistry initialization."""
        registry = AgentRegistry()
        
        assert len(registry.agents) == 0
        assert len(registry.agents_by_type) == len(AgentType)
        assert registry.max_agents_per_type == 3
    
    def test_register_agent(self):
        """Test agent registration."""
        registry = AgentRegistry()
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        success = registry.register(instance)
        
        assert success is True
        assert agent.agent_id in registry.agents
        assert instance in registry.agents_by_type[AgentType.PARSER]
    
    def test_register_duplicate_agent(self):
        """Test registering duplicate agent."""
        registry = AgentRegistry()
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        # Register first time
        success1 = registry.register(instance)
        assert success1 is True
        
        # Register second time (should fail)
        success2 = registry.register(instance)
        assert success2 is False
    
    def test_register_too_many_agents(self):
        """Test registering too many agents of same type."""
        registry = AgentRegistry()
        registry.max_agents_per_type = 2
        
        # Register maximum agents
        for i in range(2):
            agent = MockAgent()
            instance = AgentInstance(agent, AgentType.PARSER, {})
            success = registry.register(instance)
            assert success is True
        
        # Try to register one more (should fail)
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        success = registry.register(instance)
        assert success is False
    
    def test_unregister_agent(self):
        """Test agent unregistration."""
        registry = AgentRegistry()
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        # Register and then unregister
        registry.register(instance)
        success = registry.unregister(agent.agent_id)
        
        assert success is True
        assert agent.agent_id not in registry.agents
        assert instance not in registry.agents_by_type[AgentType.PARSER]
        assert instance.status == AgentStatus.STOPPED
    
    def test_unregister_nonexistent_agent(self):
        """Test unregistering non-existent agent."""
        registry = AgentRegistry()
        
        success = registry.unregister("nonexistent_id")
        assert success is False
    
    def test_get_agent(self):
        """Test getting agent by ID."""
        registry = AgentRegistry()
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        registry.register(instance)
        
        retrieved = registry.get_agent(agent.agent_id)
        assert retrieved == instance
        
        # Test non-existent agent
        assert registry.get_agent("nonexistent") is None
    
    def test_get_available_agent(self):
        """Test getting available agent."""
        registry = AgentRegistry()
        agent = MockAgent()
        instance = AgentInstance(agent, AgentType.PARSER, {})
        
        registry.register(instance)
        
        # Should return the ready agent
        available = registry.get_available_agent(AgentType.PARSER)
        assert available == instance
        
        # Set agent to busy
        instance.status = AgentStatus.BUSY
        available = registry.get_available_agent(AgentType.PARSER)
        assert available is None
        
        # Test non-existent type
        available = registry.get_available_agent(AgentType.CODER)
        assert available is None
    
    def test_get_available_agent_least_recently_used(self):
        """Test getting least recently used available agent."""
        registry = AgentRegistry()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = MockAgent()
            instance = AgentInstance(agent, AgentType.PARSER, {})
            agents.append(instance)
            registry.register(instance)
        
        # Set different last used times
        now = datetime.now()
        agents[0].last_used = now - timedelta(hours=2)
        agents[1].last_used = now - timedelta(hours=1)
        agents[2].last_used = now - timedelta(minutes=30)
        
        # Should return the least recently used
        available = registry.get_available_agent(AgentType.PARSER)
        assert available == agents[0]
    
    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        registry = AgentRegistry()
        
        # Add some agents
        for agent_type in [AgentType.PARSER, AgentType.CODER]:
            agent = MockAgent()
            instance = AgentInstance(agent, agent_type, {})
            instance.stats['total_tasks'] = 5
            instance.stats['successful_tasks'] = 3
            instance.stats['failed_tasks'] = 2
            registry.register(instance)
        
        stats = registry.get_registry_stats()
        
        assert stats['total_agents'] == 2
        assert stats['agents_by_type'][AgentType.PARSER.value] == 1
        assert stats['agents_by_type'][AgentType.CODER.value] == 1
        assert stats['agents_by_status'][AgentStatus.READY.value] == 2
        assert stats['total_tasks_executed'] == 10
        assert stats['total_successful_tasks'] == 6
        assert stats['total_failed_tasks'] == 4
    
    def test_cleanup(self):
        """Test registry cleanup."""
        registry = AgentRegistry()
        registry.cleanup_interval = 0  # Force immediate cleanup
        
        # Add stopped agent
        agent1 = MockAgent()
        instance1 = AgentInstance(agent1, AgentType.PARSER, {})
        instance1.status = AgentStatus.STOPPED
        registry.register(instance1)
        
        # Add errored agent with old last_used
        agent2 = MockAgent()
        instance2 = AgentInstance(agent2, AgentType.CODER, {})
        instance2.status = AgentStatus.ERROR
        instance2.last_used = datetime.now() - timedelta(hours=1)  # Old enough
        registry.register(instance2)
        
        # Add healthy agent
        agent3 = MockAgent()
        instance3 = AgentInstance(agent3, AgentType.TESTER, {})
        registry.register(instance3)
        
        # Cleanup should remove first two agents
        registry.cleanup()
        
        assert len(registry.agents) == 1
        assert agent3.agent_id in registry.agents


class TestAgentFactory:
    """Test cases for AgentFactory class."""
    
    def test_init(self):
        """Test AgentFactory initialization."""
        factory = AgentFactory()
        
        assert factory.config == {}
        assert factory.registry is not None
        assert len(factory.default_configs) == len(AgentType)
        assert len(factory.agent_classes) == len(AgentType)
    
    def test_init_with_config(self):
        """Test AgentFactory initialization with config."""
        config = {'test': 'value'}
        factory = AgentFactory(config)
        
        assert factory.config == config
    
    @patch('StarterKit.agents.factory.ParserAgent')
    def test_create_agent(self, mock_parser_class):
        """Test agent creation."""
        factory = AgentFactory()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_id"
        mock_parser_class.return_value = mock_agent
        
        # Create agent
        instance = factory.create_agent(AgentType.PARSER, register=False)
        
        assert instance.agent == mock_agent
        assert instance.agent_type == AgentType.PARSER
        mock_parser_class.assert_called_once()
    
    def test_create_agent_unknown_type(self):
        """Test creating agent with unknown type."""
        factory = AgentFactory()
        
        with pytest.raises(ValueError, match="Unknown agent type"):
            factory.create_agent("unknown_type")
    
    @patch('StarterKit.agents.factory.ParserAgent')
    def test_create_agent_with_config_merge(self, mock_parser_class):
        """Test agent creation with config merging."""
        factory = AgentFactory({'global': 'value'})
        
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_id"
        mock_parser_class.return_value = mock_agent
        
        # Create agent with additional config
        instance = factory.create_agent(
            AgentType.PARSER, 
            config={'specific': 'value'},
            register=False
        )
        
        # Check that config was merged
        call_args = mock_parser_class.call_args
        config_arg = call_args[1]['config']
        assert 'global' in config_arg
        assert 'specific' in config_arg
        assert config_arg['global'] == 'value'
        assert config_arg['specific'] == 'value'
    
    @patch('StarterKit.agents.factory.ParserAgent')
    def test_get_or_create_agent_existing(self, mock_parser_class):
        """Test getting existing agent."""
        factory = AgentFactory()
        
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_id"
        mock_parser_class.return_value = mock_agent
        
        # Create and register agent
        instance1 = factory.create_agent(AgentType.PARSER, register=True)
        
        # Get existing agent
        instance2 = factory.get_or_create_agent(AgentType.PARSER)
        
        assert instance1 == instance2
        assert mock_parser_class.call_count == 1  # Only called once
    
    @patch('StarterKit.agents.factory.ParserAgent')
    def test_get_or_create_agent_new(self, mock_parser_class):
        """Test creating new agent when none available."""
        factory = AgentFactory()
        
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_id"
        mock_parser_class.return_value = mock_agent
        
        # Should create new agent
        instance = factory.get_or_create_agent(AgentType.PARSER)
        
        assert instance.agent_type == AgentType.PARSER
        mock_parser_class.assert_called_once()
    
    @patch('StarterKit.agents.factory.ParserAgent')
    def test_create_agent_pool(self, mock_parser_class):
        """Test creating agent pool."""
        factory = AgentFactory()
        
        # Mock agents
        mock_agents = [Mock() for _ in range(3)]
        for i, agent in enumerate(mock_agents):
            agent.agent_id = f"test_agent_{i}"
        mock_parser_class.side_effect = mock_agents
        
        # Create pool
        pool = factory.create_agent_pool(AgentType.PARSER, pool_size=3)
        
        assert len(pool) == 3
        assert all(instance.agent_type == AgentType.PARSER for instance in pool)
        assert mock_parser_class.call_count == 3
    
    def test_get_registry(self):
        """Test getting registry."""
        factory = AgentFactory()
        registry = factory.get_registry()
        
        assert registry == factory.registry
    
    def test_shutdown(self):
        """Test factory shutdown."""
        factory = AgentFactory()
        
        # Add some agents
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_id"
        instance = AgentInstance(mock_agent, AgentType.PARSER, {})
        factory.registry.register(instance)
        
        # Shutdown
        factory.shutdown()
        
        assert len(factory.registry.agents) == 0
        assert instance.status == AgentStatus.STOPPED


class TestFactoryFunctions:
    """Test cases for factory convenience functions."""
    
    @patch('StarterKit.agents.factory.get_agent_factory')
    def test_create_agent_function(self, mock_get_factory):
        """Test create_agent convenience function."""
        mock_factory = Mock()
        mock_instance = Mock()
        mock_factory.create_agent.return_value = mock_instance
        mock_get_factory.return_value = mock_factory
        
        result = create_agent(AgentType.PARSER, {'test': 'config'}, register=False)
        
        assert result == mock_instance
        mock_factory.create_agent.assert_called_once_with(
            AgentType.PARSER, {'test': 'config'}, False
        )
    
    @patch('StarterKit.agents.factory.get_agent_factory')
    def test_create_agent_function_string_type(self, mock_get_factory):
        """Test create_agent function with string type."""
        mock_factory = Mock()
        mock_instance = Mock()
        mock_factory.create_agent.return_value = mock_instance
        mock_get_factory.return_value = mock_factory
        
        result = create_agent("parser")
        
        mock_factory.create_agent.assert_called_once_with(
            AgentType.PARSER, None, True
        )
    
    @patch('StarterKit.agents.factory.get_agent_factory')
    def test_get_or_create_agent_function(self, mock_get_factory):
        """Test get_or_create_agent convenience function."""
        mock_factory = Mock()
        mock_instance = Mock()
        mock_factory.get_or_create_agent.return_value = mock_instance
        mock_get_factory.return_value = mock_factory
        
        result = get_or_create_agent(AgentType.CODER)
        
        assert result == mock_instance
        mock_factory.get_or_create_agent.assert_called_once_with(
            AgentType.CODER, None
        )