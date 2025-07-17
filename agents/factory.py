"""
Agent Factory and Registry for the Ultimate Agentic StarterKit.

This module provides factory patterns and registry management for creating
and managing AI agent instances with lifecycle management and configuration.
"""

import asyncio
import threading
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import uuid
from enum import Enum

from agents.base_agent import BaseAgent
from agents.parser_agent import ParserAgent
from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent
from agents.advisor_agent import AdvisorAgent
from agents.project_manager_agent import ProjectManagerAgent
from agents.research_agent import ResearchAgent
from agents.documentation_agent import DocumentationAgent
from agents.testing_validation_agent import TestingValidationAgent
from agents.visual_testing_agent import VisualTestingAgent
from agents.bug_bounty_agent import BugBountyAgent
from core.models import AgentType, ProjectTask, AgentResult
from core.config import get_config
from core.logger import get_logger


class AgentStatus(str, Enum):
    """Status enumeration for agent instances."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


class AgentInstance:
    """
    Wrapper for agent instances with lifecycle management.
    """
    
    def __init__(self, agent: BaseAgent, agent_type: AgentType, config: Dict[str, Any]):
        self.agent = agent
        self.agent_type = agent_type
        self.config = config
        self.status = AgentStatus.CREATED
        self.created_at = datetime.now()
        self.last_used = None
        self.error_message = None
        self.task_queue = asyncio.Queue()
        self.is_processing = False
        self.stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0
        }
        self.logger = get_logger(f"agent_instance_{agent.agent_id}")
        
        # Mark as ready after creation
        self.status = AgentStatus.READY
        self.logger.info(f"Agent instance created: {agent.agent_id}")
    
    async def execute_task(self, task: ProjectTask) -> AgentResult:
        """
        Execute a task using this agent instance.
        
        Args:
            task: The task to execute
            
        Returns:
            AgentResult: Result of the task execution
        """
        if self.status != AgentStatus.READY:
            raise RuntimeError(f"Agent {self.agent.agent_id} is not ready (status: {self.status})")
        
        self.status = AgentStatus.BUSY
        self.is_processing = True
        self.last_used = datetime.now()
        
        try:
            self.logger.info(f"Executing task {task.id} with agent {self.agent.agent_id}")
            result = await self.agent.execute_with_retry(task)
            
            # Update stats
            self.stats['total_tasks'] += 1
            self.stats['total_execution_time'] += result.execution_time
            
            if result.success:
                self.stats['successful_tasks'] += 1
            else:
                self.stats['failed_tasks'] += 1
            
            self.status = AgentStatus.READY
            self.error_message = None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.id}: {str(e)}")
            self.status = AgentStatus.ERROR
            self.error_message = str(e)
            self.stats['failed_tasks'] += 1
            
            # Create error result
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error=str(e),
                execution_time=0.0,
                agent_id=self.agent.agent_id,
                timestamp=datetime.now()
            )
        
        finally:
            self.is_processing = False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this agent instance.
        
        Returns:
            Dict with agent instance information
        """
        return {
            'agent_id': self.agent.agent_id,
            'agent_type': self.agent_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'error_message': self.error_message,
            'is_processing': self.is_processing,
            'stats': self.stats.copy(),
            'config': self.config.copy()
        }
    
    def reset(self):
        """Reset agent instance to ready state."""
        if self.status != AgentStatus.BUSY:
            self.status = AgentStatus.READY
            self.error_message = None
            self.is_processing = False
            self.logger.info(f"Agent instance reset: {self.agent.agent_id}")
    
    def stop(self):
        """Stop the agent instance."""
        self.status = AgentStatus.STOPPED
        self.is_processing = False
        self.logger.info(f"Agent instance stopped: {self.agent.agent_id}")


class AgentRegistry:
    """
    Registry for managing agent instances with lifecycle management.
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentInstance] = {}
        self.agents_by_type: Dict[AgentType, List[AgentInstance]] = {
            agent_type: [] for agent_type in AgentType
        }
        self.lock = threading.Lock()
        self.logger = get_logger("agent_registry")
        self.max_agents_per_type = 3
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.now()
    
    def register(self, agent_instance: AgentInstance) -> bool:
        """
        Register an agent instance.
        
        Args:
            agent_instance: The agent instance to register
            
        Returns:
            True if registered successfully
        """
        with self.lock:
            agent_id = agent_instance.agent.agent_id
            
            if agent_id in self.agents:
                self.logger.warning(f"Agent {agent_id} already registered")
                return False
            
            # Check if we have too many agents of this type
            agent_type = agent_instance.agent_type
            if len(self.agents_by_type[agent_type]) >= self.max_agents_per_type:
                self.logger.warning(f"Maximum agents reached for type {agent_type}")
                return False
            
            self.agents[agent_id] = agent_instance
            self.agents_by_type[agent_type].append(agent_instance)
            
            self.logger.info(f"Registered agent {agent_id} of type {agent_type}")
            return True
    
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent instance.
        
        Args:
            agent_id: The agent ID to unregister
            
        Returns:
            True if unregistered successfully
        """
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            agent_instance = self.agents[agent_id]
            agent_instance.stop()
            
            # Remove from main registry
            del self.agents[agent_id]
            
            # Remove from type-specific registry
            agent_type = agent_instance.agent_type
            if agent_instance in self.agents_by_type[agent_type]:
                self.agents_by_type[agent_type].remove(agent_instance)
            
            self.logger.info(f"Unregistered agent {agent_id}")
            return True
    
    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """
        Get an agent instance by ID.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            AgentInstance or None
        """
        with self.lock:
            return self.agents.get(agent_id)
    
    def get_available_agent(self, agent_type: AgentType) -> Optional[AgentInstance]:
        """
        Get an available agent of the specified type.
        
        Args:
            agent_type: The type of agent needed
            
        Returns:
            Available AgentInstance or None
        """
        with self.lock:
            agents = self.agents_by_type.get(agent_type, [])
            
            # Find ready agents
            ready_agents = [agent for agent in agents if agent.status == AgentStatus.READY]
            
            if ready_agents:
                # Return the least recently used agent
                return min(ready_agents, key=lambda a: a.last_used or datetime.min)
            
            return None
    
    def get_all_agents(self) -> List[AgentInstance]:
        """
        Get all registered agents.
        
        Returns:
            List of all AgentInstance objects
        """
        with self.lock:
            return list(self.agents.values())
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentInstance]:
        """
        Get all agents of a specific type.
        
        Args:
            agent_type: The agent type
            
        Returns:
            List of AgentInstance objects
        """
        with self.lock:
            return list(self.agents_by_type.get(agent_type, []))
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        with self.lock:
            stats = {
                'total_agents': len(self.agents),
                'agents_by_type': {
                    agent_type.value: len(agents) 
                    for agent_type, agents in self.agents_by_type.items()
                },
                'agents_by_status': {},
                'total_tasks_executed': 0,
                'total_successful_tasks': 0,
                'total_failed_tasks': 0
            }
            
            # Count by status
            for agent in self.agents.values():
                status = agent.status.value
                stats['agents_by_status'][status] = stats['agents_by_status'].get(status, 0) + 1
                
                # Aggregate task stats
                stats['total_tasks_executed'] += agent.stats['total_tasks']
                stats['total_successful_tasks'] += agent.stats['successful_tasks']
                stats['total_failed_tasks'] += agent.stats['failed_tasks']
            
            return stats
    
    def cleanup(self):
        """Clean up inactive or errored agents."""
        current_time = datetime.now()
        
        # Only cleanup if enough time has passed
        if (current_time - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return
        
        with self.lock:
            agents_to_remove = []
            
            for agent_id, agent_instance in self.agents.items():
                # Remove stopped agents
                if agent_instance.status == AgentStatus.STOPPED:
                    agents_to_remove.append(agent_id)
                    continue
                
                # Remove agents that have been in error state for too long
                if (agent_instance.status == AgentStatus.ERROR and 
                    agent_instance.last_used and 
                    (current_time - agent_instance.last_used).total_seconds() > 1800):  # 30 minutes
                    agents_to_remove.append(agent_id)
                    continue
            
            # Remove identified agents
            for agent_id in agents_to_remove:
                self.unregister(agent_id)
            
            self.last_cleanup = current_time
            
            if agents_to_remove:
                self.logger.info(f"Cleaned up {len(agents_to_remove)} inactive agents")


class AgentFactory:
    """
    Factory for creating agent instances with configuration management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.system_config = get_config()
        self.registry = AgentRegistry()
        self.logger = get_logger("agent_factory")
        
        # Default configurations for each agent type
        self.default_configs = {
            AgentType.PARSER: {
                'model_name': 'all-MiniLM-L6-v2',
                'chunk_size': 256,
                'similarity_threshold': 0.7,
                'max_retries': 3,
                'timeout': 300
            },
            AgentType.CODER: {
                'model': 'claude-3-5-sonnet-20241022',
                'max_tokens': 4000,
                'temperature': 0.1,
                'max_file_size': 500,
                'max_retries': 3,
                'timeout': 600
            },
            AgentType.TESTER: {
                'test_timeout': 300,
                'max_retries_flaky': 2,
                'supported_frameworks': ['pytest', 'unittest', 'jest'],
                'max_retries': 2,
                'timeout': 450
            },
            AgentType.ADVISOR: {
                'model': 'o3-mini',
                'reasoning_effort': 'medium',
                'max_tokens': 8000,
                'temperature': 0.3,
                'max_retries': 2,
                'timeout': 900
            },
            AgentType.DOCUMENTATION: {
                'supported_formats': ['.md', '.rst', '.txt'],
                'max_retries': 2,
                'timeout': 1200  # 20 minutes - enough for complex documentation updates
            },
            AgentType.TESTING_VALIDATION: {
                'max_retries': 5,
                'retry_delay_base': 2.0,
                'retry_delay_max': 60.0,
                'timeout': 2400  # 40 minutes - enough for comprehensive test suites
            },
            AgentType.VISUAL_TESTING: {
                'screenshot_timeout': 60,    # Increased from 30s
                'page_load_timeout': 30,     # Increased from 10s  
                'supported_browsers': ['chrome', 'firefox'],
                'max_retries': 3,
                'timeout': 1800  # 30 minutes - enough for app startup + AI vision analysis
            },
            AgentType.BUG_BOUNTY: {
                'max_debugging_depth': 10,
                'analysis_timeout': 1800,
                'max_retries': 1,
                'timeout': 3600  # 60 minutes - increased for thorough debugging
            }
        }
        
        # Agent class mapping
        self.agent_classes = {
            AgentType.PARSER: ParserAgent,
            AgentType.CODER: CoderAgent,
            AgentType.TESTER: TesterAgent,
            AgentType.ADVISOR: AdvisorAgent,
            AgentType.DOCUMENTATION: DocumentationAgent,
            AgentType.TESTING_VALIDATION: TestingValidationAgent,
            AgentType.VISUAL_TESTING: VisualTestingAgent,
            AgentType.BUG_BOUNTY: BugBountyAgent,
            "project_manager": ProjectManagerAgent,
            "research": ResearchAgent
        }
    
    def create_agent(self, agent_type: AgentType, 
                    config: Optional[Dict[str, Any]] = None,
                    register: bool = True) -> AgentInstance:
        """
        Create an agent instance of the specified type.
        
        Args:
            agent_type: The type of agent to create
            config: Optional configuration overrides
            register: Whether to register the agent automatically
            
        Returns:
            AgentInstance: The created agent instance
        """
        if agent_type not in self.agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Merge configurations
        agent_config = self.default_configs.get(agent_type, {}).copy()
        if config:
            agent_config.update(config)
        if self.config:
            agent_config.update(self.config)
        
        try:
            # Create the agent
            agent_class = self.agent_classes[agent_type]
            agent = agent_class(config=agent_config)
            
            # Create agent instance wrapper
            agent_instance = AgentInstance(agent, agent_type, agent_config)
            
            # Register if requested
            if register:
                if not self.registry.register(agent_instance):
                    self.logger.warning(f"Failed to register agent {agent.agent_id}")
            
            self.logger.info(f"Created agent {agent.agent_id} of type {agent_type}")
            return agent_instance
            
        except Exception as e:
            self.logger.error(f"Failed to create agent of type {agent_type}: {str(e)}")
            raise
    
    def get_or_create_agent(self, agent_type: AgentType, 
                           config: Optional[Dict[str, Any]] = None) -> AgentInstance:
        """
        Get an available agent or create a new one if none available.
        
        Args:
            agent_type: The type of agent needed
            config: Optional configuration overrides
            
        Returns:
            AgentInstance: Available or newly created agent
        """
        # Try to get an available agent
        agent_instance = self.registry.get_available_agent(agent_type)
        
        if agent_instance:
            self.logger.debug(f"Using existing agent {agent_instance.agent.agent_id}")
            return agent_instance
        
        # Create new agent if none available
        self.logger.info(f"Creating new agent of type {agent_type}")
        return self.create_agent(agent_type, config, register=True)
    
    def create_agent_pool(self, agent_type: AgentType, pool_size: int = 2,
                         config: Optional[Dict[str, Any]] = None) -> List[AgentInstance]:
        """
        Create a pool of agents of the specified type.
        
        Args:
            agent_type: The type of agents to create
            pool_size: Number of agents to create
            config: Optional configuration overrides
            
        Returns:
            List of created AgentInstance objects
        """
        agents = []
        
        for i in range(pool_size):
            try:
                agent = self.create_agent(agent_type, config, register=True)
                agents.append(agent)
            except Exception as e:
                self.logger.error(f"Failed to create agent {i+1}/{pool_size}: {str(e)}")
        
        self.logger.info(f"Created pool of {len(agents)} agents of type {agent_type}")
        return agents
    
    def get_registry(self) -> AgentRegistry:
        """
        Get the agent registry.
        
        Returns:
            AgentRegistry instance
        """
        return self.registry
    
    def shutdown(self):
        """Shutdown the factory and all agents."""
        self.logger.info("Shutting down agent factory")
        
        # Stop all agents
        for agent_instance in self.registry.get_all_agents():
            agent_instance.stop()
        
        # Clear registry
        self.registry.agents.clear()
        for agent_type in AgentType:
            self.registry.agents_by_type[agent_type].clear()


# Global factory instance
_factory_instance: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """
    Get the global agent factory instance.
    
    Returns:
        AgentFactory: The global factory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = AgentFactory()
    return _factory_instance


def create_agent(agent_type: Union[str, AgentType], 
                config: Optional[Dict[str, Any]] = None,
                register: bool = True) -> AgentInstance:
    """
    Convenience function to create an agent.
    
    Args:
        agent_type: The type of agent to create
        config: Optional configuration overrides
        register: Whether to register the agent
        
    Returns:
        AgentInstance: The created agent instance
    """
    if isinstance(agent_type, str):
        agent_type = AgentType(agent_type)
    
    factory = get_agent_factory()
    return factory.create_agent(agent_type, config, register)


def get_or_create_agent(agent_type: Union[str, AgentType], 
                       config: Optional[Dict[str, Any]] = None) -> AgentInstance:
    """
    Convenience function to get or create an agent.
    
    Args:
        agent_type: The type of agent needed
        config: Optional configuration overrides
        
    Returns:
        AgentInstance: Available or newly created agent
    """
    if isinstance(agent_type, str):
        agent_type = AgentType(agent_type)
    
    factory = get_agent_factory()
    return factory.get_or_create_agent(agent_type, config)


def execute_task_with_agent(task: ProjectTask, 
                           agent_type: Optional[AgentType] = None) -> AgentResult:
    """
    Execute a task using an appropriate agent.
    
    Args:
        task: The task to execute
        agent_type: Optional specific agent type to use
        
    Returns:
        AgentResult: Result of the task execution
    """
    # Use specified agent type or task's agent type
    target_agent_type = agent_type or task.agent_type
    
    # Get or create agent
    agent_instance = get_or_create_agent(target_agent_type)
    
    # Execute task
    return asyncio.run(agent_instance.execute_task(task))


def get_agent_registry() -> AgentRegistry:
    """
    Get the global agent registry.
    
    Returns:
        AgentRegistry: The global registry instance
    """
    return get_agent_factory().get_registry()


def shutdown_agents():
    """Shutdown all agents and the factory."""
    global _factory_instance
    if _factory_instance:
        _factory_instance.shutdown()
        _factory_instance = None