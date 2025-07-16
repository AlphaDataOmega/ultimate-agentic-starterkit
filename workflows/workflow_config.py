"""
Workflow Configuration System for the Ultimate Agentic StarterKit.

This module provides workflow templates, parameters, agent selection,
and confidence thresholds for customizable workflow execution.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
from datetime import datetime

from core.models import AgentType, TaskStatus
from core.logger import get_logger
from core.config import get_config


class WorkflowTemplate(str, Enum):
    """Predefined workflow templates."""
    WEB_DEVELOPMENT = "web_development"
    BLOCKCHAIN_DEVELOPMENT = "blockchain_development"
    AI_DEVELOPMENT = "ai_development"
    GENERAL_DEVELOPMENT = "general_development"
    DATA_ANALYSIS = "data_analysis"
    TESTING_VALIDATION = "testing_validation"
    DOCUMENTATION = "documentation"
    CUSTOM = "custom"


class ExecutionStrategy(str, Enum):
    """Workflow execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


class FailureStrategy(str, Enum):
    """Failure handling strategies."""
    FAIL_FAST = "fail_fast"
    CONTINUE_ON_FAILURE = "continue_on_failure"
    RETRY_FAILED = "retry_failed"
    ADAPTIVE_RETRY = "adaptive_retry"


@dataclass
class AgentConfiguration:
    """Configuration for a specific agent type."""
    agent_type: AgentType
    max_retries: int = 3
    timeout: int = 600
    confidence_threshold: float = 0.8
    high_confidence_threshold: float = 0.95
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_type': self.agent_type.value,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'confidence_threshold': self.confidence_threshold,
            'high_confidence_threshold': self.high_confidence_threshold,
            'custom_config': self.custom_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfiguration':
        """Create from dictionary."""
        return cls(
            agent_type=AgentType(data['agent_type']),
            max_retries=data.get('max_retries', 3),
            timeout=data.get('timeout', 600),
            confidence_threshold=data.get('confidence_threshold', 0.8),
            high_confidence_threshold=data.get('high_confidence_threshold', 0.95),
            custom_config=data.get('custom_config', {})
        )


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel execution."""
    max_concurrent_tasks: int = 3
    max_concurrent_agents: int = 5
    group_similar_tasks: bool = True
    respect_dependencies: bool = True
    timeout_multiplier: float = 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParallelExecutionConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RetryConfiguration:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on_low_confidence: bool = True
    confidence_retry_threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetryConfiguration':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ProgressConfiguration:
    """Configuration for progress tracking."""
    enable_progress_tracking: bool = True
    enable_voice_alerts: bool = True
    snapshot_interval: int = 30
    milestone_notifications: bool = True
    detailed_logging: bool = True
    performance_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressConfiguration':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StateManagementConfig:
    """Configuration for state management."""
    enable_state_persistence: bool = True
    auto_checkpoint_interval: int = 300  # 5 minutes
    max_state_history: int = 100
    compression_enabled: bool = True
    backup_states: bool = True
    cleanup_old_states: bool = True
    max_state_age_days: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateManagementConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class WorkflowConfiguration:
    """Comprehensive workflow configuration."""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Workflow"
    description: str = ""
    template: WorkflowTemplate = WorkflowTemplate.GENERAL_DEVELOPMENT
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    failure_strategy: FailureStrategy = FailureStrategy.ADAPTIVE_RETRY
    
    # Agent configurations
    agent_configs: Dict[str, AgentConfiguration] = field(default_factory=dict)
    
    # Execution configurations
    parallel_config: ParallelExecutionConfig = field(default_factory=ParallelExecutionConfig)
    retry_config: RetryConfiguration = field(default_factory=RetryConfiguration)
    progress_config: ProgressConfiguration = field(default_factory=ProgressConfiguration)
    state_config: StateManagementConfig = field(default_factory=StateManagementConfig)
    
    # Workflow parameters
    workflow_timeout: int = 7200  # 2 hours
    task_timeout: int = 600  # 10 minutes
    overall_confidence_threshold: float = 0.8
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default agent configurations."""
        if not self.agent_configs:
            self._create_default_agent_configs()
    
    def _create_default_agent_configs(self):
        """Create default agent configurations."""
        self.agent_configs = {
            AgentType.PARSER.value: AgentConfiguration(
                agent_type=AgentType.PARSER,
                max_retries=2,
                timeout=300,
                confidence_threshold=0.7,
                high_confidence_threshold=0.9
            ),
            AgentType.CODER.value: AgentConfiguration(
                agent_type=AgentType.CODER,
                max_retries=3,
                timeout=900,
                confidence_threshold=0.8,
                high_confidence_threshold=0.95
            ),
            AgentType.TESTER.value: AgentConfiguration(
                agent_type=AgentType.TESTER,
                max_retries=2,
                timeout=600,
                confidence_threshold=0.9,
                high_confidence_threshold=0.98
            ),
            AgentType.ADVISOR.value: AgentConfiguration(
                agent_type=AgentType.ADVISOR,
                max_retries=1,
                timeout=300,
                confidence_threshold=0.7,
                high_confidence_threshold=0.85
            )
        }
    
    def get_agent_config(self, agent_type: AgentType) -> Optional[AgentConfiguration]:
        """Get configuration for a specific agent type."""
        return self.agent_configs.get(agent_type.value)
    
    def set_agent_config(self, agent_config: AgentConfiguration):
        """Set configuration for a specific agent type."""
        self.agent_configs[agent_config.agent_type.value] = agent_config
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config_id': self.config_id,
            'name': self.name,
            'description': self.description,
            'template': self.template.value,
            'execution_strategy': self.execution_strategy.value,
            'failure_strategy': self.failure_strategy.value,
            'agent_configs': {k: v.to_dict() for k, v in self.agent_configs.items()},
            'parallel_config': self.parallel_config.to_dict(),
            'retry_config': self.retry_config.to_dict(),
            'progress_config': self.progress_config.to_dict(),
            'state_config': self.state_config.to_dict(),
            'workflow_timeout': self.workflow_timeout,
            'task_timeout': self.task_timeout,
            'overall_confidence_threshold': self.overall_confidence_threshold,
            'custom_params': self.custom_params,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowConfiguration':
        """Create from dictionary."""
        # Convert agent configs
        agent_configs = {}
        for k, v in data.get('agent_configs', {}).items():
            agent_configs[k] = AgentConfiguration.from_dict(v)
        
        return cls(
            config_id=data.get('config_id', str(uuid.uuid4())),
            name=data.get('name', 'Default Workflow'),
            description=data.get('description', ''),
            template=WorkflowTemplate(data.get('template', 'general_development')),
            execution_strategy=ExecutionStrategy(data.get('execution_strategy', 'adaptive')),
            failure_strategy=FailureStrategy(data.get('failure_strategy', 'adaptive_retry')),
            agent_configs=agent_configs,
            parallel_config=ParallelExecutionConfig.from_dict(data.get('parallel_config', {})),
            retry_config=RetryConfiguration.from_dict(data.get('retry_config', {})),
            progress_config=ProgressConfiguration.from_dict(data.get('progress_config', {})),
            state_config=StateManagementConfig.from_dict(data.get('state_config', {})),
            workflow_timeout=data.get('workflow_timeout', 7200),
            task_timeout=data.get('task_timeout', 600),
            overall_confidence_threshold=data.get('overall_confidence_threshold', 0.8),
            custom_params=data.get('custom_params', {}),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
            version=data.get('version', '1.0.0'),
            tags=data.get('tags', [])
        )
    
    def save_to_file(self, file_path: str):
        """Save configuration to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'WorkflowConfiguration':
        """Load configuration from file."""
        path = Path(file_path)
        
        with open(path, 'r') as f:
            if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    def clone(self, new_name: str = None) -> 'WorkflowConfiguration':
        """Create a copy of this configuration."""
        config_dict = self.to_dict()
        config_dict['config_id'] = str(uuid.uuid4())
        config_dict['created_at'] = datetime.now().isoformat()
        config_dict['updated_at'] = datetime.now().isoformat()
        
        if new_name:
            config_dict['name'] = new_name
        
        return self.from_dict(config_dict)


class WorkflowConfigManager:
    """
    Manager for workflow configurations with templates and customization.
    """
    
    def __init__(self, config_dir: str = "workflow_configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("workflow_config_manager")
        self.system_config = get_config()
        
        # Store loaded configurations
        self.configurations: Dict[str, WorkflowConfiguration] = {}
        
        # Load default templates
        self._create_default_templates()
        
        self.logger.info("Workflow Configuration Manager initialized")
    
    def _create_default_templates(self):
        """Create default workflow templates."""
        templates = {
            WorkflowTemplate.WEB_DEVELOPMENT: self._create_web_development_template(),
            WorkflowTemplate.BLOCKCHAIN_DEVELOPMENT: self._create_blockchain_development_template(),
            WorkflowTemplate.AI_DEVELOPMENT: self._create_ai_development_template(),
            WorkflowTemplate.GENERAL_DEVELOPMENT: self._create_general_development_template(),
            WorkflowTemplate.DATA_ANALYSIS: self._create_data_analysis_template(),
            WorkflowTemplate.TESTING_VALIDATION: self._create_testing_validation_template(),
            WorkflowTemplate.DOCUMENTATION: self._create_documentation_template()
        }
        
        for template_type, config in templates.items():
            self.configurations[template_type.value] = config
    
    def _create_web_development_template(self) -> WorkflowConfiguration:
        """Create web development workflow template."""
        config = WorkflowConfiguration(
            name="Web Development Workflow",
            description="Optimized for web application development with React, Node.js, and modern frameworks",
            template=WorkflowTemplate.WEB_DEVELOPMENT,
            execution_strategy=ExecutionStrategy.HYBRID,
            failure_strategy=FailureStrategy.RETRY_FAILED
        )
        
        # Customize agent configurations for web development
        config.agent_configs[AgentType.CODER.value].timeout = 1200  # 20 minutes
        config.agent_configs[AgentType.CODER.value].custom_config = {
            'preferred_frameworks': ['react', 'next.js', 'tailwind'],
            'code_style': 'functional',
            'testing_framework': 'jest'
        }
        
        config.agent_configs[AgentType.TESTER.value].custom_config = {
            'test_types': ['unit', 'integration', 'e2e'],
            'coverage_threshold': 0.8
        }
        
        # Parallel execution for web development
        config.parallel_config.max_concurrent_tasks = 4
        config.parallel_config.group_similar_tasks = True
        
        config.tags = ['web', 'frontend', 'javascript', 'react']
        return config
    
    def _create_blockchain_development_template(self) -> WorkflowConfiguration:
        """Create blockchain development workflow template."""
        config = WorkflowConfiguration(
            name="Blockchain Development Workflow",
            description="Optimized for blockchain and smart contract development",
            template=WorkflowTemplate.BLOCKCHAIN_DEVELOPMENT,
            execution_strategy=ExecutionStrategy.SEQUENTIAL,
            failure_strategy=FailureStrategy.FAIL_FAST
        )
        
        # Higher confidence thresholds for blockchain
        config.overall_confidence_threshold = 0.95
        for agent_config in config.agent_configs.values():
            agent_config.confidence_threshold = 0.9
            agent_config.high_confidence_threshold = 0.98
        
        config.agent_configs[AgentType.CODER.value].custom_config = {
            'languages': ['solidity', 'rust', 'go'],
            'security_focused': True,
            'gas_optimization': True
        }
        
        config.agent_configs[AgentType.TESTER.value].custom_config = {
            'test_types': ['unit', 'integration', 'security', 'gas'],
            'security_tools': ['mythril', 'slither']
        }
        
        # Conservative parallel execution
        config.parallel_config.max_concurrent_tasks = 2
        config.parallel_config.respect_dependencies = True
        
        config.tags = ['blockchain', 'smart-contracts', 'solidity', 'defi']
        return config
    
    def _create_ai_development_template(self) -> WorkflowConfiguration:
        """Create AI development workflow template."""
        config = WorkflowConfiguration(
            name="AI Development Workflow",
            description="Optimized for AI and machine learning development",
            template=WorkflowTemplate.AI_DEVELOPMENT,
            execution_strategy=ExecutionStrategy.ADAPTIVE,
            failure_strategy=FailureStrategy.ADAPTIVE_RETRY
        )
        
        # Extended timeouts for AI development
        config.workflow_timeout = 14400  # 4 hours
        config.task_timeout = 1800  # 30 minutes
        
        config.agent_configs[AgentType.CODER.value].timeout = 1800
        config.agent_configs[AgentType.CODER.value].custom_config = {
            'frameworks': ['pytorch', 'tensorflow', 'transformers'],
            'optimization': 'performance',
            'gpu_support': True
        }
        
        config.agent_configs[AgentType.TESTER.value].custom_config = {
            'test_types': ['unit', 'model_validation', 'performance'],
            'metrics': ['accuracy', 'precision', 'recall', 'f1']
        }
        
        config.parallel_config.max_concurrent_tasks = 3
        config.parallel_config.timeout_multiplier = 2.0
        
        config.tags = ['ai', 'machine-learning', 'neural-networks', 'deep-learning']
        return config
    
    def _create_general_development_template(self) -> WorkflowConfiguration:
        """Create general development workflow template."""
        config = WorkflowConfiguration(
            name="General Development Workflow",
            description="Balanced configuration for general software development",
            template=WorkflowTemplate.GENERAL_DEVELOPMENT,
            execution_strategy=ExecutionStrategy.ADAPTIVE,
            failure_strategy=FailureStrategy.ADAPTIVE_RETRY
        )
        
        # Balanced settings
        config.parallel_config.max_concurrent_tasks = 3
        config.retry_config.max_retries = 3
        
        config.tags = ['general', 'development', 'balanced']
        return config
    
    def _create_data_analysis_template(self) -> WorkflowConfiguration:
        """Create data analysis workflow template."""
        config = WorkflowConfiguration(
            name="Data Analysis Workflow",
            description="Optimized for data analysis and processing tasks",
            template=WorkflowTemplate.DATA_ANALYSIS,
            execution_strategy=ExecutionStrategy.PARALLEL,
            failure_strategy=FailureStrategy.CONTINUE_ON_FAILURE
        )
        
        config.agent_configs[AgentType.CODER.value].custom_config = {
            'libraries': ['pandas', 'numpy', 'scipy', 'matplotlib'],
            'optimization': 'memory',
            'data_formats': ['csv', 'json', 'parquet']
        }
        
        config.parallel_config.max_concurrent_tasks = 5
        config.parallel_config.group_similar_tasks = True
        
        config.tags = ['data', 'analysis', 'statistics', 'visualization']
        return config
    
    def _create_testing_validation_template(self) -> WorkflowConfiguration:
        """Create testing and validation workflow template."""
        config = WorkflowConfiguration(
            name="Testing & Validation Workflow",
            description="Focused on comprehensive testing and validation",
            template=WorkflowTemplate.TESTING_VALIDATION,
            execution_strategy=ExecutionStrategy.SEQUENTIAL,
            failure_strategy=FailureStrategy.FAIL_FAST
        )
        
        # High confidence requirements
        config.overall_confidence_threshold = 0.95
        for agent_config in config.agent_configs.values():
            agent_config.confidence_threshold = 0.9
        
        config.agent_configs[AgentType.TESTER.value].max_retries = 5
        config.agent_configs[AgentType.TESTER.value].timeout = 900
        
        config.tags = ['testing', 'validation', 'quality-assurance']
        return config
    
    def _create_documentation_template(self) -> WorkflowConfiguration:
        """Create documentation workflow template."""
        config = WorkflowConfiguration(
            name="Documentation Workflow",
            description="Optimized for documentation generation and maintenance",
            template=WorkflowTemplate.DOCUMENTATION,
            execution_strategy=ExecutionStrategy.PARALLEL,
            failure_strategy=FailureStrategy.CONTINUE_ON_FAILURE
        )
        
        config.agent_configs[AgentType.CODER.value].custom_config = {
            'documentation_tools': ['sphinx', 'mkdocs', 'gitbook'],
            'formats': ['markdown', 'rst', 'html']
        }
        
        config.parallel_config.max_concurrent_tasks = 4
        config.task_timeout = 300  # 5 minutes
        
        config.tags = ['documentation', 'writing', 'maintenance']
        return config
    
    def get_template(self, template_type: WorkflowTemplate) -> Optional[WorkflowConfiguration]:
        """
        Get a workflow template.
        
        Args:
            template_type: Type of template to get
            
        Returns:
            WorkflowConfiguration or None if not found
        """
        return self.configurations.get(template_type.value)
    
    def create_configuration(self, template_type: WorkflowTemplate = WorkflowTemplate.GENERAL_DEVELOPMENT,
                           name: str = None, description: str = None) -> WorkflowConfiguration:
        """
        Create a new workflow configuration from template.
        
        Args:
            template_type: Template to base configuration on
            name: Optional custom name
            description: Optional custom description
            
        Returns:
            New WorkflowConfiguration instance
        """
        template = self.get_template(template_type)
        if not template:
            template = self._create_general_development_template()
        
        config = template.clone(name)
        
        if description:
            config.description = description
        
        return config
    
    def save_configuration(self, config: WorkflowConfiguration, filename: str = None):
        """
        Save a workflow configuration to file.
        
        Args:
            config: Configuration to save
            filename: Optional filename (defaults to config name)
        """
        if filename is None:
            filename = f"{config.name.lower().replace(' ', '_')}.json"
        
        file_path = self.config_dir / filename
        config.save_to_file(str(file_path))
        
        # Store in memory
        self.configurations[config.config_id] = config
        
        self.logger.info(f"Saved configuration: {config.name}")
    
    def load_configuration(self, filename: str) -> Optional[WorkflowConfiguration]:
        """
        Load a workflow configuration from file.
        
        Args:
            filename: Configuration filename
            
        Returns:
            WorkflowConfiguration or None if not found
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            self.logger.error(f"Configuration file not found: {filename}")
            return None
        
        try:
            config = WorkflowConfiguration.load_from_file(str(file_path))
            self.configurations[config.config_id] = config
            
            self.logger.info(f"Loaded configuration: {config.name}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration {filename}: {e}")
            return None
    
    def list_configurations(self) -> List[Dict[str, Any]]:
        """
        List all available configurations.
        
        Returns:
            List of configuration summaries
        """
        configs = []
        
        for config in self.configurations.values():
            configs.append({
                'config_id': config.config_id,
                'name': config.name,
                'description': config.description,
                'template': config.template.value,
                'execution_strategy': config.execution_strategy.value,
                'created_at': config.created_at.isoformat(),
                'updated_at': config.updated_at.isoformat(),
                'tags': config.tags
            })
        
        return configs
    
    def get_configuration(self, config_id: str) -> Optional[WorkflowConfiguration]:
        """
        Get a configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            WorkflowConfiguration or None if not found
        """
        return self.configurations.get(config_id)
    
    def delete_configuration(self, config_id: str) -> bool:
        """
        Delete a configuration.
        
        Args:
            config_id: Configuration ID to delete
            
        Returns:
            True if successful
        """
        if config_id in self.configurations:
            config = self.configurations[config_id]
            
            # Don't delete templates
            if config.template.value == config_id:
                self.logger.warning(f"Cannot delete template: {config.name}")
                return False
            
            del self.configurations[config_id]
            self.logger.info(f"Deleted configuration: {config.name}")
            return True
        
        return False
    
    def export_configurations(self, export_path: str):
        """
        Export all configurations to a single file.
        
        Args:
            export_path: Path to export file
        """
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'configurations': [config.to_dict() for config in self.configurations.values()]
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(self.configurations)} configurations")
    
    def import_configurations(self, import_path: str):
        """
        Import configurations from a file.
        
        Args:
            import_path: Path to import file
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for config_data in import_data.get('configurations', []):
                config = WorkflowConfiguration.from_dict(config_data)
                self.configurations[config.config_id] = config
                imported_count += 1
            
            self.logger.info(f"Imported {imported_count} configurations")
            
        except Exception as e:
            self.logger.error(f"Failed to import configurations: {e}")


# Global configuration manager instance
_config_manager_instance: Optional[WorkflowConfigManager] = None


def get_config_manager() -> WorkflowConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        WorkflowConfigManager instance
    """
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = WorkflowConfigManager()
    return _config_manager_instance


def get_workflow_template(template_type: WorkflowTemplate) -> Optional[WorkflowConfiguration]:
    """
    Convenience function to get a workflow template.
    
    Args:
        template_type: Template type
        
    Returns:
        WorkflowConfiguration or None
    """
    return get_config_manager().get_template(template_type)


def create_workflow_config(template_type: WorkflowTemplate = WorkflowTemplate.GENERAL_DEVELOPMENT,
                          name: str = None) -> WorkflowConfiguration:
    """
    Convenience function to create a workflow configuration.
    
    Args:
        template_type: Template to use
        name: Optional custom name
        
    Returns:
        WorkflowConfiguration instance
    """
    return get_config_manager().create_configuration(template_type, name)