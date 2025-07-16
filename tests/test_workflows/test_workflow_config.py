"""
Tests for Workflow Configuration System.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

from workflows.workflow_config import (
    WorkflowConfigManager,
    WorkflowTemplate,
    WorkflowParameter,
    AgentConfiguration,
    WorkflowConfig,
    ConfigValidationError,
    TemplateNotFoundError,
    create_workflow_config,
    load_workflow_template,
    validate_workflow_config,
    get_config_manager,
    create_agent_config,
    create_workflow_parameter
)
from core.models import AgentType


class TestWorkflowParameter:
    """Test cases for WorkflowParameter."""
    
    def test_workflow_parameter_initialization(self):
        """Test workflow parameter initialization."""
        param = WorkflowParameter(
            name="confidence_threshold",
            value=0.8,
            type="float",
            description="Confidence threshold for task completion",
            required=True,
            default_value=0.7,
            validation_rules={"min": 0.0, "max": 1.0}
        )
        
        assert param.name == "confidence_threshold"
        assert param.value == 0.8
        assert param.type == "float"
        assert param.description == "Confidence threshold for task completion"
        assert param.required is True
        assert param.default_value == 0.7
        assert param.validation_rules == {"min": 0.0, "max": 1.0}
    
    def test_workflow_parameter_with_defaults(self):
        """Test workflow parameter with default values."""
        param = WorkflowParameter(
            name="test_param",
            value="test_value"
        )
        
        assert param.name == "test_param"
        assert param.value == "test_value"
        assert param.type == "string"
        assert param.required is False
        assert param.default_value is None
        assert param.validation_rules == {}
    
    def test_workflow_parameter_to_dict(self):
        """Test workflow parameter serialization."""
        param = WorkflowParameter(
            name="confidence_threshold",
            value=0.8,
            type="float",
            description="Confidence threshold",
            required=True
        )
        
        data = param.to_dict()
        
        assert data["name"] == "confidence_threshold"
        assert data["value"] == 0.8
        assert data["type"] == "float"
        assert data["description"] == "Confidence threshold"
        assert data["required"] is True
    
    def test_workflow_parameter_validate_success(self):
        """Test successful parameter validation."""
        param = WorkflowParameter(
            name="confidence_threshold",
            value=0.8,
            type="float",
            validation_rules={"min": 0.0, "max": 1.0}
        )
        
        # Should not raise exception
        param.validate()
    
    def test_workflow_parameter_validate_failure(self):
        """Test parameter validation failure."""
        param = WorkflowParameter(
            name="confidence_threshold",
            value=1.5,  # Above max
            type="float",
            validation_rules={"min": 0.0, "max": 1.0}
        )
        
        with pytest.raises(ConfigValidationError):
            param.validate()
    
    def test_workflow_parameter_validate_required_missing(self):
        """Test validation of required parameter with None value."""
        param = WorkflowParameter(
            name="required_param",
            value=None,
            required=True
        )
        
        with pytest.raises(ConfigValidationError):
            param.validate()


class TestAgentConfiguration:
    """Test cases for AgentConfiguration."""
    
    def test_agent_configuration_initialization(self):
        """Test agent configuration initialization."""
        config = AgentConfiguration(
            agent_type=AgentType.PARSER,
            max_concurrent_tasks=2,
            timeout=300,
            retry_attempts=3,
            confidence_threshold=0.8,
            model_params={"temperature": 0.7},
            custom_settings={"debug": True}
        )
        
        assert config.agent_type == AgentType.PARSER
        assert config.max_concurrent_tasks == 2
        assert config.timeout == 300
        assert config.retry_attempts == 3
        assert config.confidence_threshold == 0.8
        assert config.model_params == {"temperature": 0.7}
        assert config.custom_settings == {"debug": True}
    
    def test_agent_configuration_with_defaults(self):
        """Test agent configuration with default values."""
        config = AgentConfiguration(
            agent_type=AgentType.CODER
        )
        
        assert config.agent_type == AgentType.CODER
        assert config.max_concurrent_tasks == 1
        assert config.timeout == 600
        assert config.retry_attempts == 3
        assert config.confidence_threshold == 0.8
        assert config.model_params == {}
        assert config.custom_settings == {}
    
    def test_agent_configuration_to_dict(self):
        """Test agent configuration serialization."""
        config = AgentConfiguration(
            agent_type=AgentType.PARSER,
            max_concurrent_tasks=2,
            timeout=300
        )
        
        data = config.to_dict()
        
        assert data["agent_type"] == "parser"
        assert data["max_concurrent_tasks"] == 2
        assert data["timeout"] == 300
    
    def test_agent_configuration_validate_success(self):
        """Test successful agent configuration validation."""
        config = AgentConfiguration(
            agent_type=AgentType.PARSER,
            max_concurrent_tasks=2,
            timeout=300,
            confidence_threshold=0.8
        )
        
        # Should not raise exception
        config.validate()
    
    def test_agent_configuration_validate_failure(self):
        """Test agent configuration validation failure."""
        config = AgentConfiguration(
            agent_type=AgentType.PARSER,
            max_concurrent_tasks=-1,  # Invalid value
            timeout=300
        )
        
        with pytest.raises(ConfigValidationError):
            config.validate()


class TestWorkflowConfig:
    """Test cases for WorkflowConfig."""
    
    def test_workflow_config_initialization(self):
        """Test workflow configuration initialization."""
        agent_config = AgentConfiguration(agent_type=AgentType.PARSER)
        parameter = WorkflowParameter(name="test_param", value="test_value")
        
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT,
            description="Test workflow configuration",
            agent_configurations=[agent_config],
            parameters=[parameter],
            max_concurrent_tasks=5,
            workflow_timeout=3600,
            retry_policy={"max_retries": 3, "delay": 1.0},
            metadata={"version": "1.0"}
        )
        
        assert config.name == "test_workflow"
        assert config.template == WorkflowTemplate.WEB_DEVELOPMENT
        assert config.description == "Test workflow configuration"
        assert len(config.agent_configurations) == 1
        assert len(config.parameters) == 1
        assert config.max_concurrent_tasks == 5
        assert config.workflow_timeout == 3600
        assert config.retry_policy == {"max_retries": 3, "delay": 1.0}
        assert config.metadata == {"version": "1.0"}
    
    def test_workflow_config_with_defaults(self):
        """Test workflow configuration with default values."""
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        assert config.name == "test_workflow"
        assert config.template == WorkflowTemplate.WEB_DEVELOPMENT
        assert config.description == ""
        assert config.agent_configurations == []
        assert config.parameters == []
        assert config.max_concurrent_tasks == 3
        assert config.workflow_timeout == 7200
        assert config.retry_policy == {"max_retries": 3, "delay": 1.0}
        assert config.metadata == {}
    
    def test_workflow_config_to_dict(self):
        """Test workflow configuration serialization."""
        agent_config = AgentConfiguration(agent_type=AgentType.PARSER)
        parameter = WorkflowParameter(name="test_param", value="test_value")
        
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT,
            agent_configurations=[agent_config],
            parameters=[parameter]
        )
        
        data = config.to_dict()
        
        assert data["name"] == "test_workflow"
        assert data["template"] == "web_development"
        assert len(data["agent_configurations"]) == 1
        assert len(data["parameters"]) == 1
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_workflow_config_get_parameter(self):
        """Test getting parameter by name."""
        parameter = WorkflowParameter(name="test_param", value="test_value")
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT,
            parameters=[parameter]
        )
        
        found_param = config.get_parameter("test_param")
        assert found_param is not None
        assert found_param.value == "test_value"
        
        missing_param = config.get_parameter("missing_param")
        assert missing_param is None
    
    def test_workflow_config_get_agent_config(self):
        """Test getting agent configuration by type."""
        agent_config = AgentConfiguration(agent_type=AgentType.PARSER)
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT,
            agent_configurations=[agent_config]
        )
        
        found_config = config.get_agent_config(AgentType.PARSER)
        assert found_config is not None
        assert found_config.agent_type == AgentType.PARSER
        
        missing_config = config.get_agent_config(AgentType.CODER)
        assert missing_config is None
    
    def test_workflow_config_validate_success(self):
        """Test successful workflow configuration validation."""
        agent_config = AgentConfiguration(agent_type=AgentType.PARSER)
        parameter = WorkflowParameter(name="test_param", value="test_value")
        
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT,
            agent_configurations=[agent_config],
            parameters=[parameter]
        )
        
        # Should not raise exception
        config.validate()
    
    def test_workflow_config_validate_failure(self):
        """Test workflow configuration validation failure."""
        config = WorkflowConfig(
            name="",  # Empty name
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        with pytest.raises(ConfigValidationError):
            config.validate()


class TestWorkflowConfigManager:
    """Test cases for WorkflowConfigManager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_config(self, temp_config_dir):
        """Mock configuration."""
        return {
            'config_storage_path': temp_config_dir,
            'default_template': 'web_development',
            'enable_validation': True,
            'auto_save_enabled': True
        }
    
    @pytest.fixture
    def config_manager(self, mock_config):
        """Create workflow config manager instance."""
        with patch('workflows.workflow_config.get_logger'):
            return WorkflowConfigManager(mock_config)
    
    def test_workflow_config_manager_initialization(self, config_manager):
        """Test workflow config manager initialization."""
        assert config_manager.config_storage_path is not None
        assert config_manager.default_template == WorkflowTemplate.WEB_DEVELOPMENT
        assert config_manager.enable_validation is True
        assert config_manager.auto_save_enabled is True
    
    def test_create_config_from_template(self, config_manager):
        """Test creating configuration from template."""
        config = config_manager.create_config_from_template(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT,
            custom_params={"confidence_threshold": 0.9}
        )
        
        assert config.name == "test_workflow"
        assert config.template == WorkflowTemplate.WEB_DEVELOPMENT
        assert len(config.agent_configurations) > 0
        assert len(config.parameters) > 0
        
        # Check custom parameter was set
        confidence_param = config.get_parameter("confidence_threshold")
        assert confidence_param is not None
        assert confidence_param.value == 0.9
    
    def test_create_config_from_template_unknown_template(self, config_manager):
        """Test creating configuration from unknown template."""
        with pytest.raises(TemplateNotFoundError):
            config_manager.create_config_from_template(
                name="test_workflow",
                template="unknown_template"
            )
    
    def test_save_config_success(self, config_manager):
        """Test successful configuration save."""
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        result = config_manager.save_config(config)
        
        assert result is True
        
        # Verify file was created
        config_file = os.path.join(
            config_manager.config_storage_path,
            f"{config.name}.json"
        )
        assert os.path.exists(config_file)
    
    def test_save_config_validation_error(self, config_manager):
        """Test configuration save with validation error."""
        config = WorkflowConfig(
            name="",  # Invalid name
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        with pytest.raises(ConfigValidationError):
            config_manager.save_config(config)
    
    def test_load_config_success(self, config_manager):
        """Test successful configuration load."""
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        # Save first
        config_manager.save_config(config)
        
        # Load
        loaded_config = config_manager.load_config("test_workflow")
        
        assert loaded_config is not None
        assert loaded_config.name == "test_workflow"
        assert loaded_config.template == WorkflowTemplate.WEB_DEVELOPMENT
    
    def test_load_config_not_found(self, config_manager):
        """Test loading non-existent configuration."""
        loaded_config = config_manager.load_config("non_existent_workflow")
        
        assert loaded_config is None
    
    def test_delete_config_success(self, config_manager):
        """Test successful configuration deletion."""
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        # Save first
        config_manager.save_config(config)
        
        # Delete
        result = config_manager.delete_config("test_workflow")
        
        assert result is True
        
        # Verify file was deleted
        config_file = os.path.join(
            config_manager.config_storage_path,
            f"{config.name}.json"
        )
        assert not os.path.exists(config_file)
    
    def test_delete_config_not_found(self, config_manager):
        """Test deleting non-existent configuration."""
        result = config_manager.delete_config("non_existent_workflow")
        
        assert result is False
    
    def test_list_configs(self, config_manager):
        """Test listing all configurations."""
        # Save multiple configs
        config1 = WorkflowConfig(name="workflow1", template=WorkflowTemplate.WEB_DEVELOPMENT)
        config2 = WorkflowConfig(name="workflow2", template=WorkflowTemplate.BLOCKCHAIN_DEVELOPMENT)
        
        config_manager.save_config(config1)
        config_manager.save_config(config2)
        
        configs = config_manager.list_configs()
        
        assert len(configs) == 2
        names = {config["name"] for config in configs}
        assert "workflow1" in names
        assert "workflow2" in names
    
    def test_get_available_templates(self, config_manager):
        """Test getting available templates."""
        templates = config_manager.get_available_templates()
        
        assert len(templates) > 0
        assert "web_development" in templates
        assert "blockchain_development" in templates
        assert "ai_model_development" in templates
        assert "data_pipeline_development" in templates
    
    def test_get_template_info(self, config_manager):
        """Test getting template information."""
        template_info = config_manager.get_template_info(WorkflowTemplate.WEB_DEVELOPMENT)
        
        assert template_info is not None
        assert "name" in template_info
        assert "description" in template_info
        assert "default_agents" in template_info
        assert "default_parameters" in template_info
    
    def test_get_template_info_unknown_template(self, config_manager):
        """Test getting information for unknown template."""
        template_info = config_manager.get_template_info("unknown_template")
        
        assert template_info is None
    
    def test_validate_config_success(self, config_manager):
        """Test successful configuration validation."""
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        # Should not raise exception
        config_manager.validate_config(config)
    
    def test_validate_config_failure(self, config_manager):
        """Test configuration validation failure."""
        config = WorkflowConfig(
            name="",  # Invalid name
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        with pytest.raises(ConfigValidationError):
            config_manager.validate_config(config)
    
    def test_export_config(self, config_manager):
        """Test configuration export."""
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        exported_data = config_manager.export_config(config)
        
        assert isinstance(exported_data, str)
        
        # Should be valid JSON
        parsed_data = json.loads(exported_data)
        assert parsed_data["name"] == "test_workflow"
        assert parsed_data["template"] == "web_development"
    
    def test_import_config(self, config_manager):
        """Test configuration import."""
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        exported_data = config_manager.export_config(config)
        imported_config = config_manager.import_config(exported_data)
        
        assert imported_config is not None
        assert imported_config.name == "test_workflow"
        assert imported_config.template == WorkflowTemplate.WEB_DEVELOPMENT
    
    def test_import_config_invalid_data(self, config_manager):
        """Test importing invalid configuration data."""
        with pytest.raises(ConfigValidationError):
            config_manager.import_config("invalid json data")


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_workflow_config(self):
        """Test creating workflow configuration."""
        config = create_workflow_config(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT,
            description="Test workflow"
        )
        
        assert config.name == "test_workflow"
        assert config.template == WorkflowTemplate.WEB_DEVELOPMENT
        assert config.description == "Test workflow"
    
    def test_load_workflow_template(self):
        """Test loading workflow template."""
        template_data = load_workflow_template(WorkflowTemplate.WEB_DEVELOPMENT)
        
        assert template_data is not None
        assert "name" in template_data
        assert "description" in template_data
        assert "default_agents" in template_data
        assert "default_parameters" in template_data
    
    def test_load_workflow_template_unknown(self):
        """Test loading unknown workflow template."""
        with pytest.raises(TemplateNotFoundError):
            load_workflow_template("unknown_template")
    
    def test_validate_workflow_config(self):
        """Test workflow configuration validation."""
        config = WorkflowConfig(
            name="test_workflow",
            template=WorkflowTemplate.WEB_DEVELOPMENT
        )
        
        # Should not raise exception
        validate_workflow_config(config)
    
    def test_create_agent_config(self):
        """Test creating agent configuration."""
        config = create_agent_config(
            agent_type=AgentType.PARSER,
            max_concurrent_tasks=2,
            timeout=300
        )
        
        assert config.agent_type == AgentType.PARSER
        assert config.max_concurrent_tasks == 2
        assert config.timeout == 300
    
    def test_create_workflow_parameter(self):
        """Test creating workflow parameter."""
        param = create_workflow_parameter(
            name="confidence_threshold",
            value=0.8,
            type="float",
            description="Confidence threshold"
        )
        
        assert param.name == "confidence_threshold"
        assert param.value == 0.8
        assert param.type == "float"
        assert param.description == "Confidence threshold"


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_get_config_manager(self):
        """Test getting global config manager."""
        with patch('workflows.workflow_config.WorkflowConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            manager = get_config_manager()
            
            assert manager is not None
            mock_manager_class.assert_called_once()