"""
Unit tests for Integration Factory.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import shutil

from StarterKit.integrations.factory import (
    IntegrationFactory, 
    IntegrationInstance, 
    IntegrationType, 
    IntegrationStatus,
    get_integration_factory
)


class TestIntegrationInstance:
    """Test suite for Integration Instance."""
    
    @pytest.fixture
    def mock_integration(self):
        """Create mock integration."""
        mock_integration = Mock()
        mock_integration.get_status.return_value = {
            "healthy": True,
            "details": "All good"
        }
        return mock_integration
    
    @pytest.fixture
    def integration_instance(self, mock_integration):
        """Create integration instance."""
        return IntegrationInstance(
            IntegrationType.CLAUDE_CODE,
            mock_integration,
            {"test": "config"}
        )
    
    def test_initialization(self, integration_instance):
        """Test integration instance initialization."""
        assert integration_instance.integration_type == IntegrationType.CLAUDE_CODE
        assert integration_instance.config == {"test": "config"}
        assert integration_instance.status == IntegrationStatus.INITIALIZING
        assert integration_instance.error_count == 0
    
    @pytest.mark.asyncio
    async def test_health_check_claude_code(self, integration_instance):
        """Test health check for Claude Code integration."""
        # Wait for initial health check to complete
        await asyncio.sleep(0.1)
        
        result = await integration_instance.health_check()
        
        assert result["healthy"] is True
        assert integration_instance.status == IntegrationStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_health_check_cached(self, integration_instance):
        """Test cached health check."""
        # First health check
        result1 = await integration_instance.health_check()
        
        # Second health check within cache interval
        result2 = await integration_instance.health_check()
        
        assert result1["healthy"] is True
        assert result2["cached"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, integration_instance, mock_integration):
        """Test health check failure."""
        mock_integration.get_status.return_value = {
            "claude_dir_exists": False,
            "commands_dir_exists": False,
            "registered_commands": 0
        }
        
        result = await integration_instance.health_check()
        
        assert result["healthy"] is False
        assert integration_instance.status == IntegrationStatus.DEGRADED
        assert integration_instance.error_count == 1
    
    @pytest.mark.asyncio
    async def test_health_check_multiple_failures(self, integration_instance, mock_integration):
        """Test multiple health check failures."""
        mock_integration.get_status.side_effect = Exception("Connection failed")
        
        # Multiple failures
        for _ in range(3):
            await integration_instance.health_check()
        
        assert integration_instance.status == IntegrationStatus.FAILED
        assert integration_instance.error_count == 3
    
    def test_update_last_used(self, integration_instance):
        """Test updating last used timestamp."""
        original_time = integration_instance.last_used
        
        integration_instance.update_last_used()
        
        assert integration_instance.last_used > original_time
    
    def test_get_info(self, integration_instance):
        """Test getting integration info."""
        info = integration_instance.get_info()
        
        assert info["integration_type"] == IntegrationType.CLAUDE_CODE
        assert info["status"] == IntegrationStatus.INITIALIZING
        assert info["error_count"] == 0
        assert info["config"] == {"test": "config"}


class TestIntegrationFactory:
    """Test suite for Integration Factory."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def integration_factory(self, temp_workspace):
        """Create integration factory."""
        return IntegrationFactory(temp_workspace)
    
    def test_initialization(self, integration_factory, temp_workspace):
        """Test integration factory initialization."""
        assert str(integration_factory.workspace_root) == temp_workspace
        assert integration_factory.auto_health_check is True
        assert integration_factory.monitoring_enabled is True
        assert len(integration_factory.instance_configs) == 3
    
    def test_default_configs(self, integration_factory):
        """Test default configurations."""
        configs = integration_factory.instance_configs
        
        assert IntegrationType.CLAUDE_CODE in configs
        assert IntegrationType.GIT_MANAGER in configs
        assert IntegrationType.OLLAMA_CLIENT in configs
        
        assert configs[IntegrationType.CLAUDE_CODE]["auto_setup"] is True
        assert configs[IntegrationType.GIT_MANAGER]["validation_gates"]["linting"] is True
        assert configs[IntegrationType.OLLAMA_CLIENT]["base_url"] == "http://localhost:11434"
    
    @pytest.mark.asyncio
    async def test_create_integration_claude_code(self, integration_factory):
        """Test creating Claude Code integration."""
        with patch('StarterKit.integrations.factory.ClaudeCodeIntegration') as mock_cls:
            mock_instance = Mock()
            mock_instance.setup_commands = AsyncMock()
            mock_instance.setup_event_handlers = Mock()
            mock_cls.return_value = mock_instance
            
            result = await integration_factory.create_integration(
                IntegrationType.CLAUDE_CODE,
                "test_instance"
            )
            
            assert result is not None
            assert result.integration_type == IntegrationType.CLAUDE_CODE
            assert "test_instance" in integration_factory.instances
            
            mock_instance.setup_commands.assert_called_once()
            mock_instance.setup_event_handlers.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_integration_git_manager(self, integration_factory):
        """Test creating Git Manager integration."""
        with patch('StarterKit.integrations.factory.GitManager') as mock_cls:
            mock_instance = Mock()
            mock_instance.configure_validation_gates = Mock()
            mock_cls.return_value = mock_instance
            
            result = await integration_factory.create_integration(
                IntegrationType.GIT_MANAGER,
                "test_git"
            )
            
            assert result is not None
            assert result.integration_type == IntegrationType.GIT_MANAGER
            
            mock_instance.configure_validation_gates.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_integration_ollama_client(self, integration_factory):
        """Test creating Ollama Client integration."""
        with patch('StarterKit.integrations.factory.OllamaClient') as mock_cls:
            mock_instance = Mock()
            mock_instance.list_models = AsyncMock()
            mock_cls.return_value = mock_instance
            
            result = await integration_factory.create_integration(
                IntegrationType.OLLAMA_CLIENT,
                "test_ollama",
                {"auto_pull_models": True}
            )
            
            assert result is not None
            assert result.integration_type == IntegrationType.OLLAMA_CLIENT
            
            mock_instance.list_models.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_integration_duplicate_id(self, integration_factory):
        """Test creating integration with duplicate ID."""
        with patch('StarterKit.integrations.factory.ClaudeCodeIntegration') as mock_cls:
            mock_instance = Mock()
            mock_instance.setup_commands = AsyncMock()
            mock_instance.setup_event_handlers = Mock()
            mock_cls.return_value = mock_instance
            
            # Create first instance
            result1 = await integration_factory.create_integration(
                IntegrationType.CLAUDE_CODE,
                "duplicate_id"
            )
            
            # Try to create with same ID
            result2 = await integration_factory.create_integration(
                IntegrationType.CLAUDE_CODE,
                "duplicate_id"
            )
            
            assert result1 is result2  # Should return existing instance
    
    @pytest.mark.asyncio
    async def test_create_integration_max_instances(self, integration_factory):
        """Test creating integration with max instances reached."""
        integration_factory.max_instances_per_type = 1
        
        with patch('StarterKit.integrations.factory.ClaudeCodeIntegration') as mock_cls:
            mock_instance = Mock()
            mock_instance.setup_commands = AsyncMock()
            mock_instance.setup_event_handlers = Mock()
            mock_cls.return_value = mock_instance
            
            # Create first instance
            result1 = await integration_factory.create_integration(
                IntegrationType.CLAUDE_CODE,
                "instance1"
            )
            
            # Try to create second instance
            result2 = await integration_factory.create_integration(
                IntegrationType.CLAUDE_CODE,
                "instance2"
            )
            
            assert result1 is not None
            assert result2 is None  # Should fail due to limit
    
    @pytest.mark.asyncio
    async def test_create_integration_creation_failure(self, integration_factory):
        """Test integration creation failure."""
        with patch('StarterKit.integrations.factory.ClaudeCodeIntegration') as mock_cls:
            mock_cls.side_effect = Exception("Creation failed")
            
            result = await integration_factory.create_integration(
                IntegrationType.CLAUDE_CODE,
                "failed_instance"
            )
            
            assert result is None
    
    def test_get_integration(self, integration_factory):
        """Test getting integration instance."""
        # Create mock instance
        mock_instance = Mock()
        mock_instance.update_last_used = Mock()
        integration_factory.instances["test_id"] = mock_instance
        
        result = integration_factory.get_integration("test_id")
        
        assert result is mock_instance
        mock_instance.update_last_used.assert_called_once()
    
    def test_get_integration_not_found(self, integration_factory):
        """Test getting non-existent integration."""
        result = integration_factory.get_integration("nonexistent")
        
        assert result is None
    
    def test_get_integrations_by_type(self, integration_factory):
        """Test getting integrations by type."""
        # Create mock instances
        mock_instance1 = Mock()
        mock_instance1.integration_type = IntegrationType.CLAUDE_CODE
        mock_instance2 = Mock()
        mock_instance2.integration_type = IntegrationType.GIT_MANAGER
        mock_instance3 = Mock()
        mock_instance3.integration_type = IntegrationType.CLAUDE_CODE
        
        integration_factory.instances["id1"] = mock_instance1
        integration_factory.instances["id2"] = mock_instance2
        integration_factory.instances["id3"] = mock_instance3
        
        result = integration_factory.get_integrations_by_type(IntegrationType.CLAUDE_CODE)
        
        assert len(result) == 2
        assert mock_instance1 in result
        assert mock_instance3 in result
    
    def test_get_healthy_integration(self, integration_factory):
        """Test getting healthy integration."""
        # Create mock instances
        mock_instance1 = Mock()
        mock_instance1.integration_type = IntegrationType.CLAUDE_CODE
        mock_instance1.status = IntegrationStatus.FAILED
        mock_instance1.update_last_used = Mock()
        
        mock_instance2 = Mock()
        mock_instance2.integration_type = IntegrationType.CLAUDE_CODE
        mock_instance2.status = IntegrationStatus.HEALTHY
        mock_instance2.update_last_used = Mock()
        
        integration_factory.instances["id1"] = mock_instance1
        integration_factory.instances["id2"] = mock_instance2
        
        result = integration_factory.get_healthy_integration(IntegrationType.CLAUDE_CODE)
        
        assert result is mock_instance2
        mock_instance2.update_last_used.assert_called_once()
    
    def test_get_healthy_integration_none(self, integration_factory):
        """Test getting healthy integration when none available."""
        # Create mock instance that's not healthy
        mock_instance = Mock()
        mock_instance.integration_type = IntegrationType.CLAUDE_CODE
        mock_instance.status = IntegrationStatus.FAILED
        
        integration_factory.instances["id1"] = mock_instance
        
        result = integration_factory.get_healthy_integration(IntegrationType.CLAUDE_CODE)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_or_create_integration_existing_healthy(self, integration_factory):
        """Test getting or creating existing healthy integration."""
        # Create mock healthy instance
        mock_instance = Mock()
        mock_instance.status = IntegrationStatus.HEALTHY
        mock_instance.update_last_used = Mock()
        integration_factory.instances["test_id"] = mock_instance
        
        result = await integration_factory.get_or_create_integration(
            IntegrationType.CLAUDE_CODE,
            "test_id"
        )
        
        assert result is mock_instance
        mock_instance.update_last_used.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_or_create_integration_create_new(self, integration_factory):
        """Test getting or creating new integration."""
        with patch.object(integration_factory, 'create_integration') as mock_create:
            mock_instance = Mock()
            mock_create.return_value = mock_instance
            
            result = await integration_factory.get_or_create_integration(
                IntegrationType.CLAUDE_CODE,
                "new_id"
            )
            
            assert result is mock_instance
            mock_create.assert_called_once_with(
                IntegrationType.CLAUDE_CODE,
                "new_id",
                None
            )
    
    @pytest.mark.asyncio
    async def test_remove_integration(self, integration_factory):
        """Test removing integration."""
        # Create mock instance with close method
        mock_instance = Mock()
        mock_instance.instance = Mock()
        mock_instance.instance.close = AsyncMock()
        
        integration_factory.instances["test_id"] = mock_instance
        
        result = await integration_factory.remove_integration("test_id")
        
        assert result is True
        assert "test_id" not in integration_factory.instances
        mock_instance.instance.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_integration_not_found(self, integration_factory):
        """Test removing non-existent integration."""
        result = await integration_factory.remove_integration("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cleanup_unhealthy_instances(self, integration_factory):
        """Test cleaning up unhealthy instances."""
        # Create mock instances
        mock_healthy = Mock()
        mock_healthy.status = IntegrationStatus.HEALTHY
        mock_failed = Mock()
        mock_failed.status = IntegrationStatus.FAILED
        mock_failed.instance = Mock()
        mock_failed.instance.close = AsyncMock()
        
        integration_factory.instances["healthy"] = mock_healthy
        integration_factory.instances["failed"] = mock_failed
        
        await integration_factory.cleanup_unhealthy_instances()
        
        assert "healthy" in integration_factory.instances
        assert "failed" not in integration_factory.instances
    
    def test_configure_integration_type(self, integration_factory):
        """Test configuring integration type."""
        integration_factory.configure_integration_type(
            IntegrationType.CLAUDE_CODE,
            {"custom_setting": "value"}
        )
        
        config = integration_factory.instance_configs[IntegrationType.CLAUDE_CODE]
        assert config["custom_setting"] == "value"
    
    def test_get_status(self, integration_factory):
        """Test getting factory status."""
        # Create mock instances
        mock_healthy = Mock()
        mock_healthy.integration_type = IntegrationType.CLAUDE_CODE
        mock_healthy.status = IntegrationStatus.HEALTHY
        
        mock_failed = Mock()
        mock_failed.integration_type = IntegrationType.CLAUDE_CODE
        mock_failed.status = IntegrationStatus.FAILED
        
        integration_factory.instances["id1"] = mock_healthy
        integration_factory.instances["id2"] = mock_failed
        
        status = integration_factory.get_status()
        
        assert status["total_instances"] == 2
        assert status["monitoring_enabled"] is True
        
        claude_code_status = status["status_by_type"][IntegrationType.CLAUDE_CODE]
        assert claude_code_status["total_instances"] == 2
        assert claude_code_status["healthy_instances"] == 1
        assert claude_code_status["failed_instances"] == 1
    
    def test_get_all_instances(self, integration_factory):
        """Test getting all instances info."""
        # Create mock instance
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"test": "info"}
        
        integration_factory.instances["test_id"] = mock_instance
        
        result = integration_factory.get_all_instances()
        
        assert result == {"test_id": {"test": "info"}}
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, integration_factory):
        """Test stopping monitoring."""
        # Mock the monitoring task
        integration_factory.monitoring_task = Mock()
        integration_factory.monitoring_task.cancel = Mock()
        
        await integration_factory.stop_monitoring()
        
        assert integration_factory.monitoring_enabled is False
        integration_factory.monitoring_task.cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, integration_factory):
        """Test factory cleanup."""
        # Create mock instance
        mock_instance = Mock()
        mock_instance.instance = Mock()
        mock_instance.instance.close = AsyncMock()
        
        integration_factory.instances["test_id"] = mock_instance
        
        with patch.object(integration_factory, 'stop_monitoring') as mock_stop:
            await integration_factory.cleanup()
            
            mock_stop.assert_called_once()
            assert len(integration_factory.instances) == 0


class TestGlobalFactory:
    """Test suite for global factory functions."""
    
    def test_get_integration_factory(self):
        """Test getting global integration factory."""
        factory1 = get_integration_factory()
        factory2 = get_integration_factory()
        
        assert factory1 is factory2  # Should be singleton
    
    @pytest.mark.asyncio
    async def test_get_claude_code_integration(self):
        """Test getting Claude Code integration."""
        with patch('StarterKit.integrations.factory.get_integration_factory') as mock_factory:
            mock_factory_instance = Mock()
            mock_integration_instance = Mock()
            mock_integration_instance.instance = "claude_code_instance"
            mock_factory_instance.get_or_create_integration = AsyncMock(return_value=mock_integration_instance)
            mock_factory.return_value = mock_factory_instance
            
            from StarterKit.integrations.factory import get_claude_code_integration
            
            result = await get_claude_code_integration()
            
            assert result == "claude_code_instance"
    
    @pytest.mark.asyncio
    async def test_get_git_manager(self):
        """Test getting Git Manager integration."""
        with patch('StarterKit.integrations.factory.get_integration_factory') as mock_factory:
            mock_factory_instance = Mock()
            mock_integration_instance = Mock()
            mock_integration_instance.instance = "git_manager_instance"
            mock_factory_instance.get_or_create_integration = AsyncMock(return_value=mock_integration_instance)
            mock_factory.return_value = mock_factory_instance
            
            from StarterKit.integrations.factory import get_git_manager
            
            result = await get_git_manager()
            
            assert result == "git_manager_instance"
    
    @pytest.mark.asyncio
    async def test_get_ollama_client(self):
        """Test getting Ollama Client integration."""
        with patch('StarterKit.integrations.factory.get_integration_factory') as mock_factory:
            mock_factory_instance = Mock()
            mock_integration_instance = Mock()
            mock_integration_instance.instance = "ollama_client_instance"
            mock_factory_instance.get_or_create_integration = AsyncMock(return_value=mock_integration_instance)
            mock_factory.return_value = mock_factory_instance
            
            from StarterKit.integrations.factory import get_ollama_client
            
            result = await get_ollama_client()
            
            assert result == "ollama_client_instance"