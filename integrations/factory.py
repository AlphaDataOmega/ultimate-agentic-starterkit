"""
Integration Factory for managing external service integrations.

This module provides a factory pattern for creating and managing integration instances
with health monitoring, fallback mechanisms, and lifecycle management.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from enum import Enum

from core.logger import get_logger
from core.config import get_config
from core.voice_alerts import get_voice_alerts
from .claude_code import ClaudeCodeIntegration
from .git_manager import GitManager
from .ollama_client import OllamaClient


class IntegrationType(str, Enum):
    """Types of integrations available."""
    CLAUDE_CODE = "claude_code"
    GIT_MANAGER = "git_manager"
    OLLAMA_CLIENT = "ollama_client"


class IntegrationStatus(str, Enum):
    """Status of an integration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    INITIALIZING = "initializing"
    STOPPED = "stopped"


class IntegrationInstance:
    """
    Wrapper for integration instances with health monitoring.
    """
    
    def __init__(self, integration_type: IntegrationType, instance: Any, config: Dict[str, Any] = None):
        """
        Initialize integration instance wrapper.
        
        Args:
            integration_type: Type of integration
            instance: The actual integration instance
            config: Configuration for the integration
        """
        self.integration_type = integration_type
        self.instance = instance
        self.config = config or {}
        self.status = IntegrationStatus.INITIALIZING
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        self.error_count = 0
        self.last_error = None
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        
        # Health monitoring
        self.health_history: List[Dict[str, Any]] = []
        self.max_health_history = 100
        
        # Initialize health status
        asyncio.create_task(self._initial_health_check())
    
    async def _initial_health_check(self):
        """Perform initial health check."""
        try:
            health_result = await self.health_check()
            if health_result["healthy"]:
                self.status = IntegrationStatus.HEALTHY
            else:
                self.status = IntegrationStatus.DEGRADED
        except Exception as e:
            self.status = IntegrationStatus.FAILED
            self.last_error = str(e)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the integration.
        
        Returns:
            Dict containing health status
        """
        try:
            current_time = time.time()
            
            # Use cached result if recent
            if (current_time - self.last_health_check) < self.health_check_interval:
                return {
                    "healthy": self.status == IntegrationStatus.HEALTHY,
                    "status": self.status,
                    "cached": True
                }
            
            # Perform health check based on integration type
            if self.integration_type == IntegrationType.CLAUDE_CODE:
                health_result = await self._check_claude_code_health()
            elif self.integration_type == IntegrationType.GIT_MANAGER:
                health_result = await self._check_git_manager_health()
            elif self.integration_type == IntegrationType.OLLAMA_CLIENT:
                health_result = await self._check_ollama_client_health()
            else:
                health_result = {"healthy": False, "error": "Unknown integration type"}
            
            # Update status based on health check
            if health_result["healthy"]:
                self.status = IntegrationStatus.HEALTHY
                self.error_count = 0
            else:
                self.error_count += 1
                self.last_error = health_result.get("error", "Unknown error")
                
                if self.error_count >= 3:
                    self.status = IntegrationStatus.FAILED
                else:
                    self.status = IntegrationStatus.DEGRADED
            
            # Update timestamps
            self.last_health_check = current_time
            
            # Add to health history
            self._add_to_health_history(health_result)
            
            return {
                "healthy": self.status == IntegrationStatus.HEALTHY,
                "status": self.status,
                "error_count": self.error_count,
                "last_error": self.last_error,
                "timestamp": datetime.now().isoformat(),
                **health_result
            }
            
        except Exception as e:
            self.status = IntegrationStatus.FAILED
            self.last_error = str(e)
            return {
                "healthy": False,
                "status": self.status,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_claude_code_health(self) -> Dict[str, Any]:
        """Check Claude Code integration health."""
        try:
            status = self.instance.get_status()
            
            healthy = (
                status["claude_dir_exists"] and
                status["commands_dir_exists"] and
                status["registered_commands"] > 0
            )
            
            return {
                "healthy": healthy,
                "details": status
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_git_manager_health(self) -> Dict[str, Any]:
        """Check Git Manager integration health."""
        try:
            status = self.instance.get_status()
            
            healthy = status.get("available", False)
            
            return {
                "healthy": healthy,
                "details": status
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_ollama_client_health(self) -> Dict[str, Any]:
        """Check Ollama Client integration health."""
        try:
            health_result = await self.instance.health_check()
            
            return {
                "healthy": health_result.get("healthy", False),
                "details": health_result
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    def _add_to_health_history(self, health_result: Dict[str, Any]):
        """Add health check result to history."""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "healthy": health_result.get("healthy", False),
            "status": self.status,
            "error": health_result.get("error"),
            "details": health_result.get("details", {})
        }
        
        self.health_history.append(history_entry)
        
        # Maintain history size limit
        if len(self.health_history) > self.max_health_history:
            self.health_history.pop(0)
    
    def update_last_used(self):
        """Update last used timestamp."""
        self.last_used = datetime.now()
    
    def get_info(self) -> Dict[str, Any]:
        """Get integration instance information."""
        return {
            "integration_type": self.integration_type,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "health_history_count": len(self.health_history),
            "config": self.config
        }


class IntegrationFactory:
    """
    Factory for creating and managing integration instances.
    
    Provides centralized management of external service integrations with
    health monitoring, fallback mechanisms, and lifecycle management.
    """
    
    def __init__(self, workspace_root: str = None):
        """
        Initialize integration factory.
        
        Args:
            workspace_root: Root directory for workspace-specific integrations
        """
        self.workspace_root = Path(workspace_root or ".")
        self.logger = get_logger("integration_factory")
        self.config = get_config()
        self.voice = get_voice_alerts()
        
        # Integration instances
        self.instances: Dict[str, IntegrationInstance] = {}
        self.instance_configs: Dict[IntegrationType, Dict[str, Any]] = {}
        
        # Factory configuration
        self.auto_health_check = True
        self.health_check_interval = 60  # seconds
        self.max_instances_per_type = 3
        
        # Background monitoring
        self.monitoring_task = None
        self.monitoring_enabled = False
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        # Start monitoring
        self._start_monitoring()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for integrations."""
        self.instance_configs = {
            IntegrationType.CLAUDE_CODE: {
                "workspace_root": str(self.workspace_root),
                "auto_setup": True
            },
            IntegrationType.GIT_MANAGER: {
                "repo_path": str(self.workspace_root),
                "validation_gates": {
                    "linting": True,
                    "type_checking": True,
                    "unit_tests": True
                }
            },
            IntegrationType.OLLAMA_CLIENT: {
                "base_url": "http://localhost:11434",
                "timeout": 30.0,
                "auto_pull_models": False
            }
        }
    
    def _start_monitoring(self):
        """Start background monitoring of integrations."""
        if self.auto_health_check and not self.monitoring_enabled:
            self.monitoring_enabled = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Integration monitoring started")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self._monitor_all_instances()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Short delay before retrying
    
    async def _monitor_all_instances(self):
        """Monitor all integration instances."""
        for instance_id, instance in self.instances.items():
            try:
                health_result = await instance.health_check()
                
                if not health_result["healthy"]:
                    self.logger.warning(f"Integration {instance_id} is unhealthy: {health_result.get('error', 'Unknown error')}")
                    
                    # Notify via voice if status changed to failed
                    if instance.status == IntegrationStatus.FAILED:
                        self.voice.speak_warning(f"Integration {instance.integration_type} has failed")
                        
            except Exception as e:
                self.logger.error(f"Error monitoring instance {instance_id}: {e}")
    
    async def create_integration(self, 
                               integration_type: IntegrationType,
                               instance_id: str = None,
                               config: Dict[str, Any] = None) -> Optional[IntegrationInstance]:
        """
        Create a new integration instance.
        
        Args:
            integration_type: Type of integration to create
            instance_id: Unique identifier for the instance
            config: Configuration for the integration
        
        Returns:
            IntegrationInstance or None if creation failed
        """
        try:
            # Generate instance ID if not provided
            if not instance_id:
                instance_id = f"{integration_type}_{int(time.time())}"
            
            # Check if instance already exists
            if instance_id in self.instances:
                self.logger.warning(f"Integration instance {instance_id} already exists")
                return self.instances[instance_id]
            
            # Check instance limit
            type_count = sum(1 for inst in self.instances.values() if inst.integration_type == integration_type)
            if type_count >= self.max_instances_per_type:
                self.logger.error(f"Maximum instances for {integration_type} reached")
                return None
            
            # Merge configuration
            merged_config = self.instance_configs.get(integration_type, {}).copy()
            if config:
                merged_config.update(config)
            
            # Create integration instance
            instance = await self._create_instance(integration_type, merged_config)
            
            if instance:
                # Wrap in IntegrationInstance
                integration_instance = IntegrationInstance(integration_type, instance, merged_config)
                self.instances[instance_id] = integration_instance
                
                self.logger.info(f"Created integration instance: {instance_id}")
                self.voice.speak_success(f"Integration {integration_type} created")
                
                return integration_instance
            else:
                self.logger.error(f"Failed to create integration instance: {integration_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating integration {integration_type}: {e}")
            self.voice.speak_error(f"Integration creation failed: {str(e)}")
            return None
    
    async def _create_instance(self, integration_type: IntegrationType, config: Dict[str, Any]) -> Optional[Any]:
        """Create the actual integration instance."""
        try:
            if integration_type == IntegrationType.CLAUDE_CODE:
                instance = ClaudeCodeIntegration(config.get("workspace_root", str(self.workspace_root)))
                
                if config.get("auto_setup", True):
                    await instance.setup_commands()
                    instance.setup_event_handlers()
                
                return instance
                
            elif integration_type == IntegrationType.GIT_MANAGER:
                instance = GitManager(config.get("repo_path", str(self.workspace_root)))
                
                # Configure validation gates
                if "validation_gates" in config:
                    instance.configure_validation_gates(**config["validation_gates"])
                
                return instance
                
            elif integration_type == IntegrationType.OLLAMA_CLIENT:
                instance = OllamaClient(
                    base_url=config.get("base_url", "http://localhost:11434"),
                    timeout=config.get("timeout", 30.0)
                )
                
                # Auto-pull models if configured
                if config.get("auto_pull_models", False):
                    await instance.list_models()
                
                return instance
                
            else:
                self.logger.error(f"Unknown integration type: {integration_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating {integration_type} instance: {e}")
            return None
    
    def get_integration(self, instance_id: str) -> Optional[IntegrationInstance]:
        """
        Get integration instance by ID.
        
        Args:
            instance_id: Instance identifier
        
        Returns:
            IntegrationInstance or None if not found
        """
        instance = self.instances.get(instance_id)
        
        if instance:
            instance.update_last_used()
            return instance
        
        return None
    
    def get_integrations_by_type(self, integration_type: IntegrationType) -> List[IntegrationInstance]:
        """
        Get all integration instances of a specific type.
        
        Args:
            integration_type: Type of integration
        
        Returns:
            List of IntegrationInstance objects
        """
        return [
            instance for instance in self.instances.values()
            if instance.integration_type == integration_type
        ]
    
    def get_healthy_integration(self, integration_type: IntegrationType) -> Optional[IntegrationInstance]:
        """
        Get first healthy integration instance of a specific type.
        
        Args:
            integration_type: Type of integration
        
        Returns:
            IntegrationInstance or None if no healthy instance found
        """
        for instance in self.get_integrations_by_type(integration_type):
            if instance.status == IntegrationStatus.HEALTHY:
                instance.update_last_used()
                return instance
        
        return None
    
    async def get_or_create_integration(self, 
                                      integration_type: IntegrationType,
                                      instance_id: str = None,
                                      config: Dict[str, Any] = None) -> Optional[IntegrationInstance]:
        """
        Get existing integration or create new one.
        
        Args:
            integration_type: Type of integration
            instance_id: Instance identifier
            config: Configuration for creation
        
        Returns:
            IntegrationInstance or None if creation failed
        """
        # Try to get existing healthy instance
        if instance_id:
            instance = self.get_integration(instance_id)
            if instance and instance.status == IntegrationStatus.HEALTHY:
                return instance
        
        # Try to get any healthy instance of this type
        healthy_instance = self.get_healthy_integration(integration_type)
        if healthy_instance:
            return healthy_instance
        
        # Create new instance
        return await self.create_integration(integration_type, instance_id, config)
    
    async def remove_integration(self, instance_id: str) -> bool:
        """
        Remove integration instance.
        
        Args:
            instance_id: Instance identifier
        
        Returns:
            True if removed successfully
        """
        try:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                
                # Cleanup instance if it has cleanup method
                if hasattr(instance.instance, 'close'):
                    await instance.instance.close()
                
                del self.instances[instance_id]
                
                self.logger.info(f"Removed integration instance: {instance_id}")
                return True
            else:
                self.logger.warning(f"Integration instance {instance_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing integration {instance_id}: {e}")
            return False
    
    async def cleanup_unhealthy_instances(self):
        """Remove all unhealthy integration instances."""
        unhealthy_instances = []
        
        for instance_id, instance in self.instances.items():
            if instance.status == IntegrationStatus.FAILED:
                unhealthy_instances.append(instance_id)
        
        for instance_id in unhealthy_instances:
            await self.remove_integration(instance_id)
        
        if unhealthy_instances:
            self.logger.info(f"Cleaned up {len(unhealthy_instances)} unhealthy instances")
    
    def configure_integration_type(self, integration_type: IntegrationType, config: Dict[str, Any]):
        """
        Configure default settings for integration type.
        
        Args:
            integration_type: Type of integration
            config: Configuration dictionary
        """
        if integration_type not in self.instance_configs:
            self.instance_configs[integration_type] = {}
        
        self.instance_configs[integration_type].update(config)
        self.logger.info(f"Updated configuration for {integration_type}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall factory status."""
        status_by_type = {}
        
        for integration_type in IntegrationType:
            instances = self.get_integrations_by_type(integration_type)
            status_by_type[integration_type] = {
                "total_instances": len(instances),
                "healthy_instances": sum(1 for inst in instances if inst.status == IntegrationStatus.HEALTHY),
                "degraded_instances": sum(1 for inst in instances if inst.status == IntegrationStatus.DEGRADED),
                "failed_instances": sum(1 for inst in instances if inst.status == IntegrationStatus.FAILED)
            }
        
        return {
            "total_instances": len(self.instances),
            "monitoring_enabled": self.monitoring_enabled,
            "workspace_root": str(self.workspace_root),
            "status_by_type": status_by_type,
            "instance_configs": self.instance_configs
        }
    
    def get_all_instances(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all instances."""
        return {
            instance_id: instance.get_info()
            for instance_id, instance in self.instances.items()
        }
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring_enabled:
            self.monitoring_enabled = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Integration monitoring stopped")
    
    async def cleanup(self):
        """Clean up all resources."""
        await self.stop_monitoring()
        
        # Clean up all instances
        for instance_id in list(self.instances.keys()):
            await self.remove_integration(instance_id)
        
        self.logger.info("Integration factory cleaned up")


# Global factory instance
_factory_instance: Optional[IntegrationFactory] = None


def get_integration_factory(workspace_root: str = None) -> IntegrationFactory:
    """
    Get global integration factory instance.
    
    Args:
        workspace_root: Root directory for workspace-specific integrations
    
    Returns:
        IntegrationFactory instance
    """
    global _factory_instance
    
    if _factory_instance is None:
        _factory_instance = IntegrationFactory(workspace_root)
    
    return _factory_instance


# Convenience functions
async def get_claude_code_integration(workspace_root: str = None) -> Optional[ClaudeCodeIntegration]:
    """Get Claude Code integration instance."""
    factory = get_integration_factory(workspace_root)
    instance = await factory.get_or_create_integration(IntegrationType.CLAUDE_CODE)
    return instance.instance if instance else None


async def get_git_manager(repo_path: str = None) -> Optional[GitManager]:
    """Get Git Manager integration instance."""
    factory = get_integration_factory(repo_path)
    instance = await factory.get_or_create_integration(IntegrationType.GIT_MANAGER)
    return instance.instance if instance else None


async def get_ollama_client(base_url: str = "http://localhost:11434") -> Optional[OllamaClient]:
    """Get Ollama Client integration instance."""
    factory = get_integration_factory()
    config = {"base_url": base_url}
    instance = await factory.get_or_create_integration(IntegrationType.OLLAMA_CLIENT, config=config)
    return instance.instance if instance else None