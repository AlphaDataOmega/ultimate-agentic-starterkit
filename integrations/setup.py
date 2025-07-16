"""
Setup script for external integrations.

This script initializes the integration system, creates necessary directories,
and sets up the Claude Code extension configuration.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ..core.logger import get_logger
from ..core.config import get_config
from ..core.voice_alerts import get_voice_alerts
from .factory import get_integration_factory, IntegrationType


class IntegrationSetup:
    """
    Setup manager for external integrations.
    """
    
    def __init__(self, workspace_root: str = None):
        """
        Initialize setup manager.
        
        Args:
            workspace_root: Root directory for workspace
        """
        self.workspace_root = Path(workspace_root or os.getcwd())
        self.logger = get_logger("integration_setup")
        self.config = get_config()
        self.voice = get_voice_alerts()
        
        # Setup paths
        self.claude_dir = self.workspace_root / ".claude"
        self.commands_dir = self.claude_dir / "commands"
        self.hooks_dir = self.claude_dir / "hooks"
        self.config_file = self.claude_dir / "config.json"
        
        # Integration factory
        self.factory = get_integration_factory(str(self.workspace_root))
    
    async def setup_all(self) -> Dict[str, Any]:
        """
        Setup all integrations.
        
        Returns:
            Dict containing setup results
        """
        self.logger.info("Starting integration setup")
        self.voice.speak("Setting up external integrations")
        
        results = {
            "success": True,
            "directories_created": [],
            "integrations_setup": [],
            "errors": []
        }
        
        try:
            # Create directories
            await self._create_directories()
            results["directories_created"] = [
                str(self.claude_dir),
                str(self.commands_dir),
                str(self.hooks_dir)
            ]
            
            # Setup integrations
            claude_code_result = await self._setup_claude_code()
            git_manager_result = await self._setup_git_manager()
            ollama_client_result = await self._setup_ollama_client()
            
            # Collect results
            if claude_code_result["success"]:
                results["integrations_setup"].append("claude_code")
            else:
                results["errors"].append(f"Claude Code setup failed: {claude_code_result['error']}")
            
            if git_manager_result["success"]:
                results["integrations_setup"].append("git_manager")
            else:
                results["errors"].append(f"Git Manager setup failed: {git_manager_result['error']}")
            
            if ollama_client_result["success"]:
                results["integrations_setup"].append("ollama_client")
            else:
                results["errors"].append(f"Ollama Client setup failed: {ollama_client_result['error']}")
            
            # Update main config
            await self._update_config()
            
            # Final success check
            if results["errors"]:
                results["success"] = False
                self.voice.speak_warning("Integration setup completed with errors")
            else:
                self.voice.speak_success("All integrations setup successfully")
            
            self.logger.info(f"Integration setup completed: {len(results['integrations_setup'])}/3 successful")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            self.logger.error(f"Integration setup failed: {e}")
            self.voice.speak_error(f"Integration setup failed: {str(e)}")
        
        return results
    
    async def _create_directories(self):
        """Create necessary directories."""
        directories = [self.claude_dir, self.commands_dir, self.hooks_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    async def _setup_claude_code(self) -> Dict[str, Any]:
        """Setup Claude Code integration."""
        try:
            self.logger.info("Setting up Claude Code integration")
            
            # Create integration instance
            instance = await self.factory.create_integration(
                IntegrationType.CLAUDE_CODE,
                "default",
                {"workspace_root": str(self.workspace_root)}
            )
            
            if instance:
                # Setup commands
                await instance.instance.setup_commands()
                instance.instance.setup_event_handlers()
                
                return {
                    "success": True,
                    "message": "Claude Code integration setup successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create Claude Code integration instance"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _setup_git_manager(self) -> Dict[str, Any]:
        """Setup Git Manager integration."""
        try:
            self.logger.info("Setting up Git Manager integration")
            
            # Create integration instance
            instance = await self.factory.create_integration(
                IntegrationType.GIT_MANAGER,
                "default",
                {"repo_path": str(self.workspace_root)}
            )
            
            if instance:
                # Configure validation gates
                instance.instance.configure_validation_gates(
                    linting=True,
                    type_checking=True,
                    unit_tests=True,
                    security_scan=False,
                    dependency_check=False
                )
                
                return {
                    "success": True,
                    "message": "Git Manager integration setup successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create Git Manager integration instance"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _setup_ollama_client(self) -> Dict[str, Any]:
        """Setup Ollama Client integration."""
        try:
            self.logger.info("Setting up Ollama Client integration")
            
            # Create integration instance
            instance = await self.factory.create_integration(
                IntegrationType.OLLAMA_CLIENT,
                "default",
                {
                    "base_url": "http://localhost:11434",
                    "timeout": 30.0,
                    "auto_pull_models": False
                }
            )
            
            if instance:
                # Test connection
                health_result = await instance.instance.health_check()
                
                if health_result["healthy"]:
                    self.logger.info("Ollama service is available")
                else:
                    self.logger.warning("Ollama service is not available - integration will work when service is started")
                
                return {
                    "success": True,
                    "message": "Ollama Client integration setup successfully",
                    "service_available": health_result["healthy"]
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create Ollama Client integration instance"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _update_config(self):
        """Update main configuration file."""
        config_data = {
            "version": "1.0.0",
            "workspace_root": str(self.workspace_root),
            "commands_dir": str(self.commands_dir),
            "hooks_dir": str(self.hooks_dir),
            "integrations": {
                "claude_code": {
                    "enabled": True,
                    "auto_setup": True
                },
                "git_manager": {
                    "enabled": True,
                    "validation_gates": {
                        "linting": True,
                        "type_checking": True,
                        "unit_tests": True,
                        "security_scan": False,
                        "dependency_check": False
                    }
                },
                "ollama_client": {
                    "enabled": True,
                    "base_url": "http://localhost:11434",
                    "timeout": 30.0,
                    "auto_pull_models": False,
                    "default_models": {
                        "chat": "llama3.2:latest",
                        "code": "codellama:latest",
                        "tools": "mistral:latest"
                    }
                }
            },
            "monitoring": {
                "enabled": True,
                "health_check_interval": 60,
                "max_instances_per_type": 3
            },
            "voice_alerts": {
                "enabled": True,
                "milestone_alerts": True,
                "error_alerts": True
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Updated configuration file: {self.config_file}")
    
    async def validate_setup(self) -> Dict[str, Any]:
        """
        Validate the integration setup.
        
        Returns:
            Dict containing validation results
        """
        self.logger.info("Validating integration setup")
        
        results = {
            "success": True,
            "checks_passed": [],
            "checks_failed": [],
            "integration_status": {}
        }
        
        try:
            # Check directories
            if self.claude_dir.exists():
                results["checks_passed"].append("claude_dir_exists")
            else:
                results["checks_failed"].append("claude_dir_missing")
            
            if self.commands_dir.exists():
                results["checks_passed"].append("commands_dir_exists")
            else:
                results["checks_failed"].append("commands_dir_missing")
            
            if self.config_file.exists():
                results["checks_passed"].append("config_file_exists")
            else:
                results["checks_failed"].append("config_file_missing")
            
            # Check integrations
            factory_status = self.factory.get_status()
            results["integration_status"] = factory_status
            
            # Check each integration type
            for integration_type in IntegrationType:
                instances = self.factory.get_integrations_by_type(integration_type)
                
                if instances:
                    healthy_count = sum(1 for inst in instances if inst.status.value == "healthy")
                    if healthy_count > 0:
                        results["checks_passed"].append(f"{integration_type}_healthy")
                    else:
                        results["checks_failed"].append(f"{integration_type}_unhealthy")
                else:
                    results["checks_failed"].append(f"{integration_type}_missing")
            
            # Final result
            if results["checks_failed"]:
                results["success"] = False
                self.voice.speak_warning("Integration validation found issues")
            else:
                self.voice.speak_success("All integration validations passed")
            
        except Exception as e:
            results["success"] = False
            results["checks_failed"].append(f"validation_error: {str(e)}")
            self.logger.error(f"Validation failed: {e}")
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        await self.factory.cleanup()
        self.logger.info("Integration setup cleanup completed")
    
    def get_setup_info(self) -> Dict[str, Any]:
        """Get information about the current setup."""
        return {
            "workspace_root": str(self.workspace_root),
            "claude_dir": str(self.claude_dir),
            "commands_dir": str(self.commands_dir),
            "hooks_dir": str(self.hooks_dir),
            "config_file": str(self.config_file),
            "directories_exist": {
                "claude_dir": self.claude_dir.exists(),
                "commands_dir": self.commands_dir.exists(),
                "hooks_dir": self.hooks_dir.exists()
            },
            "config_file_exists": self.config_file.exists(),
            "factory_status": self.factory.get_status()
        }


# Convenience functions
async def setup_integrations(workspace_root: str = None) -> Dict[str, Any]:
    """
    Convenience function to setup all integrations.
    
    Args:
        workspace_root: Root directory for workspace
    
    Returns:
        Dict containing setup results
    """
    setup_manager = IntegrationSetup(workspace_root)
    try:
        return await setup_manager.setup_all()
    finally:
        await setup_manager.cleanup()


async def validate_integrations(workspace_root: str = None) -> Dict[str, Any]:
    """
    Convenience function to validate integration setup.
    
    Args:
        workspace_root: Root directory for workspace
    
    Returns:
        Dict containing validation results
    """
    setup_manager = IntegrationSetup(workspace_root)
    try:
        return await setup_manager.validate_setup()
    finally:
        await setup_manager.cleanup()


if __name__ == "__main__":
    # Run setup if script is executed directly
    async def main():
        print("Setting up external integrations...")
        result = await setup_integrations()
        
        if result["success"]:
            print("✅ Integration setup completed successfully")
            print(f"✅ Directories created: {len(result['directories_created'])}")
            print(f"✅ Integrations setup: {len(result['integrations_setup'])}")
        else:
            print("❌ Integration setup failed")
            for error in result["errors"]:
                print(f"❌ {error}")
        
        # Run validation
        print("\nValidating integration setup...")
        validation_result = await validate_integrations()
        
        if validation_result["success"]:
            print("✅ All validation checks passed")
        else:
            print("❌ Validation failed")
            for check in validation_result["checks_failed"]:
                print(f"❌ {check}")
    
    asyncio.run(main())