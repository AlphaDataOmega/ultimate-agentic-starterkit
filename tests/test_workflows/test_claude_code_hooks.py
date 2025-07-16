"""
Tests for Claude Code Hooks Integration.
"""

import pytest
import asyncio
import json
import os
from unittest.mock import Mock, patch, AsyncMock, call
from datetime import datetime

from workflows.claude_code_hooks import (
    ClaudeCodeHookManager,
    HookType,
    HookEvent,
    HookCommand,
    HookError,
    HookExecutionResult,
    emit_hook_event,
    register_hook,
    unregister_hook,
    get_hook_manager,
    format_hook_message,
    validate_hook_config
)


class TestHookEvent:
    """Test cases for HookEvent."""
    
    def test_hook_event_initialization(self):
        """Test hook event initialization."""
        event = HookEvent(
            hook_type=HookType.WORKFLOW_START,
            workflow_id="test-workflow",
            project_id="test-project",
            timestamp=datetime.now(),
            data={"key": "value"},
            metadata={"source": "test"}
        )
        
        assert event.hook_type == HookType.WORKFLOW_START
        assert event.workflow_id == "test-workflow"
        assert event.project_id == "test-project"
        assert event.timestamp is not None
        assert event.data == {"key": "value"}
        assert event.metadata == {"source": "test"}
    
    def test_hook_event_auto_timestamp(self):
        """Test automatic timestamp generation."""
        event = HookEvent(
            hook_type=HookType.WORKFLOW_START,
            workflow_id="test-workflow",
            project_id="test-project"
        )
        
        assert event.timestamp is not None
    
    def test_hook_event_to_dict(self):
        """Test hook event serialization."""
        event = HookEvent(
            hook_type=HookType.WORKFLOW_START,
            workflow_id="test-workflow",
            project_id="test-project",
            data={"key": "value"}
        )
        
        data = event.to_dict()
        
        assert data["hook_type"] == "workflow_start"
        assert data["workflow_id"] == "test-workflow"
        assert data["project_id"] == "test-project"
        assert data["data"] == {"key": "value"}
        assert "timestamp" in data
    
    def test_hook_event_to_json(self):
        """Test hook event JSON serialization."""
        event = HookEvent(
            hook_type=HookType.WORKFLOW_START,
            workflow_id="test-workflow",
            project_id="test-project"
        )
        
        json_str = event.to_json()
        
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed_data = json.loads(json_str)
        assert parsed_data["hook_type"] == "workflow_start"
        assert parsed_data["workflow_id"] == "test-workflow"


class TestHookCommand:
    """Test cases for HookCommand."""
    
    def test_hook_command_initialization(self):
        """Test hook command initialization."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            enabled=True,
            timeout=30,
            working_directory="/home/user",
            environment_variables={"VAR": "value"},
            retry_count=3,
            description="Test hook command"
        )
        
        assert command.hook_type == HookType.WORKFLOW_START
        assert command.command == "python script.py"
        assert command.enabled is True
        assert command.timeout == 30
        assert command.working_directory == "/home/user"
        assert command.environment_variables == {"VAR": "value"}
        assert command.retry_count == 3
        assert command.description == "Test hook command"
    
    def test_hook_command_with_defaults(self):
        """Test hook command with default values."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py"
        )
        
        assert command.hook_type == HookType.WORKFLOW_START
        assert command.command == "python script.py"
        assert command.enabled is True
        assert command.timeout == 30
        assert command.working_directory is None
        assert command.environment_variables == {}
        assert command.retry_count == 1
        assert command.description == ""
    
    def test_hook_command_to_dict(self):
        """Test hook command serialization."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            enabled=True,
            timeout=30
        )
        
        data = command.to_dict()
        
        assert data["hook_type"] == "workflow_start"
        assert data["command"] == "python script.py"
        assert data["enabled"] is True
        assert data["timeout"] == 30
    
    def test_hook_command_validate_success(self):
        """Test successful hook command validation."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            timeout=30
        )
        
        # Should not raise exception
        command.validate()
    
    def test_hook_command_validate_failure(self):
        """Test hook command validation failure."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="",  # Empty command
            timeout=30
        )
        
        with pytest.raises(HookError):
            command.validate()


class TestHookExecutionResult:
    """Test cases for HookExecutionResult."""
    
    def test_hook_execution_result_initialization(self):
        """Test hook execution result initialization."""
        result = HookExecutionResult(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            success=True,
            execution_time=5.0,
            stdout="Output message",
            stderr="",
            exit_code=0,
            error_message=None
        )
        
        assert result.hook_type == HookType.WORKFLOW_START
        assert result.command == "python script.py"
        assert result.success is True
        assert result.execution_time == 5.0
        assert result.stdout == "Output message"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.error_message is None
    
    def test_hook_execution_result_to_dict(self):
        """Test hook execution result serialization."""
        result = HookExecutionResult(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            success=True,
            execution_time=5.0,
            stdout="Output message",
            exit_code=0
        )
        
        data = result.to_dict()
        
        assert data["hook_type"] == "workflow_start"
        assert data["command"] == "python script.py"
        assert data["success"] is True
        assert data["execution_time"] == 5.0
        assert data["stdout"] == "Output message"
        assert data["exit_code"] == 0
        assert "timestamp" in data


class TestClaudeCodeHookManager:
    """Test cases for ClaudeCodeHookManager."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            'claude_code_settings_path': '/home/user/.claude_code/settings.json',
            'hook_timeout': 30,
            'max_retry_attempts': 3,
            'enable_hook_logging': True,
            'hook_environment_variables': {'PATH': '/usr/bin'}
        }
    
    @pytest.fixture
    def hook_manager(self, mock_config):
        """Create hook manager instance."""
        with patch('workflows.claude_code_hooks.get_logger'):
            return ClaudeCodeHookManager(mock_config)
    
    def test_hook_manager_initialization(self, hook_manager):
        """Test hook manager initialization."""
        assert hook_manager.claude_code_settings_path == '/home/user/.claude_code/settings.json'
        assert hook_manager.hook_timeout == 30
        assert hook_manager.max_retry_attempts == 3
        assert hook_manager.enable_hook_logging is True
        assert hook_manager.hook_environment_variables == {'PATH': '/usr/bin'}
        assert len(hook_manager.registered_hooks) == 0
        assert len(hook_manager.execution_history) == 0
    
    def test_register_hook_success(self, hook_manager):
        """Test successful hook registration."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py"
        )
        
        hook_manager.register_hook(command)
        
        assert HookType.WORKFLOW_START in hook_manager.registered_hooks
        assert hook_manager.registered_hooks[HookType.WORKFLOW_START] == command
    
    def test_register_hook_duplicate(self, hook_manager):
        """Test registering duplicate hook."""
        command1 = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python script1.py"
        )
        
        command2 = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python script2.py"
        )
        
        hook_manager.register_hook(command1)
        
        # Should replace the first hook
        hook_manager.register_hook(command2)
        
        assert hook_manager.registered_hooks[HookType.WORKFLOW_START] == command2
    
    def test_unregister_hook_success(self, hook_manager):
        """Test successful hook unregistration."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py"
        )
        
        hook_manager.register_hook(command)
        result = hook_manager.unregister_hook(HookType.WORKFLOW_START)
        
        assert result is True
        assert HookType.WORKFLOW_START not in hook_manager.registered_hooks
    
    def test_unregister_hook_not_found(self, hook_manager):
        """Test unregistering non-existent hook."""
        result = hook_manager.unregister_hook(HookType.WORKFLOW_START)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_emit_event_success(self, hook_manager):
        """Test successful event emission."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="echo 'workflow started'"
        )
        
        hook_manager.register_hook(command)
        
        with patch('workflows.claude_code_hooks.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="workflow started\n",
                stderr="",
                args=["echo", "workflow started"]
            )
            
            result = await hook_manager.emit_event(
                HookType.WORKFLOW_START,
                "test-workflow",
                "test-project"
            )
            
            assert result is not None
            assert result.success is True
            assert result.exit_code == 0
            assert "workflow started" in result.stdout
    
    @pytest.mark.asyncio
    async def test_emit_event_no_hook_registered(self, hook_manager):
        """Test event emission with no hook registered."""
        result = await hook_manager.emit_event(
            HookType.WORKFLOW_START,
            "test-workflow",
            "test-project"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_emit_event_hook_disabled(self, hook_manager):
        """Test event emission with disabled hook."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="echo 'workflow started'",
            enabled=False
        )
        
        hook_manager.register_hook(command)
        
        result = await hook_manager.emit_event(
            HookType.WORKFLOW_START,
            "test-workflow",
            "test-project"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_emit_event_command_failure(self, hook_manager):
        """Test event emission with command failure."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="exit 1"
        )
        
        hook_manager.register_hook(command)
        
        with patch('workflows.claude_code_hooks.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="Command failed",
                args=["exit", "1"]
            )
            
            result = await hook_manager.emit_event(
                HookType.WORKFLOW_START,
                "test-workflow",
                "test-project"
            )
            
            assert result is not None
            assert result.success is False
            assert result.exit_code == 1
            assert "Command failed" in result.stderr
    
    @pytest.mark.asyncio
    async def test_emit_event_timeout(self, hook_manager):
        """Test event emission with timeout."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="sleep 60",
            timeout=1  # Short timeout
        )
        
        hook_manager.register_hook(command)
        
        with patch('workflows.claude_code_hooks.subprocess.run') as mock_run:
            mock_run.side_effect = asyncio.TimeoutError()
            
            result = await hook_manager.emit_event(
                HookType.WORKFLOW_START,
                "test-workflow",
                "test-project"
            )
            
            assert result is not None
            assert result.success is False
            assert "timeout" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_emit_event_with_retry(self, hook_manager):
        """Test event emission with retry on failure."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="exit 1",
            retry_count=2
        )
        
        hook_manager.register_hook(command)
        
        with patch('workflows.claude_code_hooks.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="Command failed",
                args=["exit", "1"]
            )
            
            result = await hook_manager.emit_event(
                HookType.WORKFLOW_START,
                "test-workflow",
                "test-project"
            )
            
            # Should retry 2 times
            assert mock_run.call_count == 2
            assert result.success is False
    
    def test_load_claude_code_settings_success(self, hook_manager):
        """Test successful Claude Code settings load."""
        settings_data = {
            "hooks": {
                "workflow-start": {
                    "enabled": True,
                    "command": "python hook.py workflow_start"
                },
                "task-completed": {
                    "enabled": True,
                    "command": "python hook.py task_completed"
                }
            }
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(settings_data))):
            with patch('os.path.exists', return_value=True):
                result = hook_manager.load_claude_code_settings()
                
                assert result is True
                assert len(hook_manager.registered_hooks) == 2
                assert HookType.WORKFLOW_START in hook_manager.registered_hooks
                assert HookType.TASK_COMPLETED in hook_manager.registered_hooks
    
    def test_load_claude_code_settings_file_not_found(self, hook_manager):
        """Test Claude Code settings load with file not found."""
        with patch('os.path.exists', return_value=False):
            result = hook_manager.load_claude_code_settings()
            
            assert result is False
    
    def test_load_claude_code_settings_invalid_json(self, hook_manager):
        """Test Claude Code settings load with invalid JSON."""
        with patch('builtins.open', mock_open(read_data="invalid json")):
            with patch('os.path.exists', return_value=True):
                result = hook_manager.load_claude_code_settings()
                
                assert result is False
    
    def test_save_claude_code_settings_success(self, hook_manager):
        """Test successful Claude Code settings save."""
        command = HookCommand(
            hook_type=HookType.WORKFLOW_START,
            command="python hook.py workflow_start"
        )
        
        hook_manager.register_hook(command)
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('os.makedirs'):
                result = hook_manager.save_claude_code_settings()
                
                assert result is True
                mock_file.assert_called_once()
    
    def test_get_execution_history(self, hook_manager):
        """Test getting execution history."""
        # Add some mock results to history
        result1 = HookExecutionResult(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            success=True,
            execution_time=5.0,
            exit_code=0
        )
        
        result2 = HookExecutionResult(
            hook_type=HookType.TASK_COMPLETED,
            command="python script.py",
            success=False,
            execution_time=2.0,
            exit_code=1
        )
        
        hook_manager.execution_history = [result1, result2]
        
        history = hook_manager.get_execution_history()
        
        assert len(history) == 2
        assert history[0]["hook_type"] == "workflow_start"
        assert history[1]["hook_type"] == "task_completed"
    
    def test_get_execution_history_filtered(self, hook_manager):
        """Test getting filtered execution history."""
        # Add some mock results to history
        result1 = HookExecutionResult(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            success=True,
            execution_time=5.0,
            exit_code=0
        )
        
        result2 = HookExecutionResult(
            hook_type=HookType.TASK_COMPLETED,
            command="python script.py",
            success=False,
            execution_time=2.0,
            exit_code=1
        )
        
        hook_manager.execution_history = [result1, result2]
        
        history = hook_manager.get_execution_history(hook_type=HookType.WORKFLOW_START)
        
        assert len(history) == 1
        assert history[0]["hook_type"] == "workflow_start"
    
    def test_clear_execution_history(self, hook_manager):
        """Test clearing execution history."""
        # Add some mock results to history
        result = HookExecutionResult(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            success=True,
            execution_time=5.0,
            exit_code=0
        )
        
        hook_manager.execution_history = [result]
        
        hook_manager.clear_execution_history()
        
        assert len(hook_manager.execution_history) == 0
    
    def test_get_hook_statistics(self, hook_manager):
        """Test getting hook statistics."""
        # Add some mock results to history
        result1 = HookExecutionResult(
            hook_type=HookType.WORKFLOW_START,
            command="python script.py",
            success=True,
            execution_time=5.0,
            exit_code=0
        )
        
        result2 = HookExecutionResult(
            hook_type=HookType.TASK_COMPLETED,
            command="python script.py",
            success=False,
            execution_time=2.0,
            exit_code=1
        )
        
        hook_manager.execution_history = [result1, result2]
        
        stats = hook_manager.get_hook_statistics()
        
        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 1
        assert stats["failed_executions"] == 1
        assert stats["average_execution_time"] == 3.5
        assert len(stats["hook_type_breakdown"]) == 2


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_format_hook_message(self):
        """Test hook message formatting."""
        event = HookEvent(
            hook_type=HookType.WORKFLOW_START,
            workflow_id="test-workflow",
            project_id="test-project",
            data={"progress": 0.5}
        )
        
        message = format_hook_message(event)
        
        assert isinstance(message, str)
        assert "workflow_start" in message
        assert "test-workflow" in message
        assert "test-project" in message
    
    def test_validate_hook_config_success(self):
        """Test successful hook configuration validation."""
        config = {
            "hooks": {
                "workflow-start": {
                    "enabled": True,
                    "command": "python hook.py workflow_start"
                }
            }
        }
        
        # Should not raise exception
        validate_hook_config(config)
    
    def test_validate_hook_config_failure(self):
        """Test hook configuration validation failure."""
        config = {
            "hooks": {
                "workflow-start": {
                    "enabled": True,
                    "command": ""  # Empty command
                }
            }
        }
        
        with pytest.raises(HookError):
            validate_hook_config(config)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_emit_hook_event(self):
        """Test convenience function for emitting hook event."""
        with patch('workflows.claude_code_hooks.get_hook_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.emit_event = AsyncMock(return_value=Mock(success=True))
            mock_get_manager.return_value = mock_manager
            
            result = await emit_hook_event(
                HookType.WORKFLOW_START,
                "test-workflow",
                "test-project"
            )
            
            assert result is not None
            assert result.success is True
            mock_manager.emit_event.assert_called_once_with(
                HookType.WORKFLOW_START,
                "test-workflow",
                "test-project",
                None
            )
    
    def test_register_hook(self):
        """Test convenience function for registering hook."""
        with patch('workflows.claude_code_hooks.get_hook_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            command = HookCommand(
                hook_type=HookType.WORKFLOW_START,
                command="python script.py"
            )
            
            register_hook(command)
            
            mock_manager.register_hook.assert_called_once_with(command)
    
    def test_unregister_hook(self):
        """Test convenience function for unregistering hook."""
        with patch('workflows.claude_code_hooks.get_hook_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.unregister_hook.return_value = True
            mock_get_manager.return_value = mock_manager
            
            result = unregister_hook(HookType.WORKFLOW_START)
            
            assert result is True
            mock_manager.unregister_hook.assert_called_once_with(HookType.WORKFLOW_START)
    
    def test_get_hook_manager(self):
        """Test getting global hook manager."""
        with patch('workflows.claude_code_hooks.ClaudeCodeHookManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            manager = get_hook_manager()
            
            assert manager is not None
            mock_manager_class.assert_called_once()


def mock_open(read_data=""):
    """Mock open function for file operations."""
    from unittest.mock import mock_open as _mock_open
    return _mock_open(read_data=read_data)