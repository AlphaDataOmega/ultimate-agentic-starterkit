"""
Unit tests for Claude Code Integration.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from StarterKit.integrations.claude_code import ClaudeCodeIntegration
from StarterKit.core.models import ProjectTask, AgentType


class TestClaudeCodeIntegration:
    """Test suite for Claude Code Integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def claude_integration(self, temp_workspace):
        """Create Claude Code integration instance."""
        return ClaudeCodeIntegration(temp_workspace)
    
    def test_initialization(self, claude_integration, temp_workspace):
        """Test Claude Code integration initialization."""
        assert claude_integration.workspace_root == Path(temp_workspace)
        assert claude_integration.claude_dir == Path(temp_workspace) / ".claude"
        assert claude_integration.commands_dir == Path(temp_workspace) / ".claude" / "commands"
        assert claude_integration.hooks_dir == Path(temp_workspace) / ".claude" / "hooks"
        
        # Check directories were created
        assert claude_integration.claude_dir.exists()
        assert claude_integration.commands_dir.exists()
        assert claude_integration.hooks_dir.exists()
    
    def test_command_handlers_registration(self, claude_integration):
        """Test that command handlers are registered."""
        assert "generate_prp" in claude_integration.command_handlers
        assert "execute_agent_flow" in claude_integration.command_handlers
        assert "review_code" in claude_integration.command_handlers
        assert "validate_project" in claude_integration.command_handlers
        assert "create_agent" in claude_integration.command_handlers
    
    @pytest.mark.asyncio
    async def test_setup_commands(self, claude_integration):
        """Test command setup."""
        result = await claude_integration.setup_commands()
        
        assert result is True
        
        # Check command files were created
        assert (claude_integration.commands_dir / "generate-prp.json").exists()
        assert (claude_integration.commands_dir / "execute-agent-flow.json").exists()
        assert (claude_integration.commands_dir / "review-code.json").exists()
        assert (claude_integration.commands_dir / "validate-project.json").exists()
        assert (claude_integration.commands_dir / "create-agent.json").exists()
        
        # Check config file was created
        assert claude_integration.config_file.exists()
    
    @pytest.mark.asyncio
    async def test_generate_prp_command(self, claude_integration):
        """Test PRP generation command."""
        with patch('StarterKit.integrations.claude_code.ParserAgent') as mock_parser:
            # Mock parser agent
            mock_agent = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = "Test PRP content"
            mock_result.confidence = 0.9
            mock_agent.execute = AsyncMock(return_value=mock_result)
            mock_parser.return_value = mock_agent
            
            result = await claude_integration.generate_prp_command(
                "Create a test feature",
                "general"
            )
            
            assert result["success"] is True
            assert "PRP generated" in result["message"]
            assert "file_path" in result
            assert result["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_generate_prp_command_failure(self, claude_integration):
        """Test PRP generation command failure."""
        with patch('StarterKit.integrations.claude_code.ParserAgent') as mock_parser:
            # Mock parser agent failure
            mock_agent = Mock()
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "Parser failed"
            mock_agent.execute = AsyncMock(return_value=mock_result)
            mock_parser.return_value = mock_agent
            
            result = await claude_integration.generate_prp_command(
                "Create a test feature",
                "general"
            )
            
            assert result["success"] is False
            assert result["error"] == "Parser failed"
    
    @pytest.mark.asyncio
    async def test_execute_agent_flow_command(self, claude_integration):
        """Test agent flow execution command."""
        with patch('StarterKit.integrations.claude_code.ProjectBuilder') as mock_builder:
            # Mock project builder
            mock_builder_instance = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = "Workflow completed"
            mock_result.confidence = 0.8
            mock_builder_instance.execute_from_prp = AsyncMock(return_value=mock_result)
            mock_builder.return_value = mock_builder_instance
            
            result = await claude_integration.execute_agent_flow_command(
                "test_prp.md",
                False
            )
            
            assert result["success"] is True
            assert "Workflow completed" in result["message"]
            assert result["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_review_code_command(self, claude_integration):
        """Test code review command."""
        with patch('StarterKit.integrations.claude_code.AdvisorAgent') as mock_advisor:
            # Mock advisor agent
            mock_agent = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = "Code review results"
            mock_result.confidence = 0.7
            mock_agent.execute = AsyncMock(return_value=mock_result)
            mock_advisor.return_value = mock_agent
            
            result = await claude_integration.review_code_command(
                "test_file.py",
                "security"
            )
            
            assert result["success"] is True
            assert "Code review completed" in result["message"]
            assert result["confidence"] == 0.7
    
    @pytest.mark.asyncio
    async def test_validate_project_command(self, claude_integration):
        """Test project validation command."""
        with patch('StarterKit.integrations.claude_code.TesterAgent') as mock_tester:
            # Mock tester agent
            mock_agent = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = "Validation results"
            mock_result.confidence = 0.9
            mock_agent.execute = AsyncMock(return_value=mock_result)
            mock_tester.return_value = mock_agent
            
            result = await claude_integration.validate_project_command(True)
            
            assert result["success"] is True
            assert "Project validation completed" in result["message"]
            assert result["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_create_agent_command(self, claude_integration):
        """Test agent creation command."""
        with patch('StarterKit.integrations.claude_code.CoderAgent') as mock_coder:
            # Mock coder agent
            mock_agent = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = "Agent creation results"
            mock_result.confidence = 0.8
            mock_agent.execute = AsyncMock(return_value=mock_result)
            mock_coder.return_value = mock_agent
            
            result = await claude_integration.create_agent_command(
                "test_agent",
                "general"
            )
            
            assert result["success"] is True
            assert "Agent test_agent created successfully" in result["message"]
            assert result["confidence"] == 0.8
    
    def test_event_handlers_setup(self, claude_integration):
        """Test event handlers setup."""
        claude_integration.setup_event_handlers()
        
        assert "on_file_save" in claude_integration.event_handlers
        assert "on_file_change" in claude_integration.event_handlers
        assert "on_project_open" in claude_integration.event_handlers
        assert "on_git_commit" in claude_integration.event_handlers
    
    @pytest.mark.asyncio
    async def test_trigger_event(self, claude_integration):
        """Test event triggering."""
        # Register a test event handler
        called = False
        
        async def test_handler(**kwargs):
            nonlocal called
            called = True
            assert kwargs["test_param"] == "test_value"
        
        claude_integration.register_event_handler("test_event", test_handler)
        
        # Trigger the event
        await claude_integration.trigger_event("test_event", test_param="test_value")
        
        assert called is True
    
    @pytest.mark.asyncio
    async def test_file_save_handler(self, claude_integration):
        """Test file save event handler."""
        with patch.object(claude_integration, 'review_code_command') as mock_review:
            mock_review.return_value = {"success": True}
            
            await claude_integration._handle_file_save("test_file.py")
            
            # Should trigger code review for Python files
            mock_review.assert_called_once_with("test_file.py", "style")
    
    @pytest.mark.asyncio
    async def test_project_open_handler(self, claude_integration):
        """Test project open event handler."""
        with patch.object(claude_integration, 'validate_project_command') as mock_validate:
            mock_validate.return_value = {"success": True}
            
            await claude_integration._handle_project_open("/test/project")
            
            # Should trigger project validation
            mock_validate.assert_called_once_with(deep_check=False)
    
    def test_get_status(self, claude_integration):
        """Test status retrieval."""
        status = claude_integration.get_status()
        
        assert "workspace_root" in status
        assert "claude_dir_exists" in status
        assert "commands_dir_exists" in status
        assert "hooks_dir_exists" in status
        assert "registered_commands" in status
        assert "registered_events" in status
        
        assert status["claude_dir_exists"] is True
        assert status["commands_dir_exists"] is True
        assert status["hooks_dir_exists"] is True
    
    def test_get_commands_list(self, claude_integration):
        """Test getting commands list."""
        commands = claude_integration.get_commands_list()
        
        assert "generate_prp" in commands
        assert "execute_agent_flow" in commands
        assert "review_code" in commands
        assert "validate_project" in commands
        assert "create_agent" in commands
    
    def test_get_events_list(self, claude_integration):
        """Test getting events list."""
        claude_integration.setup_event_handlers()
        events = claude_integration.get_events_list()
        
        assert "on_file_save" in events
        assert "on_file_change" in events
        assert "on_project_open" in events
        assert "on_git_commit" in events
    
    def test_format_prp_content(self, claude_integration):
        """Test PRP content formatting."""
        content = claude_integration._format_prp_content(
            "Test parser output",
            "web",
            "Create a web application"
        )
        
        assert "Create a web application" in content
        assert "web" in content
        assert "Test parser output" in content
        assert "# PRP Generated" in content