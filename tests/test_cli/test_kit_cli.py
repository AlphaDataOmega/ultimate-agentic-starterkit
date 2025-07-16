"""
Unit tests for the kit.py CLI interface.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from kit import AgenticStarterKitCLI, create_parser, main


class TestAgenticStarterKitCLI:
    """Test cases for the AgenticStarterKitCLI class."""
    
    @pytest.fixture
    def cli(self):
        """Create CLI instance for testing."""
        return AgenticStarterKitCLI()
    
    @pytest.fixture
    def sample_project_spec(self):
        """Sample project specification for testing."""
        return {
            "title": "Test Project",
            "description": "Test project description",
            "tasks": [
                {
                    "id": "task-1",
                    "title": "Test Task",
                    "description": "Test task description",
                    "type": "CREATE",
                    "agent_type": "coder"
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_cli_initialization(self, cli):
        """Test CLI initialization."""
        # Mock the dependencies
        with patch('kit.get_config') as mock_config, \
             patch('kit.get_voice_alerts') as mock_voice, \
             patch('kit.LangGraphWorkflowManager') as mock_workflow, \
             patch('kit.ValidationOrchestrator') as mock_validator:
            
            mock_config.return_value = MagicMock()
            mock_voice.return_value = MagicMock()
            mock_workflow.return_value = MagicMock()
            mock_validator.return_value = MagicMock()
            
            await cli.initialize()
            
            assert cli.config is not None
            assert cli.voice_alerts is not None
            assert cli.workflow_manager is not None
            assert cli.validator is not None
    
    @pytest.mark.asyncio
    async def test_cli_initialization_failure(self, cli):
        """Test CLI initialization failure."""
        with patch('kit.get_config', side_effect=Exception("Config error")):
            with pytest.raises(SystemExit):
                await cli.initialize()
    
    @pytest.mark.asyncio
    async def test_load_prp_file_success(self, cli, sample_project_spec):
        """Test successful PRP file loading."""
        # Create temporary PRP file
        test_prp = Path("test_prp.md")
        test_prp.write_text("# Test PRP\n\n## Tasks\n- Task 1\n- Task 2")
        
        try:
            # Mock the parser agent
            with patch('kit.ParserAgent') as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser_class.return_value = mock_parser
                
                # Mock the execute method
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output = sample_project_spec
                mock_parser.execute = AsyncMock(return_value=mock_result)
                
                result = await cli._load_prp_file(test_prp)
                
                assert result == sample_project_spec
                mock_parser.execute.assert_called_once()
        finally:
            # Clean up
            if test_prp.exists():
                test_prp.unlink()
    
    @pytest.mark.asyncio
    async def test_load_prp_file_not_found(self, cli):
        """Test PRP file not found error."""
        non_existent_file = Path("non_existent.md")
        
        with pytest.raises(FileNotFoundError):
            await cli._load_prp_file(non_existent_file)
    
    @pytest.mark.asyncio
    async def test_load_prp_file_parse_failure(self, cli):
        """Test PRP file parse failure."""
        # Create temporary PRP file
        test_prp = Path("test_prp.md")
        test_prp.write_text("# Test PRP")
        
        try:
            # Mock the parser agent to fail
            with patch('kit.ParserAgent') as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser_class.return_value = mock_parser
                
                # Mock the execute method to return failure
                mock_result = MagicMock()
                mock_result.success = False
                mock_result.error = "Parse error"
                mock_parser.execute = AsyncMock(return_value=mock_result)
                
                with pytest.raises(ValueError, match="Failed to parse PRP file"):
                    await cli._load_prp_file(test_prp)
        finally:
            # Clean up
            if test_prp.exists():
                test_prp.unlink()
    
    @pytest.mark.asyncio
    async def test_validate_project(self, cli, sample_project_spec):
        """Test project validation."""
        cli.validator = MagicMock()
        validation_result = {
            "overall_status": "passed",
            "checks": {
                "syntax": {"passed": True, "message": "Syntax OK"},
                "structure": {"passed": True, "message": "Structure OK"}
            }
        }
        cli.validator.validate_project_spec = AsyncMock(return_value=validation_result)
        
        result = await cli._validate_project(sample_project_spec)
        
        assert result == validation_result
        cli.validator.validate_project_spec.assert_called_once_with(sample_project_spec)
    
    @pytest.mark.asyncio
    async def test_dry_run_project(self, cli, sample_project_spec):
        """Test dry run project execution."""
        # Mock the orchestrator
        with patch('kit.O3Orchestrator') as mock_orchestrator_class, \
             patch('kit.ProjectSpecification') as mock_spec_class:
            
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            mock_spec = MagicMock()
            mock_spec_class.return_value = mock_spec
            
            execution_plan = {
                "execution_order": [
                    {
                        "task_id": "task-1",
                        "priority": "high",
                        "dependencies": [],
                        "agent_type": "coder"
                    }
                ],
                "critical_path": ["task-1"]
            }
            
            mock_orchestrator.create_execution_plan = AsyncMock(return_value=execution_plan)
            
            result = await cli._dry_run_project(sample_project_spec)
            
            assert result["success"] is True
            assert result["dry_run"] is True
            assert result["execution_plan"] == execution_plan
    
    @pytest.mark.asyncio
    async def test_execute_full_project(self, cli, sample_project_spec):
        """Test full project execution."""
        # Mock the workflow manager
        cli.workflow_manager = MagicMock()
        workflow_result = {
            "workflow_status": "completed",
            "overall_confidence": 0.95,
            "completed_tasks": ["task-1"],
            "failed_tasks": []
        }
        cli.workflow_manager.execute_workflow = AsyncMock(return_value=workflow_result)
        
        # Mock voice alerts
        cli.voice_alerts = MagicMock()
        
        result = await cli._execute_full_project(sample_project_spec, True)
        
        assert result == workflow_result
        cli.workflow_manager.execute_workflow.assert_called_once()
        cli.voice_alerts.speak_milestone.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_project_success(self, cli, sample_project_spec):
        """Test successful project execution."""
        # Create temporary PRP file
        test_prp = Path("test_prp.md")
        test_prp.write_text("# Test PRP")
        
        try:
            # Mock the methods
            cli._load_prp_file = AsyncMock(return_value=sample_project_spec)
            cli._execute_full_project = AsyncMock(return_value={"success": True})
            cli.voice_alerts = MagicMock()
            
            result = await cli.execute_project(str(test_prp))
            
            assert result["success"] is True
            cli._load_prp_file.assert_called_once()
            cli._execute_full_project.assert_called_once()
        finally:
            # Clean up
            if test_prp.exists():
                test_prp.unlink()
    
    @pytest.mark.asyncio
    async def test_execute_project_file_not_found(self, cli):
        """Test project execution with non-existent file."""
        result = await cli.execute_project("non_existent.md")
        
        assert result["success"] is False
        assert "not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_generate_prp(self, cli):
        """Test PRP generation."""
        description = "Build a REST API"
        
        with patch('kit.ClaudeCodeIntegration') as mock_integration_class:
            mock_integration = MagicMock()
            mock_integration_class.return_value = mock_integration
            
            mock_integration.generate_prp_command = AsyncMock(return_value={
                "success": True,
                "file_path": "PRPs/rest_api.md"
            })
            
            result = await cli.generate_prp(description)
            
            assert result["success"] is True
            assert "file_path" in result
            mock_integration.generate_prp_command.assert_called_once_with(description)
    
    @pytest.mark.asyncio
    async def test_generate_prp_failure(self, cli):
        """Test PRP generation failure."""
        description = "Build a REST API"
        
        with patch('kit.ClaudeCodeIntegration', side_effect=Exception("Integration error")):
            result = await cli.generate_prp(description)
            
            assert result["success"] is False
            assert "error" in result
    
    def test_list_examples(self, cli):
        """Test listing examples."""
        # Create temporary examples directory
        examples_dir = Path("examples")
        examples_dir.mkdir(exist_ok=True)
        
        # Create test files
        test_files = ["test1.py", "test2.py"]
        for file_name in test_files:
            (examples_dir / file_name).write_text("# Test file")
        
        try:
            # This should not raise an exception
            cli.list_examples()
        finally:
            # Clean up
            for file_name in test_files:
                (examples_dir / file_name).unlink()
    
    def test_get_status(self, cli):
        """Test getting system status."""
        cli.config = MagicMock()
        cli.voice_alerts = MagicMock()
        cli.workflow_manager = MagicMock()
        cli.validator = MagicMock()
        
        cli.voice_alerts.get_status.return_value = {"enabled": True}
        
        status = cli.get_status()
        
        assert status["system_initialized"] is True
        assert status["voice_alerts_enabled"] is True
        assert status["workflow_manager_ready"] is True
        assert status["validator_ready"] is True
        assert "voice_status" in status


class TestArgumentParser:
    """Test cases for the argument parser."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        
        assert parser is not None
        assert parser.description is not None
    
    def test_parser_prp_argument(self):
        """Test --prp argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["--prp", "test.md"])
        assert args.prp == "test.md"
    
    def test_parser_validate_argument(self):
        """Test --validate argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["--prp", "test.md", "--validate"])
        assert args.validate is True
    
    def test_parser_dry_run_argument(self):
        """Test --dry-run argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["--prp", "test.md", "--dry-run"])
        assert args.dry_run is True
    
    def test_parser_generate_prp_argument(self):
        """Test --generate-prp argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["--generate-prp", "Build an API"])
        assert args.generate_prp == "Build an API"
    
    def test_parser_list_examples_argument(self):
        """Test --list-examples argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["--list-examples"])
        assert args.list_examples is True
    
    def test_parser_status_argument(self):
        """Test --status argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["--status"])
        assert args.status is True
    
    def test_parser_no_voice_argument(self):
        """Test --no-voice argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["--prp", "test.md", "--no-voice"])
        assert args.no_voice is True
    
    def test_parser_verbose_argument(self):
        """Test --verbose argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["--prp", "test.md", "--verbose"])
        assert args.verbose is True


class TestMainFunction:
    """Test cases for the main function."""
    
    @pytest.mark.asyncio
    async def test_main_with_prp(self):
        """Test main function with --prp argument."""
        test_args = ["--prp", "test.md"]
        
        with patch('sys.argv', ['kit.py'] + test_args), \
             patch('kit.AgenticStarterKitCLI') as mock_cli_class:
            
            mock_cli = MagicMock()
            mock_cli_class.return_value = mock_cli
            mock_cli.initialize = AsyncMock()
            mock_cli.execute_project = AsyncMock(return_value={"success": True})
            
            with patch('sys.exit') as mock_exit:
                await main()
                mock_exit.assert_called_once_with(0)
    
    @pytest.mark.asyncio
    async def test_main_with_generate_prp(self):
        """Test main function with --generate-prp argument."""
        test_args = ["--generate-prp", "Build an API"]
        
        with patch('sys.argv', ['kit.py'] + test_args), \
             patch('kit.AgenticStarterKitCLI') as mock_cli_class:
            
            mock_cli = MagicMock()
            mock_cli_class.return_value = mock_cli
            mock_cli.initialize = AsyncMock()
            mock_cli.generate_prp = AsyncMock(return_value={"success": True})
            
            with patch('sys.exit') as mock_exit:
                await main()
                mock_exit.assert_called_once_with(0)
    
    @pytest.mark.asyncio
    async def test_main_with_list_examples(self):
        """Test main function with --list-examples argument."""
        test_args = ["--list-examples"]
        
        with patch('sys.argv', ['kit.py'] + test_args), \
             patch('kit.AgenticStarterKitCLI') as mock_cli_class:
            
            mock_cli = MagicMock()
            mock_cli_class.return_value = mock_cli
            mock_cli.initialize = AsyncMock()
            mock_cli.list_examples = MagicMock()
            
            await main()
            mock_cli.list_examples.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_main_with_status(self):
        """Test main function with --status argument."""
        test_args = ["--status"]
        
        with patch('sys.argv', ['kit.py'] + test_args), \
             patch('kit.AgenticStarterKitCLI') as mock_cli_class:
            
            mock_cli = MagicMock()
            mock_cli_class.return_value = mock_cli
            mock_cli.initialize = AsyncMock()
            mock_cli.get_status = MagicMock(return_value={"status": "ok"})
            
            await main()
            mock_cli.get_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_main_with_no_arguments(self):
        """Test main function with no arguments."""
        test_args = []
        
        with patch('sys.argv', ['kit.py'] + test_args), \
             patch('kit.AgenticStarterKitCLI') as mock_cli_class, \
             patch('kit.create_parser') as mock_parser_func:
            
            mock_cli = MagicMock()
            mock_cli_class.return_value = mock_cli
            mock_cli.initialize = AsyncMock()
            
            mock_parser = MagicMock()
            mock_parser_func.return_value = mock_parser
            mock_parser.parse_args.return_value = MagicMock(
                prp=None,
                generate_prp=None,
                list_examples=False,
                status=False
            )
            
            await main()
            mock_parser.print_help.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])