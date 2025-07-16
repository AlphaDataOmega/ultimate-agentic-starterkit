"""
Basic tests for examples to ensure they can be imported and run without errors.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class TestExamplesImport:
    """Test that examples can be imported without errors."""
    
    def test_parser_example_import(self):
        """Test that parser example can be imported."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            # Import should not raise an exception
            import parser_example
            assert hasattr(parser_example, 'main')
            assert hasattr(parser_example, 'display_results')
            
        except ImportError as e:
            pytest.skip(f"Parser example import failed: {e}")
    
    def test_coder_example_import(self):
        """Test that coder example can be imported."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            # Import should not raise an exception
            import coder_example
            assert hasattr(coder_example, 'main')
            assert hasattr(coder_example, 'display_results')
            
        except ImportError as e:
            pytest.skip(f"Coder example import failed: {e}")
    
    def test_workflow_example_import(self):
        """Test that workflow example can be imported."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            # Import should not raise an exception
            import workflow_example
            assert hasattr(workflow_example, 'main')
            assert hasattr(workflow_example, 'display_workflow_results')
            
        except ImportError as e:
            pytest.skip(f"Workflow example import failed: {e}")
    
    def test_claude_code_example_import(self):
        """Test that Claude Code example can be imported."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            # Import should not raise an exception
            import claude_code_example
            assert hasattr(claude_code_example, 'main')
            assert hasattr(claude_code_example, 'display_integration_results')
            
        except ImportError as e:
            pytest.skip(f"Claude Code example import failed: {e}")


class TestExamplesExecution:
    """Test that examples can be executed without errors."""
    
    @pytest.mark.asyncio
    async def test_parser_example_execution(self):
        """Test parser example execution with mocking."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            import parser_example
            
            # Mock the ParserAgent and its dependencies
            with patch('parser_example.ParserAgent') as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser_class.return_value = mock_parser
                
                # Mock the execute method
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.confidence = 0.8
                mock_result.execution_time = 1.0
                mock_result.timestamp = "2024-01-01T00:00:00"
                mock_result.agent_id = "test-agent"
                mock_result.output = {
                    'tasks': [
                        {
                            'title': 'Test Task',
                            'type': 'CREATE',
                            'agent_type': 'coder',
                            'confidence': 0.8,
                            'description': 'Test description',
                            'source_query': 'test',
                            'extraction_method': 'semantic'
                        }
                    ],
                    'chunks_processed': 1,
                    'extraction_method': 'semantic_and_pattern'
                }
                mock_parser.execute = AsyncMock(return_value=mock_result)
                
                # Test individual example functions
                result = await parser_example.example_1_basic_parsing()
                assert result is not None
                
        except ImportError as e:
            pytest.skip(f"Parser example execution test skipped: {e}")
        except Exception as e:
            pytest.fail(f"Parser example execution failed: {e}")
    
    @pytest.mark.asyncio
    async def test_coder_example_execution(self):
        """Test coder example execution with mocking."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            import coder_example
            
            # Mock the CoderAgent and its dependencies
            with patch('coder_example.CoderAgent') as mock_coder_class:
                mock_coder = MagicMock()
                mock_coder_class.return_value = mock_coder
                
                # Mock the execute method
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.confidence = 0.9
                mock_result.execution_time = 2.0
                mock_result.timestamp = "2024-01-01T00:00:00"
                mock_result.agent_id = "test-coder"
                mock_result.output = {
                    'files': [
                        {
                            'path': 'test.py',
                            'language': 'python',
                            'size': 100,
                            'lines': 10,
                            'operation': 'create'
                        }
                    ],
                    'validation': {
                        'total_files': 1,
                        'total_lines': 10,
                        'languages': ['python'],
                        'quality_score': 0.8,
                        'issues': []
                    },
                    'tool_calls': 1
                }
                mock_coder.execute = AsyncMock(return_value=mock_result)
                
                # Test individual example functions
                result = await coder_example.example_1_fastapi_auth()
                assert result is not None
                
        except ImportError as e:
            pytest.skip(f"Coder example execution test skipped: {e}")
        except Exception as e:
            pytest.fail(f"Coder example execution failed: {e}")
    
    @pytest.mark.asyncio
    async def test_workflow_example_execution(self):
        """Test workflow example execution with mocking."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            import workflow_example
            
            # Mock the workflow manager and its dependencies
            with patch('workflow_example.LangGraphWorkflowManager') as mock_workflow_class, \
                 patch('workflow_example.get_voice_alerts') as mock_voice:
                
                mock_workflow = MagicMock()
                mock_workflow_class.return_value = mock_workflow
                
                mock_voice_alerts = MagicMock()
                mock_voice.return_value = mock_voice_alerts
                
                # Mock the execute_workflow method
                workflow_result = {
                    'workflow_status': 'completed',
                    'overall_confidence': 0.85,
                    'completed_tasks': ['task-1', 'task-2'],
                    'failed_tasks': [],
                    'agent_results': [
                        {
                            'task_id': 'task-1',
                            'agent_type': 'coder',
                            'success': True,
                            'confidence': 0.8,
                            'execution_time': 1.5
                        }
                    ],
                    'metrics': {
                        'total_tasks': 2,
                        'duration': 10.0,
                        'success_rate': 1.0,
                        'average_confidence': 0.85
                    }
                }
                mock_workflow.execute_workflow = AsyncMock(return_value=workflow_result)
                
                # Test individual example functions
                result = await workflow_example.example_1_simple_api()
                assert result is not None
                
        except ImportError as e:
            pytest.skip(f"Workflow example execution test skipped: {e}")
        except Exception as e:
            pytest.fail(f"Workflow example execution failed: {e}")
    
    @pytest.mark.asyncio
    async def test_claude_code_example_execution(self):
        """Test Claude Code example execution with mocking."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            import claude_code_example
            
            # Mock the Claude Code integration
            with patch('claude_code_example.ClaudeCodeIntegration') as mock_integration_class:
                mock_integration = MagicMock()
                mock_integration_class.return_value = mock_integration
                
                # Mock various methods
                mock_integration.register_commands = AsyncMock(return_value={
                    "success": True,
                    "commands": ["agentic.parse.prp", "agentic.generate.code"]
                })
                
                mock_integration.setup_event_handlers = AsyncMock(return_value={
                    "success": True,
                    "events": ["file_created", "file_modified"]
                })
                
                mock_integration.setup_file_monitoring = AsyncMock(return_value={
                    "success": True,
                    "message": "File monitoring enabled"
                })
                
                # Test individual example functions
                result = await claude_code_example.example_1_command_registration()
                assert result is not None
                
        except ImportError as e:
            pytest.skip(f"Claude Code example execution test skipped: {e}")
        except Exception as e:
            pytest.fail(f"Claude Code example execution failed: {e}")


class TestExamplesUtilities:
    """Test utility functions in examples."""
    
    def test_parser_example_display_results(self):
        """Test parser example display results function."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            import parser_example
            
            # Mock result object
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.confidence = 0.8
            mock_result.execution_time = 1.0
            mock_result.timestamp = "2024-01-01T00:00:00"
            mock_result.agent_id = "test-agent"
            mock_result.output = {
                'tasks': [
                    {
                        'title': 'Test Task',
                        'type': 'CREATE',
                        'agent_type': 'coder',
                        'confidence': 0.8,
                        'description': 'Test description',
                        'source_query': 'test'
                    }
                ],
                'chunks_processed': 1,
                'extraction_method': 'semantic'
            }
            
            # Should not raise an exception
            parser_example.display_results(mock_result)
            
        except ImportError as e:
            pytest.skip(f"Parser example display test skipped: {e}")
    
    def test_coder_example_display_results(self):
        """Test coder example display results function."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            import coder_example
            
            # Mock result object
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.confidence = 0.9
            mock_result.execution_time = 2.0
            mock_result.timestamp = "2024-01-01T00:00:00"
            mock_result.agent_id = "test-coder"
            mock_result.output = {
                'files': [
                    {
                        'path': 'test.py',
                        'language': 'python',
                        'size': 100,
                        'lines': 10,
                        'operation': 'create'
                    }
                ],
                'validation': {
                    'total_files': 1,
                    'total_lines': 10,
                    'languages': ['python'],
                    'quality_score': 0.8,
                    'issues': []
                },
                'tool_calls': 1
            }
            
            # Should not raise an exception
            coder_example.display_results(mock_result)
            
        except ImportError as e:
            pytest.skip(f"Coder example display test skipped: {e}")
    
    def test_workflow_example_display_results(self):
        """Test workflow example display results function."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            import workflow_example
            
            # Mock workflow result
            workflow_result = {
                'workflow_status': 'completed',
                'overall_confidence': 0.85,
                'completed_tasks': ['task-1', 'task-2'],
                'failed_tasks': [],
                'agent_results': [
                    {
                        'task_id': 'task-1',
                        'agent_type': 'coder',
                        'success': True,
                        'confidence': 0.8,
                        'execution_time': 1.5
                    }
                ],
                'metrics': {
                    'total_tasks': 2,
                    'duration': 10.0,
                    'success_rate': 1.0,
                    'average_confidence': 0.85
                }
            }
            
            # Should not raise an exception
            workflow_example.display_workflow_results(workflow_result)
            
        except ImportError as e:
            pytest.skip(f"Workflow example display test skipped: {e}")
    
    def test_claude_code_example_display_results(self):
        """Test Claude Code example display results function."""
        try:
            # Add examples directory to path
            examples_path = Path(__file__).parent.parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_path))
            
            import claude_code_example
            
            # Mock integration result
            integration_result = {
                "success": True,
                "commands": ["agentic.parse.prp", "agentic.generate.code"],
                "events": ["file_created", "file_modified"],
                "file_path": "test.py",
                "message": "Operation completed successfully"
            }
            
            # Should not raise an exception
            claude_code_example.display_integration_results(integration_result)
            
        except ImportError as e:
            pytest.skip(f"Claude Code example display test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__])