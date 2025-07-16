#!/usr/bin/env python3
"""
Claude Code Integration Example - Ultimate Agentic StarterKit

This example demonstrates how to integrate with Claude Code VS Code extension,
including event handlers, file system integration, and command registration.
"""

import asyncio
import sys
import os
from pathlib import Path
import json

# Add StarterKit to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'StarterKit'))

from StarterKit.integrations.claude_code import ClaudeCodeIntegration
from StarterKit.core.logger import get_logger

# Setup logging
logger = get_logger("claude_code_example")


def display_integration_results(result):
    """Display Claude Code integration results."""
    print("=" * 60)
    print("CLAUDE CODE INTEGRATION RESULTS")
    print("=" * 60)
    
    if isinstance(result, dict):
        success = result.get("success", False)
        if success:
            print("✓ Integration operation successful!")
            
            # Display specific result data
            if "commands" in result:
                commands = result["commands"]
                print(f"  Commands registered: {len(commands)}")
                for cmd in commands:
                    print(f"    - {cmd}")
            
            if "events" in result:
                events = result["events"]
                print(f"  Events handled: {len(events)}")
                for event in events:
                    print(f"    - {event}")
            
            if "file_path" in result:
                print(f"  File path: {result['file_path']}")
            
            if "message" in result:
                print(f"  Message: {result['message']}")
        else:
            print("✗ Integration operation failed!")
            error = result.get("error", "Unknown error")
            print(f"  Error: {error}")
    else:
        print(f"Result: {result}")


async def example_1_command_registration():
    """Example 1: Register custom commands with Claude Code."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Command Registration")
    print("="*60)
    
    try:
        # Initialize integration
        integration = ClaudeCodeIntegration(".")
        
        # Register custom commands
        commands = [
            {
                "name": "agentic.parse.prp",
                "title": "Parse PRP File",
                "description": "Parse a PRP file and extract tasks",
                "handler": "parse_prp_command"
            },
            {
                "name": "agentic.generate.code",
                "title": "Generate Code",
                "description": "Generate code using AI agent",
                "handler": "generate_code_command"
            },
            {
                "name": "agentic.run.workflow",
                "title": "Run Workflow",
                "description": "Execute complete workflow",
                "handler": "run_workflow_command"
            },
            {
                "name": "agentic.validate.project",
                "title": "Validate Project",
                "description": "Validate project specification",
                "handler": "validate_project_command"
            }
        ]
        
        print("Registering commands...")
        result = await integration.register_commands(commands)
        display_integration_results(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Command registration failed: {e}")
        print(f"Command registration failed: {e}")
        return {"success": False, "error": str(e)}


async def example_2_event_handlers():
    """Example 2: Set up event handlers for file system changes."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Event Handlers")
    print("="*60)
    
    try:
        # Initialize integration
        integration = ClaudeCodeIntegration(".")
        
        # Define event handlers
        event_handlers = {
            "file_created": {
                "handler": "on_file_created",
                "description": "Handle file creation events"
            },
            "file_modified": {
                "handler": "on_file_modified",
                "description": "Handle file modification events"
            },
            "file_deleted": {
                "handler": "on_file_deleted",
                "description": "Handle file deletion events"
            },
            "prp_file_changed": {
                "handler": "on_prp_file_changed",
                "description": "Handle PRP file changes"
            }
        }
        
        print("Setting up event handlers...")
        result = await integration.setup_event_handlers(event_handlers)
        display_integration_results(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Event handler setup failed: {e}")
        print(f"Event handler setup failed: {e}")
        return {"success": False, "error": str(e)}


async def example_3_file_system_integration():
    """Example 3: File system integration and monitoring."""
    print("\n" + "="*60)
    print("EXAMPLE 3: File System Integration")
    print("="*60)
    
    try:
        # Initialize integration
        integration = ClaudeCodeIntegration(".")
        
        # Monitor specific file patterns
        patterns = [
            "*.py",
            "*.md",
            "PRPs/*.md",
            "examples/*.py",
            "tests/*.py"
        ]
        
        print("Setting up file system monitoring...")
        result = await integration.setup_file_monitoring(patterns)
        display_integration_results(result)
        
        # Test file operations
        test_file = Path("test_integration.py")
        if not test_file.exists():
            test_file.write_text("""
# Test file for Claude Code integration
def hello_world():
    print("Hello from Claude Code integration!")

if __name__ == "__main__":
    hello_world()
""")
            print(f"Created test file: {test_file}")
        
        # Simulate file modification
        print("Simulating file modification...")
        content = test_file.read_text()
        test_file.write_text(content + "\n# Modified by integration test")
        
        # Clean up
        if test_file.exists():
            test_file.unlink()
            print(f"Cleaned up test file: {test_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"File system integration failed: {e}")
        print(f"File system integration failed: {e}")
        return {"success": False, "error": str(e)}


async def example_4_prp_generation():
    """Example 4: Generate PRP from natural language description."""
    print("\n" + "="*60)
    print("EXAMPLE 4: PRP Generation")
    print("="*60)
    
    try:
        # Initialize integration
        integration = ClaudeCodeIntegration(".")
        
        # Natural language description
        description = """
        Create a web application for managing a personal library.
        The application should allow users to:
        - Add books to their library
        - Search for books by title, author, or genre
        - Mark books as read or unread
        - Add reviews and ratings
        - Generate reading statistics
        
        Technical requirements:
        - Use React for the frontend
        - Use FastAPI for the backend
        - Use PostgreSQL for data storage
        - Include user authentication
        - Add comprehensive tests
        """
        
        print("Generating PRP from description...")
        result = await integration.generate_prp_command(description)
        display_integration_results(result)
        
        return result
        
    except Exception as e:
        logger.error(f"PRP generation failed: {e}")
        print(f"PRP generation failed: {e}")
        return {"success": False, "error": str(e)}


async def example_5_workflow_integration():
    """Example 5: Integrate workflow execution with Claude Code."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Workflow Integration")
    print("="*60)
    
    try:
        # Initialize integration
        integration = ClaudeCodeIntegration(".")
        
        # Sample workflow specification
        workflow_spec = {
            "title": "Claude Code Integration Demo",
            "description": "Demonstrate workflow integration with Claude Code",
            "tasks": [
                {
                    "id": "setup",
                    "title": "Setup Project",
                    "description": "Setup basic project structure",
                    "type": "CREATE",
                    "agent_type": "coder"
                },
                {
                    "id": "implement",
                    "title": "Implement Features",
                    "description": "Implement core features",
                    "type": "CREATE",
                    "agent_type": "coder"
                },
                {
                    "id": "test",
                    "title": "Add Tests",
                    "description": "Add comprehensive tests",
                    "type": "TEST",
                    "agent_type": "tester"
                }
            ]
        }
        
        print("Executing workflow through Claude Code integration...")
        result = await integration.execute_workflow_command(workflow_spec)
        display_integration_results(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Workflow integration failed: {e}")
        print(f"Workflow integration failed: {e}")
        return {"success": False, "error": str(e)}


async def example_6_progress_reporting():
    """Example 6: Progress reporting and status updates."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Progress Reporting")
    print("="*60)
    
    try:
        # Initialize integration
        integration = ClaudeCodeIntegration(".")
        
        # Simulate progress updates
        progress_updates = [
            {"step": "initialization", "progress": 0.1, "message": "Initializing workflow"},
            {"step": "parsing", "progress": 0.3, "message": "Parsing project specification"},
            {"step": "planning", "progress": 0.5, "message": "Creating execution plan"},
            {"step": "execution", "progress": 0.8, "message": "Executing tasks"},
            {"step": "completion", "progress": 1.0, "message": "Workflow completed"}
        ]
        
        print("Sending progress updates...")
        for update in progress_updates:
            result = await integration.send_progress_update(update)
            display_integration_results(result)
            await asyncio.sleep(0.5)  # Simulate work
        
        return {"success": True, "updates": len(progress_updates)}
        
    except Exception as e:
        logger.error(f"Progress reporting failed: {e}")
        print(f"Progress reporting failed: {e}")
        return {"success": False, "error": str(e)}


async def example_7_error_handling():
    """Example 7: Error handling and recovery in integration."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Error Handling")
    print("="*60)
    
    try:
        # Initialize integration
        integration = ClaudeCodeIntegration(".")
        
        # Test error scenarios
        error_scenarios = [
            {
                "name": "Invalid Command",
                "action": lambda: integration.execute_command("invalid.command", {})
            },
            {
                "name": "Missing File",
                "action": lambda: integration.parse_prp_file("nonexistent.md")
            },
            {
                "name": "Invalid Workflow",
                "action": lambda: integration.execute_workflow_command({})
            }
        ]
        
        print("Testing error scenarios...")
        for scenario in error_scenarios:
            print(f"\nTesting: {scenario['name']}")
            try:
                result = await scenario["action"]()
                display_integration_results(result)
            except Exception as e:
                print(f"Expected error caught: {e}")
                result = {"success": False, "error": str(e)}
                display_integration_results(result)
        
        return {"success": True, "scenarios_tested": len(error_scenarios)}
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        print(f"Error handling test failed: {e}")
        return {"success": False, "error": str(e)}


async def example_8_configuration_management():
    """Example 8: Configuration management for Claude Code."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Configuration Management")
    print("="*60)
    
    try:
        # Initialize integration
        integration = ClaudeCodeIntegration(".")
        
        # Sample configuration
        config = {
            "workspace_path": ".",
            "auto_save": True,
            "voice_alerts": True,
            "progress_reporting": True,
            "file_monitoring": {
                "enabled": True,
                "patterns": ["*.py", "*.md", "PRPs/*.md"]
            },
            "agent_config": {
                "default_confidence_threshold": 0.8,
                "max_retries": 3,
                "timeout": 300
            }
        }
        
        print("Setting up configuration...")
        result = await integration.setup_configuration(config)
        display_integration_results(result)
        
        # Test configuration retrieval
        print("\nRetrieving configuration...")
        config_result = await integration.get_configuration()
        display_integration_results(config_result)
        
        return result
        
    except Exception as e:
        logger.error(f"Configuration management failed: {e}")
        print(f"Configuration management failed: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Main function to run all Claude Code integration examples."""
    print("Ultimate Agentic StarterKit - Claude Code Integration Examples")
    print("=" * 60)
    
    try:
        # Run examples
        await example_1_command_registration()
        await example_2_event_handlers()
        await example_3_file_system_integration()
        await example_4_prp_generation()
        await example_5_workflow_integration()
        await example_6_progress_reporting()
        await example_7_error_handling()
        await example_8_configuration_management()
        
        print("\n" + "="*60)
        print("All Claude Code integration examples completed!")
        print("="*60)
        print("Note: Some examples may fail if Claude Code extension is not installed")
        print("Install the Claude Code VS Code extension for full functionality")
        
    except Exception as e:
        logger.error(f"Error running Claude Code examples: {e}")
        print(f"Error running Claude Code examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)