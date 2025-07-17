"""
Claude Code VS Code extension integration.

This module provides integration with the Claude Code VS Code extension,
including command handlers, event handlers, and workspace integration.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Callable
import asyncio
from datetime import datetime

from core.logger import get_logger
from core.config import get_config
from core.models import AgentType, create_project_task
from core.voice_alerts import get_voice_alerts


class ClaudeCodeIntegration:
    """
    Integration with Claude Code VS Code extension.
    
    Provides command handlers, event handlers, and workspace integration
    for seamless development workflow with Claude Code.
    """
    
    def __init__(self, workspace_root: str):
        """
        Initialize Claude Code integration.
        
        Args:
            workspace_root: Path to the workspace root directory
        """
        self.workspace_root = Path(workspace_root)
        self.claude_dir = self.workspace_root / ".claude"
        self.commands_dir = self.claude_dir / "commands"
        self.hooks_dir = self.claude_dir / "hooks"
        self.config_file = self.claude_dir / "config.json"
        
        self.logger = get_logger("claude_code")
        self.config = get_config()
        self.voice = get_voice_alerts()
        
        # Event handlers registry
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Command handlers registry
        self.command_handlers: Dict[str, Callable] = {}
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Register default command handlers
        self._register_default_handlers()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.claude_dir, self.commands_dir, self.hooks_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Claude Code directories initialized at {self.claude_dir}")
    
    def _register_default_handlers(self):
        """Register default command handlers."""
        self.command_handlers.update({
            "generate_prp": self.generate_prp_command,
            "execute_agent_flow": self.execute_agent_flow_command,
            "review_code": self.review_code_command,
            "validate_project": self.validate_project_command,
            "create_agent": self.create_agent_command
        })
    
    async def setup_commands(self) -> bool:
        """
        Setup Claude Code commands.
        
        Creates command definition files for VS Code extension.
        
        Returns:
            bool: True if setup was successful
        """
        try:
            commands = {
                "generate-prp": {
                    "description": "Generate new PRP from description",
                    "handler": "generate_prp",
                    "parameters": {
                        "description": {
                            "type": "string",
                            "description": "Description of the feature to implement"
                        },
                        "project_type": {
                            "type": "string",
                            "description": "Type of project (web, ai, blockchain, general)",
                            "default": "general"
                        }
                    }
                },
                "execute-agent-flow": {
                    "description": "Execute full agent workflow",
                    "handler": "execute_agent_flow",
                    "parameters": {
                        "prp_file": {
                            "type": "string",
                            "description": "Path to PRP file to execute"
                        },
                        "validate_only": {
                            "type": "boolean",
                            "description": "Only validate without executing",
                            "default": False
                        }
                    }
                },
                "review-code": {
                    "description": "Review code changes with advisor agent",
                    "handler": "review_code",
                    "parameters": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to file to review"
                        },
                        "review_type": {
                            "type": "string",
                            "description": "Type of review (security, performance, style)",
                            "default": "general"
                        }
                    }
                },
                "validate-project": {
                    "description": "Validate project structure and dependencies",
                    "handler": "validate_project",
                    "parameters": {
                        "deep_check": {
                            "type": "boolean",
                            "description": "Perform deep validation checks",
                            "default": False
                        }
                    }
                },
                "create-agent": {
                    "description": "Create new agent from template",
                    "handler": "create_agent",
                    "parameters": {
                        "agent_name": {
                            "type": "string",
                            "description": "Name for the new agent"
                        },
                        "agent_type": {
                            "type": "string",
                            "description": "Type of agent to create",
                            "default": "general"
                        }
                    }
                }
            }
            
            # Write command definitions
            for cmd_name, cmd_config in commands.items():
                cmd_file = self.commands_dir / f"{cmd_name}.json"
                with open(cmd_file, 'w') as f:
                    json.dump(cmd_config, f, indent=2)
            
            # Create main configuration file
            config_data = {
                "version": "1.0.0",
                "workspace_root": str(self.workspace_root),
                "commands_dir": str(self.commands_dir),
                "hooks_dir": str(self.hooks_dir),
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Setup {len(commands)} Claude Code commands")
            self.voice.speak_success("Claude Code commands configured")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup commands: {e}")
            self.voice.speak_error(f"Command setup failed: {str(e)}")
            return False
    
    async def generate_prp_command(self, description: str, project_type: str = "general") -> Dict[str, Any]:
        """
        Command handler for generating new PRP.
        
        Args:
            description: Description of the feature to implement
            project_type: Type of project (web, ai, blockchain, general)
        
        Returns:
            Dict containing result status and file path
        """
        try:
            self.voice.speak_agent_start("prp_generator", "Generating new PRP")
            
            # Import parser agent (dynamic import to avoid circular dependencies)
            from agents.parser_agent import ParserAgent
            
            # Create parser agent
            parser = ParserAgent()
            
            # Create task for PRP generation
            task = create_project_task(
                title="Generate PRP",
                description=f"Create PRP for: {description}",
                task_type="CREATE",
                agent_type=AgentType.PARSER
            )
            
            # Execute parser agent
            result = await parser.execute(task)
            
            if result.success:
                # Generate PRP file
                prp_content = self._format_prp_content(result.output, project_type, description)
                timestamp = int(time.time())
                prp_file = self.workspace_root / "PRPs" / f"generated_{timestamp}.md"
                
                # Ensure PRPs directory exists
                prp_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(prp_file, 'w') as f:
                    f.write(prp_content)
                
                self.voice.speak_success("PRP generated successfully")
                
                return {
                    "success": True,
                    "message": f"PRP generated: {prp_file.name}",
                    "file_path": str(prp_file),
                    "confidence": result.confidence
                }
            else:
                self.voice.speak_error("PRP generation failed")
                return {
                    "success": False,
                    "error": result.error or "Unknown error during PRP generation"
                }
                
        except Exception as e:
            self.logger.error(f"PRP generation failed: {e}")
            self.voice.speak_error(f"PRP generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_agent_flow_command(self, prp_file: str, validate_only: bool = False) -> Dict[str, Any]:
        """
        Command handler for executing agent workflow.
        
        Args:
            prp_file: Path to PRP file to execute
            validate_only: Only validate without executing
        
        Returns:
            Dict containing execution result
        """
        try:
            self.voice.speak_agent_start("workflow_executor", "Starting agent workflow")
            
            # Import workflow components (dynamic import to avoid circular dependencies)
            from workflows.project_builder import ProjectBuilder
            
            # Create project builder
            builder = ProjectBuilder()
            
            # Execute workflow
            result = await builder.execute_from_prp(prp_file, validate_only)
            
            if result.success:
                self.voice.speak_success("Agent workflow completed successfully")
                return {
                    "success": True,
                    "message": f"Workflow completed with confidence: {result.confidence:.2f}",
                    "output": result.output,
                    "confidence": result.confidence
                }
            else:
                self.voice.speak_error("Agent workflow failed")
                return {
                    "success": False,
                    "error": result.error or "Workflow execution failed"
                }
                
        except Exception as e:
            self.logger.error(f"Agent flow execution failed: {e}")
            self.voice.speak_error(f"Workflow execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def review_code_command(self, file_path: str, review_type: str = "general") -> Dict[str, Any]:
        """
        Command handler for code review.
        
        Args:
            file_path: Path to file to review
            review_type: Type of review (security, performance, style, general)
        
        Returns:
            Dict containing review result
        """
        try:
            self.voice.speak_agent_start("code_reviewer", f"Reviewing {Path(file_path).name}")
            
            # Import advisor agent (dynamic import to avoid circular dependencies)
            from ..agents.advisor_agent import AdvisorAgent
            
            # Create advisor agent
            advisor = AdvisorAgent()
            
            # Create task for code review
            task = create_project_task(
                title="Code Review",
                description=f"Review {file_path} for {review_type} issues",
                task_type="VALIDATE",
                agent_type=AgentType.ADVISOR
            )
            
            # Execute review
            result = await advisor.execute(task)
            
            if result.success:
                self.voice.speak_success("Code review completed")
                return {
                    "success": True,
                    "message": "Code review completed successfully",
                    "review_results": result.output,
                    "confidence": result.confidence
                }
            else:
                self.voice.speak_error("Code review failed")
                return {
                    "success": False,
                    "error": result.error or "Code review failed"
                }
                
        except Exception as e:
            self.logger.error(f"Code review failed: {e}")
            self.voice.speak_error(f"Code review error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def validate_project_command(self, deep_check: bool = False) -> Dict[str, Any]:
        """
        Command handler for project validation.
        
        Args:
            deep_check: Whether to perform deep validation checks
        
        Returns:
            Dict containing validation result
        """
        try:
            self.voice.speak_agent_start("validator", "Validating project structure")
            
            # Import tester agent (dynamic import to avoid circular dependencies)
            from ..agents.tester_agent import TesterAgent
            
            # Create tester agent
            tester = TesterAgent()
            
            # Create task for validation
            task = create_project_task(
                title="Project Validation",
                description="Validate project structure and dependencies",
                task_type="VALIDATE",
                agent_type=AgentType.TESTER
            )
            
            # Execute validation
            result = await tester.execute(task)
            
            if result.success:
                self.voice.speak_success("Project validation completed")
                return {
                    "success": True,
                    "message": "Project validation completed successfully",
                    "validation_results": result.output,
                    "confidence": result.confidence
                }
            else:
                self.voice.speak_error("Project validation failed")
                return {
                    "success": False,
                    "error": result.error or "Project validation failed"
                }
                
        except Exception as e:
            self.logger.error(f"Project validation failed: {e}")
            self.voice.speak_error(f"Validation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_agent_command(self, agent_name: str, agent_type: str = "general") -> Dict[str, Any]:
        """
        Command handler for creating new agent.
        
        Args:
            agent_name: Name for the new agent
            agent_type: Type of agent to create
        
        Returns:
            Dict containing creation result
        """
        try:
            self.voice.speak_agent_start("agent_creator", f"Creating agent {agent_name}")
            
            # Import coder agent (dynamic import to avoid circular dependencies)
            from ..agents.coder_agent import CoderAgent
            
            # Create coder agent
            coder = CoderAgent()
            
            # Create task for agent creation
            task = create_project_task(
                title="Create Agent",
                description=f"Create new {agent_type} agent named {agent_name}",
                task_type="CREATE",
                agent_type=AgentType.CODER
            )
            
            # Execute agent creation
            result = await coder.execute(task)
            
            if result.success:
                self.voice.speak_success(f"Agent {agent_name} created successfully")
                return {
                    "success": True,
                    "message": f"Agent {agent_name} created successfully",
                    "agent_details": result.output,
                    "confidence": result.confidence
                }
            else:
                self.voice.speak_error("Agent creation failed")
                return {
                    "success": False,
                    "error": result.error or "Agent creation failed"
                }
                
        except Exception as e:
            self.logger.error(f"Agent creation failed: {e}")
            self.voice.speak_error(f"Agent creation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def setup_event_handlers(self):
        """Setup event handlers for VS Code events."""
        # Register default event handlers
        self.register_event_handler("on_file_save", self._handle_file_save)
        self.register_event_handler("on_file_change", self._handle_file_change)
        self.register_event_handler("on_project_open", self._handle_project_open)
        self.register_event_handler("on_git_commit", self._handle_git_commit)
        
        self.logger.info("Event handlers registered")
    
    def register_event_handler(self, event_name: str, handler: Callable):
        """
        Register an event handler.
        
        Args:
            event_name: Name of the event
            handler: Handler function to register
        """
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        
        self.event_handlers[event_name].append(handler)
        self.logger.debug(f"Registered handler for event: {event_name}")
    
    async def trigger_event(self, event_name: str, **kwargs):
        """
        Trigger an event and call all registered handlers.
        
        Args:
            event_name: Name of the event to trigger
            **kwargs: Event data
        """
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(**kwargs)
                    else:
                        handler(**kwargs)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_name}: {e}")
    
    async def _handle_file_save(self, file_path: str, **kwargs):
        """Handle file save events."""
        try:
            file_path_obj = Path(file_path)
            
            # Trigger code review for Python files
            if file_path_obj.suffix == '.py':
                self.voice.speak("File saved—review pending")
                
                # Trigger async code review
                asyncio.create_task(self.review_code_command(file_path, "style"))
                
            self.logger.info(f"File saved: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error handling file save: {e}")
    
    async def _handle_file_change(self, file_path: str, **kwargs):
        """Handle file change events."""
        try:
            self.logger.debug(f"File changed: {file_path}")
            
            # Add any file change logic here
            # For example, invalidate caches, update indexes, etc.
            
        except Exception as e:
            self.logger.error(f"Error handling file change: {e}")
    
    async def _handle_project_open(self, project_path: str, **kwargs):
        """Handle project open events."""
        try:
            self.voice.speak("Project opened")
            self.logger.info(f"Project opened: {project_path}")
            
            # Trigger project validation
            asyncio.create_task(self.validate_project_command(deep_check=False))
            
        except Exception as e:
            self.logger.error(f"Error handling project open: {e}")
    
    async def _handle_git_commit(self, commit_hash: str, message: str, **kwargs):
        """Handle git commit events."""
        try:
            self.voice.speak("Code committed successfully")
            self.logger.info(f"Git commit: {commit_hash[:8]} - {message}")
            
        except Exception as e:
            self.logger.error(f"Error handling git commit: {e}")
    
    def _format_prp_content(self, parser_output: Any, project_type: str, description: str) -> str:
        """
        Format PRP content from parser output.
        
        Args:
            parser_output: Output from parser agent
            project_type: Type of project
            description: Original description
        
        Returns:
            Formatted PRP content
        """
        timestamp = datetime.now().isoformat()
        
        # Basic PRP template
        prp_template = f"""# PRP Generated: {description}

## Goal
{description}

## Why
Generated PRP for {project_type} project to implement the requested feature.

## What
{parser_output if isinstance(parser_output, str) else str(parser_output)}

## Implementation Blueprint
[To be filled based on parser output]

## Success Criteria
- [ ] Implementation meets requirements
- [ ] All tests pass
- [ ] Code follows project conventions
- [ ] Documentation is updated

## Validation Loop
[To be defined based on specific requirements]

## Generated Details
- **Generated At**: {timestamp}
- **Project Type**: {project_type}
- **Original Description**: {description}
- **Parser Confidence**: N/A

**Note**: This is a generated PRP and may need refinement based on specific project requirements.
"""
        
        return prp_template
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of Claude Code integration.
        
        Returns:
            Dict containing status information
        """
        return {
            "workspace_root": str(self.workspace_root),
            "claude_dir_exists": self.claude_dir.exists(),
            "commands_dir_exists": self.commands_dir.exists(),
            "hooks_dir_exists": self.hooks_dir.exists(),
            "config_file_exists": self.config_file.exists(),
            "registered_commands": len(self.command_handlers),
            "registered_events": len(self.event_handlers),
            "event_handlers": {
                event: len(handlers) for event, handlers in self.event_handlers.items()
            }
        }
    
    def get_commands_list(self) -> List[str]:
        """
        Get list of available commands.
        
        Returns:
            List of command names
        """
        return list(self.command_handlers.keys())
    
    def get_events_list(self) -> List[str]:
        """
        Get list of available events.
        
        Returns:
            List of event names
        """
        return list(self.event_handlers.keys())