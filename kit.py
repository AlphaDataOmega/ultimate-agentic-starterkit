#!/usr/bin/env python3
"""
Ultimate Agentic StarterKit - Main CLI Interface

This is the main command-line interface for the Ultimate Agentic StarterKit.
It provides a comprehensive interface for project execution, validation, and management.
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add StarterKit to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StarterKit'))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from StarterKit.core.config import get_config
from StarterKit.core.logger import get_logger
from StarterKit.core.voice_alerts import get_voice_alerts
from StarterKit.workflows.project_builder import LangGraphWorkflowManager
from StarterKit.validation.validator import ValidationOrchestrator

console = Console()
logger = get_logger("cli")


class AgenticStarterKitCLI:
    """Main CLI application for the Ultimate Agentic StarterKit."""
    
    def __init__(self):
        self.config = None
        self.voice_alerts = None
        self.workflow_manager = None
        self.validator = None
        
    async def initialize(self):
        """Initialize CLI components."""
        try:
            # Load and validate configuration
            self.config = get_config()
            
            # Initialize components
            self.voice_alerts = get_voice_alerts()
            self.workflow_manager = LangGraphWorkflowManager()
            self.validator = ValidationOrchestrator()
            
            console.print("[green]✓[/green] Agentic StarterKit initialized successfully")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Initialization failed: {e}")
            sys.exit(1)
    
    async def execute_project(self, prp_file: str, validate_only: bool = False, 
                            voice_alerts: bool = True, dry_run: bool = False) -> Dict[str, Any]:
        """Execute project from PRP file."""
        try:
            # Load PRP file
            prp_path = Path(prp_file)
            if not prp_path.exists():
                raise FileNotFoundError(f"PRP file not found: {prp_file}")
            
            with console.status(f"[bold green]Loading PRP file: {prp_file}"):
                project_spec = await self._load_prp_file(prp_path)
            
            # Voice alert for project start
            if voice_alerts and self.voice_alerts:
                self.voice_alerts.speak_milestone(f"Starting project: {project_spec['title']}")
            
            # Validation only mode
            if validate_only:
                return await self._validate_project(project_spec)
            
            # Dry run mode
            if dry_run:
                return await self._dry_run_project(project_spec)
            
            # Full execution
            return await self._execute_full_project(project_spec, voice_alerts)
            
        except Exception as e:
            logger.error(f"Project execution failed: {e}")
            console.print(f"[red]✗[/red] Project execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _load_prp_file(self, prp_path: Path) -> Dict[str, Any]:
        """Load and parse PRP file."""
        # Import parser agent
        from StarterKit.agents.parser_agent import ParserAgent
        from StarterKit.core.models import ProjectTask, AgentType
        
        # Read PRP content
        content = prp_path.read_text()
        
        # Parse with parser agent
        parser = ParserAgent()
        task = ProjectTask(
            id="prp-parse",
            title="Parse PRP File",
            description=content,
            type="PARSE",
            agent_type=AgentType.PARSER
        )
        
        result = await parser.execute(task)
        
        if not result.success:
            raise ValueError(f"Failed to parse PRP file: {result.error}")
        
        return result.output
    
    async def _validate_project(self, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate project specification."""
        with console.status("[bold blue]Validating project specification..."):
            validation_result = await self.validator.validate_project_spec(project_spec)
        
        # Display validation results
        self._display_validation_results(validation_result)
        
        return validation_result
    
    async def _dry_run_project(self, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Perform dry run of project execution."""
        console.print("[yellow]Dry Run Mode - No actual changes will be made[/yellow]")
        
        # Create execution plan
        with console.status("[bold blue]Creating execution plan..."):
            from StarterKit.core.orchestrator import O3Orchestrator
            orchestrator = O3Orchestrator()
            
            # Convert to ProjectSpecification
            from StarterKit.core.models import ProjectSpecification
            spec = ProjectSpecification(**project_spec)
            
            execution_plan = await orchestrator.create_execution_plan(spec)
        
        # Display execution plan
        self._display_execution_plan(execution_plan)
        
        return {
            "success": True,
            "execution_plan": execution_plan,
            "dry_run": True
        }
    
    async def _execute_full_project(self, project_spec: Dict[str, Any], voice_alerts: bool) -> Dict[str, Any]:
        """Execute complete project workflow."""
        
        # Create progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True
        ) as progress:
            
            # Estimate total tasks
            total_tasks = len(project_spec.get("tasks", []))
            main_task = progress.add_task("Executing project workflow", total=total_tasks)
            
            # Execute workflow
            def progress_callback(completed, total):
                progress.update(main_task, completed=completed, total=total)
            
            result = await self.workflow_manager.execute_workflow(
                project_spec,
                progress_callback=progress_callback
            )
            
            # Voice alert for completion
            if voice_alerts and self.voice_alerts:
                status = result.get("workflow_status", "unknown")
                if status == "completed":
                    self.voice_alerts.speak_milestone("Project completed successfully!")
                else:
                    self.voice_alerts.speak_error("Project execution failed.")
        
        # Display results
        self._display_execution_results(result)
        
        return result
    
    def _display_validation_results(self, validation_result: Dict[str, Any]):
        """Display validation results in formatted table."""
        table = Table(title="Validation Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")
        
        for check_name, check_result in validation_result.get("checks", {}).items():
            status = "✓ PASS" if check_result.get("passed", False) else "✗ FAIL"
            details = check_result.get("message", "")
            table.add_row(check_name, status, details)
        
        console.print(table)
        
        # Overall result
        overall_status = validation_result.get("overall_status", "unknown")
        if overall_status == "passed":
            console.print(Panel("[green]✓ Project validation passed[/green]", title="Result"))
        else:
            console.print(Panel("[red]✗ Project validation failed[/red]", title="Result"))
    
    def _display_execution_plan(self, execution_plan: Dict[str, Any]):
        """Display execution plan in formatted table."""
        table = Table(title="Execution Plan")
        table.add_column("Task", style="cyan")
        table.add_column("Priority", style="magenta")
        table.add_column("Dependencies", style="yellow")
        table.add_column("Agent Type", style="green")
        
        for task_info in execution_plan.get("execution_order", []):
            task_id = task_info.get("task_id", "Unknown")
            priority = task_info.get("priority", "Medium")
            dependencies = ", ".join(task_info.get("dependencies", []))
            agent_type = task_info.get("agent_type", "Unknown")
            
            table.add_row(task_id, priority, dependencies, agent_type)
        
        console.print(table)
        
        # Critical path
        critical_path = execution_plan.get("critical_path", [])
        if critical_path:
            console.print(f"[bold red]Critical Path:[/bold red] {' → '.join(critical_path)}")
    
    def _display_execution_results(self, result: Dict[str, Any]):
        """Display execution results."""
        status = result.get("workflow_status", "unknown")
        confidence = result.get("overall_confidence", 0.0)
        
        if status == "completed":
            console.print(Panel(
                f"[green]✓ Project completed successfully![/green]\n"
                f"Overall Confidence: {confidence:.2f}",
                title="Execution Results"
            ))
        else:
            console.print(Panel(
                f"[red]✗ Project execution failed[/red]\n"
                f"Status: {status}\n"
                f"Confidence: {confidence:.2f}",
                title="Execution Results"
            ))
        
        # Display completed tasks
        completed_tasks = result.get("completed_tasks", [])
        failed_tasks = result.get("failed_tasks", [])
        
        if completed_tasks:
            console.print(f"[green]Completed tasks ({len(completed_tasks)}):[/green] {', '.join(completed_tasks)}")
        
        if failed_tasks:
            console.print(f"[red]Failed tasks ({len(failed_tasks)}):[/red] {', '.join(failed_tasks)}")
    
    async def generate_prp(self, description: str) -> Dict[str, Any]:
        """Generate PRP from description."""
        try:
            from StarterKit.integrations.claude_code import ClaudeCodeIntegration
            integration = ClaudeCodeIntegration(".")
            
            result = await integration.generate_prp_command(description)
            
            if result.get("success", False):
                console.print(f"[green]✓[/green] PRP generated: {result.get('file_path', 'Unknown')}")
            else:
                console.print(f"[red]✗[/red] PRP generation failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"PRP generation failed: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/red] {error_msg}")
            return {"success": False, "error": error_msg}
    
    def list_examples(self):
        """List available examples."""
        examples_dir = Path("examples")
        if examples_dir.exists():
            console.print("[bold cyan]Available Examples:[/bold cyan]")
            for example_file in examples_dir.glob("*.py"):
                console.print(f"  • {example_file.name}")
        else:
            console.print("[yellow]No examples directory found[/yellow]")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            "system_initialized": self.config is not None,
            "voice_alerts_enabled": self.voice_alerts is not None,
            "workflow_manager_ready": self.workflow_manager is not None,
            "validator_ready": self.validator is not None
        }
        
        if self.voice_alerts:
            status["voice_status"] = self.voice_alerts.get_status()
        
        return status


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Ultimate Agentic StarterKit - Automated Project Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prp PRPs/web_app.md                    # Execute web app project
  %(prog)s --prp PRPs/api_service.md --validate     # Validate API service project
  %(prog)s --prp PRPs/ml_model.md --dry-run         # Dry run ML model project
  %(prog)s --generate-prp "Build a REST API"        # Generate PRP from description
  %(prog)s --list-examples                           # List available examples
  %(prog)s --status                                  # Show system status
        """
    )
    
    # Main commands
    parser.add_argument(
        "--prp", 
        type=str, 
        help="Path to PRP (Project Requirements & Planning) file"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate project specification only (no execution)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without actual execution"
    )
    
    parser.add_argument(
        "--generate-prp",
        type=str,
        help="Generate PRP from description"
    )
    
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List available examples"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status"
    )
    
    # Options
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice alerts"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume workflow from saved state file"
    )
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize CLI
    cli = AgenticStarterKitCLI()
    await cli.initialize()
    
    # Handle different commands
    if args.prp:
        result = await cli.execute_project(
            args.prp,
            validate_only=args.validate,
            voice_alerts=not args.no_voice,
            dry_run=args.dry_run
        )
        
        # Exit with appropriate code
        sys.exit(0 if result.get("success", False) else 1)
    
    elif args.generate_prp:
        result = await cli.generate_prp(args.generate_prp)
        sys.exit(0 if result.get("success", False) else 1)
    
    elif args.list_examples:
        cli.list_examples()
    
    elif args.status:
        status = cli.get_status()
        console.print(json.dumps(status, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)