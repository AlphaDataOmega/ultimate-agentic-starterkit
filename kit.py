#!/usr/bin/env python3
"""
Ultimate Agentic StarterKit - Main CLI Interface

This is the main command-line interface for the Ultimate Agentic StarterKit.
It provides a comprehensive interface for AI-driven project execution.
"""

import argparse
import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from core.config import get_config
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts
from workflows.project_builder import LangGraphWorkflowManager
from validation.validator import ValidationOrchestrator

console = Console()
logger = get_logger("cli")


class StarterKitOrchestrator:
    """Main orchestrator for the complete agentic workflow."""
    
    def __init__(self):
        self.project_manager = None
        self.work_order_manager = None
        self.knowledge_base = None
        self.voice_alerts = get_voice_alerts()
        self.logger = get_logger("orchestrator")
    
    async def run_complete_workflow(self, overview_path: str) -> Dict[str, Any]:
        """Run the complete agentic workflow with incremental work order execution."""
        try:
            console.print("[bold blue]🚀 Starting Complete Agentic Workflow[/bold blue]")
            
            # Initialize components
            from agents.enhanced_project_manager import EnhancedProjectManager
            from core.work_order_manager import WorkOrderManager
            from core.knowledge_base import ProjectKnowledgeBase
            
            self.project_manager = EnhancedProjectManager()
            self.work_order_manager = WorkOrderManager()
            self.knowledge_base = ProjectKnowledgeBase()
            
            # Phase 1: Interactive Learning
            console.print("\n[cyan]Phase 1: Interactive Learning[/cyan]")
            overview_content = Path(overview_path).read_text()
            
            # Create project task for enhanced project manager
            from core.models import ProjectTask, AgentType
            task = ProjectTask(
                id="interactive-learning",
                title="Interactive Learning Phase",
                description=overview_content,
                type="CREATE",
                agent_type=AgentType.ADVISOR
            )
            
            # Execute enhanced project manager (includes interactive learning)
            with console.status("[bold green]Running interactive learning phase..."):
                pm_result = await self.project_manager.execute(task)
            
            if not pm_result.success:
                raise Exception(f"Interactive learning failed: {pm_result.error}")
            
            console.print(f"[green]✓[/green] Interactive learning completed")
            console.print(f"[green]✓[/green] Asked {pm_result.output['questions_asked']} questions")
            console.print(f"[green]✓[/green] Project type: {pm_result.output['project_type']}")
            console.print(f"[green]✓[/green] Initialized {len(pm_result.output['initialized_documents'])} documents")
            
            # Phase 2: Project Setup
            console.print("\n[cyan]Phase 2: Project Setup[/cyan]")
            project_type = pm_result.output['project_type']
            project_context = pm_result.output['project_context']
            
            # Initialize work order manager for incremental execution
            await self.work_order_manager.initialize_project_for_incremental_execution(
                project_type, project_context
            )
            
            console.print(f"[green]✓[/green] Project initialized for incremental execution")
            
            # Phase 3: Incremental Work Order Execution
            console.print("\n[cyan]Phase 3: Incremental Work Order Execution[/cyan]")
            
            executed_orders = []
            failed_orders = []
            
            while True:
                # Create next work order
                work_order_id = await self.work_order_manager.create_next_work_order(
                    await self.knowledge_base.get_project_context()
                )
                
                if not work_order_id:
                    console.print("[green]✓[/green] All work orders completed!")
                    break
                
                console.print(f"\n[yellow]Executing work order: {work_order_id}[/yellow]")
                
                # Voice notification
                if self.voice_alerts:
                    self.voice_alerts.speak_milestone(f"Executing work order {work_order_id}")
                
                # Execute work order
                with console.status(f"[bold green]Executing {work_order_id}..."):
                    result = await self.work_order_manager.execute_work_order(work_order_id)
                
                if result["success"]:
                    console.print(f"[green]✓[/green] Completed {work_order_id}")
                    executed_orders.append(work_order_id)
                    
                    # Analyze completion and update context
                    completion_analysis = await self.work_order_manager.analyze_completion_and_plan_next(result)
                    
                    # Update knowledge base with completion
                    await self.knowledge_base.update_from_completion(result)
                    
                    # Voice notification for milestone
                    if self.voice_alerts:
                        self.voice_alerts.speak_milestone(f"Work order {work_order_id} completed")
                
                else:
                    console.print(f"[red]✗[/red] Failed {work_order_id}: {result.get('error', 'Unknown error')}")
                    failed_orders.append(work_order_id)
                    
                    # Voice notification for failure
                    if self.voice_alerts:
                        self.voice_alerts.speak_error(f"Work order {work_order_id} failed")
                    
                    # Break on failure for now (could implement retry logic)
                    break
            
            # Phase 4: Project Completion
            console.print("\n[cyan]Phase 4: Project Completion[/cyan]")
            await self.finalize_project()
            
            # Final voice notification
            if self.voice_alerts:
                if failed_orders:
                    self.voice_alerts.speak_error("Project execution completed with failures")
                else:
                    self.voice_alerts.speak_milestone("Project execution completed successfully!")
            
            return {
                "success": len(failed_orders) == 0,
                "executed_orders": executed_orders,
                "failed_orders": failed_orders,
                "total_executed": len(executed_orders),
                "project_type": project_type,
                "interactive_learning_completed": True
            }
            
        except Exception as e:
            self.logger.error(f"Complete workflow failed: {e}")
            console.print(f"[red]✗[/red] Complete workflow failed: {e}")
            
            if self.voice_alerts:
                self.voice_alerts.speak_error("Complete workflow failed")
            
            return {"success": False, "error": str(e)}
    
    async def finalize_project(self):
        """Finalize project after all work orders are complete."""
        console.print("[green]✓[/green] Project finalization complete")
        
        # Could add final validation, documentation updates, etc.
        # For now, just confirm completion
        pass


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
    
    async def run_complete_workflow(self, overview_file: str = "OVERVIEW.md") -> Dict[str, Any]:
        """Run the complete agentic workflow with new orchestrator."""
        try:
            console.print("[bold blue]🚀 Starting Complete Agentic Workflow[/bold blue]")
            
            # Initialize and run the orchestrator
            orchestrator = StarterKitOrchestrator()
            result = await orchestrator.run_complete_workflow(overview_file)
            
            # Display results
            if result["success"]:
                console.print(f"\n[green]✓ Complete workflow succeeded![/green]")
                console.print(f"Project type: {result.get('project_type', 'Unknown')}")
                console.print(f"Work orders executed: {result.get('total_executed', 0)}")
                console.print(f"Interactive learning: {'✓' if result.get('interactive_learning_completed') else '✗'}")
            else:
                console.print(f"\n[red]✗ Complete workflow failed: {result.get('error', 'Unknown error')}[/red]")
            
            return result
            
        except Exception as e:
            logger.error(f"Complete workflow execution failed: {e}")
            console.print(f"[red]✗[/red] Complete workflow execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_prp(self, description: str) -> Dict[str, Any]:
        """Generate PRP from description."""
        try:
            from integrations.claude_code import ClaudeCodeIntegration
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
        description="Ultimate Agentic StarterKit - AI-Driven Project Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --workflow OVERVIEW.md                   # Complete agentic workflow with AI-driven work orders
  %(prog)s --generate-prp "Build a REST API"        # Generate PRP from description
  %(prog)s --status                                  # Show system status
        """
    )
    
    # Main commands
    parser.add_argument(
        "--workflow",
        type=str,
        help="Run complete agentic workflow with AI-driven work orders and documentation"
    )
    
    parser.add_argument(
        "--generate-prp",
        type=str,
        help="Generate PRP from description"
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
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize CLI
    cli = AgenticStarterKitCLI()
    await cli.initialize()
    
    # Handle different commands
    if args.workflow:
        result = await cli.run_complete_workflow(args.workflow)
        sys.exit(0 if result.get("success", False) else 1)
    
    elif args.generate_prp:
        result = await cli.generate_prp(args.generate_prp)
        sys.exit(0 if result.get("success", False) else 1)
    
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