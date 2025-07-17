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
        """Execute interactive learning phase for project knowledge transfer."""
        try:
            console.print("[bold blue]🎓 Starting Interactive Learning Phase[/bold blue]")
            
            # Load OVERVIEW.md file
            overview_path = Path(overview_file)
            if not overview_path.exists():
                raise FileNotFoundError(f"OVERVIEW.md file not found: {overview_file}")
            
            overview_content = overview_path.read_text()
            
            # Voice alert
            if self.voice_alerts:
                self.voice_alerts.speak_milestone("Starting project learning phase")
            
            # Run learning loop
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                console.print(f"\n[cyan]Learning Iteration {iteration}/{max_iterations}[/cyan]")
                
                # Step 1: Project Manager generates questions
                with console.status("[bold green]Analyzing project and generating questions..."):
                    pm_result = await self._run_project_manager(overview_content)
                
                if not pm_result['success']:
                    console.print(f"[red]✗[/red] Project analysis failed: {pm_result.get('error')}")
                    break
                
                questions_file = pm_result['output']['questions_file']
                total_questions = pm_result['output']['total_questions']
                critical_questions = pm_result['output']['critical_questions']
                
                console.print(f"[green]✓[/green] Generated {total_questions} questions ({critical_questions} critical)")
                
                # Step 2: Research Agent fills in answers
                with console.status("[bold green]Researching answers..."):
                    research_result = await self._run_research_agent(questions_file)
                
                if research_result['success']:
                    researched_count = research_result['output']['questions_researched']
                    console.print(f"[green]✓[/green] Researched {researched_count} questions")
                
                # Step 3: User review prompt
                console.print(f"\n[yellow]📝 Questions ready for review![/yellow]")
                console.print(f"Please review and edit: [bold]{questions_file}[/bold]")
                console.print("Add your answers and check boxes [x] when complete")
                
                if self.voice_alerts:
                    self.voice_alerts.speak_milestone("Questions ready - edit questions.md, check boxes and add answers, then accept learn")
                
                console.print("\n[bold cyan]Run `python kit.py --accept-learn` when ready to continue[/bold cyan]")
                
                return {
                    'success': True,
                    'status': 'awaiting_user_input',
                    'questions_file': questions_file,
                    'iteration': iteration,
                    'total_questions': total_questions,
                    'critical_questions': critical_questions
                }
            
            # Max iterations reached
            console.print(f"[yellow]⚠️[/yellow] Maximum learning iterations ({max_iterations}) reached")
            return {
                'success': True,
                'status': 'max_iterations_reached',
                'iteration': iteration
            }
            
        except Exception as e:
            logger.error(f"Learning phase failed: {e}")
            console.print(f"[red]✗[/red] Learning phase failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def accept_learning_results(self) -> Dict[str, Any]:
        """Accept learning results and incorporate into project overview."""
        try:
            console.print("[bold blue]📚 Accepting Learning Results[/bold blue]")
            
            # Check if questions.md exists
            questions_file = Path("questions.md")
            if not questions_file.exists():
                raise FileNotFoundError("No questions.md file found. Run --learn first.")
            
            # Parse answered questions
            questions_content = questions_file.read_text()
            answered_questions = self._parse_answered_questions(questions_content)
            
            # Check if user provided answers
            answered_count = len([q for q in answered_questions if q.get('user_answer')])
            total_questions = len(answered_questions)
            
            console.print(f"Found {answered_count}/{total_questions} answered questions")
            
            if answered_count == 0:
                console.print("[yellow]⚠️[/yellow] No answers found. Please edit questions.md first.")
                return {"success": False, "error": "No answers provided"}
            
            # Incorporate answers into OVERVIEW.md
            with console.status("[bold green]Incorporating answers into project overview..."):
                updated_overview = await self._incorporate_answers_into_overview(answered_questions)
            
            # Save updated overview with version backup
            overview_path = Path("OVERVIEW.md")
            if overview_path.exists():
                # Create backup
                backup_path = Path(f"OVERVIEW_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                backup_path.write_text(overview_path.read_text())
                console.print(f"[green]✓[/green] Backed up original to {backup_path}")
            
            overview_path.write_text(updated_overview)
            console.print(f"[green]✓[/green] Updated OVERVIEW.md with learning results")
            
            # Archive questions
            self._archive_questions(questions_file)
            
            # Check if learning is complete
            from agents.project_manager_agent import ProjectManagerAgent
            pm_agent = ProjectManagerAgent()
            is_complete = await pm_agent.check_learning_completion(updated_overview)
            
            if is_complete:
                console.print("[green]🎉 Learning phase complete![/green]")
                if self.voice_alerts:
                    self.voice_alerts.speak_milestone("Learning complete - overview updated. Proceed to context and planning?")
                
                # Auto-generate context if requested
                console.print("\n[cyan]Would you like to proceed with project execution? (y/n)[/cyan]")
                user_input = input().strip().lower()
                
                if user_input in ['y', 'yes']:
                    return await self.execute_overview("OVERVIEW.md")
                
                return {
                    'success': True,
                    'status': 'learning_complete',
                    'updated_overview': str(overview_path),
                    'answered_questions': answered_count
                }
            else:
                console.print("[yellow]📝 Learning continues - more details needed[/yellow]")
                if self.voice_alerts:
                    self.voice_alerts.speak_milestone("Learning continues - running another iteration")
                
                # Run another learning iteration
                return await self.execute_learning_phase()
            
        except Exception as e:
            logger.error(f"Accept learning failed: {e}")
            console.print(f"[red]✗[/red] Accept learning failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_project_manager(self, overview_content: str) -> Dict[str, Any]:
        """Run Project Manager Agent to generate questions."""
        from agents.project_manager_agent import ProjectManagerAgent
        from core.models import ProjectTask, AgentType
        
        pm_agent = ProjectManagerAgent()
        task = ProjectTask(
            id="pm-analysis",
            title="Project Analysis and Question Generation",
            description=overview_content,
            type="CREATE",
            agent_type=AgentType.ADVISOR
        )
        
        result = await pm_agent.execute(task)
        return {
            'success': result.success,
            'output': result.output if result.success else {},
            'error': result.error if not result.success else None
        }
    
    async def _run_research_agent(self, questions_file: str) -> Dict[str, Any]:
        """Run Research Agent to fill in suggested answers."""
        from agents.research_agent import ResearchAgent
        from core.models import ProjectTask, AgentType
        
        research_agent = ResearchAgent()
        task = ProjectTask(
            id="research-questions",
            title="Research Project Questions",
            description=questions_file,
            type="MODIFY",
            agent_type=AgentType.ADVISOR
        )
        
        result = await research_agent.execute(task)
        return {
            'success': result.success,
            'output': result.output if result.success else {},
            'error': result.error if not result.success else None
        }
    
    def _parse_answered_questions(self, questions_content: str) -> List[Dict[str, Any]]:
        """Parse questions.md to extract user answers."""
        from agents.research_agent import ResearchAgent
        
        research_agent = ResearchAgent()
        return research_agent._parse_questions_document(questions_content)
    
    async def _incorporate_answers_into_overview(self, answered_questions: List[Dict[str, Any]]) -> str:
        """Incorporate answered questions into OVERVIEW.md."""
        
        # Load current overview
        overview_path = Path("OVERVIEW.md")
        current_overview = overview_path.read_text() if overview_path.exists() else ""
        
        # Create sections for answered questions
        sections_to_add = []
        
        # Group answers by category
        tech_answers = [q for q in answered_questions if 'tech' in q['question'].lower() or 'stack' in q['question'].lower()]
        auth_answers = [q for q in answered_questions if 'auth' in q['question'].lower()]
        db_answers = [q for q in answered_questions if 'database' in q['question'].lower() or 'data' in q['question'].lower()]
        feature_answers = [q for q in answered_questions if 'feature' in q['question'].lower() or 'requirement' in q['question'].lower()]
        
        # Add technology stack section
        if tech_answers:
            tech_section = "## Technology Stack Details\n\n"
            for q in tech_answers:
                if q.get('user_answer'):
                    tech_section += f"- **{q['question'].replace('?', '')}**: {q['user_answer']}\n"
            sections_to_add.append(tech_section)
        
        # Add authentication section
        if auth_answers:
            auth_section = "## Authentication & Security\n\n"
            for q in auth_answers:
                if q.get('user_answer'):
                    auth_section += f"- **{q['question'].replace('?', '')}**: {q['user_answer']}\n"
            sections_to_add.append(auth_section)
        
        # Add database section
        if db_answers:
            db_section = "## Data Management\n\n"
            for q in db_answers:
                if q.get('user_answer'):
                    db_section += f"- **{q['question'].replace('?', '')}**: {q['user_answer']}\n"
            sections_to_add.append(db_section)
        
        # Add additional requirements
        if feature_answers:
            req_section = "## Additional Requirements\n\n"
            for q in feature_answers:
                if q.get('user_answer'):
                    req_section += f"- **{q['question'].replace('?', '')}**: {q['user_answer']}\n"
            sections_to_add.append(req_section)
        
        # Insert sections into overview
        if sections_to_add:
            # Add learning results section
            learning_section = f"\n\n<!-- Learning Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"
            learning_section += "\n".join(sections_to_add)
            learning_section += "\n<!-- End Learning Results -->\n"
            
            updated_overview = current_overview + learning_section
        else:
            updated_overview = current_overview
        
        return updated_overview
    
    def _archive_questions(self, questions_file: Path):
        """Archive questions.md to questions/history/."""
        
        # Create history directory
        history_dir = Path("questions/history")
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Move questions file to history
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = history_dir / f"questions_completed_{timestamp}.md"
        
        questions_file.rename(archive_path)
        console.print(f"[green]✓[/green] Archived questions to {archive_path}")
    
    async def build_complete_project(self, overview_file: str = "OVERVIEW.md") -> Dict[str, Any]:
        """Execute complete project build workflow with enhanced project manager."""
        try:
            console.print("[bold blue]🏗️ Starting Complete Project Build Workflow[/bold blue]")
            
            # Load OVERVIEW.md file
            overview_path = Path(overview_file)
            if not overview_path.exists():
                raise FileNotFoundError(f"OVERVIEW.md file not found: {overview_file}")
            
            overview_content = overview_path.read_text()
            
            # Voice alert
            if self.voice_alerts:
                self.voice_alerts.speak_milestone("Starting complete project build workflow")
            
            # Phase 1: Enhanced Project Management
            with console.status("[bold green]Creating multi-document knowledge base..."):
                from agents.enhanced_project_manager import EnhancedProjectManager
                from core.models import ProjectTask, AgentType
                
                enhanced_pm = EnhancedProjectManager()
                pm_task = ProjectTask(
                    id="enhanced-pm-setup",
                    title="Enhanced Project Management Setup",
                    description=overview_content,
                    type="CREATE",
                    agent_type=AgentType.ADVISOR
                )
                
                pm_result = await enhanced_pm.execute(pm_task)
            
            if not pm_result.success:
                raise Exception(f"Enhanced project management failed: {pm_result.error}")
            
            console.print(f"[green]✓[/green] Created {len(pm_result.output['initialized_documents'])} project documents")
            console.print(f"[green]✓[/green] Generated {pm_result.output['total_work_orders']} work orders")
            
            # Phase 2: Execute Work Orders
            return await self.execute_work_orders()
            
        except Exception as e:
            logger.error(f"Complete project build failed: {e}")
            console.print(f"[red]✗[/red] Complete project build failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_work_orders(self) -> Dict[str, Any]:
        """Execute all pending work orders in sequence."""
        try:
            console.print("[bold blue]⚙️ Executing Work Orders[/bold blue]")
            
            from core.work_order_manager import WorkOrderManager
            
            work_order_manager = WorkOrderManager()
            
            executed_orders = []
            failed_orders = []
            
            # Execute work orders until none are pending
            while True:
                # Get next work order
                next_wo_id = await work_order_manager.knowledge_base.get_next_work_order()
                if not next_wo_id:
                    break
                
                console.print(f"\n[cyan]Executing work order: {next_wo_id}[/cyan]")
                
                if self.voice_alerts:
                    self.voice_alerts.speak_milestone(f"Executing work order {next_wo_id}")
                
                # Execute work order
                with console.status(f"[bold green]Executing {next_wo_id} via Claude..."):
                    result = await work_order_manager.execute_work_order(next_wo_id)
                
                if result["success"]:
                    console.print(f"[green]✓[/green] Completed {next_wo_id}")
                    console.print(f"    Method: {result.get('claude_method', 'unknown')}")
                    console.print(f"    Files: {result.get('files_created', 0)} created")
                    executed_orders.append(next_wo_id)
                else:
                    console.print(f"[red]✗[/red] Failed {next_wo_id}: {result.get('error', 'Unknown error')}")
                    failed_orders.append(next_wo_id)
                    break  # Stop on first failure
            
            # Voice alert for completion
            if self.voice_alerts:
                if failed_orders:
                    self.voice_alerts.speak_error("Work order execution failed")
                else:
                    self.voice_alerts.speak_milestone("All work orders completed successfully!")
            
            # Display results
            console.print(f"\n[bold]Work Order Execution Results[/bold]")
            console.print(f"[green]Completed:[/green] {len(executed_orders)} work orders")
            if failed_orders:
                console.print(f"[red]Failed:[/red] {len(failed_orders)} work orders")
            
            return {
                "success": len(failed_orders) == 0,
                "executed_orders": executed_orders,
                "failed_orders": failed_orders,
                "total_executed": len(executed_orders)
            }
            
        except Exception as e:
            logger.error(f"Work order execution failed: {e}")
            console.print(f"[red]✗[/red] Work order execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def show_work_order_status(self) -> Dict[str, Any]:
        """Show current work order status and progress."""
        try:
            console.print("[bold blue]📊 Work Order Status[/bold blue]")
            
            from core.work_order_manager import WorkOrderManager
            
            work_order_manager = WorkOrderManager()
            status = await work_order_manager.get_work_order_status()
            
            # Display work order summary
            wo_summary = status["work_orders"]
            console.print(f"\n[bold]Work Order Summary[/bold]")
            console.print(f"Total: {wo_summary['total']}")
            console.print(f"[green]Completed:[/green] {wo_summary['completed']}")
            console.print(f"[yellow]Pending:[/yellow] {wo_summary['pending']}")
            console.print(f"[blue]In Progress:[/blue] {wo_summary['in_progress']}")
            
            # Display document status
            console.print(f"\n[bold]Project Documents[/bold]")
            for doc in status["documents"]:
                if doc["exists"]:
                    console.print(f"[green]✓[/green] {doc['type']} ({doc['size']} chars)")
                else:
                    console.print(f"[red]✗[/red] {doc['type']} (not created)")
            
            console.print(f"\nCompletion History: {status['completion_history']} completed work orders")
            
            return status
            
        except Exception as e:
            logger.error(f"Work order status check failed: {e}")
            console.print(f"[red]✗[/red] Status check failed: {e}")
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


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Ultimate Agentic StarterKit - Automated Project Builder",
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
    if args.workflow:
        result = await cli.run_complete_workflow(args.workflow)
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