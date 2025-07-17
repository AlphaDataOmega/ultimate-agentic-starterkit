"""
Work Order Management System for the Ultimate Agentic StarterKit.

This module manages the progressive work order execution system that builds
up project knowledge and creates work orders based on comprehensive context.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from core.knowledge_base import ProjectKnowledgeBase, WorkOrderStatus, ProjectType
from core.models import ProjectTask, AgentResult, AgentType
from core.logger import get_logger
from agents.factory import get_or_create_agent
from integrations.claude_code_executor import ClaudeCodeExecutor


class WorkOrderManager:
    """
    Manages work order creation, execution, and completion tracking.
    Uses AI-driven analysis to determine next steps based on project context.
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize work order manager."""
        self.project_root = Path(project_root)
        self.knowledge_base = ProjectKnowledgeBase(project_root)
        self.logger = get_logger("work_order_manager")
        self.claude_executor = ClaudeCodeExecutor()
        
        # Project state for incremental work order creation
        self.project_type = None
        self.current_step = 0
        self.project_context = None
        
        # Work order execution strategies
        self.execution_strategies = {
            "setup": self._execute_setup_order,
            "implementation": self._execute_implementation_order,
            "testing": self._execute_testing_order,
            "documentation": self._execute_documentation_order,
            "integration": self._execute_integration_order,
            "visual": self._execute_visual_order
        }
    
    async def initialize_project_for_incremental_execution(self, project_type: ProjectType, project_context: Dict[str, Any]) -> bool:
        """
        Initialize project for incremental work order creation.
        
        Returns True if initialization successful.
        """
        self.project_type = project_type
        self.current_step = 0
        self.project_context = project_context
        
        self.logger.info(f"Initialized project for incremental execution: {project_type}")
        return True
    
    async def create_next_work_order(self, project_context: Dict[str, Any]) -> Optional[str]:
        """
        Create the next work order based on current project state using AI analysis.
        
        Returns work order ID or None if project is complete.
        """
        if not self.project_type:
            self.logger.error("Project type not set - call initialize_project_for_incremental_execution first")
            return None
        
        # Use AI to analyze current state and determine next work order
        next_work_order = await self._ai_analyze_next_work_order(project_context)
        
        if not next_work_order:
            self.logger.info("AI analysis indicates project is complete")
            return None
        
        # Create the work order
        wo_id = await self.knowledge_base.create_work_order(
            title=next_work_order["title"],
            description=next_work_order["description"],
            dependencies=next_work_order.get("dependencies", [])
        )
        
        # Increment step counter
        self.current_step += 1
        
        self.logger.info(f"Created AI-generated work order {wo_id}: {next_work_order['title']}")
        return wo_id
    
    async def _ai_analyze_next_work_order(self, project_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use AI to analyze current project state and determine next work order."""
        
        # Get current project state
        current_state = self._build_current_state_summary(project_context)
        
        # Create analysis prompt for AI
        analysis_prompt = f"""
# Project Work Order Analysis

## Current Project State
- **Project Type**: {self.project_type}
- **Current Step**: {self.current_step}
- **Completed Work Orders**: {len(project_context.get('completion_history', []))}

## Project Context
{current_state}

## Completed Work Orders
{self._format_completed_work_orders(project_context)}

## Available Project Documents
{self._format_available_documents(project_context)}

## Analysis Task
Based on the current project state, determine the next logical work order needed to progress toward project completion.

Consider:
1. What has been completed so far
2. What is the natural next step in development
3. Dependencies between different components
4. Project type-specific requirements

## Required Response Format
If there is a next work order needed, respond with JSON:
{{
    "title": "Clear, specific work order title",
    "description": "Detailed description of what needs to be done, including specific tasks and context",
    "dependencies": ["list", "of", "dependency", "work", "order", "ids"],
    "estimated_effort": "low/medium/high",
    "priority": "high/medium/low"
}}

If the project is complete, respond with:
{{
    "complete": true,
    "reason": "Explanation of why the project is complete"
}}
"""
        
        # Use enhanced project manager for analysis
        from agents.enhanced_project_manager import EnhancedProjectManager
        from core.models import ProjectTask, AgentType
        
        pm = EnhancedProjectManager()
        analysis_task = ProjectTask(
            id="work-order-analysis",
            title="Analyze Next Work Order",
            description=analysis_prompt,
            type="ANALYZE",
            agent_type=AgentType.ADVISOR
        )
        
        try:
            result = await pm.execute(analysis_task)
            
            if result.success:
                # Parse the AI response
                response_text = result.output.get("analysis", "").strip()
                
                # Try to extract JSON from response
                import json
                import re
                
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        analysis_result = json.loads(json_match.group())
                        
                        if analysis_result.get("complete"):
                            return None
                        
                        return analysis_result
                    except json.JSONDecodeError:
                        self.logger.warning("Could not parse AI analysis response as JSON")
                
                # Fallback: create a basic next work order
                return self._create_fallback_work_order(project_context)
            
            else:
                self.logger.error(f"AI analysis failed: {result.error}")
                return self._create_fallback_work_order(project_context)
        
        except Exception as e:
            self.logger.error(f"AI analysis error: {e}")
            return self._create_fallback_work_order(project_context)
    
    def _build_current_state_summary(self, project_context: Dict[str, Any]) -> str:
        """Build a summary of current project state."""
        summary = []
        
        # Add project overview
        overview = project_context.get("overview_content", "")
        if overview:
            summary.append(f"**Project Overview**:\n{overview[:500]}...")
        
        # Add refined scope if available
        refined_scope = project_context.get("refined_scope", "")
        if refined_scope and refined_scope != overview:
            summary.append(f"**Refined Scope**:\n{refined_scope[:500]}...")
        
        # Add user responses if available
        user_responses = project_context.get("user_responses", {})
        if user_responses:
            summary.append("**User Clarifications**:")
            for key, value in user_responses.items():
                summary.append(f"- {key}: {value}")
        
        return "\n\n".join(summary)
    
    def _format_completed_work_orders(self, project_context: Dict[str, Any]) -> str:
        """Format completed work orders for AI analysis."""
        completed = project_context.get("completion_history", [])
        if not completed:
            return "None"
        
        formatted = []
        for completion in completed:
            formatted.append(f"- {completion.get('title', 'Unknown')}: {completion.get('status', 'completed')}")
        
        return "\n".join(formatted)
    
    def _format_available_documents(self, project_context: Dict[str, Any]) -> str:
        """Format available documents for AI analysis."""
        documents = project_context.get("documents", {})
        if not documents:
            return "None"
        
        formatted = []
        for doc_name, doc_content in documents.items():
            content_length = len(doc_content) if doc_content else 0
            formatted.append(f"- {doc_name}: {content_length} characters")
        
        return "\n".join(formatted)
    
    def _create_fallback_work_order(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback work order when AI analysis fails."""
        
        # Basic fallback based on current step
        if self.current_step == 0:
            return {
                "title": "Initialize Project Structure",
                "description": "Set up the basic project structure, configuration files, and development environment.",
                "dependencies": [],
                "estimated_effort": "medium",
                "priority": "high"
            }
        elif self.current_step == 1:
            return {
                "title": "Implement Core Functionality",
                "description": "Implement the main functionality based on project requirements.",
                "dependencies": [],
                "estimated_effort": "high",
                "priority": "high"
            }
        elif self.current_step == 2:
            return {
                "title": "Add Testing and Documentation",
                "description": "Create comprehensive tests and update documentation.",
                "dependencies": [],
                "estimated_effort": "medium",
                "priority": "medium"
            }
        else:
            # Project likely complete
            return None
    
    async def analyze_completion_and_plan_next(self, completed_work_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze completed work order and determine next steps.
        
        Returns analysis of what should happen next.
        """
        completion_analysis = {
            "work_order_id": completed_work_order.get("id"),
            "step_completed": self.current_step - 1,  # -1 because we incremented after creating
            "total_steps": len(self.WORK_ORDER_FLOWS.get(self.project_type, [])),
            "next_step_available": self.current_step < len(self.WORK_ORDER_FLOWS.get(self.project_type, [])),
            "lessons_learned": completed_work_order.get("completion_data", {}).get("lessons_learned", []),
            "context_updates": completed_work_order.get("completion_data", {}).get("context_updates", [])
        }
        
        # Update project context with completion results
        if self.project_context:
            self.project_context.update({
                "last_completion": completed_work_order,
                "completion_analysis": completion_analysis
            })
        
        self.logger.info(f"Analyzed completion: Step {completion_analysis['step_completed']} of {completion_analysis['total_steps']}")
        return completion_analysis
    
    async def execute_next_work_order(self) -> Optional[Dict[str, Any]]:
        """Execute the next available work order."""
        next_wo_id = await self.knowledge_base.get_next_work_order()
        if not next_wo_id:
            self.logger.info("No work orders ready for execution")
            return None
        
        return await self.execute_work_order(next_wo_id)
    
    async def execute_work_order(self, work_order_id: str) -> Dict[str, Any]:
        """Execute a specific work order."""
        self.logger.info(f"Executing work order: {work_order_id}")
        
        # Load work order
        wo_file = self.knowledge_base.work_orders_dir / f"{work_order_id}.json"
        if not wo_file.exists():
            raise ValueError(f"Work order file not found: {work_order_id}")
        
        work_order = json.loads(wo_file.read_text())
        
        # Update status to in progress
        work_order["status"] = WorkOrderStatus.IN_PROGRESS
        work_order["updated_at"] = datetime.now().isoformat()
        wo_file.write_text(json.dumps(work_order, indent=2))
        
        try:
            # Get comprehensive context
            context = await self.knowledge_base.get_project_context(work_order_id)
            
            # Determine execution strategy
            wo_type = work_order.get("type", "implementation")
            if wo_type in self.execution_strategies:
                execution_func = self.execution_strategies[wo_type]
            else:
                execution_func = self.execution_strategies["implementation"]
            
            # Execute work order
            result = await execution_func(work_order, context)
            
            if result["success"]:
                # Mark as completed
                await self.knowledge_base.complete_work_order(work_order_id, result)
                
                # Update knowledge base documents if needed
                await self._update_knowledge_from_completion(result)
                
                self.logger.info(f"Successfully completed work order: {work_order_id}")
            else:
                # Mark as failed
                work_order["status"] = WorkOrderStatus.BLOCKED
                work_order["error"] = result.get("error", "Unknown error")
                wo_file.write_text(json.dumps(work_order, indent=2))
                
                self.logger.error(f"Work order failed: {work_order_id} - {result.get('error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Work order execution failed: {work_order_id} - {e}")
            
            # Mark as failed
            work_order["status"] = WorkOrderStatus.BLOCKED
            work_order["error"] = str(e)
            wo_file.write_text(json.dumps(work_order, indent=2))
            
            return {"success": False, "error": str(e)}
    
    async def _execute_setup_order(self, work_order: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a setup work order using Claude Code."""
        
        # Use Claude Code for setup with full context
        result = await self.claude_executor.execute_coding_work_order(work_order, context)
        
        if result["success"]:
            return {
                "success": True,
                "artifacts": result.get("artifacts", []),
                "code_changes": "Project structure and configuration files created via Claude",
                "lessons_learned": ["Project structure established with Claude Code"],
                "context_updates": ["Basic project environment is set up", "Claude Code integration working"],
                "next_steps": "Proceed with data model implementation",
                "claude_method": result.get("method", "unknown")
            }
        else:
            return {"success": False, "error": result.get("error", "Claude execution failed")}
    
    async def _execute_implementation_order(self, work_order: Dict[str, Any],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an implementation work order using Claude Code."""
        
        # Use Claude Code for implementation with full project context
        result = await self.claude_executor.execute_coding_work_order(work_order, context)
        
        if result["success"]:
            return {
                "success": True,
                "artifacts": result.get("artifacts", []),
                "code_changes": f"Implemented {work_order['title']} via Claude",
                "lessons_learned": [f"{work_order['title']} implementation completed with Claude"],
                "context_updates": [f"{work_order['title']} is now functional"],
                "next_steps": "Feature ready for testing",
                "claude_method": result.get("method", "unknown"),
                "files_created": result.get("files_created", 0)
            }
        else:
            return {"success": False, "error": result.get("error", "Claude implementation failed")}
    
    async def _execute_testing_order(self, work_order: Dict[str, Any],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a testing work order."""
        
        # Create task for tester agent
        task = ProjectTask(
            id=f"test-{work_order['id']}",
            title=work_order["title"],
            description=f"""Testing Task: {work_order['title']}

{work_order['description']}

Project Context:
{self._format_context_for_agent(context)}

Implemented Features:
{self._format_completed_implementations(context)}

Create comprehensive tests for all implemented functionality.
""",
            type="TEST",
            agent_type=AgentType.TESTER
        )
        
        # Execute with tester agent
        tester_agent = get_or_create_agent(AgentType.TESTER)
        result = await tester_agent.execute(task)
        
        if result.success:
            return {
                "success": True,
                "artifacts": result.output.get("test_files", []),
                "tests_added": f"Created tests for {work_order['title']}",
                "lessons_learned": ["Test coverage improved"],
                "context_updates": ["All features have test coverage"],
                "next_steps": "Tests can be run to validate functionality"
            }
        else:
            return {"success": False, "error": result.error}
    
    async def _execute_documentation_order(self, work_order: Dict[str, Any],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a documentation work order."""
        
        # Create task for advisor agent
        task = ProjectTask(
            id=f"docs-{work_order['id']}",
            title=work_order["title"],
            description=f"""Documentation Task: {work_order['title']}

{work_order['description']}

Project Context:
{self._format_context_for_agent(context)}

Implementation Details:
{self._format_implementation_details(context)}

Update all project documentation with implementation details and usage instructions.
""",
            type="MODIFY",
            agent_type=AgentType.ADVISOR
        )
        
        # Execute with advisor agent
        advisor_agent = get_or_create_agent(AgentType.ADVISOR)
        result = await advisor_agent.execute(task)
        
        if result.success:
            return {
                "success": True,
                "artifacts": result.output.get("documentation_files", []),
                "documentation_updates": f"Updated documentation for {work_order['title']}",
                "lessons_learned": ["Documentation is current with implementation"],
                "context_updates": ["Project documentation is comprehensive"],
                "next_steps": "Documentation ready for end users"
            }
        else:
            return {"success": False, "error": result.error}
    
    async def _execute_integration_order(self, work_order: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an integration work order."""
        return await self._execute_implementation_order(work_order, context)
    
    async def _execute_visual_order(self, work_order: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a visual work order using Claude's vision capabilities."""
        
        # Check if screenshots are provided
        screenshots = work_order.get("screenshots", [])
        if not screenshots:
            # Look for screenshots in work order directory
            wo_dir = self.knowledge_base.work_orders_dir / f"{work_order['id']}_assets"
            if wo_dir.exists():
                screenshots = [str(f) for f in wo_dir.glob("*.png")]
        
        if screenshots:
            result = await self.claude_executor.execute_visual_work_order(work_order, context, screenshots)
        else:
            # Fall back to regular implementation if no visuals
            result = await self.claude_executor.execute_coding_work_order(work_order, context)
        
        if result["success"]:
            return {
                "success": True,
                "artifacts": result.get("artifacts", []),
                "code_changes": f"Implemented visual feature: {work_order['title']}",
                "lessons_learned": [f"Visual implementation completed for {work_order['title']}"],
                "context_updates": [f"Visual feature {work_order['title']} is functional"],
                "next_steps": "Visual feature ready for testing",
                "claude_method": result.get("method", "unknown"),
                "visual_analysis": result.get("visual_analysis", "None"),
                "screenshots_analyzed": result.get("screenshots_analyzed", 0)
            }
        else:
            return {"success": False, "error": result.get("error", "Visual implementation failed")}
    
    def _format_context_for_agent(self, context: Dict[str, Any]) -> str:
        """Format project context for agent consumption."""
        formatted = ""
        
        # Add key documents
        if "REQUIREMENTS.md" in context["documents"]:
            formatted += "=== REQUIREMENTS ===\n"
            formatted += context["documents"]["REQUIREMENTS.md"][:1000] + "...\n\n"
        
        if "ARCHITECTURE.md" in context["documents"]:
            formatted += "=== ARCHITECTURE ===\n"
            formatted += context["documents"]["ARCHITECTURE.md"][:1000] + "...\n\n"
        
        if "CONTEXT.md" in context["documents"]:
            formatted += "=== WORKING ASSUMPTIONS ===\n"
            formatted += context["documents"]["CONTEXT.md"][:800] + "...\n\n"
        
        return formatted
    
    def _format_completed_work_orders(self, context: Dict[str, Any]) -> str:
        """Format completed work orders for context."""
        completed = context["work_orders"]["completed"]
        if not completed:
            return "None"
        
        formatted = ""
        for wo in completed[-5:]:  # Last 5 completed orders
            formatted += f"- {wo.get('title', 'Unknown')}: {wo.get('status', 'completed')}\n"
        
        return formatted
    
    def _format_completed_implementations(self, context: Dict[str, Any]) -> str:
        """Format completed implementations for testing context."""
        implementations = []
        for completion in context["completion_history"]:
            if "Implement" in completion.get("title", ""):
                implementations.append(completion["title"])
        
        return "\n".join(f"- {impl}" for impl in implementations)
    
    def _format_implementation_details(self, context: Dict[str, Any]) -> str:
        """Format implementation details for documentation."""
        details = []
        for completion in context["completion_history"]:
            artifacts = completion.get("artifacts", [])
            if artifacts:
                details.append(f"{completion['title']}: {', '.join(artifacts)}")
        
        return "\n".join(details)
    
    async def _update_knowledge_from_completion(self, completion_result: Dict[str, Any]):
        """Update knowledge base documents from work order completion."""
        
        # Extract any new assumptions or technical decisions
        context_updates = completion_result.get("context_updates", [])
        if context_updates:
            await self.knowledge_base._update_context_from_completion(completion_result)
        
        # Update architecture document if technical decisions were made
        if "architecture" in completion_result.get("lessons_learned", []):
            # Could update ARCHITECTURE.md with new details
            pass
        
        self.logger.info("Knowledge base updated from work order completion")
    
    async def get_work_order_status(self) -> Dict[str, Any]:
        """Get current status of all work orders."""
        return await self.knowledge_base.get_knowledge_summary()