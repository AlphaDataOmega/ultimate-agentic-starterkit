"""
Knowledge Base Management for the Ultimate Agentic StarterKit.

This module manages the multi-document project knowledge base that builds up
progressively as work orders are completed.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime
from enum import Enum

from core.models import ProjectTask, AgentResult
from core.logger import get_logger
from core.config import get_config


class ProjectType(str, Enum):
    """Types of projects that can be created."""
    GAME = "game"
    WEB_APP = "web_app"
    CLI_TOOL = "cli_tool"
    API_SERVICE = "api_service"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    ML_PROJECT = "ml_project"
    GENERIC = "generic"


class DocumentType(str, Enum):
    """Types of project documents in the knowledge base."""
    OVERVIEW = "OVERVIEW.md"
    CONTEXT = "CONTEXT.md"
    REQUIREMENTS = "REQUIREMENTS.md"
    ARCHITECTURE = "ARCHITECTURE.md"
    USER_STORIES = "USER_STORIES.md"
    DATA_MODELS = "DATA_MODELS.md"
    API_SPEC = "API_SPEC.md"
    SECURITY = "SECURITY.md"
    DEPLOYMENT = "DEPLOYMENT.md"
    BUSINESS_RULES = "BUSINESS_RULES.md"
    WORK_ORDERS = "work_orders/"
    COMPLETIONS = "completions/"


class WorkOrderStatus(str, Enum):
    """Status of work orders."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ProjectKnowledgeBase:
    """
    Manages the comprehensive project knowledge base with multiple specialized documents
    and progressive work order tracking.
    """
    
    def __init__(self, project_root: str = ".", project_name: str = None):
        """Initialize the knowledge base."""
        self.project_root = Path(project_root)
        self.project_name = project_name
        self.logger = get_logger("knowledge_base")
        self.config = get_config()
        
        # Knowledge base structure - use workspace/[project_name]/docs if project_name provided
        if project_name:
            project_workspace = self.project_root / "workspace" / project_name
            self.docs_dir = project_workspace / "docs"
            self.work_orders_dir = self.project_root / "work_orders"
            self.completions_dir = self.project_root / "completions"
        else:
            # Fallback to root docs for backward compatibility
            self.docs_dir = self.project_root / "docs"
            self.work_orders_dir = self.project_root / "work_orders"
            self.completions_dir = self.project_root / "completions"
        
        # Create directories
        for dir_path in [self.docs_dir, self.work_orders_dir, self.completions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Document templates dictionary removed - using AI generation instead
        
        # Work order tracking
        self.work_order_index = self._load_work_order_index()
    
    async def detect_project_type(self, overview_content: str) -> ProjectType:
        """Detect project type from overview content."""
        content_lower = overview_content.lower()
        
        # Game indicators
        game_keywords = ['game', 'player', 'score', 'level', 'paddle', 'ball', 'pygame', 'unity', 'godot']
        if any(keyword in content_lower for keyword in game_keywords):
            return ProjectType.GAME
        
        # CLI tool indicators
        cli_keywords = ['cli', 'command', 'terminal', 'script', 'tool', 'argparse', 'click', 'typer']
        if any(keyword in content_lower for keyword in cli_keywords):
            return ProjectType.CLI_TOOL
        
        # API service indicators
        api_keywords = ['api', 'service', 'endpoint', 'microservice', 'rest', 'graphql', 'fastapi', 'flask']
        if any(keyword in content_lower for keyword in api_keywords):
            return ProjectType.API_SERVICE
        
        # Mobile app indicators
        mobile_keywords = ['mobile', 'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin']
        if any(keyword in content_lower for keyword in mobile_keywords):
            return ProjectType.MOBILE_APP
        
        # Desktop app indicators
        desktop_keywords = ['desktop', 'electron', 'tkinter', 'pyqt', 'wxpython', 'kivy']
        if any(keyword in content_lower for keyword in desktop_keywords):
            return ProjectType.DESKTOP_APP
        
        # ML project indicators
        ml_keywords = ['machine learning', 'ml', 'ai', 'neural network', 'tensorflow', 'pytorch', 'sklearn', 'model']
        if any(keyword in content_lower for keyword in ml_keywords):
            return ProjectType.ML_PROJECT
        
        # Web app indicators (more comprehensive check)
        web_keywords = ['web', 'frontend', 'backend', 'database', 'react', 'vue', 'angular', 'django', 'express']
        if any(keyword in content_lower for keyword in web_keywords):
            return ProjectType.WEB_APP
        
        # Default to web app if uncertain
        return ProjectType.WEB_APP
    
    
    def _load_work_order_index(self) -> Dict[str, Any]:
        """Load work order tracking index."""
        index_file = self.work_orders_dir / "index.json"
        if index_file.exists():
            return json.loads(index_file.read_text())
        return {
            "next_id": 1,
            "work_orders": {},
            "completion_history": []
        }
    
    def _save_work_order_index(self):
        """Save work order tracking index."""
        index_file = self.work_orders_dir / "index.json"
        index_file.write_text(json.dumps(self.work_order_index, indent=2))
    
    async def determine_needed_documents(self, overview_content: str, project_type: ProjectType) -> List[DocumentType]:
        """Determine which documents are needed based on project type."""
        needed_docs = [DocumentType.CONTEXT]  # Always need context
        
        # Core documents for all projects
        core_docs = [
            DocumentType.REQUIREMENTS,
            DocumentType.ARCHITECTURE
        ]
        needed_docs.extend(core_docs)
        
        # Project-type-specific document selection
        if project_type == ProjectType.GAME:
            # Games: Simple structure, no complex business logic
            needed_docs.append(DocumentType.USER_STORIES)
            # Skip: API_SPEC, DATA_MODELS, SECURITY, BUSINESS_RULES
            # Only add deployment if specifically mentioned
            if any(keyword in overview_content.lower() for keyword in ['deploy', 'hosting', 'distribution']):
                needed_docs.append(DocumentType.DEPLOYMENT)
        
        elif project_type == ProjectType.CLI_TOOL:
            # CLI tools: Focus on commands and configuration
            needed_docs.append(DocumentType.USER_STORIES)
            # Add deployment for distribution
            needed_docs.append(DocumentType.DEPLOYMENT)
            # Add security only if auth/permissions mentioned
            if any(keyword in overview_content.lower() for keyword in ['auth', 'permission', 'secure']):
                needed_docs.append(DocumentType.SECURITY)
        
        elif project_type == ProjectType.WEB_APP:
            # Web apps: Comprehensive documentation
            needed_docs.extend([
                DocumentType.USER_STORIES,
                DocumentType.DATA_MODELS,
                DocumentType.API_SPEC,
                DocumentType.SECURITY,
                DocumentType.DEPLOYMENT,
                DocumentType.BUSINESS_RULES
            ])
        
        elif project_type == ProjectType.API_SERVICE:
            # API services: Focus on API spec and security
            needed_docs.extend([
                DocumentType.API_SPEC,
                DocumentType.DATA_MODELS,
                DocumentType.SECURITY,
                DocumentType.DEPLOYMENT,
                DocumentType.BUSINESS_RULES
            ])
        
        elif project_type == ProjectType.MOBILE_APP:
            # Mobile apps: Similar to web apps but different deployment
            needed_docs.extend([
                DocumentType.USER_STORIES,
                DocumentType.DATA_MODELS,
                DocumentType.SECURITY,
                DocumentType.DEPLOYMENT
            ])
            # Add API spec if backend integration
            if any(keyword in overview_content.lower() for keyword in ['api', 'backend', 'server']):
                needed_docs.append(DocumentType.API_SPEC)
        
        elif project_type == ProjectType.DESKTOP_APP:
            # Desktop apps: Similar to mobile but different deployment
            needed_docs.extend([
                DocumentType.USER_STORIES,
                DocumentType.DEPLOYMENT
            ])
            # Add data models if local storage
            if any(keyword in overview_content.lower() for keyword in ['database', 'data', 'storage']):
                needed_docs.append(DocumentType.DATA_MODELS)
        
        elif project_type == ProjectType.ML_PROJECT:
            # ML projects: Focus on data and models
            needed_docs.extend([
                DocumentType.DATA_MODELS,
                DocumentType.DEPLOYMENT
            ])
            # Add API spec if serving model
            if any(keyword in overview_content.lower() for keyword in ['api', 'serve', 'endpoint']):
                needed_docs.append(DocumentType.API_SPEC)
        
        else:  # GENERIC or unknown
            # Default to web app approach
            needed_docs.extend([
                DocumentType.USER_STORIES,
                DocumentType.DATA_MODELS,
                DocumentType.SECURITY,
                DocumentType.DEPLOYMENT
            ])
        
        self.logger.info(f"Project type: {project_type}, needed documents: {[doc.value for doc in needed_docs]}")
        return needed_docs
    
    async def initialize_document(self, doc_type: DocumentType, project_type: ProjectType = ProjectType.WEB_APP, project_context: Dict[str, Any] = None) -> Path:
        """Initialize a document with AI-generated content based on project context."""
        doc_path = self.docs_dir / doc_type.value
        
        if not doc_path.exists():
            # Generate document content using AI
            document_content = await self._ai_generate_document_content(doc_type, project_type, project_context or {})
            
            if document_content:
                doc_path.write_text(document_content)
                self.logger.info(f"Initialized AI-generated {project_type} document: {doc_path}")
            else:
                # Fallback to basic template
                template_content = self._get_project_specific_template(doc_type, project_type)
                if template_content:
                    doc_path.write_text(template_content)
                    self.logger.info(f"Initialized template-based {project_type} document: {doc_path}")
                else:
                    raise ValueError(f"No template available for document type: {doc_type} with project type: {project_type}")
        
        return doc_path
    
    async def _ai_generate_document_content(self, doc_type: DocumentType, project_type: ProjectType, project_context: Dict[str, Any]) -> str:
        """Generate document content using AI based on project context."""
        
        try:
            from integrations.ollama_client import OllamaClient
            
            # Build comprehensive context for AI generation
            context_summary = self._build_generation_context(project_context)
            
            # Create generation prompt  
            generation_prompt = f"""
Generate a comprehensive {doc_type.value} document specifically tailored for this {project_type} project.

Project Context:
{context_summary[:2000]}  # Truncate context to keep prompt manageable

Requirements:
1. Be specific to this project - Use the actual project details provided
2. Match project complexity - Simple projects get simple docs, complex projects get detailed docs
3. Include relevant sections only - Don't add unnecessary sections for this project type
4. Use markdown format - Well-structured with headers, lists, and code blocks where appropriate
5. Be actionable - Include specific, implementable details

Document Purpose:
- CONTEXT.md: Working assumptions and project context
- REQUIREMENTS.md: Functional and technical requirements
- ARCHITECTURE.md: System design and technical decisions
- USER_STORIES.md: User stories and acceptance criteria

Response Format:
Provide only the markdown content for the document, no additional text or explanations.
"""
            
            client = OllamaClient()
            result = await client.generate_response(
                model="llama3.2:latest",  # Use a default model
                prompt=generation_prompt
            )
            
            if result.get("success", False):
                document_content = result.get("response", "")
            else:
                document_content = ""
            
            if document_content:
                self.logger.info(f"Generated AI document content for {doc_type.value}")
                return document_content
            else:
                self.logger.warning(f"AI generation produced empty content for {doc_type.value}")
                return ""
        
        except Exception as e:
            self.logger.error(f"AI document generation error: {e}")
            return ""
    
    def _build_generation_context(self, project_context: Dict[str, Any]) -> str:
        """Build comprehensive context for AI document generation."""
        context_parts = []
        
        # Add project metadata
        project_type = project_context.get("project_type", "unknown")
        complexity = project_context.get("complexity", "medium")
        features = project_context.get("features", [])
        tech_stack = project_context.get("tech_stack", [])
        
        context_parts.append(f"**Project Metadata**:")
        context_parts.append(f"- Type: {project_type}")
        context_parts.append(f"- Complexity: {complexity}")
        if features:
            context_parts.append(f"- Features: {', '.join(features)}")
        if tech_stack:
            context_parts.append(f"- Technology Stack: {', '.join(tech_stack)}")
        
        # Add project overview
        overview = project_context.get("overview_content", "")
        if overview:
            context_parts.append(f"**Project Overview**:\n{overview}")
        
        # Add refined scope if available
        refined_scope = project_context.get("refined_scope", "")
        if refined_scope and refined_scope != overview:
            context_parts.append(f"**Refined Scope**:\n{refined_scope}")
        
        # Add user responses/clarifications
        user_responses = project_context.get("user_responses", {})
        if user_responses:
            context_parts.append("**User Clarifications**:")
            for key, value in user_responses.items():
                context_parts.append(f"- {key}: {value}")
        
        # Add project-specific requirements
        requirements = project_context.get("requirements", [])
        if requirements:
            context_parts.append("**Specific Requirements**:")
            for req in requirements:
                context_parts.append(f"- {req}")
        
        # Add architectural decisions
        architecture = project_context.get("architecture", {})
        if architecture:
            context_parts.append("**Architecture Decisions**:")
            for key, value in architecture.items():
                context_parts.append(f"- {key}: {value}")
        
        # Add game-specific details for game projects
        if project_type.lower() == "game":
            game_details = self._extract_game_specific_details(project_context)
            if game_details:
                context_parts.append("**Game-Specific Details**:")
                context_parts.extend(game_details)
        
        # Add web app specific details for web projects
        if project_type.lower() in ["web_app", "webapp"]:
            web_details = self._extract_web_specific_details(project_context)
            if web_details:
                context_parts.append("**Web Application Details**:")
                context_parts.extend(web_details)
        
        # Add API specific details for API projects
        if project_type.lower() in ["api", "api_service"]:
            api_details = self._extract_api_specific_details(project_context)
            if api_details:
                context_parts.append("**API Service Details**:")
                context_parts.extend(api_details)
        
        # Add existing documents for context
        existing_docs = project_context.get("documents", {})
        if existing_docs:
            context_parts.append("**Existing Documents**:")
            for doc_name, doc_content in existing_docs.items():
                if doc_content:
                    # Include summary of existing docs
                    summary = doc_content[:200] + "..." if len(doc_content) > 200 else doc_content
                    context_parts.append(f"- {doc_name}: {summary}")
        
        # Add completion history for context
        completion_history = project_context.get("completion_history", [])
        if completion_history:
            context_parts.append("**Implementation Progress**:")
            for completion in completion_history[-3:]:  # Last 3 completions
                context_parts.append(f"- {completion.get('title', 'Unknown')}: {completion.get('status', 'completed')}")
        
        return "\n\n".join(context_parts) if context_parts else "No additional context available"
    
    def _extract_game_specific_details(self, project_context: Dict[str, Any]) -> List[str]:
        """Extract game-specific details from project context."""
        details = []
        
        # Extract game genre and mechanics
        overview = project_context.get("overview_content", "").lower()
        if "pong" in overview:
            details.append("- Game Genre: Classic Arcade")
            details.append("- Game Mechanics: Paddle movement, ball physics, collision detection")
            details.append("- Player Controls: Keyboard input for paddle movement")
            details.append("- Game Objectives: First to reach score limit wins")
        elif "puzzle" in overview:
            details.append("- Game Genre: Puzzle")
            details.append("- Game Mechanics: Logic-based challenges, problem solving")
        elif "rpg" in overview:
            details.append("- Game Genre: Role-Playing Game")
            details.append("- Game Mechanics: Character progression, inventory, quest system")
        elif "strategy" in overview:
            details.append("- Game Genre: Strategy")
            details.append("- Game Mechanics: Resource management, tactical decisions")
        
        # Extract rendering and graphics details
        if "pygame" in str(project_context.get("tech_stack", [])).lower():
            details.append("- Graphics: Pygame-based 2D rendering")
            details.append("- Display: Real-time game loop with frame updates")
        
        # Extract gameplay features
        if "multiplayer" in overview:
            details.append("- Multiplayer: Multiple player support")
        if "ai" in overview:
            details.append("- AI: Computer-controlled opponents")
        if "score" in overview:
            details.append("- Scoring System: Points-based gameplay")
        
        return details
    
    def _extract_web_specific_details(self, project_context: Dict[str, Any]) -> List[str]:
        """Extract web application specific details from project context."""
        details = []
        
        overview = project_context.get("overview_content", "").lower()
        tech_stack = project_context.get("tech_stack", [])
        
        # Extract frontend framework
        if "react" in str(tech_stack).lower() or "react" in overview:
            details.append("- Frontend Framework: React.js")
        elif "vue" in str(tech_stack).lower() or "vue" in overview:
            details.append("- Frontend Framework: Vue.js")
        elif "angular" in str(tech_stack).lower() or "angular" in overview:
            details.append("- Frontend Framework: Angular")
        
        # Extract backend framework
        if "flask" in str(tech_stack).lower() or "flask" in overview:
            details.append("- Backend Framework: Flask")
        elif "django" in str(tech_stack).lower() or "django" in overview:
            details.append("- Backend Framework: Django")
        elif "fastapi" in str(tech_stack).lower() or "fastapi" in overview:
            details.append("- Backend Framework: FastAPI")
        
        # Extract database information
        if "postgresql" in overview or "postgres" in overview:
            details.append("- Database: PostgreSQL")
        elif "mysql" in overview:
            details.append("- Database: MySQL")
        elif "mongodb" in overview:
            details.append("- Database: MongoDB")
        elif "sqlite" in overview:
            details.append("- Database: SQLite")
        
        # Extract deployment and hosting details
        if "docker" in overview:
            details.append("- Deployment: Docker containerization")
        if "aws" in overview:
            details.append("- Hosting: AWS")
        elif "heroku" in overview:
            details.append("- Hosting: Heroku")
        elif "vercel" in overview:
            details.append("- Hosting: Vercel")
        
        # Extract authentication details
        if "auth" in overview or "login" in overview:
            details.append("- Authentication: User login/registration system")
        if "jwt" in overview:
            details.append("- Auth Method: JWT tokens")
        if "oauth" in overview:
            details.append("- Auth Method: OAuth integration")
        
        return details
    
    def _extract_api_specific_details(self, project_context: Dict[str, Any]) -> List[str]:
        """Extract API service specific details from project context."""
        details = []
        
        overview = project_context.get("overview_content", "").lower()
        tech_stack = project_context.get("tech_stack", [])
        
        # Extract API framework
        if "fastapi" in str(tech_stack).lower() or "fastapi" in overview:
            details.append("- API Framework: FastAPI")
        elif "flask" in str(tech_stack).lower() or "flask" in overview:
            details.append("- API Framework: Flask")
        elif "django" in str(tech_stack).lower() or "django" in overview:
            details.append("- API Framework: Django REST Framework")
        elif "express" in str(tech_stack).lower() or "express" in overview:
            details.append("- API Framework: Express.js")
        
        # Extract API type
        if "rest" in overview:
            details.append("- API Type: RESTful API")
        elif "graphql" in overview:
            details.append("- API Type: GraphQL API")
        
        # Extract authentication
        if "api key" in overview:
            details.append("- Authentication: API Key")
        elif "jwt" in overview:
            details.append("- Authentication: JWT tokens")
        elif "oauth" in overview:
            details.append("- Authentication: OAuth 2.0")
        
        # Extract data formats
        if "json" in overview:
            details.append("- Data Format: JSON")
        elif "xml" in overview:
            details.append("- Data Format: XML")
        
        # Extract rate limiting
        if "rate limit" in overview:
            details.append("- Rate Limiting: Request throttling implemented")
        
        return details
    
    def _get_project_specific_template(self, doc_type: DocumentType, project_type: ProjectType) -> str:
        """Get project-type-specific template content."""
        # Basic fallback templates when AI generation fails
        
        if doc_type == DocumentType.CONTEXT:
            return f"""# Project Context

## Working Assumptions
- This is a {project_type} project
- Standard development practices apply
- Testing will be implemented
- Documentation will be maintained

## Technical Environment
- Development environment: Local development
- Target environment: Production ready
- Dependencies: As specified in requirements

## Project Scope
- Primary focus: Core functionality
- Secondary focus: Testing and documentation
- Future enhancements: To be determined

## Success Criteria
- All requirements implemented
- Tests passing
- Documentation complete
"""
        
        elif doc_type == DocumentType.REQUIREMENTS:
            return f"""# Requirements

## Functional Requirements
- Core features as specified in project overview
- User interface requirements
- Performance requirements

## Non-Functional Requirements
- Reliability and stability
- Performance optimization
- Code quality standards

## Technical Requirements
- {project_type} specific implementation
- Testing coverage requirements
- Documentation standards
"""
        
        elif doc_type == DocumentType.ARCHITECTURE:
            return f"""# Architecture

## System Overview
- {project_type} architecture
- Component organization
- Technology stack

## Technical Decisions
- Framework selection
- Architecture patterns
- Implementation approach

## Development Guidelines
- Code organization
- Testing strategy
- Deployment approach
"""
        
        elif doc_type == DocumentType.USER_STORIES:
            return f"""# User Stories

## Primary User Stories
- As a user, I want to use the {project_type} functionality
- As a user, I want the system to be reliable
- As a user, I want clear documentation

## Acceptance Criteria
- All features work as expected
- Performance meets requirements
- Documentation is complete
"""
        
        else:
            return f"""# {doc_type.value.replace('.md', '')}

## Overview
This is a {project_type} project document.

## Details
To be filled in based on project requirements.
"""
    
    async def get_project_context(self, work_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive project context for work order execution."""
        context = {
            "documents": {},
            "work_orders": {
                "completed": [],
                "pending": [],
                "current": work_order_id
            },
            "assumptions": [],
            "technical_decisions": [],
            "completion_history": self.work_order_index.get("completion_history", [])
        }
        
        # Load all document content
        for doc_type in DocumentType:
            if doc_type.value.endswith('.md'):
                doc_path = self.docs_dir / doc_type.value
                if doc_path.exists():
                    context["documents"][doc_type.value] = doc_path.read_text()
        
        # Load work order status
        for wo_id, wo_data in self.work_order_index.get("work_orders", {}).items():
            if wo_data["status"] == WorkOrderStatus.COMPLETED:
                context["work_orders"]["completed"].append(wo_data)
            elif wo_data["status"] == WorkOrderStatus.PENDING:
                context["work_orders"]["pending"].append(wo_data)
        
        # Extract assumptions from CONTEXT.md
        context_doc = context["documents"].get("CONTEXT.md", "")
        assumptions = self._extract_assumptions(context_doc)
        context["assumptions"] = assumptions
        
        return context
    
    def _extract_assumptions(self, context_content: str) -> List[str]:
        """Extract checked assumptions from CONTEXT.md."""
        assumptions = []
        lines = context_content.split('\n')
        
        for line in lines:
            if '- [x]' in line:
                assumption = line.replace('- [x]', '').strip()
                if assumption:
                    assumptions.append(assumption)
        
        return assumptions
    
    async def create_work_order(self, title: str, description: str, 
                              dependencies: List[str] = None) -> str:
        """Create a new work order."""
        work_order_id = f"WO-{self.work_order_index['next_id']:04d}"
        self.work_order_index['next_id'] += 1
        
        work_order = {
            "id": work_order_id,
            "title": title,
            "description": description,
            "status": WorkOrderStatus.PENDING,
            "dependencies": dependencies or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "context_snapshot": await self.get_project_context(),
            "completion_data": None
        }
        
        self.work_order_index["work_orders"][work_order_id] = work_order
        self._save_work_order_index()
        
        # Save work order file
        wo_file = self.work_orders_dir / f"{work_order_id}.json"
        wo_file.write_text(json.dumps(work_order, indent=2))
        
        self.logger.info(f"Created work order: {work_order_id}")
        return work_order_id
    
    async def complete_work_order(self, work_order_id: str, 
                                completion_data: Dict[str, Any]):
        """Mark work order as complete and update knowledge base."""
        if work_order_id not in self.work_order_index["work_orders"]:
            raise ValueError(f"Work order not found: {work_order_id}")
        
        work_order = self.work_order_index["work_orders"][work_order_id]
        work_order["status"] = WorkOrderStatus.COMPLETED
        work_order["updated_at"] = datetime.now().isoformat()
        work_order["completion_data"] = completion_data
        
        # Save completion document
        completion_file = self.completions_dir / f"{work_order_id}_completion.md"
        completion_content = self._format_completion_document(work_order, completion_data)
        completion_file.write_text(completion_content)
        
        # Update completion history
        self.work_order_index["completion_history"].append({
            "work_order_id": work_order_id,
            "completed_at": datetime.now().isoformat(),
            "title": work_order["title"],
            "artifacts": completion_data.get("artifacts", []),
            "lessons_learned": completion_data.get("lessons_learned", []),
            "context_updates": completion_data.get("context_updates", [])
        })
        
        self._save_work_order_index()
        
        # Update context document with new assumptions/learnings
        await self._update_context_from_completion(completion_data)
        
        self.logger.info(f"Completed work order: {work_order_id}")
    
    def _format_completion_document(self, work_order: Dict[str, Any], 
                                  completion_data: Dict[str, Any]) -> str:
        """Format work order completion document."""
        return f"""# Work Order Completion: {work_order['id']}

## Work Order Details
- **Title**: {work_order['title']}
- **Created**: {work_order['created_at']}
- **Completed**: {work_order['updated_at']}
- **Status**: {work_order['status']}

## Description
{work_order['description']}

## Completion Results

### Artifacts Created
{chr(10).join(f"- {artifact}" for artifact in completion_data.get('artifacts', []))}

### Code Changes
{completion_data.get('code_changes', 'No code changes documented')}

### Tests Added
{completion_data.get('tests_added', 'No tests documented')}

### Documentation Updates
{completion_data.get('documentation_updates', 'No documentation updates')}

## Lessons Learned
{chr(10).join(f"- {lesson}" for lesson in completion_data.get('lessons_learned', []))}

## Context Updates
{chr(10).join(f"- {update}" for update in completion_data.get('context_updates', []))}

## Next Steps
{completion_data.get('next_steps', 'No next steps identified')}

---
*Generated automatically on work order completion*
"""
    
    async def _update_context_from_completion(self, completion_data: Dict[str, Any]):
        """Update CONTEXT.md with learnings from work order completion."""
        context_path = self.docs_dir / "CONTEXT.md"
        if not context_path.exists():
            await self.initialize_document(DocumentType.CONTEXT)
        
        context_content = context_path.read_text()
        
        # Add new assumptions if any
        new_assumptions = completion_data.get('context_updates', [])
        if new_assumptions:
            assumptions_section = "\n### Validated Assumptions from Development\n"
            for assumption in new_assumptions:
                assumptions_section += f"- [x] {assumption}\n"
            
            context_content += f"\n{assumptions_section}"
            context_path.write_text(context_content)
            self.logger.info(f"Updated CONTEXT.md with {len(new_assumptions)} new assumptions")
    
    async def get_next_work_order(self) -> Optional[str]:
        """Get the next work order ready for execution."""
        for wo_id, work_order in self.work_order_index["work_orders"].items():
            if work_order["status"] == WorkOrderStatus.PENDING:
                # Check if dependencies are completed
                dependencies_met = all(
                    self.work_order_index["work_orders"][dep_id]["status"] == WorkOrderStatus.COMPLETED
                    for dep_id in work_order["dependencies"]
                    if dep_id in self.work_order_index["work_orders"]
                )
                
                if dependencies_met:
                    return wo_id
        
        return None
    
    async def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of current knowledge base state."""
        documents = []
        for doc_type in DocumentType:
            if doc_type.value.endswith('.md'):
                doc_path = self.docs_dir / doc_type.value
                if doc_path.exists():
                    documents.append({
                        "type": doc_type.value,
                        "exists": True,
                        "size": len(doc_path.read_text()),
                        "last_modified": datetime.fromtimestamp(doc_path.stat().st_mtime).isoformat()
                    })
                else:
                    documents.append({
                        "type": doc_type.value,
                        "exists": False
                    })
        
        work_orders = self.work_order_index["work_orders"]
        wo_summary = {
            "total": len(work_orders),
            "completed": len([wo for wo in work_orders.values() if wo["status"] == WorkOrderStatus.COMPLETED]),
            "pending": len([wo for wo in work_orders.values() if wo["status"] == WorkOrderStatus.PENDING]),
            "in_progress": len([wo for wo in work_orders.values() if wo["status"] == WorkOrderStatus.IN_PROGRESS])
        }
        
        return {
            "documents": documents,
            "work_orders": wo_summary,
            "completion_history": len(self.work_order_index["completion_history"]),
            "project_root": str(self.project_root)
        }
    
    async def update_from_completion(self, completion_result: Dict[str, Any]):
        """Update knowledge base from work order completion."""
        # This method is called by the orchestrator after each work order completion
        # to update the knowledge base with new learnings
        
        # Extract any new assumptions or technical decisions
        context_updates = completion_result.get("context_updates", [])
        if context_updates:
            await self._update_context_from_completion(completion_result)
        
        # Update architecture document if technical decisions were made
        if "architecture" in completion_result.get("lessons_learned", []):
            # Could update ARCHITECTURE.md with new details
            pass
        
        self.logger.info("Knowledge base updated from work order completion")
    
    # AI-Enhanced Features (Future Automation Opportunities)
    # Templates are now only used as fallbacks - AI generates most content
    
    
    async def _ai_code_review(self, code_content: str) -> Dict[str, Any]:
        """AI-powered code review for quality, security, and performance."""
        from integrations.ollama_client import OllamaClient
        
        try:
            client = OllamaClient()
            
            # Limit code content to prevent overwhelming the AI
            if len(code_content) > 5000:
                code_content = code_content[:5000] + "\n... [truncated for review]"
            
            prompt = f"""
            Review the following code for quality, security, and performance issues:
            
            ```
            {code_content}
            ```
            
            Please provide feedback on:
            1. Code quality and best practices
            2. Security vulnerabilities
            3. Performance improvements
            4. Maintainability concerns
            
            Format your response as constructive feedback with specific suggestions.
            """
            
            response = await client.generate_response(
                model="llama3.2:latest",
                prompt=prompt
            )
            
            if response.get("success", False):
                return {
                    "review_result": response.get("response", ""),
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "review_result": "Code review failed - AI service unavailable",
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            self.logger.error(f"AI code review failed: {e}")
            return {
                "review_result": "Code review failed due to technical error",
                "success": False,
                "error": str(e)
            }
    
    async def _ai_deployment_strategy(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-driven deployment strategy based on project requirements."""
        from integrations.ollama_client import OllamaClient
        
        try:
            client = OllamaClient()
            
            project_type = project_context.get('project_type', 'unknown')
            features = project_context.get('features', [])
            tech_stack = project_context.get('tech_stack', [])
            complexity = project_context.get('complexity', 'medium')
            
            prompt = f"""
            Generate a deployment strategy for a {project_type} project with these characteristics:
            
            Project Type: {project_type}
            Complexity: {complexity}
            Features: {', '.join(features) if features else 'Standard features'}
            Technology Stack: {', '.join(tech_stack) if tech_stack else 'Standard stack'}
            
            Please provide a comprehensive deployment strategy including:
            
            1. **Deployment Architecture**:
               - Recommended hosting platform
               - Infrastructure requirements
               - Scalability considerations
            
            2. **Environment Configuration**:
               - Development environment setup
               - Staging environment configuration
               - Production environment requirements
            
            3. **CI/CD Pipeline**:
               - Build process automation
               - Testing integration
               - Deployment automation
            
            4. **Monitoring & Maintenance**:
               - Performance monitoring
               - Error tracking
               - Update strategies
            
            Focus on practical, implementable recommendations for this project type.
            """
            
            response = await client.generate_response(
                model="llama3.2:latest",
                prompt=prompt
            )
            
            if response.get("success", False):
                return {
                    "deployment_strategy": response.get("response", ""),
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "deployment_strategy": "Standard deployment strategy recommended",
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            self.logger.error(f"AI deployment strategy failed: {e}")
            return {
                "deployment_strategy": "Standard deployment strategy recommended",
                "success": False,
                "error": str(e)
            }
    
    async def _ai_error_recovery(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-based error recovery and self-healing suggestions."""
        from integrations.ollama_client import OllamaClient
        
        try:
            client = OllamaClient()
            
            error_type = error_context.get('error_type', 'unknown')
            error_message = error_context.get('error_message', '')
            stack_trace = error_context.get('stack_trace', '')
            previous_failures = error_context.get('previous_failures', [])
            
            # Truncate long error messages
            if len(error_message) > 1000:
                error_message = error_message[:1000] + "... [truncated]"
            if len(stack_trace) > 2000:
                stack_trace = stack_trace[:2000] + "... [truncated]"
            
            prompt = f"""
            Analyze the following error and suggest recovery strategies:
            
            Error Type: {error_type}
            Error Message: {error_message}
            Stack Trace: {stack_trace}
            
            Previous Failures: {len(previous_failures)} similar errors
            
            Please provide:
            1. **Root Cause Analysis**: What likely caused this error?
            2. **Immediate Recovery**: Quick fixes to restore functionality
            3. **Long-term Solutions**: Prevent similar errors in the future
            4. **Monitoring**: What to watch for to detect early warnings
            
            Focus on actionable, specific recommendations.
            """
            
            response = await client.generate_response(
                model="llama3.2:latest",
                prompt=prompt
            )
            
            if response.get("success", False):
                return {
                    "recovery_strategy": response.get("response", ""),
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "recovery_strategy": "Standard error recovery procedures recommended",
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            self.logger.error(f"AI error recovery failed: {e}")
            return {
                "recovery_strategy": "Standard error recovery procedures recommended",
                "success": False,
                "error": str(e)
            }
    
    async def _ai_architecture_advisor(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """AI-driven architecture recommendations based on requirements."""
        from integrations.ollama_client import OllamaClient
        
        try:
            client = OllamaClient()
            
            project_type = requirements.get('project_type', 'unknown')
            features = requirements.get('features', [])
            scale = requirements.get('scale', 'small')
            performance_needs = requirements.get('performance_needs', 'standard')
            budget = requirements.get('budget', 'medium')
            
            prompt = f"""
            Provide architecture recommendations for a {project_type} project with these requirements:
            
            Project Type: {project_type}
            Scale: {scale}
            Performance Needs: {performance_needs}
            Budget: {budget}
            Required Features: {', '.join(features) if features else 'Standard features'}
            
            Please recommend:
            
            1. **Architecture Pattern**:
               - Monolithic vs Microservices
               - Recommended design patterns
               - Scalability approach
            
            2. **Technology Stack**:
               - Backend framework and language
               - Database choices
               - Frontend technology (if applicable)
               - Infrastructure components
            
            3. **Data Architecture**:
               - Database design approach
               - Caching strategy
               - Data flow patterns
            
            4. **Integration Strategy**:
               - API design approach
               - Third-party integrations
               - Communication patterns
            
            Focus on practical, cost-effective recommendations that match the project scale.
            """
            
            response = await client.generate_response(
                model="llama3.2:latest",
                prompt=prompt
            )
            
            if response.get("success", False):
                return {
                    "architecture_recommendations": response.get("response", ""),
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "architecture_recommendations": "Standard architecture patterns recommended",
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            self.logger.error(f"AI architecture advisor failed: {e}")
            return {
                "architecture_recommendations": "Standard architecture patterns recommended",
                "success": False,
                "error": str(e)
            }
    
    async def ai_generate_test_strategy(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI-driven test strategy based on project type and complexity.
        
        Args:
            project_context: Context about the project including type, features, etc.
            
        Returns:
            Dict containing test strategy recommendations
        """
        from integrations.ollama_client import OllamaClient
        
        try:
            client = OllamaClient()
            
            # Get project information
            project_type = project_context.get('project_type', 'unknown')
            features = project_context.get('features', [])
            complexity = project_context.get('complexity', 'medium')
            tech_stack = project_context.get('tech_stack', [])
            
            # Create AI prompt for test strategy generation
            prompt = f"""
            Generate a comprehensive test strategy for a {project_type} project with the following characteristics:
            
            Project Type: {project_type}
            Complexity: {complexity}
            Features: {', '.join(features) if features else 'Standard features'}
            Technology Stack: {', '.join(tech_stack) if tech_stack else 'Standard stack'}
            
            Please provide a detailed test strategy that includes:
            
            1. **Test Framework Recommendations**:
               - Unit testing framework
               - Integration testing approach
               - End-to-end testing tools
               - Performance testing strategy
            
            2. **Test Types by Priority**:
               - Unit tests (what to test, coverage targets)
               - Integration tests (API, database, external services)
               - End-to-end tests (user workflows)
               - Performance tests (load, stress, scalability)
               - Security tests (if applicable)
            
            3. **Testing Configuration**:
               - Test environment setup
               - CI/CD integration
               - Test data management
               - Mock/stub strategies
            
            4. **Test Automation Strategy**:
               - Automated test execution
               - Test reporting
               - Continuous testing approach
               - Quality gates
            
            5. **Project-Specific Considerations**:
               - Special testing needs for this project type
               - Risk areas that need extra testing
               - Performance benchmarks
               - User acceptance criteria
            
            Format the response as a structured markdown document with clear sections.
            """
            
            # Generate test strategy using AI
            response = await client.generate_completion(prompt)
            
            # Parse the response and create strategy object
            strategy = {
                'test_framework': self._extract_test_framework(response, project_type),
                'test_types': self._extract_test_types(response, project_type),
                'automation_strategy': self._extract_automation_strategy(response),
                'configuration': self._extract_test_configuration(response),
                'project_specific': self._extract_project_specific_tests(response, project_type),
                'ai_generated_content': response,
                'generated_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"Generated AI test strategy for {project_type} project")
            return strategy
            
        except Exception as e:
            self.logger.error(f"Failed to generate AI test strategy: {str(e)}")
            return self._fallback_test_strategy(project_context)
    
    def _extract_test_framework(self, ai_response: str, project_type: str) -> Dict[str, str]:
        """Extract test framework recommendations from AI response."""
        # Basic framework recommendations based on project type
        frameworks = {
            'GAME': {
                'unit': 'pytest',
                'integration': 'pytest',
                'e2e': 'pygame testing',
                'performance': 'performance profiling'
            },
            'WEB_APP': {
                'unit': 'pytest/jest',
                'integration': 'pytest/supertest',
                'e2e': 'playwright/cypress',
                'performance': 'k6/locust'
            },
            'CLI_TOOL': {
                'unit': 'pytest',
                'integration': 'pytest',
                'e2e': 'subprocess testing',
                'performance': 'benchmark testing'
            }
        }
        
        return frameworks.get(project_type, frameworks['WEB_APP'])
    
    def _extract_test_types(self, ai_response: str, project_type: str) -> List[Dict[str, Any]]:
        """Extract test types and priorities from AI response."""
        base_tests = [
            {'type': 'unit', 'priority': 'high', 'coverage_target': '80%'},
            {'type': 'integration', 'priority': 'medium', 'coverage_target': '60%'},
            {'type': 'e2e', 'priority': 'medium', 'coverage_target': '40%'}
        ]
        
        if project_type == 'WEB_APP':
            base_tests.extend([
                {'type': 'security', 'priority': 'high', 'coverage_target': '100%'},
                {'type': 'performance', 'priority': 'medium', 'coverage_target': '100%'}
            ])
        elif project_type == 'GAME':
            base_tests.extend([
                {'type': 'performance', 'priority': 'high', 'coverage_target': '100%'},
                {'type': 'usability', 'priority': 'medium', 'coverage_target': '80%'}
            ])
        
        return base_tests
    
    def _extract_automation_strategy(self, ai_response: str) -> Dict[str, Any]:
        """Extract automation strategy from AI response."""
        return {
            'ci_integration': True,
            'automated_execution': True,
            'test_reporting': 'junit/html',
            'quality_gates': ['unit_tests_pass', 'coverage_threshold', 'integration_tests_pass'],
            'continuous_testing': True
        }
    
    def _extract_test_configuration(self, ai_response: str) -> Dict[str, Any]:
        """Extract test configuration from AI response."""
        return {
            'test_environment': 'isolated',
            'test_data_strategy': 'fixtures_and_mocks',
            'mock_strategy': 'external_services',
            'parallel_execution': True,
            'test_isolation': True
        }
    
    def _extract_project_specific_tests(self, ai_response: str, project_type: str) -> List[str]:
        """Extract project-specific test considerations."""
        specific_tests = {
            'GAME': [
                'Performance testing for frame rate',
                'Input response time testing',
                'Memory leak detection',
                'Cross-platform compatibility'
            ],
            'WEB_APP': [
                'API security testing',
                'Database transaction testing',
                'Authentication flow testing',
                'Browser compatibility testing'
            ],
            'CLI_TOOL': [
                'Command-line argument testing',
                'Exit code validation',
                'Error message testing',
                'Platform-specific behavior'
            ]
        }
        
        return specific_tests.get(project_type, [])
    
    def _fallback_test_strategy(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback test strategy if AI generation fails."""
        project_type = project_context.get('project_type', 'WEB_APP')
        
        return {
            'test_framework': self._extract_test_framework('', project_type),
            'test_types': self._extract_test_types('', project_type),
            'automation_strategy': self._extract_automation_strategy(''),
            'configuration': self._extract_test_configuration(''),
            'project_specific': self._extract_project_specific_tests('', project_type),
            'ai_generated_content': 'Fallback strategy used due to AI generation failure',
            'generated_at': datetime.now().isoformat()
        }