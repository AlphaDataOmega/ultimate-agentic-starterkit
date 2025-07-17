"""
Enhanced Project Manager Agent for Multi-Document Knowledge Base.

This agent manages the complete project lifecycle from OVERVIEW.md through
multi-document specification generation and work order creation.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from agents.base_agent import BaseAgent
from core.models import AgentResult, ProjectTask, TaskStatus, AgentType
from core.logger import get_logger
from core.knowledge_base import ProjectKnowledgeBase, DocumentType
from core.work_order_manager import WorkOrderManager


class EnhancedProjectManager(BaseAgent):
    """
    Enhanced Project Manager that creates comprehensive multi-document
    project specifications and manages the complete development workflow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Enhanced Project Manager."""
        super().__init__("enhanced_project_manager", config)
        
        self.knowledge_base = ProjectKnowledgeBase()
        self.work_order_manager = WorkOrderManager()
        
        # Document generation priorities
        self.document_priority = [
            DocumentType.CONTEXT,      # Working assumptions first
            DocumentType.REQUIREMENTS, # Detailed requirements
            DocumentType.ARCHITECTURE, # Technical decisions
            DocumentType.USER_STORIES,  # User perspectives
            DocumentType.DATA_MODELS,   # Data structure
            DocumentType.API_SPEC,      # API contracts
            DocumentType.SECURITY,      # Security requirements
            DocumentType.DEPLOYMENT,    # Deployment strategy
            DocumentType.BUSINESS_RULES # Domain logic
        ]
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute enhanced project management workflow.
        
        This creates the complete multi-document knowledge base and
        sets up work orders for project execution.
        """
        self.logger.info(f"Starting enhanced project management for: {task.title}")
        
        try:
            # Parse the OVERVIEW.md content
            overview_content = task.description
            
            # Phase 1: Interactive Learning - Generate clarifying questions
            questions = await self._generate_clarifying_questions(overview_content)
            self.logger.info(f"Generated {len(questions)} clarifying questions")
            
            # Phase 2: Present questions to user and gather responses
            user_responses = await self._present_questions_to_user(questions)
            self.logger.info(f"Received user responses for {len(user_responses)} questions")
            
            # Phase 3: Refine project scope based on user input
            refined_scope = await self._refine_project_scope(overview_content, user_responses)
            
            # Phase 4: Determine project type and needed documents
            project_type = await self.knowledge_base.detect_project_type(refined_scope)
            needed_docs = await self.knowledge_base.determine_needed_documents(refined_scope, project_type)
            self.logger.info(f"Project type: {project_type}, needed documents: {[doc.value for doc in needed_docs]}")
            
            # Phase 5: Initialize project-appropriate documents with AI-generated content
            initialized_docs = []
            for doc_type in needed_docs:
                # Build comprehensive context for document generation
                doc_context = {
                    "overview_content": overview_content,
                    "refined_scope": refined_scope,
                    "user_responses": user_responses,
                    "project_type": project_type,
                    "documents": {}  # Will be populated as documents are created
                }
                
                doc_path = await self.knowledge_base.initialize_document(doc_type, project_type, doc_context)
                initialized_docs.append(str(doc_path))
                
                # Add newly created document to context for next documents
                if doc_path.exists():
                    doc_context["documents"][doc_type.value] = doc_path.read_text()
            
            # Phase 6: Generate comprehensive project context
            project_context = await self._generate_comprehensive_context(refined_scope, needed_docs)
            
            # Phase 7: NOTE: Work order creation is now handled incrementally by WorkOrderManager
            # We only prepare the project for incremental execution
            execution_plan = await self._prepare_incremental_execution(project_context, project_type)
            
            result = AgentResult(
                agent_id=self.agent_id,
                success=True,
                output={
                    'interactive_learning_completed': True,
                    'questions_asked': len(questions),
                    'user_responses': user_responses,
                    'refined_scope': refined_scope,
                    'project_type': project_type,
                    'initialized_documents': initialized_docs,
                    'needed_documents': [doc.value for doc in needed_docs],
                    'project_context': project_context,
                    'execution_plan': execution_plan,
                    'ready_for_incremental_execution': True
                },
                confidence=0.9,
                execution_time=0.0
            )
            
            self.logger.info(f"Enhanced project management completed: {len(work_order_ids)} work orders created")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced project management failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                confidence=0.0,
                execution_time=0.0
            )
    
    async def _generate_comprehensive_context(self, overview_content: str, 
                                            needed_docs: List[DocumentType]) -> Dict[str, Any]:
        """Generate comprehensive project context and populate documents."""
        
        context = {
            "project_analysis": await self._analyze_project_overview(overview_content),
            "technical_decisions": [],
            "working_assumptions": [],
            "feature_breakdown": [],
            "integration_requirements": []
        }
        
        # Analyze project type and complexity
        project_analysis = context["project_analysis"]
        project_type = project_analysis.get("type", "generic")
        
        # Generate working assumptions for CONTEXT.md
        assumptions = await self._generate_working_assumptions(overview_content, project_type)
        context["working_assumptions"] = assumptions
        
        # Update CONTEXT.md with project-specific assumptions
        await self._update_context_document(assumptions, project_type)
        
        # Generate feature breakdown
        features = await self._extract_and_analyze_features(overview_content)
        context["feature_breakdown"] = features
        
        # Generate technical decisions
        tech_decisions = await self._make_technical_decisions(project_analysis, features)
        context["technical_decisions"] = tech_decisions
        
        # Update ARCHITECTURE.md with decisions
        await self._update_architecture_document(tech_decisions, project_type)
        
        # Update REQUIREMENTS.md with detailed specs
        await self._update_requirements_document(features, project_analysis)
        
        return context
    
    async def _analyze_project_overview(self, overview_content: str) -> Dict[str, Any]:
        """Analyze project overview for type, complexity, and requirements."""
        
        lines = overview_content.split('\n')
        project_title = "Unknown Project"
        project_type = "generic"
        
        # Extract basic information
        for line in lines:
            if line.startswith('# ') and project_title == "Unknown Project":
                project_title = line[2:].strip()
            elif 'project type' in line.lower() or 'type:' in line.lower():
                project_type = self._extract_project_type(line)
        
        # Analyze complexity
        analysis = {
            'title': project_title,
            'type': project_type,
            'has_features': '## Feature' in overview_content or '### Feature' in overview_content,
            'has_tech_stack': 'tech' in overview_content.lower() or 'stack' in overview_content.lower(),
            'has_file_structure': 'structure' in overview_content.lower() or '```' in overview_content,
            'has_success_criteria': 'success' in overview_content.lower() or 'criteria' in overview_content,
            'has_tasks': '- [ ]' in overview_content or '- [x]' in overview_content,
            'word_count': len(overview_content.split()),
            'complexity_score': self._calculate_complexity_score(overview_content),
            'estimated_work_orders': self._estimate_work_orders(overview_content)
        }
        
        return analysis
    
    def _extract_project_type(self, line: str) -> str:
        """Extract project type from a line of text."""
        line_lower = line.lower()
        type_mapping = {
            'web': 'web-app',
            'api': 'api-service', 
            'ml': 'ml-project',
            'cli': 'cli-tool',
            'game': 'game',
            'mobile': 'mobile-app',
            'desktop': 'desktop-app',
            'machine learning': 'ml-project',
            'artificial intelligence': 'ml-project',
            'rest': 'api-service',
            'service': 'api-service',
            'tool': 'cli-tool',
            'application': 'web-app'
        }
        
        for keyword, project_type in type_mapping.items():
            if keyword in line_lower:
                return project_type
        
        return 'generic'
    
    def _calculate_complexity_score(self, overview_content: str) -> float:
        """Calculate project complexity score (0.0 to 1.0)."""
        factors = []
        
        # Feature count
        feature_count = overview_content.count('### Feature') + overview_content.count('## Feature')
        factors.append(min(feature_count / 10, 1.0))
        
        # Task count
        task_count = overview_content.count('- [ ]') + overview_content.count('- [x]')
        factors.append(min(task_count / 20, 1.0))
        
        # Integration mentions
        integration_keywords = ['api', 'database', 'auth', 'payment', 'email', 'file', 'storage']
        integration_score = sum(1 for keyword in integration_keywords if keyword in overview_content.lower())
        factors.append(min(integration_score / 5, 1.0))
        
        # Technical complexity
        tech_keywords = ['security', 'scalability', 'performance', 'deployment', 'testing']
        tech_score = sum(1 for keyword in tech_keywords if keyword in overview_content.lower())
        factors.append(min(tech_score / 5, 1.0))
        
        return sum(factors) / len(factors)
    
    def _estimate_work_orders(self, overview_content: str) -> int:
        """Estimate number of work orders needed."""
        base_orders = 3  # Setup, core implementation, testing
        
        # Add orders for features
        feature_count = overview_content.count('### Feature') + overview_content.count('## Feature')
        feature_orders = max(feature_count, 1)
        
        # Add orders for integrations
        integration_keywords = ['api', 'database', 'auth', 'payment']
        integration_orders = sum(1 for keyword in integration_keywords if keyword in overview_content.lower())
        
        return base_orders + feature_orders + integration_orders
    
    async def _generate_working_assumptions(self, overview_content: str, 
                                          project_type: str) -> List[str]:
        """Generate working assumptions based on project type and content."""
        
        assumptions = []
        
        # Base assumptions for all projects
        assumptions.extend([
            "Users have basic computer literacy",
            "System will handle expected user load",
            "Error states are handled gracefully",
            "Data validation happens on both client and server"
        ])
        
        # Project type specific assumptions
        if project_type == 'web-app':
            assumptions.extend([
                "Users can navigate using mouse/keyboard/touch",
                "Target browsers support modern JavaScript (ES6+)",
                "Users have stable internet connection",
                "Responsive design works on mobile and desktop"
            ])
        elif project_type == 'game':
            assumptions.extend([
                "Users can use mouse and keyboard for input",
                "Game physics behave predictably",
                "Frame rate stays above 30 FPS",
                "Game state is properly maintained"
            ])
        elif project_type == 'api-service':
            assumptions.extend([
                "API endpoints follow RESTful conventions",
                "Response times are under 200ms for most operations",
                "API versioning strategy is implemented",
                "Rate limiting protects against abuse"
            ])
        elif project_type == 'mobile-app':
            assumptions.extend([
                "Touch gestures work intuitively",
                "App works offline for core features",
                "Push notifications are delivered reliably",
                "App starts up in under 3 seconds"
            ])
        
        # Content-specific assumptions
        if 'auth' in overview_content.lower():
            assumptions.append("Authentication tokens are secure and time-limited")
        
        if 'database' in overview_content.lower():
            assumptions.append("Database maintains data consistency and integrity")
        
        if 'payment' in overview_content.lower():
            assumptions.append("Payment processing follows security standards")
        
        return assumptions
    
    async def _update_context_document(self, assumptions: List[str], project_type: str):
        """Update CONTEXT.md with project-specific assumptions."""
        
        context_path = self.knowledge_base.docs_dir / "CONTEXT.md"
        if context_path.exists():
            content = context_path.read_text()
            
            # Add project-specific assumptions section
            project_section = f"\n## Project-Specific Assumptions ({project_type})\n\n"
            for assumption in assumptions:
                project_section += f"- [ ] {assumption}\n"
            
            # Insert before the closing line
            if "---" in content:
                content = content.replace("---", f"{project_section}\n---")
            else:
                content += f"\n{project_section}"
            
            context_path.write_text(content)
            self.logger.info(f"Updated CONTEXT.md with {len(assumptions)} project-specific assumptions")
    
    async def _extract_and_analyze_features(self, overview_content: str) -> List[Dict[str, Any]]:
        """Extract and analyze features from overview content."""
        
        features = []
        lines = overview_content.split('\n')
        current_feature = None
        
        for line in lines:
            # Look for feature headers
            if line.startswith('### Feature') or line.startswith('## Feature'):
                if current_feature:
                    features.append(current_feature)
                
                feature_name = line.split(':')[0].replace('#', '').replace('Feature', '').strip()
                if ':' in line:
                    feature_desc = line.split(':', 1)[1].strip()
                else:
                    feature_desc = feature_name
                    
                current_feature = {
                    "name": feature_name,
                    "description": feature_desc,
                    "requirements": [],
                    "complexity": "medium",
                    "dependencies": [],
                    "work_orders_needed": 1
                }
            
            elif current_feature and line.strip().startswith('- [ ]'):
                requirement = line.replace('- [ ]', '').strip()
                current_feature["requirements"].append(requirement)
        
        if current_feature:
            features.append(current_feature)
        
        # Analyze each feature
        for feature in features:
            feature["complexity"] = self._analyze_feature_complexity(feature)
            feature["work_orders_needed"] = self._estimate_feature_work_orders(feature)
        
        return features
    
    def _analyze_feature_complexity(self, feature: Dict[str, Any]) -> str:
        """Analyze feature complexity."""
        req_count = len(feature["requirements"])
        
        if req_count <= 2:
            return "low"
        elif req_count <= 5:
            return "medium"
        else:
            return "high"
    
    def _estimate_feature_work_orders(self, feature: Dict[str, Any]) -> int:
        """Estimate work orders needed for a feature."""
        complexity = feature["complexity"]
        
        if complexity == "low":
            return 1
        elif complexity == "medium":
            return 2
        else:
            return 3
    
    async def _make_technical_decisions(self, project_analysis: Dict[str, Any], 
                                      features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make technical decisions based on project analysis."""
        
        decisions = []
        project_type = project_analysis["type"]
        complexity = project_analysis["complexity_score"]
        
        # Database decision
        if any("database" in f["description"].lower() for f in features):
            if complexity > 0.7:
                decisions.append({
                    "category": "database",
                    "decision": "PostgreSQL with Redis for caching",
                    "reasoning": "High complexity project needs ACID compliance and performance"
                })
            else:
                decisions.append({
                    "category": "database", 
                    "decision": "SQLite for development, PostgreSQL for production",
                    "reasoning": "Simple setup for development, scalable for production"
                })
        
        # Authentication decision
        if any("auth" in f["description"].lower() for f in features):
            decisions.append({
                "category": "authentication",
                "decision": "JWT tokens with refresh token rotation",
                "reasoning": "Stateless, secure, and scalable authentication"
            })
        
        # Frontend framework decision
        if project_type == "web-app":
            if complexity > 0.6:
                decisions.append({
                    "category": "frontend",
                    "decision": "React with TypeScript and state management",
                    "reasoning": "Complex UI needs robust state management and type safety"
                })
            else:
                decisions.append({
                    "category": "frontend",
                    "decision": "React with JavaScript",
                    "reasoning": "Standard choice for modern web applications"
                })
        
        return decisions
    
    async def _update_architecture_document(self, tech_decisions: List[Dict[str, Any]], 
                                          project_type: str):
        """Update ARCHITECTURE.md with technical decisions."""
        
        arch_path = self.knowledge_base.docs_dir / "ARCHITECTURE.md"
        if arch_path.exists():
            content = arch_path.read_text()
            
            # Add technical decisions section
            decisions_section = f"\n## Technical Decisions Made\n\n"
            for decision in tech_decisions:
                decisions_section += f"### {decision['category'].title()}\n"
                decisions_section += f"**Decision**: {decision['decision']}\n\n"
                decisions_section += f"**Reasoning**: {decision['reasoning']}\n\n"
            
            # Insert before the closing line
            if "---" in content:
                content = content.replace("---", f"{decisions_section}\n---")
            else:
                content += f"\n{decisions_section}"
            
            arch_path.write_text(content)
            self.logger.info(f"Updated ARCHITECTURE.md with {len(tech_decisions)} technical decisions")
    
    async def _update_requirements_document(self, features: List[Dict[str, Any]], 
                                          project_analysis: Dict[str, Any]):
        """Update REQUIREMENTS.md with detailed feature specifications."""
        
        req_path = self.knowledge_base.docs_dir / "REQUIREMENTS.md"
        if req_path.exists():
            content = req_path.read_text()
            
            # Add detailed feature requirements
            features_section = f"\n## Detailed Feature Requirements\n\n"
            for i, feature in enumerate(features, 1):
                features_section += f"### {i}. {feature['name']}\n"
                features_section += f"**Description**: {feature['description']}\n\n"
                features_section += f"**Complexity**: {feature['complexity']}\n\n"
                features_section += f"**Requirements**:\n"
                for req in feature['requirements']:
                    features_section += f"- {req}\n"
                features_section += f"\n**Estimated Work Orders**: {feature['work_orders_needed']}\n\n"
            
            # Insert before the closing line
            if "---" in content:
                content = content.replace("---", f"{features_section}\n---")
            else:
                content += f"\n{features_section}"
            
            req_path.write_text(content)
            self.logger.info(f"Updated REQUIREMENTS.md with {len(features)} detailed features")
    
    async def _create_execution_plan(self, work_order_ids: List[str]) -> Dict[str, Any]:
        """Create execution plan for work orders."""
        
        plan = {
            "total_work_orders": len(work_order_ids),
            "execution_phases": {
                "setup": [],
                "core_development": [],
                "testing": [],
                "finalization": []
            },
            "estimated_duration": "TBD",
            "next_work_order": work_order_ids[0] if work_order_ids else None
        }
        
        # Categorize work orders by phase
        for wo_id in work_order_ids:
            wo_file = self.knowledge_base.work_orders_dir / f"{wo_id}.json"
            if wo_file.exists():
                work_order = json.loads(wo_file.read_text())
                wo_type = work_order.get("type", "implementation")
                
                if wo_type == "setup":
                    plan["execution_phases"]["setup"].append(wo_id)
                elif wo_type in ["implementation", "integration"]:
                    plan["execution_phases"]["core_development"].append(wo_id)
                elif wo_type == "testing":
                    plan["execution_phases"]["testing"].append(wo_id)
                else:
                    plan["execution_phases"]["finalization"].append(wo_id)
        
        return plan
    
    async def _generate_clarifying_questions(self, overview_content: str) -> List[Dict[str, Any]]:
        """Generate project-specific clarifying questions based on the overview."""
        questions = []
        
        # Detect project type from overview content
        project_type = await self._detect_basic_project_type(overview_content)
        
        # Base questions for all projects
        questions.extend([
            {
                "id": "target_audience",
                "type": "text",
                "question": "Who is the target audience for this project?",
                "default": "General users",
                "required": False
            },
            {
                "id": "complexity_preference",
                "type": "choice",
                "question": "What level of complexity do you prefer?",
                "choices": ["Simple and minimal", "Moderate complexity", "Full-featured"],
                "default": "Moderate complexity",
                "required": True
            }
        ])
        
        # Project-type-specific questions
        if project_type == "game":
            questions.extend([
                {
                    "id": "game_controls",
                    "type": "choice",
                    "question": "What control scheme do you prefer?",
                    "choices": ["Arrow keys", "WASD", "Mouse", "Both keyboard options"],
                    "default": "Both keyboard options",
                    "required": True
                },
                {
                    "id": "ai_difficulty",
                    "type": "choice",
                    "question": "What AI difficulty levels should be included?",
                    "choices": ["Easy only", "Easy and Hard", "Easy, Medium, Hard", "Adaptive difficulty"],
                    "default": "Easy and Hard",
                    "required": True
                },
                {
                    "id": "graphics_style",
                    "type": "choice",
                    "question": "What graphics style do you prefer?",
                    "choices": ["Simple shapes", "Retro pixel art", "Modern graphics", "Customizable"],
                    "default": "Simple shapes",
                    "required": False
                },
                {
                    "id": "sound_effects",
                    "type": "yes_no",
                    "question": "Should the game include sound effects?",
                    "default": False,
                    "required": False
                },
                {
                    "id": "screen_size",
                    "type": "choice",
                    "question": "What screen resolution should be supported?",
                    "choices": ["800x600", "1024x768", "1280x720", "Resizable window"],
                    "default": "1024x768",
                    "required": False
                }
            ])
        elif project_type == "web-app":
            questions.extend([
                {
                    "id": "authentication",
                    "type": "yes_no",
                    "question": "Does your application require user authentication?",
                    "default": True,
                    "required": True
                },
                {
                    "id": "database_type",
                    "type": "choice",
                    "question": "What type of database do you prefer?",
                    "choices": ["SQLite (simple)", "PostgreSQL (robust)", "MongoDB (NoSQL)", "No database"],
                    "default": "PostgreSQL (robust)",
                    "required": True
                },
                {
                    "id": "api_requirements",
                    "type": "yes_no",
                    "question": "Will this application provide API endpoints for other services?",
                    "default": False,
                    "required": True
                },
                {
                    "id": "real_time_features",
                    "type": "yes_no",
                    "question": "Do you need real-time features (websockets, live updates)?",
                    "default": False,
                    "required": False
                },
                {
                    "id": "deployment_target",
                    "type": "choice",
                    "question": "Where do you plan to deploy this application?",
                    "choices": ["Local development only", "Cloud hosting", "Docker containers", "Not sure yet"],
                    "default": "Cloud hosting",
                    "required": False
                }
            ])
        elif project_type == "cli-tool":
            questions.extend([
                {
                    "id": "command_structure",
                    "type": "choice",
                    "question": "What command structure do you prefer?",
                    "choices": ["Single command with flags", "Multiple subcommands", "Interactive menu", "Mixed approach"],
                    "default": "Multiple subcommands",
                    "required": True
                },
                {
                    "id": "configuration_file",
                    "type": "yes_no",
                    "question": "Should the tool support configuration files?",
                    "default": True,
                    "required": False
                },
                {
                    "id": "output_format",
                    "type": "choice",
                    "question": "What output formats should be supported?",
                    "choices": ["Text only", "JSON", "Both text and JSON", "Multiple formats"],
                    "default": "Both text and JSON",
                    "required": False
                },
                {
                    "id": "distribution_method",
                    "type": "choice",
                    "question": "How should the tool be distributed?",
                    "choices": ["Python package (pip)", "Standalone executable", "Docker image", "Source code only"],
                    "default": "Python package (pip)",
                    "required": False
                }
            ])
        
        return questions
    
    async def _detect_basic_project_type(self, overview_content: str) -> str:
        """Detect basic project type from overview content."""
        content_lower = overview_content.lower()
        
        # Game indicators
        if any(keyword in content_lower for keyword in ['game', 'player', 'score', 'level', 'paddle', 'ball']):
            return "game"
        
        # Web app indicators
        if any(keyword in content_lower for keyword in ['web', 'api', 'server', 'frontend', 'backend', 'database']):
            return "web-app"
        
        # CLI tool indicators
        if any(keyword in content_lower for keyword in ['cli', 'command', 'terminal', 'script', 'tool']):
            return "cli-tool"
        
        # API service indicators
        if any(keyword in content_lower for keyword in ['api', 'service', 'endpoint', 'microservice']):
            return "api-service"
        
        # Default to web-app if uncertain
        return "web-app"
    
    async def _present_questions_to_user(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Present questions to user via CLI and capture responses."""
        responses = {}
        
        print("\n" + "="*60)
        print("🤖 StarterKit Interactive Learning Phase")
        print("="*60)
        print("Please answer the following questions to refine your project:")
        print()
        
        for i, question in enumerate(questions, 1):
            print(f"Question {i}/{len(questions)}: {question['question']}")
            
            if question["type"] == "yes_no":
                while True:
                    response = input(f"  [y/n] (default: {'y' if question['default'] else 'n'}): ").strip().lower()
                    if response == "":
                        responses[question["id"]] = question["default"]
                        break
                    elif response in ["y", "yes"]:
                        responses[question["id"]] = True
                        break
                    elif response in ["n", "no"]:
                        responses[question["id"]] = False
                        break
                    else:
                        print("  Please enter 'y' for yes or 'n' for no.")
            
            elif question["type"] == "choice":
                print("  Choices:")
                for j, choice in enumerate(question["choices"], 1):
                    print(f"    {j}. {choice}")
                
                while True:
                    response = input(f"  Enter choice number (1-{len(question['choices'])}) or press Enter for default: ").strip()
                    if response == "":
                        responses[question["id"]] = question["default"]
                        break
                    try:
                        choice_index = int(response) - 1
                        if 0 <= choice_index < len(question["choices"]):
                            responses[question["id"]] = question["choices"][choice_index]
                            break
                        else:
                            print(f"  Please enter a number between 1 and {len(question['choices'])}")
                    except ValueError:
                        print("  Please enter a valid number")
            
            elif question["type"] == "text":
                response = input(f"  Enter your answer (default: {question['default']}): ").strip()
                if response == "":
                    responses[question["id"]] = question["default"]
                else:
                    responses[question["id"]] = response
            
            print()
        
        print("="*60)
        print("✅ Interactive learning complete! Refining project scope...")
        print("="*60)
        print()
        
        return responses
    
    async def _refine_project_scope(self, overview_content: str, user_responses: Dict[str, Any]) -> str:
        """Refine project scope based on user responses."""
        
        # Start with original overview
        refined_content = overview_content
        
        # Add a refinement section based on user responses
        refinement_section = "\n\n## Project Refinement (Based on User Input)\n\n"
        
        # Add target audience
        if "target_audience" in user_responses:
            refinement_section += f"**Target Audience**: {user_responses['target_audience']}\n\n"
        
        # Add complexity preference
        if "complexity_preference" in user_responses:
            refinement_section += f"**Complexity Level**: {user_responses['complexity_preference']}\n\n"
        
        # Add project-specific refinements
        if "game_controls" in user_responses:
            refinement_section += f"**Game Controls**: {user_responses['game_controls']}\n"
        
        if "ai_difficulty" in user_responses:
            refinement_section += f"**AI Difficulty**: {user_responses['ai_difficulty']}\n"
        
        if "graphics_style" in user_responses:
            refinement_section += f"**Graphics Style**: {user_responses['graphics_style']}\n"
        
        if "sound_effects" in user_responses:
            sound_status = "Yes" if user_responses["sound_effects"] else "No"
            refinement_section += f"**Sound Effects**: {sound_status}\n"
        
        if "screen_size" in user_responses:
            refinement_section += f"**Screen Resolution**: {user_responses['screen_size']}\n"
        
        # Web app specific refinements
        if "authentication" in user_responses:
            auth_status = "Required" if user_responses["authentication"] else "Not required"
            refinement_section += f"**Authentication**: {auth_status}\n"
        
        if "database_type" in user_responses:
            refinement_section += f"**Database**: {user_responses['database_type']}\n"
        
        if "api_requirements" in user_responses:
            api_status = "Yes" if user_responses["api_requirements"] else "No"
            refinement_section += f"**API Endpoints**: {api_status}\n"
        
        if "real_time_features" in user_responses:
            realtime_status = "Yes" if user_responses["real_time_features"] else "No"
            refinement_section += f"**Real-time Features**: {realtime_status}\n"
        
        if "deployment_target" in user_responses:
            refinement_section += f"**Deployment**: {user_responses['deployment_target']}\n"
        
        # CLI tool specific refinements
        if "command_structure" in user_responses:
            refinement_section += f"**Command Structure**: {user_responses['command_structure']}\n"
        
        if "configuration_file" in user_responses:
            config_status = "Yes" if user_responses["configuration_file"] else "No"
            refinement_section += f"**Configuration File**: {config_status}\n"
        
        if "output_format" in user_responses:
            refinement_section += f"**Output Format**: {user_responses['output_format']}\n"
        
        if "distribution_method" in user_responses:
            refinement_section += f"**Distribution**: {user_responses['distribution_method']}\n"
        
        # Add technical implications
        refinement_section += "\n### Technical Implications\n"
        
        # Add implications based on responses
        if user_responses.get("complexity_preference") == "Simple and minimal":
            refinement_section += "- Focus on core functionality only\n"
            refinement_section += "- Minimal external dependencies\n"
            refinement_section += "- Simple architecture and file structure\n"
        elif user_responses.get("complexity_preference") == "Full-featured":
            refinement_section += "- Comprehensive feature set\n"
            refinement_section += "- Robust error handling and logging\n"
            refinement_section += "- Extensible architecture\n"
        
        if user_responses.get("authentication") == False:
            refinement_section += "- No user management system needed\n"
            refinement_section += "- Simplified security requirements\n"
        
        if user_responses.get("database_type") == "No database":
            refinement_section += "- File-based storage or in-memory data structures\n"
            refinement_section += "- No database setup or migration scripts\n"
        
        # Append refinement section to original content
        refined_content += refinement_section
        
        # Save refined overview
        overview_path = Path("OVERVIEW.md")
        overview_path.write_text(refined_content)
        
        self.logger.info("Project scope refined and OVERVIEW.md updated")
        return refined_content
    
    async def _prepare_incremental_execution(self, project_context: Dict[str, Any], 
                                           project_type: str) -> Dict[str, Any]:
        """Prepare project for incremental work order execution."""
        
        return {
            "execution_type": "incremental",
            "project_type": project_type,
            "project_context": project_context,
            "ready_for_work_orders": True,
            "next_step": "Create first work order via WorkOrderManager.create_next_work_order()",
            "notes": "Work orders will be created one at a time based on project progress"
        }