"""
Research Agent for the Ultimate Agentic StarterKit.

This agent uses Ollama with small models (like Mistral) to research and suggest
answers to project questions. Supports web search integration and interactive prompts.
"""

import json
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
from datetime import datetime

from agents.base_agent import BaseAgent
from core.models import AgentResult, ProjectTask, TaskStatus, AgentType
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts


class ResearchAgent(BaseAgent):
    """
    Research Agent that fills in suggested answers for project questions.
    
    Uses Ollama for local LLM reasoning and supports web search integration
    for external knowledge gathering.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Research Agent."""
        super().__init__("research", config)
        
        self.ollama_model = self.config.get('ollama_model', 'mistral:7b')
        self.voice = get_voice_alerts()
        self.max_research_attempts = 3
        
        # Research knowledge base for common questions
        self.knowledge_base = {
            'authentication': {
                'web-app': 'Consider JWT tokens, OAuth 2.0, or session-based auth. For simple apps, JWT is recommended.',
                'api-service': 'Use API keys for service-to-service, JWT for user authentication, OAuth for third-party access.'
            },
            'database': {
                'web-app': 'PostgreSQL for complex queries, MongoDB for document storage, SQLite for simple apps.',
                'api-service': 'PostgreSQL for ACID compliance, Redis for caching, MongoDB for flexible schemas.'
            },
            'tech_stack': {
                'web-app': 'Frontend: React/Vue/Angular, Backend: Node.js/Python FastAPI/Django, Database: PostgreSQL',
                'api-service': 'Python FastAPI for fast development, Node.js Express for JavaScript, Go for performance',
                'ml-project': 'Python with scikit-learn/TensorFlow/PyTorch, Jupyter for experimentation, MLflow for tracking'
            }
        }
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute research to fill in suggested answers for questions.
        
        Args:
            task: ProjectTask containing questions file path to research
            
        Returns:
            AgentResult with updated questions and research results
        """
        self.logger.info(f"Starting research for: {task.title}")
        
        try:
            # Load questions from file
            questions_file = task.description.strip()
            if not Path(questions_file).exists():
                questions_file = "questions.md"
            
            questions_content = Path(questions_file).read_text()
            questions = self._parse_questions_document(questions_content)
            
            # Research each question that requires it
            research_results = []
            for question in questions:
                if question['requires_research'] and not question['suggested_answer']:
                    result = await self._research_question(question)
                    research_results.append(result)
                    question['suggested_answer'] = result['answer']
                    question['research_confidence'] = result['confidence']
            
            # Update questions document
            updated_content = self._update_questions_document(questions_content, questions)
            Path(questions_file).write_text(updated_content)
            
            result = AgentResult(
                agent_id=self.agent_id,
                success=True,
                output={
                    'questions_file': questions_file,
                    'research_results': research_results,
                    'questions_researched': len(research_results),
                    'updated_questions': questions
                },
                confidence=0.8,
                execution_time=0.0
            )
            
            self.logger.info(f"Completed research for {len(research_results)} questions")
            return result
            
        except Exception as e:
            self.logger.error(f"Research agent execution failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                confidence=0.0,
                execution_time=0.0
            )
    
    def _parse_questions_document(self, content: str) -> List[Dict[str, Any]]:
        """Parse questions document to extract structured question data."""
        
        questions = []
        lines = content.split('\n')
        current_question = None
        
        for line in lines:
            # Match question headers like "### [Critical] Question: What is...?"
            question_match = re.match(r'### \[(\w+)\] Question: (.+?)(\*.*)?$', line)
            if question_match:
                if current_question:
                    questions.append(current_question)
                
                priority = question_match.group(1)
                question_text = question_match.group(2).strip()
                requires_research = '*(Research Required)*' in line
                
                current_question = {
                    'priority': priority,
                    'question': question_text,
                    'requires_research': requires_research,
                    'suggested_answer': '',
                    'user_answer': '',
                    'why': '',
                    'research_confidence': 0.0
                }
            
            elif current_question:
                # Parse other fields
                if line.startswith('**Why**:'):
                    current_question['why'] = line.replace('**Why**:', '').strip()
                elif line.startswith('**Suggested Answer**:'):
                    answer = line.replace('**Suggested Answer**:', '').strip()
                    if answer != '_To be filled by Research Agent_':
                        current_question['suggested_answer'] = answer
                elif line.startswith('**Your Answer**:'):
                    user_answer = line.replace('**Your Answer**:', '').strip()
                    # Remove checkbox markers
                    user_answer = re.sub(r'\[.\]\s*', '', user_answer)
                    if user_answer:
                        current_question['user_answer'] = user_answer
        
        # Add last question
        if current_question:
            questions.append(current_question)
        
        return questions
    
    async def _research_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Research a single question to provide a suggested answer."""
        
        question_text = question['question']
        question_type = self._categorize_question(question_text)
        
        # Try knowledge base first
        kb_answer = self._get_knowledge_base_answer(question_text, question_type)
        if kb_answer:
            return {
                'question': question_text,
                'answer': kb_answer,
                'method': 'knowledge_base',
                'confidence': 0.8
            }
        
        # Try Ollama research
        ollama_answer = await self._research_with_ollama(question_text, question_type)
        if ollama_answer:
            return ollama_answer
        
        # Fallback to interactive prompt
        interactive_answer = await self._interactive_research(question_text)
        return interactive_answer
    
    def _categorize_question(self, question: str) -> str:
        """Categorize question to determine research approach."""
        
        question_lower = question.lower()
        
        categories = {
            'authentication': ['auth', 'login', 'security', 'permission', 'user'],
            'database': ['database', 'storage', 'data', 'persist', 'sql'],
            'tech_stack': ['technology', 'framework', 'language', 'stack', 'tool'],
            'api': ['api', 'endpoint', 'service', 'rest', 'graphql'],
            'performance': ['performance', 'speed', 'scalability', 'load'],
            'deployment': ['deploy', 'hosting', 'server', 'cloud', 'infrastructure'],
            'ui_ux': ['interface', 'user experience', 'design', 'frontend'],
            'testing': ['test', 'quality', 'validation', 'verification']
        }
        
        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _get_knowledge_base_answer(self, question: str, question_type: str) -> Optional[str]:
        """Get answer from built-in knowledge base."""
        
        if question_type in self.knowledge_base:
            kb_section = self.knowledge_base[question_type]
            
            # Try to match project type or return general answer
            if isinstance(kb_section, dict):
                # Return first available answer if no specific type match
                return next(iter(kb_section.values()))
            else:
                return kb_section
        
        return None
    
    async def _research_with_ollama(self, question: str, question_type: str) -> Optional[Dict[str, Any]]:
        """Research question using Ollama local LLM."""
        
        try:
            # Check if Ollama is available
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                self.logger.warning("Ollama not available, skipping LLM research")
                return None
            
            # Create research prompt
            prompt = f"""You are a software engineering consultant. Answer this project question concisely and practically:

Question: {question}
Category: {question_type}

Provide a brief, actionable answer (2-3 sentences max) suitable for a project overview. Focus on common best practices and popular choices.

Answer:"""
            
            # Call Ollama
            ollama_result = subprocess.run(
                ['ollama', 'run', self.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if ollama_result.returncode == 0:
                answer = ollama_result.stdout.strip()
                if answer and len(answer) > 10:  # Ensure meaningful response
                    return {
                        'question': question,
                        'answer': answer,
                        'method': 'ollama',
                        'confidence': 0.7
                    }
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning(f"Ollama research failed: {e}")
        
        return None
    
    async def _interactive_research(self, question: str) -> Dict[str, Any]:
        """Use interactive prompt for questions requiring user input."""
        
        try:
            # Voice alert
            if self.voice:
                self.voice.speak(f"Quick question: {question}")
            
            # CLI prompt
            print(f"\n🤔 Research needed: {question}")
            print("Enter your answer (or 'skip' to leave blank):")
            
            user_input = input("> ").strip()
            
            if user_input.lower() in ['skip', 'none', '']:
                answer = "User input required - please provide answer manually"
                confidence = 0.2
            else:
                answer = user_input
                confidence = 0.9  # High confidence for user-provided answers
            
            return {
                'question': question,
                'answer': answer,
                'method': 'interactive',
                'confidence': confidence
            }
            
        except (KeyboardInterrupt, EOFError):
            return {
                'question': question,
                'answer': "User input required - please provide answer manually",
                'method': 'interactive_skipped',
                'confidence': 0.1
            }
    
    def _update_questions_document(self, original_content: str, questions: List[Dict[str, Any]]) -> str:
        """Update questions document with research results."""
        
        lines = original_content.split('\n')
        updated_lines = []
        current_question_index = -1
        
        for line in lines:
            # Check if this is a question header
            if re.match(r'### \[(\w+)\] Question:', line):
                current_question_index += 1
            
            # Update suggested answer lines
            if line.startswith('**Suggested Answer**:') and current_question_index < len(questions):
                question = questions[current_question_index]
                if question['suggested_answer']:
                    confidence_note = ""
                    if question.get('research_confidence', 0) > 0:
                        confidence_note = f" *(Confidence: {question['research_confidence']:.1%})*"
                    updated_lines.append(f"**Suggested Answer**: {question['suggested_answer']}{confidence_note}")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        return '\n'.join(updated_lines)
    
    async def quick_research(self, query: str) -> str:
        """Quick research for simple questions during learning loop."""
        
        # Try knowledge base first
        answer = self._get_knowledge_base_answer(query, self._categorize_question(query))
        if answer:
            return answer
        
        # Try Ollama for quick answer
        ollama_result = await self._research_with_ollama(query, 'general')
        if ollama_result:
            return ollama_result['answer']
        
        # Fallback prompt
        if self.voice:
            self.voice.speak(f"Quick question: {query}")
        
        print(f"\n🔍 {query}")
        user_answer = input("Quick answer: ").strip()
        
        return user_answer if user_answer else "No answer provided"
    
    async def generate_comprehensive_prp(self, work_order: Dict[str, Any], 
                                       project_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive PRP (Project Requirements Package) for work order.
        
        This method analyzes the codebase and creates detailed context for Claude SDK
        to ensure it has comprehensive understanding of the project state.
        
        Args:
            work_order: Work order specification
            project_context: Current project context
            
        Returns:
            Comprehensive PRP with codebase analysis and context
        """
        self.logger.info(f"Generating comprehensive PRP for work order: {work_order['id']}")
        
        try:
            # 1. Analyze existing codebase structure
            codebase_analysis = await self._analyze_codebase_structure(project_context)
            
            # 2. Review previous work order completions
            history_analysis = await self._analyze_completion_history(project_context)
            
            # 3. Create comprehensive implementation guidance
            implementation_guide = await self._generate_implementation_guidance(
                work_order, codebase_analysis, history_analysis
            )
            
            # 4. Generate project-specific context
            project_specific_context = await self._generate_project_specific_context(
                work_order, project_context
            )
            
            # 5. Create final PRP package
            prp = {
                'work_order_id': work_order['id'],
                'work_order_title': work_order['title'],
                'work_order_description': work_order['description'],
                'generated_at': datetime.now().isoformat(),
                'codebase_analysis': codebase_analysis,
                'completion_history': history_analysis,
                'implementation_guidance': implementation_guide,
                'project_context': project_specific_context,
                'comprehensive_context': self._create_comprehensive_context(
                    work_order, codebase_analysis, history_analysis, implementation_guide
                ),
                'prp_version': '2.0',
                'research_method': 'comprehensive_codebase_analysis'
            }
            
            self.logger.info(f"Successfully generated comprehensive PRP for {work_order['id']}")
            return prp
            
        except Exception as e:
            self.logger.error(f"Failed to generate PRP for {work_order['id']}: {e}")
            return {
                'work_order_id': work_order['id'],
                'error': str(e),
                'fallback_context': project_context,
                'prp_version': '2.0-fallback'
            }
    
    async def _analyze_codebase_structure(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze existing codebase structure and components."""
        from pathlib import Path
        
        workspace_path = Path(project_context.get('workspace_path', 'workspace'))
        
        analysis = {
            'file_structure': {},
            'existing_components': [],
            'code_patterns': [],
            'dependencies': [],
            'architecture_notes': []
        }
        
        try:
            # Analyze directory structure
            if workspace_path.exists():
                analysis['file_structure'] = self._scan_directory_structure(workspace_path)
                
                # Identify existing components
                analysis['existing_components'] = self._identify_components(workspace_path)
                
                # Analyze code patterns
                analysis['code_patterns'] = self._analyze_code_patterns(workspace_path)
                
                # Identify dependencies
                analysis['dependencies'] = self._identify_dependencies(workspace_path)
                
                # Generate architecture notes
                analysis['architecture_notes'] = self._generate_architecture_notes(workspace_path)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Codebase analysis failed: {e}")
            return {
                'error': str(e),
                'file_structure': {},
                'existing_components': [],
                'code_patterns': [],
                'dependencies': [],
                'architecture_notes': []
            }
    
    def _scan_directory_structure(self, path: Path) -> Dict[str, Any]:
        """Scan directory structure and file types."""
        structure = {}
        
        try:
            for item in path.iterdir():
                if item.is_file():
                    structure[item.name] = {
                        'type': 'file',
                        'size': item.stat().st_size,
                        'extension': item.suffix,
                        'last_modified': item.stat().st_mtime
                    }
                elif item.is_dir() and not item.name.startswith('.'):
                    structure[item.name] = {
                        'type': 'directory',
                        'contents': self._scan_directory_structure(item)
                    }
        except Exception as e:
            self.logger.error(f"Directory scan failed for {path}: {e}")
        
        return structure
    
    def _identify_components(self, path: Path) -> List[Dict[str, Any]]:
        """Identify existing code components and their purposes."""
        components = []
        
        try:
            # Look for Python files
            for py_file in path.glob('**/*.py'):
                if py_file.name != '__init__.py':
                    component = {
                        'name': py_file.name,
                        'path': str(py_file.relative_to(path)),
                        'type': 'python_module',
                        'purpose': self._infer_component_purpose(py_file)
                    }
                    components.append(component)
            
            # Look for config files
            for config_file in path.glob('**/requirements*.txt'):
                component = {
                    'name': config_file.name,
                    'path': str(config_file.relative_to(path)),
                    'type': 'dependency_config',
                    'purpose': 'Project dependencies'
                }
                components.append(component)
            
            # Look for test files
            for test_file in path.glob('**/test_*.py'):
                component = {
                    'name': test_file.name,
                    'path': str(test_file.relative_to(path)),
                    'type': 'test_file',
                    'purpose': 'Unit tests'
                }
                components.append(component)
        
        except Exception as e:
            self.logger.error(f"Component identification failed: {e}")
        
        return components
    
    def _infer_component_purpose(self, file_path: Path) -> str:
        """Infer the purpose of a code component from its name and content."""
        name = file_path.name.lower()
        
        # Purpose mapping based on common patterns
        purpose_map = {
            'main': 'Application entry point',
            'game': 'Game logic implementation',
            'pong': 'Pong game implementation',
            'setup': 'Project setup and configuration',
            'config': 'Configuration management',
            'utils': 'Utility functions',
            'models': 'Data models',
            'api': 'API endpoints',
            'auth': 'Authentication logic',
            'database': 'Database operations',
            'test': 'Test implementation'
        }
        
        for keyword, purpose in purpose_map.items():
            if keyword in name:
                return purpose
        
        return 'General application component'
    
    def _analyze_code_patterns(self, path: Path) -> List[Dict[str, Any]]:
        """Analyze code patterns and architectural decisions."""
        patterns = []
        
        try:
            # Look for common patterns in Python files
            for py_file in path.glob('**/*.py'):
                try:
                    content = py_file.read_text()
                    
                    # Check for class definitions
                    if 'class ' in content:
                        patterns.append({
                            'pattern': 'Object-Oriented Design',
                            'file': str(py_file.relative_to(path)),
                            'evidence': 'Contains class definitions'
                        })
                    
                    # Check for async patterns
                    if 'async def' in content:
                        patterns.append({
                            'pattern': 'Asynchronous Programming',
                            'file': str(py_file.relative_to(path)),
                            'evidence': 'Uses async/await'
                        })
                    
                    # Check for testing patterns
                    if 'import pytest' in content or 'def test_' in content:
                        patterns.append({
                            'pattern': 'Unit Testing',
                            'file': str(py_file.relative_to(path)),
                            'evidence': 'Contains test functions'
                        })
                
                except Exception as e:
                    self.logger.debug(f"Could not analyze {py_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
        
        return patterns
    
    def _identify_dependencies(self, path: Path) -> List[Dict[str, Any]]:
        """Identify project dependencies and their purposes."""
        dependencies = []
        
        try:
            # Check requirements.txt
            req_file = path / 'requirements.txt'
            if req_file.exists():
                content = req_file.read_text()
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep_name = line.split('==')[0].split('>=')[0].split('~=')[0]
                        dependencies.append({
                            'name': dep_name,
                            'purpose': self._infer_dependency_purpose(dep_name),
                            'source': 'requirements.txt'
                        })
            
            # Check for import statements in Python files
            for py_file in path.glob('**/*.py'):
                try:
                    content = py_file.read_text()
                    for line in content.split('\n'):
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            # Extract module name
                            if 'import ' in line:
                                module = line.split('import ')[1].split()[0].split('.')[0]
                                if module not in ['os', 'sys', 'json', 'datetime']:  # Skip standard library
                                    dependencies.append({
                                        'name': module,
                                        'purpose': self._infer_dependency_purpose(module),
                                        'source': str(py_file.relative_to(path))
                                    })
                except Exception as e:
                    self.logger.debug(f"Could not analyze imports in {py_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}")
        
        # Remove duplicates
        unique_deps = []
        seen = set()
        for dep in dependencies:
            if dep['name'] not in seen:
                unique_deps.append(dep)
                seen.add(dep['name'])
        
        return unique_deps
    
    def _infer_dependency_purpose(self, dep_name: str) -> str:
        """Infer the purpose of a dependency."""
        purpose_map = {
            'pygame': 'Game development framework',
            'flask': 'Web application framework',
            'fastapi': 'Modern web API framework',
            'django': 'Full-featured web framework',
            'pytest': 'Testing framework',
            'requests': 'HTTP client library',
            'numpy': 'Numerical computing',
            'pandas': 'Data manipulation',
            'matplotlib': 'Data visualization',
            'pydantic': 'Data validation',
            'sqlalchemy': 'Database ORM',
            'asyncio': 'Asynchronous programming',
            'pathlib': 'Path manipulation',
            'json': 'JSON handling',
            'datetime': 'Date/time handling'
        }
        
        return purpose_map.get(dep_name.lower(), 'General purpose library')
    
    def _generate_architecture_notes(self, path: Path) -> List[str]:
        """Generate architecture notes based on codebase analysis."""
        notes = []
        
        try:
            # Check project structure
            if (path / 'src').exists():
                notes.append("Project uses src/ directory for source code organization")
            
            if (path / 'tests').exists():
                notes.append("Project includes dedicated tests/ directory")
            
            if (path / 'docs').exists():
                notes.append("Project includes documentation directory")
            
            # Check for configuration files
            config_files = ['setup.py', 'pyproject.toml', 'requirements.txt', 'Dockerfile']
            for config_file in config_files:
                if (path / config_file).exists():
                    notes.append(f"Project uses {config_file} for configuration")
            
            # Check for game-specific patterns
            if any(f.name.lower() == 'game.py' for f in path.rglob('*.py')):
                notes.append("Game development project with dedicated game module")
            
            # Check for web-specific patterns
            if any('app' in f.name.lower() for f in path.rglob('*.py')):
                notes.append("Web application project with app modules")
        
        except Exception as e:
            self.logger.error(f"Architecture analysis failed: {e}")
        
        return notes
    
    async def _analyze_completion_history(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze previous work order completions for context."""
        history = {
            'completed_work_orders': [],
            'lessons_learned': [],
            'implementation_patterns': [],
            'success_factors': []
        }
        
        try:
            completion_history = project_context.get('completion_history', [])
            
            for completion in completion_history:
                work_order_info = {
                    'id': completion.get('work_order_id', 'unknown'),
                    'title': completion.get('title', 'Unknown'),
                    'success': completion.get('success', False),
                    'lessons': completion.get('lessons_learned', []),
                    'artifacts': completion.get('artifacts', [])
                }
                history['completed_work_orders'].append(work_order_info)
                
                # Extract lessons learned
                if completion.get('lessons_learned'):
                    history['lessons_learned'].extend(completion['lessons_learned'])
                
                # Extract implementation patterns
                if completion.get('context_updates'):
                    history['implementation_patterns'].extend(completion['context_updates'])
                
                # Extract success factors
                if completion.get('success') and completion.get('success_factors'):
                    history['success_factors'].extend(completion['success_factors'])
        
        except Exception as e:
            self.logger.error(f"History analysis failed: {e}")
        
        return history
    
    async def _generate_implementation_guidance(self, work_order: Dict[str, Any], 
                                             codebase_analysis: Dict[str, Any], 
                                             history_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed implementation guidance for Claude SDK."""
        guidance = {
            'recommended_approach': '',
            'key_considerations': [],
            'integration_points': [],
            'best_practices': [],
            'potential_challenges': []
        }
        
        try:
            # Use AI to generate guidance based on analysis
            from integrations.ollama_client import OllamaClient
            
            client = OllamaClient()
            
            prompt = f"""
            Based on the following codebase analysis and work order, provide detailed implementation guidance:
            
            Work Order: {work_order['title']}
            Description: {work_order['description']}
            
            Existing Components: {[c['name'] for c in codebase_analysis['existing_components']]}
            Code Patterns: {[p['pattern'] for p in codebase_analysis['code_patterns']]}
            Dependencies: {[d['name'] for d in codebase_analysis['dependencies']]}
            
            Previous Lessons: {history_analysis['lessons_learned'][:5]}
            
            Please provide:
            1. Recommended implementation approach
            2. Key considerations for this work order
            3. Integration points with existing code
            4. Best practices to follow
            5. Potential challenges to watch for
            
            Keep the response concise but comprehensive.
            """
            
            response = await client.generate_response(
                model=self.ollama_model,
                prompt=prompt
            )
            
            # Parse response into structured guidance
            if response.get("success", False):
                guidance = self._parse_guidance_response(response.get("response", ""))
            else:
                self.logger.error(f"Failed to get AI guidance: {response.get('error', 'Unknown error')}")
                raise Exception(f"AI guidance generation failed: {response.get('error', 'Unknown error')}")
            
        except Exception as e:
            self.logger.error(f"AI guidance generation failed: {e}")
            # Fallback to basic guidance
            guidance = {
                'recommended_approach': f"Implement {work_order['title']} following existing project patterns",
                'key_considerations': ['Maintain code consistency', 'Follow existing patterns', 'Add appropriate tests'],
                'integration_points': ['Review existing components for integration opportunities'],
                'best_practices': ['Use type hints', 'Add docstrings', 'Include error handling'],
                'potential_challenges': ['Integration complexity', 'Dependency management']
            }
        
        return guidance
    
    def _parse_guidance_response(self, response: str) -> Dict[str, Any]:
        """Parse AI-generated guidance response into structured format."""
        guidance = {
            'recommended_approach': '',
            'key_considerations': [],
            'integration_points': [],
            'best_practices': [],
            'potential_challenges': []
        }
        
        try:
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Identify sections
                if 'recommended' in line.lower() and 'approach' in line.lower():
                    current_section = 'recommended_approach'
                elif 'key considerations' in line.lower():
                    current_section = 'key_considerations'
                elif 'integration points' in line.lower():
                    current_section = 'integration_points'
                elif 'best practices' in line.lower():
                    current_section = 'best_practices'
                elif 'potential challenges' in line.lower():
                    current_section = 'potential_challenges'
                elif line.startswith('-') or line.startswith('*'):
                    # List item
                    item = line.lstrip('-* ').strip()
                    if current_section and current_section != 'recommended_approach':
                        guidance[current_section].append(item)
                else:
                    # Regular text
                    if current_section == 'recommended_approach':
                        guidance['recommended_approach'] += line + ' '
        
        except Exception as e:
            self.logger.error(f"Response parsing failed: {e}")
        
        return guidance
    
    async def _generate_project_specific_context(self, work_order: Dict[str, Any], 
                                               project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate project-specific context for work order."""
        context = {
            'project_type': project_context.get('project_type', 'unknown'),
            'project_phase': project_context.get('current_phase', 'unknown'),
            'relevant_documents': [],
            'key_requirements': [],
            'technical_constraints': []
        }
        
        try:
            # Extract relevant documents
            documents = project_context.get('documents', {})
            for doc_type, doc_content in documents.items():
                if doc_content and len(doc_content) > 100:  # Only include substantial documents
                    context['relevant_documents'].append({
                        'type': doc_type,
                        'summary': doc_content[:200] + '...' if len(doc_content) > 200 else doc_content
                    })
            
            # Extract key requirements from work order
            description = work_order.get('description', '')
            if description:
                context['key_requirements'] = self._extract_requirements(description)
            
            # Generate technical constraints
            context['technical_constraints'] = self._generate_constraints(project_context)
        
        except Exception as e:
            self.logger.error(f"Project context generation failed: {e}")
        
        return context
    
    def _extract_requirements(self, description: str) -> List[str]:
        """Extract key requirements from work order description."""
        requirements = []
        
        # Look for requirement indicators
        requirement_indicators = [
            'must', 'should', 'shall', 'need to', 'required to', 'implement',
            'create', 'build', 'develop', 'add', 'include', 'ensure'
        ]
        
        sentences = description.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in requirement_indicators):
                requirements.append(sentence)
        
        return requirements[:5]  # Limit to most important requirements
    
    def _generate_constraints(self, project_context: Dict[str, Any]) -> List[str]:
        """Generate technical constraints based on project context."""
        constraints = []
        
        # Technology constraints
        project_type = project_context.get('project_type', '')
        if project_type == 'GAME':
            constraints.append('Must use game development best practices')
            constraints.append('Performance optimization for real-time gameplay')
        elif project_type == 'WEB_APP':
            constraints.append('Must be web-compatible')
            constraints.append('Security considerations for web deployment')
        elif project_type == 'CLI_TOOL':
            constraints.append('Must be command-line compatible')
            constraints.append('Cross-platform compatibility')
        
        # Dependencies constraints
        dependencies = project_context.get('dependencies', [])
        if dependencies:
            constraints.append(f'Must work with existing dependencies: {", ".join(dependencies[:3])}')
        
        return constraints
    
    def _create_comprehensive_context(self, work_order: Dict[str, Any], 
                                    codebase_analysis: Dict[str, Any], 
                                    history_analysis: Dict[str, Any], 
                                    implementation_guidance: Dict[str, Any]) -> str:
        """Create comprehensive context string for Claude SDK."""
        context_parts = []
        
        # Work order context
        context_parts.append(f"## Work Order: {work_order['title']}")
        context_parts.append(f"**Description**: {work_order['description']}")
        context_parts.append("")
        
        # Codebase analysis
        context_parts.append("## Existing Codebase Analysis")
        if codebase_analysis['existing_components']:
            context_parts.append("**Existing Components**:")
            for comp in codebase_analysis['existing_components'][:10]:  # Limit to avoid context overflow
                context_parts.append(f"- {comp['name']}: {comp['purpose']}")
        
        if codebase_analysis['code_patterns']:
            context_parts.append("**Code Patterns**:")
            for pattern in codebase_analysis['code_patterns'][:5]:
                context_parts.append(f"- {pattern['pattern']}: {pattern['evidence']}")
        
        if codebase_analysis['dependencies']:
            context_parts.append("**Dependencies**:")
            for dep in codebase_analysis['dependencies'][:10]:
                context_parts.append(f"- {dep['name']}: {dep['purpose']}")
        
        context_parts.append("")
        
        # Implementation guidance
        context_parts.append("## Implementation Guidance")
        context_parts.append(f"**Recommended Approach**: {implementation_guidance['recommended_approach']}")
        
        if implementation_guidance['key_considerations']:
            context_parts.append("**Key Considerations**:")
            for consideration in implementation_guidance['key_considerations']:
                context_parts.append(f"- {consideration}")
        
        if implementation_guidance['best_practices']:
            context_parts.append("**Best Practices**:")
            for practice in implementation_guidance['best_practices']:
                context_parts.append(f"- {practice}")
        
        context_parts.append("")
        
        # History context
        if history_analysis['lessons_learned']:
            context_parts.append("## Lessons from Previous Work Orders")
            for lesson in history_analysis['lessons_learned'][:5]:
                context_parts.append(f"- {lesson}")
        
        # Join all parts
        full_context = "\n".join(context_parts)
        
        # Truncate if too long for Claude context
        if len(full_context) > 180000:  # Leave room for other context
            full_context = full_context[:180000] + "\n\n[Context truncated for length]"
        
        return full_context