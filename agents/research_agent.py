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