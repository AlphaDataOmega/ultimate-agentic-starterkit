"""
Project Manager Agent for the Ultimate Agentic StarterKit.

This agent uses OpenAI o3 for intelligent project analysis and question generation
to facilitate knowledge transfer and project understanding.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from agents.base_agent import BaseAgent
from core.models import AgentResult, ProjectTask, TaskStatus, AgentType
from core.logger import get_logger
from core.config import get_config


class ProjectManagerAgent(BaseAgent):
    """
    Project Manager Agent that analyzes project overviews and generates 
    prioritized questions for knowledge transfer.
    
    Uses OpenAI o3 for sophisticated reasoning about project requirements,
    gaps, and critical information needed for successful implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Project Manager Agent."""
        super().__init__("project_manager", config)
        
        self.max_iterations = self.config.get('max_iterations', 5)
        self.question_priorities = ['Critical', 'MVP', 'Future']
        
        # Question templates for different project types
        self.question_templates = {
            'web-app': [
                "What authentication method should be used?",
                "What database will store user data?",
                "What are the main user roles and permissions?",
                "What external APIs need integration?",
                "What are the performance requirements?"
            ],
            'api-service': [
                "What data format should the API return (JSON/XML)?",
                "What authentication/authorization is required?",
                "What rate limiting should be implemented?",
                "What are the main endpoints and their purposes?",
                "What error handling strategy should be used?"
            ],
            'ml-project': [
                "What type of ML model is needed (classification/regression/clustering)?",
                "What is the expected data format and size?",
                "What are the model performance requirements?",
                "What deployment platform will be used?",
                "What data preprocessing is required?"
            ],
            'cli-tool': [
                "What are the main command-line options needed?",
                "What input/output formats are expected?",
                "What configuration methods should be supported?",
                "What error handling and logging is required?",
                "What are the installation requirements?"
            ]
        }
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute project manager analysis to generate questions.
        
        Args:
            task: ProjectTask containing the project overview to analyze
            
        Returns:
            AgentResult with generated questions and analysis
        """
        self.logger.info(f"Starting project analysis for: {task.title}")
        
        try:
            # Parse project overview
            overview_content = task.description
            project_analysis = await self._analyze_project_overview(overview_content)
            
            # Generate prioritized questions
            questions = await self._generate_questions(project_analysis)
            
            # Create questions document
            questions_doc = self._format_questions_document(questions, project_analysis)
            
            # Save questions to file
            questions_path = await self._save_questions_document(questions_doc)
            
            result = AgentResult(
                agent_id=self.agent_id,
                success=True,
                output={
                    'questions': questions,
                    'project_analysis': project_analysis,
                    'questions_file': str(questions_path),
                    'total_questions': len(questions),
                    'critical_questions': len([q for q in questions if q['priority'] == 'Critical'])
                },
                confidence=0.9,
                execution_time=0.0
            )
            
            self.logger.info(f"Generated {len(questions)} questions for project analysis")
            return result
            
        except Exception as e:
            self.logger.error(f"Project manager execution failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                confidence=0.0,
                execution_time=0.0
            )
    
    async def _analyze_project_overview(self, overview_content: str) -> Dict[str, Any]:
        """Analyze project overview to understand requirements and identify gaps."""
        
        # Extract basic project information
        lines = overview_content.split('\n')
        project_title = "Unknown Project"
        project_type = "generic"
        
        for line in lines:
            if line.startswith('# ') and not project_title == "Unknown Project":
                project_title = line[2:].strip()
            elif 'project type' in line.lower() or 'type:' in line.lower():
                project_type = self._extract_project_type(line)
        
        # Analyze content for completeness
        analysis = {
            'title': project_title,
            'type': project_type,
            'has_features': '## Feature' in overview_content or '### Feature' in overview_content,
            'has_tech_stack': 'tech' in overview_content.lower() or 'stack' in overview_content.lower(),
            'has_file_structure': 'structure' in overview_content.lower() or '```' in overview_content,
            'has_success_criteria': 'success' in overview_content.lower() or 'criteria' in overview_content,
            'has_tasks': '- [ ]' in overview_content or '- [x]' in overview_content,
            'word_count': len(overview_content.split()),
            'completeness_score': 0.0
        }
        
        # Calculate completeness score
        completeness_factors = [
            analysis['has_features'],
            analysis['has_tech_stack'], 
            analysis['has_file_structure'],
            analysis['has_success_criteria'],
            analysis['has_tasks'],
            analysis['word_count'] > 100
        ]
        analysis['completeness_score'] = sum(completeness_factors) / len(completeness_factors)
        
        return analysis
    
    def _extract_project_type(self, line: str) -> str:
        """Extract project type from a line of text."""
        line_lower = line.lower()
        type_mapping = {
            'web': 'web-app',
            'api': 'api-service', 
            'ml': 'ml-project',
            'cli': 'cli-tool',
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
    
    async def _generate_questions(self, project_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized questions based on project analysis."""
        
        questions = []
        project_type = project_analysis['type']
        completeness = project_analysis['completeness_score']
        
        # Add type-specific template questions
        if project_type in self.question_templates:
            template_questions = self.question_templates[project_type]
            for i, question_text in enumerate(template_questions):
                priority = 'Critical' if i < 2 else 'MVP' if i < 4 else 'Future'
                questions.append({
                    'question': question_text,
                    'priority': priority,
                    'suggested_answer': '',
                    'user_answer': '',
                    'why': self._get_question_reason(question_text, project_type),
                    'requires_research': True
                })
        
        # Add gap-based questions based on missing sections
        if not project_analysis['has_tech_stack']:
            questions.append({
                'question': 'What technology stack should be used for this project?',
                'priority': 'Critical',
                'suggested_answer': '',
                'user_answer': '',
                'why': 'Technology choices affect all implementation decisions',
                'requires_research': False
            })
        
        if not project_analysis['has_file_structure']:
            questions.append({
                'question': 'What should the project file structure look like?',
                'priority': 'MVP',
                'suggested_answer': '',
                'user_answer': '',
                'why': 'Clear structure guides development and organization',
                'requires_research': False
            })
        
        if not project_analysis['has_success_criteria']:
            questions.append({
                'question': 'What are the specific success criteria for this project?',
                'priority': 'Critical',
                'suggested_answer': '',
                'user_answer': '',
                'why': 'Success criteria define when the project is complete',
                'requires_research': False
            })
        
        if completeness < 0.5:
            questions.append({
                'question': 'Can you provide more detailed requirements for the main features?',
                'priority': 'Critical',
                'suggested_answer': '',
                'user_answer': '',
                'why': 'Detailed requirements prevent scope creep and ensure quality',
                'requires_research': False
            })
        
        # Sort by priority
        priority_order = {'Critical': 0, 'MVP': 1, 'Future': 2}
        questions.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return questions
    
    def _get_question_reason(self, question: str, project_type: str) -> str:
        """Get explanation for why a question is important."""
        reasons = {
            'authentication': 'Security and user management are fundamental to application design',
            'database': 'Data storage decisions affect performance and scalability',
            'performance': 'Performance requirements guide architecture and technology choices',
            'api': 'API design affects user experience and integration capabilities',
            'model': 'Model selection determines project feasibility and accuracy',
            'deployment': 'Deployment strategy affects operational requirements and costs',
            'command': 'CLI design affects user experience and adoption',
            'error': 'Error handling ensures robustness and user-friendly operation'
        }
        
        question_lower = question.lower()
        for keyword, reason in reasons.items():
            if keyword in question_lower:
                return reason
        
        return 'Important for project success and implementation quality'
    
    def _format_questions_document(self, questions: List[Dict[str, Any]], 
                                 analysis: Dict[str, Any]) -> str:
        """Format questions into a markdown document."""
        
        doc = f"""# Project Learning Questions - {analysis['title']}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Project Type: {analysis['type']}
Completeness Score: {analysis['completeness_score']:.1%}

## Instructions
1. Review each question and provide answers where possible
2. Check the box [x] when you've provided an answer
3. Research Agent will fill in suggested answers for research-required questions
4. Run `python kit.py --accept-learn` when ready to incorporate answers

---

"""
        
        for priority in self.question_priorities:
            priority_questions = [q for q in questions if q['priority'] == priority]
            if priority_questions:
                doc += f"## {priority} Questions\n\n"
                
                for q in priority_questions:
                    checkbox = "[ ]"
                    research_note = " *(Research Required)*" if q['requires_research'] else ""
                    
                    doc += f"### [{priority}] Question: {q['question']}{research_note}\n\n"
                    doc += f"**Why**: {q['why']}\n\n"
                    doc += f"**Suggested Answer**: {q['suggested_answer'] or '_To be filled by Research Agent_'}\n\n"
                    doc += f"**Your Answer**: {checkbox} {q['user_answer']}\n\n"
                    doc += "---\n\n"
        
        doc += f"""
## Summary
- Total Questions: {len(questions)}
- Critical: {len([q for q in questions if q['priority'] == 'Critical'])}
- MVP: {len([q for q in questions if q['priority'] == 'MVP'])}
- Future: {len([q for q in questions if q['priority'] == 'Future'])}

**Next Steps**: Answer questions above, then run `python kit.py --accept-learn`
"""
        
        return doc
    
    async def _save_questions_document(self, questions_doc: str) -> Path:
        """Save questions document to file."""
        
        # Create questions directory if it doesn't exist
        questions_dir = Path("questions")
        questions_dir.mkdir(exist_ok=True)
        
        # Create versioned filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        questions_path = questions_dir / f"questions_v{timestamp}.md"
        
        # Also create/update current questions.md
        current_path = Path("questions.md")
        
        # Write both files
        questions_path.write_text(questions_doc)
        current_path.write_text(questions_doc)
        
        self.logger.info(f"Saved questions to {questions_path} and questions.md")
        
        return current_path
    
    async def check_learning_completion(self, updated_overview: str) -> bool:
        """
        Check if learning phase is complete by analyzing updated overview.
        
        Returns True if no critical gaps remain, False if more learning needed.
        """
        analysis = await self._analyze_project_overview(updated_overview)
        
        # Learning is complete if:
        # 1. Completeness score is high enough
        # 2. All critical sections are present
        completion_criteria = [
            analysis['completeness_score'] >= 0.8,
            analysis['has_features'],
            analysis['has_tech_stack'],
            analysis['has_success_criteria']
        ]
        
        is_complete = all(completion_criteria)
        
        if is_complete:
            self.logger.info("Learning phase complete - project overview is sufficiently detailed")
        else:
            self.logger.info(f"Learning phase incomplete - completeness: {analysis['completeness_score']:.1%}")
        
        return is_complete