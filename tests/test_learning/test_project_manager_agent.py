"""
Tests for Project Manager Agent - Interactive Learning Phase
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.project_manager_agent import ProjectManagerAgent
from core.models import ProjectTask, AgentType


class TestProjectManagerAgent:
    """Test cases for Project Manager Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ProjectManagerAgent()
        
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test project manager agent initialization."""
        assert self.agent.agent_type == "project_manager"
        assert self.agent.max_iterations == 5
        assert len(self.agent.question_priorities) == 3
        assert "Critical" in self.agent.question_priorities
        
    def test_extract_project_type(self):
        """Test project type extraction from text."""
        test_cases = [
            ("This is a web application", "web-app"),
            ("Building an API service", "api-service"),
            ("Machine learning project", "ml-project"),
            ("CLI tool development", "cli-tool"),
            ("Generic project", "generic")
        ]
        
        for text, expected_type in test_cases:
            result = self.agent._extract_project_type(text)
            assert result == expected_type
    
    @pytest.mark.asyncio
    async def test_analyze_project_overview(self):
        """Test project overview analysis."""
        overview_content = """
        # Test Web Application
        
        ## Technology Stack
        - Frontend: React
        - Backend: Python FastAPI
        
        ## Features
        ### Feature 1: User Authentication
        - [ ] Login functionality
        - [ ] Registration
        
        ## Success Criteria
        - All tests pass
        - Application runs without errors
        """
        
        analysis = await self.agent._analyze_project_overview(overview_content)
        
        assert analysis['type'] == 'web-app'
        assert analysis['has_features'] is True
        assert analysis['has_tech_stack'] is True
        assert analysis['has_success_criteria'] is True
        assert analysis['has_tasks'] is True
        assert analysis['completeness_score'] > 0.5
    
    @pytest.mark.asyncio
    async def test_generate_questions_web_app(self):
        """Test question generation for web applications."""
        project_analysis = {
            'title': 'Test Web App',
            'type': 'web-app',
            'has_features': True,
            'has_tech_stack': True,
            'has_file_structure': False,
            'has_success_criteria': True,
            'has_tasks': True,
            'completeness_score': 0.7
        }
        
        questions = await self.agent._generate_questions(project_analysis)
        
        assert len(questions) > 0
        
        # Check that web-app specific questions are included
        question_texts = [q['question'] for q in questions]
        assert any('authentication' in q.lower() for q in question_texts)
        
        # Check priority distribution
        priorities = [q['priority'] for q in questions]
        assert 'Critical' in priorities
        assert 'MVP' in priorities or 'Future' in priorities
        
        # Check that missing file structure generated a question
        file_structure_questions = [q for q in questions if 'structure' in q['question'].lower()]
        assert len(file_structure_questions) > 0
    
    @pytest.mark.asyncio 
    async def test_execute_success(self):
        """Test successful execution of project manager agent."""
        overview_content = """
        # Simple CLI Tool
        
        ## Description
        A command-line tool for processing text files.
        
        ## Features
        - Read files
        - Process content
        - Output results
        """
        
        task = ProjectTask(
            title="Analyze CLI Tool Project",
            description=overview_content,
            type="CREATE",
            agent_type=AgentType.ADVISOR
        )
        
        with patch.object(self.agent, '_save_questions_document', 
                         return_value=AsyncMock(return_value=Path("questions.md"))):
            result = await self.agent.execute(task)
        
        assert result.success is True
        assert 'questions' in result.output
        assert 'project_analysis' in result.output
        assert result.output['total_questions'] > 0
        assert result.confidence > 0.8
    
    def test_format_questions_document(self):
        """Test questions document formatting."""
        questions = [
            {
                'question': 'What authentication method should be used?',
                'priority': 'Critical',
                'suggested_answer': 'JWT tokens',
                'user_answer': '',
                'why': 'Security is fundamental',
                'requires_research': True
            },
            {
                'question': 'What database will be used?',
                'priority': 'MVP',
                'suggested_answer': '',
                'user_answer': '',
                'why': 'Data storage affects architecture',
                'requires_research': False
            }
        ]
        
        analysis = {
            'title': 'Test Project',
            'type': 'web-app',
            'completeness_score': 0.6
        }
        
        doc = self.agent._format_questions_document(questions, analysis)
        
        assert '# Project Learning Questions - Test Project' in doc
        assert 'Project Type: web-app' in doc
        assert 'Completeness Score: 60%' in doc
        assert '## Critical Questions' in doc
        assert '## MVP Questions' in doc
        assert '[Critical] Question: What authentication method should be used?' in doc
        assert '**Why**: Security is fundamental' in doc
        assert '*(Research Required)*' in doc
        assert 'python kit.py --accept-learn' in doc
    
    @pytest.mark.asyncio
    async def test_check_learning_completion_complete(self):
        """Test learning completion check for complete overview."""
        complete_overview = """
        # Complete Project
        
        ## Technology Stack
        - Frontend: React
        - Backend: FastAPI
        
        ## Features
        ### Feature 1: Authentication
        - Login system
        
        ## Success Criteria
        - All tests pass
        - 90% code coverage
        
        ## File Structure
        project/
        ├── src/
        └── tests/
        """
        
        is_complete = await self.agent.check_learning_completion(complete_overview)
        assert is_complete is True
    
    @pytest.mark.asyncio
    async def test_check_learning_completion_incomplete(self):
        """Test learning completion check for incomplete overview."""
        incomplete_overview = """
        # Incomplete Project
        
        Basic description only.
        """
        
        is_complete = await self.agent.check_learning_completion(incomplete_overview)
        assert is_complete is False
    
    def test_get_question_reason(self):
        """Test question reasoning generation."""
        test_cases = [
            ("What authentication method?", "authentication"),
            ("Which database to use?", "database"),
            ("How to handle errors?", "error"),
            ("What are performance requirements?", "performance")
        ]
        
        for question, expected_keyword in test_cases:
            reason = self.agent._get_question_reason(question, "web-app")
            assert len(reason) > 0
            assert isinstance(reason, str)
    
    @pytest.mark.asyncio
    async def test_execute_failure_handling(self):
        """Test execution failure handling."""
        task = ProjectTask(
            title="Invalid Task",
            description="",  # Empty description should cause validation error
            type="CREATE",
            agent_type=AgentType.ADVISOR
        )
        
        result = await self.agent.execute(task)
        
        assert result.success is False
        assert result.error is not None
        assert result.confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__])