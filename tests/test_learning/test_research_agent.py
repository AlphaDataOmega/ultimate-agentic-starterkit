"""
Tests for Research Agent - Interactive Learning Phase
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import subprocess

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.research_agent import ResearchAgent
from core.models import ProjectTask, AgentType


class TestResearchAgent:
    """Test cases for Research Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ResearchAgent()
        
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test research agent initialization."""
        assert self.agent.agent_type == "research"
        assert self.agent.ollama_model == 'mistral:7b'
        assert self.agent.max_research_attempts == 3
        assert len(self.agent.knowledge_base) > 0
        
    def test_categorize_question(self):
        """Test question categorization."""
        test_cases = [
            ("What authentication method should be used?", "authentication"),
            ("Which database should we choose?", "database"),
            ("What technology stack is recommended?", "tech_stack"),
            ("How should we handle API endpoints?", "api"),
            ("What are the performance requirements?", "performance"),
            ("How should we deploy this?", "deployment"),
            ("What testing strategy should we use?", "testing"),
            ("Random question about something", "general")
        ]
        
        for question, expected_category in test_cases:
            result = self.agent._categorize_question(question)
            assert result == expected_category
    
    def test_get_knowledge_base_answer(self):
        """Test knowledge base answer retrieval."""
        # Test authentication question
        auth_answer = self.agent._get_knowledge_base_answer(
            "What authentication method?", "authentication"
        )
        assert auth_answer is not None
        assert len(auth_answer) > 0
        
        # Test database question
        db_answer = self.agent._get_knowledge_base_answer(
            "What database?", "database"
        )
        assert db_answer is not None
        assert len(db_answer) > 0
        
        # Test unknown category
        unknown_answer = self.agent._get_knowledge_base_answer(
            "Unknown question", "unknown_category"
        )
        assert unknown_answer is None
    
    def test_parse_questions_document(self):
        """Test parsing of questions markdown document."""
        questions_content = """
# Project Learning Questions - Test Project

## Critical Questions

### [Critical] Question: What authentication method should be used? *(Research Required)*

**Why**: Security is fundamental

**Suggested Answer**: JWT tokens with refresh tokens

**Your Answer**: [ ] 

---

### [Critical] Question: What database should be used?

**Why**: Data storage affects architecture

**Suggested Answer**: PostgreSQL for ACID compliance

**Your Answer**: [x] We'll use PostgreSQL with Redis for caching

---

## MVP Questions

### [MVP] Question: What frontend framework?

**Why**: UI development approach

**Suggested Answer**: _To be filled by Research Agent_

**Your Answer**: [ ] 

---
"""
        
        questions = self.agent._parse_questions_document(questions_content)
        
        assert len(questions) == 3
        
        # Check first question
        q1 = questions[0]
        assert q1['priority'] == 'Critical'
        assert q1['question'] == 'What authentication method should be used?'
        assert q1['requires_research'] is True
        assert q1['why'] == 'Security is fundamental'
        assert q1['suggested_answer'] == 'JWT tokens with refresh tokens'
        assert q1['user_answer'] == ''
        
        # Check second question (has user answer)
        q2 = questions[1]
        assert q2['priority'] == 'Critical'
        assert q2['question'] == 'What database should be used?'
        assert q2['requires_research'] is False
        assert q2['user_answer'] == "We'll use PostgreSQL with Redis for caching"
        
        # Check third question
        q3 = questions[2]
        assert q3['priority'] == 'MVP'
        assert q3['question'] == 'What frontend framework?'
        assert q3['suggested_answer'] == ''
    
    @pytest.mark.asyncio
    async def test_research_question_knowledge_base(self):
        """Test research using knowledge base."""
        question = {
            'question': 'What authentication method should be used?',
            'priority': 'Critical',
            'requires_research': True
        }
        
        result = await self.agent._research_question(question)
        
        assert result['question'] == question['question']
        assert result['method'] == 'knowledge_base'
        assert result['confidence'] == 0.8
        assert len(result['answer']) > 0
        assert 'JWT' in result['answer'] or 'OAuth' in result['answer']
    
    @pytest.mark.asyncio
    async def test_research_with_ollama_not_available(self):
        """Test Ollama research when Ollama is not available."""
        with patch('subprocess.run') as mock_run:
            # Mock Ollama not available
            mock_run.return_value.returncode = 1
            
            result = await self.agent._research_with_ollama("Test question", "general")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_research_with_ollama_success(self):
        """Test successful Ollama research."""
        with patch('subprocess.run') as mock_run:
            # Mock successful Ollama list command
            mock_run.side_effect = [
                MagicMock(returncode=0),  # ollama list
                MagicMock(returncode=0, stdout="This is a test answer from Ollama.")  # ollama run
            ]
            
            result = await self.agent._research_with_ollama("Test question", "general")
            
            assert result is not None
            assert result['method'] == 'ollama'
            assert result['confidence'] == 0.7
            assert result['answer'] == "This is a test answer from Ollama."
    
    @pytest.mark.asyncio
    async def test_interactive_research(self):
        """Test interactive research with user input."""
        with patch('builtins.input', return_value='User provided answer'):
            with patch.object(self.agent.voice, 'speak') as mock_speak:
                result = await self.agent._interactive_research("Test question?")
                
                assert result['question'] == "Test question?"
                assert result['method'] == 'interactive'
                assert result['confidence'] == 0.9
                assert result['answer'] == 'User provided answer'
                mock_speak.assert_called()
    
    @pytest.mark.asyncio
    async def test_interactive_research_skip(self):
        """Test interactive research when user skips."""
        with patch('builtins.input', return_value='skip'):
            result = await self.agent._interactive_research("Test question?")
            
            assert result['method'] == 'interactive'
            assert result['confidence'] == 0.2
            assert 'User input required' in result['answer']
    
    def test_update_questions_document(self):
        """Test updating questions document with research results."""
        original_content = """### [Critical] Question: What database?

**Suggested Answer**: _To be filled by Research Agent_

**Your Answer**: [ ] 

---

### [MVP] Question: What frontend framework?

**Suggested Answer**: _To be filled by Research Agent_

**Your Answer**: [ ] """
        
        questions = [
            {
                'question': 'What database?',
                'suggested_answer': 'PostgreSQL for production apps',
                'research_confidence': 0.8
            },
            {
                'question': 'What frontend framework?',
                'suggested_answer': 'React for component-based development',
                'research_confidence': 0.7
            }
        ]
        
        updated_content = self.agent._update_questions_document(original_content, questions)
        
        assert 'PostgreSQL for production apps *(Confidence: 80%)*' in updated_content
        assert 'React for component-based development *(Confidence: 70%)*' in updated_content
        assert '_To be filled by Research Agent_' not in updated_content
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution of research agent."""
        # Create a temporary questions file
        questions_content = """### [Critical] Question: What authentication? *(Research Required)*

**Suggested Answer**: _To be filled by Research Agent_

**Your Answer**: [ ] """
        
        questions_file = Path("test_questions.md")
        questions_file.write_text(questions_content)
        
        try:
            task = ProjectTask(
                title="Research Questions",
                description="test_questions.md",
                type="MODIFY",
                agent_type=AgentType.ADVISOR
            )
            
            result = await self.agent.execute(task)
            
            assert result.success is True
            assert 'research_results' in result.output
            assert result.output['questions_researched'] >= 0
            assert result.confidence > 0.5
            
        finally:
            # Clean up
            if questions_file.exists():
                questions_file.unlink()
    
    @pytest.mark.asyncio
    async def test_execute_file_not_found(self):
        """Test execution when questions file not found."""
        task = ProjectTask(
            title="Research Missing File",
            description="nonexistent_questions.md",
            type="MODIFY",
            agent_type=AgentType.ADVISOR
        )
        
        # Should fall back to questions.md
        with patch('pathlib.Path.exists', return_value=False):
            result = await self.agent.execute(task)
            
            assert result.success is False
            assert 'error' in result.error.lower() or 'not found' in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_quick_research(self):
        """Test quick research functionality."""
        # Test knowledge base answer
        kb_result = await self.agent.quick_research("What authentication method?")
        assert len(kb_result) > 0
        
        # Test with interactive fallback
        with patch('builtins.input', return_value='Quick answer'):
            with patch.object(self.agent.voice, 'speak'):
                interactive_result = await self.agent.quick_research("Unknown question?")
                assert interactive_result == 'Quick answer'


if __name__ == "__main__":
    pytest.main([__file__])