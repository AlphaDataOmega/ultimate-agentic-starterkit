"""
Unit tests for confidence scoring engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from validation.confidence import ConfidenceScorer, ConfidenceFactors
from core.models import AgentResult, ProjectTask, AgentType, create_agent_result, create_project_task


class TestConfidenceFactors:
    """Test ConfidenceFactors dataclass"""
    
    def test_confidence_factors_initialization(self):
        """Test ConfidenceFactors initialization"""
        factors = ConfidenceFactors()
        
        assert factors.syntax_correctness == 0.0
        assert factors.type_safety == 0.0
        assert factors.test_coverage == 0.0
        assert factors.code_quality == 0.0
        assert factors.performance == 0.0
        assert factors.completeness == 0.0
        assert factors.agent_self_confidence == 0.0
    
    def test_confidence_factors_with_values(self):
        """Test ConfidenceFactors with custom values"""
        factors = ConfidenceFactors(
            syntax_correctness=0.9,
            type_safety=0.8,
            test_coverage=0.7,
            code_quality=0.85,
            performance=0.75,
            completeness=0.9,
            agent_self_confidence=0.8
        )
        
        assert factors.syntax_correctness == 0.9
        assert factors.type_safety == 0.8
        assert factors.test_coverage == 0.7
        assert factors.code_quality == 0.85
        assert factors.performance == 0.75
        assert factors.completeness == 0.9
        assert factors.agent_self_confidence == 0.8


class TestConfidenceScorer:
    """Test ConfidenceScorer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.scorer = ConfidenceScorer()
        
        # Mock agent result
        self.agent_result = create_agent_result(
            success=True,
            confidence=0.8,
            output="test output",
            agent_id="test-agent",
            execution_time=1.0
        )
        
        # Mock project task
        self.project_task = create_project_task(
            title="Test Task",
            description="Test description",
            task_type="CREATE",
            agent_type=AgentType.CODER
        )
    
    def test_scorer_initialization(self):
        """Test ConfidenceScorer initialization"""
        assert self.scorer.logger is not None
        assert self.scorer.voice is not None
        assert len(self.scorer.factor_weights) == 5
        assert AgentType.CODER in self.scorer.factor_weights
        assert AgentType.PARSER in self.scorer.factor_weights
        assert AgentType.TESTER in self.scorer.factor_weights
        assert AgentType.ADVISOR in self.scorer.factor_weights
        assert AgentType.ORCHESTRATOR in self.scorer.factor_weights
    
    def test_factor_weights_structure(self):
        """Test factor weights structure"""
        coder_weights = self.scorer.factor_weights[AgentType.CODER]
        
        assert "syntax_correctness" in coder_weights
        assert "type_safety" in coder_weights
        assert "code_quality" in coder_weights
        assert "completeness" in coder_weights
        assert "agent_self_confidence" in coder_weights
        
        # Verify weights sum to reasonable value
        total_weight = sum(coder_weights.values())
        assert 0.8 <= total_weight <= 1.2  # Allow some flexibility
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_success(self):
        """Test successful confidence calculation"""
        with patch.object(self.scorer, '_extract_confidence_factors') as mock_extract:
            mock_factors = ConfidenceFactors(
                syntax_correctness=0.9,
                type_safety=0.8,
                code_quality=0.85,
                completeness=0.9,
                agent_self_confidence=0.8
            )
            mock_extract.return_value = mock_factors
            
            confidence = await self.scorer.calculate_confidence(
                self.agent_result,
                self.project_task
            )
            
            assert 0.0 <= confidence <= 1.0
            assert confidence > 0.7  # Should be reasonably high
            mock_extract.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_with_penalties(self):
        """Test confidence calculation with penalties"""
        with patch.object(self.scorer, '_extract_confidence_factors') as mock_extract:
            mock_factors = ConfidenceFactors(
                syntax_correctness=0.2,  # Low syntax correctness
                type_safety=0.4,  # Low type safety
                code_quality=0.9,
                completeness=0.9,
                agent_self_confidence=0.8
            )
            mock_extract.return_value = mock_factors
            
            confidence = await self.scorer.calculate_confidence(
                self.agent_result,
                self.project_task
            )
            
            assert 0.0 <= confidence <= 1.0
            assert confidence < 0.5  # Should be penalized
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_exception_handling(self):
        """Test confidence calculation exception handling"""
        with patch.object(self.scorer, '_extract_confidence_factors') as mock_extract:
            mock_extract.side_effect = Exception("Test error")
            
            confidence = await self.scorer.calculate_confidence(
                self.agent_result,
                self.project_task
            )
            
            assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_extract_confidence_factors_coder(self):
        """Test confidence factors extraction for coder agent"""
        coder_task = create_project_task(
            title="Coder Task",
            description="Test coder task",
            task_type="CREATE",
            agent_type=AgentType.CODER
        )
        
        with patch.object(self.scorer, '_check_syntax_correctness') as mock_syntax, \
             patch.object(self.scorer, '_check_type_safety') as mock_type, \
             patch.object(self.scorer, '_check_code_quality') as mock_quality, \
             patch.object(self.scorer, '_check_completeness') as mock_completeness:
            
            mock_syntax.return_value = 0.9
            mock_type.return_value = 0.8
            mock_quality.return_value = 0.85
            mock_completeness.return_value = 0.9
            
            factors = await self.scorer._extract_confidence_factors(
                self.agent_result,
                coder_task
            )
            
            assert factors.syntax_correctness == 0.9
            assert factors.type_safety == 0.8
            assert factors.code_quality == 0.85
            assert factors.completeness == 0.9
            assert factors.agent_self_confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_extract_confidence_factors_parser(self):
        """Test confidence factors extraction for parser agent"""
        parser_task = create_project_task(
            title="Parser Task",
            description="Test parser task",
            task_type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        with patch.object(self.scorer, '_check_completeness') as mock_completeness:
            mock_completeness.return_value = 0.9
            
            factors = await self.scorer._extract_confidence_factors(
                self.agent_result,
                parser_task
            )
            
            assert factors.completeness == 0.9
            assert factors.agent_self_confidence == 0.8
            # Other factors should be 0.0 for parser
            assert factors.syntax_correctness == 0.0
            assert factors.type_safety == 0.0
    
    @pytest.mark.asyncio
    async def test_check_syntax_correctness_valid_code(self):
        """Test syntax correctness check with valid code"""
        valid_code = "def hello():\n    print('Hello, World!')"
        
        result = await self.scorer._check_syntax_correctness(valid_code)
        
        assert result == 1.0
    
    @pytest.mark.asyncio
    async def test_check_syntax_correctness_invalid_code(self):
        """Test syntax correctness check with invalid code"""
        invalid_code = "def hello(\n    print('Hello, World!')"
        
        result = await self.scorer._check_syntax_correctness(invalid_code)
        
        assert result == 0.0
    
    @pytest.mark.asyncio
    async def test_check_syntax_correctness_no_code(self):
        """Test syntax correctness check with no code"""
        result = await self.scorer._check_syntax_correctness("")
        
        assert result == 0.5
    
    @pytest.mark.asyncio
    async def test_check_type_safety_success(self):
        """Test type safety check success"""
        valid_code = "def hello(name: str) -> str:\n    return f'Hello, {name}!'"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="")
            
            result = await self.scorer._check_type_safety(valid_code)
            
            assert result == 1.0
    
    @pytest.mark.asyncio
    async def test_check_type_safety_with_errors(self):
        """Test type safety check with errors"""
        invalid_code = "def hello(name: str) -> str:\n    return 42"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="error: error: error:")
            
            result = await self.scorer._check_type_safety(invalid_code)
            
            assert result == 0.7  # 1.0 - (3 errors * 0.1)
    
    @pytest.mark.asyncio
    async def test_check_code_quality_success(self):
        """Test code quality check success"""
        clean_code = "def hello(name: str) -> str:\n    return f'Hello, {name}!'"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="")
            
            result = await self.scorer._check_code_quality(clean_code)
            
            assert result == 1.0
    
    @pytest.mark.asyncio
    async def test_check_code_quality_with_issues(self):
        """Test code quality check with issues"""
        messy_code = "def hello():\n    print('test')"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="issue1\nissue2\nissue3")
            
            result = await self.scorer._check_code_quality(messy_code)
            
            assert result == 0.85  # 1.0 - (3 issues * 0.05)
    
    def test_extract_code_from_output_string(self):
        """Test code extraction from string output"""
        code = "def hello():\n    print('Hello')"
        
        result = self.scorer._extract_code_from_output(code)
        
        assert result == code
    
    def test_extract_code_from_output_dict(self):
        """Test code extraction from dict output"""
        output = {
            "response": "def hello():\n    print('Hello')",
            "metadata": {}
        }
        
        result = self.scorer._extract_code_from_output(output)
        
        assert result == "def hello():\n    print('Hello')"
    
    def test_extract_code_from_output_code_field(self):
        """Test code extraction from output with code field"""
        output = {
            "code": "def hello():\n    print('Hello')",
            "metadata": {}
        }
        
        result = self.scorer._extract_code_from_output(output)
        
        assert result == "def hello():\n    print('Hello')"
    
    def test_extract_code_from_output_empty(self):
        """Test code extraction from empty output"""
        result = self.scorer._extract_code_from_output(None)
        
        assert result == ""
    
    def test_apply_confidence_penalties(self):
        """Test confidence penalties application"""
        factors = ConfidenceFactors(
            syntax_correctness=0.2,  # Severe penalty
            type_safety=0.4,  # Penalty
            test_coverage=0.5,  # Penalty
            completeness=0.4  # Penalty
        )
        
        result = self.scorer._apply_confidence_penalties(1.0, factors)
        
        # Should have penalties: 0.3 + 0.2 + 0.1 + 0.15 = 0.75
        assert result == 0.25
    
    def test_apply_confidence_penalties_minimum(self):
        """Test confidence penalties don't go below 0"""
        factors = ConfidenceFactors(
            syntax_correctness=0.0,  # Max penalty
            type_safety=0.0,  # Max penalty
            test_coverage=0.0,  # Max penalty
            completeness=0.0  # Max penalty
        )
        
        result = self.scorer._apply_confidence_penalties(0.5, factors)
        
        assert result == 0.0
    
    def test_get_confidence_breakdown(self):
        """Test confidence breakdown generation"""
        factors = ConfidenceFactors(
            syntax_correctness=0.9,
            type_safety=0.8,
            code_quality=0.85,
            completeness=0.9,
            agent_self_confidence=0.8
        )
        
        breakdown = self.scorer.get_confidence_breakdown(factors, AgentType.CODER)
        
        assert "factors" in breakdown
        assert "weights" in breakdown
        assert "weighted_scores" in breakdown
        
        assert breakdown["factors"]["syntax_correctness"] == 0.9
        assert breakdown["factors"]["type_safety"] == 0.8
        
        # Check that weighted scores are calculated
        assert "syntax_correctness" in breakdown["weighted_scores"]
        assert "type_safety" in breakdown["weighted_scores"]
    
    def test_check_parser_completeness_success(self):
        """Test parser completeness check success"""
        output = {
            "milestones": [
                {
                    "title": "Milestone 1",
                    "description": "Description 1",
                    "tasks": ["task1", "task2"]
                },
                {
                    "title": "Milestone 2",
                    "description": "Description 2",
                    "tasks": ["task3", "task4"]
                }
            ]
        }
        
        result = self.scorer._check_parser_completeness(output, self.project_task)
        
        assert result == 1.0
    
    def test_check_parser_completeness_partial(self):
        """Test parser completeness check with partial completion"""
        output = {
            "milestones": [
                {
                    "title": "Milestone 1",
                    "description": "Description 1",
                    "tasks": ["task1", "task2"]
                },
                {
                    "title": "Milestone 2",
                    # Missing description
                    "tasks": ["task3", "task4"]
                }
            ]
        }
        
        result = self.scorer._check_parser_completeness(output, self.project_task)
        
        assert result == 0.5  # 1 complete out of 2 milestones
    
    def test_check_coder_completeness_success(self):
        """Test coder completeness check success"""
        output = """
        import os
        from typing import List
        
        def hello(name: str) -> str:
            '''
            Say hello to someone.
            
            Args:
                name: Person's name
                
            Returns:
                Greeting string
            '''
            try:
                return f'Hello, {name}!'
            except Exception as e:
                return f'Error: {e}'
        """
        
        result = self.scorer._check_coder_completeness(output, self.project_task)
        
        assert result == 1.0  # Should have all components
    
    def test_check_coder_completeness_minimal(self):
        """Test coder completeness check with minimal code"""
        output = "print('hello')"
        
        result = self.scorer._check_coder_completeness(output, self.project_task)
        
        assert result == 0.0  # Very minimal code
    
    def test_check_tester_completeness_success(self):
        """Test tester completeness check success"""
        output = {
            "test_results": {
                "passed": 10,
                "failed": 0,
                "total": 10
            }
        }
        
        result = self.scorer._check_tester_completeness(output, self.project_task)
        
        assert result == 1.0
    
    def test_check_tester_completeness_with_coverage(self):
        """Test tester completeness check with coverage"""
        output = {
            "coverage": 85.0
        }
        
        result = self.scorer._check_tester_completeness(output, self.project_task)
        
        assert result == 0.8
    
    def test_check_advisor_completeness_success(self):
        """Test advisor completeness check success"""
        output = {
            "recommendations": [
                "Improve error handling",
                "Add more tests",
                "Optimize performance"
            ]
        }
        
        result = self.scorer._check_advisor_completeness(output, self.project_task)
        
        assert result == 1.0
    
    def test_check_orchestrator_completeness_success(self):
        """Test orchestrator completeness check success"""
        output = {
            "tasks": [
                {"id": "task1", "type": "CREATE"},
                {"id": "task2", "type": "TEST"}
            ]
        }
        
        result = self.scorer._check_orchestrator_completeness(output, self.project_task)
        
        assert result == 1.0