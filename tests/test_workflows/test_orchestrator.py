"""
Tests for OpenAI o3 Orchestrator.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from core.orchestrator import O3Orchestrator
from core.models import ProjectSpecification, ProjectTask, AgentType


class TestO3Orchestrator:
    """Test cases for O3Orchestrator."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.api_keys = Mock()
        config.api_keys.has_openai.return_value = True
        config.api_keys.openai_api_key = "test-key"
        return config
    
    @pytest.fixture
    def sample_project_spec(self):
        """Sample project specification."""
        task1 = ProjectTask(
            id="task-1",
            title="Parse Requirements",
            description="Parse project requirements",
            type="CREATE",
            agent_type=AgentType.PARSER,
            dependencies=[]
        )
        
        task2 = ProjectTask(
            id="task-2",
            title="Generate Code",
            description="Generate application code",
            type="CREATE",
            agent_type=AgentType.CODER,
            dependencies=["task-1"]
        )
        
        task3 = ProjectTask(
            id="task-3",
            title="Run Tests",
            description="Execute test suite",
            type="VALIDATE",
            agent_type=AgentType.TESTER,
            dependencies=["task-2"]
        )
        
        return ProjectSpecification(
            title="Test Project",
            description="A test project",
            project_type="web",
            tasks=[task1, task2, task3],
            requirements={"framework": "react"},
            validation_criteria={"coverage": 0.8}
        )
    
    @pytest.fixture
    def orchestrator(self, mock_config):
        """Create orchestrator instance."""
        with patch('core.orchestrator.get_config', return_value=mock_config):
            with patch('core.orchestrator.get_logger'):
                with patch('core.orchestrator.get_voice_alerts'):
                    with patch('openai.OpenAI'):
                        return O3Orchestrator()
    
    @pytest.mark.asyncio
    async def test_create_execution_plan_success(self, orchestrator, sample_project_spec):
        """Test successful execution plan creation."""
        # Mock OpenAI response
        mock_response = {
            "execution_order": [
                {
                    "task_id": "task-1",
                    "priority": "high",
                    "dependencies": [],
                    "parallel_group": None,
                    "risk_level": "low",
                    "confidence_threshold": 0.8,
                    "estimated_duration": 300,
                    "resource_requirements": {"cpu": "medium", "memory": "low"},
                    "fallback_strategy": "retry with increased timeout"
                },
                {
                    "task_id": "task-2",
                    "priority": "high",
                    "dependencies": ["task-1"],
                    "parallel_group": None,
                    "risk_level": "medium",
                    "confidence_threshold": 0.8,
                    "estimated_duration": 600,
                    "resource_requirements": {"cpu": "high", "memory": "medium"},
                    "fallback_strategy": "retry with increased timeout"
                },
                {
                    "task_id": "task-3",
                    "priority": "medium",
                    "dependencies": ["task-2"],
                    "parallel_group": None,
                    "risk_level": "low",
                    "confidence_threshold": 0.9,
                    "estimated_duration": 300,
                    "resource_requirements": {"cpu": "medium", "memory": "low"},
                    "fallback_strategy": "retry with increased timeout"
                }
            ],
            "parallel_groups": {},
            "critical_path": ["task-1", "task-2", "task-3"],
            "risk_mitigation": {
                "high_risk_tasks": [],
                "mitigation_strategies": []
            },
            "estimated_total_duration": 1200,
            "optimization_notes": "Sequential execution based on dependencies"
        }
        
        with patch.object(orchestrator, '_make_o3_request', return_value=json.dumps(mock_response)):
            plan = await orchestrator.create_execution_plan(sample_project_spec)
            
            assert "execution_order" in plan
            assert len(plan["execution_order"]) == 3
            assert plan["execution_order"][0]["task_id"] == "task-1"
            assert plan["execution_order"][1]["task_id"] == "task-2"
            assert plan["execution_order"][2]["task_id"] == "task-3"
            assert "metadata" in plan
    
    @pytest.mark.asyncio
    async def test_create_execution_plan_fallback(self, orchestrator, sample_project_spec):
        """Test fallback plan creation when o3 fails."""
        with patch.object(orchestrator, '_make_o3_request', side_effect=Exception("API Error")):
            plan = await orchestrator.create_execution_plan(sample_project_spec)
            
            assert "execution_order" in plan
            assert len(plan["execution_order"]) == 3
            assert plan["optimization_notes"] == "Fallback plan - simple sequential execution"
            assert plan["metadata"]["model"] == "fallback"
    
    @pytest.mark.asyncio
    async def test_adapt_plan_during_execution(self, orchestrator, sample_project_spec):
        """Test plan adaptation during execution."""
        current_plan = {
            "execution_order": [
                {"task_id": "task-1", "dependencies": []},
                {"task_id": "task-2", "dependencies": ["task-1"]},
                {"task_id": "task-3", "dependencies": ["task-2"]}
            ],
            "parallel_groups": {},
            "critical_path": ["task-1", "task-2", "task-3"]
        }
        
        failed_tasks = ["task-2"]
        agent_results = [
            {"task_id": "task-1", "success": True, "confidence": 0.9},
            {"task_id": "task-2", "success": False, "confidence": 0.3}
        ]
        
        adapted_response = {
            "execution_order": [
                {"task_id": "task-1", "dependencies": []},
                {"task_id": "task-2-retry", "dependencies": ["task-1"]},
                {"task_id": "task-3", "dependencies": ["task-2-retry"]}
            ],
            "parallel_groups": {},
            "critical_path": ["task-1", "task-2-retry", "task-3"]
        }
        
        with patch.object(orchestrator, '_make_o3_request', return_value=json.dumps(adapted_response)):
            adapted_plan = await orchestrator.adapt_plan_during_execution(
                current_plan, failed_tasks, agent_results
            )
            
            assert "execution_order" in adapted_plan
            assert "metadata" in adapted_plan
            assert "adapted_at" in adapted_plan["metadata"]
    
    def test_has_circular_dependencies_none(self, orchestrator):
        """Test circular dependency detection with no cycles."""
        execution_order = [
            {"task_id": "task-1", "dependencies": []},
            {"task_id": "task-2", "dependencies": ["task-1"]},
            {"task_id": "task-3", "dependencies": ["task-2"]}
        ]
        
        assert not orchestrator._has_circular_dependencies(execution_order)
    
    def test_has_circular_dependencies_simple_cycle(self, orchestrator):
        """Test circular dependency detection with simple cycle."""
        execution_order = [
            {"task_id": "task-1", "dependencies": ["task-2"]},
            {"task_id": "task-2", "dependencies": ["task-1"]}
        ]
        
        assert orchestrator._has_circular_dependencies(execution_order)
    
    def test_has_circular_dependencies_complex_cycle(self, orchestrator):
        """Test circular dependency detection with complex cycle."""
        execution_order = [
            {"task_id": "task-1", "dependencies": ["task-3"]},
            {"task_id": "task-2", "dependencies": ["task-1"]},
            {"task_id": "task-3", "dependencies": ["task-2"]}
        ]
        
        assert orchestrator._has_circular_dependencies(execution_order)
    
    def test_calculate_execution_time(self, orchestrator):
        """Test execution time calculation."""
        plan = {
            "execution_order": [
                {"task_id": "task-1", "dependencies": [], "estimated_duration": 300},
                {"task_id": "task-2", "dependencies": ["task-1"], "estimated_duration": 600},
                {"task_id": "task-3", "dependencies": ["task-2"], "estimated_duration": 300}
            ],
            "parallel_groups": {}
        }
        
        total_time = orchestrator._calculate_execution_time(plan)
        assert total_time == 1200  # Sequential execution
    
    def test_resolve_circular_dependencies(self, orchestrator):
        """Test circular dependency resolution."""
        plan = {
            "execution_order": [
                {"task_id": "task-1", "dependencies": ["task-2"]},
                {"task_id": "task-2", "dependencies": ["task-1"]}
            ]
        }
        
        resolved_plan = orchestrator._resolve_circular_dependencies(plan)
        
        # Should remove at least one dependency to break the cycle
        deps_1 = resolved_plan["execution_order"][0]["dependencies"]
        deps_2 = resolved_plan["execution_order"][1]["dependencies"]
        
        assert not (deps_1 and deps_2)  # At least one should be empty
    
    def test_get_orchestrator_stats(self, orchestrator):
        """Test orchestrator statistics."""
        stats = orchestrator.get_orchestrator_stats()
        
        assert "request_count" in stats
        assert "total_tokens" in stats
        assert "estimated_cost" in stats
        assert "model" in stats
        assert "reasoning_effort" in stats
    
    def test_reset_stats(self, orchestrator):
        """Test statistics reset."""
        # Set some stats
        orchestrator.request_count = 5
        orchestrator.total_tokens = 1000
        orchestrator.total_cost = 0.05
        
        orchestrator.reset_stats()
        
        assert orchestrator.request_count == 0
        assert orchestrator.total_tokens == 0
        assert orchestrator.total_cost == 0.0