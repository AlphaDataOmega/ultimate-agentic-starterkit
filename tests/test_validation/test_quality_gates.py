"""
Unit tests for quality gates system.
"""

import pytest
import asyncio
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from validation.quality_gates import (
    QualityGate, QualityGateSystem, QualityGateStatus, QualityGateResult,
    ConfidenceGate, SyntaxQualityGate, TypeSafetyQualityGate, TestCoverageQualityGate,
    PerformanceQualityGate, CustomQualityGate
)
from validation.confidence import ConfidenceFactors


class TestQualityGateResult:
    """Test QualityGateResult dataclass"""
    
    def test_quality_gate_result_creation(self):
        """Test QualityGateResult creation"""
        result = QualityGateResult(
            gate_name="test_gate",
            status=QualityGateStatus.PASSED,
            score=0.85,
            threshold=0.8,
            message="Test passed",
            details={"info": "test"}
        )
        
        assert result.gate_name == "test_gate"
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 0.85
        assert result.threshold == 0.8
        assert result.message == "Test passed"
        assert result.details == {"info": "test"}
        assert result.timestamp is not None
        assert result.execution_time == 0.0
    
    def test_quality_gate_result_defaults(self):
        """Test QualityGateResult with defaults"""
        result = QualityGateResult(
            gate_name="test_gate",
            status=QualityGateStatus.PASSED,
            score=0.85,
            threshold=0.8,
            message="Test passed"
        )
        
        assert result.details == {}
        assert result.execution_time == 0.0
        assert result.timestamp is not None


class TestQualityGateStatus:
    """Test QualityGateStatus enum"""
    
    def test_quality_gate_status_values(self):
        """Test QualityGateStatus enum values"""
        assert QualityGateStatus.PASSED == "passed"
        assert QualityGateStatus.FAILED == "failed"
        assert QualityGateStatus.WARNING == "warning"
        assert QualityGateStatus.BLOCKED == "blocked"
        assert QualityGateStatus.SKIPPED == "skipped"


class TestConfidenceGate:
    """Test ConfidenceGate implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gate = ConfidenceGate("confidence_gate", 0.8, 1.0)
    
    def test_confidence_gate_initialization(self):
        """Test ConfidenceGate initialization"""
        assert self.gate.name == "confidence_gate"
        assert self.gate.threshold == 0.8
        assert self.gate.weight == 1.0
        assert self.gate.enabled is True
        assert self.gate.blocking is True
    
    @pytest.mark.asyncio
    async def test_confidence_gate_success(self):
        """Test ConfidenceGate success case"""
        factors = ConfidenceFactors(
            syntax_correctness=0.9,
            type_safety=0.8,
            code_quality=0.85,
            test_coverage=0.9,
            performance=0.8,
            completeness=0.9,
            agent_self_confidence=0.85
        )
        
        context = {"confidence_factors": factors}
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score >= 0.8
        assert result.threshold == 0.8
        assert "meets threshold" in result.message
    
    @pytest.mark.asyncio
    async def test_confidence_gate_failure(self):
        """Test ConfidenceGate failure case"""
        factors = ConfidenceFactors(
            syntax_correctness=0.5,
            type_safety=0.4,
            code_quality=0.6,
            test_coverage=0.5,
            performance=0.6,
            completeness=0.5,
            agent_self_confidence=0.4
        )
        
        context = {"confidence_factors": factors}
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score < 0.8
        assert result.threshold == 0.8
        assert "below threshold" in result.message
    
    @pytest.mark.asyncio
    async def test_confidence_gate_missing_factors(self):
        """Test ConfidenceGate with missing factors"""
        context = {}
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.0
        assert "No confidence factors provided" in result.message
    
    @pytest.mark.asyncio
    async def test_confidence_gate_invalid_factors(self):
        """Test ConfidenceGate with invalid factors"""
        context = {"confidence_factors": "invalid"}
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.0
        assert "No confidence factors provided" in result.message


class TestSyntaxQualityGate:
    """Test SyntaxQualityGate implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gate = SyntaxQualityGate("syntax_gate", 0.9, 1.0)
    
    @pytest.mark.asyncio
    async def test_syntax_gate_success(self):
        """Test SyntaxQualityGate success case"""
        factors = ConfidenceFactors(syntax_correctness=0.95)
        context = {"confidence_factors": factors}
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 0.95
        assert result.threshold == 0.9
        assert "meets threshold" in result.message
    
    @pytest.mark.asyncio
    async def test_syntax_gate_failure(self):
        """Test SyntaxQualityGate failure case"""
        factors = ConfidenceFactors(syntax_correctness=0.7)
        context = {"confidence_factors": factors}
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.7
        assert result.threshold == 0.9
        assert "below threshold" in result.message


class TestTypeSafetyQualityGate:
    """Test TypeSafetyQualityGate implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gate = TypeSafetyQualityGate("type_safety_gate", 0.7, 1.0)
    
    @pytest.mark.asyncio
    async def test_type_safety_gate_success(self):
        """Test TypeSafetyQualityGate success case"""
        factors = ConfidenceFactors(type_safety=0.85)
        context = {"confidence_factors": factors}
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 0.85
        assert result.threshold == 0.7
        assert "meets threshold" in result.message
    
    @pytest.mark.asyncio
    async def test_type_safety_gate_failure(self):
        """Test TypeSafetyQualityGate failure case"""
        factors = ConfidenceFactors(type_safety=0.5)
        context = {"confidence_factors": factors}
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.5
        assert result.threshold == 0.7
        assert "below threshold" in result.message


class TestTestCoverageQualityGate:
    """Test TestCoverageQualityGate implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gate = TestCoverageQualityGate("coverage_gate", 0.8, 1.0)
    
    @pytest.mark.asyncio
    async def test_coverage_gate_from_factors(self):
        """Test coverage gate using confidence factors"""
        factors = ConfidenceFactors(test_coverage=0.9)
        context = {"confidence_factors": factors}
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 0.9
        assert result.threshold == 0.8
    
    @pytest.mark.asyncio
    async def test_coverage_gate_from_test_results(self):
        """Test coverage gate using test results"""
        context = {
            "test_results": {
                "coverage": {
                    "coverage_percent": 85.0
                }
            }
        }
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 0.85
        assert result.threshold == 0.8
    
    @pytest.mark.asyncio
    async def test_coverage_gate_failure(self):
        """Test coverage gate failure"""
        factors = ConfidenceFactors(test_coverage=0.6)
        context = {"confidence_factors": factors}
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.6
        assert result.threshold == 0.8


class TestPerformanceQualityGate:
    """Test PerformanceQualityGate implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gate = PerformanceQualityGate("performance_gate", 30.0, "execution_time", 1.0)
    
    @pytest.mark.asyncio
    async def test_performance_gate_execution_time_success(self):
        """Test performance gate with execution time success"""
        context = {
            "performance_data": {
                "execution_time": 25.0
            }
        }
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score > 0.8  # Should be high since under threshold
        assert result.threshold == 30.0
        assert "within threshold" in result.message
    
    @pytest.mark.asyncio
    async def test_performance_gate_execution_time_failure(self):
        """Test performance gate with execution time failure"""
        context = {
            "performance_data": {
                "execution_time": 45.0
            }
        }
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score < 0.5  # Should be low since over threshold
        assert result.threshold == 30.0
        assert "exceeds threshold" in result.message
    
    @pytest.mark.asyncio
    async def test_performance_gate_memory_usage(self):
        """Test performance gate with memory usage"""
        memory_gate = PerformanceQualityGate("memory_gate", 1000.0, "memory_usage", 1.0)
        
        context = {
            "performance_data": {
                "memory_usage": 800.0
            }
        }
        
        result = await memory_gate.evaluate(context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score > 0.8
        assert "within threshold" in result.message


class TestCustomQualityGate:
    """Test CustomQualityGate implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        async def custom_eval(context, threshold):
            value = context.get("custom_value", 0.0)
            return {
                "status": "passed" if value >= threshold else "failed",
                "score": value,
                "message": f"Custom evaluation: {value}",
                "details": {"custom_value": value}
            }
        
        self.gate = CustomQualityGate("custom_gate", 0.8, custom_eval, 1.0)
    
    @pytest.mark.asyncio
    async def test_custom_gate_success(self):
        """Test custom gate success"""
        context = {"custom_value": 0.9}
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 0.9
        assert result.threshold == 0.8
        assert "Custom evaluation" in result.message
    
    @pytest.mark.asyncio
    async def test_custom_gate_failure(self):
        """Test custom gate failure"""
        context = {"custom_value": 0.6}
        
        result = await self.gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.6
        assert result.threshold == 0.8
        assert "Custom evaluation" in result.message
    
    @pytest.mark.asyncio
    async def test_custom_gate_exception(self):
        """Test custom gate exception handling"""
        async def failing_eval(context, threshold):
            raise ValueError("Test error")
        
        failing_gate = CustomQualityGate("failing_gate", 0.8, failing_eval, 1.0)
        context = {}
        
        result = await failing_gate.evaluate(context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.0
        assert "Evaluation error" in result.message


class TestQualityGateSystem:
    """Test QualityGateSystem implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.system = QualityGateSystem()
    
    def test_system_initialization(self):
        """Test QualityGateSystem initialization"""
        assert len(self.system.gates) == 0
        assert self.system.logger is not None
        assert self.system.voice is not None
    
    def test_add_gate(self):
        """Test adding a gate to the system"""
        gate = ConfidenceGate("test_gate", 0.8, 1.0)
        
        self.system.add_gate(gate)
        
        assert "test_gate" in self.system.gates
        assert self.system.gates["test_gate"] == gate
    
    def test_remove_gate(self):
        """Test removing a gate from the system"""
        gate = ConfidenceGate("test_gate", 0.8, 1.0)
        self.system.add_gate(gate)
        
        self.system.remove_gate("test_gate")
        
        assert "test_gate" not in self.system.gates
    
    def test_remove_nonexistent_gate(self):
        """Test removing a nonexistent gate"""
        self.system.remove_gate("nonexistent_gate")
        # Should not raise an exception
    
    def test_get_gate(self):
        """Test getting a gate from the system"""
        gate = ConfidenceGate("test_gate", 0.8, 1.0)
        self.system.add_gate(gate)
        
        retrieved_gate = self.system.get_gate("test_gate")
        
        assert retrieved_gate == gate
    
    def test_get_nonexistent_gate(self):
        """Test getting a nonexistent gate"""
        retrieved_gate = self.system.get_gate("nonexistent_gate")
        
        assert retrieved_gate is None
    
    def test_list_gates(self):
        """Test listing all gates"""
        gate1 = ConfidenceGate("gate1", 0.8, 1.0)
        gate2 = SyntaxQualityGate("gate2", 0.9, 1.0)
        
        self.system.add_gate(gate1)
        self.system.add_gate(gate2)
        
        gate_names = self.system.list_gates()
        
        assert "gate1" in gate_names
        assert "gate2" in gate_names
        assert len(gate_names) == 2
    
    @pytest.mark.asyncio
    async def test_evaluate_single_gate(self):
        """Test evaluating a single gate"""
        gate = ConfidenceGate("test_gate", 0.8, 1.0)
        self.system.add_gate(gate)
        
        factors = ConfidenceFactors(
            syntax_correctness=0.9,
            type_safety=0.8,
            code_quality=0.85,
            test_coverage=0.9,
            performance=0.8,
            completeness=0.9,
            agent_self_confidence=0.85
        )
        context = {"confidence_factors": factors}
        
        result = await self.system.evaluate_gate("test_gate", context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score >= 0.8
        assert result.execution_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_nonexistent_gate(self):
        """Test evaluating a nonexistent gate"""
        context = {}
        
        result = await self.system.evaluate_gate("nonexistent_gate", context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.0
        assert "not found" in result.message
    
    @pytest.mark.asyncio
    async def test_evaluate_disabled_gate(self):
        """Test evaluating a disabled gate"""
        gate = ConfidenceGate("test_gate", 0.8, 1.0, enabled=False)
        self.system.add_gate(gate)
        
        context = {}
        
        result = await self.system.evaluate_gate("test_gate", context)
        
        assert result.status == QualityGateStatus.SKIPPED
        assert result.score == 0.0
        assert "disabled" in result.message
    
    @pytest.mark.asyncio
    async def test_evaluate_all_gates_success(self):
        """Test evaluating all gates with success"""
        gate1 = ConfidenceGate("gate1", 0.8, 1.0)
        gate2 = SyntaxQualityGate("gate2", 0.9, 1.0)
        
        self.system.add_gate(gate1)
        self.system.add_gate(gate2)
        
        factors = ConfidenceFactors(
            syntax_correctness=0.95,
            type_safety=0.9,
            code_quality=0.9,
            test_coverage=0.9,
            performance=0.9,
            completeness=0.9,
            agent_self_confidence=0.9
        )
        context = {"confidence_factors": factors}
        
        result = await self.system.evaluate_all_gates(context)
        
        assert result["overall_status"] == QualityGateStatus.PASSED
        assert result["overall_score"] > 0.8
        assert result["total_gates"] == 2
        assert result["passed_gates"] == 2
        assert result["failed_gates"] == 0
        assert len(result["blocking_failures"]) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_all_gates_failure(self):
        """Test evaluating all gates with failure"""
        gate1 = ConfidenceGate("gate1", 0.8, 1.0)
        gate2 = SyntaxQualityGate("gate2", 0.9, 1.0)
        
        self.system.add_gate(gate1)
        self.system.add_gate(gate2)
        
        factors = ConfidenceFactors(
            syntax_correctness=0.7,  # Will fail gate2
            type_safety=0.4,
            code_quality=0.4,
            test_coverage=0.4,
            performance=0.4,
            completeness=0.4,
            agent_self_confidence=0.4
        )
        context = {"confidence_factors": factors}
        
        result = await self.system.evaluate_all_gates(context)
        
        assert result["overall_status"] == QualityGateStatus.FAILED
        assert result["overall_score"] < 0.8
        assert result["total_gates"] == 2
        assert result["passed_gates"] == 0
        assert result["failed_gates"] == 2
        assert len(result["blocking_failures"]) == 2
    
    @pytest.mark.asyncio
    async def test_evaluate_all_gates_non_blocking_failure(self):
        """Test evaluating all gates with non-blocking failure"""
        gate1 = ConfidenceGate("gate1", 0.8, 1.0)
        gate2 = SyntaxQualityGate("gate2", 0.9, 1.0, blocking=False)
        
        self.system.add_gate(gate1)
        self.system.add_gate(gate2)
        
        factors = ConfidenceFactors(
            syntax_correctness=0.7,  # Will fail gate2 (non-blocking)
            type_safety=0.9,
            code_quality=0.9,
            test_coverage=0.9,
            performance=0.9,
            completeness=0.9,
            agent_self_confidence=0.9
        )
        context = {"confidence_factors": factors}
        
        result = await self.system.evaluate_all_gates(context)
        
        assert result["overall_status"] == QualityGateStatus.WARNING
        assert result["total_gates"] == 2
        assert result["passed_gates"] == 1
        assert result["failed_gates"] == 1
        assert len(result["blocking_failures"]) == 0
    
    def test_load_config_json(self):
        """Test loading configuration from JSON file"""
        config_data = {
            "gates": [
                {
                    "name": "confidence_gate",
                    "type": "confidence",
                    "threshold": 0.8,
                    "weight": 1.0,
                    "enabled": True,
                    "blocking": True
                },
                {
                    "name": "syntax_gate",
                    "type": "syntax",
                    "threshold": 0.9,
                    "weight": 1.5,
                    "enabled": True,
                    "blocking": True
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            self.system.load_config(config_path)
            
            assert len(self.system.gates) == 2
            assert "confidence_gate" in self.system.gates
            assert "syntax_gate" in self.system.gates
            
            confidence_gate = self.system.gates["confidence_gate"]
            assert confidence_gate.threshold == 0.8
            assert confidence_gate.weight == 1.0
            
            syntax_gate = self.system.gates["syntax_gate"]
            assert syntax_gate.threshold == 0.9
            assert syntax_gate.weight == 1.5
            
        finally:
            import os
            os.unlink(config_path)
    
    def test_load_config_performance_gate(self):
        """Test loading performance gate configuration"""
        config_data = {
            "gates": [
                {
                    "name": "performance_gate",
                    "type": "performance",
                    "threshold": 30.0,
                    "metric_type": "execution_time",
                    "weight": 1.0,
                    "enabled": True,
                    "blocking": True
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            self.system.load_config(config_path)
            
            assert "performance_gate" in self.system.gates
            gate = self.system.gates["performance_gate"]
            assert isinstance(gate, PerformanceQualityGate)
            assert gate.threshold == 30.0
            assert gate.metric_type == "execution_time"
            
        finally:
            import os
            os.unlink(config_path)
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration from nonexistent file"""
        self.system.load_config("nonexistent_file.json")
        
        # Should not raise an exception
        assert len(self.system.gates) == 0
    
    def test_save_config(self):
        """Test saving configuration to file"""
        gate1 = ConfidenceGate("gate1", 0.8, 1.0)
        gate2 = SyntaxQualityGate("gate2", 0.9, 1.5)
        
        self.system.add_gate(gate1)
        self.system.add_gate(gate2)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            self.system.save_config(config_path)
            
            # Load and verify
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            
            assert "gates" in saved_config
            assert len(saved_config["gates"]) == 2
            
            # Find gates in saved config
            gate_names = [g["name"] for g in saved_config["gates"]]
            assert "gate1" in gate_names
            assert "gate2" in gate_names
            
        finally:
            import os
            os.unlink(config_path)
    
    def test_get_system_status(self):
        """Test getting system status"""
        gate1 = ConfidenceGate("gate1", 0.8, 1.0)
        gate2 = SyntaxQualityGate("gate2", 0.9, 1.0, enabled=False)
        gate3 = TypeSafetyQualityGate("gate3", 0.7, 1.0, blocking=False)
        
        self.system.add_gate(gate1)
        self.system.add_gate(gate2)
        self.system.add_gate(gate3)
        
        status = self.system.get_system_status()
        
        assert status["total_gates"] == 3
        assert status["enabled_gates"] == 2
        assert status["disabled_gates"] == 1
        assert status["blocking_gates"] == 1
        assert status["non_blocking_gates"] == 2
        assert len(status["gate_names"]) == 3
        assert len(status["gate_details"]) == 3
        
        # Check gate details
        assert status["gate_details"]["gate1"]["enabled"] is True
        assert status["gate_details"]["gate1"]["blocking"] is True
        assert status["gate_details"]["gate2"]["enabled"] is False
        assert status["gate_details"]["gate3"]["blocking"] is False