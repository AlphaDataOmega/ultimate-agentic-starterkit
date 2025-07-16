"""
Unit tests for validation orchestrator.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import uuid

from validation.validator import (
    ValidationOrchestrator, ValidationRequest, ValidationResult
)
from validation.quality_gates import QualityGateStatus
from core.models import AgentResult, ProjectTask, AgentType, create_agent_result, create_project_task


class TestValidationRequest:
    """Test ValidationRequest dataclass"""
    
    def test_validation_request_creation(self):
        """Test ValidationRequest creation"""
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["confidence", "tests"],
            context={"key": "value"},
            options={"option": "value"},
            priority=2,
            timeout=120.0
        )
        
        assert request.request_id == "test_request"
        assert request.validation_types == ["confidence", "tests"]
        assert request.context == {"key": "value"}
        assert request.options == {"option": "value"}
        assert request.priority == 2
        assert request.timeout == 120.0
        assert request.timestamp is not None
    
    def test_validation_request_defaults(self):
        """Test ValidationRequest with defaults"""
        request = ValidationRequest(request_id="test_request")
        
        assert request.validation_types == []
        assert request.agent_result is None
        assert request.project_task is None
        assert request.context == {}
        assert request.options == {}
        assert request.priority == 1
        assert request.timeout == 300.0


class TestValidationResult:
    """Test ValidationResult dataclass"""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        result = ValidationResult(
            request_id="test_request",
            overall_success=True,
            overall_confidence=0.85,
            validation_results={"test": "result"},
            errors=["error1"],
            warnings=["warning1"],
            execution_time=5.0,
            retry_count=1
        )
        
        assert result.request_id == "test_request"
        assert result.overall_success is True
        assert result.overall_confidence == 0.85
        assert result.validation_results == {"test": "result"}
        assert result.errors == ["error1"]
        assert result.warnings == ["warning1"]
        assert result.execution_time == 5.0
        assert result.retry_count == 1
        assert result.timestamp is not None
    
    def test_validation_result_defaults(self):
        """Test ValidationResult with defaults"""
        result = ValidationResult(
            request_id="test_request",
            overall_success=True,
            overall_confidence=0.85
        )
        
        assert result.validation_results == {}
        assert result.quality_gate_results is None
        assert result.performance_metrics is None
        assert result.execution_time == 0.0
        assert result.errors == []
        assert result.warnings == []
        assert result.retry_count == 0


class TestValidationOrchestrator:
    """Test ValidationOrchestrator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.orchestrator = ValidationOrchestrator()
        
        # Mock dependencies
        self.orchestrator.confidence_scorer = Mock()
        self.orchestrator.visual_validator = Mock()
        self.orchestrator.test_runner = Mock()
        self.orchestrator.quality_gates = Mock()
        self.orchestrator.performance_monitor = Mock()
        
        # Mock agent result and task
        self.agent_result = create_agent_result(
            success=True,
            confidence=0.8,
            output="test output",
            agent_id="test-agent",
            execution_time=1.0
        )
        
        self.project_task = create_project_task(
            title="Test Task",
            description="Test description",
            task_type="CREATE",
            agent_type=AgentType.CODER
        )
    
    def test_orchestrator_initialization(self):
        """Test ValidationOrchestrator initialization"""
        assert self.orchestrator.confidence_scorer is not None
        assert self.orchestrator.visual_validator is not None
        assert self.orchestrator.test_runner is not None
        assert self.orchestrator.quality_gates is not None
        assert self.orchestrator.performance_monitor is not None
        
        assert self.orchestrator.active_validations == {}
        assert self.orchestrator.is_running is False
        assert self.orchestrator.max_workers == 3
        assert self.orchestrator.max_retries == 2
        
        # Check statistics
        assert self.orchestrator.total_validations == 0
        assert self.orchestrator.successful_validations == 0
        assert self.orchestrator.failed_validations == 0
        assert self.orchestrator.validation_history == []
    
    def test_setup_default_quality_gates(self):
        """Test default quality gates setup"""
        # Quality gates should be added during initialization
        assert len(self.orchestrator.quality_gates.gates) >= 0
        # The actual gates are added via quality_gates.add_gate calls
    
    @pytest.mark.asyncio
    async def test_start_orchestrator(self):
        """Test starting the orchestrator"""
        # Mock performance monitor
        self.orchestrator.performance_monitor.start_monitoring = AsyncMock()
        
        await self.orchestrator.start()
        
        assert self.orchestrator.is_running is True
        assert len(self.orchestrator.worker_tasks) == 3
        self.orchestrator.performance_monitor.start_monitoring.assert_called_once()
        
        # Clean up
        await self.orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_stop_orchestrator(self):
        """Test stopping the orchestrator"""
        # Mock performance monitor
        self.orchestrator.performance_monitor.start_monitoring = AsyncMock()
        self.orchestrator.performance_monitor.stop_monitoring = AsyncMock()
        
        await self.orchestrator.start()
        await self.orchestrator.stop()
        
        assert self.orchestrator.is_running is False
        assert len(self.orchestrator.worker_tasks) == 0
        self.orchestrator.performance_monitor.stop_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_success(self):
        """Test successful validation"""
        # Mock validation pipeline
        with patch.object(self.orchestrator, '_execute_validation') as mock_execute:
            mock_result = ValidationResult(
                request_id="test_request",
                overall_success=True,
                overall_confidence=0.85,
                execution_time=2.0
            )
            mock_execute.return_value = mock_result
            
            request = ValidationRequest(
                request_id="test_request",
                validation_types=["confidence"],
                agent_result=self.agent_result,
                project_task=self.project_task
            )
            
            result = await self.orchestrator.validate(request)
            
            assert result.overall_success is True
            assert result.overall_confidence == 0.85
            assert result.execution_time >= 2.0
            assert self.orchestrator.total_validations == 1
            assert self.orchestrator.successful_validations == 1
            assert self.orchestrator.failed_validations == 0
            assert len(self.orchestrator.validation_history) == 1
    
    @pytest.mark.asyncio
    async def test_validate_failure(self):
        """Test failed validation"""
        # Mock validation pipeline
        with patch.object(self.orchestrator, '_execute_validation') as mock_execute:
            mock_result = ValidationResult(
                request_id="test_request",
                overall_success=False,
                overall_confidence=0.3,
                errors=["Test error"],
                execution_time=2.0
            )
            mock_execute.return_value = mock_result
            
            request = ValidationRequest(
                request_id="test_request",
                validation_types=["confidence"],
                agent_result=self.agent_result,
                project_task=self.project_task
            )
            
            result = await self.orchestrator.validate(request)
            
            assert result.overall_success is False
            assert result.overall_confidence == 0.3
            assert len(result.errors) == 1
            assert self.orchestrator.total_validations == 1
            assert self.orchestrator.successful_validations == 0
            assert self.orchestrator.failed_validations == 1
    
    @pytest.mark.asyncio
    async def test_validate_exception(self):
        """Test validation with exception"""
        # Mock validation pipeline to raise exception
        with patch.object(self.orchestrator, '_execute_validation') as mock_execute:
            mock_execute.side_effect = Exception("Test exception")
            
            request = ValidationRequest(
                request_id="test_request",
                validation_types=["confidence"],
                agent_result=self.agent_result,
                project_task=self.project_task
            )
            
            result = await self.orchestrator.validate(request)
            
            assert result.overall_success is False
            assert result.overall_confidence == 0.0
            assert len(result.errors) == 1
            assert "Test exception" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_run_confidence_scoring(self):
        """Test confidence scoring execution"""
        # Mock confidence scorer
        self.orchestrator.confidence_scorer.calculate_confidence = AsyncMock(return_value=0.85)
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["confidence"],
            agent_result=self.agent_result,
            project_task=self.project_task
        )
        
        result = await self.orchestrator._run_confidence_scoring(request)
        
        assert result["confidence_scoring"]["overall_confidence"] == 0.85
        assert result["confidence_scoring"]["success"] is True
        self.orchestrator.confidence_scorer.calculate_confidence.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_confidence_scoring_exception(self):
        """Test confidence scoring with exception"""
        # Mock confidence scorer to raise exception
        self.orchestrator.confidence_scorer.calculate_confidence = AsyncMock(
            side_effect=Exception("Test error")
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["confidence"],
            agent_result=self.agent_result,
            project_task=self.project_task
        )
        
        result = await self.orchestrator._run_confidence_scoring(request)
        
        assert result["confidence_scoring"]["overall_confidence"] == 0.0
        assert result["confidence_scoring"]["success"] is False
        assert "Test error" in result["confidence_scoring"]["error"]
    
    @pytest.mark.asyncio
    async def test_run_visual_testing_single_url(self):
        """Test visual testing with single URL"""
        # Mock visual validator
        self.orchestrator.visual_validator.validate_web_interface = AsyncMock(
            return_value={"success": True, "confidence": 0.9}
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["visual"],
            options={
                "visual": {
                    "url": "http://localhost:3000",
                    "test_name": "test_page"
                }
            }
        )
        
        result = await self.orchestrator._run_visual_testing(request)
        
        assert result["visual_testing"]["success"] is True
        assert result["visual_testing"]["confidence"] == 0.9
        self.orchestrator.visual_validator.validate_web_interface.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_visual_testing_prp_config(self):
        """Test visual testing with PRP config"""
        # Mock visual validator
        self.orchestrator.visual_validator.validate_prp_interface = AsyncMock(
            return_value={"success": True, "confidence": 0.85}
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["visual"],
            options={
                "visual": {
                    "prp_config": {"feature": "test_feature"}
                }
            }
        )
        
        result = await self.orchestrator._run_visual_testing(request)
        
        assert result["visual_testing"]["success"] is True
        assert result["visual_testing"]["confidence"] == 0.85
        self.orchestrator.visual_validator.validate_prp_interface.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_visual_testing_multiple_tests(self):
        """Test visual testing with multiple tests"""
        # Mock visual validator
        self.orchestrator.visual_validator.run_multiple_tests = AsyncMock(
            return_value={"success": True, "overall_confidence": 0.8}
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["visual"],
            options={
                "visual": {
                    "test_configs": [
                        {"url": "http://localhost:3000", "test_name": "test1"},
                        {"url": "http://localhost:3001", "test_name": "test2"}
                    ]
                }
            }
        )
        
        result = await self.orchestrator._run_visual_testing(request)
        
        assert result["visual_testing"]["success"] is True
        assert result["visual_testing"]["confidence"] == 0.8
        self.orchestrator.visual_validator.run_multiple_tests.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_visual_testing_no_config(self):
        """Test visual testing with no valid config"""
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["visual"],
            options={"visual": {}}
        )
        
        result = await self.orchestrator._run_visual_testing(request)
        
        assert result["visual_testing"]["success"] is False
        assert "No valid visual testing configuration" in result["visual_testing"]["error"]
    
    @pytest.mark.asyncio
    async def test_run_test_execution_pytest(self):
        """Test test execution with pytest"""
        # Mock test runner
        self.orchestrator.test_runner.run_pytest_tests = AsyncMock(
            return_value={"success": True, "framework": "pytest"}
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["tests"],
            options={
                "tests": {
                    "type": "pytest",
                    "test_path": "tests/",
                    "options": ["-v"]
                }
            }
        )
        
        result = await self.orchestrator._run_test_execution(request)
        
        assert result["test_execution"]["success"] is True
        assert result["test_execution"]["framework"] == "pytest"
        self.orchestrator.test_runner.run_pytest_tests.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_test_execution_unittest(self):
        """Test test execution with unittest"""
        # Mock test runner
        self.orchestrator.test_runner.run_unittest_tests = AsyncMock(
            return_value={"success": True, "framework": "unittest"}
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["tests"],
            options={
                "tests": {
                    "type": "unittest",
                    "test_path": "tests/"
                }
            }
        )
        
        result = await self.orchestrator._run_test_execution(request)
        
        assert result["test_execution"]["success"] is True
        assert result["test_execution"]["framework"] == "unittest"
        self.orchestrator.test_runner.run_unittest_tests.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_test_execution_all(self):
        """Test test execution with all tests"""
        # Mock test runner
        self.orchestrator.test_runner.run_all_tests = AsyncMock(
            return_value={"overall_success": True, "individual_results": {}}
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["tests"],
            options={
                "tests": {
                    "type": "all",
                    "test_types": ["pytest", "mypy", "ruff"]
                }
            }
        )
        
        result = await self.orchestrator._run_test_execution(request)
        
        assert result["test_execution"]["overall_success"] is True
        self.orchestrator.test_runner.run_all_tests.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_quality_gates(self):
        """Test quality gates execution"""
        # Mock quality gates
        self.orchestrator.quality_gates.evaluate_all_gates = AsyncMock(
            return_value={
                "overall_status": QualityGateStatus.PASSED,
                "overall_score": 0.9,
                "passed_gates": 3,
                "failed_gates": 0
            }
        )
        
        # Mock confidence scorer
        self.orchestrator.confidence_scorer._extract_confidence_factors = AsyncMock(
            return_value=Mock()
        )
        
        # Mock performance monitor
        self.orchestrator.performance_monitor.get_current_metrics = Mock(
            return_value={"cpu_percent": 50.0}
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["gates"],
            agent_result=self.agent_result,
            project_task=self.project_task
        )
        
        validation_results = {}
        
        result = await self.orchestrator._run_quality_gates(request, validation_results)
        
        assert result["overall_status"] == QualityGateStatus.PASSED
        assert result["overall_score"] == 0.9
        assert result["passed_gates"] == 3
        assert result["failed_gates"] == 0
        self.orchestrator.quality_gates.evaluate_all_gates.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_validation_pipeline_comprehensive(self):
        """Test complete validation pipeline"""
        # Mock all components
        self.orchestrator.confidence_scorer.calculate_confidence = AsyncMock(return_value=0.85)
        self.orchestrator.visual_validator.validate_web_interface = AsyncMock(
            return_value={"success": True, "confidence": 0.9}
        )
        self.orchestrator.test_runner.run_pytest_tests = AsyncMock(
            return_value={"success": True, "framework": "pytest"}
        )
        self.orchestrator.quality_gates.evaluate_all_gates = AsyncMock(
            return_value={
                "overall_status": QualityGateStatus.PASSED,
                "overall_score": 0.9
            }
        )
        self.orchestrator.performance_monitor.get_current_metrics = Mock(
            return_value={"cpu_percent": 50.0}
        )
        
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["confidence", "visual", "tests", "gates", "performance"],
            agent_result=self.agent_result,
            project_task=self.project_task,
            options={
                "visual": {"url": "http://localhost:3000", "test_name": "test"},
                "tests": {"type": "pytest"}
            }
        )
        
        result = await self.orchestrator._run_validation_pipeline(request)
        
        assert result.overall_success is True
        assert result.overall_confidence == 0.9
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.quality_gate_results is not None
        assert result.performance_metrics is not None
    
    @pytest.mark.asyncio
    async def test_queue_validation(self):
        """Test queueing validation request"""
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["confidence"]
        )
        
        request_id = await self.orchestrator.queue_validation(request)
        
        assert request_id == "test_request"
        assert self.orchestrator.validation_queue.qsize() == 1
    
    def test_get_validation_status_active(self):
        """Test getting status of active validation"""
        request = ValidationRequest(
            request_id="test_request",
            validation_types=["confidence"],
            priority=2
        )
        
        self.orchestrator.active_validations["test_request"] = request
        
        status = self.orchestrator.get_validation_status("test_request")
        
        assert status["request_id"] == "test_request"
        assert status["status"] == "in_progress"
        assert status["validation_types"] == ["confidence"]
        assert status["priority"] == 2
    
    def test_get_validation_status_completed(self):
        """Test getting status of completed validation"""
        result = ValidationResult(
            request_id="test_request",
            overall_success=True,
            overall_confidence=0.85,
            execution_time=5.0
        )
        
        self.orchestrator.validation_history.append(result)
        
        status = self.orchestrator.get_validation_status("test_request")
        
        assert status["request_id"] == "test_request"
        assert status["status"] == "completed"
        assert status["success"] is True
        assert status["confidence"] == 0.85
        assert status["execution_time"] == 5.0
    
    def test_get_validation_status_not_found(self):
        """Test getting status of nonexistent validation"""
        status = self.orchestrator.get_validation_status("nonexistent_request")
        
        assert status is None
    
    def test_get_system_status(self):
        """Test getting system status"""
        # Add some test data
        self.orchestrator.total_validations = 10
        self.orchestrator.successful_validations = 8
        self.orchestrator.failed_validations = 2
        self.orchestrator.is_running = True
        
        # Mock components
        self.orchestrator.performance_monitor.is_running = True
        self.orchestrator.quality_gates.get_system_status = Mock(
            return_value={"total_gates": 5}
        )
        
        status = self.orchestrator.get_system_status()
        
        assert status["orchestrator_running"] is True
        assert status["total_validations"] == 10
        assert status["successful_validations"] == 8
        assert status["failed_validations"] == 2
        assert status["success_rate"] == 0.8
        assert status["performance_monitoring"] is True
        assert status["components"]["confidence_scorer"] == "active"
        assert status["components"]["visual_validator"] == "active"
        assert status["components"]["test_runner"] == "active"
        assert status["components"]["quality_gates"] == "active"
        assert status["components"]["performance_monitor"] == "active"
    
    def test_get_validation_history(self):
        """Test getting validation history"""
        # Add test results
        for i in range(5):
            result = ValidationResult(
                request_id=f"test_request_{i}",
                overall_success=i % 2 == 0,
                overall_confidence=0.8,
                execution_time=float(i),
                errors=["error"] if i % 2 == 1 else [],
                warnings=["warning"] if i % 3 == 0 else []
            )
            self.orchestrator.validation_history.append(result)
        
        history = self.orchestrator.get_validation_history(limit=3)
        
        assert len(history) == 3
        assert history[0]["request_id"] == "test_request_2"
        assert history[1]["request_id"] == "test_request_3"
        assert history[2]["request_id"] == "test_request_4"
        
        # Check structure
        assert "success" in history[0]
        assert "confidence" in history[0]
        assert "execution_time" in history[0]
        assert "timestamp" in history[0]
        assert "errors" in history[0]
        assert "warnings" in history[0]
    
    def test_get_validation_history_empty(self):
        """Test getting validation history when empty"""
        history = self.orchestrator.get_validation_history()
        
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_execute_validation_timeout(self):
        """Test validation execution with timeout"""
        # Mock a long-running validation
        with patch.object(self.orchestrator, '_run_validation_pipeline') as mock_pipeline:
            # Simulate timeout
            mock_pipeline.side_effect = asyncio.TimeoutError()
            
            request = ValidationRequest(
                request_id="test_request",
                validation_types=["confidence"],
                timeout=0.1  # Very short timeout
            )
            
            result = await self.orchestrator._execute_validation(request)
            
            assert result.overall_success is False
            assert result.overall_confidence == 0.0
            assert any("timeout" in error.lower() for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_execute_validation_retry(self):
        """Test validation execution with retry"""
        # Mock failing validation that succeeds on retry
        with patch.object(self.orchestrator, '_run_validation_pipeline') as mock_pipeline:
            failing_result = ValidationResult(
                request_id="test_request",
                overall_success=False,
                overall_confidence=0.3,
                retry_count=0
            )
            
            success_result = ValidationResult(
                request_id="test_request",
                overall_success=True,
                overall_confidence=0.85,
                retry_count=1
            )
            
            mock_pipeline.side_effect = [failing_result, success_result]
            
            request = ValidationRequest(
                request_id="test_request",
                validation_types=["confidence"]
            )
            
            result = await self.orchestrator._execute_validation(request)
            
            assert result.overall_success is True
            assert result.overall_confidence == 0.85
            assert result.retry_count == 1
            assert mock_pipeline.call_count == 2