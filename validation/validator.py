"""
Validation Orchestration System for Ultimate Agentic StarterKit.

This module coordinates all validation activities, manages parallel execution,
aggregates results, and provides re-execution logic for failed validations.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import traceback

from .confidence import ConfidenceScorer, ConfidenceFactors
from .visual_validator import VisualTestingValidator
from .test_runner import TestExecutionFramework
from .quality_gates import QualityGateSystem, QualityGateStatus
from .performance_monitor import PerformanceMonitor
from core.models import AgentResult, ProjectTask, AgentType
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts
from core.config import get_config


@dataclass
class ValidationRequest:
    """Validation request configuration"""
    request_id: str
    validation_types: List[str] = field(default_factory=list)  # confidence, visual, tests, gates, performance
    agent_result: Optional[AgentResult] = None
    project_task: Optional[ProjectTask] = None
    context: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    timeout: float = 300.0  # 5 minutes default


@dataclass
class ValidationResult:
    """Result of validation execution"""
    request_id: str
    overall_success: bool
    overall_confidence: float
    validation_results: Dict[str, Any] = field(default_factory=dict)
    quality_gate_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0


class ValidationOrchestrator:
    """Main validation orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("validation_orchestrator")
        self.voice = get_voice_alerts()
        self.config = get_config()
        
        # Initialize components
        self.confidence_scorer = ConfidenceScorer()
        self.visual_validator = VisualTestingValidator()
        self.test_runner = TestExecutionFramework()
        self.quality_gates = QualityGateSystem(config_path)
        self.performance_monitor = PerformanceMonitor()
        
        # Runtime state
        self.active_validations = {}
        self.validation_queue = asyncio.Queue()
        self.worker_tasks = []
        self.is_running = False
        self.max_workers = 3
        self.max_retries = 2
        
        # Statistics
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.validation_history = []
        
        # Default quality gates
        self._setup_default_quality_gates()
    
    def _setup_default_quality_gates(self):
        """Setup default quality gates"""
        from .quality_gates import (
            ConfidenceGate, SyntaxQualityGate, TypeSafetyQualityGate,
            TestCoverageQualityGate, PerformanceQualityGate
        )
        
        # Add default gates
        self.quality_gates.add_gate(ConfidenceGate("overall_confidence", 0.8, 2.0))
        self.quality_gates.add_gate(SyntaxQualityGate("syntax_correctness", 0.9, 1.5))
        self.quality_gates.add_gate(TypeSafetyQualityGate("type_safety", 0.7, 1.0))
        self.quality_gates.add_gate(TestCoverageQualityGate("test_coverage", 0.8, 1.5))
        self.quality_gates.add_gate(PerformanceQualityGate("execution_time", 30.0, "execution_time", 1.0))
    
    async def start(self):
        """Start the validation orchestrator"""
        if self.is_running:
            self.logger.warning("Validation orchestrator already running")
            return
        
        self.is_running = True
        
        # Start performance monitoring
        await self.performance_monitor.start_monitoring()
        
        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.worker_tasks.append(task)
        
        self.logger.info(f"Validation orchestrator started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the validation orchestrator"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop performance monitoring
        await self.performance_monitor.stop_monitoring()
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        self.logger.info("Validation orchestrator stopped")
    
    async def validate(self, request: ValidationRequest) -> ValidationResult:
        """Execute validation request"""
        try:
            self.logger.info(f"Starting validation: {request.request_id}")
            start_time = datetime.now()
            
            # Add to active validations
            self.active_validations[request.request_id] = request
            
            # Execute validation
            result = await self._execute_validation(request)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Update statistics
            self.total_validations += 1
            if result.overall_success:
                self.successful_validations += 1
            else:
                self.failed_validations += 1
            
            # Add to history
            self.validation_history.append(result)
            if len(self.validation_history) > 1000:
                self.validation_history.pop(0)
            
            # Remove from active validations
            self.active_validations.pop(request.request_id, None)
            
            # Voice notification
            if result.overall_success:
                self.voice.speak_success(f"Validation {request.request_id} completed successfully")
            else:
                self.voice.speak_error(f"Validation {request.request_id} failed")
            
            self.logger.info(f"Validation completed: {request.request_id} - Success: {result.overall_success}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {request.request_id} - {str(e)}")
            
            # Remove from active validations
            self.active_validations.pop(request.request_id, None)
            
            return ValidationResult(
                request_id=request.request_id,
                overall_success=False,
                overall_confidence=0.0,
                errors=[str(e)],
                timestamp=datetime.now()
            )
    
    async def _execute_validation(self, request: ValidationRequest) -> ValidationResult:
        """Execute validation with timeout and error handling"""
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_validation_pipeline(request),
                timeout=request.timeout
            )
            
            # Retry logic for failed validations
            if not result.overall_success and result.retry_count < self.max_retries:
                self.logger.info(f"Retrying validation: {request.request_id} (attempt {result.retry_count + 1})")
                
                # Modify request for retry
                request.options["retry_attempt"] = result.retry_count + 1
                
                # Retry validation
                retry_result = await self._run_validation_pipeline(request)
                retry_result.retry_count = result.retry_count + 1
                
                return retry_result
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Validation timeout: {request.request_id}")
            return ValidationResult(
                request_id=request.request_id,
                overall_success=False,
                overall_confidence=0.0,
                errors=[f"Validation timeout after {request.timeout} seconds"]
            )
        except Exception as e:
            self.logger.error(f"Validation execution failed: {request.request_id} - {str(e)}")
            return ValidationResult(
                request_id=request.request_id,
                overall_success=False,
                overall_confidence=0.0,
                errors=[str(e)]
            )
    
    async def _run_validation_pipeline(self, request: ValidationRequest) -> ValidationResult:
        """Run the validation pipeline"""
        validation_results = {}
        overall_confidence = 0.0
        errors = []
        warnings = []
        
        # Execute validation tasks based on request types
        tasks = []
        
        # Confidence scoring
        if "confidence" in request.validation_types and request.agent_result and request.project_task:
            tasks.append(self._run_confidence_scoring(request))
        
        # Visual testing
        if "visual" in request.validation_types:
            tasks.append(self._run_visual_testing(request))
        
        # Test execution
        if "tests" in request.validation_types:
            tasks.append(self._run_test_execution(request))
        
        # Execute validation tasks in parallel
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, task_result in enumerate(task_results):
                if isinstance(task_result, Exception):
                    errors.append(f"Task {i} failed: {str(task_result)}")
                else:
                    validation_results.update(task_result)
        
        # Quality gates evaluation
        quality_gate_results = None
        if "gates" in request.validation_types:
            quality_gate_results = await self._run_quality_gates(request, validation_results)
            
            if quality_gate_results:
                overall_confidence = quality_gate_results.get("overall_score", 0.0)
                
                # Check for gate failures
                if quality_gate_results.get("overall_status") == QualityGateStatus.FAILED:
                    errors.append("Quality gates failed")
                elif quality_gate_results.get("overall_status") == QualityGateStatus.WARNING:
                    warnings.append("Quality gates have warnings")
        
        # Performance metrics
        performance_metrics = None
        if "performance" in request.validation_types:
            performance_metrics = self.performance_monitor.get_current_metrics()
        
        # Determine overall success
        overall_success = len(errors) == 0 and (
            quality_gate_results is None or 
            quality_gate_results.get("overall_status") != QualityGateStatus.FAILED
        )
        
        return ValidationResult(
            request_id=request.request_id,
            overall_success=overall_success,
            overall_confidence=overall_confidence,
            validation_results=validation_results,
            quality_gate_results=quality_gate_results,
            performance_metrics=performance_metrics,
            errors=errors,
            warnings=warnings
        )
    
    async def _run_confidence_scoring(self, request: ValidationRequest) -> Dict[str, Any]:
        """Run confidence scoring"""
        try:
            confidence_score = await self.confidence_scorer.calculate_confidence(
                request.agent_result,
                request.project_task
            )
            
            return {
                "confidence_scoring": {
                    "overall_confidence": confidence_score,
                    "success": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Confidence scoring failed: {e}")
            return {
                "confidence_scoring": {
                    "overall_confidence": 0.0,
                    "success": False,
                    "error": str(e)
                }
            }
    
    async def _run_visual_testing(self, request: ValidationRequest) -> Dict[str, Any]:
        """Run visual testing"""
        try:
            visual_options = request.options.get("visual", {})
            
            if "url" in visual_options:
                # Single URL testing
                result = await self.visual_validator.validate_web_interface(
                    visual_options["url"],
                    visual_options.get("test_name", request.request_id),
                    visual_options.get("options", {})
                )
                
                return {
                    "visual_testing": {
                        "single_test": result,
                        "success": result.get("success", False),
                        "confidence": result.get("confidence", 0.0)
                    }
                }
            
            elif "prp_config" in visual_options:
                # PRP validation
                result = await self.visual_validator.validate_prp_interface(
                    visual_options["prp_config"]
                )
                
                return {
                    "visual_testing": {
                        "prp_validation": result,
                        "success": result.get("success", False),
                        "confidence": result.get("confidence", 0.0)
                    }
                }
            
            elif "test_configs" in visual_options:
                # Multiple tests
                result = await self.visual_validator.run_multiple_tests(
                    visual_options["test_configs"]
                )
                
                return {
                    "visual_testing": {
                        "multiple_tests": result,
                        "success": result.get("success", False),
                        "confidence": result.get("overall_confidence", 0.0)
                    }
                }
            
            else:
                return {
                    "visual_testing": {
                        "success": False,
                        "error": "No valid visual testing configuration provided"
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Visual testing failed: {e}")
            return {
                "visual_testing": {
                    "success": False,
                    "error": str(e)
                }
            }
    
    async def _run_test_execution(self, request: ValidationRequest) -> Dict[str, Any]:
        """Run test execution"""
        try:
            test_options = request.options.get("tests", {})
            test_type = test_options.get("type", "pytest")
            
            if test_type == "pytest":
                result = await self.test_runner.run_pytest_tests(
                    test_path=test_options.get("test_path"),
                    options=test_options.get("options", [])
                )
            elif test_type == "unittest":
                result = await self.test_runner.run_unittest_tests(
                    test_path=test_options.get("test_path")
                )
            elif test_type == "type_checking":
                result = await self.test_runner.run_type_checking(
                    target_path=test_options.get("target_path")
                )
            elif test_type == "linting":
                result = await self.test_runner.run_linting(
                    target_path=test_options.get("target_path")
                )
            elif test_type == "all":
                result = await self.test_runner.run_all_tests(test_options)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown test type: {test_type}"
                }
            
            return {
                "test_execution": result
            }
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return {
                "test_execution": {
                    "success": False,
                    "error": str(e)
                }
            }
    
    async def _run_quality_gates(self, request: ValidationRequest, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run quality gates evaluation"""
        try:
            # Prepare context for quality gates
            context = request.context.copy()
            
            # Add validation results to context
            context.update(validation_results)
            
            # Add confidence factors if available
            if request.agent_result and request.project_task:
                confidence_factors = await self.confidence_scorer._extract_confidence_factors(
                    request.agent_result,
                    request.project_task
                )
                context["confidence_factors"] = confidence_factors
            
            # Add performance data
            context["performance_data"] = self.performance_monitor.get_current_metrics()
            
            # Evaluate quality gates
            result = await self.quality_gates.evaluate_all_gates(context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quality gates evaluation failed: {e}")
            return {
                "overall_status": QualityGateStatus.FAILED,
                "overall_score": 0.0,
                "error": str(e)
            }
    
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing validation requests"""
        self.logger.info(f"Validation worker {worker_id} started")
        
        while self.is_running:
            try:
                # Wait for validation request
                request = await asyncio.wait_for(
                    self.validation_queue.get(),
                    timeout=1.0
                )
                
                # Process validation
                result = await self.validate(request)
                
                # Mark task as done
                self.validation_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.info(f"Validation worker {worker_id} stopped")
    
    async def queue_validation(self, request: ValidationRequest) -> str:
        """Queue a validation request"""
        await self.validation_queue.put(request)
        self.logger.info(f"Queued validation request: {request.request_id}")
        return request.request_id
    
    def get_validation_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a validation request"""
        if request_id in self.active_validations:
            request = self.active_validations[request_id]
            return {
                "request_id": request_id,
                "status": "in_progress",
                "validation_types": request.validation_types,
                "started_at": request.timestamp.isoformat(),
                "priority": request.priority
            }
        
        # Check history
        for result in reversed(self.validation_history):
            if result.request_id == request_id:
                return {
                    "request_id": request_id,
                    "status": "completed",
                    "success": result.overall_success,
                    "confidence": result.overall_confidence,
                    "execution_time": result.execution_time,
                    "completed_at": result.timestamp.isoformat()
                }
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "orchestrator_running": self.is_running,
            "active_validations": len(self.active_validations),
            "queue_size": self.validation_queue.qsize(),
            "worker_count": len(self.worker_tasks),
            "total_validations": self.total_validations,
            "successful_validations": self.successful_validations,
            "failed_validations": self.failed_validations,
            "success_rate": self.successful_validations / self.total_validations if self.total_validations > 0 else 0.0,
            "performance_monitoring": self.performance_monitor.is_running,
            "quality_gates": self.quality_gates.get_system_status(),
            "components": {
                "confidence_scorer": "active",
                "visual_validator": "active",
                "test_runner": "active",
                "quality_gates": "active",
                "performance_monitor": "active" if self.performance_monitor.is_running else "inactive"
            }
        }
    
    def get_validation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get validation history"""
        recent_validations = self.validation_history[-limit:]
        
        return [
            {
                "request_id": result.request_id,
                "success": result.overall_success,
                "confidence": result.overall_confidence,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat(),
                "errors": result.errors,
                "warnings": result.warnings
            }
            for result in recent_validations
        ]