"""
Quality Gates System for Ultimate Agentic StarterKit.

This module provides configurable quality gates with validation checkpoints,
thresholds, and gate failure handling for comprehensive quality assurance.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import yaml
from pathlib import Path
import asyncio
from datetime import datetime

from .confidence import ConfidenceFactors
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts
from core.config import get_config


class QualityGateStatus(str, Enum):
    """Status enumeration for quality gates"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate evaluation"""
    gate_name: str
    status: QualityGateStatus
    score: float
    threshold: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time: float = 0.0


class QualityGate(ABC):
    """Abstract base class for quality gates"""
    
    def __init__(self, name: str, threshold: float, weight: float = 1.0, 
                 enabled: bool = True, blocking: bool = True):
        self.name = name
        self.threshold = threshold
        self.weight = weight
        self.enabled = enabled
        self.blocking = blocking
        self.logger = get_logger(f"quality_gate.{name}")
        
    @abstractmethod
    async def evaluate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Evaluate the quality gate"""
        pass
    
    def _create_result(self, status: QualityGateStatus, score: float, 
                      message: str, details: Optional[Dict[str, Any]] = None) -> QualityGateResult:
        """Create a quality gate result"""
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            threshold=self.threshold,
            message=message,
            details=details or {}
        )


class ConfidenceGate(QualityGate):
    """Quality gate for overall confidence scoring"""
    
    async def evaluate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Evaluate confidence-based quality gate"""
        try:
            confidence_factors = context.get("confidence_factors")
            if not confidence_factors or not isinstance(confidence_factors, ConfidenceFactors):
                return self._create_result(
                    QualityGateStatus.FAILED,
                    0.0,
                    "No confidence factors provided",
                    {"error": "Missing confidence_factors in context"}
                )
            
            # Calculate overall confidence score
            overall_confidence = (
                confidence_factors.syntax_correctness * 0.2 +
                confidence_factors.type_safety * 0.15 +
                confidence_factors.code_quality * 0.15 +
                confidence_factors.test_coverage * 0.2 +
                confidence_factors.performance * 0.1 +
                confidence_factors.completeness * 0.1 +
                confidence_factors.agent_self_confidence * 0.1
            )
            
            # Determine status
            if overall_confidence >= self.threshold:
                status = QualityGateStatus.PASSED
                message = f"Confidence score {overall_confidence:.3f} meets threshold {self.threshold:.3f}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Confidence score {overall_confidence:.3f} below threshold {self.threshold:.3f}"
            
            return self._create_result(
                status,
                overall_confidence,
                message,
                {
                    "confidence_breakdown": {
                        "syntax_correctness": confidence_factors.syntax_correctness,
                        "type_safety": confidence_factors.type_safety,
                        "code_quality": confidence_factors.code_quality,
                        "test_coverage": confidence_factors.test_coverage,
                        "performance": confidence_factors.performance,
                        "completeness": confidence_factors.completeness,
                        "agent_self_confidence": confidence_factors.agent_self_confidence
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Confidence gate evaluation failed: {e}")
            return self._create_result(
                QualityGateStatus.FAILED,
                0.0,
                f"Evaluation error: {str(e)}",
                {"error": str(e)}
            )


class SyntaxQualityGate(QualityGate):
    """Quality gate for syntax correctness"""
    
    async def evaluate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Evaluate syntax correctness quality gate"""
        try:
            confidence_factors = context.get("confidence_factors")
            if not confidence_factors:
                return self._create_result(
                    QualityGateStatus.FAILED,
                    0.0,
                    "No confidence factors provided"
                )
            
            syntax_score = confidence_factors.syntax_correctness
            
            if syntax_score >= self.threshold:
                status = QualityGateStatus.PASSED
                message = f"Syntax correctness {syntax_score:.3f} meets threshold {self.threshold:.3f}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Syntax correctness {syntax_score:.3f} below threshold {self.threshold:.3f}"
            
            return self._create_result(
                status,
                syntax_score,
                message,
                {"syntax_correctness": syntax_score}
            )
            
        except Exception as e:
            self.logger.error(f"Syntax gate evaluation failed: {e}")
            return self._create_result(
                QualityGateStatus.FAILED,
                0.0,
                f"Evaluation error: {str(e)}"
            )


class TypeSafetyQualityGate(QualityGate):
    """Quality gate for type safety"""
    
    async def evaluate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Evaluate type safety quality gate"""
        try:
            confidence_factors = context.get("confidence_factors")
            if not confidence_factors:
                return self._create_result(
                    QualityGateStatus.FAILED,
                    0.0,
                    "No confidence factors provided"
                )
            
            type_safety_score = confidence_factors.type_safety
            
            if type_safety_score >= self.threshold:
                status = QualityGateStatus.PASSED
                message = f"Type safety {type_safety_score:.3f} meets threshold {self.threshold:.3f}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Type safety {type_safety_score:.3f} below threshold {self.threshold:.3f}"
            
            return self._create_result(
                status,
                type_safety_score,
                message,
                {"type_safety": type_safety_score}
            )
            
        except Exception as e:
            self.logger.error(f"Type safety gate evaluation failed: {e}")
            return self._create_result(
                QualityGateStatus.FAILED,
                0.0,
                f"Evaluation error: {str(e)}"
            )


class TestCoverageQualityGate(QualityGate):
    """Quality gate for test coverage"""
    
    async def evaluate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Evaluate test coverage quality gate"""
        try:
            # Check confidence factors first
            confidence_factors = context.get("confidence_factors")
            if confidence_factors:
                coverage_score = confidence_factors.test_coverage
            else:
                # Check test results
                test_results = context.get("test_results", {})
                coverage_data = test_results.get("coverage", {})
                coverage_score = coverage_data.get("coverage_percent", 0.0) / 100.0
            
            if coverage_score >= self.threshold:
                status = QualityGateStatus.PASSED
                message = f"Test coverage {coverage_score:.3f} meets threshold {self.threshold:.3f}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Test coverage {coverage_score:.3f} below threshold {self.threshold:.3f}"
            
            return self._create_result(
                status,
                coverage_score,
                message,
                {"test_coverage": coverage_score}
            )
            
        except Exception as e:
            self.logger.error(f"Test coverage gate evaluation failed: {e}")
            return self._create_result(
                QualityGateStatus.FAILED,
                0.0,
                f"Evaluation error: {str(e)}"
            )


class PerformanceQualityGate(QualityGate):
    """Quality gate for performance metrics"""
    
    def __init__(self, name: str, threshold: float, metric_type: str = "execution_time", 
                 weight: float = 1.0, enabled: bool = True, blocking: bool = True):
        super().__init__(name, threshold, weight, enabled, blocking)
        self.metric_type = metric_type
    
    async def evaluate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Evaluate performance quality gate"""
        try:
            performance_data = context.get("performance_data", {})
            
            if self.metric_type == "execution_time":
                metric_value = performance_data.get("execution_time", 0.0)
                # For execution time, lower is better
                score = max(0.0, 1.0 - (metric_value / self.threshold))
                
                if metric_value <= self.threshold:
                    status = QualityGateStatus.PASSED
                    message = f"Execution time {metric_value:.2f}s within threshold {self.threshold:.2f}s"
                else:
                    status = QualityGateStatus.FAILED
                    message = f"Execution time {metric_value:.2f}s exceeds threshold {self.threshold:.2f}s"
            
            elif self.metric_type == "memory_usage":
                metric_value = performance_data.get("memory_usage", 0.0)
                # For memory usage, lower is better
                score = max(0.0, 1.0 - (metric_value / self.threshold))
                
                if metric_value <= self.threshold:
                    status = QualityGateStatus.PASSED
                    message = f"Memory usage {metric_value:.2f}MB within threshold {self.threshold:.2f}MB"
                else:
                    status = QualityGateStatus.FAILED
                    message = f"Memory usage {metric_value:.2f}MB exceeds threshold {self.threshold:.2f}MB"
            
            else:
                # Generic metric handling
                metric_value = performance_data.get(self.metric_type, 0.0)
                score = min(1.0, metric_value / self.threshold)
                
                if metric_value >= self.threshold:
                    status = QualityGateStatus.PASSED
                    message = f"{self.metric_type} {metric_value:.2f} meets threshold {self.threshold:.2f}"
                else:
                    status = QualityGateStatus.FAILED
                    message = f"{self.metric_type} {metric_value:.2f} below threshold {self.threshold:.2f}"
            
            return self._create_result(
                status,
                score,
                message,
                {
                    "metric_type": self.metric_type,
                    "metric_value": metric_value,
                    "performance_data": performance_data
                }
            )
            
        except Exception as e:
            self.logger.error(f"Performance gate evaluation failed: {e}")
            return self._create_result(
                QualityGateStatus.FAILED,
                0.0,
                f"Evaluation error: {str(e)}"
            )


class CustomQualityGate(QualityGate):
    """Custom quality gate with user-defined evaluation function"""
    
    def __init__(self, name: str, threshold: float, evaluation_function: Callable,
                 weight: float = 1.0, enabled: bool = True, blocking: bool = True):
        super().__init__(name, threshold, weight, enabled, blocking)
        self.evaluation_function = evaluation_function
    
    async def evaluate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Evaluate custom quality gate"""
        try:
            # Call the custom evaluation function
            result = await self.evaluation_function(context, self.threshold)
            
            if isinstance(result, QualityGateResult):
                return result
            elif isinstance(result, dict):
                return QualityGateResult(
                    gate_name=self.name,
                    status=QualityGateStatus(result.get("status", "failed")),
                    score=result.get("score", 0.0),
                    threshold=self.threshold,
                    message=result.get("message", "Custom evaluation completed"),
                    details=result.get("details", {})
                )
            else:
                # Assume result is a score
                score = float(result)
                
                if score >= self.threshold:
                    status = QualityGateStatus.PASSED
                    message = f"Custom evaluation score {score:.3f} meets threshold {self.threshold:.3f}"
                else:
                    status = QualityGateStatus.FAILED
                    message = f"Custom evaluation score {score:.3f} below threshold {self.threshold:.3f}"
                
                return self._create_result(
                    status,
                    score,
                    message,
                    {"custom_score": score}
                )
            
        except Exception as e:
            self.logger.error(f"Custom gate evaluation failed: {e}")
            return self._create_result(
                QualityGateStatus.FAILED,
                0.0,
                f"Evaluation error: {str(e)}"
            )


class QualityGateSystem:
    """System for managing quality gates"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.gates = {}
        self.gate_configs = {}
        self.logger = get_logger("quality_gates")
        self.voice = get_voice_alerts()
        self.config = get_config()
        
        if config_path:
            self.load_config(config_path)
    
    def add_gate(self, gate: QualityGate):
        """Add a quality gate to the system"""
        self.gates[gate.name] = gate
        self.logger.info(f"Added quality gate: {gate.name}")
    
    def remove_gate(self, gate_name: str):
        """Remove a quality gate from the system"""
        if gate_name in self.gates:
            del self.gates[gate_name]
            self.logger.info(f"Removed quality gate: {gate_name}")
    
    def get_gate(self, gate_name: str) -> Optional[QualityGate]:
        """Get a specific quality gate"""
        return self.gates.get(gate_name)
    
    def list_gates(self) -> List[str]:
        """List all quality gate names"""
        return list(self.gates.keys())
    
    async def evaluate_gate(self, gate_name: str, context: Dict[str, Any]) -> QualityGateResult:
        """Evaluate a specific quality gate"""
        gate = self.gates.get(gate_name)
        if not gate:
            return QualityGateResult(
                gate_name=gate_name,
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=0.0,
                message=f"Gate '{gate_name}' not found"
            )
        
        if not gate.enabled:
            return QualityGateResult(
                gate_name=gate_name,
                status=QualityGateStatus.SKIPPED,
                score=0.0,
                threshold=gate.threshold,
                message=f"Gate '{gate_name}' is disabled"
            )
        
        start_time = datetime.now()
        result = await gate.evaluate(context)
        execution_time = (datetime.now() - start_time).total_seconds()
        result.execution_time = execution_time
        
        # Voice alert for failures
        if result.status == QualityGateStatus.FAILED and gate.blocking:
            self.voice.speak_error(f"Quality gate {gate_name} failed")
        
        return result
    
    async def evaluate_all_gates(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all quality gates"""
        results = {}
        overall_score = 0.0
        total_weight = 0.0
        blocking_failures = []
        
        # Evaluate gates in parallel
        tasks = []
        gate_names = []
        
        for gate_name, gate in self.gates.items():
            tasks.append(self.evaluate_gate(gate_name, context))
            gate_names.append(gate_name)
        
        gate_results = await asyncio.gather(*tasks)
        
        # Process results
        for gate_name, result in zip(gate_names, gate_results):
            results[gate_name] = result
            gate = self.gates[gate_name]
            
            # Calculate weighted score
            if result.status == QualityGateStatus.PASSED:
                overall_score += gate.weight * result.score
            elif result.status == QualityGateStatus.WARNING:
                overall_score += gate.weight * result.score * 0.5
            
            total_weight += gate.weight
            
            # Track blocking failures
            if result.status == QualityGateStatus.FAILED and gate.blocking:
                blocking_failures.append(gate_name)
        
        # Calculate overall status
        if blocking_failures:
            overall_status = QualityGateStatus.FAILED
        else:
            failed_gates = [name for name, result in results.items() 
                          if result.status == QualityGateStatus.FAILED]
            if failed_gates:
                overall_status = QualityGateStatus.WARNING
            else:
                overall_status = QualityGateStatus.PASSED
        
        overall_score = overall_score / total_weight if total_weight > 0 else 0.0
        
        # Voice alert for overall status
        if overall_status == QualityGateStatus.FAILED:
            self.voice.speak_error("Quality gates failed - review required")
        elif overall_status == QualityGateStatus.PASSED:
            self.voice.speak_success("All quality gates passed")
        
        return {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "total_gates": len(self.gates),
            "passed_gates": len([r for r in results.values() if r.status == QualityGateStatus.PASSED]),
            "failed_gates": len([r for r in results.values() if r.status == QualityGateStatus.FAILED]),
            "warning_gates": len([r for r in results.values() if r.status == QualityGateStatus.WARNING]),
            "blocking_failures": blocking_failures,
            "gate_results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def load_config(self, config_path: str):
        """Load quality gate configuration from file"""
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                self.logger.error(f"Config file not found: {config_path}")
                return
            
            # Load configuration
            if config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    config = json.load(f)
            elif config_file.suffix.lower() in ['.yml', '.yaml']:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported config file format: {config_file.suffix}")
                return
            
            # Create gates from configuration
            self._create_gates_from_config(config)
            
            self.logger.info(f"Loaded quality gate configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
    
    def _create_gates_from_config(self, config: Dict[str, Any]):
        """Create quality gates from configuration"""
        gates_config = config.get("gates", [])
        
        for gate_config in gates_config:
            gate_type = gate_config.get("type", "")
            name = gate_config.get("name", "")
            threshold = gate_config.get("threshold", 0.0)
            weight = gate_config.get("weight", 1.0)
            enabled = gate_config.get("enabled", True)
            blocking = gate_config.get("blocking", True)
            
            if gate_type == "confidence":
                gate = ConfidenceGate(name, threshold, weight, enabled, blocking)
            elif gate_type == "syntax":
                gate = SyntaxQualityGate(name, threshold, weight, enabled, blocking)
            elif gate_type == "type_safety":
                gate = TypeSafetyQualityGate(name, threshold, weight, enabled, blocking)
            elif gate_type == "test_coverage":
                gate = TestCoverageQualityGate(name, threshold, weight, enabled, blocking)
            elif gate_type == "performance":
                metric_type = gate_config.get("metric_type", "execution_time")
                gate = PerformanceQualityGate(name, threshold, metric_type, weight, enabled, blocking)
            else:
                self.logger.warning(f"Unknown gate type: {gate_type}")
                continue
            
            self.add_gate(gate)
    
    def save_config(self, config_path: str):
        """Save current quality gate configuration to file"""
        try:
            config = {
                "gates": []
            }
            
            for gate_name, gate in self.gates.items():
                gate_config = {
                    "name": gate.name,
                    "threshold": gate.threshold,
                    "weight": gate.weight,
                    "enabled": gate.enabled,
                    "blocking": gate.blocking
                }
                
                # Add type-specific configuration
                if isinstance(gate, ConfidenceGate):
                    gate_config["type"] = "confidence"
                elif isinstance(gate, SyntaxQualityGate):
                    gate_config["type"] = "syntax"
                elif isinstance(gate, TypeSafetyQualityGate):
                    gate_config["type"] = "type_safety"
                elif isinstance(gate, TestCoverageQualityGate):
                    gate_config["type"] = "test_coverage"
                elif isinstance(gate, PerformanceQualityGate):
                    gate_config["type"] = "performance"
                    gate_config["metric_type"] = gate.metric_type
                else:
                    gate_config["type"] = "custom"
                
                config["gates"].append(gate_config)
            
            # Save configuration
            config_file = Path(config_path)
            if config_file.suffix.lower() == '.json':
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
            elif config_file.suffix.lower() in ['.yml', '.yaml']:
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Saved quality gate configuration to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current status of quality gate system"""
        return {
            "total_gates": len(self.gates),
            "enabled_gates": len([g for g in self.gates.values() if g.enabled]),
            "disabled_gates": len([g for g in self.gates.values() if not g.enabled]),
            "blocking_gates": len([g for g in self.gates.values() if g.blocking]),
            "non_blocking_gates": len([g for g in self.gates.values() if not g.blocking]),
            "gate_names": list(self.gates.keys()),
            "gate_details": {
                name: {
                    "threshold": gate.threshold,
                    "weight": gate.weight,
                    "enabled": gate.enabled,
                    "blocking": gate.blocking,
                    "type": type(gate).__name__
                }
                for name, gate in self.gates.items()
            }
        }