"""
Validation Hooks Integration for Ultimate Agentic StarterKit.

This module provides Claude Code hooks integration specifically for validation events,
quality gate reporting, and validation advisor integration.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .validator import ValidationOrchestrator, ValidationRequest, ValidationResult
from .quality_gates import QualityGateStatus
from workflows.claude_code_hooks import (
    ClaudeCodeHookManager, HookType, HookEvent, WorkflowHookIntegration,
    get_hook_integration
)
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts
from core.config import get_config


class ValidationHookType(str, Enum):
    """Validation-specific hook types"""
    VALIDATION_START = "validation_start"
    VALIDATION_COMPLETE = "validation_complete"
    VALIDATION_FAILED = "validation_failed"
    QUALITY_GATE_PASSED = "quality_gate_passed"
    QUALITY_GATE_FAILED = "quality_gate_failed"
    CONFIDENCE_THRESHOLD_MET = "confidence_threshold_met"
    CONFIDENCE_THRESHOLD_FAILED = "confidence_threshold_failed"
    VALIDATION_RETRY = "validation_retry"
    PERFORMANCE_ALERT = "performance_alert"
    VISUAL_TEST_COMPLETE = "visual_test_complete"
    TEST_EXECUTION_COMPLETE = "test_execution_complete"


@dataclass
class ValidationHookEvent:
    """Event data for validation hooks"""
    hook_type: ValidationHookType
    timestamp: datetime
    validation_id: str
    workflow_id: str
    project_id: str
    validation_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'hook_type': self.hook_type.value,
            'timestamp': self.timestamp.isoformat(),
            'validation_id': self.validation_id,
            'workflow_id': self.workflow_id,
            'project_id': self.project_id,
            'validation_data': self.validation_data
        }


class ValidationHookManager:
    """Manager for validation-specific Claude Code hooks"""
    
    def __init__(self, orchestrator: ValidationOrchestrator):
        self.orchestrator = orchestrator
        self.logger = get_logger("validation_hooks")
        self.voice = get_voice_alerts()
        self.config = get_config()
        
        # Get global hook integration
        self.hook_integration = get_hook_integration()
        self.hook_manager = self.hook_integration.hook_manager
        
        # Validation-specific hook handlers
        self.validation_hooks: Dict[ValidationHookType, List[callable]] = {
            hook_type: [] for hook_type in ValidationHookType
        }
        
        # Configuration
        self.enable_validation_hooks = self.config.get("enable_validation_hooks", True)
        self.validation_confidence_threshold = self.config.agent.high_confidence_threshold
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'quality_gate_failures': 0,
            'confidence_threshold_failures': 0
        }
        
        # Register validation hooks with main hook manager
        self._register_validation_hooks()
        
        self.logger.info("Validation Hook Manager initialized")
    
    def _register_validation_hooks(self):
        """Register validation hooks with the main hook manager"""
        # Register validation event handlers
        self.hook_manager.register_hook(HookType.TASK_COMPLETE, self._on_task_complete)
        self.hook_manager.register_hook(HookType.WORKFLOW_COMPLETE, self._on_workflow_complete)
        self.hook_manager.register_hook(HookType.ERROR_OCCURRED, self._on_error_occurred)
        
        # Register validation-specific event handlers
        self.register_validation_hook(
            ValidationHookType.VALIDATION_COMPLETE,
            self._on_validation_complete_advisor
        )
        self.register_validation_hook(
            ValidationHookType.QUALITY_GATE_FAILED,
            self._on_quality_gate_failed
        )
        self.register_validation_hook(
            ValidationHookType.CONFIDENCE_THRESHOLD_FAILED,
            self._on_confidence_threshold_failed
        )
    
    def register_validation_hook(self, hook_type: ValidationHookType, handler: callable):
        """Register a validation-specific hook handler"""
        if hook_type not in self.validation_hooks:
            self.validation_hooks[hook_type] = []
        
        self.validation_hooks[hook_type].append(handler)
        self.logger.info(f"Registered validation hook: {hook_type.value}")
    
    def unregister_validation_hook(self, hook_type: ValidationHookType, handler: callable):
        """Unregister a validation-specific hook handler"""
        if hook_type in self.validation_hooks:
            try:
                self.validation_hooks[hook_type].remove(handler)
                self.logger.info(f"Unregistered validation hook: {hook_type.value}")
            except ValueError:
                self.logger.warning(f"Handler not found for {hook_type.value}")
    
    async def emit_validation_event(self, hook_type: ValidationHookType, validation_id: str,
                                   workflow_id: str, project_id: str,
                                   validation_data: Dict[str, Any] = None):
        """Emit a validation-specific hook event"""
        if not self.enable_validation_hooks:
            return
        
        event = ValidationHookEvent(
            hook_type=hook_type,
            timestamp=datetime.now(),
            validation_id=validation_id,
            workflow_id=workflow_id,
            project_id=project_id,
            validation_data=validation_data or {}
        )
        
        # Execute validation-specific handlers
        await self._execute_validation_hooks(event)
        
        # Also emit to main hook system for general event handling
        await self.hook_manager.emit_event(
            HookType.MILESTONE_REACHED,
            workflow_id,
            project_id,
            {
                'milestone_name': f'validation_{hook_type.value}',
                'milestone_value': validation_id,
                'validation_data': validation_data
            }
        )
    
    async def _execute_validation_hooks(self, event: ValidationHookEvent):
        """Execute validation-specific hook handlers"""
        handlers = self.validation_hooks.get(event.hook_type, [])
        
        if not handlers:
            return
        
        self.logger.debug(f"Executing {len(handlers)} validation hooks for {event.hook_type.value}")
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    await asyncio.get_event_loop().run_in_executor(None, handler, event)
            except Exception as e:
                self.logger.error(f"Validation hook execution failed: {e}")
    
    async def _on_task_complete(self, event: HookEvent):
        """Handle task completion - trigger validation if needed"""
        try:
            task_id = event.event_data.get('task_id')
            success = event.event_data.get('success', False)
            confidence = event.event_data.get('confidence', 0.0)
            
            # Check if validation is needed based on confidence
            if success and confidence < self.validation_confidence_threshold:
                self.logger.info(f"Task {task_id} completed with low confidence, triggering validation")
                
                # Create validation request
                validation_request = ValidationRequest(
                    request_id=f"task_{task_id}_validation",
                    validation_types=["confidence", "tests", "gates"],
                    context={
                        'task_id': task_id,
                        'confidence': confidence,
                        'workflow_id': event.workflow_id,
                        'project_id': event.project_id
                    }
                )
                
                # Queue validation
                await self.orchestrator.queue_validation(validation_request)
                
                # Emit validation start event
                await self.emit_validation_event(
                    ValidationHookType.VALIDATION_START,
                    validation_request.request_id,
                    event.workflow_id,
                    event.project_id,
                    {'trigger': 'low_confidence_task', 'task_id': task_id}
                )
            
        except Exception as e:
            self.logger.error(f"Task completion validation hook failed: {e}")
    
    async def _on_workflow_complete(self, event: HookEvent):
        """Handle workflow completion - trigger comprehensive validation"""
        try:
            success = event.event_data.get('success', False)
            final_confidence = event.event_data.get('final_confidence', 0.0)
            
            # Always trigger validation for workflow completion
            validation_request = ValidationRequest(
                request_id=f"workflow_{event.workflow_id}_validation",
                validation_types=["confidence", "visual", "tests", "gates", "performance"],
                context={
                    'workflow_id': event.workflow_id,
                    'project_id': event.project_id,
                    'success': success,
                    'final_confidence': final_confidence
                },
                priority=3  # High priority for workflow validation
            )
            
            # Queue validation
            await self.orchestrator.queue_validation(validation_request)
            
            # Emit validation start event
            await self.emit_validation_event(
                ValidationHookType.VALIDATION_START,
                validation_request.request_id,
                event.workflow_id,
                event.project_id,
                {'trigger': 'workflow_completion', 'success': success}
            )
            
        except Exception as e:
            self.logger.error(f"Workflow completion validation hook failed: {e}")
    
    async def _on_error_occurred(self, event: HookEvent):
        """Handle error occurrence - trigger validation for error analysis"""
        try:
            error_type = event.event_data.get('error_type')
            error_message = event.event_data.get('error_message', '')
            task_id = event.event_data.get('task_id')
            
            # Trigger validation for error analysis
            validation_request = ValidationRequest(
                request_id=f"error_{event.workflow_id}_{task_id}_validation",
                validation_types=["tests", "gates"],
                context={
                    'workflow_id': event.workflow_id,
                    'project_id': event.project_id,
                    'error_type': error_type,
                    'error_message': error_message,
                    'task_id': task_id
                },
                priority=4  # Critical priority for error validation
            )
            
            # Queue validation
            await self.orchestrator.queue_validation(validation_request)
            
            # Emit validation start event
            await self.emit_validation_event(
                ValidationHookType.VALIDATION_START,
                validation_request.request_id,
                event.workflow_id,
                event.project_id,
                {'trigger': 'error_analysis', 'error_type': error_type}
            )
            
        except Exception as e:
            self.logger.error(f"Error validation hook failed: {e}")
    
    async def _on_validation_complete_advisor(self, event: ValidationHookEvent):
        """Handle validation completion - trigger advisor review if needed"""
        try:
            validation_data = event.validation_data
            success = validation_data.get('success', False)
            confidence = validation_data.get('confidence', 0.0)
            
            # Trigger advisor review based on validation results
            if not success:
                # Failed validation - detailed advisor review needed
                await self._trigger_validation_advisor_review(
                    event.workflow_id,
                    event.project_id,
                    event.validation_id,
                    review_type="validation_failure",
                    validation_data=validation_data
                )
            elif confidence < self.validation_confidence_threshold:
                # Low confidence - brief advisor review
                await self._trigger_validation_advisor_review(
                    event.workflow_id,
                    event.project_id,
                    event.validation_id,
                    review_type="low_confidence",
                    validation_data=validation_data
                )
            
        except Exception as e:
            self.logger.error(f"Validation completion advisor hook failed: {e}")
    
    async def _on_quality_gate_failed(self, event: ValidationHookEvent):
        """Handle quality gate failure"""
        try:
            validation_data = event.validation_data
            failed_gates = validation_data.get('failed_gates', [])
            
            self.logger.warning(f"Quality gates failed: {failed_gates}")
            
            # Voice alert for quality gate failures
            self.voice.speak_error(f"Quality gates failed: {len(failed_gates)} gates")
            
            # Update statistics
            self.validation_stats['quality_gate_failures'] += 1
            
            # Trigger advisor review for quality gate failures
            await self._trigger_validation_advisor_review(
                event.workflow_id,
                event.project_id,
                event.validation_id,
                review_type="quality_gate_failure",
                validation_data=validation_data
            )
            
        except Exception as e:
            self.logger.error(f"Quality gate failure hook failed: {e}")
    
    async def _on_confidence_threshold_failed(self, event: ValidationHookEvent):
        """Handle confidence threshold failure"""
        try:
            validation_data = event.validation_data
            confidence = validation_data.get('confidence', 0.0)
            threshold = validation_data.get('threshold', self.validation_confidence_threshold)
            
            self.logger.warning(f"Confidence threshold failed: {confidence:.3f} < {threshold:.3f}")
            
            # Voice alert for confidence threshold failures
            self.voice.speak_warning(f"Confidence threshold failed: {confidence:.1%}")
            
            # Update statistics
            self.validation_stats['confidence_threshold_failures'] += 1
            
            # Trigger advisor review for confidence issues
            await self._trigger_validation_advisor_review(
                event.workflow_id,
                event.project_id,
                event.validation_id,
                review_type="confidence_threshold_failure",
                validation_data=validation_data
            )
            
        except Exception as e:
            self.logger.error(f"Confidence threshold failure hook failed: {e}")
    
    async def _trigger_validation_advisor_review(self, workflow_id: str, project_id: str,
                                                validation_id: str, review_type: str,
                                                validation_data: Dict[str, Any]):
        """Trigger advisor review for validation issues"""
        try:
            review_request = {
                'type': 'validation_advisor_review',
                'workflow_id': workflow_id,
                'project_id': project_id,
                'validation_id': validation_id,
                'review_type': review_type,
                'validation_data': validation_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Emit advisor review event
            await self.hook_manager.emit_event(
                HookType.ADVISOR_REVIEW,
                workflow_id,
                project_id,
                review_request
            )
            
            self.logger.info(f"Triggered validation advisor review: {review_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger validation advisor review: {e}")
    
    async def handle_validation_result(self, result: ValidationResult):
        """Handle validation result and emit appropriate events"""
        try:
            # Emit validation completion event
            if result.overall_success:
                await self.emit_validation_event(
                    ValidationHookType.VALIDATION_COMPLETE,
                    result.request_id,
                    result.validation_results.get('workflow_id', 'unknown'),
                    result.validation_results.get('project_id', 'unknown'),
                    {
                        'success': True,
                        'confidence': result.overall_confidence,
                        'execution_time': result.execution_time
                    }
                )
                
                # Check confidence threshold
                if result.overall_confidence >= self.validation_confidence_threshold:
                    await self.emit_validation_event(
                        ValidationHookType.CONFIDENCE_THRESHOLD_MET,
                        result.request_id,
                        result.validation_results.get('workflow_id', 'unknown'),
                        result.validation_results.get('project_id', 'unknown'),
                        {
                            'confidence': result.overall_confidence,
                            'threshold': self.validation_confidence_threshold
                        }
                    )
                else:
                    await self.emit_validation_event(
                        ValidationHookType.CONFIDENCE_THRESHOLD_FAILED,
                        result.request_id,
                        result.validation_results.get('workflow_id', 'unknown'),
                        result.validation_results.get('project_id', 'unknown'),
                        {
                            'confidence': result.overall_confidence,
                            'threshold': self.validation_confidence_threshold
                        }
                    )
            else:
                await self.emit_validation_event(
                    ValidationHookType.VALIDATION_FAILED,
                    result.request_id,
                    result.validation_results.get('workflow_id', 'unknown'),
                    result.validation_results.get('project_id', 'unknown'),
                    {
                        'success': False,
                        'confidence': result.overall_confidence,
                        'errors': result.errors,
                        'execution_time': result.execution_time
                    }
                )
            
            # Handle quality gate results
            if result.quality_gate_results:
                gate_status = result.quality_gate_results.get('overall_status')
                
                if gate_status == QualityGateStatus.PASSED:
                    await self.emit_validation_event(
                        ValidationHookType.QUALITY_GATE_PASSED,
                        result.request_id,
                        result.validation_results.get('workflow_id', 'unknown'),
                        result.validation_results.get('project_id', 'unknown'),
                        {
                            'gate_results': result.quality_gate_results,
                            'passed_gates': result.quality_gate_results.get('passed_gates', 0)
                        }
                    )
                elif gate_status == QualityGateStatus.FAILED:
                    await self.emit_validation_event(
                        ValidationHookType.QUALITY_GATE_FAILED,
                        result.request_id,
                        result.validation_results.get('workflow_id', 'unknown'),
                        result.validation_results.get('project_id', 'unknown'),
                        {
                            'gate_results': result.quality_gate_results,
                            'failed_gates': result.quality_gate_results.get('blocking_failures', [])
                        }
                    )
            
            # Handle component-specific results
            if 'visual_testing' in result.validation_results:
                await self.emit_validation_event(
                    ValidationHookType.VISUAL_TEST_COMPLETE,
                    result.request_id,
                    result.validation_results.get('workflow_id', 'unknown'),
                    result.validation_results.get('project_id', 'unknown'),
                    {
                        'visual_results': result.validation_results['visual_testing']
                    }
                )
            
            if 'test_execution' in result.validation_results:
                await self.emit_validation_event(
                    ValidationHookType.TEST_EXECUTION_COMPLETE,
                    result.request_id,
                    result.validation_results.get('workflow_id', 'unknown'),
                    result.validation_results.get('project_id', 'unknown'),
                    {
                        'test_results': result.validation_results['test_execution']
                    }
                )
            
            # Update statistics
            self.validation_stats['total_validations'] += 1
            if result.overall_success:
                self.validation_stats['successful_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
            
        except Exception as e:
            self.logger.error(f"Error handling validation result: {e}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        
        # Calculate rates
        total = stats['total_validations']
        if total > 0:
            stats['success_rate'] = stats['successful_validations'] / total
            stats['failure_rate'] = stats['failed_validations'] / total
            stats['quality_gate_failure_rate'] = stats['quality_gate_failures'] / total
            stats['confidence_threshold_failure_rate'] = stats['confidence_threshold_failures'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['quality_gate_failure_rate'] = 0.0
            stats['confidence_threshold_failure_rate'] = 0.0
        
        return stats
    
    def reset_validation_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'quality_gate_failures': 0,
            'confidence_threshold_failures': 0
        }
        self.logger.info("Validation statistics reset")


# Global validation hook manager instance
_validation_hook_manager: Optional[ValidationHookManager] = None


def get_validation_hook_manager(orchestrator: ValidationOrchestrator) -> ValidationHookManager:
    """Get the global validation hook manager instance"""
    global _validation_hook_manager
    if _validation_hook_manager is None:
        _validation_hook_manager = ValidationHookManager(orchestrator)
    return _validation_hook_manager


# Convenience functions for validation events
async def emit_validation_start(validation_id: str, workflow_id: str, project_id: str,
                               trigger: str = "manual"):
    """Emit validation start event"""
    manager = get_validation_hook_manager(None)  # Manager should be initialized
    await manager.emit_validation_event(
        ValidationHookType.VALIDATION_START,
        validation_id,
        workflow_id,
        project_id,
        {'trigger': trigger}
    )


async def emit_validation_complete(validation_id: str, workflow_id: str, project_id: str,
                                  success: bool, confidence: float, execution_time: float):
    """Emit validation complete event"""
    manager = get_validation_hook_manager(None)  # Manager should be initialized
    await manager.emit_validation_event(
        ValidationHookType.VALIDATION_COMPLETE,
        validation_id,
        workflow_id,
        project_id,
        {
            'success': success,
            'confidence': confidence,
            'execution_time': execution_time
        }
    )


async def emit_quality_gate_result(validation_id: str, workflow_id: str, project_id: str,
                                  passed: bool, gate_results: Dict[str, Any]):
    """Emit quality gate result event"""
    manager = get_validation_hook_manager(None)  # Manager should be initialized
    
    if passed:
        await manager.emit_validation_event(
            ValidationHookType.QUALITY_GATE_PASSED,
            validation_id,
            workflow_id,
            project_id,
            {'gate_results': gate_results}
        )
    else:
        await manager.emit_validation_event(
            ValidationHookType.QUALITY_GATE_FAILED,
            validation_id,
            workflow_id,
            project_id,
            {'gate_results': gate_results}
        )