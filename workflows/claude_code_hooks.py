"""
Claude Code Hooks Integration for the Ultimate Agentic StarterKit.

This module provides integration with Claude Code's hook system for workflow events,
progress reporting, and advisor integration.
"""

import json
import asyncio
import subprocess
import sys
import os
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.logger import get_logger
from core.voice_alerts import get_voice_alerts
from core.config import get_config


class HookType(str, Enum):
    """Types of Claude Code hooks."""
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_FAILURE = "workflow_failure"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAILURE = "task_failure"
    MILESTONE_REACHED = "milestone_reached"
    PROGRESS_UPDATE = "progress_update"
    ADVISOR_REVIEW = "advisor_review"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class HookEvent:
    """Event data for Claude Code hooks."""
    hook_type: HookType
    timestamp: datetime
    workflow_id: str
    project_id: str
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'hook_type': self.hook_type.value,
            'timestamp': self.timestamp.isoformat(),
            'workflow_id': self.workflow_id,
            'project_id': self.project_id,
            'event_data': self.event_data
        }


class ClaudeCodeHookManager:
    """
    Manager for Claude Code hooks integration.
    
    This class handles the integration with Claude Code's hook system,
    providing workflow event notifications and advisor integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Claude Code hook manager.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.logger = get_logger("claude_code_hooks")
        self.voice = get_voice_alerts()
        
        # Configuration
        self.enable_hooks = self.config.get('enable_hooks', True)
        self.hook_timeout = self.config.get('hook_timeout', 30)
        self.retry_failed_hooks = self.config.get('retry_failed_hooks', True)
        self.max_hook_retries = self.config.get('max_hook_retries', 3)
        
        # Hook registration
        self.registered_hooks: Dict[HookType, List[Callable]] = {
            hook_type: [] for hook_type in HookType
        }
        
        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.hook_stats = {
            'total_events': 0,
            'successful_hooks': 0,
            'failed_hooks': 0,
            'hook_execution_times': []
        }
        
        # Start event processing
        if self.enable_hooks:
            self._start_event_processing()
        
        self.logger.info("Claude Code Hook Manager initialized")
    
    def _start_event_processing(self):
        """Start asynchronous event processing."""
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_events())
    
    async def _process_events(self):
        """Process hook events from the queue."""
        while True:
            try:
                event = await self.event_queue.get()
                await self._execute_hooks(event)
                self.event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing hook event: {e}")
    
    async def _execute_hooks(self, event: HookEvent):
        """Execute all registered hooks for an event."""
        hook_type = event.hook_type
        hooks = self.registered_hooks.get(hook_type, [])
        
        if not hooks:
            return
        
        self.logger.debug(f"Executing {len(hooks)} hooks for {hook_type.value}")
        
        for hook in hooks:
            try:
                start_time = datetime.now()
                
                # Execute hook with timeout
                if asyncio.iscoroutinefunction(hook):
                    await asyncio.wait_for(hook(event), timeout=self.hook_timeout)
                else:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, hook, event),
                        timeout=self.hook_timeout
                    )
                
                # Record execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                self.hook_stats['hook_execution_times'].append(execution_time)
                self.hook_stats['successful_hooks'] += 1
                
                self.logger.debug(f"Hook executed successfully in {execution_time:.3f}s")
                
            except asyncio.TimeoutError:
                self.logger.error(f"Hook timeout for {hook_type.value}")
                self.hook_stats['failed_hooks'] += 1
                
            except Exception as e:
                self.logger.error(f"Hook execution failed for {hook_type.value}: {e}")
                self.hook_stats['failed_hooks'] += 1
        
        self.hook_stats['total_events'] += 1
    
    def register_hook(self, hook_type: HookType, hook_function: Callable):
        """
        Register a hook function for a specific event type.
        
        Args:
            hook_type: Type of hook event
            hook_function: Function to call when event occurs
        """
        if hook_type not in self.registered_hooks:
            self.registered_hooks[hook_type] = []
        
        self.registered_hooks[hook_type].append(hook_function)
        self.logger.info(f"Registered hook for {hook_type.value}")
    
    def unregister_hook(self, hook_type: HookType, hook_function: Callable):
        """
        Unregister a hook function.
        
        Args:
            hook_type: Type of hook event
            hook_function: Function to remove
        """
        if hook_type in self.registered_hooks:
            try:
                self.registered_hooks[hook_type].remove(hook_function)
                self.logger.info(f"Unregistered hook for {hook_type.value}")
            except ValueError:
                self.logger.warning(f"Hook function not found for {hook_type.value}")
    
    async def emit_event(self, hook_type: HookType, workflow_id: str, project_id: str,
                        event_data: Dict[str, Any] = None):
        """
        Emit a hook event.
        
        Args:
            hook_type: Type of hook event
            workflow_id: Workflow identifier
            project_id: Project identifier
            event_data: Additional event data
        """
        if not self.enable_hooks:
            return
        
        event = HookEvent(
            hook_type=hook_type,
            timestamp=datetime.now(),
            workflow_id=workflow_id,
            project_id=project_id,
            event_data=event_data or {}
        )
        
        await self.event_queue.put(event)
        self.logger.debug(f"Emitted hook event: {hook_type.value}")
    
    def emit_event_sync(self, hook_type: HookType, workflow_id: str, project_id: str,
                       event_data: Dict[str, Any] = None):
        """
        Emit a hook event synchronously.
        
        Args:
            hook_type: Type of hook event
            workflow_id: Workflow identifier
            project_id: Project identifier
            event_data: Additional event data
        """
        try:
            asyncio.create_task(self.emit_event(hook_type, workflow_id, project_id, event_data))
        except RuntimeError:
            # If no event loop is running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.emit_event(hook_type, workflow_id, project_id, event_data))
            finally:
                loop.close()
    
    def get_hook_stats(self) -> Dict[str, Any]:
        """Get hook execution statistics."""
        stats = self.hook_stats.copy()
        
        # Calculate averages
        if stats['hook_execution_times']:
            stats['average_execution_time'] = sum(stats['hook_execution_times']) / len(stats['hook_execution_times'])
            stats['max_execution_time'] = max(stats['hook_execution_times'])
            stats['min_execution_time'] = min(stats['hook_execution_times'])
        else:
            stats['average_execution_time'] = 0.0
            stats['max_execution_time'] = 0.0
            stats['min_execution_time'] = 0.0
        
        # Calculate success rate
        total_hooks = stats['successful_hooks'] + stats['failed_hooks']
        stats['success_rate'] = stats['successful_hooks'] / total_hooks if total_hooks > 0 else 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset hook statistics."""
        self.hook_stats = {
            'total_events': 0,
            'successful_hooks': 0,
            'failed_hooks': 0,
            'hook_execution_times': []
        }
        self.logger.info("Hook statistics reset")
    
    def shutdown(self):
        """Shutdown the hook manager."""
        self.logger.info("Shutting down Claude Code Hook Manager")
        
        if self.processing_task:
            self.processing_task.cancel()
        
        self.registered_hooks.clear()
        self.logger.info("Claude Code Hook Manager shutdown complete")


class ClaudeCodeIntegration:
    """
    Integration with Claude Code for workflow management.
    
    This class provides specific integrations for Claude Code's features
    including advisor reviews, progress reporting, and command execution.
    """
    
    def __init__(self, hook_manager: ClaudeCodeHookManager):
        """
        Initialize Claude Code integration.
        
        Args:
            hook_manager: Hook manager instance
        """
        self.hook_manager = hook_manager
        self.logger = get_logger("claude_code_integration")
        self.voice = get_voice_alerts()
        
        # Register default hooks
        self._register_default_hooks()
        
        self.logger.info("Claude Code Integration initialized")
    
    def _register_default_hooks(self):
        """Register default hooks for Claude Code integration."""
        # Workflow completion hook for advisor review
        self.hook_manager.register_hook(
            HookType.WORKFLOW_COMPLETE,
            self._on_workflow_complete
        )
        
        # Progress update hook for status reporting
        self.hook_manager.register_hook(
            HookType.PROGRESS_UPDATE,
            self._on_progress_update
        )
        
        # Error handling hook
        self.hook_manager.register_hook(
            HookType.ERROR_OCCURRED,
            self._on_error_occurred
        )
        
        # Milestone reached hook
        self.hook_manager.register_hook(
            HookType.MILESTONE_REACHED,
            self._on_milestone_reached
        )
    
    async def _on_workflow_complete(self, event: HookEvent):
        """Handle workflow completion event."""
        workflow_id = event.workflow_id
        project_id = event.project_id
        event_data = event.event_data
        
        self.logger.info(f"Workflow {workflow_id} completed, triggering advisor review")
        
        # Trigger advisor review
        success = event_data.get('success', False)
        final_confidence = event_data.get('final_confidence', 0.0)
        
        if success and final_confidence > 0.8:
            # High confidence completion - brief review
            await self._trigger_advisor_review(workflow_id, project_id, review_type="brief")
        elif success:
            # Lower confidence completion - detailed review
            await self._trigger_advisor_review(workflow_id, project_id, review_type="detailed")
        else:
            # Failed workflow - failure analysis
            await self._trigger_advisor_review(workflow_id, project_id, review_type="failure_analysis")
        
        # Voice notification
        self.voice.speak_milestone("Workflow completed, advisor review initiated")
    
    async def _on_progress_update(self, event: HookEvent):
        """Handle progress update event."""
        workflow_id = event.workflow_id
        event_data = event.event_data
        
        # Extract progress information
        progress = event_data.get('progress', 0.0)
        completed_tasks = event_data.get('completed_tasks', 0)
        total_tasks = event_data.get('total_tasks', 0)
        
        # Log progress update
        self.logger.info(f"Progress update: {progress:.1%} ({completed_tasks}/{total_tasks})")
        
        # Update Claude Code status (if integration available)
        await self._update_claude_code_status(workflow_id, progress, completed_tasks, total_tasks)
    
    async def _on_error_occurred(self, event: HookEvent):
        """Handle error occurrence event."""
        workflow_id = event.workflow_id
        event_data = event.event_data
        
        error_type = event_data.get('error_type', 'unknown')
        error_message = event_data.get('error_message', '')
        task_id = event_data.get('task_id')
        
        self.logger.error(f"Error occurred in workflow {workflow_id}: {error_type}")
        
        # Voice notification for critical errors
        if error_type in ['workflow_failure', 'agent_failure', 'system_error']:
            self.voice.speak_error(f"Critical error in workflow")
        
        # Log error details
        error_details = {
            'workflow_id': workflow_id,
            'error_type': error_type,
            'error_message': error_message,
            'task_id': task_id,
            'timestamp': event.timestamp.isoformat()
        }
        
        await self._log_error_to_claude_code(error_details)
    
    async def _on_milestone_reached(self, event: HookEvent):
        """Handle milestone reached event."""
        workflow_id = event.workflow_id
        event_data = event.event_data
        
        milestone_name = event_data.get('milestone_name', 'Unknown')
        milestone_value = event_data.get('milestone_value', 0)
        
        self.logger.info(f"Milestone reached: {milestone_name} = {milestone_value}")
        
        # Voice notification
        self.voice.speak_milestone(f"Milestone reached: {milestone_name}")
        
        # Update Claude Code with milestone information
        await self._report_milestone_to_claude_code(workflow_id, milestone_name, milestone_value)
    
    async def _trigger_advisor_review(self, workflow_id: str, project_id: str, review_type: str):
        """Trigger advisor review through Claude Code."""
        try:
            # Create advisor review request
            review_request = {
                'workflow_id': workflow_id,
                'project_id': project_id,
                'review_type': review_type,
                'timestamp': datetime.now().isoformat()
            }
            
            # Execute advisor review (placeholder for actual implementation)
            self.logger.info(f"Triggering {review_type} advisor review for workflow {workflow_id}")
            
            # This would integrate with actual Claude Code advisor agent
            # For now, we log the request
            self.logger.debug(f"Advisor review request: {json.dumps(review_request, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger advisor review: {e}")
    
    async def _update_claude_code_status(self, workflow_id: str, progress: float, 
                                       completed_tasks: int, total_tasks: int):
        """Update Claude Code with workflow status."""
        try:
            status_update = {
                'workflow_id': workflow_id,
                'progress': progress,
                'completed_tasks': completed_tasks,
                'total_tasks': total_tasks,
                'timestamp': datetime.now().isoformat()
            }
            
            # This would integrate with Claude Code's status system
            self.logger.debug(f"Status update: {json.dumps(status_update, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to update Claude Code status: {e}")
    
    async def _log_error_to_claude_code(self, error_details: Dict[str, Any]):
        """Log error details to Claude Code."""
        try:
            # This would integrate with Claude Code's error logging system
            self.logger.debug(f"Error logged: {json.dumps(error_details, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log error to Claude Code: {e}")
    
    async def _report_milestone_to_claude_code(self, workflow_id: str, milestone_name: str, milestone_value: Any):
        """Report milestone achievement to Claude Code."""
        try:
            milestone_report = {
                'workflow_id': workflow_id,
                'milestone_name': milestone_name,
                'milestone_value': milestone_value,
                'timestamp': datetime.now().isoformat()
            }
            
            # This would integrate with Claude Code's milestone tracking
            self.logger.debug(f"Milestone report: {json.dumps(milestone_report, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to report milestone to Claude Code: {e}")


# Integration with workflow components
class WorkflowHookIntegration:
    """
    Integration layer between workflow components and Claude Code hooks.
    """
    
    def __init__(self):
        """Initialize workflow hook integration."""
        self.hook_manager = ClaudeCodeHookManager()
        self.claude_code = ClaudeCodeIntegration(self.hook_manager)
        self.logger = get_logger("workflow_hook_integration")
        
        self.logger.info("Workflow Hook Integration initialized")
    
    async def on_workflow_start(self, workflow_id: str, project_id: str, total_tasks: int):
        """Handle workflow start event."""
        await self.hook_manager.emit_event(
            HookType.WORKFLOW_START,
            workflow_id,
            project_id,
            {'total_tasks': total_tasks}
        )
    
    async def on_workflow_complete(self, workflow_id: str, project_id: str, 
                                 success: bool, final_confidence: float):
        """Handle workflow completion event."""
        await self.hook_manager.emit_event(
            HookType.WORKFLOW_COMPLETE,
            workflow_id,
            project_id,
            {
                'success': success,
                'final_confidence': final_confidence
            }
        )
    
    async def on_task_start(self, workflow_id: str, project_id: str, task_id: str, agent_type: str):
        """Handle task start event."""
        await self.hook_manager.emit_event(
            HookType.TASK_START,
            workflow_id,
            project_id,
            {
                'task_id': task_id,
                'agent_type': agent_type
            }
        )
    
    async def on_task_complete(self, workflow_id: str, project_id: str, task_id: str, 
                             success: bool, confidence: float):
        """Handle task completion event."""
        await self.hook_manager.emit_event(
            HookType.TASK_COMPLETE,
            workflow_id,
            project_id,
            {
                'task_id': task_id,
                'success': success,
                'confidence': confidence
            }
        )
    
    async def on_milestone_reached(self, workflow_id: str, project_id: str, 
                                 milestone_name: str, milestone_value: Any):
        """Handle milestone reached event."""
        await self.hook_manager.emit_event(
            HookType.MILESTONE_REACHED,
            workflow_id,
            project_id,
            {
                'milestone_name': milestone_name,
                'milestone_value': milestone_value
            }
        )
    
    async def on_progress_update(self, workflow_id: str, project_id: str, 
                               progress: float, completed_tasks: int, total_tasks: int):
        """Handle progress update event."""
        await self.hook_manager.emit_event(
            HookType.PROGRESS_UPDATE,
            workflow_id,
            project_id,
            {
                'progress': progress,
                'completed_tasks': completed_tasks,
                'total_tasks': total_tasks
            }
        )
    
    async def on_error_occurred(self, workflow_id: str, project_id: str, 
                              error_type: str, error_message: str, task_id: str = None):
        """Handle error occurrence event."""
        await self.hook_manager.emit_event(
            HookType.ERROR_OCCURRED,
            workflow_id,
            project_id,
            {
                'error_type': error_type,
                'error_message': error_message,
                'task_id': task_id
            }
        )
    
    def shutdown(self):
        """Shutdown the hook integration."""
        self.hook_manager.shutdown()


# Global hook integration instance
_hook_integration_instance: Optional[WorkflowHookIntegration] = None


def get_hook_integration() -> WorkflowHookIntegration:
    """
    Get the global hook integration instance.
    
    Returns:
        WorkflowHookIntegration instance
    """
    global _hook_integration_instance
    if _hook_integration_instance is None:
        _hook_integration_instance = WorkflowHookIntegration()
    return _hook_integration_instance


# Convenience functions for common hook events
async def emit_workflow_start(workflow_id: str, project_id: str, total_tasks: int):
    """Emit workflow start event."""
    await get_hook_integration().on_workflow_start(workflow_id, project_id, total_tasks)


async def emit_workflow_complete(workflow_id: str, project_id: str, success: bool, final_confidence: float):
    """Emit workflow completion event."""
    await get_hook_integration().on_workflow_complete(workflow_id, project_id, success, final_confidence)


async def emit_task_complete(workflow_id: str, project_id: str, task_id: str, success: bool, confidence: float):
    """Emit task completion event."""
    await get_hook_integration().on_task_complete(workflow_id, project_id, task_id, success, confidence)


async def emit_milestone_reached(workflow_id: str, project_id: str, milestone_name: str, milestone_value: Any):
    """Emit milestone reached event."""
    await get_hook_integration().on_milestone_reached(workflow_id, project_id, milestone_name, milestone_value)


async def emit_progress_update(workflow_id: str, project_id: str, progress: float, completed_tasks: int, total_tasks: int):
    """Emit progress update event."""
    await get_hook_integration().on_progress_update(workflow_id, project_id, progress, completed_tasks, total_tasks)


async def emit_error_occurred(workflow_id: str, project_id: str, error_type: str, error_message: str, task_id: str = None):
    """Emit error occurrence event."""
    await get_hook_integration().on_error_occurred(workflow_id, project_id, error_type, error_message, task_id)