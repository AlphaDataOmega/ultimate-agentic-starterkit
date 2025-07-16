"""
CLI interface for StarterKit workflow hooks.

This module provides a command-line interface for Claude Code hooks integration.
"""

import sys
import json
import asyncio
import argparse
from typing import Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.claude_code_hooks import (
    HookType, 
    get_hook_integration,
    emit_workflow_start,
    emit_workflow_complete,
    emit_task_complete,
    emit_milestone_reached,
    emit_progress_update,
    emit_error_occurred
)
from core.logger import get_logger


def parse_hook_data(data_str: str) -> Dict[str, Any]:
    """Parse hook data from command line argument."""
    try:
        if data_str.startswith('{') and data_str.endswith('}'):
            return json.loads(data_str)
        else:
            # Try to parse as key=value pairs
            data = {}
            for pair in data_str.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    # Try to convert to appropriate type
                    try:
                        value = json.loads(value)
                    except:
                        pass  # Keep as string
                    data[key.strip()] = value
            return data
    except:
        return {}


async def handle_workflow_start(args):
    """Handle workflow start hook."""
    logger = get_logger("hook_workflow_start")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    total_tasks = args.total_tasks or 0
    
    logger.info(f"Workflow start hook: {workflow_id}")
    
    try:
        await emit_workflow_start(workflow_id, project_id, total_tasks)
        print(f"✓ Workflow start hook executed successfully")
    except Exception as e:
        logger.error(f"Workflow start hook failed: {e}")
        print(f"✗ Workflow start hook failed: {e}")
        return 1
    
    return 0


async def handle_workflow_complete(args):
    """Handle workflow complete hook."""
    logger = get_logger("hook_workflow_complete")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    success = args.success if args.success is not None else True
    final_confidence = args.confidence or 0.8
    
    logger.info(f"Workflow complete hook: {workflow_id}, success={success}")
    
    try:
        await emit_workflow_complete(workflow_id, project_id, success, final_confidence)
        print(f"✓ Workflow complete hook executed successfully")
    except Exception as e:
        logger.error(f"Workflow complete hook failed: {e}")
        print(f"✗ Workflow complete hook failed: {e}")
        return 1
    
    return 0


async def handle_workflow_failure(args):
    """Handle workflow failure hook."""
    logger = get_logger("hook_workflow_failure")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    error_message = args.error_message or "Unknown error"
    
    logger.info(f"Workflow failure hook: {workflow_id}")
    
    try:
        await emit_workflow_complete(workflow_id, project_id, False, 0.0)
        await emit_error_occurred(workflow_id, project_id, "workflow_failure", error_message)
        print(f"✓ Workflow failure hook executed successfully")
    except Exception as e:
        logger.error(f"Workflow failure hook failed: {e}")
        print(f"✗ Workflow failure hook failed: {e}")
        return 1
    
    return 0


async def handle_task_start(args):
    """Handle task start hook."""
    logger = get_logger("hook_task_start")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    task_id = args.task_id or "unknown"
    agent_type = args.agent_type or "unknown"
    
    logger.info(f"Task start hook: {task_id}")
    
    try:
        hook_integration = get_hook_integration()
        await hook_integration.on_task_start(workflow_id, project_id, task_id, agent_type)
        print(f"✓ Task start hook executed successfully")
    except Exception as e:
        logger.error(f"Task start hook failed: {e}")
        print(f"✗ Task start hook failed: {e}")
        return 1
    
    return 0


async def handle_task_complete(args):
    """Handle task complete hook."""
    logger = get_logger("hook_task_complete")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    task_id = args.task_id or "unknown"
    success = args.success if args.success is not None else True
    confidence = args.confidence or 0.8
    
    logger.info(f"Task complete hook: {task_id}, success={success}")
    
    try:
        await emit_task_complete(workflow_id, project_id, task_id, success, confidence)
        print(f"✓ Task complete hook executed successfully")
    except Exception as e:
        logger.error(f"Task complete hook failed: {e}")
        print(f"✗ Task complete hook failed: {e}")
        return 1
    
    return 0


async def handle_milestone_reached(args):
    """Handle milestone reached hook."""
    logger = get_logger("hook_milestone_reached")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    milestone_name = args.milestone_name or "Unknown Milestone"
    milestone_value = args.milestone_value or 0
    
    logger.info(f"Milestone reached hook: {milestone_name}={milestone_value}")
    
    try:
        await emit_milestone_reached(workflow_id, project_id, milestone_name, milestone_value)
        print(f"✓ Milestone reached hook executed successfully")
    except Exception as e:
        logger.error(f"Milestone reached hook failed: {e}")
        print(f"✗ Milestone reached hook failed: {e}")
        return 1
    
    return 0


async def handle_progress_update(args):
    """Handle progress update hook."""
    logger = get_logger("hook_progress_update")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    progress = args.progress or 0.0
    completed_tasks = args.completed_tasks or 0
    total_tasks = args.total_tasks or 0
    
    logger.info(f"Progress update hook: {progress:.1%}")
    
    try:
        await emit_progress_update(workflow_id, project_id, progress, completed_tasks, total_tasks)
        print(f"✓ Progress update hook executed successfully")
    except Exception as e:
        logger.error(f"Progress update hook failed: {e}")
        print(f"✗ Progress update hook failed: {e}")
        return 1
    
    return 0


async def handle_error_occurred(args):
    """Handle error occurred hook."""
    logger = get_logger("hook_error_occurred")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    error_type = args.error_type or "unknown"
    error_message = args.error_message or "Unknown error"
    task_id = args.task_id
    
    logger.info(f"Error occurred hook: {error_type}")
    
    try:
        await emit_error_occurred(workflow_id, project_id, error_type, error_message, task_id)
        print(f"✓ Error occurred hook executed successfully")
    except Exception as e:
        logger.error(f"Error occurred hook failed: {e}")
        print(f"✗ Error occurred hook failed: {e}")
        return 1
    
    return 0


async def handle_advisor_review(args):
    """Handle advisor review hook."""
    logger = get_logger("hook_advisor_review")
    
    workflow_id = args.workflow_id or "unknown"
    project_id = args.project_id or "unknown"
    review_type = args.review_type or "standard"
    
    logger.info(f"Advisor review hook: {review_type}")
    
    try:
        # For now, just log the advisor review request
        print(f"✓ Advisor review hook executed successfully (review_type: {review_type})")
    except Exception as e:
        logger.error(f"Advisor review hook failed: {e}")
        print(f"✗ Advisor review hook failed: {e}")
        return 1
    
    return 0


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="StarterKit Workflow Hooks CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Workflow start
    workflow_start_parser = subparsers.add_parser('workflow_start', help='Handle workflow start')
    workflow_start_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    workflow_start_parser.add_argument('--project-id', required=True, help='Project ID')
    workflow_start_parser.add_argument('--total-tasks', type=int, default=0, help='Total number of tasks')
    
    # Workflow complete
    workflow_complete_parser = subparsers.add_parser('workflow_complete', help='Handle workflow completion')
    workflow_complete_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    workflow_complete_parser.add_argument('--project-id', required=True, help='Project ID')
    workflow_complete_parser.add_argument('--success', type=bool, default=True, help='Success status')
    workflow_complete_parser.add_argument('--confidence', type=float, default=0.8, help='Final confidence')
    
    # Workflow failure
    workflow_failure_parser = subparsers.add_parser('workflow_failure', help='Handle workflow failure')
    workflow_failure_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    workflow_failure_parser.add_argument('--project-id', required=True, help='Project ID')
    workflow_failure_parser.add_argument('--error-message', help='Error message')
    
    # Task start
    task_start_parser = subparsers.add_parser('task_start', help='Handle task start')
    task_start_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    task_start_parser.add_argument('--project-id', required=True, help='Project ID')
    task_start_parser.add_argument('--task-id', required=True, help='Task ID')
    task_start_parser.add_argument('--agent-type', help='Agent type')
    
    # Task complete
    task_complete_parser = subparsers.add_parser('task_complete', help='Handle task completion')
    task_complete_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    task_complete_parser.add_argument('--project-id', required=True, help='Project ID')
    task_complete_parser.add_argument('--task-id', required=True, help='Task ID')
    task_complete_parser.add_argument('--success', type=bool, default=True, help='Success status')
    task_complete_parser.add_argument('--confidence', type=float, default=0.8, help='Task confidence')
    
    # Milestone reached
    milestone_parser = subparsers.add_parser('milestone_reached', help='Handle milestone reached')
    milestone_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    milestone_parser.add_argument('--project-id', required=True, help='Project ID')
    milestone_parser.add_argument('--milestone-name', required=True, help='Milestone name')
    milestone_parser.add_argument('--milestone-value', help='Milestone value')
    
    # Progress update
    progress_parser = subparsers.add_parser('progress_update', help='Handle progress update')
    progress_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    progress_parser.add_argument('--project-id', required=True, help='Project ID')
    progress_parser.add_argument('--progress', type=float, default=0.0, help='Progress value (0.0-1.0)')
    progress_parser.add_argument('--completed-tasks', type=int, default=0, help='Completed tasks')
    progress_parser.add_argument('--total-tasks', type=int, default=0, help='Total tasks')
    
    # Error occurred
    error_parser = subparsers.add_parser('error_occurred', help='Handle error occurrence')
    error_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    error_parser.add_argument('--project-id', required=True, help='Project ID')
    error_parser.add_argument('--error-type', required=True, help='Error type')
    error_parser.add_argument('--error-message', help='Error message')
    error_parser.add_argument('--task-id', help='Task ID (optional)')
    
    # Advisor review
    advisor_parser = subparsers.add_parser('advisor_review', help='Handle advisor review')
    advisor_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    advisor_parser.add_argument('--project-id', required=True, help='Project ID')
    advisor_parser.add_argument('--review-type', default='standard', help='Review type')
    
    return parser


async def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Map commands to handlers
    handlers = {
        'workflow_start': handle_workflow_start,
        'workflow_complete': handle_workflow_complete,
        'workflow_failure': handle_workflow_failure,
        'task_start': handle_task_start,
        'task_complete': handle_task_complete,
        'milestone_reached': handle_milestone_reached,
        'progress_update': handle_progress_update,
        'error_occurred': handle_error_occurred,
        'advisor_review': handle_advisor_review
    }
    
    handler = handlers.get(args.command)
    if not handler:
        print(f"Unknown command: {args.command}")
        return 1
    
    try:
        return await handler(args)
    except Exception as e:
        print(f"Hook execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)