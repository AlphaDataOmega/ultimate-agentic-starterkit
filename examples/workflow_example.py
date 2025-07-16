#!/usr/bin/env python3
"""
Workflow Example - Ultimate Agentic StarterKit

This example demonstrates complete workflow execution from PRP to working code,
showing agent coordination, progress tracking, and voice alerts.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add StarterKit to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'StarterKit'))

from StarterKit.workflows.project_builder import LangGraphWorkflowManager
from StarterKit.core.voice_alerts import get_voice_alerts
from StarterKit.core.logger import get_logger

# Setup logging
logger = get_logger("workflow_example")


def display_workflow_results(result):
    """Display workflow execution results in a formatted way."""
    print("=" * 60)
    print("WORKFLOW EXECUTION RESULTS")
    print("=" * 60)
    
    status = result.get("workflow_status", "unknown")
    confidence = result.get("overall_confidence", 0.0)
    
    print(f"Status: {status}")
    print(f"Overall Confidence: {confidence:.2f}")
    
    # Display task results
    completed_tasks = result.get("completed_tasks", [])
    failed_tasks = result.get("failed_tasks", [])
    
    print(f"Completed Tasks: {len(completed_tasks)}")
    print(f"Failed Tasks: {len(failed_tasks)}")
    
    if completed_tasks:
        print("\n--- COMPLETED TASKS ---")
        for task_id in completed_tasks:
            print(f"  ✓ {task_id}")
    
    if failed_tasks:
        print("\n--- FAILED TASKS ---")
        for task_id in failed_tasks:
            print(f"  ✗ {task_id}")
    
    # Display agent results
    agent_results = result.get("agent_results", [])
    if agent_results:
        print(f"\n--- AGENT EXECUTION RESULTS ---")
        for i, agent_result in enumerate(agent_results, 1):
            print(f"{i}. Task: {agent_result.get('task_id', 'Unknown')}")
            print(f"   Agent: {agent_result.get('agent_type', 'Unknown')}")
            print(f"   Success: {agent_result.get('success', False)}")
            print(f"   Confidence: {agent_result.get('confidence', 0.0):.2f}")
            print(f"   Execution Time: {agent_result.get('execution_time', 0.0):.2f}s")
            if agent_result.get('error'):
                print(f"   Error: {agent_result.get('error')}")
    
    # Display metrics
    metrics = result.get("metrics", {})
    if metrics:
        print(f"\n--- WORKFLOW METRICS ---")
        print(f"  Total Tasks: {metrics.get('total_tasks', 0)}")
        print(f"  Duration: {metrics.get('duration', 0.0):.2f}s")
        print(f"  Success Rate: {metrics.get('success_rate', 0.0):.2f}")
        print(f"  Average Confidence: {metrics.get('average_confidence', 0.0):.2f}")


async def example_1_simple_api():
    """Example 1: Simple REST API project workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple REST API Workflow")
    print("="*60)
    
    # Sample project specification
    project_spec = {
        "title": "Simple REST API",
        "description": "Build a REST API for todo management",
        "project_type": "api",
        "tasks": [
            {
                "id": "task-1",
                "title": "Setup FastAPI Project",
                "description": "Create FastAPI application structure with dependencies",
                "type": "CREATE",
                "agent_type": "coder"
            },
            {
                "id": "task-2",
                "title": "Create Todo Model",
                "description": "Define Pydantic models for todo items",
                "type": "CREATE",
                "agent_type": "coder"
            },
            {
                "id": "task-3",
                "title": "Implement CRUD Endpoints",
                "description": "Create REST endpoints for todo operations",
                "type": "CREATE",
                "agent_type": "coder"
            },
            {
                "id": "task-4",
                "title": "Add Unit Tests",
                "description": "Create pytest tests for all endpoints",
                "type": "CREATE",
                "agent_type": "coder"
            },
            {
                "id": "task-5",
                "title": "Run Tests",
                "description": "Execute unit tests and validate results",
                "type": "TEST",
                "agent_type": "tester"
            }
        ],
        "requirements": {
            "python_version": "3.11",
            "frameworks": ["FastAPI", "pytest"],
            "database": "in-memory"
        },
        "validation_criteria": {
            "test_coverage": 80,
            "code_quality": "high"
        }
    }
    
    # Initialize components
    workflow_manager = LangGraphWorkflowManager()
    voice_alerts = get_voice_alerts()
    
    # Voice notification
    voice_alerts.speak_milestone("Starting workflow execution example")
    
    # Execute workflow
    print("Executing complete workflow...")
    result = await workflow_manager.execute_workflow(project_spec)
    
    # Display results
    display_workflow_results(result)
    
    if result.get('workflow_status') == 'completed':
        voice_alerts.speak_milestone("Workflow completed successfully!")
        print("✓ Workflow completed successfully!")
    else:
        voice_alerts.speak_error("Workflow execution failed")
        print("✗ Workflow execution failed")
    
    return result


async def example_2_web_application():
    """Example 2: Web application with authentication workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Web Application Workflow")
    print("="*60)
    
    # Web application project specification
    project_spec = {
        "title": "Web Application with Authentication",
        "description": "Build a web application with user authentication",
        "project_type": "web",
        "tasks": [
            {
                "id": "parse-requirements",
                "title": "Parse Requirements",
                "description": "Extract detailed requirements from project specification",
                "type": "PARSE",
                "agent_type": "parser"
            },
            {
                "id": "setup-backend",
                "title": "Setup Backend",
                "description": "Create FastAPI backend with authentication",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["parse-requirements"]
            },
            {
                "id": "create-frontend",
                "title": "Create Frontend",
                "description": "Build React frontend with authentication UI",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["setup-backend"]
            },
            {
                "id": "add-tests",
                "title": "Add Tests",
                "description": "Create comprehensive test suite",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["create-frontend"]
            },
            {
                "id": "validate-code",
                "title": "Validate Code",
                "description": "Review and validate code quality",
                "type": "VALIDATE",
                "agent_type": "advisor",
                "dependencies": ["add-tests"]
            },
            {
                "id": "run-tests",
                "title": "Run Tests",
                "description": "Execute all tests and generate reports",
                "type": "TEST",
                "agent_type": "tester",
                "dependencies": ["validate-code"]
            }
        ],
        "requirements": {
            "backend": "FastAPI",
            "frontend": "React",
            "database": "PostgreSQL",
            "authentication": "JWT"
        },
        "validation_criteria": {
            "test_coverage": 85,
            "code_quality": "high",
            "security": "high"
        }
    }
    
    # Initialize components
    workflow_manager = LangGraphWorkflowManager()
    voice_alerts = get_voice_alerts()
    
    # Voice notification
    voice_alerts.speak_milestone("Starting web application workflow")
    
    # Execute workflow
    print("Executing web application workflow...")
    result = await workflow_manager.execute_workflow(project_spec)
    
    # Display results
    display_workflow_results(result)
    
    if result.get('workflow_status') == 'completed':
        voice_alerts.speak_milestone("Web application workflow completed!")
        print("✓ Web application workflow completed successfully!")
    else:
        voice_alerts.speak_error("Web application workflow failed")
        print("✗ Web application workflow failed")
    
    return result


async def example_3_data_processing():
    """Example 3: Data processing pipeline workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Processing Pipeline")
    print("="*60)
    
    # Data processing project specification
    project_spec = {
        "title": "Data Processing Pipeline",
        "description": "Build a data processing pipeline for analytics",
        "project_type": "data",
        "tasks": [
            {
                "id": "data-extractor",
                "title": "Data Extractor",
                "description": "Create data extraction from various sources",
                "type": "CREATE",
                "agent_type": "coder"
            },
            {
                "id": "data-transformer",
                "title": "Data Transformer",
                "description": "Transform and clean extracted data",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["data-extractor"]
            },
            {
                "id": "data-loader",
                "title": "Data Loader",
                "description": "Load processed data into target system",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["data-transformer"]
            },
            {
                "id": "pipeline-orchestrator",
                "title": "Pipeline Orchestrator",
                "description": "Orchestrate the entire pipeline",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["data-loader"]
            },
            {
                "id": "test-pipeline",
                "title": "Test Pipeline",
                "description": "Test the complete pipeline",
                "type": "TEST",
                "agent_type": "tester",
                "dependencies": ["pipeline-orchestrator"]
            }
        ],
        "requirements": {
            "python_version": "3.11",
            "frameworks": ["pandas", "airflow"],
            "database": "PostgreSQL"
        },
        "validation_criteria": {
            "performance": "high",
            "reliability": "high"
        }
    }
    
    # Initialize components
    workflow_manager = LangGraphWorkflowManager()
    voice_alerts = get_voice_alerts()
    
    # Voice notification
    voice_alerts.speak_milestone("Starting data processing pipeline workflow")
    
    # Execute workflow
    print("Executing data processing pipeline workflow...")
    result = await workflow_manager.execute_workflow(project_spec)
    
    # Display results
    display_workflow_results(result)
    
    if result.get('workflow_status') == 'completed':
        voice_alerts.speak_milestone("Data processing pipeline completed!")
        print("✓ Data processing pipeline completed successfully!")
    else:
        voice_alerts.speak_error("Data processing pipeline failed")
        print("✗ Data processing pipeline failed")
    
    return result


async def example_4_workflow_control():
    """Example 4: Workflow control and monitoring."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Workflow Control and Monitoring")
    print("="*60)
    
    # Simple project for control demonstration
    project_spec = {
        "title": "Workflow Control Demo",
        "description": "Demonstrate workflow control capabilities",
        "project_type": "general",
        "tasks": [
            {
                "id": "task-1",
                "title": "First Task",
                "description": "Execute first task",
                "type": "CREATE",
                "agent_type": "coder"
            },
            {
                "id": "task-2",
                "title": "Second Task",
                "description": "Execute second task",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["task-1"]
            },
            {
                "id": "task-3",
                "title": "Third Task",
                "description": "Execute third task",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["task-2"]
            }
        ]
    }
    
    # Initialize workflow manager
    workflow_manager = LangGraphWorkflowManager()
    voice_alerts = get_voice_alerts()
    
    # Start workflow execution
    print("Starting workflow with monitoring...")
    voice_alerts.speak_milestone("Starting workflow control demo")
    
    # Execute workflow with monitoring
    async def monitor_workflow():
        """Monitor workflow execution."""
        while workflow_manager.is_running:
            await asyncio.sleep(1)
            current_state = workflow_manager.get_current_state()
            if current_state:
                status = current_state.get("workflow_status", "unknown")
                current_task = current_state.get("current_task", {})
                if current_task:
                    task_id = current_task.get("id", "unknown")
                    print(f"Status: {status}, Current Task: {task_id}")
    
    # Run workflow and monitoring concurrently
    workflow_task = asyncio.create_task(
        workflow_manager.execute_workflow(project_spec)
    )
    monitor_task = asyncio.create_task(monitor_workflow())
    
    # Wait for workflow completion
    result = await workflow_task
    monitor_task.cancel()
    
    # Display results
    display_workflow_results(result)
    
    # Display workflow metrics
    metrics = workflow_manager.get_workflow_metrics()
    print(f"\n--- WORKFLOW METRICS ---")
    print(f"Metrics: {metrics}")
    
    return result


async def example_5_error_recovery():
    """Example 5: Error handling and recovery."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Error Handling and Recovery")
    print("="*60)
    
    # Project with potential failure points
    project_spec = {
        "title": "Error Recovery Demo",
        "description": "Demonstrate error handling and recovery",
        "project_type": "general",
        "tasks": [
            {
                "id": "good-task",
                "title": "Good Task",
                "description": "A task that should succeed",
                "type": "CREATE",
                "agent_type": "coder"
            },
            {
                "id": "problematic-task",
                "title": "Problematic Task",
                "description": "A task that might fail",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["good-task"]
            },
            {
                "id": "recovery-task",
                "title": "Recovery Task",
                "description": "A task to recover from failure",
                "type": "CREATE",
                "agent_type": "coder",
                "dependencies": ["problematic-task"]
            }
        ]
    }
    
    # Initialize workflow manager
    workflow_manager = LangGraphWorkflowManager()
    voice_alerts = get_voice_alerts()
    
    # Voice notification
    voice_alerts.speak_milestone("Starting error recovery demo")
    
    # Execute workflow
    print("Executing workflow with error handling...")
    result = await workflow_manager.execute_workflow(project_spec)
    
    # Display results
    display_workflow_results(result)
    
    # Check state history
    history = workflow_manager.get_state_history()
    print(f"\n--- STATE HISTORY ---")
    print(f"History entries: {len(history)}")
    
    return result


async def main():
    """Main function to run all workflow examples."""
    print("Ultimate Agentic StarterKit - Workflow Examples")
    print("=" * 60)
    
    try:
        # Run examples
        await example_1_simple_api()
        await example_2_web_application()
        await example_3_data_processing()
        await example_4_workflow_control()
        await example_5_error_recovery()
        
        print("\n" + "="*60)
        print("All workflow examples completed!")
        print("="*60)
        print("Note: Some examples may fail if dependencies are not configured")
        print("Check the configuration files and ensure all agents are properly set up")
        
    except Exception as e:
        logger.error(f"Error running workflow examples: {e}")
        print(f"Error running workflow examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)