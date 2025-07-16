#!/usr/bin/env python3
"""
Agent Framework Demo

This script demonstrates the basic functionality of the AI Agent Framework.
"""

import asyncio
import sys
import os

# Add the StarterKit directory to the path
sys.path.append(os.path.dirname(__file__))

from agents.factory import create_agent, get_agent_factory
from core.models import ProjectTask, AgentType, create_project_task


async def demo_parser_agent():
    """Demo the Parser Agent functionality."""
    print("=== Parser Agent Demo ===")
    
    # Create a parser agent
    parser_agent = create_agent(AgentType.PARSER)
    print(f"Created parser agent: {parser_agent.agent.agent_id}")
    
    # Create a task for parsing
    task = create_project_task(
        title="Parse Project Requirements",
        description="""
        # Project Requirements Document
        
        ## Core Features
        - Create user authentication system
        - Implement data validation
        - Add error handling
        - Write unit tests
        
        ## Secondary Features
        - Build dashboard interface
        - Set up logging system
        - Create API documentation
        
        ## Tasks
        1. Set up development environment
        2. Create database schema
        3. Implement user registration
        4. Add authentication middleware
        TODO: Write comprehensive tests
        [ ] Deploy to production
        """,
        task_type="CREATE",
        agent_type=AgentType.PARSER
    )
    
    # Execute the task
    try:
        result = await parser_agent.execute_task(task)
        print(f"Parser result - Success: {result.success}")
        print(f"Parser result - Confidence: {result.confidence:.2f}")
        print(f"Parser result - Execution time: {result.execution_time:.2f}s")
        
        if result.success and result.output:
            tasks = result.output.get('tasks', [])
            print(f"Extracted {len(tasks)} tasks:")
            for i, task_data in enumerate(tasks[:3]):  # Show first 3 tasks
                print(f"  {i+1}. {task_data['title']}")
                print(f"     Type: {task_data['type']}")
                print(f"     Agent: {task_data['agent_type']}")
                print(f"     Confidence: {task_data['confidence']:.2f}")
        else:
            print(f"Parser failed: {result.error}")
            
    except Exception as e:
        print(f"Parser demo failed: {str(e)}")
    
    print()


async def demo_coder_agent():
    """Demo the Coder Agent functionality."""
    print("=== Coder Agent Demo ===")
    
    # Create a coder agent
    coder_agent = create_agent(AgentType.CODER, config={'timeout': 60})
    print(f"Created coder agent: {coder_agent.agent.agent_id}")
    
    # Create a simple coding task
    task = create_project_task(
        title="Create Hello World Function",
        description="""
        Create a simple Python function that:
        1. Takes a name parameter
        2. Returns a greeting message
        3. Includes proper error handling
        4. Has comprehensive docstring
        5. Includes a simple test function
        """,
        task_type="CREATE",
        agent_type=AgentType.CODER
    )
    
    # Execute the task
    try:
        result = await coder_agent.execute_task(task)
        print(f"Coder result - Success: {result.success}")
        print(f"Coder result - Confidence: {result.confidence:.2f}")
        print(f"Coder result - Execution time: {result.execution_time:.2f}s")
        
        if result.success and result.output:
            files = result.output.get('files', [])
            print(f"Generated {len(files)} files:")
            for file_info in files:
                print(f"  - {file_info.get('path', 'unknown')}")
                print(f"    Language: {file_info.get('language', 'unknown')}")
                print(f"    Lines: {file_info.get('lines', 0)}")
        else:
            print(f"Coder failed: {result.error}")
            
    except Exception as e:
        print(f"Coder demo failed: {str(e)}")
    
    print()


async def demo_tester_agent():
    """Demo the Tester Agent functionality."""
    print("=== Tester Agent Demo ===")
    
    # Create a tester agent
    tester_agent = create_agent(AgentType.TESTER, config={'timeout': 30})
    print(f"Created tester agent: {tester_agent.agent.agent_id}")
    
    # Create a testing task
    task = create_project_task(
        title="Run Unit Tests",
        description="Run all unit tests in the project and report results",
        task_type="TEST",
        agent_type=AgentType.TESTER
    )
    
    # Execute the task
    try:
        result = await tester_agent.execute_task(task)
        print(f"Tester result - Success: {result.success}")
        print(f"Tester result - Confidence: {result.confidence:.2f}")
        print(f"Tester result - Execution time: {result.execution_time:.2f}s")
        
        if result.output:
            test_results = result.output.get('test_results', {})
            print(f"Framework: {test_results.get('framework', 'unknown')}")
            print(f"Total tests: {test_results.get('total_tests', 0)}")
            print(f"Passed: {test_results.get('passed_tests', 0)}")
            print(f"Failed: {test_results.get('failed_tests', 0)}")
            print(f"Errors: {test_results.get('error_tests', 0)}")
        else:
            print(f"Tester failed: {result.error}")
            
    except Exception as e:
        print(f"Tester demo failed: {str(e)}")
    
    print()


async def demo_advisor_agent():
    """Demo the Advisor Agent functionality."""
    print("=== Advisor Agent Demo ===")
    
    # Create an advisor agent
    advisor_agent = create_agent(AgentType.ADVISOR, config={'timeout': 60})
    print(f"Created advisor agent: {advisor_agent.agent.agent_id}")
    
    # Create a code review task
    task = create_project_task(
        title="Review Code Quality",
        description="Review the generated code for quality, security, and best practices",
        task_type="VALIDATE",
        agent_type=AgentType.ADVISOR
    )
    
    # Execute the task
    try:
        result = await advisor_agent.execute_task(task)
        print(f"Advisor result - Success: {result.success}")
        print(f"Advisor result - Confidence: {result.confidence:.2f}")
        print(f"Advisor result - Execution time: {result.execution_time:.2f}s")
        
        if result.success and result.output:
            review_results = result.output.get('review_results', {})
            print(f"Overall score: {review_results.get('overall_score', 0.0):.2f}")
            print(f"Files reviewed: {result.output.get('files_reviewed', 0)}")
            
            issues = review_results.get('issues_found', [])
            if issues:
                print(f"Issues found: {len(issues)}")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"  - {issue.get('description', 'No description')}")
            else:
                print("No issues found")
        else:
            print(f"Advisor failed: {result.error}")
            
    except Exception as e:
        print(f"Advisor demo failed: {str(e)}")
    
    print()


async def demo_agent_factory():
    """Demo the Agent Factory functionality."""
    print("=== Agent Factory Demo ===")
    
    factory = get_agent_factory()
    registry = factory.get_registry()
    
    # Show registry stats
    stats = registry.get_registry_stats()
    print(f"Total agents: {stats['total_agents']}")
    print(f"Agents by type: {stats['agents_by_type']}")
    print(f"Agents by status: {stats['agents_by_status']}")
    
    print()


async def main():
    """Run all demos."""
    print("AI Agent Framework Demo")
    print("=" * 50)
    
    try:
        await demo_parser_agent()
        await demo_coder_agent()
        await demo_tester_agent()
        await demo_advisor_agent()
        await demo_agent_factory()
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())