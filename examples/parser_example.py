#!/usr/bin/env python3
"""
Parser Agent Example - Ultimate Agentic StarterKit

This example demonstrates how to use the Parser Agent to extract tasks and milestones
from project specifications using semantic search and pattern matching.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.parser_agent import ParserAgent
from core.models import ProjectTask, AgentType, create_project_task
from core.logger import get_logger

# Setup logging
logger = get_logger("parser_example")


def display_results(result):
    """Display parser agent results in a formatted way."""
    print("=" * 60)
    print("PARSER AGENT EXECUTION RESULTS")
    print("=" * 60)
    
    if result.success:
        print(f"✓ Parser execution successful!")
        print(f"  Agent ID: {result.agent_id}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        print(f"  Timestamp: {result.timestamp}")
        
        # Display extracted tasks
        output = result.output
        if output and isinstance(output, dict):
            tasks = output.get('tasks', [])
            print(f"  Extracted tasks: {len(tasks)}")
            print(f"  Chunks processed: {output.get('chunks_processed', 0)}")
            print(f"  Extraction method: {output.get('extraction_method', 'unknown')}")
            
            # Display individual tasks
            if tasks:
                print("\n--- EXTRACTED TASKS ---")
                for i, task in enumerate(tasks, 1):
                    print(f"{i}. {task.get('title', 'Unknown Title')}")
                    print(f"   Type: {task.get('type', 'Unknown')}")
                    print(f"   Agent: {task.get('agent_type', 'Unknown')}")
                    print(f"   Confidence: {task.get('confidence', 0.0):.2f}")
                    print(f"   Description: {task.get('description', 'No description')[:100]}...")
                    print(f"   Source: {task.get('source_query', 'Unknown')}")
                    print()
        else:
            print("  No task data in output")
    else:
        print(f"✗ Parser execution failed: {result.error}")
        print(f"  Agent ID: {result.agent_id}")
        print(f"  Execution time: {result.execution_time:.2f}s")


async def example_1_basic_parsing():
    """Example 1: Basic parsing of a simple project specification."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Project Specification Parsing")
    print("="*60)
    
    # Sample project specification
    project_spec = """
    # Web Application Project
    
    ## Goal
    Build a modern web application with user authentication and data management.
    
    ## Tasks
    
    ### Task 1: Setup Project Structure
    - Create Next.js application with TypeScript
    - Configure ESLint and Prettier
    - Setup project dependencies
    
    ### Task 2: Implement Authentication
    - Create login/register forms
    - Setup JWT authentication
    - Add protected routes
    - Implement password reset functionality
    
    ### Task 3: Database Integration
    - Setup PostgreSQL database
    - Create user model with proper validation
    - Implement CRUD operations
    - Add database migrations
    
    ### Task 4: API Development
    - Create REST API endpoints
    - Add input validation
    - Implement error handling
    - Add API documentation
    
    ### Task 5: Frontend Development
    - Build user interface components
    - Implement responsive design
    - Add state management
    - Create user dashboard
    
    ### Task 6: Testing
    - Write unit tests for components
    - Add integration tests
    - Setup end-to-end testing
    - Add test coverage reporting
    """
    
    # Create parser agent
    parser = ParserAgent()
    
    # Create task
    task = create_project_task(
        title="Parse Web App Project",
        description=project_spec,
        task_type="PARSE",
        agent_type=AgentType.PARSER
    )
    
    # Execute parser
    print("Executing parser agent...")
    result = await parser.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def example_2_markdown_list_parsing():
    """Example 2: Parsing markdown lists with different formats."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Markdown List Parsing")
    print("="*60)
    
    # Sample markdown with different list formats
    markdown_spec = """
    # API Service Development
    
    ## Implementation Tasks
    
    * Create FastAPI application structure
    * Setup database models using SQLAlchemy
    * Implement authentication middleware
    * Add CORS configuration
    
    ## Development Tasks
    
    1. Create user registration endpoint
    2. Implement login with JWT tokens
    3. Add password hashing with bcrypt
    4. Create protected routes
    5. Add input validation with Pydantic
    
    ## Quality Assurance
    
    - [ ] Write comprehensive unit tests
    - [ ] Add integration tests for API endpoints
    - [ ] Setup automated testing pipeline
    - [ ] Add code coverage reporting
    - [ ] Create API documentation
    
    ## Deployment
    
    - Deploy to production environment
    - Setup monitoring and logging
    - Configure SSL certificates
    - Add health check endpoints
    """
    
    # Create parser agent
    parser = ParserAgent()
    
    # Create task
    task = create_project_task(
        title="Parse API Service Markdown",
        description=markdown_spec,
        task_type="PARSE",
        agent_type=AgentType.PARSER
    )
    
    # Execute parser
    print("Executing parser agent...")
    result = await parser.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def example_3_complex_specification():
    """Example 3: Complex project specification with nested tasks."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Complex Project Specification")
    print("="*60)
    
    # Complex project specification
    complex_spec = """
    # E-commerce Platform Development
    
    ## Phase 1: Foundation
    ### Backend Infrastructure
    - Setup microservices architecture
    - Configure API gateway
    - Implement service discovery
    - Setup message queuing system
    
    ### Database Design
    - Design user management schema
    - Create product catalog structure
    - Setup order management tables
    - Implement payment processing schema
    
    ## Phase 2: Core Features
    ### User Management
    TODO: Implement user registration system
    TODO: Add social media authentication
    TODO: Create user profile management
    TODO: Add role-based access control
    
    ### Product Management
    - Create product catalog API
    - Implement product search functionality
    - Add product reviews and ratings
    - Setup inventory management
    
    ### Order Processing
    - Build shopping cart functionality
    - Implement checkout process
    - Add payment gateway integration
    - Create order tracking system
    
    ## Phase 3: Advanced Features
    ### Analytics and Reporting
    - Add user behavior tracking
    - Implement sales reporting
    - Create admin dashboard
    - Add performance metrics
    
    ### Optimization
    - Implement caching strategies
    - Add search optimization
    - Setup CDN for static assets
    - Add performance monitoring
    
    ## Phase 4: Testing and Deployment
    ### Quality Assurance
    - Write comprehensive test suite
    - Add automated testing pipeline
    - Implement security testing
    - Add performance testing
    
    ### Deployment
    - Setup containerization with Docker
    - Configure Kubernetes deployment
    - Add monitoring and alerting
    - Create deployment automation
    """
    
    # Create parser agent with custom configuration
    parser_config = {
        'max_tasks_per_chunk': 10,
        'similarity_threshold': 0.6,
        'chunk_size': 400
    }
    parser = ParserAgent(config=parser_config)
    
    # Create task
    task = create_project_task(
        title="Parse E-commerce Platform",
        description=complex_spec,
        task_type="PARSE",
        agent_type=AgentType.PARSER
    )
    
    # Execute parser
    print("Executing parser agent...")
    result = await parser.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def example_4_error_handling():
    """Example 4: Error handling and edge cases."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Error Handling and Edge Cases")
    print("="*60)
    
    # Test with empty content
    print("Testing with empty content...")
    parser = ParserAgent()
    
    empty_task = create_project_task(
        title="Parse Empty Content",
        description="",
        task_type="PARSE",
        agent_type=AgentType.PARSER
    )
    
    result = await parser.execute(empty_task)
    display_results(result)
    
    # Test with invalid task type
    print("\nTesting with invalid task type...")
    invalid_task = create_project_task(
        title="Invalid Task",
        description="This is a test with invalid task type",
        task_type="INVALID",
        agent_type=AgentType.PARSER
    )
    
    result = await parser.execute(invalid_task)
    display_results(result)
    
    return result


async def example_5_performance_test():
    """Example 5: Performance testing with large content."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Performance Testing")
    print("="*60)
    
    # Generate large content
    large_content = """
    # Large Scale System Development
    
    ## System Architecture
    """
    
    # Add many tasks
    for i in range(1, 51):
        large_content += f"""
    ### Component {i}
    - Task {i}.1: Design component architecture
    - Task {i}.2: Implement core functionality
    - Task {i}.3: Add error handling
    - Task {i}.4: Write unit tests
    - Task {i}.5: Create documentation
    """
    
    # Create parser agent
    parser = ParserAgent()
    
    # Create task
    task = create_project_task(
        title="Parse Large System",
        description=large_content,
        task_type="PARSE",
        agent_type=AgentType.PARSER
    )
    
    # Execute parser
    print("Executing parser agent on large content...")
    result = await parser.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def main():
    """Main function to run all examples."""
    print("Ultimate Agentic StarterKit - Parser Agent Examples")
    print("=" * 60)
    
    try:
        # Run examples
        await example_1_basic_parsing()
        await example_2_markdown_list_parsing()
        await example_3_complex_specification()
        await example_4_error_handling()
        await example_5_performance_test()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)