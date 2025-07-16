#!/usr/bin/env python3
"""
Coder Agent Example - Ultimate Agentic StarterKit

This example demonstrates how to use the Coder Agent to generate code using
Claude API with tool calling capabilities for file operations.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add StarterKit to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'StarterKit'))

from StarterKit.agents.coder_agent import CoderAgent
from StarterKit.core.models import ProjectTask, AgentType, create_project_task
from StarterKit.core.logger import get_logger

# Setup logging
logger = get_logger("coder_example")


def display_results(result):
    """Display coder agent results in a formatted way."""
    print("=" * 60)
    print("CODER AGENT EXECUTION RESULTS")
    print("=" * 60)
    
    if result.success:
        print(f"✓ Coder execution successful!")
        print(f"  Agent ID: {result.agent_id}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        print(f"  Timestamp: {result.timestamp}")
        
        # Display output details
        output = result.output
        if output and isinstance(output, dict):
            files = output.get('files', [])
            print(f"  Files created: {len(files)}")
            print(f"  Tool calls: {output.get('tool_calls', 0)}")
            
            # Display file details
            if files:
                print("\n--- GENERATED FILES ---")
                for i, file_info in enumerate(files, 1):
                    print(f"{i}. {file_info.get('path', 'Unknown path')}")
                    print(f"   Language: {file_info.get('language', 'Unknown')}")
                    print(f"   Size: {file_info.get('size', 0)} characters")
                    print(f"   Lines: {file_info.get('lines', 0)}")
                    print(f"   Operation: {file_info.get('operation', 'Unknown')}")
                    print()
            
            # Display validation results
            validation = output.get('validation', {})
            if validation:
                print("--- VALIDATION RESULTS ---")
                print(f"  Total files: {validation.get('total_files', 0)}")
                print(f"  Total lines: {validation.get('total_lines', 0)}")
                print(f"  Languages: {', '.join(validation.get('languages', []))}")
                print(f"  Quality score: {validation.get('quality_score', 0.0):.2f}")
                
                issues = validation.get('issues', [])
                if issues:
                    print(f"  Issues found: {len(issues)}")
                    for issue in issues:
                        print(f"    - {issue}")
        else:
            print("  No output data available")
    else:
        print(f"✗ Coder execution failed: {result.error}")
        print(f"  Agent ID: {result.agent_id}")
        print(f"  Execution time: {result.execution_time:.2f}s")


async def example_1_fastapi_auth():
    """Example 1: Create FastAPI application with authentication."""
    print("\n" + "="*60)
    print("EXAMPLE 1: FastAPI Authentication System")
    print("="*60)
    
    # Create coder agent
    coder = CoderAgent()
    
    # Create coding task
    task = create_project_task(
        title="Create FastAPI Authentication",
        description="""
        Create a FastAPI application with user authentication:
        1. User registration endpoint
        2. Login endpoint with JWT tokens
        3. Protected route that requires authentication
        4. User model with password hashing
        5. Proper error handling and validation
        
        Requirements:
        - Use FastAPI framework
        - Use Pydantic for data validation
        - Use JWT for authentication
        - Use bcrypt for password hashing
        - Include proper error handling
        - Add comprehensive docstrings
        - Use async/await patterns
        """,
        task_type="CREATE",
        agent_type=AgentType.CODER
    )
    
    # Execute coder
    print("Executing coder agent...")
    result = await coder.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def example_2_rest_api():
    """Example 2: Create REST API for todo management."""
    print("\n" + "="*60)
    print("EXAMPLE 2: REST API for Todo Management")
    print("="*60)
    
    # Create coder agent
    coder = CoderAgent()
    
    # Create coding task
    task = create_project_task(
        title="Create Todo REST API",
        description="""
        Create a complete REST API for todo management:
        1. Todo model with Pydantic
        2. CRUD endpoints (Create, Read, Update, Delete)
        3. In-memory storage for simplicity
        4. Input validation
        5. Error handling
        6. API documentation
        
        Endpoints needed:
        - POST /todos - Create new todo
        - GET /todos - Get all todos
        - GET /todos/{id} - Get specific todo
        - PUT /todos/{id} - Update todo
        - DELETE /todos/{id} - Delete todo
        
        Requirements:
        - Use FastAPI
        - Use Pydantic models
        - Add proper HTTP status codes
        - Include comprehensive documentation
        - Add input validation
        """,
        task_type="CREATE",
        agent_type=AgentType.CODER
    )
    
    # Execute coder
    print("Executing coder agent...")
    result = await coder.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def example_3_data_processing():
    """Example 3: Create data processing utility."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Processing Utility")
    print("="*60)
    
    # Create coder agent
    coder = CoderAgent()
    
    # Create coding task
    task = create_project_task(
        title="Create Data Processing Utility",
        description="""
        Create a Python utility for data processing:
        1. CSV file reader and writer
        2. Data validation and cleaning
        3. Statistical analysis functions
        4. Data transformation utilities
        5. Export to different formats
        
        Features needed:
        - Read CSV files with error handling
        - Clean and validate data
        - Calculate basic statistics (mean, median, mode)
        - Filter and transform data
        - Export to JSON and Excel formats
        - Command-line interface
        
        Requirements:
        - Use pandas for data manipulation
        - Use click for CLI
        - Add comprehensive error handling
        - Include type hints
        - Add logging
        - Create modular design
        """,
        task_type="CREATE",
        agent_type=AgentType.CODER
    )
    
    # Execute coder
    print("Executing coder agent...")
    result = await coder.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def example_4_async_web_scraper():
    """Example 4: Create async web scraper."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Async Web Scraper")
    print("="*60)
    
    # Create coder agent
    coder = CoderAgent()
    
    # Create coding task
    task = create_project_task(
        title="Create Async Web Scraper",
        description="""
        Create an asynchronous web scraper:
        1. Async HTTP requests using aiohttp
        2. HTML parsing with BeautifulSoup
        3. Rate limiting and respectful scraping
        4. Data extraction and storage
        5. Error handling and retries
        
        Features:
        - Scrape multiple URLs concurrently
        - Extract structured data from HTML
        - Handle different content types
        - Rate limiting to respect servers
        - Retry mechanism for failed requests
        - Save results to JSON/CSV
        
        Requirements:
        - Use aiohttp for async HTTP requests
        - Use BeautifulSoup for HTML parsing
        - Add proper error handling
        - Include rate limiting
        - Add logging and monitoring
        - Create configurable scraper
        """,
        task_type="CREATE",
        agent_type=AgentType.CODER
    )
    
    # Execute coder
    print("Executing coder agent...")
    result = await coder.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def example_5_file_modification():
    """Example 5: Modify existing code files."""
    print("\n" + "="*60)
    print("EXAMPLE 5: File Modification")
    print("="*60)
    
    # First, create a simple file to modify
    sample_code = """
def hello_world():
    print("Hello, World!")

def add_numbers(a, b):
    return a + b

if __name__ == "__main__":
    hello_world()
    result = add_numbers(5, 3)
    print(f"Result: {result}")
"""
    
    # Write sample file
    sample_file = Path("sample_code.py")
    sample_file.write_text(sample_code)
    
    # Create coder agent
    coder = CoderAgent()
    
    # Create modification task
    task = create_project_task(
        title="Modify Sample Code",
        description=f"""
        Modify the existing file 'sample_code.py' to:
        1. Add type hints to all functions
        2. Add docstrings to all functions
        3. Add error handling to add_numbers function
        4. Add a new function for multiplication
        5. Add logging capabilities
        6. Improve the main block
        
        Current file content:
        {sample_code}
        
        Requirements:
        - Preserve existing functionality
        - Add proper type hints
        - Add comprehensive docstrings
        - Add error handling
        - Add logging
        - Follow PEP 8 style guide
        """,
        task_type="MODIFY",
        agent_type=AgentType.CODER
    )
    
    # Execute coder
    print("Executing coder agent...")
    result = await coder.execute(task)
    
    # Display results
    display_results(result)
    
    # Clean up
    if sample_file.exists():
        sample_file.unlink()
    
    return result


async def example_6_testing_code():
    """Example 6: Create unit tests for code."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Unit Test Creation")
    print("="*60)
    
    # Create coder agent
    coder = CoderAgent()
    
    # Create testing task
    task = create_project_task(
        title="Create Unit Tests",
        description="""
        Create comprehensive unit tests for a calculator module:
        1. Test basic arithmetic operations
        2. Test edge cases and error conditions
        3. Test input validation
        4. Use pytest framework
        5. Add test fixtures and parameterized tests
        
        Calculator functions to test:
        - add(a, b) - addition
        - subtract(a, b) - subtraction
        - multiply(a, b) - multiplication
        - divide(a, b) - division with zero handling
        - power(a, b) - exponentiation
        - sqrt(a) - square root
        
        Requirements:
        - Use pytest framework
        - Create test fixtures
        - Add parameterized tests
        - Test error conditions
        - Add test coverage
        - Include docstrings
        - Test both positive and negative cases
        """,
        task_type="CREATE",
        agent_type=AgentType.CODER
    )
    
    # Execute coder
    print("Executing coder agent...")
    result = await coder.execute(task)
    
    # Display results
    display_results(result)
    
    return result


async def example_7_error_handling():
    """Example 7: Error handling and edge cases."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Error Handling")
    print("="*60)
    
    # Test with invalid task type
    print("Testing with invalid agent type...")
    coder = CoderAgent()
    
    invalid_task = create_project_task(
        title="Invalid Task",
        description="This should fail",
        task_type="CREATE",
        agent_type=AgentType.PARSER  # Wrong agent type
    )
    
    result = await coder.execute(invalid_task)
    display_results(result)
    
    # Test with empty description
    print("\nTesting with empty description...")
    empty_task = create_project_task(
        title="Empty Description Task",
        description="",
        task_type="CREATE",
        agent_type=AgentType.CODER
    )
    
    result = await coder.execute(empty_task)
    display_results(result)
    
    return result


async def main():
    """Main function to run all examples."""
    print("Ultimate Agentic StarterKit - Coder Agent Examples")
    print("=" * 60)
    
    try:
        # Run examples
        await example_1_fastapi_auth()
        await example_2_rest_api()
        await example_3_data_processing()
        await example_4_async_web_scraper()
        await example_5_file_modification()
        await example_6_testing_code()
        await example_7_error_handling()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        print("Note: Some examples may fail if API keys are not configured")
        print("Check the configuration files for proper API key setup")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)