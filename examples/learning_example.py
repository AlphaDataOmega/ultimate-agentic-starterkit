#!/usr/bin/env python3
"""
Learning Phase Example - Ultimate Agentic StarterKit

This example demonstrates how to use the Interactive Learning Phase
for project knowledge transfer and requirement refinement.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.project_manager_agent import ProjectManagerAgent
from agents.research_agent import ResearchAgent
from core.models import ProjectTask, AgentType
from core.logger import get_logger

# Setup logging
logger = get_logger("learning_example")


def create_sample_overview():
    """Create a sample OVERVIEW.md for testing."""
    overview_content = """# E-Commerce Platform

## Project Description
Build a modern e-commerce platform for selling digital products with user accounts and payment processing.

## Project Type
web-app

## Core Features & Requirements

### Feature 1: User Management
**Description**: User registration and authentication system
**Tasks**:
- [ ] User registration
- [ ] Login/logout functionality  
- [ ] Profile management

### Feature 2: Product Catalog
**Description**: Display and manage digital products
**Tasks**:
- [ ] Product listing
- [ ] Product details
- [ ] Search functionality

### Feature 3: Shopping Cart
**Description**: Cart and checkout process
**Tasks**:
- [ ] Add/remove items
- [ ] Checkout process
- [ ] Payment integration

## Success Criteria
- [ ] All core features implemented and tested
- [ ] Application runs without errors
- [ ] Payment processing works correctly

## Notes
- Focus on digital products only
- Mobile-responsive design required
"""
    
    overview_path = Path("SAMPLE_OVERVIEW.md")
    overview_path.write_text(overview_content)
    print(f"✓ Created sample overview: {overview_path}")
    return overview_path


async def demonstrate_project_manager():
    """Demonstrate Project Manager Agent functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Project Manager Agent")
    print("="*60)
    
    # Create sample overview
    overview_path = create_sample_overview()
    overview_content = overview_path.read_text()
    
    # Create Project Manager Agent
    pm_agent = ProjectManagerAgent()
    
    # Create analysis task
    task = ProjectTask(
        id="demo-pm-analysis",
        title="Analyze E-Commerce Project",
        description=overview_content,
        type="CREATE",
        agent_type=AgentType.ADVISOR
    )
    
    print(f"📊 Analyzing project overview...")
    result = await pm_agent.execute(task)
    
    if result.success:
        print(f"✓ Analysis completed successfully!")
        print(f"  Generated: {result.output['total_questions']} total questions")
        print(f"  Critical: {result.output['critical_questions']} critical questions")
        print(f"  Questions file: {result.output['questions_file']}")
        print(f"  Confidence: {result.confidence:.1%}")
        
        # Display some sample questions
        questions = result.output['questions'][:3]  # First 3 questions
        print(f"\n📝 Sample Generated Questions:")
        for i, question in enumerate(questions, 1):
            print(f"  {i}. [{question['priority']}] {question['question']}")
            print(f"     Why: {question['why']}")
        
        return result.output['questions_file']
        
    else:
        print(f"✗ Analysis failed: {result.error}")
        return None


async def demonstrate_research_agent(questions_file):
    """Demonstrate Research Agent functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Research Agent")
    print("="*60)
    
    if not questions_file or not Path(questions_file).exists():
        print("⚠️ No questions file available for research demonstration")
        return
    
    # Create Research Agent
    research_agent = ResearchAgent()
    
    # Create research task
    task = ProjectTask(
        id="demo-research",
        title="Research E-Commerce Questions",
        description=questions_file,
        type="MODIFY",
        agent_type=AgentType.ADVISOR
    )
    
    print(f"🔍 Researching answers for questions in {questions_file}...")
    result = await research_agent.execute(task)
    
    if result.success:
        print(f"✓ Research completed successfully!")
        print(f"  Researched: {result.output['questions_researched']} questions")
        print(f"  Updated file: {result.output['questions_file']}")
        print(f"  Confidence: {result.confidence:.1%}")
        
        # Show research results
        if result.output['research_results']:
            print(f"\n🔬 Research Results:")
            for i, research in enumerate(result.output['research_results'][:2], 1):
                print(f"  {i}. Method: {research['method']}")
                print(f"     Question: {research['question']}")
                print(f"     Answer: {research['answer'][:100]}...")
                print(f"     Confidence: {research['confidence']:.1%}")
        
    else:
        print(f"✗ Research failed: {result.error}")


async def demonstrate_quick_research():
    """Demonstrate quick research functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Quick Research")
    print("="*60)
    
    research_agent = ResearchAgent()
    
    test_queries = [
        "What authentication method for web apps?",
        "Which database for e-commerce?",
        "What payment processing options?"
    ]
    
    for query in test_queries:
        print(f"\n🤔 Quick research: {query}")
        answer = await research_agent.quick_research(query)
        print(f"💡 Answer: {answer}")


async def demonstrate_learning_completion():
    """Demonstrate learning completion check."""
    print("\n" + "="*60)
    print("DEMONSTRATION 4: Learning Completion Check")
    print("="*60)
    
    pm_agent = ProjectManagerAgent()
    
    # Test with incomplete overview
    incomplete_overview = "# Basic Project\n\nJust a simple description."
    is_complete = await pm_agent.check_learning_completion(incomplete_overview)
    print(f"📋 Incomplete overview learning status: {'Complete' if is_complete else 'Needs more details'}")
    
    # Test with complete overview
    complete_overview = """# Complete E-Commerce Platform

## Technology Stack
- Frontend: React with TypeScript
- Backend: Node.js with Express
- Database: PostgreSQL with Redis
- Payment: Stripe integration

## Features
### User Management
Complete authentication system with JWT tokens

### Product Catalog  
Full product management with search and filtering

## Success Criteria
- 95% test coverage
- Sub-200ms response times
- PCI compliance for payments

## File Structure
Complete project structure defined
"""
    
    is_complete = await pm_agent.check_learning_completion(complete_overview)
    print(f"📋 Complete overview learning status: {'Complete' if is_complete else 'Needs more details'}")


def cleanup_demo_files():
    """Clean up demonstration files."""
    demo_files = [
        "SAMPLE_OVERVIEW.md",
        "questions.md"
    ]
    
    for file_path in demo_files:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            print(f"🗑️ Cleaned up: {file_path}")


async def main():
    """Run all learning phase demonstrations."""
    print("Ultimate Agentic StarterKit - Learning Phase Examples")
    print("=" * 60)
    
    try:
        # Demonstration 1: Project Manager Agent
        questions_file = await demonstrate_project_manager()
        
        # Demonstration 2: Research Agent
        await demonstrate_research_agent(questions_file)
        
        # Demonstration 3: Quick Research
        await demonstrate_quick_research()
        
        # Demonstration 4: Learning Completion
        await demonstrate_learning_completion()
        
        print("\n" + "="*60)
        print("🎉 All learning phase demonstrations completed!")
        print("="*60)
        
        print("\nTo use the learning phase in your projects:")
        print("1. Create your OVERVIEW.md file")
        print("2. Run: python kit.py --learn --overview OVERVIEW.md")
        print("3. Edit the generated questions.md file")
        print("4. Run: python kit.py --accept-learn")
        print("5. Repeat until learning is complete")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        print(f"❌ Example failed: {e}")
        
    finally:
        cleanup_demo_files()


if __name__ == "__main__":
    asyncio.run(main())