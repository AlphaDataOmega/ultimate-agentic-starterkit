#!/usr/bin/env python3
"""
Agent Framework Validation Script

This script validates that the agent framework is working correctly.
"""

import sys
import os

# Add the StarterKit directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all agent modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.models import AgentResult, ProjectTask, AgentType, create_project_task
        print("✓ Core models imported successfully")
        
        from agents.base_agent import BaseAgent
        print("✓ BaseAgent imported successfully")
        
        from agents.parser_agent import ParserAgent
        print("✓ ParserAgent imported successfully")
        
        from agents.coder_agent import CoderAgent
        print("✓ CoderAgent imported successfully")
        
        from agents.tester_agent import TesterAgent
        print("✓ TesterAgent imported successfully")
        
        from agents.advisor_agent import AdvisorAgent
        print("✓ AdvisorAgent imported successfully")
        
        from agents.factory import AgentFactory, create_agent
        print("✓ AgentFactory imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        return False

def test_agent_creation():
    """Test that agents can be created."""
    print("\nTesting agent creation...")
    
    try:
        from agents.factory import create_agent
        from core.models import AgentType
        
        # Test creating each agent type
        for agent_type in AgentType:
            try:
                agent_instance = create_agent(agent_type, register=False)
                print(f"✓ Created {agent_type.value} agent: {agent_instance.agent.agent_id}")
            except Exception as e:
                print(f"✗ Failed to create {agent_type.value} agent: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Agent creation failed: {str(e)}")
        return False

def test_task_creation():
    """Test that tasks can be created."""
    print("\nTesting task creation...")
    
    try:
        from core.models import create_project_task, AgentType
        
        task = create_project_task(
            title="Test Task",
            description="This is a test task",
            task_type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        print(f"✓ Created task: {task.id}")
        print(f"  Title: {task.title}")
        print(f"  Type: {task.type}")
        print(f"  Agent Type: {task.agent_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Task creation failed: {str(e)}")
        return False

def test_agent_stats():
    """Test that agent statistics work."""
    print("\nTesting agent statistics...")
    
    try:
        from agents.factory import get_agent_factory
        
        factory = get_agent_factory()
        registry = factory.get_registry()
        
        stats = registry.get_registry_stats()
        print(f"✓ Registry stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent stats failed: {str(e)}")
        return False

def test_confidence_calculation():
    """Test confidence calculation."""
    print("\nTesting confidence calculation...")
    
    try:
        from agents.parser_agent import ParserAgent
        
        agent = ParserAgent()
        
        # Test different confidence scenarios
        tests = [
            ({}, "empty indicators"),
            ({'error_count': 1}, "with errors"),
            ({'completion_status': 'complete'}, "completed"),
            ({'validation_passed': True}, "validation passed"),
            ({'error_count': 0, 'completion_status': 'complete', 'validation_passed': True}, "all good")
        ]
        
        for indicators, description in tests:
            confidence = agent._calculate_confidence(indicators)
            print(f"✓ Confidence for {description}: {confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Confidence calculation failed: {str(e)}")
        return False

def main():
    """Run all validation tests."""
    print("AI Agent Framework Validation")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_agent_creation,
        test_task_creation,
        test_agent_stats,
        test_confidence_calculation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {str(e)}")
    
    print(f"\n" + "=" * 40)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All validation tests passed!")
        return 0
    else:
        print("✗ Some validation tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())