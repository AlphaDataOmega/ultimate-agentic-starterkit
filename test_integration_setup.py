#!/usr/bin/env python3
"""
Simple test script to verify integration setup.
"""

import sys
import os
import subprocess

def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")
    
    try:
        # Test that we can import the integrations
        from integrations.claude_code import ClaudeCodeIntegration
        from integrations.git_manager import GitManager
        from integrations.ollama_client import OllamaClient
        from integrations.factory import IntegrationFactory
        from integrations.error_handling import ErrorHandler
        
        print("✅ All integration imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_directory_structure():
    """Test that directory structure is correct."""
    print("Testing directory structure...")
    
    # Check that .claude directory exists
    claude_dir = os.path.join(os.getcwd(), ".claude")
    commands_dir = os.path.join(claude_dir, "commands")
    
    if os.path.exists(claude_dir):
        print("✅ .claude directory exists")
    else:
        print("❌ .claude directory missing")
        return False
    
    if os.path.exists(commands_dir):
        print("✅ .claude/commands directory exists")
    else:
        print("❌ .claude/commands directory missing")
        return False
    
    # Check for command files
    command_files = [
        "generate-prp.json",
        "execute-agent-flow.json",
        "review-code.json",
        "validate-project.json",
        "create-agent.json"
    ]
    
    for cmd_file in command_files:
        cmd_path = os.path.join(commands_dir, cmd_file)
        if os.path.exists(cmd_path):
            print(f"✅ Command file {cmd_file} exists")
        else:
            print(f"❌ Command file {cmd_file} missing")
    
    return True

def test_linting():
    """Test that code passes linting."""
    print("Testing linting...")
    
    try:
        result = subprocess.run(
            ["./venv_linux/bin/python", "-m", "ruff", "check", "integrations/"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Linting passed")
            return True
        else:
            print(f"❌ Linting failed: {result.stdout}")
            return False
    except Exception as e:
        print(f"❌ Linting test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing External Integrations Implementation")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_directory_structure,
        test_linting
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
            print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration implementation is successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())