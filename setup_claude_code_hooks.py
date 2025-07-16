#!/usr/bin/env python3
"""
Setup script for Claude Code hooks integration.

This script helps configure Claude Code to work with StarterKit workflow hooks.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any


def get_claude_code_config_path() -> Path:
    """Get the Claude Code configuration directory."""
    # Try common locations for Claude Code configuration
    possible_paths = [
        Path.home() / ".config" / "claude-code",
        Path.home() / ".claude-code",
        Path(os.environ.get("CLAUDE_CODE_CONFIG_DIR", "")).expanduser(),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Default to ~/.config/claude-code
    default_path = Path.home() / ".config" / "claude-code"
    default_path.mkdir(parents=True, exist_ok=True)
    return default_path


def create_hook_settings() -> Dict[str, Any]:
    """Create hook settings configuration."""
    current_dir = Path(__file__).parent.absolute()
    
    return {
        "hooks": {
            "workflow-start": {
                "enabled": True,
                "command": f"cd {current_dir} && python -m StarterKit.workflows workflow_start --workflow-id {{workflow_id}} --project-id {{project_id}} --total-tasks {{total_tasks}}",
                "timeout": 30000,
                "shell": True,
                "description": "Triggered when a StarterKit workflow begins execution"
            },
            "workflow-complete": {
                "enabled": True,
                "command": f"cd {current_dir} && python -m StarterKit.workflows workflow_complete --workflow-id {{workflow_id}} --project-id {{project_id}} --success {{success}} --confidence {{confidence}}",
                "timeout": 60000,
                "shell": True,
                "description": "Triggered when a StarterKit workflow completes successfully"
            },
            "workflow-failure": {
                "enabled": True,
                "command": f"cd {current_dir} && python -m StarterKit.workflows workflow_failure --workflow-id {{workflow_id}} --project-id {{project_id}} --error-message {{error_message}}",
                "timeout": 45000,
                "shell": True,
                "description": "Triggered when a StarterKit workflow fails"
            },
            "task-start": {
                "enabled": True,
                "command": f"cd {current_dir} && python -m StarterKit.workflows task_start --workflow-id {{workflow_id}} --project-id {{project_id}} --task-id {{task_id}} --agent-type {{agent_type}}",
                "timeout": 15000,
                "shell": True,
                "description": "Triggered when a StarterKit task begins execution"
            },
            "task-complete": {
                "enabled": True,
                "command": f"cd {current_dir} && python -m StarterKit.workflows task_complete --workflow-id {{workflow_id}} --project-id {{project_id}} --task-id {{task_id}} --success {{success}} --confidence {{confidence}}",
                "timeout": 30000,
                "shell": True,
                "description": "Triggered when a StarterKit task completes"
            },
            "milestone-reached": {
                "enabled": True,
                "command": f"cd {current_dir} && python -m StarterKit.workflows milestone_reached --workflow-id {{workflow_id}} --project-id {{project_id}} --milestone-name {{milestone_name}} --milestone-value {{milestone_value}}",
                "timeout": 20000,
                "shell": True,
                "description": "Triggered when a StarterKit milestone is reached"
            },
            "progress-update": {
                "enabled": False,
                "command": f"cd {current_dir} && python -m StarterKit.workflows progress_update --workflow-id {{workflow_id}} --project-id {{project_id}} --progress {{progress}} --completed-tasks {{completed_tasks}} --total-tasks {{total_tasks}}",
                "timeout": 10000,
                "shell": True,
                "description": "Triggered on StarterKit progress updates (disabled by default to reduce noise)"
            },
            "error-occurred": {
                "enabled": True,
                "command": f"cd {current_dir} && python -m StarterKit.workflows error_occurred --workflow-id {{workflow_id}} --project-id {{project_id}} --error-type {{error_type}} --error-message {{error_message}} --task-id {{task_id}}",
                "timeout": 30000,
                "shell": True,
                "description": "Triggered when errors occur in StarterKit workflows"
            },
            "advisor-review": {
                "enabled": True,
                "command": f"cd {current_dir} && python -m StarterKit.workflows advisor_review --workflow-id {{workflow_id}} --project-id {{project_id}} --review-type {{review_type}}",
                "timeout": 120000,
                "shell": True,
                "description": "Triggered for advisor review after workflow completion"
            }
        },
        "settings": {
            "starterkit_integration": {
                "enabled": True,
                "base_path": str(current_dir / "StarterKit"),
                "python_path": "python",
                "log_level": "INFO",
                "voice_alerts": True,
                "progress_tracking": True,
                "advisor_integration": True
            },
            "workflow_defaults": {
                "confidence_threshold": 0.8,
                "high_confidence_threshold": 0.95,
                "max_retries": 3,
                "timeout": 600,
                "enable_parallel_execution": True,
                "auto_milestone_creation": True
            },
            "hook_configuration": {
                "max_concurrent_hooks": 5,
                "hook_retry_count": 3,
                "hook_retry_delay": 2000,
                "enable_hook_logging": True,
                "hook_log_file": "logs/claude_code_hooks.log"
            }
        }
    }


def setup_claude_code_hooks():
    """Set up Claude Code hooks for StarterKit integration."""
    print("🔧 Setting up Claude Code hooks for StarterKit integration...")
    
    # Get Claude Code config directory
    config_dir = get_claude_code_config_path()
    print(f"📁 Using Claude Code config directory: {config_dir}")
    
    # Create settings file
    settings_file = config_dir / "settings.json"
    hook_settings = create_hook_settings()
    
    # Backup existing settings if they exist
    if settings_file.exists():
        backup_file = config_dir / "settings.json.backup"
        shutil.copy2(settings_file, backup_file)
        print(f"💾 Backed up existing settings to: {backup_file}")
        
        # Try to merge with existing settings
        try:
            with open(settings_file, 'r') as f:
                existing_settings = json.load(f)
            
            # Merge hooks section
            if "hooks" not in existing_settings:
                existing_settings["hooks"] = {}
            
            existing_settings["hooks"].update(hook_settings["hooks"])
            
            # Merge settings section
            if "settings" not in existing_settings:
                existing_settings["settings"] = {}
            
            existing_settings["settings"].update(hook_settings["settings"])
            
            hook_settings = existing_settings
            print("🔄 Merged with existing settings")
            
        except Exception as e:
            print(f"⚠️  Could not merge with existing settings: {e}")
            print("🔄 Using new settings (backup saved)")
    
    # Write settings file
    with open(settings_file, 'w') as f:
        json.dump(hook_settings, f, indent=2)
    
    print(f"✅ Created Claude Code settings: {settings_file}")
    
    # Create logs directory
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    print(f"📁 Created logs directory: {logs_dir}")
    
    # Create a test script
    test_script = Path(__file__).parent / "test_claude_code_hooks.py"
    test_script_content = '''#!/usr/bin/env python3
"""
Test script for Claude Code hooks integration.
"""

import asyncio
import sys
from pathlib import Path

# Add StarterKit to path
sys.path.insert(0, str(Path(__file__).parent / "StarterKit"))

from workflows.claude_code_hooks import (
    get_hook_integration,
    emit_workflow_start,
    emit_workflow_complete,
    emit_milestone_reached
)

async def test_hooks():
    """Test the hooks integration."""
    print("🧪 Testing Claude Code hooks integration...")
    
    workflow_id = "test-workflow-123"
    project_id = "test-project-456"
    
    try:
        # Test workflow start
        print("📊 Testing workflow start hook...")
        await emit_workflow_start(workflow_id, project_id, 5)
        print("✅ Workflow start hook test passed")
        
        # Test milestone reached
        print("🎯 Testing milestone reached hook...")
        await emit_milestone_reached(workflow_id, project_id, "50% Complete", 50)
        print("✅ Milestone reached hook test passed")
        
        # Test workflow complete
        print("🏁 Testing workflow complete hook...")
        await emit_workflow_complete(workflow_id, project_id, True, 0.95)
        print("✅ Workflow complete hook test passed")
        
        print("🎉 All hook tests passed!")
        
    except Exception as e:
        print(f"❌ Hook test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_hooks())
    sys.exit(0 if success else 1)
'''
    
    with open(test_script, 'w') as f:
        f.write(test_script_content)
    
    test_script.chmod(0o755)
    print(f"🧪 Created test script: {test_script}")
    
    # Print setup completion message
    print("\n🎉 Claude Code hooks setup complete!")
    print("\n📋 Next steps:")
    print("1. Restart Claude Code to load the new hook configuration")
    print("2. Run the test script to verify hooks are working:")
    print(f"   python {test_script}")
    print("3. Use StarterKit workflows - hooks will be triggered automatically")
    print("\n⚙️  Configuration details:")
    print(f"   - Settings file: {settings_file}")
    print(f"   - Logs directory: {logs_dir}")
    print(f"   - Hook commands use: {Path(__file__).parent}")
    print("\n🔧 To customize hooks, edit the settings.json file")
    print("📚 For more information, see the StarterKit documentation")


if __name__ == "__main__":
    setup_claude_code_hooks()