#!/usr/bin/env python3
"""
Integration initialization script.

This script initializes and tests the external integrations system.
Run this script to set up Claude Code, Git Manager, and Ollama Client integrations.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.setup import IntegrationSetup


async def main():
    """Main initialization function."""
    print("🚀 Initializing External Integrations System")
    print("=" * 50)
    
    # Initialize setup manager
    workspace_root = Path(__file__).parent.parent.parent
    setup_manager = IntegrationSetup(str(workspace_root))
    
    try:
        # Display setup information
        setup_info = setup_manager.get_setup_info()
        print(f"📁 Workspace Root: {setup_info['workspace_root']}")
        print(f"📁 Claude Directory: {setup_info['claude_dir']}")
        print(f"📁 Commands Directory: {setup_info['commands_dir']}")
        print()
        
        # Setup all integrations
        print("🔧 Setting up integrations...")
        setup_result = await setup_manager.setup_all()
        
        if setup_result["success"]:
            print("✅ Integration setup completed successfully!")
            
            # Display setup results
            print(f"📂 Directories created: {len(setup_result['directories_created'])}")
            for directory in setup_result['directories_created']:
                print(f"   • {directory}")
            
            print(f"🔗 Integrations setup: {len(setup_result['integrations_setup'])}")
            for integration in setup_result['integrations_setup']:
                print(f"   • {integration}")
            
            if setup_result['errors']:
                print(f"⚠️  Warnings: {len(setup_result['errors'])}")
                for error in setup_result['errors']:
                    print(f"   • {error}")
        else:
            print("❌ Integration setup failed!")
            for error in setup_result['errors']:
                print(f"   • {error}")
            return 1
        
        print()
        
        # Validate setup
        print("🔍 Validating integration setup...")
        validation_result = await setup_manager.validate_setup()
        
        if validation_result["success"]:
            print("✅ All validation checks passed!")
            
            print(f"✅ Checks passed: {len(validation_result['checks_passed'])}")
            for check in validation_result['checks_passed']:
                print(f"   • {check}")
        else:
            print("❌ Validation failed!")
            
            if validation_result['checks_failed']:
                print(f"❌ Checks failed: {len(validation_result['checks_failed'])}")
                for check in validation_result['checks_failed']:
                    print(f"   • {check}")
            
            return 1
        
        print()
        
        # Display integration status
        print("📊 Integration Status:")
        factory_status = setup_manager.factory.get_status()
        
        print(f"   Total instances: {factory_status['total_instances']}")
        print(f"   Monitoring enabled: {factory_status['monitoring_enabled']}")
        
        for integration_type, status in factory_status['status_by_type'].items():
            print(f"   {integration_type}:")
            print(f"     • Total: {status['total_instances']}")
            print(f"     • Healthy: {status['healthy_instances']}")
            print(f"     • Degraded: {status['degraded_instances']}")
            print(f"     • Failed: {status['failed_instances']}")
        
        print()
        print("🎉 Integration system is ready!")
        print()
        print("Next steps:")
        print("1. Launch Claude Code in VS Code (Ctrl+ESC or Cmd+ESC)")
        print("2. Use commands like /generate-prp, /execute-agent-flow")
        print("3. Start Ollama service for local model support:")
        print("   • Install Ollama: https://ollama.ai/")
        print("   • Run: ollama serve")
        print("   • Pull models: ollama pull llama3.2")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1
    
    finally:
        await setup_manager.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)