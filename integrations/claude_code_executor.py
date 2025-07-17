"""
Claude Code Integration for Work Order Execution.

This module handles the integration with Claude Code/SDK for actual code generation,
leveraging Claude's massive context windows and visual capabilities.
"""

import json
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from core.logger import get_logger
from core.config import get_config


class ClaudeCodeExecutor:
    """
    Manages code generation through Claude Code/SDK with comprehensive context.
    
    Uses Claude's massive context windows to provide full project context
    for intelligent code generation.
    """
    
    def __init__(self):
        """Initialize Claude Code executor."""
        self.logger = get_logger("claude_code_executor")
        self.config = get_config()
        
        # Claude Code integration settings
        self.use_claude_code = True  # Use Claude Code extension if available
        self.use_claude_sdk = True   # Fallback to SDK if extension not available
        self.max_context_size = 200000  # Claude's large context window
        
    async def execute_coding_work_order(self, work_order: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a coding work order using Claude Code/SDK.
        
        Args:
            work_order: Work order specification
            context: Full project context including all documents and history
            
        Returns:
            Execution result with generated files and artifacts
        """
        self.logger.info(f"Executing coding work order via Claude: {work_order['id']}")
        
        # Prepare comprehensive context for Claude
        claude_context = await self._prepare_claude_context(work_order, context)
        
        # Create coding prompt with full context
        coding_prompt = self._create_coding_prompt(work_order, claude_context)
        
        # Execute via Claude Code or SDK
        if self.use_claude_code and await self._is_claude_code_available():
            result = await self._execute_via_claude_code(coding_prompt, work_order)
        elif self.use_claude_sdk:
            result = await self._execute_via_claude_sdk(coding_prompt, work_order)
        else:
            raise RuntimeError("Neither Claude Code nor Claude SDK available")
        
        return result
    
    async def _prepare_claude_context(self, work_order: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for Claude with full project knowledge."""
        
        claude_context = {
            "work_order": work_order,
            "project_documents": {},
            "working_assumptions": [],
            "completed_work_orders": [],
            "current_codebase": {},
            "file_structure": {},
            "technical_decisions": []
        }
        
        # Add all project documents (Claude can handle large context)
        for doc_name, doc_content in context.get("documents", {}).items():
            claude_context["project_documents"][doc_name] = doc_content
        
        # Extract working assumptions from CONTEXT.md
        if "CONTEXT.md" in context.get("documents", {}):
            assumptions = self._extract_checked_assumptions(context["documents"]["CONTEXT.md"])
            claude_context["working_assumptions"] = assumptions
        
        # Add completion history with full details
        claude_context["completed_work_orders"] = context.get("completion_history", [])
        
        # Scan current codebase if it exists
        current_code = await self._scan_current_codebase()
        claude_context["current_codebase"] = current_code
        
        # Get current file structure
        file_structure = await self._get_file_structure()
        claude_context["file_structure"] = file_structure
        
        return claude_context
    
    def _extract_checked_assumptions(self, context_content: str) -> List[str]:
        """Extract validated assumptions from CONTEXT.md."""
        assumptions = []
        lines = context_content.split('\n')
        
        for line in lines:
            if '- [x]' in line:
                assumption = line.replace('- [x]', '').strip()
                if assumption:
                    assumptions.append(assumption)
        
        return assumptions
    
    async def _scan_current_codebase(self) -> Dict[str, str]:
        """Scan existing codebase for context."""
        codebase = {}
        
        # Common code directories to scan
        code_dirs = ['src', 'lib', 'app', 'components', 'services', 'models', 'api']
        
        for dir_name in code_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs']:
                        try:
                            codebase[str(file_path)] = file_path.read_text()
                        except Exception as e:
                            self.logger.warning(f"Could not read {file_path}: {e}")
        
        return codebase
    
    async def _get_file_structure(self) -> Dict[str, Any]:
        """Get current project file structure."""
        try:
            result = subprocess.run(['find', '.', '-type', 'f', '-not', '-path', './.git/*'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return {"files": files, "count": len(files)}
        except Exception as e:
            self.logger.warning(f"Could not get file structure: {e}")
        
        return {"files": [], "count": 0}
    
    def _create_coding_prompt(self, work_order: Dict[str, Any], 
                            claude_context: Dict[str, Any]) -> str:
        """Create comprehensive coding prompt for Claude."""
        
        prompt = f"""# Coding Work Order: {work_order['title']}

## Work Order Description
{work_order['description']}

## Project Context

### Working Assumptions (Validated)
{chr(10).join(f"✓ {assumption}" for assumption in claude_context['working_assumptions'])}

### Project Documents
"""
        
        # Add key project documents
        for doc_name, doc_content in claude_context['project_documents'].items():
            if doc_content:
                prompt += f"\n#### {doc_name}\n```\n{doc_content[:2000]}{'...' if len(doc_content) > 2000 else ''}\n```\n"
        
        prompt += f"""
### Completed Work Orders
{chr(10).join(f"- {wo.get('title', 'Unknown')}: {wo.get('completed_at', 'Unknown')}" for wo in claude_context['completed_work_orders'][-10:])}

### Current Codebase
"""
        
        # Add existing code for context
        for file_path, file_content in claude_context['current_codebase'].items():
            if file_content:
                prompt += f"\n#### {file_path}\n```\n{file_content[:1000]}{'...' if len(file_content) > 1000 else ''}\n```\n"
        
        prompt += f"""
### Current File Structure
Files: {claude_context['file_structure']['count']}
Key files: {', '.join(claude_context['file_structure']['files'][:20])}

## Coding Instructions

1. **Follow Project Architecture**: Use the established patterns from ARCHITECTURE.md
2. **Respect Working Assumptions**: All checked assumptions in CONTEXT.md are true
3. **Build on Existing Code**: Extend and integrate with completed work orders
4. **Follow Conventions**: Match coding style and patterns from existing codebase
5. **Create Tests**: Include unit tests for new functionality
6. **Update Documentation**: Update relevant .md files with implementation details

## Deliverables Required

1. **Source Code**: All necessary implementation files
2. **Tests**: Unit tests for new functionality  
3. **Documentation Updates**: Update relevant project documents
4. **Integration Points**: Show how this integrates with existing code
5. **Working Assumptions Updates**: Any new assumptions that become true

## Implementation Guidelines

- Use the technology stack defined in ARCHITECTURE.md
- Follow security guidelines from SECURITY.md
- Implement data models as specified in DATA_MODELS.md
- Ensure API endpoints match API_SPEC.md if applicable
- Write code that can handle the assumptions in CONTEXT.md

## IMPORTANT: File Organization

**ALL project files must be created in the workspace directory structure:**
- Main project files go in: `workspace/pong_game/`
- Game code goes in: `workspace/pong_game/` (not workspace/game/)
- Tests go in: `workspace/pong_game/tests/` or alongside main files
- Documentation goes in: `workspace/pong_game/`

**Example file paths:**
```
workspace/pong_game/pong.py
workspace/pong_game/requirements.txt
workspace/pong_game/README.md
workspace/pong_game/tests/test_pong.py
```

Please implement the requested functionality with full consideration of the project context.
"""
        
        return prompt
    
    async def _is_claude_code_available(self) -> bool:
        """Check if Claude Code extension is available."""
        try:
            # Try to detect Claude Code extension
            # This would be project-specific detection logic
            result = subprocess.run(['code', '--list-extensions'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                extensions = result.stdout
                return 'claude' in extensions.lower() or 'anthropic' in extensions.lower()
        except Exception as e:
            self.logger.debug(f"Claude Code detection failed: {e}")
        
        return False
    
    async def _execute_via_claude_code(self, prompt: str, work_order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coding via Claude Code extension."""
        self.logger.info("Executing via Claude Code extension")
        
        try:
            # Create a temporary prompt file
            prompt_file = Path(f"work_order_{work_order['id']}_prompt.md")
            prompt_file.write_text(prompt)
            
            # This would integrate with Claude Code extension
            # For now, simulate the interaction
            await asyncio.sleep(1)  # Simulate processing time
            
            # In real implementation, this would:
            # 1. Open Claude Code with the prompt
            # 2. Let user interact with Claude for coding
            # 3. Collect the generated files
            # 4. Return results
            
            result = {
                "success": True,
                "method": "claude_code",
                "artifacts": ["src/new_feature.py", "tests/test_new_feature.py"],
                "files_created": 2,
                "files_modified": 1,
                "prompt_file": str(prompt_file),
                "message": "Claude Code execution completed (simulated)"
            }
            
            # Clean up
            if prompt_file.exists():
                prompt_file.unlink()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Claude Code execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_via_claude_sdk(self, prompt: str, work_order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coding via Claude SDK."""
        self.logger.info("Executing via Claude SDK")
        
        try:
            # Import Claude SDK
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.config.api_keys.anthropic_api_key)
            
            # Use Claude's large context window
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            # Process Claude's response
            claude_response = response.content[0].text
            
            # Extract code blocks and file information from response
            files_info = self._extract_files_from_claude_response(claude_response)
            
            # Save generated files
            artifacts = await self._save_generated_files(files_info, work_order['id'])
            
            return {
                "success": True,
                "method": "claude_sdk",
                "artifacts": artifacts,
                "files_created": len(artifacts),
                "claude_response": claude_response,
                "message": "Claude SDK execution completed"
            }
            
        except Exception as e:
            self.logger.error(f"Claude SDK execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_files_from_claude_response(self, response: str) -> List[Dict[str, str]]:
        """Extract file information from Claude's response."""
        files = []
        
        # Look for code blocks with file paths
        import re
        
        # Pattern to match: ```language filename or ```filename
        pattern = r'```(?:(\w+)\s+)?([^\n]+\.[\w]+)?\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for language, filename, content in matches:
            if filename and content.strip():
                files.append({
                    "filename": filename.strip(),
                    "language": language or "text",
                    "content": content.strip()
                })
            elif content.strip() and (language in ['python', 'javascript', 'typescript', 'java']):
                # Infer filename from language and content
                ext_map = {
                    'python': '.py',
                    'javascript': '.js', 
                    'typescript': '.ts',
                    'java': '.java'
                }
                files.append({
                    "filename": f"generated_file{ext_map.get(language, '.txt')}",
                    "language": language,
                    "content": content.strip()
                })
        
        return files
    
    async def _save_generated_files(self, files_info: List[Dict[str, str]], 
                                  work_order_id: str) -> List[str]:
        """Save files generated by Claude."""
        artifacts = []
        
        # Define workspace directory
        workspace_dir = Path("workspace")
        workspace_dir.mkdir(exist_ok=True)
        
        for file_info in files_info:
            filename = file_info['filename']
            content = file_info['content']
            
            # Clean up filename - remove # prefix and ensure it goes to workspace
            clean_filename = filename.lstrip('# ').strip()
            
            # Ensure all files go to workspace directory
            if not clean_filename.startswith('workspace/'):
                file_path = workspace_dir / clean_filename
            else:
                file_path = Path(clean_filename)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path.write_text(content)
            artifacts.append(str(file_path))
            
            self.logger.info(f"Saved generated file: {file_path}")
        
        return artifacts
    
    async def execute_visual_work_order(self, work_order: Dict[str, Any], 
                                      context: Dict[str, Any], 
                                      screenshots: List[str] = None) -> Dict[str, Any]:
        """
        Execute work order that requires visual analysis using Claude's vision capabilities.
        
        Args:
            work_order: Work order specification
            context: Project context
            screenshots: List of screenshot file paths for visual analysis
            
        Returns:
            Execution result with visual analysis and code generation
        """
        self.logger.info(f"Executing visual work order: {work_order['id']}")
        
        if not screenshots:
            return {"success": False, "error": "No screenshots provided for visual work order"}
        
        try:
            import anthropic
            import base64
            
            client = anthropic.Anthropic(api_key=self.config.api_keys.anthropic_api_key)
            
            # Prepare images for Claude
            images = []
            for screenshot_path in screenshots:
                if Path(screenshot_path).exists():
                    with open(screenshot_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode()
                        images.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        })
            
            # Create visual prompt
            visual_prompt = f"""# Visual Coding Work Order: {work_order['title']}

## Work Order Description
{work_order['description']}

## Visual Analysis Required
Please analyze the provided screenshots and implement the requested functionality.

## Project Context
{self._create_coding_prompt(work_order, await self._prepare_claude_context(work_order, context))}

Please analyze the visual elements and implement accordingly.
"""
            
            # Create message with text and images
            message_content = [{"type": "text", "text": visual_prompt}]
            message_content.extend(images)
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": message_content}]
            )
            
            # Process response
            claude_response = response.content[0].text
            files_info = self._extract_files_from_claude_response(claude_response)
            artifacts = await self._save_generated_files(files_info, work_order['id'])
            
            return {
                "success": True,
                "method": "claude_vision",
                "artifacts": artifacts,
                "visual_analysis": claude_response,
                "files_created": len(artifacts),
                "screenshots_analyzed": len(screenshots)
            }
            
        except Exception as e:
            self.logger.error(f"Visual work order execution failed: {e}")
            return {"success": False, "error": str(e)}