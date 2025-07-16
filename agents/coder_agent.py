"""
Coder Agent for the Ultimate Agentic StarterKit.

This module implements the Coder Agent that generates code using Claude API
with tool calling capabilities for file operations and code generation.
"""

import json
import asyncio
import tempfile
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from agents.base_agent import BaseAgent
from core.models import AgentResult, ProjectTask, AgentType
from core.config import get_config
from core.logger import get_logger


class CoderAgent(BaseAgent):
    """
    Coder Agent that generates code using Claude API with tool calling.
    
    Specializes in:
    - Code generation and modification
    - File creation and editing
    - Code quality validation
    - Multi-language support
    - Tool calling for file operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Coder Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("coder", config)
        
        self.client = None
        self.system_config = get_config()
        self.model = self.config.get('model', 'claude-3-5-sonnet-20241022')
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.temperature = self.config.get('temperature', 0.1)
        self.max_file_size = self.config.get('max_file_size', 500)  # Max lines per file
        
        # Code quality patterns
        self.quality_patterns = {
            'python': {
                'required': ['def ', 'import ', 'from '],
                'good': ['"""', "'''", 'type hint', 'try:', 'except:', 'class '],
                'bad': ['print(', 'TODO', 'FIXME', 'hack', 'quick fix']
            },
            'javascript': {
                'required': ['function', 'const', 'let', 'var'],
                'good': ['/**', 'async', 'await', 'try', 'catch', 'class'],
                'bad': ['console.log', 'TODO', 'FIXME', 'hack']
            }
        }
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            self.logger.error("Anthropic library not available")
            return
        
        try:
            api_key = self.system_config.api_keys.anthropic_api_key
            if not api_key:
                self.logger.error("Anthropic API key not configured")
                return
            
            self.client = anthropic.Anthropic(api_key=api_key)
            self.logger.info("Anthropic client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            self.client = None
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute the coder agent to generate code.
        
        Args:
            task: The project task to execute
            
        Returns:
            AgentResult: Result with generated code
        """
        start_time = datetime.now()
        
        if not self._validate_task(task):
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error="Invalid task provided",
                execution_time=0.0,
                agent_id=self.agent_id,
                timestamp=start_time
            )
        
        if not self.client:
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error="Anthropic client not available",
                execution_time=0.0,
                agent_id=self.agent_id,
                timestamp=start_time
            )
        
        try:
            self.logger.info(f"Generating code for task: {task.title}")
            
            # Define tools for file operations
            tools = self._get_file_tools()
            
            # Create context-aware system prompt
            system_prompt = self._create_system_prompt(task)
            
            # Generate code using Claude
            response = await self._generate_code(task, system_prompt, tools)
            
            # Process tool calls and create files
            output_files = await self._process_tool_calls(response)
            
            # Validate generated code
            validation_results = self._validate_generated_code(output_files)
            
            # Calculate confidence based on code quality
            confidence = self._calculate_code_confidence(response, validation_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Generated {len(output_files)} files with confidence {confidence:.2f}")
            
            return AgentResult(
                success=True,
                confidence=confidence,
                output={
                    'files': output_files,
                    'response': self._serialize_response(response),
                    'validation': validation_results,
                    'tool_calls': len(output_files)
                },
                execution_time=execution_time,
                agent_id=self.agent_id,
                timestamp=start_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.exception(f"Error during code generation: {str(e)}")
            
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error=str(e),
                execution_time=execution_time,
                agent_id=self.agent_id,
                timestamp=start_time
            )
    
    def _get_file_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for file operations.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "create_file",
                "description": "Create a new file with content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path where to create the file"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language of the content"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "name": "modify_file",
                "description": "Modify existing file content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to modify"
                        },
                        "changes": {
                            "type": "string",
                            "description": "Description of changes to make"
                        },
                        "new_content": {
                            "type": "string",
                            "description": "New content for the file"
                        }
                    },
                    "required": ["file_path", "new_content"]
                }
            },
            {
                "name": "validate_syntax",
                "description": "Validate syntax of code",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to validate"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language"
                        }
                    },
                    "required": ["code", "language"]
                }
            }
        ]
    
    def _create_system_prompt(self, task: ProjectTask) -> str:
        """
        Create a context-aware system prompt for code generation.
        
        Args:
            task: The project task
            
        Returns:
            System prompt string
        """
        return f"""
You are an expert software engineer specializing in creating high-quality code.

Task: {task.title}
Description: {task.description}
Type: {task.type}

Requirements:
- Write clean, maintainable, and well-documented code
- Use proper error handling and input validation
- Include comprehensive docstrings and comments
- Follow language-specific best practices and conventions
- Ensure code is production-ready and secure
- Maximum {self.max_file_size} lines per file
- Use async/await patterns where appropriate
- Implement proper logging and monitoring

Code Quality Standards:
- Type hints for Python code
- Proper exception handling
- Input validation
- Security best practices
- Performance considerations
- Testing considerations

Use the provided tools to create or modify files as needed.
Focus on creating working, tested, and documented code.
"""
    
    async def _generate_code(self, task: ProjectTask, system_prompt: str, 
                           tools: List[Dict[str, Any]]) -> Any:
        """
        Generate code using Claude API.
        
        Args:
            task: The project task
            system_prompt: System prompt for context
            tools: Available tools for file operations
            
        Returns:
            Claude API response
        """
        try:
            # Create the message
            message = f"""
Please implement the following task:

{task.description}

Additional context:
- Task type: {task.type}
- Target agent: {task.agent_type}

Please use the provided tools to create the necessary files and implement the functionality.
Focus on creating complete, working solutions that follow best practices.
"""
            
            # Call Claude API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    tools=tools,
                    messages=[{"role": "user", "content": message}]
                )
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling Claude API: {str(e)}")
            raise
    
    async def _process_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Process tool calls from Claude response.
        
        Args:
            response: Claude API response
            
        Returns:
            List of created files with metadata
        """
        output_files = []
        
        try:
            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    
                    if tool_name == "create_file":
                        file_info = await self._create_file(tool_input)
                        if file_info:
                            output_files.append(file_info)
                    
                    elif tool_name == "modify_file":
                        file_info = await self._modify_file(tool_input)
                        if file_info:
                            output_files.append(file_info)
                    
                    elif tool_name == "validate_syntax":
                        validation_result = self._validate_syntax(tool_input)
                        self.logger.info(f"Syntax validation: {validation_result}")
        
        except Exception as e:
            self.logger.error(f"Error processing tool calls: {str(e)}")
        
        return output_files
    
    async def _create_file(self, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a file from tool input.
        
        Args:
            tool_input: Tool input parameters
            
        Returns:
            File information dictionary
        """
        try:
            file_path = tool_input.get('file_path')
            content = tool_input.get('content')
            language = tool_input.get('language', 'unknown')
            
            if not file_path or not content:
                self.logger.error("Missing file_path or content in create_file tool")
                return None
            
            # Ensure file path is safe
            safe_path = self._sanitize_file_path(file_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            
            # Write file
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get file info
            file_info = {
                'path': safe_path,
                'language': language,
                'size': len(content),
                'lines': len(content.splitlines()),
                'created': datetime.now().isoformat(),
                'operation': 'create'
            }
            
            self.logger.info(f"Created file: {safe_path} ({file_info['lines']} lines)")
            return file_info
            
        except Exception as e:
            self.logger.error(f"Error creating file: {str(e)}")
            return None
    
    async def _modify_file(self, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Modify a file from tool input.
        
        Args:
            tool_input: Tool input parameters
            
        Returns:
            File information dictionary
        """
        try:
            file_path = tool_input.get('file_path')
            new_content = tool_input.get('new_content')
            changes = tool_input.get('changes', 'Modified by coder agent')
            
            if not file_path or not new_content:
                self.logger.error("Missing file_path or new_content in modify_file tool")
                return None
            
            # Ensure file path is safe
            safe_path = self._sanitize_file_path(file_path)
            
            # Create backup if file exists
            if os.path.exists(safe_path):
                backup_path = f"{safe_path}.bak"
                os.rename(safe_path, backup_path)
                self.logger.info(f"Created backup: {backup_path}")
            
            # Write new content
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Get file info
            file_info = {
                'path': safe_path,
                'size': len(new_content),
                'lines': len(new_content.splitlines()),
                'modified': datetime.now().isoformat(),
                'operation': 'modify',
                'changes': changes
            }
            
            self.logger.info(f"Modified file: {safe_path} ({file_info['lines']} lines)")
            return file_info
            
        except Exception as e:
            self.logger.error(f"Error modifying file: {str(e)}")
            return None
    
    def _sanitize_file_path(self, file_path: str) -> str:
        """
        Sanitize file path to prevent directory traversal.
        
        Args:
            file_path: Input file path
            
        Returns:
            Safe file path
        """
        # Remove any path traversal attempts
        file_path = file_path.replace('..', '')
        file_path = file_path.replace('//', '/')
        
        # Ensure path is within workspace
        workspace = self.system_config.project_work_dir
        if not file_path.startswith(workspace):
            file_path = os.path.join(workspace, file_path.lstrip('/'))
        
        return os.path.normpath(file_path)
    
    def _validate_syntax(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate syntax of code.
        
        Args:
            tool_input: Tool input with code and language
            
        Returns:
            Validation result dictionary
        """
        code = tool_input.get('code', '')
        language = tool_input.get('language', 'unknown')
        
        result = {
            'language': language,
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if language == 'python':
                # Try to compile Python code
                compile(code, '<string>', 'exec')
                result['valid'] = True
            else:
                # Basic validation for other languages
                if not code.strip():
                    result['valid'] = False
                    result['errors'].append('Empty code')
        
        except SyntaxError as e:
            result['valid'] = False
            result['errors'].append(f"Syntax error: {str(e)}")
        
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_generated_code(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the quality of generated code.
        
        Args:
            files: List of generated files
            
        Returns:
            Validation results
        """
        results = {
            'total_files': len(files),
            'total_lines': sum(f.get('lines', 0) for f in files),
            'languages': set(),
            'quality_score': 0.0,
            'issues': []
        }
        
        for file_info in files:
            if not os.path.exists(file_info['path']):
                results['issues'].append(f"File not found: {file_info['path']}")
                continue
            
            try:
                with open(file_info['path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                language = file_info.get('language', 'unknown')
                results['languages'].add(language)
                
                # Check code quality patterns
                if language in self.quality_patterns:
                    patterns = self.quality_patterns[language]
                    
                    # Check for required patterns
                    required_found = sum(1 for pattern in patterns['required'] if pattern in content)
                    required_score = required_found / len(patterns['required']) if patterns['required'] else 1.0
                    
                    # Check for good patterns
                    good_found = sum(1 for pattern in patterns['good'] if pattern in content)
                    good_score = min(good_found / len(patterns['good']), 1.0) if patterns['good'] else 0.5
                    
                    # Check for bad patterns
                    bad_found = sum(1 for pattern in patterns['bad'] if pattern in content)
                    bad_penalty = min(bad_found * 0.1, 0.5)
                    
                    file_quality = (required_score * 0.5 + good_score * 0.4) - bad_penalty
                    results['quality_score'] += max(file_quality, 0.0)
                
            except Exception as e:
                results['issues'].append(f"Error reading file {file_info['path']}: {str(e)}")
        
        # Average quality score
        if results['total_files'] > 0:
            results['quality_score'] /= results['total_files']
        
        results['languages'] = list(results['languages'])
        return results
    
    def _calculate_code_confidence(self, response: Any, validation: Dict[str, Any]) -> float:
        """
        Calculate confidence score for generated code.
        
        Args:
            response: Claude API response
            validation: Code validation results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from successful response
        base_confidence = 0.6
        
        # Quality score contribution
        quality_contribution = validation.get('quality_score', 0.0) * 0.3
        
        # File count contribution (more files = more complex, potentially higher confidence)
        file_count = validation.get('total_files', 0)
        file_contribution = min(file_count * 0.05, 0.2)
        
        # Issues penalty
        issues_count = len(validation.get('issues', []))
        issues_penalty = min(issues_count * 0.1, 0.3)
        
        # Line count contribution (reasonable amount of code)
        line_count = validation.get('total_lines', 0)
        line_contribution = min(line_count / 100, 0.1)  # 100 lines = 0.1 contribution
        
        # Combine all factors
        confidence = base_confidence + quality_contribution + file_contribution + line_contribution - issues_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        """
        Serialize Claude response for output.
        
        Args:
            response: Claude API response
            
        Returns:
            Serialized response dictionary
        """
        try:
            return {
                'model': response.model,
                'usage': response.usage.dict() if hasattr(response, 'usage') else {},
                'content_blocks': len(response.content),
                'tool_calls': sum(1 for c in response.content if c.type == 'tool_use'),
                'text_content': '\n'.join(c.text for c in response.content if c.type == 'text')
            }
        except Exception as e:
            self.logger.error(f"Error serializing response: {str(e)}")
            return {'error': str(e)}
    
    def _validate_task(self, task: ProjectTask) -> bool:
        """
        Validate that the task is appropriate for the coder agent.
        
        Args:
            task: The task to validate
            
        Returns:
            True if task is valid
        """
        if not super()._validate_task(task):
            return False
        
        # Coder-specific validation
        if task.agent_type != AgentType.CODER:
            self.logger.warning(f"Task {task.id} is not for coder agent: {task.agent_type}")
            return False
        
        return True