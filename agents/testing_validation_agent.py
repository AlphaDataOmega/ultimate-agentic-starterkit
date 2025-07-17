"""
Testing Validation Agent for the Ultimate Agentic StarterKit.

This agent handles comprehensive error capture, retry logic coordination, and creates 
revision documents for failed work orders. It implements max 5 retry attempts with 
exponential backoff and escalates to BugBountyAgent when max retries are exceeded.
"""

import json
import asyncio
import subprocess
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import traceback

from agents.base_agent import BaseAgent
from core.models import ProjectTask, AgentResult, AgentType
from core.logger import get_logger
from core.config import get_config
from core.knowledge_base import ProjectKnowledgeBase


class TestingValidationAgent(BaseAgent):
    """
    Agent responsible for comprehensive testing validation and retry logic.
    
    Handles:
    - Running unit tests, integration tests
    - Capturing console errors, backend logs
    - Creating revision documents for retry PRPs
    - Max 5 retry attempts with exponential backoff
    - Escalation to BugBountyAgent after max retries
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Testing Validation Agent."""
        super().__init__("testing_validation", config)
        self.system_config = get_config()
        self.knowledge_base = ProjectKnowledgeBase(".", project_name=self.config.get('project_name', None))
        
        # Testing framework detection
        self.supported_frameworks = {
            'python': ['pytest', 'unittest', 'nose2'],
            'javascript': ['jest', 'mocha', 'jasmine', 'vitest'],
            'java': ['junit', 'testng'],
            'go': ['go test'],
            'rust': ['cargo test']
        }
        
        # Error patterns for analysis
        self.error_patterns = {
            'import_error': r'ImportError|ModuleNotFoundError|import.*not found',
            'syntax_error': r'SyntaxError|IndentationError|TabError',
            'runtime_error': r'RuntimeError|ValueError|TypeError|AttributeError',
            'test_failure': r'FAILED|AssertionError|Test failed',
            'network_error': r'ConnectionError|TimeoutError|NetworkError',
            'database_error': r'DatabaseError|ConnectionError.*database',
            'permission_error': r'PermissionError|Access denied',
            'file_error': r'FileNotFoundError|IOError|OSError'
        }
        
        # Retry configuration
        self.max_retries = 5
        self.retry_delay_base = 2.0  # Base delay in seconds
        self.retry_delay_max = 60.0  # Max delay in seconds
        
        # Log capture configuration
        self.log_directories = ['logs', 'log', '.']
        self.log_extensions = ['.log', '.txt', '.out', '.err']
        
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute testing validation task.
        
        Args:
            task: Testing validation task with work order results
            
        Returns:
            AgentResult with validation results and retry logic
        """
        self.logger.info(f"Starting testing validation for task: {task.title}")
        
        try:
            # Validate task
            if not self._validate_task(task):
                return AgentResult(
                    success=False,
                    confidence=0.0,
                    output=None,
                    error="Invalid testing validation task",
                    execution_time=0.0,
                    agent_id=self.agent_id,
                    timestamp=datetime.now()
                )
            
            # Parse work order results from task
            work_order_data = self._parse_work_order_data(task.description)
            
            # Run comprehensive testing validation
            validation_results = await self._run_comprehensive_validation(work_order_data)
            
            # Process validation results
            if validation_results['success']:
                # Success - prepare for handoff to next agent
                return await self._handle_validation_success(validation_results, work_order_data)
            else:
                # Failure - implement retry logic
                return await self._handle_validation_failure(validation_results, work_order_data, task)
        
        except Exception as e:
            self.logger.exception(f"Testing validation agent execution failed: {str(e)}")
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error=str(e),
                execution_time=0.0,
                agent_id=self.agent_id,
                timestamp=datetime.now()
            )
    
    def _parse_work_order_data(self, task_description: str) -> Dict[str, Any]:
        """Parse work order data from task description."""
        work_order_data = {
            'work_order_id': 'unknown',
            'files_created': [],
            'files_modified': [],
            'test_files': [],
            'expected_browser_appearance': None,
            'technology_stack': [],
            'attempt_number': 1,
            'previous_failures': []
        }
        
        try:
            # Try to parse as JSON first
            if task_description.strip().startswith('{'):
                parsed = json.loads(task_description)
                work_order_data.update(parsed)
            else:
                # Parse from structured text
                lines = task_description.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('Work Order ID:'):
                        work_order_data['work_order_id'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Files Created:'):
                        work_order_data['files_created'] = [f.strip() for f in line.split(':', 1)[1].split(',') if f.strip()]
                    elif line.startswith('Files Modified:'):
                        work_order_data['files_modified'] = [f.strip() for f in line.split(':', 1)[1].split(',') if f.strip()]
                    elif line.startswith('Test Files:'):
                        work_order_data['test_files'] = [f.strip() for f in line.split(':', 1)[1].split(',') if f.strip()]
                    elif line.startswith('Technology Stack:'):
                        work_order_data['technology_stack'] = [t.strip() for t in line.split(':', 1)[1].split(',') if t.strip()]
                    elif line.startswith('Attempt:'):
                        work_order_data['attempt_number'] = int(line.split(':', 1)[1].strip())
        
        except Exception as e:
            self.logger.warning(f"Failed to parse work order data: {e}")
        
        return work_order_data
    
    async def _run_comprehensive_validation(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive testing validation."""
        validation_results = {
            'success': True,
            'test_results': {},
            'console_errors': [],
            'log_errors': [],
            'code_analysis': {},
            'performance_metrics': {},
            'error_categories': {},
            'execution_time': 0.0,
            'detailed_failures': []
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: Run unit tests
            test_results = await self._run_unit_tests(work_order_data)
            validation_results['test_results'] = test_results
            
            # Step 2: Capture console errors
            console_errors = await self._capture_console_errors(work_order_data)
            validation_results['console_errors'] = console_errors
            
            # Step 3: Analyze log files
            log_errors = await self._analyze_log_files()
            validation_results['log_errors'] = log_errors
            
            # Step 4: Static code analysis
            code_analysis = await self._run_code_analysis(work_order_data)
            validation_results['code_analysis'] = code_analysis
            
            # Step 5: Performance validation
            performance_metrics = await self._validate_performance(work_order_data)
            validation_results['performance_metrics'] = performance_metrics
            
            # Step 6: Categorize errors
            error_categories = self._categorize_errors(
                console_errors, log_errors, test_results.get('failures', [])
            )
            validation_results['error_categories'] = error_categories
            
            # Determine overall success
            validation_results['success'] = self._determine_validation_success(validation_results)
            
        except Exception as e:
            validation_results['success'] = False
            validation_results['error'] = str(e)
            self.logger.error(f"Comprehensive validation failed: {e}")
        
        validation_results['execution_time'] = (datetime.now() - start_time).total_seconds()
        return validation_results
    
    async def _run_unit_tests(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run unit tests for the work order."""
        test_results = {
            'framework': 'unknown',
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': [],
            'output': '',
            'execution_time': 0.0
        }
        
        try:
            # Detect testing framework
            framework = self._detect_testing_framework(work_order_data)
            test_results['framework'] = framework
            
            if framework == 'pytest':
                result = await self._run_pytest(work_order_data)
            elif framework == 'unittest':
                result = await self._run_unittest(work_order_data)
            elif framework == 'jest':
                result = await self._run_jest(work_order_data)
            elif framework == 'mocha':
                result = await self._run_mocha(work_order_data)
            else:
                # Try to run any available test files
                result = await self._run_generic_tests(work_order_data)
            
            test_results.update(result)
            
        except Exception as e:
            test_results['error'] = str(e)
            self.logger.error(f"Unit test execution failed: {e}")
        
        return test_results
    
    def _detect_testing_framework(self, work_order_data: Dict[str, Any]) -> str:
        """Detect the testing framework being used."""
        # Check for framework-specific files
        if Path('pytest.ini').exists() or Path('pyproject.toml').exists():
            return 'pytest'
        elif Path('package.json').exists():
            try:
                with open('package.json', 'r') as f:
                    package_data = json.load(f)
                    deps = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
                    if 'jest' in deps:
                        return 'jest'
                    elif 'mocha' in deps:
                        return 'mocha'
            except:
                pass
        
        # Check test file patterns
        test_files = work_order_data.get('test_files', [])
        for test_file in test_files:
            if 'test_' in test_file or '_test' in test_file:
                if test_file.endswith('.py'):
                    return 'pytest'
                elif test_file.endswith('.js') or test_file.endswith('.ts'):
                    return 'jest'
        
        return 'unknown'
    
    async def _run_pytest(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run pytest tests."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': [],
            'output': '',
            'execution_time': 0.0
        }
        
        try:
            # Run pytest with verbose output
            cmd = ['python', '-m', 'pytest', '-v', '--tb=short']
            
            # Add specific test files if provided
            test_files = work_order_data.get('test_files', [])
            if test_files:
                cmd.extend([f for f in test_files if Path(f).exists()])
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='.'
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode() + stderr.decode()
            result['output'] = output
            
            # Parse pytest output
            result.update(self._parse_pytest_output(output))
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Pytest execution failed: {e}")
        
        return result
    
    async def _run_unittest(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run unittest tests."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': [],
            'output': '',
            'execution_time': 0.0
        }
        
        try:
            # Run unittest discover
            cmd = ['python', '-m', 'unittest', 'discover', '-v']
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='.'
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode() + stderr.decode()
            result['output'] = output
            
            # Parse unittest output
            result.update(self._parse_unittest_output(output))
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Unittest execution failed: {e}")
        
        return result
    
    async def _run_jest(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Jest tests."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': [],
            'output': '',
            'execution_time': 0.0
        }
        
        try:
            # Run Jest with verbose output
            cmd = ['npm', 'test', '--', '--verbose']
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='.'
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode() + stderr.decode()
            result['output'] = output
            
            # Parse Jest output
            result.update(self._parse_jest_output(output))
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Jest execution failed: {e}")
        
        return result
    
    async def _run_mocha(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Mocha tests."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': [],
            'output': '',
            'execution_time': 0.0
        }
        
        try:
            # Run Mocha with reporter
            cmd = ['npx', 'mocha', '--reporter', 'json']
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='.'
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode() + stderr.decode()
            result['output'] = output
            
            # Parse Mocha output
            result.update(self._parse_mocha_output(output))
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Mocha execution failed: {e}")
        
        return result
    
    async def _run_generic_tests(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run generic tests when framework is unknown."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': [],
            'output': 'Generic test execution attempted',
            'execution_time': 0.0
        }
        
        # Check if test files exist and try to run them
        test_files = work_order_data.get('test_files', [])
        for test_file in test_files:
            if Path(test_file).exists():
                try:
                    if test_file.endswith('.py'):
                        # Try to run Python test file
                        cmd = ['python', test_file]
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd='.'
                        )
                        stdout, stderr = await process.communicate()
                        if process.returncode == 0:
                            result['passed_tests'] += 1
                        else:
                            result['failed_tests'] += 1
                            result['failures'].append({
                                'file': test_file,
                                'error': stderr.decode()
                            })
                        result['total_tests'] += 1
                except Exception as e:
                    result['failures'].append({
                        'file': test_file,
                        'error': str(e)
                    })
        
        return result
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': []
        }
        
        # Extract test summary
        summary_match = re.search(r'(\d+) passed.*?(\d+) failed.*?(\d+) skipped', output)
        if summary_match:
            result['passed_tests'] = int(summary_match.group(1))
            result['failed_tests'] = int(summary_match.group(2))
            result['skipped_tests'] = int(summary_match.group(3))
            result['total_tests'] = result['passed_tests'] + result['failed_tests'] + result['skipped_tests']
        
        # Extract failure details
        failure_pattern = r'FAILED (.*?) - (.*?)(?=\n\n|\nCOLLECTED|\n=|$)'
        failures = re.findall(failure_pattern, output, re.DOTALL)
        for test_name, error in failures:
            result['failures'].append({
                'test': test_name.strip(),
                'error': error.strip()
            })
        
        return result
    
    def _parse_unittest_output(self, output: str) -> Dict[str, Any]:
        """Parse unittest output to extract test results."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': []
        }
        
        # Count test results
        lines = output.split('\n')
        for line in lines:
            if 'OK' in line:
                result['passed_tests'] += 1
            elif 'FAIL' in line or 'ERROR' in line:
                result['failed_tests'] += 1
            elif 'SKIP' in line:
                result['skipped_tests'] += 1
        
        result['total_tests'] = result['passed_tests'] + result['failed_tests'] + result['skipped_tests']
        
        return result
    
    def _parse_jest_output(self, output: str) -> Dict[str, Any]:
        """Parse Jest output to extract test results."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': []
        }
        
        # Extract test summary
        summary_match = re.search(r'Tests:\s+(\d+) failed.*?(\d+) passed.*?(\d+) total', output)
        if summary_match:
            result['failed_tests'] = int(summary_match.group(1))
            result['passed_tests'] = int(summary_match.group(2))
            result['total_tests'] = int(summary_match.group(3))
        
        return result
    
    def _parse_mocha_output(self, output: str) -> Dict[str, Any]:
        """Parse Mocha output to extract test results."""
        result = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'failures': []
        }
        
        try:
            # Try to parse JSON output
            if output.strip().startswith('{'):
                json_result = json.loads(output)
                result['total_tests'] = json_result.get('tests', 0)
                result['passed_tests'] = json_result.get('passes', 0)
                result['failed_tests'] = json_result.get('failures', 0)
                result['skipped_tests'] = json_result.get('pending', 0)
        except:
            # Fallback to text parsing
            pass
        
        return result
    
    async def _capture_console_errors(self, work_order_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Capture console errors from running the application."""
        console_errors = []
        
        try:
            # Look for common error patterns in files
            files_to_check = work_order_data.get('files_created', []) + work_order_data.get('files_modified', [])
            
            for file_path in files_to_check:
                if Path(file_path).exists():
                    try:
                        content = Path(file_path).read_text()
                        # Look for error patterns in the code
                        for error_type, pattern in self.error_patterns.items():
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            for match in matches:
                                console_errors.append({
                                    'type': error_type,
                                    'file': file_path,
                                    'error': match,
                                    'line': self._find_line_number(content, match)
                                })
                    except Exception as e:
                        console_errors.append({
                            'type': 'file_error',
                            'file': file_path,
                            'error': str(e),
                            'line': 0
                        })
        
        except Exception as e:
            self.logger.error(f"Console error capture failed: {e}")
        
        return console_errors
    
    async def _analyze_log_files(self) -> List[Dict[str, Any]]:
        """Analyze log files for errors."""
        log_errors = []
        
        try:
            # Find log files
            log_files = []
            for log_dir in self.log_directories:
                log_path = Path(log_dir)
                if log_path.exists():
                    for ext in self.log_extensions:
                        log_files.extend(log_path.glob(f'*{ext}'))
            
            # Analyze each log file
            for log_file in log_files:
                try:
                    content = log_file.read_text()
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines):
                        # Look for error patterns
                        for error_type, pattern in self.error_patterns.items():
                            if re.search(pattern, line, re.IGNORECASE):
                                log_errors.append({
                                    'type': error_type,
                                    'file': str(log_file),
                                    'line': i + 1,
                                    'error': line.strip(),
                                    'timestamp': self._extract_timestamp(line)
                                })
                
                except Exception as e:
                    log_errors.append({
                        'type': 'log_analysis_error',
                        'file': str(log_file),
                        'error': str(e),
                        'line': 0
                    })
        
        except Exception as e:
            self.logger.error(f"Log file analysis failed: {e}")
        
        return log_errors
    
    async def _run_code_analysis(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run static code analysis."""
        analysis_results = {
            'syntax_errors': [],
            'style_issues': [],
            'complexity_issues': [],
            'security_issues': []
        }
        
        try:
            files_to_analyze = work_order_data.get('files_created', []) + work_order_data.get('files_modified', [])
            
            for file_path in files_to_analyze:
                if Path(file_path).exists():
                    try:
                        # Basic syntax check
                        if file_path.endswith('.py'):
                            analysis_results.update(await self._analyze_python_file(file_path))
                        elif file_path.endswith(('.js', '.ts')):
                            analysis_results.update(await self._analyze_javascript_file(file_path))
                    except Exception as e:
                        analysis_results['syntax_errors'].append({
                            'file': file_path,
                            'error': str(e)
                        })
        
        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
        
        return analysis_results
    
    async def _analyze_python_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze Python file for issues."""
        analysis = {
            'syntax_errors': [],
            'style_issues': [],
            'complexity_issues': [],
            'security_issues': []
        }
        
        try:
            # Try to compile the file
            with open(file_path, 'r') as f:
                source = f.read()
            
            compile(source, file_path, 'exec')
            
        except SyntaxError as e:
            analysis['syntax_errors'].append({
                'file': file_path,
                'line': e.lineno,
                'error': str(e)
            })
        except Exception as e:
            analysis['syntax_errors'].append({
                'file': file_path,
                'error': str(e)
            })
        
        return analysis
    
    async def _analyze_javascript_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript file for issues."""
        analysis = {
            'syntax_errors': [],
            'style_issues': [],
            'complexity_issues': [],
            'security_issues': []
        }
        
        # Basic analysis - could be extended with proper JS/TS parser
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for common syntax issues
            if 'undefined' in content and 'typeof' not in content:
                analysis['style_issues'].append({
                    'file': file_path,
                    'issue': 'Potential undefined variable usage'
                })
        
        except Exception as e:
            analysis['syntax_errors'].append({
                'file': file_path,
                'error': str(e)
            })
        
        return analysis
    
    async def _validate_performance(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance metrics."""
        performance_metrics = {
            'response_time': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'issues': []
        }
        
        # Basic performance validation - could be extended
        try:
            # Check file sizes
            files_to_check = work_order_data.get('files_created', [])
            for file_path in files_to_check:
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    if file_size > 1024 * 1024:  # 1MB
                        performance_metrics['issues'].append({
                            'type': 'large_file',
                            'file': file_path,
                            'size': file_size
                        })
        
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
        
        return performance_metrics
    
    def _categorize_errors(self, console_errors: List[Dict], log_errors: List[Dict], 
                          test_failures: List[Dict]) -> Dict[str, Any]:
        """Categorize all errors by type."""
        categories = {
            'import_errors': [],
            'syntax_errors': [],
            'runtime_errors': [],
            'test_failures': [],
            'network_errors': [],
            'database_errors': [],
            'permission_errors': [],
            'file_errors': [],
            'other_errors': []
        }
        
        # Categorize console errors
        for error in console_errors:
            error_type = error.get('type', 'other_errors')
            if error_type in categories:
                categories[error_type].append(error)
            else:
                categories['other_errors'].append(error)
        
        # Categorize log errors
        for error in log_errors:
            error_type = error.get('type', 'other_errors')
            if error_type in categories:
                categories[error_type].append(error)
            else:
                categories['other_errors'].append(error)
        
        # Add test failures
        categories['test_failures'].extend(test_failures)
        
        return categories
    
    def _determine_validation_success(self, validation_results: Dict[str, Any]) -> bool:
        """Determine if validation was successful."""
        # Check test results
        test_results = validation_results.get('test_results', {})
        if test_results.get('failed_tests', 0) > 0:
            return False
        
        # Check for critical errors
        console_errors = validation_results.get('console_errors', [])
        log_errors = validation_results.get('log_errors', [])
        
        critical_error_types = ['syntax_error', 'import_error', 'runtime_error']
        for error in console_errors + log_errors:
            if error.get('type') in critical_error_types:
                return False
        
        # Check code analysis
        code_analysis = validation_results.get('code_analysis', {})
        if code_analysis.get('syntax_errors'):
            return False
        
        return True
    
    async def _handle_validation_success(self, validation_results: Dict[str, Any], 
                                       work_order_data: Dict[str, Any]) -> AgentResult:
        """Handle successful validation."""
        self.logger.info("Validation successful - preparing handoff")
        
        # Calculate confidence based on test coverage and quality
        confidence = self._calculate_confidence({
            'test_success_rate': self._calculate_test_success_rate(validation_results),
            'error_count': len(validation_results.get('console_errors', [])),
            'code_quality': self._assess_code_quality(validation_results),
            'performance': self._assess_performance(validation_results)
        })
        
        return AgentResult(
            success=True,
            confidence=confidence,
            output={
                'validation_results': validation_results,
                'work_order_data': work_order_data,
                'next_agent': 'visual_testing' if self._requires_visual_testing(work_order_data) else 'documentation',
                'handoff_data': {
                    'test_passed': True,
                    'errors_resolved': True,
                    'ready_for_next_stage': True
                }
            },
            error=None,
            execution_time=validation_results.get('execution_time', 0.0),
            agent_id=self.agent_id,
            timestamp=datetime.now()
        )
    
    async def _handle_validation_failure(self, validation_results: Dict[str, Any], 
                                       work_order_data: Dict[str, Any], 
                                       task: ProjectTask) -> AgentResult:
        """Handle validation failure with retry logic."""
        attempt_number = work_order_data.get('attempt_number', 1)
        
        if attempt_number >= self.max_retries:
            # Max retries reached - escalate to BugBountyAgent
            self.logger.error(f"Max retries ({self.max_retries}) reached for work order {work_order_data.get('work_order_id')}")
            return await self._escalate_to_bug_bounty(validation_results, work_order_data)
        
        # Create revision PRP for retry
        revision_prp = await self._create_revision_prp(validation_results, work_order_data, attempt_number)
        
        # Calculate retry delay
        retry_delay = min(
            self.retry_delay_base * (2 ** (attempt_number - 1)),
            self.retry_delay_max
        )
        
        self.logger.info(f"Validation failed - creating revision PRP for attempt {attempt_number + 1}")
        
        return AgentResult(
            success=False,
            confidence=0.2,  # Low confidence due to failure
            output={
                'validation_results': validation_results,
                'work_order_data': work_order_data,
                'retry_info': {
                    'attempt_number': attempt_number,
                    'max_retries': self.max_retries,
                    'retry_delay': retry_delay,
                    'revision_prp': revision_prp
                },
                'next_action': 'retry_with_revision'
            },
            error=f"Validation failed on attempt {attempt_number}",
            execution_time=validation_results.get('execution_time', 0.0),
            agent_id=self.agent_id,
            timestamp=datetime.now()
        )
    
    async def _create_revision_prp(self, validation_results: Dict[str, Any], 
                                 work_order_data: Dict[str, Any], 
                                 attempt_number: int) -> str:
        """Create revision PRP for retry."""
        work_order_id = work_order_data.get('work_order_id', 'unknown')
        revision_filename = f"PRPs/{work_order_id}_revision_{attempt_number + 1}.md"
        
        # Load revision template
        template_path = Path("PRPs/templates/prp_revision_template.md")
        if template_path.exists():
            template_content = template_path.read_text()
        else:
            template_content = self._get_default_revision_template()
        
        # Fill in template with failure analysis
        revision_content = template_content.replace('[PROJECT_NAME]', work_order_data.get('feature_name', 'Unknown'))
        revision_content = revision_content.replace('[ORIGINAL_PRP_FILE_PATH]', f"PRPs/{work_order_id}_prp.md")
        revision_content = revision_content.replace('[1-5]', str(attempt_number + 1))
        revision_content = revision_content.replace('[DATE]', datetime.now().strftime('%Y-%m-%d'))
        revision_content = revision_content.replace('[WORK_ORDER_ID]', work_order_id)
        revision_content = revision_content.replace('[BRIEF_DESCRIPTION]', self._summarize_failures(validation_results))
        
        # Add detailed failure analysis
        failure_analysis = self._generate_failure_analysis(validation_results)
        revision_content = revision_content.replace('[SPECIFIC_TEST_FAILURES]', failure_analysis['test_failures'])
        revision_content = revision_content.replace('[CONSOLE_ERRORS_LOGGED]', failure_analysis['console_errors'])
        revision_content = revision_content.replace('[INTEGRATION_PROBLEMS]', failure_analysis['integration_problems'])
        revision_content = revision_content.replace('[FULL_ERROR_LOGS_AND_STACK_TRACES]', failure_analysis['full_logs'])
        revision_content = revision_content.replace('[ANALYSIS_OF_WHY_THE_IMPLEMENTATION_FAILED]', failure_analysis['root_cause'])
        
        # Save revision PRP
        revision_path = Path(revision_filename)
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        revision_path.write_text(revision_content)
        
        self.logger.info(f"Created revision PRP: {revision_filename}")
        return revision_filename
    
    async def _escalate_to_bug_bounty(self, validation_results: Dict[str, Any], 
                                    work_order_data: Dict[str, Any]) -> AgentResult:
        """Escalate to BugBountyAgent after max retries."""
        self.logger.info("Escalating to BugBountyAgent for deep debugging")
        
        return AgentResult(
            success=False,
            confidence=0.1,
            output={
                'validation_results': validation_results,
                'work_order_data': work_order_data,
                'escalation_info': {
                    'reason': 'max_retries_exceeded',
                    'attempts_made': self.max_retries,
                    'failure_summary': self._summarize_failures(validation_results)
                },
                'next_agent': 'bug_bounty',
                'requires_deep_debugging': True
            },
            error=f"Max retries ({self.max_retries}) exceeded - escalating to BugBountyAgent",
            execution_time=validation_results.get('execution_time', 0.0),
            agent_id=self.agent_id,
            timestamp=datetime.now()
        )
    
    def _requires_visual_testing(self, work_order_data: Dict[str, Any]) -> bool:
        """Check if work order requires visual testing."""
        # Check if there's expected browser appearance data
        if work_order_data.get('expected_browser_appearance'):
            return True
        
        # Check technology stack for frontend frameworks
        tech_stack = work_order_data.get('technology_stack', [])
        frontend_indicators = ['react', 'vue', 'angular', 'html', 'css', 'javascript', 'typescript']
        
        for tech in tech_stack:
            if any(indicator in tech.lower() for indicator in frontend_indicators):
                return True
        
        return False
    
    def _calculate_test_success_rate(self, validation_results: Dict[str, Any]) -> float:
        """Calculate test success rate."""
        test_results = validation_results.get('test_results', {})
        total_tests = test_results.get('total_tests', 0)
        passed_tests = test_results.get('passed_tests', 0)
        
        if total_tests == 0:
            return 0.5  # No tests is neutral
        
        return passed_tests / total_tests
    
    def _assess_code_quality(self, validation_results: Dict[str, Any]) -> float:
        """Assess code quality based on analysis."""
        code_analysis = validation_results.get('code_analysis', {})
        
        syntax_errors = len(code_analysis.get('syntax_errors', []))
        style_issues = len(code_analysis.get('style_issues', []))
        
        if syntax_errors > 0:
            return 0.0
        elif style_issues > 5:
            return 0.5
        else:
            return 1.0
    
    def _assess_performance(self, validation_results: Dict[str, Any]) -> float:
        """Assess performance metrics."""
        performance = validation_results.get('performance_metrics', {})
        issues = performance.get('issues', [])
        
        if len(issues) > 3:
            return 0.3
        elif len(issues) > 0:
            return 0.7
        else:
            return 1.0
    
    def _summarize_failures(self, validation_results: Dict[str, Any]) -> str:
        """Summarize validation failures."""
        failures = []
        
        test_results = validation_results.get('test_results', {})
        if test_results.get('failed_tests', 0) > 0:
            failures.append(f"{test_results['failed_tests']} test failures")
        
        console_errors = validation_results.get('console_errors', [])
        if console_errors:
            failures.append(f"{len(console_errors)} console errors")
        
        log_errors = validation_results.get('log_errors', [])
        if log_errors:
            failures.append(f"{len(log_errors)} log errors")
        
        return ', '.join(failures) if failures else 'Unknown failure'
    
    def _generate_failure_analysis(self, validation_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate detailed failure analysis."""
        analysis = {
            'test_failures': '',
            'console_errors': '',
            'integration_problems': '',
            'full_logs': '',
            'root_cause': ''
        }
        
        # Analyze test failures
        test_results = validation_results.get('test_results', {})
        failures = test_results.get('failures', [])
        if failures:
            analysis['test_failures'] = '\n'.join([f"- {f.get('test', 'Unknown')}: {f.get('error', 'Unknown error')}" for f in failures])
        
        # Analyze console errors
        console_errors = validation_results.get('console_errors', [])
        if console_errors:
            analysis['console_errors'] = '\n'.join([f"- {e.get('file', 'Unknown')}: {e.get('error', 'Unknown error')}" for e in console_errors])
        
        # Generate root cause analysis
        error_categories = validation_results.get('error_categories', {})
        if error_categories.get('import_errors'):
            analysis['root_cause'] = "Import/dependency issues detected"
        elif error_categories.get('syntax_errors'):
            analysis['root_cause'] = "Syntax errors in code"
        elif error_categories.get('runtime_errors'):
            analysis['root_cause'] = "Runtime errors during execution"
        else:
            analysis['root_cause'] = "Multiple failure modes detected"
        
        return analysis
    
    def _get_default_revision_template(self) -> str:
        """Get default revision template if file doesn't exist."""
        return """# Project Revision: [PROJECT_NAME]
**Original PRP**: [ORIGINAL_PRP_FILE_PATH]
**Revision Number**: [1-5]
**Created**: [DATE]
**Work Order ID**: [WORK_ORDER_ID]
**Status**: RETRY
**Previous Failure**: [BRIEF_DESCRIPTION]

## Previous Failure Analysis
### What Failed
- **Testing Issues**: [SPECIFIC_TEST_FAILURES]
- **Console Errors**: [CONSOLE_ERRORS_LOGGED]
- **Integration Issues**: [INTEGRATION_PROBLEMS]

### Error Details
```
[FULL_ERROR_LOGS_AND_STACK_TRACES]
```

### Root Cause Analysis
[ANALYSIS_OF_WHY_THE_IMPLEMENTATION_FAILED]

## Claude Code Agent Instructions - Revision
Please address the specific failures mentioned above and implement a corrected version.
"""
    
    def _find_line_number(self, content: str, match: str) -> int:
        """Find line number of a match in content."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if match in line:
                return i + 1
        return 0
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        """Extract timestamp from log line."""
        # Common timestamp patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(0)
        
        return None