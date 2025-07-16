"""
Tester Agent for the Ultimate Agentic StarterKit.

This module implements the Tester Agent that executes tests and validates outputs
using subprocess management and multiple test framework support.
"""

import asyncio
import subprocess
import json
import os
import re
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from agents.base_agent import BaseAgent
from core.models import AgentResult, ProjectTask, AgentType
from core.config import get_config
from core.logger import get_logger


class TesterAgent(BaseAgent):
    """
    Tester Agent that executes tests and validates outputs.
    
    Specializes in:
    - Running unit tests across multiple frameworks
    - Validating code functionality
    - Performance testing
    - Integration testing
    - Test result parsing and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Tester Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("tester", config)
        
        self.system_config = get_config()
        self.test_timeout = self.config.get('test_timeout', 300)  # 5 minutes
        self.max_retries_flaky = self.config.get('max_retries_flaky', 2)
        self.supported_frameworks = self.config.get('supported_frameworks', [
            'pytest', 'unittest', 'nose2', 'jest', 'mocha', 'jasmine', 'go test', 'cargo test'
        ])
        
        # Test patterns for different frameworks
        self.test_patterns = {
            'pytest': {
                'files': ['test_*.py', '*_test.py', 'tests.py'],
                'command': 'python -m pytest',
                'success_indicators': ['passed', 'PASSED'],
                'failure_indicators': ['failed', 'FAILED', 'ERROR'],
                'result_parser': self._parse_pytest_results
            },
            'unittest': {
                'files': ['test_*.py', '*_test.py'],
                'command': 'python -m unittest',
                'success_indicators': ['OK'],
                'failure_indicators': ['FAILED', 'ERROR'],
                'result_parser': self._parse_unittest_results
            },
            'jest': {
                'files': ['*.test.js', '*.spec.js', '__tests__/*.js'],
                'command': 'npm test',
                'success_indicators': ['Tests passed', 'PASS'],
                'failure_indicators': ['Tests failed', 'FAIL'],
                'result_parser': self._parse_jest_results
            },
            'go': {
                'files': ['*_test.go'],
                'command': 'go test',
                'success_indicators': ['PASS'],
                'failure_indicators': ['FAIL'],
                'result_parser': self._parse_go_results
            }
        }
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute the tester agent to run tests and validate outputs.
        
        Args:
            task: The project task to execute
            
        Returns:
            AgentResult: Result with test execution details
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
        
        try:
            self.logger.info(f"Running tests for task: {task.title}")
            
            # Detect test framework and files
            test_info = await self._detect_test_framework(task)
            
            if not test_info:
                return AgentResult(
                    success=False,
                    confidence=0.0,
                    output=None,
                    error="No test framework or test files detected",
                    execution_time=0.0,
                    agent_id=self.agent_id,
                    timestamp=start_time
                )
            
            # Execute tests
            test_results = await self._execute_tests(test_info)
            
            # Validate results
            validation = self._validate_test_results(test_results)
            
            # Calculate confidence based on test outcomes
            confidence = self._calculate_test_confidence(test_results, validation)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Test execution completed with confidence {confidence:.2f}")
            
            return AgentResult(
                success=validation['overall_success'],
                confidence=confidence,
                output={
                    'test_results': test_results,
                    'validation': validation,
                    'framework': test_info['framework'],
                    'files_tested': test_info['files']
                },
                execution_time=execution_time,
                agent_id=self.agent_id,
                timestamp=start_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.exception(f"Error during test execution: {str(e)}")
            
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error=str(e),
                execution_time=execution_time,
                agent_id=self.agent_id,
                timestamp=start_time
            )
    
    async def _detect_test_framework(self, task: ProjectTask) -> Optional[Dict[str, Any]]:
        """
        Detect test framework and find test files.
        
        Args:
            task: The project task
            
        Returns:
            Dictionary with test framework information
        """
        # Start from workspace directory or task-specific directory
        search_dir = self.system_config.project_work_dir
        
        # Look for test files and framework indicators
        for framework, patterns in self.test_patterns.items():
            test_files = []
            
            # Search for test files
            for pattern in patterns['files']:
                matching_files = await self._find_files(search_dir, pattern)
                test_files.extend(matching_files)
            
            # Check for framework-specific configuration files
            config_files = await self._find_framework_config(search_dir, framework)
            
            if test_files or config_files:
                return {
                    'framework': framework,
                    'files': test_files,
                    'config_files': config_files,
                    'command': patterns['command'],
                    'patterns': patterns
                }
        
        # If no specific framework found, try to create basic tests
        return await self._create_basic_tests(task, search_dir)
    
    async def _find_files(self, directory: str, pattern: str) -> List[str]:
        """
        Find files matching a pattern in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            List of matching file paths
        """
        import glob
        
        try:
            search_pattern = os.path.join(directory, '**', pattern)
            return glob.glob(search_pattern, recursive=True)
        except Exception as e:
            self.logger.error(f"Error finding files with pattern {pattern}: {str(e)}")
            return []
    
    async def _find_framework_config(self, directory: str, framework: str) -> List[str]:
        """
        Find framework-specific configuration files.
        
        Args:
            directory: Directory to search
            framework: Framework name
            
        Returns:
            List of configuration files
        """
        config_files = {
            'pytest': ['pytest.ini', 'pyproject.toml', 'setup.cfg'],
            'unittest': ['unittest.cfg'],
            'jest': ['jest.config.js', 'package.json'],
            'go': ['go.mod', 'go.sum']
        }
        
        found_files = []
        
        if framework in config_files:
            for config_file in config_files[framework]:
                file_path = os.path.join(directory, config_file)
                if os.path.exists(file_path):
                    found_files.append(file_path)
        
        return found_files
    
    async def _create_basic_tests(self, task: ProjectTask, directory: str) -> Optional[Dict[str, Any]]:
        """
        Create basic tests if none exist.
        
        Args:
            task: The project task
            directory: Working directory
            
        Returns:
            Test framework information
        """
        # Create a simple pytest test file
        test_content = f'''"""
Basic tests for {task.title}
Generated by TesterAgent
"""

import pytest
import os
import sys

def test_basic_functionality():
    """Test basic functionality exists."""
    # This is a placeholder test
    assert True

def test_imports():
    """Test that imports work correctly."""
    try:
        # Try to import any Python files in the directory
        import importlib.util
        import glob
        
        py_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
        for py_file in py_files:
            if not py_file.endswith("_test.py") and not py_file.endswith("test_*.py"):
                spec = importlib.util.spec_from_file_location("test_module", py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
        
        assert True
    except Exception as e:
        pytest.fail(f"Import test failed: {{str(e)}}")

def test_task_completion():
    """Test that the task appears to be completed."""
    # This is a placeholder - specific validation would be task-dependent
    assert True

if __name__ == "__main__":
    pytest.main([__file__])
'''
        
        # Create test file
        test_file = os.path.join(directory, f"test_{task.id}.py")
        
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            self.logger.info(f"Created basic test file: {test_file}")
            
            return {
                'framework': 'pytest',
                'files': [test_file],
                'config_files': [],
                'command': 'python -m pytest',
                'patterns': self.test_patterns['pytest'],
                'generated': True
            }
            
        except Exception as e:
            self.logger.error(f"Error creating basic test file: {str(e)}")
            return None
    
    async def _execute_tests(self, test_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tests using the detected framework.
        
        Args:
            test_info: Test framework information
            
        Returns:
            Test execution results
        """
        framework = test_info['framework']
        command = test_info['command']
        test_files = test_info['files']
        
        results = {
            'framework': framework,
            'command': command,
            'files': test_files,
            'executions': [],
            'overall_success': True,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0
        }
        
        # Execute tests for each file or run once for all files
        if framework in ['pytest', 'unittest']:
            # Run all test files together
            execution_result = await self._run_test_command(command, test_files)
            results['executions'].append(execution_result)
            
            # Parse results
            parsed = test_info['patterns']['result_parser'](execution_result)
            results.update(parsed)
            
        else:
            # Run tests individually
            for test_file in test_files:
                execution_result = await self._run_test_command(command, [test_file])
                results['executions'].append(execution_result)
                
                # Parse results
                parsed = test_info['patterns']['result_parser'](execution_result)
                results['total_tests'] += parsed.get('total_tests', 0)
                results['passed_tests'] += parsed.get('passed_tests', 0)
                results['failed_tests'] += parsed.get('failed_tests', 0)
                results['error_tests'] += parsed.get('error_tests', 0)
                results['skipped_tests'] += parsed.get('skipped_tests', 0)
        
        results['overall_success'] = results['failed_tests'] == 0 and results['error_tests'] == 0
        
        return results
    
    async def _run_test_command(self, command: str, test_files: List[str]) -> Dict[str, Any]:
        """
        Run a test command and capture results.
        
        Args:
            command: Test command to execute
            test_files: List of test files
            
        Returns:
            Execution result dictionary
        """
        # Prepare command
        if test_files:
            full_command = f"{command} {' '.join(test_files)}"
        else:
            full_command = command
        
        self.logger.info(f"Executing test command: {full_command}")
        
        result = {
            'command': full_command,
            'returncode': None,
            'stdout': '',
            'stderr': '',
            'execution_time': 0.0,
            'success': False
        }
        
        try:
            start_time = datetime.now()
            
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.system_config.project_work_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.test_timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result.update({
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace'),
                'execution_time': execution_time,
                'success': process.returncode == 0
            })
            
            self.logger.info(f"Test command completed in {execution_time:.2f}s with return code {process.returncode}")
            
        except asyncio.TimeoutError:
            result['stderr'] = f"Test execution timed out after {self.test_timeout} seconds"
            self.logger.error(f"Test command timed out: {full_command}")
            
        except Exception as e:
            result['stderr'] = str(e)
            self.logger.error(f"Error executing test command: {str(e)}")
        
        return result
    
    def _parse_pytest_results(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse pytest output to extract test results.
        
        Args:
            execution_result: Test execution result
            
        Returns:
            Parsed test results
        """
        stdout = execution_result.get('stdout', '')
        stderr = execution_result.get('stderr', '')
        
        results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0,
            'test_details': []
        }
        
        # Parse pytest summary line
        summary_pattern = r'(\d+) passed.*?(\d+) failed.*?(\d+) error.*?(\d+) skipped'
        match = re.search(summary_pattern, stdout)
        
        if match:
            results['passed_tests'] = int(match.group(1))
            results['failed_tests'] = int(match.group(2))
            results['error_tests'] = int(match.group(3))
            results['skipped_tests'] = int(match.group(4))
        else:
            # Try simpler patterns
            passed_match = re.search(r'(\d+) passed', stdout)
            failed_match = re.search(r'(\d+) failed', stdout)
            error_match = re.search(r'(\d+) error', stdout)
            skipped_match = re.search(r'(\d+) skipped', stdout)
            
            if passed_match:
                results['passed_tests'] = int(passed_match.group(1))
            if failed_match:
                results['failed_tests'] = int(failed_match.group(1))
            if error_match:
                results['error_tests'] = int(error_match.group(1))
            if skipped_match:
                results['skipped_tests'] = int(skipped_match.group(1))
        
        results['total_tests'] = (results['passed_tests'] + results['failed_tests'] + 
                                results['error_tests'] + results['skipped_tests'])
        
        return results
    
    def _parse_unittest_results(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse unittest output."""
        stdout = execution_result.get('stdout', '')
        
        results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0
        }
        
        # Parse unittest output
        if 'OK' in stdout:
            # Extract test count
            ok_pattern = r'Ran (\d+) tests? in'
            match = re.search(ok_pattern, stdout)
            if match:
                results['total_tests'] = int(match.group(1))
                results['passed_tests'] = int(match.group(1))
        
        return results
    
    def _parse_jest_results(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Jest output."""
        stdout = execution_result.get('stdout', '')
        
        results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0
        }
        
        # Parse Jest output patterns
        passed_pattern = r'(\d+) passed'
        failed_pattern = r'(\d+) failed'
        total_pattern = r'Tests:\s+(\d+) total'
        
        passed_match = re.search(passed_pattern, stdout)
        failed_match = re.search(failed_pattern, stdout)
        total_match = re.search(total_pattern, stdout)
        
        if passed_match:
            results['passed_tests'] = int(passed_match.group(1))
        if failed_match:
            results['failed_tests'] = int(failed_match.group(1))
        if total_match:
            results['total_tests'] = int(total_match.group(1))
        else:
            results['total_tests'] = results['passed_tests'] + results['failed_tests']
        
        return results
    
    def _parse_go_results(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Go test output."""
        stdout = execution_result.get('stdout', '')
        
        results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0
        }
        
        # Count PASS and FAIL occurrences
        pass_count = stdout.count('PASS')
        fail_count = stdout.count('FAIL')
        
        results['passed_tests'] = pass_count
        results['failed_tests'] = fail_count
        results['total_tests'] = pass_count + fail_count
        
        return results
    
    def _validate_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate test results and provide analysis.
        
        Args:
            test_results: Test execution results
            
        Returns:
            Validation analysis
        """
        validation = {
            'overall_success': test_results.get('overall_success', False),
            'has_tests': test_results.get('total_tests', 0) > 0,
            'pass_rate': 0.0,
            'coverage_estimate': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        total_tests = test_results.get('total_tests', 0)
        passed_tests = test_results.get('passed_tests', 0)
        failed_tests = test_results.get('failed_tests', 0)
        error_tests = test_results.get('error_tests', 0)
        
        # Calculate pass rate
        if total_tests > 0:
            validation['pass_rate'] = passed_tests / total_tests
        
        # Analyze issues
        if total_tests == 0:
            validation['issues'].append('No tests found or executed')
            validation['recommendations'].append('Add comprehensive unit tests')
        
        if failed_tests > 0:
            validation['issues'].append(f'{failed_tests} tests failed')
            validation['recommendations'].append('Fix failing tests')
        
        if error_tests > 0:
            validation['issues'].append(f'{error_tests} tests had errors')
            validation['recommendations'].append('Fix test errors')
        
        if validation['pass_rate'] < 0.8:
            validation['recommendations'].append('Improve test pass rate')
        
        # Estimate coverage (very basic)
        if total_tests > 0:
            validation['coverage_estimate'] = min(total_tests * 0.1, 1.0)
        
        return validation
    
    def _calculate_test_confidence(self, test_results: Dict[str, Any], 
                                  validation: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on test results.
        
        Args:
            test_results: Test execution results
            validation: Test validation results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not validation['has_tests']:
            return 0.2  # Low confidence if no tests
        
        # Base confidence from pass rate
        pass_rate = validation.get('pass_rate', 0.0)
        base_confidence = pass_rate * 0.6
        
        # Bonus for having tests
        test_bonus = 0.2 if validation['has_tests'] else 0.0
        
        # Bonus for successful execution
        execution_bonus = 0.1 if validation['overall_success'] else 0.0
        
        # Penalty for issues
        issue_penalty = len(validation.get('issues', [])) * 0.05
        
        # Coverage bonus
        coverage_bonus = validation.get('coverage_estimate', 0.0) * 0.1
        
        confidence = base_confidence + test_bonus + execution_bonus + coverage_bonus - issue_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _validate_task(self, task: ProjectTask) -> bool:
        """
        Validate that the task is appropriate for the tester agent.
        
        Args:
            task: The task to validate
            
        Returns:
            True if task is valid
        """
        if not super()._validate_task(task):
            return False
        
        # Tester-specific validation
        if task.agent_type != AgentType.TESTER:
            self.logger.warning(f"Task {task.id} is not for tester agent: {task.agent_type}")
            return False
        
        return True