"""
Test Execution Framework for Ultimate Agentic StarterKit.

This module provides automated test execution for multiple frameworks with
test result parsing, coverage analysis, and performance benchmarking.
"""

import subprocess
import asyncio
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import tempfile
import os
import re
import time
from datetime import datetime
from dataclasses import dataclass

from core.logger import get_logger
from core.config import get_config


@dataclass
class TestResult:
    """Individual test result"""
    name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None


@dataclass
class TestSuite:
    """Test suite results"""
    name: str
    tests: List[TestResult]
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    coverage: Optional[float] = None


class TestExecutionFramework:
    """Automated test execution for multiple frameworks"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = get_logger("test_runner")
        self.config = get_config()
        
        # Test framework commands
        self.test_commands = {
            "pytest": ["python", "-m", "pytest"],
            "unittest": ["python", "-m", "unittest"],
            "nose2": ["nose2"],
            "tox": ["tox"],
            "coverage": ["coverage"],
            "mypy": ["mypy"],
            "ruff": ["ruff"],
            "black": ["black"],
            "isort": ["isort"]
        }
        
        # Environment setup
        self.test_env = os.environ.copy()
        self.test_env["PYTHONPATH"] = str(self.project_root)
        
    async def run_pytest_tests(self, test_path: Optional[str] = None, 
                              options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run pytest tests with coverage and performance analysis
        
        Args:
            test_path: Path to test directory or file
            options: Additional pytest options
            
        Returns:
            Dict containing test results and metrics
        """
        try:
            # Build pytest command
            cmd = self.test_commands["pytest"].copy()
            
            # Add coverage options
            cmd.extend([
                "--cov=.",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--junit-xml=test-results.xml",
                "-v",
                "--tb=short"
            ])
            
            # Add custom options
            if options:
                cmd.extend(options)
            
            # Add test path
            if test_path:
                cmd.append(test_path)
            
            self.logger.info(f"Running pytest: {' '.join(cmd)}")
            
            # Execute tests
            start_time = time.time()
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse results
            test_results = await self._parse_pytest_results(result)
            coverage_data = await self._parse_coverage_results()
            
            return {
                "framework": "pytest",
                "success": result["returncode"] == 0,
                "execution_time": execution_time,
                "test_results": test_results,
                "coverage": coverage_data,
                "command": " ".join(cmd),
                "output": result["stdout"],
                "errors": result["stderr"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Pytest execution failed: {e}")
            return {
                "framework": "pytest",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_unittest_tests(self, test_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run unittest tests
        
        Args:
            test_path: Path to test module or directory
            
        Returns:
            Dict containing test results
        """
        try:
            # Build unittest command
            cmd = self.test_commands["unittest"].copy()
            cmd.extend(["discover", "-v"])
            
            if test_path:
                cmd.extend(["-s", test_path])
            
            self.logger.info(f"Running unittest: {' '.join(cmd)}")
            
            # Execute tests
            start_time = time.time()
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse results
            test_results = await self._parse_unittest_results(result)
            
            return {
                "framework": "unittest",
                "success": result["returncode"] == 0,
                "execution_time": execution_time,
                "test_results": test_results,
                "command": " ".join(cmd),
                "output": result["stdout"],
                "errors": result["stderr"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Unittest execution failed: {e}")
            return {
                "framework": "unittest",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_type_checking(self, target_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run type checking with mypy
        
        Args:
            target_path: Path to check
            
        Returns:
            Dict containing type checking results
        """
        try:
            # Build mypy command
            cmd = self.test_commands["mypy"].copy()
            cmd.extend([
                "--json-report", "mypy-report.json",
                "--ignore-missing-imports",
                "--show-error-codes"
            ])
            
            if target_path:
                cmd.append(target_path)
            else:
                cmd.append(".")
            
            self.logger.info(f"Running mypy: {' '.join(cmd)}")
            
            # Execute type checking
            start_time = time.time()
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse results
            type_results = await self._parse_mypy_results(result)
            
            return {
                "framework": "mypy",
                "success": result["returncode"] == 0,
                "execution_time": execution_time,
                "type_results": type_results,
                "command": " ".join(cmd),
                "output": result["stdout"],
                "errors": result["stderr"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Type checking failed: {e}")
            return {
                "framework": "mypy",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_linting(self, target_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run linting with ruff
        
        Args:
            target_path: Path to lint
            
        Returns:
            Dict containing linting results
        """
        try:
            # Build ruff command
            cmd = self.test_commands["ruff"].copy()
            cmd.extend(["check", "--output-format=json"])
            
            if target_path:
                cmd.append(target_path)
            else:
                cmd.append(".")
            
            self.logger.info(f"Running ruff: {' '.join(cmd)}")
            
            # Execute linting
            start_time = time.time()
            result = await self._run_command(cmd)
            execution_time = time.time() - start_time
            
            # Parse results
            lint_results = await self._parse_ruff_results(result)
            
            return {
                "framework": "ruff",
                "success": result["returncode"] == 0,
                "execution_time": execution_time,
                "lint_results": lint_results,
                "command": " ".join(cmd),
                "output": result["stdout"],
                "errors": result["stderr"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Linting failed: {e}")
            return {
                "framework": "ruff",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_all_tests(self, test_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run all configured tests
        
        Args:
            test_config: Configuration for test execution
            
        Returns:
            Dict containing all test results
        """
        try:
            config = test_config or {}
            
            # Determine which tests to run
            test_types = config.get("test_types", ["pytest", "mypy", "ruff"])
            
            # Run tests in parallel
            tasks = []
            
            if "pytest" in test_types:
                tasks.append(self.run_pytest_tests(
                    test_path=config.get("pytest_path"),
                    options=config.get("pytest_options", [])
                ))
            
            if "unittest" in test_types:
                tasks.append(self.run_unittest_tests(
                    test_path=config.get("unittest_path")
                ))
            
            if "mypy" in test_types:
                tasks.append(self.run_type_checking(
                    target_path=config.get("mypy_path")
                ))
            
            if "ruff" in test_types:
                tasks.append(self.run_linting(
                    target_path=config.get("ruff_path")
                ))
            
            # Execute all tests
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            all_results = {}
            overall_success = True
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    test_type = test_types[i]
                    all_results[test_type] = {
                        "success": False,
                        "error": str(result),
                        "framework": test_type
                    }
                    overall_success = False
                else:
                    framework = result.get("framework", f"test_{i}")
                    all_results[framework] = result
                    if not result.get("success", False):
                        overall_success = False
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(all_results)
            
            return {
                "overall_success": overall_success,
                "overall_metrics": overall_metrics,
                "individual_results": all_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return {
                "overall_success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a command and return results"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
                env=self.test_env
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode()
            }
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }
    
    async def _parse_pytest_results(self, result: Dict[str, Any]) -> Optional[TestSuite]:
        """Parse pytest results from XML output"""
        try:
            xml_path = self.project_root / "test-results.xml"
            
            if not xml_path.exists():
                return self._parse_pytest_text_output(result["stdout"])
            
            # Parse JUnit XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            test_results = []
            
            for testcase in root.findall(".//testcase"):
                name = testcase.get("name", "unknown")
                duration = float(testcase.get("time", 0))
                
                # Check for failures/errors
                failure = testcase.find("failure")
                error = testcase.find("error")
                skipped = testcase.find("skipped")
                
                if failure is not None:
                    status = "failed"
                    message = failure.get("message", "")
                    error_text = failure.text
                elif error is not None:
                    status = "error"
                    message = error.get("message", "")
                    error_text = error.text
                elif skipped is not None:
                    status = "skipped"
                    message = skipped.get("message", "")
                    error_text = None
                else:
                    status = "passed"
                    message = None
                    error_text = None
                
                test_results.append(TestResult(
                    name=name,
                    status=status,
                    duration=duration,
                    message=message,
                    error=error_text
                ))
            
            # Calculate totals
            total = len(test_results)
            passed = len([t for t in test_results if t.status == "passed"])
            failed = len([t for t in test_results if t.status == "failed"])
            skipped = len([t for t in test_results if t.status == "skipped"])
            errors = len([t for t in test_results if t.status == "error"])
            duration = sum(t.duration for t in test_results)
            
            return TestSuite(
                name="pytest",
                tests=test_results,
                total=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                duration=duration
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse pytest results: {e}")
            return None
    
    def _parse_pytest_text_output(self, stdout: str) -> Optional[TestSuite]:
        """Parse pytest results from text output"""
        try:
            # Extract test results using regex
            test_results = []
            
            # Pattern for test results
            pattern = r"(\S+\.py::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)"
            matches = re.findall(pattern, stdout)
            
            for match in matches:
                test_name, status = match
                test_results.append(TestResult(
                    name=test_name,
                    status=status.lower(),
                    duration=0.0
                ))
            
            # Extract summary
            summary_pattern = r"(\d+) passed|(\d+) failed|(\d+) skipped|(\d+) error"
            summary_matches = re.findall(summary_pattern, stdout)
            
            passed = failed = skipped = errors = 0
            for match in summary_matches:
                if match[0]:  # passed
                    passed = int(match[0])
                elif match[1]:  # failed
                    failed = int(match[1])
                elif match[2]:  # skipped
                    skipped = int(match[2])
                elif match[3]:  # errors
                    errors = int(match[3])
            
            total = passed + failed + skipped + errors
            
            return TestSuite(
                name="pytest",
                tests=test_results,
                total=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                duration=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse pytest text output: {e}")
            return None
    
    async def _parse_unittest_results(self, result: Dict[str, Any]) -> Optional[TestSuite]:
        """Parse unittest results from text output"""
        try:
            stdout = result["stdout"]
            
            # Extract test results
            test_results = []
            
            # Pattern for test results
            pattern = r"(\S+)\s+\.\.\.\s+(ok|FAIL|ERROR|SKIP)"
            matches = re.findall(pattern, stdout)
            
            for match in matches:
                test_name, status = match
                status_map = {
                    "ok": "passed",
                    "FAIL": "failed",
                    "ERROR": "error",
                    "SKIP": "skipped"
                }
                
                test_results.append(TestResult(
                    name=test_name,
                    status=status_map.get(status, "unknown"),
                    duration=0.0
                ))
            
            # Calculate totals
            total = len(test_results)
            passed = len([t for t in test_results if t.status == "passed"])
            failed = len([t for t in test_results if t.status == "failed"])
            skipped = len([t for t in test_results if t.status == "skipped"])
            errors = len([t for t in test_results if t.status == "error"])
            
            return TestSuite(
                name="unittest",
                tests=test_results,
                total=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                duration=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse unittest results: {e}")
            return None
    
    async def _parse_coverage_results(self) -> Optional[Dict[str, Any]]:
        """Parse coverage results from JSON output"""
        try:
            coverage_path = self.project_root / "coverage.json"
            
            if not coverage_path.exists():
                return None
            
            with open(coverage_path, 'r') as f:
                coverage_data = json.load(f)
            
            # Extract summary
            summary = coverage_data.get("totals", {})
            
            return {
                "coverage_percent": summary.get("percent_covered", 0.0),
                "lines_covered": summary.get("covered_lines", 0),
                "lines_total": summary.get("num_statements", 0),
                "branches_covered": summary.get("covered_branches", 0),
                "branches_total": summary.get("num_branches", 0),
                "files": coverage_data.get("files", {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse coverage results: {e}")
            return None
    
    async def _parse_mypy_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse mypy results"""
        try:
            stdout = result["stdout"]
            
            # Count errors and warnings
            error_count = stdout.count("error:")
            warning_count = stdout.count("warning:")
            note_count = stdout.count("note:")
            
            # Extract specific errors
            errors = []
            for line in stdout.split('\n'):
                if "error:" in line:
                    errors.append(line.strip())
            
            return {
                "error_count": error_count,
                "warning_count": warning_count,
                "note_count": note_count,
                "errors": errors,
                "success": error_count == 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse mypy results: {e}")
            return {"error": str(e)}
    
    async def _parse_ruff_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ruff results"""
        try:
            stdout = result["stdout"]
            
            if not stdout.strip():
                return {
                    "issues": [],
                    "issue_count": 0,
                    "success": True
                }
            
            # Try to parse as JSON
            try:
                issues = json.loads(stdout)
                return {
                    "issues": issues,
                    "issue_count": len(issues),
                    "success": len(issues) == 0
                }
            except json.JSONDecodeError:
                # Parse as text
                issue_count = len(stdout.strip().split('\n'))
                return {
                    "issues": stdout.strip().split('\n'),
                    "issue_count": issue_count,
                    "success": issue_count == 0
                }
            
        except Exception as e:
            self.logger.error(f"Failed to parse ruff results: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics from all test results"""
        try:
            metrics = {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "error_tests": 0,
                "total_execution_time": 0.0,
                "coverage_percent": 0.0,
                "type_errors": 0,
                "lint_issues": 0,
                "overall_success": True
            }
            
            for framework, result in results.items():
                if not result.get("success", False):
                    metrics["overall_success"] = False
                
                # Add execution time
                metrics["total_execution_time"] += result.get("execution_time", 0.0)
                
                # Process test results
                if "test_results" in result:
                    test_suite = result["test_results"]
                    if test_suite:
                        metrics["total_tests"] += test_suite.total
                        metrics["passed_tests"] += test_suite.passed
                        metrics["failed_tests"] += test_suite.failed
                        metrics["skipped_tests"] += test_suite.skipped
                        metrics["error_tests"] += test_suite.errors
                
                # Process coverage
                if "coverage" in result and result["coverage"]:
                    metrics["coverage_percent"] = result["coverage"]["coverage_percent"]
                
                # Process type checking
                if "type_results" in result:
                    type_results = result["type_results"]
                    metrics["type_errors"] += type_results.get("error_count", 0)
                
                # Process linting
                if "lint_results" in result:
                    lint_results = result["lint_results"]
                    metrics["lint_issues"] += lint_results.get("issue_count", 0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overall metrics: {e}")
            return {"error": str(e)}
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current status of test execution system"""
        return {
            "project_root": str(self.project_root),
            "available_frameworks": list(self.test_commands.keys()),
            "framework_availability": {
                framework: self._check_framework_availability(framework)
                for framework in self.test_commands.keys()
            }
        }
    
    def _check_framework_availability(self, framework: str) -> bool:
        """Check if a testing framework is available"""
        try:
            cmd = self.test_commands[framework]
            result = subprocess.run(
                cmd + ["--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False