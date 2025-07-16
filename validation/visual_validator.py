"""
Visual Testing Integration for Ultimate Agentic StarterKit.

This module integrates with the existing Puppeteer visual testing system to provide
automated visual validation with screenshot capture, log analysis, and performance monitoring.
"""

import subprocess
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import tempfile
import os
from datetime import datetime

from core.logger import get_logger
from core.config import get_config


class VisualTestingValidator:
    """Integration with existing Puppeteer visual testing system"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.testing_dir = self.project_root / "testing"
        self.screenshots_dir = self.testing_dir / "screenshots"
        self.scripts_dir = self.testing_dir / "scripts"
        self.logger = get_logger("visual_validator")
        self.config = get_config()
        
        # Ensure directories exist
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
    async def validate_web_interface(self, url: str, test_name: str, 
                                   options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate web interface using visual testing
        
        Args:
            url: URL to test
            test_name: Name for the test run
            options: Additional options for the test
            
        Returns:
            Dict containing test results and confidence score
        """
        try:
            # Build command for visual testing
            cmd = [
                'node', 
                str(self.scripts_dir / 'visual-test.js'),
                url,
                '--name', test_name,
                '--output-dir', str(self.screenshots_dir)
            ]
            
            # Add optional parameters
            if options:
                if 'viewport' in options:
                    viewport = options['viewport']
                    cmd.extend(['--viewport', f"{viewport.get('width', 1920)}x{viewport.get('height', 1080)}"])
                
                if 'timeout' in options:
                    cmd.extend(['--timeout', str(options['timeout'])])
                
                if 'waitFor' in options:
                    cmd.extend(['--wait-for', options['waitFor']])
                
                if 'interactions' in options:
                    # Write interactions to temporary file
                    interactions_file = await self._write_interactions_file(options['interactions'])
                    cmd.extend(['--interactions', interactions_file])
            
            self.logger.info(f"Running visual test: {' '.join(cmd)}")
            
            # Run the visual test
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=60)
            
            if result.returncode == 0:
                # Parse test results from stdout
                stdout_str = stdout.decode()
                test_results = await self._parse_visual_test_results(stdout_str, test_name)
                
                confidence = self._calculate_visual_confidence(test_results)
                
                return {
                    "success": True,
                    "confidence": confidence,
                    "results": test_results,
                    "screenshot_path": str(self.screenshots_dir / f"{test_name}.png"),
                    "url": url,
                    "test_name": test_name,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"Visual test failed: {error_msg}")
                
                return {
                    "success": False,
                    "confidence": 0.0,
                    "error": error_msg,
                    "stdout": stdout.decode() if stdout else "",
                    "url": url,
                    "test_name": test_name,
                    "timestamp": datetime.now().isoformat()
                }
                
        except asyncio.TimeoutError:
            self.logger.error(f"Visual test timeout for {url}")
            return {
                "success": False,
                "confidence": 0.0,
                "error": "Visual test timeout",
                "url": url,
                "test_name": test_name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Visual testing failed: {e}")
            return {
                "success": False,
                "confidence": 0.0,
                "error": str(e),
                "url": url,
                "test_name": test_name,
                "timestamp": datetime.now().isoformat()
            }
    
    async def validate_prp_interface(self, prp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate PRP using comprehensive validation
        
        Args:
            prp_config: PRP validation configuration
            
        Returns:
            Dict containing validation results and confidence score
        """
        try:
            # Write PRP config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(prp_config, f, indent=2)
                config_file = f.name
            
            try:
                # Run PRP validation using existing system
                cmd = [
                    'node',
                    str(self.scripts_dir / 'prp-visual-validator.js'),
                    config_file
                ]
                
                self.logger.info(f"Running PRP validation: {' '.join(cmd)}")
                
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.project_root
                )
                
                stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=120)
                
                if result.returncode == 0:
                    # Parse validation results
                    stdout_str = stdout.decode()
                    validation_results = await self._parse_prp_validation_results(stdout_str)
                    
                    confidence = self._calculate_prp_confidence(validation_results)
                    
                    return {
                        "success": True,
                        "confidence": confidence,
                        "validation_results": validation_results,
                        "config": prp_config,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    self.logger.error(f"PRP validation failed: {error_msg}")
                    
                    return {
                        "success": False,
                        "confidence": 0.0,
                        "error": error_msg,
                        "stdout": stdout.decode() if stdout else "",
                        "config": prp_config,
                        "timestamp": datetime.now().isoformat()
                    }
            finally:
                # Clean up temporary file
                os.unlink(config_file)
                
        except Exception as e:
            self.logger.error(f"PRP validation failed: {e}")
            return {
                "success": False,
                "confidence": 0.0,
                "error": str(e),
                "config": prp_config,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _write_interactions_file(self, interactions: List[Dict[str, Any]]) -> str:
        """Write interactions to temporary file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(interactions, f, indent=2)
            return f.name
    
    async def _parse_visual_test_results(self, stdout: str, test_name: str) -> Dict[str, Any]:
        """Parse visual test results from stdout"""
        try:
            # Look for report files
            report_files = list(self.screenshots_dir.glob(f"report-*.json"))
            
            if report_files:
                # Get the most recent report file
                latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
                
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                    
                return report_data
            
            # Fallback parsing from stdout
            results = {
                "screenshots_captured": "screenshot" in stdout.lower(),
                "console_errors": "error" in stdout.lower(),
                "network_errors": "network" in stdout.lower(),
                "load_time": self._extract_load_time(stdout),
                "test_success": "success: true" in stdout.lower(),
                "raw_output": stdout
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to parse visual test results: {e}")
            return {
                "parse_error": str(e),
                "raw_output": stdout
            }
    
    async def _parse_prp_validation_results(self, stdout: str) -> Dict[str, Any]:
        """Parse PRP validation results from stdout"""
        try:
            # Look for validation report files
            report_files = list(self.screenshots_dir.glob("prp-validation-report-*.json"))
            
            if report_files:
                # Get the most recent report file
                latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
                
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                    
                return report_data
            
            # Fallback parsing from stdout
            results = {
                "validation_complete": "validation report" in stdout.lower(),
                "tests_passed": self._extract_test_count(stdout, "passed"),
                "tests_failed": self._extract_test_count(stdout, "failed"),
                "total_tests": self._extract_test_count(stdout, "total"),
                "raw_output": stdout
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to parse PRP validation results: {e}")
            return {
                "parse_error": str(e),
                "raw_output": stdout
            }
    
    def _extract_load_time(self, stdout: str) -> float:
        """Extract load time from stdout"""
        try:
            # Look for load time patterns
            import re
            
            # Pattern: "Load time: 1234ms"
            pattern = r"load time:?\s*(\d+(?:\.\d+)?)(?:ms|s)?"
            match = re.search(pattern, stdout, re.IGNORECASE)
            
            if match:
                time_value = float(match.group(1))
                # Convert to seconds if it looks like milliseconds
                if time_value > 100:  # Assume ms if > 100
                    return time_value / 1000
                return time_value
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _extract_test_count(self, stdout: str, count_type: str) -> int:
        """Extract test count from stdout"""
        try:
            import re
            
            # Pattern: "Passed: 5" or "Failed: 2"
            pattern = rf"{count_type}:?\s*(\d+)"
            match = re.search(pattern, stdout, re.IGNORECASE)
            
            if match:
                return int(match.group(1))
            
            return 0
            
        except Exception:
            return 0
    
    def _calculate_visual_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence based on visual test results"""
        base_confidence = 1.0
        
        # Check for JavaScript errors
        if results.get("console_errors", False):
            base_confidence -= 0.2
        
        # Check for network errors
        if results.get("network_errors", False):
            base_confidence -= 0.3
        
        # Check if screenshot was captured
        if not results.get("screenshots_captured", False):
            base_confidence -= 0.4
        
        # Check test success
        if not results.get("test_success", True):
            base_confidence -= 0.5
        
        # Performance penalty
        load_time = results.get("load_time", 0)
        if load_time > 5.0:  # Slow load
            base_confidence -= min(0.2, (load_time - 5.0) * 0.05)
        
        # Check for specific error patterns in logs
        if "logs" in results:
            logs = results["logs"]
            
            # JavaScript errors
            js_errors = logs.get("errors", [])
            if js_errors:
                js_error_count = len([e for e in js_errors if e.get("type") == "javascript"])
                base_confidence -= min(0.3, js_error_count * 0.1)
            
            # Network failures
            network_errors = logs.get("network", [])
            if network_errors:
                network_error_count = len([e for e in network_errors if e.get("type") == "failed"])
                base_confidence -= min(0.2, network_error_count * 0.05)
        
        return max(0.0, base_confidence)
    
    def _calculate_prp_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence based on PRP validation results"""
        try:
            # Check for summary data
            if "summary" in results:
                summary = results["summary"]
                total_tests = summary.get("totalTests", 0)
                passed_tests = summary.get("passedTests", 0)
                failed_tests = summary.get("failedTests", 0)
                
                if total_tests > 0:
                    pass_rate = passed_tests / total_tests
                    return pass_rate
            
            # Fallback calculation
            total_tests = results.get("total_tests", 0)
            passed_tests = results.get("tests_passed", 0)
            failed_tests = results.get("tests_failed", 0)
            
            if total_tests > 0:
                pass_rate = passed_tests / total_tests
                return pass_rate
            
            # If no test data, check for validation completion
            if results.get("validation_complete", False):
                return 0.5
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate PRP confidence: {e}")
            return 0.0
    
    async def run_multiple_tests(self, test_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run multiple visual tests in parallel
        
        Args:
            test_configs: List of test configurations
            
        Returns:
            Dict containing all test results and overall confidence
        """
        try:
            # Run tests in parallel
            tasks = []
            for config in test_configs:
                if "prp_config" in config:
                    task = self.validate_prp_interface(config["prp_config"])
                else:
                    task = self.validate_web_interface(
                        config["url"],
                        config["test_name"],
                        config.get("options", {})
                    )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_tests = []
            failed_tests = []
            total_confidence = 0.0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_tests.append({
                        "test_name": test_configs[i].get("test_name", f"test_{i}"),
                        "error": str(result),
                        "confidence": 0.0
                    })
                else:
                    if result.get("success", False):
                        successful_tests.append(result)
                        total_confidence += result.get("confidence", 0.0)
                    else:
                        failed_tests.append(result)
            
            # Calculate overall confidence
            total_tests = len(test_configs)
            overall_confidence = total_confidence / total_tests if total_tests > 0 else 0.0
            
            return {
                "success": len(failed_tests) == 0,
                "overall_confidence": overall_confidence,
                "total_tests": total_tests,
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "results": successful_tests + failed_tests,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Multiple test execution failed: {e}")
            return {
                "success": False,
                "overall_confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current status of visual testing system"""
        return {
            "project_root": str(self.project_root),
            "testing_dir": str(self.testing_dir),
            "screenshots_dir": str(self.screenshots_dir),
            "scripts_available": {
                "visual_test": (self.scripts_dir / "visual-test.js").exists(),
                "prp_validator": (self.scripts_dir / "prp-visual-validator.js").exists()
            },
            "recent_screenshots": len(list(self.screenshots_dir.glob("*.png"))),
            "recent_reports": len(list(self.screenshots_dir.glob("*.json")))
        }