"""
Confidence Scoring Engine for Ultimate Agentic StarterKit.

This module provides multi-factor confidence calculation with agent-specific weighting,
code quality analysis, and configurable confidence thresholds.
"""

from typing import Dict, List, Any, Optional
import ast
import subprocess
import json
import tempfile
import os
from dataclasses import dataclass
from core.models import AgentResult, ProjectTask, AgentType
from core.logger import get_logger
from core.voice_alerts import get_voice_alerts


@dataclass
class ConfidenceFactors:
    """Factors that contribute to confidence scoring"""
    syntax_correctness: float = 0.0
    type_safety: float = 0.0
    test_coverage: float = 0.0
    code_quality: float = 0.0
    performance: float = 0.0
    completeness: float = 0.0
    agent_self_confidence: float = 0.0


class ConfidenceScorer:
    """Multi-factor confidence scoring system"""
    
    def __init__(self):
        self.logger = get_logger("confidence_scorer")
        self.voice = get_voice_alerts()
        self.factor_weights = {
            AgentType.PARSER: {
                "syntax_correctness": 0.3,
                "completeness": 0.4,
                "agent_self_confidence": 0.3
            },
            AgentType.CODER: {
                "syntax_correctness": 0.25,
                "type_safety": 0.2,
                "code_quality": 0.2,
                "completeness": 0.2,
                "agent_self_confidence": 0.15
            },
            AgentType.TESTER: {
                "test_coverage": 0.4,
                "performance": 0.3,
                "completeness": 0.3
            },
            AgentType.ADVISOR: {
                "code_quality": 0.5,
                "completeness": 0.3,
                "agent_self_confidence": 0.2
            },
            AgentType.ORCHESTRATOR: {
                "completeness": 0.6,
                "agent_self_confidence": 0.4
            }
        }
    
    async def calculate_confidence(self, result: AgentResult, task: ProjectTask) -> float:
        """Calculate overall confidence score"""
        try:
            # Extract factors based on agent type
            factors = await self._extract_confidence_factors(result, task)
            
            # Get weights for agent type
            weights = self.factor_weights.get(task.agent_type, self.factor_weights[AgentType.CODER])
            
            # Calculate weighted confidence
            weighted_score = 0.0
            total_weight = 0.0
            
            for factor_name, weight in weights.items():
                factor_value = getattr(factors, factor_name, 0.0)
                weighted_score += factor_value * weight
                total_weight += weight
            
            # Normalize to 0-1 range
            final_confidence = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Apply penalties for specific issues
            final_confidence = self._apply_confidence_penalties(final_confidence, factors)
            
            if final_confidence < 0.5:
                self.voice.speak_warning('Low confidence—review needed')
            
            self.logger.info(f"Calculated confidence: {final_confidence:.3f} for {task.agent_type}")
            return min(max(final_confidence, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    async def _extract_confidence_factors(self, result: AgentResult, task: ProjectTask) -> ConfidenceFactors:
        """Extract individual confidence factors"""
        factors = ConfidenceFactors()
        
        # Agent self-confidence (from the agent's own assessment)
        factors.agent_self_confidence = result.confidence
        
        # Syntax correctness (for code-generating agents)
        if task.agent_type == AgentType.CODER and result.output:
            factors.syntax_correctness = await self._check_syntax_correctness(result.output)
            factors.type_safety = await self._check_type_safety(result.output)
            factors.code_quality = await self._check_code_quality(result.output)
        
        # Test coverage (for test-related agents)
        if task.agent_type == AgentType.TESTER and result.output:
            factors.test_coverage = await self._check_test_coverage(result.output)
            factors.performance = await self._check_performance(result.output)
        
        # Completeness (for all agents)
        factors.completeness = await self._check_completeness(result, task)
        
        return factors
    
    async def _check_syntax_correctness(self, output: Any) -> float:
        """Check syntax correctness of generated code"""
        try:
            if isinstance(output, dict) and "response" in output:
                # Extract code from tool calls or direct content
                code_content = self._extract_code_from_output(output)
                
                if not code_content:
                    return 0.5  # No code to check
                
                # Parse with AST
                try:
                    ast.parse(code_content)
                    return 1.0  # Valid syntax
                except SyntaxError as e:
                    self.logger.warning(f"Syntax error detected: {e}")
                    return 0.0
                
            return 0.5  # Unknown output format
            
        except Exception as e:
            self.logger.error(f"Syntax check failed: {e}")
            return 0.0
    
    async def _check_type_safety(self, output: Any) -> float:
        """Check type safety using mypy"""
        try:
            code_content = self._extract_code_from_output(output)
            if not code_content:
                return 0.5
            
            # Write temporary file for mypy check
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_content)
                temp_file = f.name
            
            try:
                # Run mypy
                result = subprocess.run(
                    ['mypy', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    return 1.0  # No type errors
                else:
                    # Count type errors
                    error_count = result.stderr.count('error:')
                    # Penalize based on error count
                    return max(0.0, 1.0 - (error_count * 0.1))
                    
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            self.logger.error(f"Type safety check failed: {e}")
            return 0.0
    
    async def _check_code_quality(self, output: Any) -> float:
        """Check code quality using ruff"""
        try:
            code_content = self._extract_code_from_output(output)
            if not code_content:
                return 0.5
            
            # Write temporary file for ruff check
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_content)
                temp_file = f.name
            
            try:
                # Run ruff
                result = subprocess.run(
                    ['ruff', 'check', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    return 1.0  # No quality issues
                else:
                    # Count quality issues
                    issue_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                    # Penalize based on issue count
                    return max(0.0, 1.0 - (issue_count * 0.05))
                    
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            self.logger.error(f"Code quality check failed: {e}")
            return 0.0
    
    async def _check_test_coverage(self, output: Any) -> float:
        """Check test coverage from test results"""
        try:
            if isinstance(output, dict) and "coverage" in output:
                coverage = output["coverage"]
                if isinstance(coverage, (int, float)):
                    return coverage / 100.0  # Convert percentage to 0-1 range
                elif isinstance(coverage, str) and coverage.endswith('%'):
                    return float(coverage[:-1]) / 100.0
            
            # If no coverage info, assume basic coverage
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Test coverage check failed: {e}")
            return 0.0
    
    async def _check_performance(self, output: Any) -> float:
        """Check performance metrics from test results"""
        try:
            if isinstance(output, dict) and "performance" in output:
                perf = output["performance"]
                if isinstance(perf, dict):
                    # Check execution time
                    exec_time = perf.get("execution_time", 0)
                    if exec_time > 0:
                        # Penalize slow execution (>10s is bad)
                        return max(0.0, 1.0 - (exec_time - 1.0) * 0.1)
                    
                    # Check memory usage
                    memory = perf.get("memory_usage", 0)
                    if memory > 0:
                        # Penalize high memory usage (>500MB is bad)
                        return max(0.0, 1.0 - (memory - 100) * 0.002)
            
            # If no performance info, assume good performance
            return 0.8
            
        except Exception as e:
            self.logger.error(f"Performance check failed: {e}")
            return 0.0
    
    async def _check_completeness(self, result: AgentResult, task: ProjectTask) -> float:
        """Check if the result addresses all task requirements"""
        try:
            # Basic completeness checks
            if not result.success:
                return 0.0
            
            if not result.output:
                return 0.0
            
            # Task-specific completeness checks
            if task.agent_type == AgentType.PARSER:
                return self._check_parser_completeness(result.output, task)
            elif task.agent_type == AgentType.CODER:
                return self._check_coder_completeness(result.output, task)
            elif task.agent_type == AgentType.TESTER:
                return self._check_tester_completeness(result.output, task)
            elif task.agent_type == AgentType.ADVISOR:
                return self._check_advisor_completeness(result.output, task)
            elif task.agent_type == AgentType.ORCHESTRATOR:
                return self._check_orchestrator_completeness(result.output, task)
            
            return 0.5  # Unknown agent type
            
        except Exception as e:
            self.logger.error(f"Completeness check failed: {e}")
            return 0.0
    
    def _check_parser_completeness(self, output: Any, task: ProjectTask) -> float:
        """Check parser agent completeness"""
        try:
            if isinstance(output, dict) and "milestones" in output:
                milestones = output["milestones"]
                if isinstance(milestones, list) and len(milestones) > 0:
                    # Check if milestones have required fields
                    required_fields = ["title", "description", "tasks"]
                    complete_milestones = 0
                    
                    for milestone in milestones:
                        if isinstance(milestone, dict):
                            if all(field in milestone for field in required_fields):
                                complete_milestones += 1
                    
                    return complete_milestones / len(milestones)
                else:
                    return 0.0
            
            return 0.5  # Partial output
            
        except Exception:
            return 0.0
    
    def _check_coder_completeness(self, output: Any, task: ProjectTask) -> float:
        """Check coder agent completeness"""
        try:
            code_content = self._extract_code_from_output(output)
            if not code_content:
                return 0.0
            
            # Check for basic code structure
            completeness_score = 0.0
            
            # Check for functions/classes
            if 'def ' in code_content or 'class ' in code_content:
                completeness_score += 0.3
            
            # Check for imports
            if 'import ' in code_content or 'from ' in code_content:
                completeness_score += 0.2
            
            # Check for docstrings
            if '"""' in code_content or "'''" in code_content:
                completeness_score += 0.2
            
            # Check for type hints
            if ':' in code_content and '->' in code_content:
                completeness_score += 0.2
            
            # Check for error handling
            if 'try:' in code_content and 'except' in code_content:
                completeness_score += 0.1
            
            return min(completeness_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _check_tester_completeness(self, output: Any, task: ProjectTask) -> float:
        """Check tester agent completeness"""
        try:
            if isinstance(output, dict):
                # Check for test results
                if "test_results" in output:
                    test_results = output["test_results"]
                    if isinstance(test_results, dict):
                        # Check for required test metrics
                        required_metrics = ["passed", "failed", "total"]
                        if all(metric in test_results for metric in required_metrics):
                            return 1.0
                        else:
                            return 0.5
                
                # Check for test coverage
                if "coverage" in output:
                    return 0.8
                
                # Check for performance metrics
                if "performance" in output:
                    return 0.6
            
            return 0.3  # Minimal output
            
        except Exception:
            return 0.0
    
    def _check_advisor_completeness(self, output: Any, task: ProjectTask) -> float:
        """Check advisor agent completeness"""
        try:
            if isinstance(output, dict):
                # Check for recommendations
                if "recommendations" in output:
                    recommendations = output["recommendations"]
                    if isinstance(recommendations, list) and len(recommendations) > 0:
                        return 1.0
                
                # Check for code review
                if "code_review" in output:
                    return 0.8
                
                # Check for suggestions
                if "suggestions" in output:
                    return 0.6
            
            return 0.3  # Minimal output
            
        except Exception:
            return 0.0
    
    def _check_orchestrator_completeness(self, output: Any, task: ProjectTask) -> float:
        """Check orchestrator agent completeness"""
        try:
            if isinstance(output, dict):
                # Check for task breakdown
                if "tasks" in output:
                    tasks = output["tasks"]
                    if isinstance(tasks, list) and len(tasks) > 0:
                        return 1.0
                
                # Check for planning
                if "plan" in output:
                    return 0.8
                
                # Check for agent assignments
                if "agent_assignments" in output:
                    return 0.6
            
            return 0.3  # Minimal output
            
        except Exception:
            return 0.0
    
    def _extract_code_from_output(self, output: Any) -> str:
        """Extract code content from agent output"""
        if isinstance(output, str):
            return output
        elif isinstance(output, dict):
            # Handle tool calling output
            if "response" in output:
                response = output["response"]
                if isinstance(response, list):
                    # Extract from tool calls
                    code_parts = []
                    for item in response:
                        if hasattr(item, 'type') and item.type == 'tool_use':
                            if hasattr(item, 'input') and 'content' in item.input:
                                code_parts.append(item.input['content'])
                    return '\n'.join(code_parts)
                elif isinstance(response, str):
                    return response
            
            # Handle direct code content
            if "code" in output:
                return output["code"]
            
            # Handle content field
            if "content" in output:
                return output["content"]
        
        return ""
    
    def _apply_confidence_penalties(self, base_confidence: float, factors: ConfidenceFactors) -> float:
        """Apply penalties for specific quality issues"""
        penalties = 0.0
        
        # Severe syntax errors
        if factors.syntax_correctness < 0.3:
            penalties += 0.3
        
        # Type safety issues
        if factors.type_safety < 0.5:
            penalties += 0.2
        
        # Poor test coverage
        if factors.test_coverage < 0.6:
            penalties += 0.1
        
        # Low completeness
        if factors.completeness < 0.5:
            penalties += 0.15
        
        return max(0.0, base_confidence - penalties)
    
    def get_confidence_breakdown(self, factors: ConfidenceFactors, agent_type: AgentType) -> Dict[str, Any]:
        """Get detailed confidence breakdown for debugging"""
        weights = self.factor_weights.get(agent_type, self.factor_weights[AgentType.CODER])
        
        breakdown = {
            "factors": {
                "syntax_correctness": factors.syntax_correctness,
                "type_safety": factors.type_safety,
                "test_coverage": factors.test_coverage,
                "code_quality": factors.code_quality,
                "performance": factors.performance,
                "completeness": factors.completeness,
                "agent_self_confidence": factors.agent_self_confidence
            },
            "weights": weights,
            "weighted_scores": {}
        }
        
        for factor_name, weight in weights.items():
            factor_value = getattr(factors, factor_name, 0.0)
            breakdown["weighted_scores"][factor_name] = factor_value * weight
        
        return breakdown