"""
Advisor Agent for the Ultimate Agentic StarterKit.

This module implements the Advisor Agent that provides code review and improvement
suggestions using OpenAI's o3 model for advanced reasoning capabilities.
"""

import asyncio
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

from agents.base_agent import BaseAgent
from core.models import AgentResult, ProjectTask, AgentType
from core.config import get_config
from core.logger import get_logger


class AdvisorAgent(BaseAgent):
    """
    Advisor Agent that provides code review and improvement suggestions.
    
    Specializes in:
    - Code review with advanced reasoning
    - Architecture improvement suggestions
    - Performance optimization recommendations
    - Security vulnerability detection
    - Best practice validation
    - Code quality assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Advisor Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("advisor", config)
        
        self.client = None
        self.system_config = get_config()
        self.model = self.config.get('model', 'o3-mini')
        self.reasoning_effort = self.config.get('reasoning_effort', 'medium')
        self.max_tokens = self.config.get('max_tokens', 8000)
        self.temperature = self.config.get('temperature', 0.3)
        self.max_file_size = self.config.get('max_file_size', 10000)  # Max chars per file
        
        # Review categories
        self.review_categories = {
            'code_quality': {
                'weight': 0.25,
                'criteria': [
                    'readability', 'maintainability', 'consistency', 
                    'documentation', 'naming_conventions'
                ]
            },
            'architecture': {
                'weight': 0.20,
                'criteria': [
                    'design_patterns', 'modularity', 'separation_of_concerns',
                    'scalability', 'extensibility'
                ]
            },
            'performance': {
                'weight': 0.15,
                'criteria': [
                    'efficiency', 'resource_usage', 'algorithmic_complexity',
                    'caching', 'optimization_opportunities'
                ]
            },
            'security': {
                'weight': 0.20,
                'criteria': [
                    'input_validation', 'authentication', 'authorization',
                    'data_protection', 'vulnerability_prevention'
                ]
            },
            'testing': {
                'weight': 0.10,
                'criteria': [
                    'test_coverage', 'test_quality', 'testability',
                    'edge_cases', 'error_handling'
                ]
            },
            'best_practices': {
                'weight': 0.10,
                'criteria': [
                    'language_conventions', 'framework_usage', 'dependencies',
                    'error_handling', 'logging'
                ]
            }
        }
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        if not OPENAI_AVAILABLE:
            self.logger.error("OpenAI library not available")
            return
        
        try:
            api_key = self.system_config.api_keys.openai_api_key
            if not api_key:
                self.logger.error("OpenAI API key not configured")
                return
            
            self.client = openai.OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute the advisor agent to provide code review and suggestions.
        
        Args:
            task: The project task to execute
            
        Returns:
            AgentResult: Result with review and suggestions
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
                error="OpenAI client not available",
                execution_time=0.0,
                agent_id=self.agent_id,
                timestamp=start_time
            )
        
        try:
            self.logger.info(f"Performing code review for task: {task.title}")
            
            # Gather code files for review
            code_files = await self._gather_code_files(task)
            
            if not code_files:
                return AgentResult(
                    success=False,
                    confidence=0.0,
                    output=None,
                    error="No code files found for review",
                    execution_time=0.0,
                    agent_id=self.agent_id,
                    timestamp=start_time
                )
            
            # Perform comprehensive code review
            review_results = await self._perform_code_review(code_files, task)
            
            # Generate improvement suggestions
            suggestions = await self._generate_suggestions(review_results, task)
            
            # Calculate confidence based on review quality
            confidence = self._calculate_review_confidence(review_results, suggestions)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Code review completed with confidence {confidence:.2f}")
            
            return AgentResult(
                success=True,
                confidence=confidence,
                output={
                    'review_results': review_results,
                    'suggestions': suggestions,
                    'files_reviewed': len(code_files),
                    'categories_analyzed': list(self.review_categories.keys())
                },
                execution_time=execution_time,
                agent_id=self.agent_id,
                timestamp=start_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.exception(f"Error during code review: {str(e)}")
            
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error=str(e),
                execution_time=execution_time,
                agent_id=self.agent_id,
                timestamp=start_time
            )
    
    async def _gather_code_files(self, task: ProjectTask) -> List[Dict[str, Any]]:
        """
        Gather code files for review.
        
        Args:
            task: The project task
            
        Returns:
            List of code file information
        """
        code_files = []
        workspace = self.system_config.project_work_dir
        
        # Code file extensions
        code_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        try:
            # Walk through workspace directory
            for root, dirs, files in os.walk(workspace):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file)
                    
                    if ext.lower() in code_extensions:
                        # Check file size
                        if os.path.getsize(file_path) > self.max_file_size:
                            self.logger.warning(f"File {file_path} too large, skipping")
                            continue
                        
                        # Read file content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            code_files.append({
                                'path': file_path,
                                'relative_path': os.path.relpath(file_path, workspace),
                                'language': code_extensions[ext.lower()],
                                'content': content,
                                'size': len(content),
                                'lines': len(content.splitlines())
                            })
                            
                        except Exception as e:
                            self.logger.error(f"Error reading file {file_path}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error gathering code files: {str(e)}")
        
        self.logger.info(f"Gathered {len(code_files)} code files for review")
        return code_files
    
    async def _perform_code_review(self, code_files: List[Dict[str, Any]], 
                                  task: ProjectTask) -> Dict[str, Any]:
        """
        Perform comprehensive code review using OpenAI o3.
        
        Args:
            code_files: List of code files
            task: The project task
            
        Returns:
            Review results
        """
        review_results = {
            'overall_score': 0.0,
            'category_scores': {},
            'files_analysis': [],
            'issues_found': [],
            'strengths': [],
            'summary': ''
        }
        
        try:
            # Create comprehensive review prompt
            review_prompt = self._create_review_prompt(code_files, task)
            
            # Call OpenAI o3 with reasoning
            response = await self._call_openai_o3(review_prompt)
            
            # Parse the response
            parsed_review = self._parse_review_response(response)
            review_results.update(parsed_review)
            
            # Analyze individual files
            for code_file in code_files:
                file_analysis = await self._analyze_file(code_file)
                review_results['files_analysis'].append(file_analysis)
            
            # Calculate overall score
            review_results['overall_score'] = self._calculate_overall_score(review_results)
            
        except Exception as e:
            self.logger.error(f"Error during code review: {str(e)}")
            review_results['summary'] = f"Review failed: {str(e)}"
        
        return review_results
    
    def _create_review_prompt(self, code_files: List[Dict[str, Any]], 
                             task: ProjectTask) -> str:
        """
        Create a comprehensive review prompt for OpenAI o3.
        
        Args:
            code_files: List of code files
            task: The project task
            
        Returns:
            Review prompt string
        """
        files_summary = []
        for file_info in code_files:
            files_summary.append(f"- {file_info['relative_path']} ({file_info['language']}, {file_info['lines']} lines)")
        
        # Truncate file contents if too long
        code_content = ""
        for file_info in code_files[:5]:  # Limit to first 5 files
            code_content += f"\n\n--- {file_info['relative_path']} ---\n"
            content = file_info['content']
            if len(content) > 2000:  # Truncate long files
                content = content[:2000] + "\n... (truncated)"
            code_content += content
        
        return f"""
You are an expert code reviewer with deep knowledge of software engineering best practices, 
security, performance, and architecture. Please perform a comprehensive code review of the following project.

## Task Context
- Task: {task.title}
- Description: {task.description}
- Type: {task.type}

## Files to Review
{chr(10).join(files_summary)}

## Code Content
{code_content}

## Review Categories
Please analyze the code across these categories:

1. **Code Quality** (25%): Readability, maintainability, consistency, documentation, naming conventions
2. **Architecture** (20%): Design patterns, modularity, separation of concerns, scalability, extensibility
3. **Performance** (15%): Efficiency, resource usage, algorithmic complexity, optimization opportunities
4. **Security** (20%): Input validation, authentication, authorization, data protection, vulnerability prevention
5. **Testing** (10%): Test coverage, test quality, testability, edge cases, error handling
6. **Best Practices** (10%): Language conventions, framework usage, dependencies, error handling, logging

## Required Output Format
Please provide your analysis in the following JSON format:

```json
{{
  "overall_score": <float 0.0-1.0>,
  "category_scores": {{
    "code_quality": <float 0.0-1.0>,
    "architecture": <float 0.0-1.0>,
    "performance": <float 0.0-1.0>,
    "security": <float 0.0-1.0>,
    "testing": <float 0.0-1.0>,
    "best_practices": <float 0.0-1.0>
  }},
  "issues_found": [
    {{
      "category": "<category>",
      "severity": "<high|medium|low>",
      "file": "<file_path>",
      "line": <line_number>,
      "description": "<detailed_description>",
      "recommendation": "<how_to_fix>"
    }}
  ],
  "strengths": [
    "<strength_1>",
    "<strength_2>"
  ],
  "summary": "<comprehensive_summary>"
}}
```

Focus on actionable feedback and specific improvements. Use your advanced reasoning capabilities to provide deep insights into the code quality and suggest concrete improvements.
"""
    
    async def _call_openai_o3(self, prompt: str) -> str:
        """
        Call OpenAI o3 model with reasoning.
        
        Args:
            prompt: The review prompt
            
        Returns:
            Model response
        """
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert code reviewer with advanced reasoning capabilities."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort
                )
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI o3: {str(e)}")
            raise
    
    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the OpenAI o3 response.
        
        Args:
            response: Raw response from OpenAI
            
        Returns:
            Parsed review data
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                return parsed
            else:
                # Fallback parsing
                return {
                    'overall_score': 0.7,
                    'category_scores': {cat: 0.7 for cat in self.review_categories},
                    'issues_found': [],
                    'strengths': [],
                    'summary': response
                }
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {str(e)}")
            return {
                'overall_score': 0.5,
                'category_scores': {cat: 0.5 for cat in self.review_categories},
                'issues_found': [],
                'strengths': [],
                'summary': f"Failed to parse review response: {str(e)}"
            }
    
    async def _analyze_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze individual file for specific metrics.
        
        Args:
            file_info: File information
            
        Returns:
            File analysis results
        """
        analysis = {
            'file': file_info['relative_path'],
            'language': file_info['language'],
            'lines': file_info['lines'],
            'size': file_info['size'],
            'complexity_estimate': 0,
            'documentation_ratio': 0.0,
            'issues': []
        }
        
        content = file_info['content']
        lines = content.splitlines()
        
        # Basic complexity estimation
        complexity_keywords = ['if', 'for', 'while', 'switch', 'case', 'try', 'catch', 'except']
        analysis['complexity_estimate'] = sum(line.count(keyword) for keyword in complexity_keywords for line in lines)
        
        # Documentation ratio
        comment_lines = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//') or line.strip().startswith('/*'))
        doc_lines = sum(1 for line in lines if '"""' in line or "'''" in line or '/**' in line)
        analysis['documentation_ratio'] = (comment_lines + doc_lines) / len(lines) if lines else 0.0
        
        # Basic issue detection
        if analysis['complexity_estimate'] > 20:
            analysis['issues'].append("High complexity detected")
        
        if analysis['documentation_ratio'] < 0.1:
            analysis['issues'].append("Low documentation ratio")
        
        if analysis['lines'] > 500:
            analysis['issues'].append("File is very large")
        
        return analysis
    
    async def _generate_suggestions(self, review_results: Dict[str, Any], 
                                  task: ProjectTask) -> List[Dict[str, Any]]:
        """
        Generate improvement suggestions based on review results.
        
        Args:
            review_results: Code review results
            task: The project task
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Generate suggestions based on category scores
        for category, score in review_results.get('category_scores', {}).items():
            if score < 0.7:  # Below threshold
                suggestion = self._generate_category_suggestion(category, score)
                if suggestion:
                    suggestions.append(suggestion)
        
        # Generate suggestions from issues
        for issue in review_results.get('issues_found', []):
            suggestion = {
                'type': 'issue_fix',
                'category': issue.get('category', 'general'),
                'priority': issue.get('severity', 'medium'),
                'title': f"Fix {issue.get('category', 'issue')} in {issue.get('file', 'unknown')}",
                'description': issue.get('description', ''),
                'recommendation': issue.get('recommendation', ''),
                'file': issue.get('file'),
                'line': issue.get('line')
            }
            suggestions.append(suggestion)
        
        # Generate overall suggestions
        overall_score = review_results.get('overall_score', 0.5)
        if overall_score < 0.8:
            suggestions.append({
                'type': 'general_improvement',
                'category': 'overall',
                'priority': 'high',
                'title': 'Improve overall code quality',
                'description': f'Current overall score: {overall_score:.2f}',
                'recommendation': 'Focus on addressing the highest priority issues first'
            })
        
        # Sort suggestions by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    def _generate_category_suggestion(self, category: str, score: float) -> Optional[Dict[str, Any]]:
        """
        Generate suggestion for a specific category.
        
        Args:
            category: Review category
            score: Category score
            
        Returns:
            Suggestion dictionary
        """
        suggestions_map = {
            'code_quality': {
                'title': 'Improve code quality',
                'description': 'Focus on readability, maintainability, and consistency',
                'recommendation': 'Add docstrings, improve naming, and refactor complex code'
            },
            'architecture': {
                'title': 'Improve architecture',
                'description': 'Enhance design patterns and modularity',
                'recommendation': 'Apply SOLID principles and improve separation of concerns'
            },
            'performance': {
                'title': 'Optimize performance',
                'description': 'Improve efficiency and resource usage',
                'recommendation': 'Profile code, optimize algorithms, and implement caching'
            },
            'security': {
                'title': 'Enhance security',
                'description': 'Address security vulnerabilities and improve protection',
                'recommendation': 'Add input validation, improve authentication, and sanitize data'
            },
            'testing': {
                'title': 'Improve testing',
                'description': 'Increase test coverage and quality',
                'recommendation': 'Add unit tests, integration tests, and edge case testing'
            },
            'best_practices': {
                'title': 'Follow best practices',
                'description': 'Adhere to language and framework conventions',
                'recommendation': 'Follow style guides, use proper error handling, and improve logging'
            }
        }
        
        if category in suggestions_map:
            template = suggestions_map[category]
            return {
                'type': 'category_improvement',
                'category': category,
                'priority': 'high' if score < 0.5 else 'medium',
                'title': template['title'],
                'description': f"{template['description']} (current score: {score:.2f})",
                'recommendation': template['recommendation'],
                'current_score': score
            }
        
        return None
    
    def _calculate_overall_score(self, review_results: Dict[str, Any]) -> float:
        """
        Calculate overall score from category scores.
        
        Args:
            review_results: Review results
            
        Returns:
            Overall score
        """
        category_scores = review_results.get('category_scores', {})
        
        if not category_scores:
            return 0.5
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, weight_info in self.review_categories.items():
            if category in category_scores:
                score = category_scores[category]
                weight = weight_info['weight']
                weighted_score += score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _calculate_review_confidence(self, review_results: Dict[str, Any], 
                                   suggestions: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for the review.
        
        Args:
            review_results: Review results
            suggestions: Generated suggestions
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from successful analysis
        base_confidence = 0.7
        
        # Boost confidence if we have detailed results
        if review_results.get('category_scores'):
            base_confidence += 0.1
        
        if review_results.get('issues_found'):
            base_confidence += 0.1
        
        # Boost confidence if we generated useful suggestions
        if suggestions:
            base_confidence += 0.1
        
        # Adjust based on overall score
        overall_score = review_results.get('overall_score', 0.5)
        confidence_adjustment = (overall_score - 0.5) * 0.2
        
        final_confidence = base_confidence + confidence_adjustment
        
        return max(0.0, min(1.0, final_confidence))
    
    def _validate_task(self, task: ProjectTask) -> bool:
        """
        Validate that the task is appropriate for the advisor agent.
        
        Args:
            task: The task to validate
            
        Returns:
            True if task is valid
        """
        if not super()._validate_task(task):
            return False
        
        # Advisor-specific validation
        if task.agent_type != AgentType.ADVISOR:
            self.logger.warning(f"Task {task.id} is not for advisor agent: {task.agent_type}")
            return False
        
        return True