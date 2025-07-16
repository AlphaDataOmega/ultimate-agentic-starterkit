"""
OpenAI o3 Orchestrator for the Ultimate Agentic StarterKit.

This module implements the orchestration layer using OpenAI o3 for intelligent planning
and task decomposition with sophisticated reasoning capabilities.
"""

import openai
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

from .models import ProjectSpecification, ProjectTask, TaskStatus, AgentType
from .config import get_config
from .logger import get_logger
from .voice_alerts import get_voice_alerts


class O3Orchestrator:
    """
    OpenAI o3-based orchestration and planning system.
    
    This class provides intelligent task decomposition, dependency analysis,
    and adaptive planning using OpenAI's o3 reasoning model.
    """
    
    def __init__(self):
        """Initialize the O3 orchestrator."""
        self.config = get_config()
        self.logger = get_logger("o3_orchestrator")
        self.voice = get_voice_alerts()
        
        # Initialize OpenAI client
        if not self.config.api_keys.has_openai():
            raise ValueError("OpenAI API key required for o3 orchestration")
        
        self.client = openai.OpenAI(api_key=self.config.api_keys.openai_api_key)
        
        # O3 configuration
        self.model = "o3-mini"  # Cost-effective option
        self.reasoning_effort = "medium"  # Balance cost and quality
        self.max_retries = 3
        self.request_timeout = 120  # 2 minutes
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0  # Estimated cost tracking
        
        self.logger.info("O3 Orchestrator initialized successfully")
    
    async def create_execution_plan(self, project_spec: ProjectSpecification) -> Dict[str, Any]:
        """
        Create optimized execution plan from project specification.
        
        Args:
            project_spec: The project specification to plan for
            
        Returns:
            Dict containing the execution plan with task ordering, dependencies,
            parallel execution opportunities, and risk assessment
        """
        self.logger.info(f"Creating execution plan for project: {project_spec.title}")
        
        try:
            # Generate planning prompt
            planning_prompt = self._create_planning_prompt(project_spec)
            
            # Call OpenAI o3 for planning
            with self.logger.performance_context("o3_planning"):
                response = await self._make_o3_request(
                    system_prompt="You are an AI orchestrator specialized in creating optimal task execution plans. Analyze dependencies, resource allocation, and risk factors to create efficient execution strategies.",
                    user_prompt=planning_prompt,
                    max_tokens=3000
                )
            
            # Parse and validate the response
            try:
                execution_plan = json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse o3 response as JSON: {e}")
                return self._create_fallback_plan(project_spec)
            
            # Validate and enhance the plan
            validated_plan = await self._validate_execution_plan(execution_plan, project_spec)
            
            # Add metadata
            validated_plan["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "project_type": project_spec.project_type,
                "total_tasks": len(project_spec.tasks)
            }
            
            self.voice.speak_success("Execution plan ready")
            self.logger.info(f"Created execution plan with {len(validated_plan['execution_order'])} tasks")
            
            return validated_plan
            
        except Exception as e:
            self.logger.error(f"Execution plan creation failed: {e}")
            self.voice.speak_error("Planning failed, using fallback")
            return self._create_fallback_plan(project_spec)
    
    def _create_planning_prompt(self, project_spec: ProjectSpecification) -> str:
        """Create detailed planning prompt for o3."""
        # Format tasks for the prompt
        tasks_info = []
        for task in project_spec.tasks:
            task_info = {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "type": task.type,
                "agent_type": task.agent_type.value,
                "dependencies": task.dependencies,
                "max_attempts": task.max_attempts
            }
            tasks_info.append(task_info)
        
        return f"""
        Analyze this project specification and create an optimal execution plan:
        
        Project Details:
        - Title: {project_spec.title}
        - Description: {project_spec.description}
        - Type: {project_spec.project_type}
        - Total Tasks: {len(project_spec.tasks)}
        
        Available Tasks:
        {json.dumps(tasks_info, indent=2)}
        
        Project Requirements:
        {json.dumps(project_spec.requirements, indent=2)}
        
        Validation Criteria:
        {json.dumps(project_spec.validation_criteria, indent=2)}
        
        Create a comprehensive JSON execution plan with:
        1. Optimized task ordering considering dependencies
        2. Parallel execution opportunities (identify tasks that can run simultaneously)
        3. Risk assessment for each task (based on complexity, dependencies, agent type)
        4. Resource allocation recommendations
        5. Confidence thresholds for each task
        6. Fallback strategies for high-risk tasks
        7. Critical path analysis
        
        Response must be valid JSON with this exact structure:
        {{
            "execution_order": [
                {{
                    "task_id": "string",
                    "priority": "high|medium|low",
                    "dependencies": ["task_id1", "task_id2"],
                    "parallel_group": "group_name|null",
                    "risk_level": "high|medium|low", 
                    "confidence_threshold": 0.8,
                    "estimated_duration": 300,
                    "resource_requirements": {{"cpu": "medium", "memory": "low"}},
                    "fallback_strategy": "string"
                }}
            ],
            "parallel_groups": {{
                "group_name": ["task_id1", "task_id2"]
            }},
            "critical_path": ["task_id1", "task_id2"],
            "risk_mitigation": {{
                "high_risk_tasks": ["task_id"],
                "mitigation_strategies": ["strategy1", "strategy2"]
            }},
            "estimated_total_duration": 1800,
            "optimization_notes": "Brief explanation of the optimization strategy"
        }}
        """
    
    async def _make_o3_request(self, system_prompt: str, user_prompt: str, 
                              max_tokens: int = 2000) -> str:
        """
        Make a request to OpenAI o3 with proper error handling and retries.
        
        Args:
            system_prompt: System instruction for the model
            user_prompt: User prompt with the task
            max_tokens: Maximum tokens to generate
            
        Returns:
            The response content from o3
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = await asyncio.wait_for(
                    self.client.chat.completions.acreate(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        reasoning_effort=self.reasoning_effort,
                        max_completion_tokens=max_tokens,
                        response_format={"type": "json_object"},
                        temperature=0.1  # Low temperature for consistent planning
                    ),
                    timeout=self.request_timeout
                )
                
                # Update tracking
                self.request_count += 1
                duration = time.time() - start_time
                
                # Log API call
                self.logger.log_api_call(
                    provider="openai",
                    model=self.model,
                    tokens_used=response.usage.total_tokens,
                    duration=duration,
                    success=True
                )
                
                return response.choices[0].message.content
                
            except asyncio.TimeoutError:
                self.logger.warning(f"O3 request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                self.logger.error(f"O3 request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded for o3 request")
    
    async def _validate_execution_plan(self, plan: Dict[str, Any], 
                                     project_spec: ProjectSpecification) -> Dict[str, Any]:
        """
        Validate and enhance execution plan.
        
        Args:
            plan: The execution plan to validate
            project_spec: The project specification
            
        Returns:
            Validated and enhanced execution plan
        """
        self.logger.info("Validating execution plan")
        
        # Ensure required fields exist
        if "execution_order" not in plan:
            self.logger.error("Missing execution_order in plan")
            return self._create_fallback_plan(project_spec)
        
        # Validate task references
        task_ids = {task.id for task in project_spec.tasks}
        valid_tasks = []
        
        for task_order in plan["execution_order"]:
            if "task_id" not in task_order:
                self.logger.warning("Missing task_id in execution order entry")
                continue
                
            task_id = task_order["task_id"]
            if task_id not in task_ids:
                self.logger.error(f"Invalid task reference: {task_id}")
                continue
                
            # Ensure required fields with defaults
            task_order.setdefault("priority", "medium")
            task_order.setdefault("dependencies", [])
            task_order.setdefault("parallel_group", None)
            task_order.setdefault("risk_level", "medium")
            task_order.setdefault("confidence_threshold", 0.8)
            task_order.setdefault("estimated_duration", 300)
            task_order.setdefault("resource_requirements", {"cpu": "medium", "memory": "low"})
            task_order.setdefault("fallback_strategy", "retry with increased timeout")
            
            valid_tasks.append(task_order)
        
        plan["execution_order"] = valid_tasks
        
        # Check for circular dependencies
        if self._has_circular_dependencies(plan["execution_order"]):
            self.logger.warning("Circular dependencies detected, resolving...")
            plan = self._resolve_circular_dependencies(plan)
        
        # Ensure all required fields exist with defaults
        plan.setdefault("parallel_groups", {})
        plan.setdefault("critical_path", [])
        plan.setdefault("risk_mitigation", {
            "high_risk_tasks": [],
            "mitigation_strategies": []
        })
        plan.setdefault("estimated_total_duration", 1800)
        plan.setdefault("optimization_notes", "Standard dependency-based ordering")
        
        # Add timing estimates
        plan["estimated_total_duration"] = self._calculate_execution_time(plan)
        
        self.logger.info("Execution plan validated successfully")
        return plan
    
    def _has_circular_dependencies(self, execution_order: List[Dict]) -> bool:
        """Check for circular dependencies in execution order."""
        # Build dependency graph
        graph = {}
        for task in execution_order:
            graph[task["task_id"]] = task.get("dependencies", [])
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor in graph and has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def _resolve_circular_dependencies(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve circular dependencies by removing problematic edges."""
        execution_order = plan["execution_order"]
        
        # Build dependency graph
        graph = {}
        for task in execution_order:
            graph[task["task_id"]] = task.get("dependencies", [])
        
        # Find and remove circular dependencies using DFS
        visited = set()
        rec_stack = set()
        removed_edges = []
        
        def remove_cycle(node, path):
            if node in rec_stack:
                # Found cycle, remove the edge that creates it
                cycle_start = path.index(node)
                cycle_path = path[cycle_start:]
                
                # Remove the last edge in the cycle
                if len(cycle_path) > 1:
                    source = cycle_path[-2]
                    target = cycle_path[-1]
                    if target in graph.get(source, []):
                        graph[source].remove(target)
                        removed_edges.append((source, target))
                
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor in graph and remove_cycle(neighbor, path):
                    path.pop()
                    rec_stack.remove(node)
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        # Remove cycles
        for node in list(graph.keys()):
            if node not in visited:
                remove_cycle(node, [])
        
        # Update execution order with resolved dependencies
        for task in execution_order:
            task["dependencies"] = graph.get(task["task_id"], [])
        
        if removed_edges:
            self.logger.info(f"Resolved circular dependencies by removing edges: {removed_edges}")
        
        return plan
    
    def _calculate_execution_time(self, plan: Dict[str, Any]) -> int:
        """Calculate estimated total execution time."""
        execution_order = plan["execution_order"]
        parallel_groups = plan.get("parallel_groups", {})
        
        # Build dependency graph and timing information
        task_times = {}
        dependencies = {}
        
        for task in execution_order:
            task_id = task["task_id"]
            task_times[task_id] = task.get("estimated_duration", 300)
            dependencies[task_id] = task.get("dependencies", [])
        
        # Calculate critical path timing
        def calculate_earliest_start(task_id, memo=None):
            if memo is None:
                memo = {}
            
            if task_id in memo:
                return memo[task_id]
            
            deps = dependencies.get(task_id, [])
            if not deps:
                memo[task_id] = 0
                return 0
            
            max_finish_time = 0
            for dep in deps:
                if dep in task_times:
                    dep_start = calculate_earliest_start(dep, memo)
                    dep_finish = dep_start + task_times[dep]
                    max_finish_time = max(max_finish_time, dep_finish)
            
            memo[task_id] = max_finish_time
            return max_finish_time
        
        # Calculate total time
        total_time = 0
        for task_id in task_times:
            start_time = calculate_earliest_start(task_id)
            finish_time = start_time + task_times[task_id]
            total_time = max(total_time, finish_time)
        
        return total_time
    
    def _create_fallback_plan(self, project_spec: ProjectSpecification) -> Dict[str, Any]:
        """Create a simple fallback plan when o3 planning fails."""
        self.logger.info("Creating fallback execution plan")
        
        execution_order = []
        for task in project_spec.tasks:
            execution_order.append({
                "task_id": task.id,
                "priority": "medium",
                "dependencies": task.dependencies,
                "parallel_group": None,
                "risk_level": "medium",
                "confidence_threshold": 0.8,
                "estimated_duration": 300,
                "resource_requirements": {"cpu": "medium", "memory": "low"},
                "fallback_strategy": "retry with increased timeout"
            })
        
        return {
            "execution_order": execution_order,
            "parallel_groups": {},
            "critical_path": [task.id for task in project_spec.tasks],
            "risk_mitigation": {
                "high_risk_tasks": [],
                "mitigation_strategies": ["Use fallback plan", "Increase retry attempts"]
            },
            "estimated_total_duration": len(project_spec.tasks) * 300,
            "optimization_notes": "Fallback plan - simple sequential execution",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "model": "fallback",
                "reasoning_effort": "none",
                "project_type": project_spec.project_type,
                "total_tasks": len(project_spec.tasks)
            }
        }
    
    async def adapt_plan_during_execution(self, current_plan: Dict[str, Any],
                                        failed_tasks: List[str],
                                        agent_results: List[Dict]) -> Dict[str, Any]:
        """
        Adapt execution plan based on runtime results.
        
        Args:
            current_plan: Current execution plan
            failed_tasks: List of task IDs that failed
            agent_results: List of recent agent execution results
            
        Returns:
            Adapted execution plan
        """
        self.logger.info(f"Adapting plan for {len(failed_tasks)} failed tasks")
        
        try:
            # Create adaptation prompt
            adaptation_prompt = self._create_adaptation_prompt(
                current_plan, failed_tasks, agent_results
            )
            
            # Request adaptation from o3
            with self.logger.performance_context("o3_adaptation"):
                response = await self._make_o3_request(
                    system_prompt="You are adapting an execution plan based on runtime feedback. Maintain project objectives while optimizing for success probability given the current situation.",
                    user_prompt=adaptation_prompt,
                    max_tokens=2000
                )
            
            # Parse adapted plan
            try:
                adapted_plan = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse adaptation response")
                return current_plan
            
            # Validate adapted plan
            adapted_plan = await self._validate_adaptation(adapted_plan, current_plan)
            
            # Add adaptation metadata
            adapted_plan["metadata"] = {
                **current_plan.get("metadata", {}),
                "adapted_at": datetime.now().isoformat(),
                "adaptation_reason": f"Failed tasks: {failed_tasks}",
                "adaptation_count": current_plan.get("metadata", {}).get("adaptation_count", 0) + 1
            }
            
            self.logger.info("Execution plan adapted successfully")
            return adapted_plan
            
        except Exception as e:
            self.logger.error(f"Plan adaptation failed: {e}")
            return current_plan
    
    def _create_adaptation_prompt(self, current_plan: Dict[str, Any],
                                failed_tasks: List[str], 
                                agent_results: List[Dict]) -> str:
        """Create prompt for plan adaptation."""
        # Get recent results summary
        recent_results = agent_results[-5:] if len(agent_results) > 5 else agent_results
        
        return f"""
        Current execution plan needs adaptation due to runtime conditions:
        
        Failed Tasks: {failed_tasks}
        
        Recent Agent Results (last 5):
        {json.dumps(recent_results, indent=2)}
        
        Current Plan Summary:
        - Total tasks: {len(current_plan.get('execution_order', []))}
        - Parallel groups: {list(current_plan.get('parallel_groups', {}).keys())}
        - Critical path length: {len(current_plan.get('critical_path', []))}
        - High risk tasks: {current_plan.get('risk_mitigation', {}).get('high_risk_tasks', [])}
        
        Current Execution Order:
        {json.dumps(current_plan.get('execution_order', []), indent=2)}
        
        Provide an adapted plan that:
        1. Handles failed tasks with alternative approaches or removes them if non-critical
        2. Adjusts dependencies based on actual results and failures
        3. Reorders tasks to optimize success probability
        4. Maintains overall project objectives
        5. Updates risk assessments based on actual execution data
        6. Considers resource constraints and performance patterns
        
        Response must be valid JSON with the same structure as the original plan.
        """
    
    async def _validate_adaptation(self, adapted_plan: Dict[str, Any],
                                 current_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate adapted plan against current plan."""
        # Ensure adapted plan has required structure
        if "execution_order" not in adapted_plan:
            self.logger.error("Adapted plan missing execution_order")
            return current_plan
        
        # Validate that critical tasks are still present
        current_tasks = {task["task_id"] for task in current_plan.get("execution_order", [])}
        adapted_tasks = {task["task_id"] for task in adapted_plan.get("execution_order", [])}
        
        if not adapted_tasks.issubset(current_tasks):
            self.logger.warning("Adapted plan contains unknown tasks, reverting to current plan")
            return current_plan
        
        # Ensure no new circular dependencies
        if self._has_circular_dependencies(adapted_plan["execution_order"]):
            self.logger.warning("Adapted plan has circular dependencies, resolving...")
            adapted_plan = self._resolve_circular_dependencies(adapted_plan)
        
        return adapted_plan
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.total_cost,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_retries": self.max_retries
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.logger.info("Orchestrator statistics reset")