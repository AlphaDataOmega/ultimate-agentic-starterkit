"""
Documentation Agent for the Ultimate Agentic StarterKit.

This agent handles updating project documentation after successful work order completion.
It updates README.md, docs/ files, and other project documentation based on completed work.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from agents.base_agent import BaseAgent
from core.models import ProjectTask, AgentResult, AgentType
from core.logger import get_logger
from core.config import get_config
from core.knowledge_base import ProjectKnowledgeBase


class DocumentationAgent(BaseAgent):
    """
    Agent responsible for updating project documentation after work order completion.
    
    Handles:
    - README.md updates
    - API documentation updates
    - Architecture documentation updates
    - User guide updates
    - CHANGELOG.md updates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Documentation Agent."""
        super().__init__("documentation", config)
        self.system_config = get_config()
        self.knowledge_base = ProjectKnowledgeBase(".")
        
        # Documentation configuration
        self.supported_formats = ['.md', '.rst', '.txt']
        self.doc_directories = ['docs', 'documentation', '.']
        self.key_files = [
            'README.md',
            'CHANGELOG.md',
            'API.md',
            'ARCHITECTURE.md',
            'USER_GUIDE.md',
            'INSTALLATION.md'
        ]
        
        # Documentation templates
        self.templates = {
            'feature_addition': """
## {feature_name}

{description}

### Usage
```{language}
{usage_example}
```

### API Reference
{api_reference}
""",
            'changelog_entry': """
### {version} - {date}
#### Added
- {feature_description}

#### Changed
- {changes}

#### Technical Details
- {technical_details}
""",
            'api_documentation': """
## {endpoint_name}

**Method**: {method}
**URL**: {url}
**Description**: {description}

### Parameters
{parameters}

### Response
{response_format}

### Example
{example}
"""
        }
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute documentation update task.
        
        Args:
            task: Documentation task with completion results
            
        Returns:
            AgentResult with documentation update results
        """
        self.logger.info(f"Starting documentation update for task: {task.title}")
        
        try:
            # Validate task
            if not self._validate_task(task):
                return AgentResult(
                    success=False,
                    confidence=0.0,
                    output=None,
                    error="Invalid documentation task",
                    execution_time=0.0,
                    agent_id=self.agent_id,
                    timestamp=datetime.now()
                )
            
            # Parse task description to get completion results
            completion_data = self._parse_completion_data(task.description)
            
            # Determine what documentation needs updating
            update_plan = await self._analyze_documentation_needs(completion_data)
            
            # Execute documentation updates
            update_results = await self._execute_documentation_updates(update_plan, completion_data)
            
            # Calculate confidence based on success rate
            confidence = self._calculate_confidence({
                'updates_successful': update_results['successful_updates'],
                'updates_total': update_results['total_updates'],
                'files_modified': len(update_results['modified_files']),
                'validation_passed': update_results['validation_passed']
            })
            
            execution_time = update_results.get('execution_time', 0.0)
            
            if update_results['success']:
                self.logger.info(f"Documentation update completed successfully")
                return AgentResult(
                    success=True,
                    confidence=confidence,
                    output=update_results,
                    error=None,
                    execution_time=execution_time,
                    agent_id=self.agent_id,
                    timestamp=datetime.now()
                )
            else:
                self.logger.error(f"Documentation update failed: {update_results.get('error')}")
                return AgentResult(
                    success=False,
                    confidence=confidence,
                    output=update_results,
                    error=update_results.get('error', 'Unknown error'),
                    execution_time=execution_time,
                    agent_id=self.agent_id,
                    timestamp=datetime.now()
                )
        
        except Exception as e:
            self.logger.exception(f"Documentation agent execution failed: {str(e)}")
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error=str(e),
                execution_time=0.0,
                agent_id=self.agent_id,
                timestamp=datetime.now()
            )
    
    def _parse_completion_data(self, task_description: str) -> Dict[str, Any]:
        """Parse completion data from task description."""
        completion_data = {
            'work_order_id': 'unknown',
            'feature_name': 'New Feature',
            'description': 'Feature implementation completed',
            'files_created': [],
            'files_modified': [],
            'api_endpoints': [],
            'technical_details': [],
            'usage_examples': [],
            'version': '1.0.0'
        }
        
        try:
            # Try to parse as JSON first
            if task_description.strip().startswith('{'):
                parsed = json.loads(task_description)
                completion_data.update(parsed)
            else:
                # Parse from structured text
                lines = task_description.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('Work Order:'):
                        completion_data['work_order_id'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Feature:'):
                        completion_data['feature_name'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Description:'):
                        completion_data['description'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Files Created:'):
                        completion_data['files_created'] = [f.strip() for f in line.split(':', 1)[1].split(',')]
                    elif line.startswith('Files Modified:'):
                        completion_data['files_modified'] = [f.strip() for f in line.split(':', 1)[1].split(',')]
                    elif line.startswith('API Endpoints:'):
                        completion_data['api_endpoints'] = [e.strip() for e in line.split(':', 1)[1].split(',')]
        
        except Exception as e:
            self.logger.warning(f"Failed to parse completion data: {e}")
        
        return completion_data
    
    async def _analyze_documentation_needs(self, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what documentation needs to be updated."""
        update_plan = {
            'readme_update': False,
            'changelog_update': False,
            'api_docs_update': False,
            'architecture_update': False,
            'user_guide_update': False,
            'new_doc_files': [],
            'priority': 'medium'
        }
        
        # Check if README needs updating
        if completion_data.get('feature_name') and completion_data.get('description'):
            update_plan['readme_update'] = True
        
        # Check if CHANGELOG needs updating
        if completion_data.get('files_created') or completion_data.get('files_modified'):
            update_plan['changelog_update'] = True
        
        # Check if API documentation needs updating
        if completion_data.get('api_endpoints'):
            update_plan['api_docs_update'] = True
        
        # Check if architecture docs need updating
        if completion_data.get('technical_details'):
            update_plan['architecture_update'] = True
        
        # Check if user guide needs updating
        if completion_data.get('usage_examples'):
            update_plan['user_guide_update'] = True
        
        # Determine priority based on scope
        total_updates = sum([
            update_plan['readme_update'],
            update_plan['changelog_update'],
            update_plan['api_docs_update'],
            update_plan['architecture_update'],
            update_plan['user_guide_update']
        ])
        
        if total_updates >= 3:
            update_plan['priority'] = 'high'
        elif total_updates >= 1:
            update_plan['priority'] = 'medium'
        else:
            update_plan['priority'] = 'low'
        
        self.logger.info(f"Documentation update plan: {update_plan}")
        return update_plan
    
    async def _execute_documentation_updates(self, update_plan: Dict[str, Any], 
                                           completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned documentation updates."""
        results = {
            'success': True,
            'total_updates': 0,
            'successful_updates': 0,
            'modified_files': [],
            'validation_passed': True,
            'execution_time': 0.0,
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            # Update README.md
            if update_plan['readme_update']:
                results['total_updates'] += 1
                if await self._update_readme(completion_data):
                    results['successful_updates'] += 1
                    results['modified_files'].append('README.md')
                else:
                    results['errors'].append('Failed to update README.md')
            
            # Update CHANGELOG.md
            if update_plan['changelog_update']:
                results['total_updates'] += 1
                if await self._update_changelog(completion_data):
                    results['successful_updates'] += 1
                    results['modified_files'].append('CHANGELOG.md')
                else:
                    results['errors'].append('Failed to update CHANGELOG.md')
            
            # Update API documentation
            if update_plan['api_docs_update']:
                results['total_updates'] += 1
                if await self._update_api_docs(completion_data):
                    results['successful_updates'] += 1
                    results['modified_files'].append('API.md')
                else:
                    results['errors'].append('Failed to update API.md')
            
            # Update architecture documentation
            if update_plan['architecture_update']:
                results['total_updates'] += 1
                if await self._update_architecture_docs(completion_data):
                    results['successful_updates'] += 1
                    results['modified_files'].append('ARCHITECTURE.md')
                else:
                    results['errors'].append('Failed to update ARCHITECTURE.md')
            
            # Update user guide
            if update_plan['user_guide_update']:
                results['total_updates'] += 1
                if await self._update_user_guide(completion_data):
                    results['successful_updates'] += 1
                    results['modified_files'].append('USER_GUIDE.md')
                else:
                    results['errors'].append('Failed to update USER_GUIDE.md')
            
            # Calculate success rate
            if results['total_updates'] > 0:
                success_rate = results['successful_updates'] / results['total_updates']
                results['success'] = success_rate >= 0.7  # 70% success threshold
            
            results['execution_time'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            results['execution_time'] = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Documentation update execution failed: {e}")
        
        return results
    
    async def _update_readme(self, completion_data: Dict[str, Any]) -> bool:
        """Update README.md with new feature information."""
        try:
            readme_path = Path('README.md')
            
            # Read existing README or create new one
            if readme_path.exists():
                content = readme_path.read_text()
            else:
                content = f"# {completion_data.get('project_name', 'Project')}\n\n"
            
            # Add feature section
            feature_section = self.templates['feature_addition'].format(
                feature_name=completion_data.get('feature_name', 'New Feature'),
                description=completion_data.get('description', 'Feature implementation'),
                language=completion_data.get('language', 'python'),
                usage_example=completion_data.get('usage_examples', ['# Usage example here'])[0] if completion_data.get('usage_examples') else '# Usage example here',
                api_reference=self._format_api_reference(completion_data.get('api_endpoints', []))
            )
            
            # Insert feature section (look for Features section or add at end)
            if '## Features' in content:
                # Insert after Features header
                parts = content.split('## Features')
                if len(parts) > 1:
                    parts[1] = feature_section + parts[1]
                    content = '## Features'.join(parts)
            else:
                # Add Features section
                content += f"\n## Features\n{feature_section}\n"
            
            # Write updated README
            readme_path.write_text(content)
            self.logger.info("Updated README.md successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update README.md: {e}")
            return False
    
    async def _update_changelog(self, completion_data: Dict[str, Any]) -> bool:
        """Update CHANGELOG.md with new changes."""
        try:
            changelog_path = Path('CHANGELOG.md')
            
            # Read existing changelog or create new one
            if changelog_path.exists():
                content = changelog_path.read_text()
            else:
                content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
            
            # Create changelog entry
            changelog_entry = self.templates['changelog_entry'].format(
                version=completion_data.get('version', '1.0.0'),
                date=datetime.now().strftime('%Y-%m-%d'),
                feature_description=completion_data.get('description', 'New feature added'),
                changes='\n- '.join(completion_data.get('files_modified', [])),
                technical_details='\n- '.join(completion_data.get('technical_details', []))
            )
            
            # Insert at the top after header
            lines = content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('##') or line.startswith('###'):
                    header_end = i
                    break
            
            if header_end > 0:
                lines.insert(header_end, changelog_entry)
            else:
                lines.append(changelog_entry)
            
            content = '\n'.join(lines)
            
            # Write updated changelog
            changelog_path.write_text(content)
            self.logger.info("Updated CHANGELOG.md successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update CHANGELOG.md: {e}")
            return False
    
    async def _update_api_docs(self, completion_data: Dict[str, Any]) -> bool:
        """Update API documentation."""
        try:
            api_docs_path = Path('API.md')
            
            # Read existing API docs or create new one
            if api_docs_path.exists():
                content = api_docs_path.read_text()
            else:
                content = "# API Documentation\n\n"
            
            # Add API endpoint documentation
            for endpoint in completion_data.get('api_endpoints', []):
                api_doc = self.templates['api_documentation'].format(
                    endpoint_name=endpoint.get('name', 'Endpoint'),
                    method=endpoint.get('method', 'GET'),
                    url=endpoint.get('url', '/api/endpoint'),
                    description=endpoint.get('description', 'API endpoint'),
                    parameters=endpoint.get('parameters', 'None'),
                    response_format=endpoint.get('response', 'JSON response'),
                    example=endpoint.get('example', 'Example usage')
                )
                content += api_doc + '\n'
            
            # Write updated API docs
            api_docs_path.write_text(content)
            self.logger.info("Updated API.md successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update API.md: {e}")
            return False
    
    async def _update_architecture_docs(self, completion_data: Dict[str, Any]) -> bool:
        """Update architecture documentation."""
        try:
            arch_docs_path = Path('ARCHITECTURE.md')
            
            # Read existing architecture docs or create new one
            if arch_docs_path.exists():
                content = arch_docs_path.read_text()
            else:
                content = "# Architecture Documentation\n\n"
            
            # Add technical details section
            if completion_data.get('technical_details'):
                tech_section = f"\n## {completion_data.get('feature_name', 'New Component')}\n\n"
                tech_section += '\n'.join(f"- {detail}" for detail in completion_data['technical_details'])
                content += tech_section + '\n'
            
            # Write updated architecture docs
            arch_docs_path.write_text(content)
            self.logger.info("Updated ARCHITECTURE.md successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update ARCHITECTURE.md: {e}")
            return False
    
    async def _update_user_guide(self, completion_data: Dict[str, Any]) -> bool:
        """Update user guide documentation."""
        try:
            guide_path = Path('USER_GUIDE.md')
            
            # Read existing user guide or create new one
            if guide_path.exists():
                content = guide_path.read_text()
            else:
                content = "# User Guide\n\n"
            
            # Add usage examples
            if completion_data.get('usage_examples'):
                usage_section = f"\n## Using {completion_data.get('feature_name', 'New Feature')}\n\n"
                for example in completion_data['usage_examples']:
                    usage_section += f"```\n{example}\n```\n\n"
                content += usage_section
            
            # Write updated user guide
            guide_path.write_text(content)
            self.logger.info("Updated USER_GUIDE.md successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update USER_GUIDE.md: {e}")
            return False
    
    def _format_api_reference(self, api_endpoints: List[Dict[str, Any]]) -> str:
        """Format API endpoints for documentation."""
        if not api_endpoints:
            return "No API endpoints"
        
        formatted = []
        for endpoint in api_endpoints:
            formatted.append(f"- **{endpoint.get('method', 'GET')}** `{endpoint.get('url', '')}`")
        
        return '\n'.join(formatted)
    
    def _calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """Calculate confidence score for documentation updates."""
        base_confidence = 0.5
        
        # Success rate factor
        if indicators.get('updates_total', 0) > 0:
            success_rate = indicators['updates_successful'] / indicators['updates_total']
            base_confidence += success_rate * 0.3
        
        # Files modified factor
        files_modified = indicators.get('files_modified', 0)
        if files_modified > 0:
            base_confidence += min(files_modified * 0.1, 0.2)
        
        # Validation factor
        if indicators.get('validation_passed', False):
            base_confidence += 0.1
        
        return max(0.0, min(1.0, base_confidence))