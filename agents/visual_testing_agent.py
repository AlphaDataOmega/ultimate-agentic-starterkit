"""
Visual Testing Agent for the Ultimate Agentic StarterKit.

This agent handles visual validation for any browser-based projects through dual AI vision calls:
1. Pure visual analysis of screenshots
2. Code analysis + expected view comparison

Supports any frontend framework (React, Vue, Angular, vanilla HTML/CSS/JS, etc.)
"""

import json
import asyncio
import base64
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

from agents.base_agent import BaseAgent
from core.models import ProjectTask, AgentResult, AgentType
from core.logger import get_logger
from core.config import get_config


class VisualTestingAgent(BaseAgent):
    """
    Agent responsible for visual validation of browser-based applications.
    
    Handles:
    - Screenshot capture from running applications
    - Dual AI vision analysis (pure visual + code analysis)
    - Expected vs actual appearance comparison
    - Error boundary detection and reading
    - Framework-agnostic visual validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Visual Testing Agent."""
        super().__init__("visual_testing", config)
        self.system_config = get_config()
        
        # Browser automation configuration
        self.supported_browsers = ['chrome', 'firefox', 'safari', 'edge']
        self.default_browser = 'chrome'
        self.screenshot_timeout = 30  # seconds
        self.page_load_timeout = 10   # seconds
        
        # Visual analysis configuration
        self.screenshot_dir = Path("screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        
        # Framework detection patterns
        self.framework_patterns = {
            'react': ['react', 'jsx', 'tsx', 'react-dom'],
            'vue': ['vue', 'vue-loader', '@vue/'],
            'angular': ['angular', '@angular/', 'ng-'],
            'svelte': ['svelte', 'svelte-loader'],
            'vanilla': ['html', 'css', 'javascript']
        }
        
        # Error boundary patterns
        self.error_boundary_patterns = [
            'Error Boundary',
            'Something went wrong',
            'Application Error',
            'Uncaught Error',
            'React Error',
            'Vue Error',
            'Angular Error'
        ]
        
        # Vision analysis prompts
        self.vision_prompts = {
            'pure_visual': """
            Analyze this screenshot of a web application. Describe what you see in detail:
            1. Layout and structure
            2. UI components and their appearance
            3. Text content and typography
            4. Colors and styling
            5. Any visible errors or broken elements
            6. Overall visual quality and completeness
            
            Focus only on what is visually present in the screenshot.
            """,
            
            'code_comparison': """
            Compare this screenshot with the expected appearance description provided.
            Analyze:
            1. Does the actual appearance match the expected description?
            2. Are all expected elements present and positioned correctly?
            3. Are there any missing or extra elements?
            4. Do colors, fonts, and styling match expectations?
            5. Are there any visual errors or inconsistencies?
            6. Rate the match quality from 1-10
            
            Expected appearance: {expected_appearance}
            
            Provide a detailed comparison analysis.
            """
        }
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute visual testing task.
        
        Args:
            task: Visual testing task with work order results and expected appearance
            
        Returns:
            AgentResult with visual validation results
        """
        self.logger.info(f"Starting visual testing for task: {task.title}")
        
        try:
            # Validate task
            if not self._validate_task(task):
                return AgentResult(
                    success=False,
                    confidence=0.0,
                    output=None,
                    error="Invalid visual testing task",
                    execution_time=0.0,
                    agent_id=self.agent_id,
                    timestamp=datetime.now()
                )
            
            # Parse work order data from task
            work_order_data = self._parse_work_order_data(task.description)
            
            # Check if visual testing is required
            if not self._requires_visual_testing(work_order_data):
                return await self._skip_visual_testing(work_order_data)
            
            # Run visual testing validation
            testing_results = await self._run_visual_testing(work_order_data)
            
            # Calculate confidence based on visual analysis
            confidence = self._calculate_confidence({
                'visual_match_score': testing_results.get('visual_match_score', 0.0),
                'error_boundaries_detected': testing_results.get('error_boundaries_detected', 0),
                'screenshot_quality': testing_results.get('screenshot_quality', 0.0),
                'elements_matched': testing_results.get('elements_matched', 0),
                'elements_expected': testing_results.get('elements_expected', 1)
            })
            
            execution_time = testing_results.get('execution_time', 0.0)
            
            if testing_results['success']:
                self.logger.info("Visual testing completed successfully")
                return AgentResult(
                    success=True,
                    confidence=confidence,
                    output=testing_results,
                    error=None,
                    execution_time=execution_time,
                    agent_id=self.agent_id,
                    timestamp=datetime.now()
                )
            else:
                self.logger.error(f"Visual testing failed: {testing_results.get('error')}")
                return AgentResult(
                    success=False,
                    confidence=confidence,
                    output=testing_results,
                    error=testing_results.get('error', 'Unknown visual testing error'),
                    execution_time=execution_time,
                    agent_id=self.agent_id,
                    timestamp=datetime.now()
                )
        
        except Exception as e:
            self.logger.exception(f"Visual testing agent execution failed: {str(e)}")
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
            'expected_browser_appearance': None,
            'technology_stack': [],
            'application_url': 'http://localhost:3000',
            'application_port': 3000,
            'start_command': None,
            'framework': 'unknown'
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
                    elif line.startswith('Technology Stack:'):
                        work_order_data['technology_stack'] = [t.strip() for t in line.split(':', 1)[1].split(',') if t.strip()]
                    elif line.startswith('Application URL:'):
                        work_order_data['application_url'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Start Command:'):
                        work_order_data['start_command'] = line.split(':', 1)[1].strip()
                
                # Extract expected browser appearance from XML
                expected_appearance = self._extract_expected_appearance(task_description)
                if expected_appearance:
                    work_order_data['expected_browser_appearance'] = expected_appearance
        
        except Exception as e:
            self.logger.warning(f"Failed to parse work order data: {e}")
        
        return work_order_data
    
    def _extract_expected_appearance(self, task_description: str) -> Optional[Dict[str, Any]]:
        """Extract expected browser appearance from XML in task description."""
        try:
            # Look for XML block in task description
            import re
            xml_match = re.search(r'<expected_appearance>(.*?)</expected_appearance>', task_description, re.DOTALL)
            if xml_match:
                xml_content = xml_match.group(1).strip()
                # Parse XML
                root = ET.fromstring(f'<expected_appearance>{xml_content}</expected_appearance>')
                
                expected = {
                    'description': '',
                    'elements': [],
                    'interactions': [],
                    'error_scenarios': []
                }
                
                # Extract description
                desc_elem = root.find('description')
                if desc_elem is not None:
                    expected['description'] = desc_elem.text or ''
                
                # Extract elements
                elements = root.findall('elements/element')
                for elem in elements:
                    expected['elements'].append(elem.text or '')
                
                # Extract interactions
                interactions = root.findall('interactions/interaction')
                for interaction in interactions:
                    expected['interactions'].append(interaction.text or '')
                
                # Extract error scenarios
                error_scenarios = root.findall('error_scenarios/error_scenario')
                for scenario in error_scenarios:
                    expected['error_scenarios'].append(scenario.text or '')
                
                return expected
        
        except Exception as e:
            self.logger.warning(f"Failed to extract expected appearance: {e}")
        
        return None
    
    def _requires_visual_testing(self, work_order_data: Dict[str, Any]) -> bool:
        """Check if visual testing is required."""
        # Check if expected browser appearance is provided
        if work_order_data.get('expected_browser_appearance'):
            return True
        
        # Check technology stack for frontend frameworks
        tech_stack = work_order_data.get('technology_stack', [])
        frontend_indicators = ['react', 'vue', 'angular', 'html', 'css', 'javascript', 'typescript', 'svelte', 'next.js', 'nuxt']
        
        for tech in tech_stack:
            if any(indicator in tech.lower() for indicator in frontend_indicators):
                return True
        
        # Check file extensions
        files = work_order_data.get('files_created', []) + work_order_data.get('files_modified', [])
        frontend_extensions = ['.html', '.css', '.js', '.ts', '.jsx', '.tsx', '.vue', '.svelte']
        
        for file_path in files:
            if any(file_path.endswith(ext) for ext in frontend_extensions):
                return True
        
        return False
    
    async def _skip_visual_testing(self, work_order_data: Dict[str, Any]) -> AgentResult:
        """Skip visual testing for non-frontend projects."""
        self.logger.info("Skipping visual testing - not a frontend project")
        
        return AgentResult(
            success=True,
            confidence=1.0,
            output={
                'visual_testing_skipped': True,
                'reason': 'Not a frontend project',
                'work_order_data': work_order_data,
                'next_agent': 'documentation'
            },
            error=None,
            execution_time=0.0,
            agent_id=self.agent_id,
            timestamp=datetime.now()
        )
    
    async def _run_visual_testing(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive visual testing."""
        testing_results = {
            'success': True,
            'framework_detected': 'unknown',
            'application_started': False,
            'screenshot_captured': False,
            'screenshot_path': None,
            'pure_visual_analysis': {},
            'code_comparison_analysis': {},
            'visual_match_score': 0.0,
            'error_boundaries_detected': 0,
            'visual_issues': [],
            'screenshot_quality': 0.0,
            'elements_matched': 0,
            'elements_expected': 0,
            'execution_time': 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: Detect framework
            framework = self._detect_framework(work_order_data)
            testing_results['framework_detected'] = framework
            
            # Step 2: Start application
            app_process = await self._start_application(work_order_data)
            testing_results['application_started'] = app_process is not None
            
            if not testing_results['application_started']:
                testing_results['success'] = False
                testing_results['error'] = 'Failed to start application'
                return testing_results
            
            try:
                # Step 3: Wait for application to be ready
                await self._wait_for_application_ready(work_order_data)
                
                # Step 4: Capture screenshot
                screenshot_path = await self._capture_screenshot(work_order_data)
                testing_results['screenshot_captured'] = screenshot_path is not None
                testing_results['screenshot_path'] = screenshot_path
                
                if not testing_results['screenshot_captured']:
                    testing_results['success'] = False
                    testing_results['error'] = 'Failed to capture screenshot'
                    return testing_results
                
                # Step 5: Run pure visual analysis
                pure_visual_analysis = await self._run_pure_visual_analysis(screenshot_path)
                testing_results['pure_visual_analysis'] = pure_visual_analysis
                
                # Step 6: Run code comparison analysis
                expected_appearance = work_order_data.get('expected_browser_appearance')
                if expected_appearance:
                    comparison_analysis = await self._run_code_comparison_analysis(screenshot_path, expected_appearance)
                    testing_results['code_comparison_analysis'] = comparison_analysis
                    
                    # Calculate visual match score
                    testing_results['visual_match_score'] = comparison_analysis.get('match_score', 0.0)
                    testing_results['elements_matched'] = comparison_analysis.get('elements_matched', 0)
                    testing_results['elements_expected'] = comparison_analysis.get('elements_expected', 1)
                
                # Step 7: Detect error boundaries
                error_boundaries = await self._detect_error_boundaries(screenshot_path, pure_visual_analysis)
                testing_results['error_boundaries_detected'] = len(error_boundaries)
                
                # Step 8: Assess screenshot quality
                testing_results['screenshot_quality'] = await self._assess_screenshot_quality(screenshot_path)
                
                # Step 9: Identify visual issues
                visual_issues = self._identify_visual_issues(pure_visual_analysis, testing_results.get('code_comparison_analysis', {}))
                testing_results['visual_issues'] = visual_issues
                
                # Determine overall success
                testing_results['success'] = self._determine_visual_testing_success(testing_results)
                
            finally:
                # Clean up: stop application
                if app_process:
                    await self._stop_application(app_process)
        
        except Exception as e:
            testing_results['success'] = False
            testing_results['error'] = str(e)
            self.logger.error(f"Visual testing failed: {e}")
        
        testing_results['execution_time'] = (datetime.now() - start_time).total_seconds()
        return testing_results
    
    def _detect_framework(self, work_order_data: Dict[str, Any]) -> str:
        """Detect the frontend framework being used."""
        # Check technology stack
        tech_stack = work_order_data.get('technology_stack', [])
        for tech in tech_stack:
            tech_lower = tech.lower()
            for framework, patterns in self.framework_patterns.items():
                if any(pattern in tech_lower for pattern in patterns):
                    return framework
        
        # Check file types
        files = work_order_data.get('files_created', []) + work_order_data.get('files_modified', [])
        
        # React indicators
        if any(f.endswith(('.jsx', '.tsx')) for f in files):
            return 'react'
        
        # Vue indicators
        if any(f.endswith('.vue') for f in files):
            return 'vue'
        
        # Angular indicators
        if any('component.ts' in f or 'module.ts' in f for f in files):
            return 'angular'
        
        # Svelte indicators
        if any(f.endswith('.svelte') for f in files):
            return 'svelte'
        
        # Check for package.json
        if Path('package.json').exists():
            try:
                with open('package.json', 'r') as f:
                    package_data = json.load(f)
                    deps = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
                    
                    for framework, patterns in self.framework_patterns.items():
                        if any(pattern in deps for pattern in patterns):
                            return framework
            except:
                pass
        
        return 'vanilla'
    
    async def _start_application(self, work_order_data: Dict[str, Any]) -> Optional[asyncio.subprocess.Process]:
        """Start the application for visual testing."""
        start_command = work_order_data.get('start_command')
        framework = work_order_data.get('framework', 'unknown')
        
        if not start_command:
            # Try to detect default start command based on framework
            if framework == 'react':
                start_command = 'npm start'
            elif framework == 'vue':
                start_command = 'npm run serve'
            elif framework == 'angular':
                start_command = 'ng serve'
            elif framework == 'svelte':
                start_command = 'npm run dev'
            elif Path('package.json').exists():
                start_command = 'npm start'
            else:
                # Try to serve static files
                start_command = 'python -m http.server 8000'
        
        try:
            self.logger.info(f"Starting application with command: {start_command}")
            
            # Split command for subprocess
            cmd_parts = start_command.split()
            
            # Start the application
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='.'
            )
            
            # Give it a moment to start
            await asyncio.sleep(3)
            
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            return None
    
    async def _wait_for_application_ready(self, work_order_data: Dict[str, Any]) -> bool:
        """Wait for application to be ready."""
        url = work_order_data.get('application_url', 'http://localhost:3000')
        timeout = self.page_load_timeout
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                for attempt in range(timeout):
                    try:
                        async with session.get(url, timeout=1) as response:
                            if response.status < 500:
                                self.logger.info(f"Application ready at {url}")
                                return True
                    except:
                        pass
                    
                    await asyncio.sleep(1)
            
            self.logger.warning(f"Application not ready after {timeout} seconds")
            return False
            
        except ImportError:
            # Fallback without aiohttp
            self.logger.warning("aiohttp not available, assuming application is ready")
            await asyncio.sleep(5)
            return True
    
    async def _capture_screenshot(self, work_order_data: Dict[str, Any]) -> Optional[str]:
        """Capture screenshot of the running application."""
        url = work_order_data.get('application_url', 'http://localhost:3000')
        work_order_id = work_order_data.get('work_order_id', 'unknown')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_filename = f"screenshot_{work_order_id}_{timestamp}.png"
        screenshot_path = self.screenshot_dir / screenshot_filename
        
        try:
            # Try using playwright first (if available)
            if await self._capture_with_playwright(url, screenshot_path):
                return str(screenshot_path)
            
            # Fallback to selenium
            if await self._capture_with_selenium(url, screenshot_path):
                return str(screenshot_path)
            
            # Fallback to puppeteer
            if await self._capture_with_puppeteer(url, screenshot_path):
                return str(screenshot_path)
            
            self.logger.error("All screenshot capture methods failed")
            return None
            
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return None
    
    async def _capture_with_playwright(self, url: str, screenshot_path: Path) -> bool:
        """Capture screenshot using playwright."""
        try:
            # Try to import playwright
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                await page.wait_for_timeout(2000)  # Wait 2 seconds for page to load
                await page.screenshot(path=screenshot_path)
                await browser.close()
                
                self.logger.info(f"Screenshot captured with playwright: {screenshot_path}")
                return True
                
        except ImportError:
            self.logger.debug("Playwright not available")
            return False
        except Exception as e:
            self.logger.error(f"Playwright screenshot failed: {e}")
            return False
    
    async def _capture_with_selenium(self, url: str, screenshot_path: Path) -> bool:
        """Capture screenshot using selenium."""
        try:
            # Try to import selenium
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            driver.save_screenshot(str(screenshot_path))
            driver.quit()
            
            self.logger.info(f"Screenshot captured with selenium: {screenshot_path}")
            return True
            
        except ImportError:
            self.logger.debug("Selenium not available")
            return False
        except Exception as e:
            self.logger.error(f"Selenium screenshot failed: {e}")
            return False
    
    async def _capture_with_puppeteer(self, url: str, screenshot_path: Path) -> bool:
        """Capture screenshot using puppeteer."""
        try:
            # Create a temporary puppeteer script
            script_content = f"""
            const puppeteer = require('puppeteer');
            
            (async () => {{
                const browser = await puppeteer.launch({{headless: true}});
                const page = await browser.newPage();
                await page.setViewport({{width: 1920, height: 1080}});
                await page.goto('{url}');
                await page.waitForTimeout(2000);
                await page.screenshot({{path: '{screenshot_path}'}});
                await browser.close();
            }})();
            """
            
            script_path = Path("temp_screenshot.js")
            script_path.write_text(script_content)
            
            # Run puppeteer script
            process = await asyncio.create_subprocess_exec(
                'node', str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            # Clean up
            script_path.unlink()
            
            if screenshot_path.exists():
                self.logger.info(f"Screenshot captured with puppeteer: {screenshot_path}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Puppeteer screenshot failed: {e}")
            return False
    
    async def _run_pure_visual_analysis(self, screenshot_path: str) -> Dict[str, Any]:
        """Run pure visual analysis using Claude's vision capabilities."""
        analysis_results = {
            'description': '',
            'layout_analysis': {},
            'ui_components': [],
            'visual_quality': 0.0,
            'errors_detected': [],
            'completeness_score': 0.0
        }
        
        try:
            # Check if Claude API is available
            if not self.system_config.api_keys.anthropic_api_key:
                analysis_results['error'] = 'Claude API key not available'
                return analysis_results
            
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.system_config.api_keys.anthropic_api_key)
            
            # Read and encode image
            with open(screenshot_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            # Create vision analysis request
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.vision_prompts['pure_visual']
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        }
                    ]
                }]
            )
            
            # Parse response
            analysis_text = response.content[0].text
            analysis_results['description'] = analysis_text
            
            # Extract structured information from analysis
            analysis_results.update(self._parse_visual_analysis(analysis_text))
            
        except Exception as e:
            analysis_results['error'] = str(e)
            self.logger.error(f"Pure visual analysis failed: {e}")
        
        return analysis_results
    
    async def _run_code_comparison_analysis(self, screenshot_path: str, expected_appearance: Dict[str, Any]) -> Dict[str, Any]:
        """Run code comparison analysis using Claude's vision capabilities."""
        comparison_results = {
            'match_score': 0.0,
            'elements_matched': 0,
            'elements_expected': 0,
            'missing_elements': [],
            'extra_elements': [],
            'style_differences': [],
            'interaction_validation': [],
            'overall_assessment': ''
        }
        
        try:
            # Check if Claude API is available
            if not self.system_config.api_keys.anthropic_api_key:
                comparison_results['error'] = 'Claude API key not available'
                return comparison_results
            
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.system_config.api_keys.anthropic_api_key)
            
            # Read and encode image
            with open(screenshot_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            # Format expected appearance for prompt
            expected_description = self._format_expected_appearance(expected_appearance)
            
            # Create comparison analysis request
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.vision_prompts['code_comparison'].format(expected_appearance=expected_description)
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        }
                    ]
                }]
            )
            
            # Parse response
            comparison_text = response.content[0].text
            comparison_results['overall_assessment'] = comparison_text
            
            # Extract structured information from comparison
            comparison_results.update(self._parse_comparison_analysis(comparison_text, expected_appearance))
            
        except Exception as e:
            comparison_results['error'] = str(e)
            self.logger.error(f"Code comparison analysis failed: {e}")
        
        return comparison_results
    
    def _format_expected_appearance(self, expected_appearance: Dict[str, Any]) -> str:
        """Format expected appearance for Claude prompt."""
        formatted = []
        
        if expected_appearance.get('description'):
            formatted.append(f"Description: {expected_appearance['description']}")
        
        if expected_appearance.get('elements'):
            formatted.append("Expected Elements:")
            for element in expected_appearance['elements']:
                formatted.append(f"- {element}")
        
        if expected_appearance.get('interactions'):
            formatted.append("Expected Interactions:")
            for interaction in expected_appearance['interactions']:
                formatted.append(f"- {interaction}")
        
        if expected_appearance.get('error_scenarios'):
            formatted.append("Error Scenarios:")
            for scenario in expected_appearance['error_scenarios']:
                formatted.append(f"- {scenario}")
        
        return '\n'.join(formatted)
    
    def _parse_visual_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse visual analysis text into structured data."""
        parsed = {
            'layout_analysis': {},
            'ui_components': [],
            'visual_quality': 0.7,  # Default
            'errors_detected': [],
            'completeness_score': 0.8  # Default
        }
        
        # Extract UI components mentioned
        import re
        
        # Look for component mentions
        component_patterns = [
            r'button',
            r'form',
            r'navigation',
            r'header',
            r'footer',
            r'sidebar',
            r'modal',
            r'dropdown',
            r'table',
            r'list',
            r'card',
            r'menu'
        ]
        
        for pattern in component_patterns:
            matches = re.findall(f'\\b{pattern}\\b', analysis_text, re.IGNORECASE)
            if matches:
                parsed['ui_components'].append(pattern)
        
        # Look for error indicators
        error_indicators = [
            'error',
            'broken',
            'missing',
            'incorrect',
            'failed',
            'problem',
            'issue'
        ]
        
        for indicator in error_indicators:
            matches = re.findall(f'\\b{indicator}\\b.*', analysis_text, re.IGNORECASE)
            parsed['errors_detected'].extend(matches)
        
        # Assess visual quality based on keywords
        quality_keywords = ['good', 'well', 'clear', 'proper', 'correct', 'complete']
        quality_score = sum(1 for keyword in quality_keywords if keyword in analysis_text.lower()) / len(quality_keywords)
        parsed['visual_quality'] = min(1.0, max(0.0, quality_score))
        
        return parsed
    
    def _parse_comparison_analysis(self, comparison_text: str, expected_appearance: Dict[str, Any]) -> Dict[str, Any]:
        """Parse comparison analysis text into structured data."""
        parsed = {
            'match_score': 0.0,
            'elements_matched': 0,
            'elements_expected': len(expected_appearance.get('elements', [])),
            'missing_elements': [],
            'extra_elements': [],
            'style_differences': []
        }
        
        # Extract match score
        import re
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|\/)\s*10', comparison_text)
        if score_match:
            parsed['match_score'] = float(score_match.group(1)) / 10.0
        else:
            # Try to assess based on keywords
            if 'matches' in comparison_text.lower() and 'well' in comparison_text.lower():
                parsed['match_score'] = 0.8
            elif 'matches' in comparison_text.lower():
                parsed['match_score'] = 0.6
            elif 'missing' in comparison_text.lower() or 'incorrect' in comparison_text.lower():
                parsed['match_score'] = 0.3
            else:
                parsed['match_score'] = 0.5
        
        # Count elements mentioned as present
        expected_elements = expected_appearance.get('elements', [])
        for element in expected_elements:
            if element.lower() in comparison_text.lower():
                parsed['elements_matched'] += 1
        
        # Look for missing elements
        missing_patterns = [
            r'missing:?\s*([^\n]+)',
            r'not found:?\s*([^\n]+)',
            r'absent:?\s*([^\n]+)'
        ]
        
        for pattern in missing_patterns:
            matches = re.findall(pattern, comparison_text, re.IGNORECASE)
            parsed['missing_elements'].extend(matches)
        
        return parsed
    
    async def _detect_error_boundaries(self, screenshot_path: str, visual_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect error boundaries in the screenshot."""
        error_boundaries = []
        
        # Check visual analysis for error indicators
        errors_detected = visual_analysis.get('errors_detected', [])
        for error in errors_detected:
            if any(pattern in error.lower() for pattern in self.error_boundary_patterns):
                error_boundaries.append({
                    'type': 'visual_error_boundary',
                    'description': error,
                    'detected_in': 'visual_analysis'
                })
        
        # Check for common error boundary text patterns
        description = visual_analysis.get('description', '')
        for pattern in self.error_boundary_patterns:
            if pattern.lower() in description.lower():
                error_boundaries.append({
                    'type': 'error_boundary_pattern',
                    'pattern': pattern,
                    'detected_in': 'screenshot_analysis'
                })
        
        return error_boundaries
    
    async def _assess_screenshot_quality(self, screenshot_path: str) -> float:
        """Assess the quality of the screenshot."""
        try:
            # Check if screenshot exists and has reasonable size
            screenshot_file = Path(screenshot_path)
            if not screenshot_file.exists():
                return 0.0
            
            file_size = screenshot_file.stat().st_size
            if file_size < 1000:  # Less than 1KB
                return 0.1
            elif file_size < 10000:  # Less than 10KB
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Screenshot quality assessment failed: {e}")
            return 0.0
    
    def _identify_visual_issues(self, pure_visual_analysis: Dict[str, Any], 
                              comparison_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify visual issues from analyses."""
        issues = []
        
        # Issues from pure visual analysis
        errors_detected = pure_visual_analysis.get('errors_detected', [])
        for error in errors_detected:
            issues.append({
                'type': 'visual_error',
                'description': error,
                'severity': 'high',
                'source': 'pure_visual_analysis'
            })
        
        # Issues from comparison analysis
        missing_elements = comparison_analysis.get('missing_elements', [])
        for element in missing_elements:
            issues.append({
                'type': 'missing_element',
                'description': f"Missing element: {element}",
                'severity': 'medium',
                'source': 'comparison_analysis'
            })
        
        style_differences = comparison_analysis.get('style_differences', [])
        for diff in style_differences:
            issues.append({
                'type': 'style_difference',
                'description': diff,
                'severity': 'low',
                'source': 'comparison_analysis'
            })
        
        return issues
    
    def _determine_visual_testing_success(self, testing_results: Dict[str, Any]) -> bool:
        """Determine if visual testing was successful."""
        # Check if screenshot was captured
        if not testing_results.get('screenshot_captured', False):
            return False
        
        # Check visual match score
        visual_match_score = testing_results.get('visual_match_score', 0.0)
        if visual_match_score < 0.6:  # 60% threshold
            return False
        
        # Check for error boundaries
        if testing_results.get('error_boundaries_detected', 0) > 0:
            return False
        
        # Check visual issues
        visual_issues = testing_results.get('visual_issues', [])
        high_severity_issues = [issue for issue in visual_issues if issue.get('severity') == 'high']
        if len(high_severity_issues) > 0:
            return False
        
        return True
    
    async def _stop_application(self, process: asyncio.subprocess.Process):
        """Stop the application process."""
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        except Exception as e:
            self.logger.error(f"Failed to stop application: {e}")
    
    def _calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """Calculate confidence score for visual testing."""
        base_confidence = 0.5
        
        # Visual match score factor
        visual_match_score = indicators.get('visual_match_score', 0.0)
        base_confidence += visual_match_score * 0.3
        
        # Error boundaries factor
        error_boundaries = indicators.get('error_boundaries_detected', 0)
        if error_boundaries == 0:
            base_confidence += 0.1
        else:
            base_confidence -= error_boundaries * 0.1
        
        # Screenshot quality factor
        screenshot_quality = indicators.get('screenshot_quality', 0.0)
        base_confidence += screenshot_quality * 0.1
        
        # Elements matched factor
        elements_matched = indicators.get('elements_matched', 0)
        elements_expected = indicators.get('elements_expected', 1)
        if elements_expected > 0:
            match_ratio = elements_matched / elements_expected
            base_confidence += match_ratio * 0.1
        
        return max(0.0, min(1.0, base_confidence))