"""
Tests for ParserAgent class.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from StarterKit.agents.parser_agent import ParserAgent
from StarterKit.core.models import ProjectTask, AgentType, TaskStatus


class TestParserAgent:
    """Test cases for ParserAgent class."""
    
    def test_init_default_config(self):
        """Test ParserAgent initialization with default config."""
        agent = ParserAgent()
        
        assert agent.agent_type == "parser"
        assert agent.agent_id.startswith("parser_")
        assert agent.model_name == 'all-MiniLM-L6-v2'
        assert agent.chunk_size == 256
        assert agent.similarity_threshold == 0.7
        assert agent.max_tasks_per_chunk == 5
    
    def test_init_custom_config(self):
        """Test ParserAgent initialization with custom config."""
        config = {
            'model_name': 'custom-model',
            'chunk_size': 512,
            'similarity_threshold': 0.8,
            'max_tasks_per_chunk': 3
        }
        agent = ParserAgent(config)
        
        assert agent.model_name == 'custom-model'
        assert agent.chunk_size == 512
        assert agent.similarity_threshold == 0.8
        assert agent.max_tasks_per_chunk == 3
    
    @patch('StarterKit.agents.parser_agent.DEPENDENCIES_AVAILABLE', False)
    def test_init_without_dependencies(self):
        """Test ParserAgent initialization without required dependencies."""
        agent = ParserAgent()
        assert agent.model is None
    
    @patch('StarterKit.agents.parser_agent.DEPENDENCIES_AVAILABLE', True)
    @patch('StarterKit.agents.parser_agent.SentenceTransformer')
    def test_init_with_dependencies(self, mock_transformer):
        """Test ParserAgent initialization with dependencies."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        agent = ParserAgent()
        
        mock_transformer.assert_called_once_with('all-MiniLM-L6-v2')
        assert agent.model == mock_model
    
    @pytest.mark.asyncio
    async def test_execute_invalid_task(self):
        """Test execute with invalid task."""
        agent = ParserAgent()
        
        # Test with None task
        result = await agent.execute(None)
        assert result.success is False
        assert "Invalid task" in result.error
        
        # Test with empty description
        task = ProjectTask(
            title="Test Task",
            description="",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        result = await agent.execute(task)
        assert result.success is False
        assert "Invalid task" in result.error
    
    @pytest.mark.asyncio
    @patch('StarterKit.agents.parser_agent.DEPENDENCIES_AVAILABLE', False)
    async def test_execute_without_model(self):
        """Test execute without model available."""
        agent = ParserAgent()
        
        task = ProjectTask(
            title="Test Task",
            description="Test description with some content",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        result = await agent.execute(task)
        assert result.success is False
        assert "not available" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_no_content(self):
        """Test execute with no content to parse."""
        agent = ParserAgent()
        agent.model = Mock()  # Mock model
        
        task = ProjectTask(
            title="Test Task",
            description="",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        result = await agent.execute(task)
        assert result.success is False
        assert "No content to parse" in result.error
    
    @pytest.mark.asyncio
    @patch('StarterKit.agents.parser_agent.DEPENDENCIES_AVAILABLE', True)
    async def test_execute_success(self):
        """Test successful task execution."""
        agent = ParserAgent()
        
        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        agent.model = mock_model
        
        # Mock the methods
        agent._create_chunks = Mock(return_value=["chunk1", "chunk2"])
        agent._extract_tasks_semantic = Mock(return_value=[])
        agent._extract_tasks_patterns = Mock(return_value=[])
        agent._combine_and_deduplicate = Mock(return_value=[
            {
                'title': 'Test Task',
                'description': 'Test description',
                'type': 'CREATE',
                'agent_type': AgentType.CODER,
                'confidence': 0.8
            }
        ])
        
        task = ProjectTask(
            title="Test Task",
            description="# Task 1\n- Create something\n- Test it",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        result = await agent.execute(task)
        
        assert result.success is True
        assert result.confidence > 0.0
        assert 'tasks' in result.output
        assert len(result.output['tasks']) == 1
    
    def test_create_chunks_with_headers(self):
        """Test chunk creation with headers."""
        agent = ParserAgent()
        
        text = """# Header 1
This is content under header 1.
More content here.

# Header 2
This is content under header 2.
Even more content.

# Header 3
Final header content."""
        
        chunks = agent._create_chunks(text)
        
        assert len(chunks) >= 3
        assert all("# Header" in chunk for chunk in chunks)
    
    def test_create_chunks_long_content(self):
        """Test chunk creation with long content."""
        agent = ParserAgent({'chunk_size': 10})  # Small chunk size
        
        text = " ".join(["word"] * 100)  # 100 words
        chunks = agent._create_chunks(text)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 15 for chunk in chunks)  # Some tolerance
    
    def test_extract_task_from_chunk(self):
        """Test task extraction from chunk."""
        agent = ParserAgent()
        
        chunk = """# Development Tasks
- Create user authentication system
- Implement data validation
- Add error handling
- Write unit tests"""
        
        tasks = agent._extract_task_from_chunk(chunk, "Create component", 0.8)
        
        assert len(tasks) > 0
        assert any("authentication" in task['title'] for task in tasks)
        assert all(task['confidence'] == 0.8 for task in tasks)
    
    def test_extract_tasks_patterns(self):
        """Test pattern-based task extraction."""
        agent = ParserAgent()
        
        content = """Tasks to complete:
1. Set up development environment
2. Create database schema
- Implement user registration
- Add authentication middleware
TODO: Write documentation
[ ] Deploy to production"""
        
        tasks = agent._extract_tasks_patterns(content)
        
        assert len(tasks) > 0
        assert any("development environment" in task['title'] for task in tasks)
        assert any("database schema" in task['title'] for task in tasks)
        assert any("user registration" in task['title'] for task in tasks)
    
    def test_classify_task_type(self):
        """Test task type classification."""
        agent = ParserAgent()
        
        # Test CREATE type
        assert agent._classify_task_type("Create user authentication") == "CREATE"
        assert agent._classify_task_type("Build new feature") == "CREATE"
        assert agent._classify_task_type("Implement API endpoint") == "CREATE"
        
        # Test TEST type
        assert agent._classify_task_type("Test user login") == "TEST"
        assert agent._classify_task_type("Validate input data") == "TEST"
        assert agent._classify_task_type("Verify functionality") == "TEST"
        
        # Test MODIFY type
        assert agent._classify_task_type("Update user profile") == "MODIFY"
        assert agent._classify_task_type("Fix authentication bug") == "MODIFY"
        assert agent._classify_task_type("Refactor code structure") == "MODIFY"
        
        # Test VALIDATE type
        assert agent._classify_task_type("Review code quality") == "VALIDATE"
        assert agent._classify_task_type("Audit security") == "VALIDATE"
        
        # Test default
        assert agent._classify_task_type("Something else") == "CREATE"
    
    def test_suggest_agent_type(self):
        """Test agent type suggestion."""
        agent = ParserAgent()
        
        # Test TESTER
        assert agent._suggest_agent_type("Test user login") == AgentType.TESTER
        assert agent._suggest_agent_type("Validate input data") == AgentType.TESTER
        
        # Test ADVISOR
        assert agent._suggest_agent_type("Review code quality") == AgentType.ADVISOR
        assert agent._suggest_agent_type("Improve performance") == AgentType.ADVISOR
        
        # Test PARSER
        assert agent._suggest_agent_type("Parse configuration file") == AgentType.PARSER
        assert agent._suggest_agent_type("Extract data from document") == AgentType.PARSER
        
        # Test CODER (default)
        assert agent._suggest_agent_type("Create user interface") == AgentType.CODER
        assert agent._suggest_agent_type("Something else") == AgentType.CODER
    
    def test_combine_and_deduplicate(self):
        """Test task combination and deduplication."""
        agent = ParserAgent()
        
        semantic_tasks = [
            {
                'title': 'Create user authentication',
                'description': 'Implement user auth system',
                'type': 'CREATE',
                'agent_type': AgentType.CODER,
                'confidence': 0.8
            },
            {
                'title': 'Test user login',
                'description': 'Test the login functionality',
                'type': 'TEST',
                'agent_type': AgentType.TESTER,
                'confidence': 0.7
            }
        ]
        
        pattern_tasks = [
            {
                'title': 'Create user authentication',  # Duplicate
                'description': 'Build auth system',
                'type': 'CREATE',
                'agent_type': AgentType.CODER,
                'confidence': 0.6
            },
            {
                'title': 'Add database migration',
                'description': 'Create migration scripts',
                'type': 'CREATE',
                'agent_type': AgentType.CODER,
                'confidence': 0.9
            }
        ]
        
        combined = agent._combine_and_deduplicate(semantic_tasks, pattern_tasks)
        
        assert len(combined) == 3  # One duplicate removed
        titles = [task['title'] for task in combined]
        assert 'Create user authentication' in titles
        assert 'Test user login' in titles
        assert 'Add database migration' in titles
    
    def test_titles_similar(self):
        """Test title similarity detection."""
        agent = ParserAgent()
        
        # Similar titles
        assert agent._titles_similar("create user authentication", "create user auth") is True
        assert agent._titles_similar("test user login", "test login functionality") is True
        
        # Different titles
        assert agent._titles_similar("create user authentication", "test database") is False
        assert agent._titles_similar("completely different", "nothing in common") is False
    
    def test_calculate_parsing_confidence(self):
        """Test parsing confidence calculation."""
        agent = ParserAgent()
        
        # Test with no tasks
        confidence = agent._calculate_parsing_confidence([], "some content")
        assert confidence == 0.0
        
        # Test with tasks
        tasks = [
            {
                'title': 'Task 1',
                'confidence': 0.8,
                'extraction_method': 'semantic'
            },
            {
                'title': 'Task 2',
                'confidence': 0.9,
                'extraction_method': 'pattern'
            }
        ]
        
        confidence = agent._calculate_parsing_confidence(tasks, "some content here")
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high
    
    def test_validate_task_valid(self):
        """Test task validation with valid parser task."""
        agent = ParserAgent()
        
        task = ProjectTask(
            title="Parse Document",
            description="Parse this document and extract tasks",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        assert agent._validate_task(task) is True
    
    def test_validate_task_invalid_agent_type(self):
        """Test task validation with wrong agent type."""
        agent = ParserAgent()
        
        task = ProjectTask(
            title="Code Generation",
            description="Generate some code",
            type="CREATE",
            agent_type=AgentType.CODER
        )
        
        assert agent._validate_task(task) is False
    
    def test_validate_task_insufficient_content(self):
        """Test task validation with insufficient content."""
        agent = ParserAgent()
        
        task = ProjectTask(
            title="Parse",
            description="Short",
            type="CREATE",
            agent_type=AgentType.PARSER
        )
        
        assert agent._validate_task(task) is False