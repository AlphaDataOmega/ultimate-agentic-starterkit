# Ultimate Agentic StarterKit

A comprehensive, production-ready framework for building AI-powered autonomous agents that can plan, execute, and validate complex software projects from natural language descriptions.

> **The Ultimate Agentic StarterKit transforms natural language descriptions into working code through intelligent agent coordination.**

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- API keys for Anthropic Claude and/or OpenAI
- Node.js (for visual testing features)
- Git

### Installation

1. **Clone and setup**:
```bash
git clone https://github.com/AlphaDataOmega/Ultimate-Agentic-StarterKit.git
cd Ultimate-Agentic-StarterKit/StarterKit
python -m venv venv_linux
source venv_linux/bin/activate
pip install -r requirements.txt
```

2. **Configure API keys**:
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
ANTHROPIC_API_KEY="your-anthropic-key"
OPENAI_API_KEY="your-openai-key"
```

3. **Test installation**:
```bash
# Go to project root
cd ..
python kit.py --help
```

### Basic Usage

#### Using the CLI

```bash
# Interactive Learning Phase (NEW!)
python kit.py --learn --overview OVERVIEW.md        # Start learning phase
python kit.py --accept-learn                         # Accept learning results

# Execute project from OVERVIEW.md
python kit.py --overview OVERVIEW.md                 # Execute project from overview
python kit.py --overview OVERVIEW.md --validate      # Validate project first
python kit.py --overview OVERVIEW.md --dry-run       # Show work orders

# Execute a project from PRP file
python kit.py --prp PRPs/001A_foundation_core.md

# Validate project specification only
python kit.py --prp PRPs/001A_foundation_core.md --validate

# Show execution plan without running
python kit.py --prp PRPs/001A_foundation_core.md --dry-run

# Generate PRP from description
python kit.py --generate-prp "Build a REST API for todo management"

# List available examples
python kit.py --list-examples

# Show system status
python kit.py --status
```

#### Using the Framework

```python
from StarterKit.core import get_config, get_logger
from StarterKit.core.models import ProjectSpecification, ProjectTask, AgentType
from StarterKit.agents.parser_agent import ParserAgent
from StarterKit.workflows.project_builder import LangGraphWorkflowManager

# Load configuration
config = get_config()

# Get logger with agent context
logger = get_logger('my_agent')

# Create a project specification
project = ProjectSpecification(
    title="My AI Project",
    description="A sample AI-powered project",
    project_type="ai"
)

# Add tasks
task = ProjectTask(
    title="Setup Environment",
    description="Initialize project environment",
    type="CREATE",
    agent_type=AgentType.CODER
)
project.add_task(task)

# Execute with workflow manager
workflow_manager = LangGraphWorkflowManager()
result = await workflow_manager.execute_workflow(project.dict())
```

## 🎓 Interactive Learning Phase

The Ultimate Agentic StarterKit includes an **Interactive Learning Phase** that helps bridge the gap between high-level project ideas and detailed implementation requirements through intelligent questioning and knowledge transfer.

### How It Works

1. **Project Analysis**: The Project Manager Agent (powered by OpenAI o3) analyzes your OVERVIEW.md and identifies knowledge gaps
2. **Question Generation**: Generates prioritized questions (Critical/MVP/Future) to fill these gaps
3. **Research Assistance**: The Research Agent provides suggested answers using local knowledge base, Ollama LLM, or web research
4. **User Review**: You review questions, provide answers, and check off completed items
5. **Incorporation**: Answers are automatically incorporated back into your OVERVIEW.md
6. **Iteration**: The process repeats until the project specification is complete

### Learning Phase Workflow

```bash
# Step 1: Start learning phase
python kit.py --learn --overview OVERVIEW.md

# The system will:
# - Analyze your OVERVIEW.md for completeness
# - Generate prioritized questions in questions.md
# - Research suggested answers automatically
# - Prompt you to review and edit questions.md

# Step 2: Edit questions.md
# - Review generated questions
# - Fill in your answers
# - Check boxes [x] for completed questions

# Step 3: Accept learning results
python kit.py --accept-learn

# The system will:
# - Incorporate your answers into OVERVIEW.md
# - Archive completed questions
# - Check if more learning is needed
# - Optionally proceed to project execution
```

### Question Types & Research

**Question Priorities:**
- **Critical**: Must-have requirements that block development
- **MVP**: Important features for initial version
- **Future**: Nice-to-have features for later iterations

**Research Methods:**
- **Knowledge Base**: Built-in answers for common project questions
- **Ollama Integration**: Local LLM research using Mistral or other models
- **Interactive Prompts**: Voice alerts + CLI prompts for user input
- **Web Research**: Future enhancement for external knowledge

### Example Learning Session

```
🎓 Starting Interactive Learning Phase

Learning Iteration 1/5
✓ Generated 8 questions (3 critical)
✓ Researched 5 questions

📝 Questions ready for review!
Please review and edit: questions.md
Add your answers and check boxes [x] when complete

Run `python kit.py --accept-learn` when ready to continue
```

### Sample Generated Questions

```markdown
### [Critical] Question: What authentication method should be used?

**Why**: Security and user management are fundamental to application design

**Suggested Answer**: JWT tokens with refresh tokens for stateless authentication *(Confidence: 80%)*

**Your Answer**: [ ] We'll use Firebase Auth for social login integration

---

### [MVP] Question: What database will store user data?

**Why**: Data storage decisions affect performance and scalability

**Suggested Answer**: PostgreSQL for complex queries, MongoDB for document storage *(Confidence: 75%)*

**Your Answer**: [x] PostgreSQL with Redis for session storage
```

## 🏗️ Architecture

### Core Components

1. **Agent Framework** (`agents/`): Specialized AI agents for different tasks
   - **Parser Agent**: Extracts tasks from project specifications
   - **Coder Agent**: Generates code using AI models
   - **Tester Agent**: Executes tests and validates outputs
   - **Advisor Agent**: Provides code review and improvement suggestions
   - **Project Manager Agent**: Analyzes projects and generates learning questions (o3-powered)
   - **Research Agent**: Researches answers using knowledge base and Ollama

2. **Workflow System** (`workflows/`): LangGraph-based orchestration
   - **Project Builder**: Main workflow execution engine
   - **Task Executor**: Manages individual task execution
   - **State Manager**: Handles workflow state persistence
   - **Progress Tracker**: Real-time progress monitoring

3. **Validation System** (`validation/`): Quality assurance and testing
   - **Confidence Scoring**: Multi-factor confidence calculation
   - **Quality Gates**: Configurable validation checkpoints
   - **Test Runner**: Automated test execution
   - **Performance Monitor**: Real-time performance tracking

4. **Integration Layer** (`integrations/`): External tool integrations
   - **Claude Code**: VS Code extension integration
   - **Git Manager**: Version control integration
   - **Ollama Client**: Local model integration

5. **Core Infrastructure** (`core/`): Foundation components
   - **Configuration**: Environment and API key management
   - **Logging**: Structured logging with agent context
   - **Models**: Type-safe data structures
   - **Voice Alerts**: Cross-platform notifications

## 📁 Project Structure

```
StarterKit/
├── agents/                    # AI Agent implementations
│   ├── base_agent.py         # Common agent interface
│   ├── parser_agent.py       # Task extraction agent
│   ├── coder_agent.py        # Code generation agent
│   ├── tester_agent.py       # Testing and validation agent
│   ├── advisor_agent.py      # Code review agent
│   └── factory.py            # Agent factory
├── core/                     # Core infrastructure
│   ├── config.py            # Configuration management
│   ├── logger.py            # Logging system
│   ├── models.py            # Data models
│   ├── orchestrator.py      # O3 orchestration
│   └── voice_alerts.py      # Voice notification system
├── workflows/               # Workflow orchestration
│   ├── project_builder.py   # Main workflow engine
│   ├── task_executor.py     # Task execution
│   ├── state_manager.py     # State persistence
│   └── progress_tracker.py  # Progress monitoring
├── validation/              # Quality assurance
│   ├── confidence.py        # Confidence scoring
│   ├── quality_gates.py     # Validation checkpoints
│   ├── test_runner.py       # Test execution
│   └── validator.py         # Main validation orchestrator
├── integrations/            # External integrations
│   ├── claude_code.py       # VS Code integration
│   ├── git_manager.py       # Git operations
│   └── ollama_client.py     # Local model client
├── tests/                   # Comprehensive test suite
│   ├── test_agents/         # Agent tests
│   ├── test_core/           # Core component tests
│   ├── test_workflows/      # Workflow tests
│   ├── test_validation/     # Validation tests
│   └── test_integrations/   # Integration tests
├── examples/                # Usage examples
│   └── agent_demo.py        # Agent demonstration
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker configuration
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables

Configure the framework by editing `.env`:

```bash
# API Keys (required for full functionality)
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_API_KEY=your-huggingface-key  # Optional

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/starterkit.log
LOG_STRUCTURED=true

# Voice Alerts
VOICE_ALERTS_ENABLED=true
VOICE_RATE=200
VOICE_VOLUME=0.7

# Agent Configuration
MIN_CONFIDENCE_THRESHOLD=0.7
MAX_RETRY_ATTEMPTS=3
TASK_TIMEOUT=600
WORKFLOW_TIMEOUT=7200

# Project Configuration
PROJECT_WORK_DIR=workspace
OUTPUT_DIR=output
TEMP_DIR=temp
```

### Agent Configuration

```python
# Example agent configuration
parser_config = {
    'model_name': 'all-MiniLM-L6-v2',
    'similarity_threshold': 0.7,
    'max_tasks_per_chunk': 5,
    'confidence_threshold': 0.8
}

coder_config = {
    'model': 'claude-3-5-sonnet-20241022',
    'max_tokens': 4000,
    'temperature': 0.1,
    'max_file_size': 500
}
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_agents/ -v
python -m pytest tests/test_workflows/ -v
python -m pytest tests/test_validation/ -v

# Run with coverage
python -m pytest tests/ --cov=StarterKit --cov-report=html
```

### Integration Tests

```bash
# Test agent integration
python test_integration_setup.py

# Validate agent functionality
python validate_agents.py
```

### Example Tests

```bash
# Test examples from project root
python examples/parser_example.py
python examples/coder_example.py
python examples/workflow_example.py
python examples/claude_code_example.py
```

## 🔍 Examples

### Parser Agent Example

```python
from StarterKit.agents.parser_agent import ParserAgent
from StarterKit.core.models import create_project_task, AgentType

# Create parser agent
parser = ParserAgent()

# Create task
task = create_project_task(
    title="Parse Web App Project",
    description="# Web App\n## Tasks\n- Setup backend\n- Create frontend",
    task_type="PARSE",
    agent_type=AgentType.PARSER
)

# Execute parser
result = await parser.execute(task)
print(f"Extracted {len(result.output['tasks'])} tasks")
```

### Coder Agent Example

```python
from StarterKit.agents.coder_agent import CoderAgent
from StarterKit.core.models import create_project_task, AgentType

# Create coder agent
coder = CoderAgent()

# Create coding task
task = create_project_task(
    title="Create FastAPI Authentication",
    description="Create FastAPI app with JWT authentication",
    task_type="CREATE",
    agent_type=AgentType.CODER
)

# Execute coder
result = await coder.execute(task)
print(f"Generated {len(result.output['files'])} files")
```

### Workflow Example

```python
from StarterKit.workflows.project_builder import LangGraphWorkflowManager

# Create workflow manager
workflow_manager = LangGraphWorkflowManager()

# Project specification
project_spec = {
    "title": "Simple REST API",
    "description": "Build a REST API for todo management",
    "tasks": [
        {
            "id": "task-1",
            "title": "Setup FastAPI Project",
            "description": "Create FastAPI application structure",
            "type": "CREATE",
            "agent_type": "coder"
        }
    ]
}

# Execute workflow
result = await workflow_manager.execute_workflow(project_spec)
print(f"Workflow status: {result['workflow_status']}")
```

## 🔌 VS Code Integration

The framework includes comprehensive VS Code integration via the Claude Code extension:

```python
from StarterKit.integrations.claude_code import ClaudeCodeIntegration

# Initialize integration
integration = ClaudeCodeIntegration(".")

# Register commands
commands = [
    {
        "name": "agentic.parse.prp",
        "title": "Parse PRP File",
        "description": "Parse a PRP file and extract tasks",
        "handler": "parse_prp_command"
    }
]

result = await integration.register_commands(commands)
```

## 🐳 Docker Development

### Quick Start with Docker

```bash
# Start development environment
docker-compose up -d

# Run tests in container
docker-compose exec app pytest tests/ -v

# Access development shell
docker-compose exec app bash
```

### Custom Docker Setup

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "kit.py", "--help"]
```

## 📊 Performance Monitoring

The framework includes comprehensive performance monitoring:

```python
from StarterKit.validation.performance_monitor import PerformanceMonitor

# Initialize monitor
monitor = PerformanceMonitor()

# Track performance
with monitor.track_operation("task_execution"):
    # Your code here
    pass

# Get metrics
metrics = monitor.get_metrics()
```

## 🔊 Voice Alerts

Cross-platform voice notifications for important events:

```python
from StarterKit.core.voice_alerts import get_voice_alerts

# Get voice alerts instance
voice = get_voice_alerts()

# Speak milestone
voice.speak_milestone("Project completed successfully!")

# Speak error
voice.speak_error("Task execution failed")

# Test voice system
voice.test_voice()
```

## 🚀 Development

### Code Quality

```bash
# Format code
black StarterKit/

# Check linting
ruff check StarterKit/

# Type checking
mypy StarterKit/
```

### Running in Development Mode

```bash
# Install in development mode
pip install -e .

# Run with verbose logging
python kit.py --verbose --status

# Run specific example
python examples/parser_example.py
```

## 📚 Documentation

- [Project Overview](../docs/PROJECT_OVERVIEW.md)
- [Visual Testing](../docs/VISUAL_TESTING.md)
- [Context Documentation](../docs/CONTEXT.md)
- [API Reference](../docs/api/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Guidelines

- Follow PEP8 style guide
- Use type hints for all functions
- Add comprehensive docstrings
- Maximum 500 lines per file
- Create unit tests for all new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support:
- Check the documentation in `docs/`
- Review example implementations
- Open an issue on GitHub
- Review system status with `python kit.py --status`

## 🙏 Acknowledgments

- Built with Claude AI and OpenAI GPT models
- Uses LangGraph for workflow orchestration
- Integrates with VS Code via Claude Code extension
- Semantic search powered by sentence-transformers
- Voice alerts via pyttsx3

---

**The Ultimate Agentic StarterKit: Where Natural Language Meets Production Code**