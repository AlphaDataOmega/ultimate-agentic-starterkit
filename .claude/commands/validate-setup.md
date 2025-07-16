# Validate Setup Command

Validate that the StarterKit system is properly configured and ready for use.

## Usage

```
/validate-setup [level]
```

## Parameters

- `level`: Validation level (optional, defaults to basic)
  - `basic`: Basic setup validation
  - `comprehensive`: Full system validation
  - `production`: Production readiness check

## Examples

```
/validate-setup
/validate-setup basic
/validate-setup comprehensive
/validate-setup production
```

## Validation Levels

### Basic Validation
- Project structure exists
- Required files are present
- Python environment is correct
- Dependencies are installed

### Comprehensive Validation
- All basic checks pass
- Configuration loading works
- API keys are valid
- Logging system functions
- Voice alerts work
- Data models validate

### Production Validation
- All comprehensive checks pass
- Security configuration
- Performance benchmarks
- Resource limits
- Monitoring setup

## Validation Commands

This command will run these validation steps:

### Level 1: Basic Setup
```bash
# Verify project structure
ls -la StarterKit/core/

# Test configuration loading
python -c "from StarterKit.core.config import get_config; print('Config loaded successfully')"

# Test imports
python -c "from StarterKit.core import get_logger, get_config; print('Imports successful')"
```

### Level 2: Component Testing
```bash
# Test configuration validation
python -c "
from StarterKit.core.config import load_config
config = load_config()
print(f'Configuration valid: {config is not None}')
"

# Test logger system
python -c "
from StarterKit.core.logger import get_logger
logger = get_logger('test')
logger.info('Test message')
print('Logger working')
"

# Test voice alerts
python -c "
from StarterKit.core.voice_alerts import test_voice_system
result = test_voice_system()
print(f'Voice system: {\"working\" if result else \"not working\"}')
"

# Test data models
python -c "
from StarterKit.core.models import ProjectTask, AgentType, TaskStatus
task = ProjectTask(
    title='Test Task',
    description='Test description',
    type='CREATE',
    agent_type=AgentType.PARSER
)
print(f'Model validation: {task.status == TaskStatus.PENDING}')
"
```

### Level 3: Integration Testing
```bash
# Run full test suite
pytest tests/ -v

# Check code quality
ruff check StarterKit/
mypy StarterKit/

# Performance validation
python -c "
from StarterKit.core.logger import get_logger
from StarterKit.core.config import get_config
import time

logger = get_logger('perf_test')
config = get_config()

start = time.time()
for i in range(1000):
    logger.info(f'Performance test message {i}')
duration = time.time() - start

print(f'Logging performance: {1000/duration:.2f} messages/second')
"
```

## Expected Results

### Successful Validation
- ✅ Project structure valid
- ✅ Configuration loaded
- ✅ All imports successful
- ✅ API keys configured
- ✅ Logging system working
- ✅ Voice alerts functional
- ✅ Data models valid
- ✅ Tests passing

### Common Issues and Solutions

#### Configuration Issues
- Missing .env file → Copy .env.example to .env
- Invalid API keys → Check key format and permissions
- Permission errors → Check file permissions

#### Dependency Issues
- Import errors → Run `pip install -r requirements.txt`
- Version conflicts → Use virtual environment
- System dependencies → Install system packages

#### Voice Alert Issues
- No voices available → Install system TTS packages
- Permission denied → Check audio system permissions
- Platform compatibility → Review platform-specific requirements

## Output Format

The validation will output:
- ✅ Passed checks
- ❌ Failed checks
- ⚠️ Warnings
- 📋 Recommendations

## Recovery Actions

If validation fails, the command will suggest:
1. Specific commands to fix issues
2. Configuration changes needed
3. Dependencies to install
4. Permissions to adjust

## Notes

- Run validation after any system changes
- Use Docker for consistent validation environment
- Production validation requires all API keys
- Some checks may be platform-specific