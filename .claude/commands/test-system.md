# Test System Command

Run comprehensive tests for the StarterKit system components.

## Usage

```
/test-system [component] [test-type]
```

## Parameters

- `component`: Specific component to test (optional, defaults to all)
  - `config`: Test configuration system
  - `logger`: Test logging system
  - `voice`: Test voice alerts system
  - `models`: Test data models
  - `core`: Test all core components
- `test-type`: Type of test to run (optional, defaults to unit)
  - `unit`: Unit tests
  - `integration`: Integration tests
  - `performance`: Performance tests
  - `all`: All test types

## Examples

```
/test-system
/test-system config
/test-system voice integration
/test-system core all
```

## Test Commands

This command will execute the appropriate test commands:

### Unit Tests
```bash
pytest tests/test_core/ -v
pytest tests/test_core/test_config.py -v
pytest tests/test_core/test_logger.py -v
pytest tests/test_core/test_voice_alerts.py -v
pytest tests/test_core/test_models.py -v
```

### Integration Tests
```bash
pytest tests/test_integration/ -v
```

### Performance Tests
```bash
pytest tests/test_performance/ -v
```

### Coverage Analysis
```bash
pytest tests/ --cov=StarterKit --cov-report=html
```

## Test Categories

### Configuration Tests
- Environment variable loading
- API key validation
- Configuration validation
- Settings parsing

### Logging Tests
- Logger initialization
- Message formatting
- Agent context tracking
- Performance monitoring

### Voice Alert Tests
- Engine initialization
- Message queuing
- Platform compatibility
- Error handling

### Data Model Tests
- Pydantic validation
- Model serialization
- Factory functions
- Business logic

## Expected Outputs

- Test results summary
- Coverage report
- Performance metrics
- Error reports if any tests fail

## Notes

- Tests are run in isolated environments
- All tests should pass for a healthy system
- Performance tests may take longer to execute
- Use Docker for consistent test environments