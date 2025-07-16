"""
Test configuration management system.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from StarterKit.core.config import (
    APIKeys,
    LoggingConfig,
    VoiceConfig,
    AgentConfig,
    PerformanceConfig,
    StarterKitConfig,
    load_config,
    get_config,
    reload_config,
    get_config_summary
)


class TestAPIKeys:
    """Test API keys container."""
    
    def test_valid_openai_key(self):
        """Test valid OpenAI API key."""
        api_keys = APIKeys(openai_api_key="sk-test123")
        assert api_keys.has_openai()
    
    def test_invalid_openai_key(self):
        """Test invalid OpenAI API key format."""
        with pytest.raises(ValueError, match="OpenAI API key must start with 'sk-'"):
            APIKeys(openai_api_key="invalid_key")
    
    def test_valid_anthropic_key(self):
        """Test valid Anthropic API key."""
        api_keys = APIKeys(anthropic_api_key="sk-ant-test123")
        assert api_keys.has_anthropic()
    
    def test_invalid_anthropic_key(self):
        """Test invalid Anthropic API key format."""
        with pytest.raises(ValueError, match="Anthropic API key must start with 'sk-ant-'"):
            APIKeys(anthropic_api_key="invalid_key")
    
    def test_huggingface_key(self):
        """Test HuggingFace API key."""
        api_keys = APIKeys(huggingface_api_key="hf_test123")
        assert api_keys.has_huggingface()
    
    def test_missing_keys(self):
        """Test missing API keys."""
        api_keys = APIKeys()
        assert not api_keys.has_openai()
        assert not api_keys.has_anthropic()
        assert not api_keys.has_huggingface()


class TestLoggingConfig:
    """Test logging configuration."""
    
    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.structured is True
        assert config.rotation is True
    
    def test_invalid_log_level(self):
        """Test invalid log level."""
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")
    
    def test_file_path_creation(self):
        """Test that log directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "test_logs" / "test.log"
            config = LoggingConfig(file_path=str(log_path))
            assert log_path.parent.exists()


class TestVoiceConfig:
    """Test voice configuration."""
    
    def test_default_config(self):
        """Test default voice configuration."""
        config = VoiceConfig()
        assert config.enabled is True
        assert config.rate == 200
        assert config.volume == 0.7
    
    def test_invalid_rate(self):
        """Test invalid speech rate."""
        with pytest.raises(ValueError):
            VoiceConfig(rate=1000)  # Too high
    
    def test_invalid_volume(self):
        """Test invalid volume level."""
        with pytest.raises(ValueError):
            VoiceConfig(volume=2.0)  # Too high


class TestAgentConfig:
    """Test agent configuration."""
    
    def test_default_config(self):
        """Test default agent configuration."""
        config = AgentConfig()
        assert config.min_confidence_threshold == 0.7
        assert config.high_confidence_threshold == 0.9
        assert config.max_retry_attempts == 3
    
    def test_invalid_confidence_thresholds(self):
        """Test invalid confidence threshold configuration."""
        with pytest.raises(ValueError):
            AgentConfig(
                min_confidence_threshold=0.9,
                high_confidence_threshold=0.7  # Should be higher than min
            )


class TestPerformanceConfig:
    """Test performance configuration."""
    
    def test_default_config(self):
        """Test default performance configuration."""
        config = PerformanceConfig()
        assert config.async_concurrency_limit == 10
        assert config.rate_limit_rpm == 60
        assert config.memory_limit_mb == 2048


class TestStarterKitConfig:
    """Test main configuration class."""
    
    def test_default_config(self):
        """Test default configuration."""
        with patch.dict(os.environ, {}, clear=True):
            config = StarterKitConfig()
            assert config.environment == "development"
            assert config.debug is True
            assert config.openai_model == "gpt-4"
    
    def test_directory_creation(self):
        """Test that configured directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / "work"
            temp_path = Path(temp_dir) / "temp"
            output_dir = Path(temp_dir) / "output"
            
            config = StarterKitConfig(
                project_work_dir=str(work_dir),
                temp_dir=str(temp_path),
                output_dir=str(output_dir)
            )
            
            assert work_dir.exists()
            assert temp_path.exists()
            assert output_dir.exists()
    
    def test_api_key_validation(self):
        """Test API key validation."""
        with patch.dict(os.environ, {}, clear=True):
            config = StarterKitConfig()
            missing_keys = config.validate_api_keys()
            assert "OPENAI_API_KEY" in missing_keys
            assert "ANTHROPIC_API_KEY" in missing_keys
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-test123'
        }):
            config = StarterKitConfig()
            model_config = config.get_model_config("openai")
            assert model_config["model"] == "gpt-4"
            assert model_config["api_key"] == "sk-test123"
    
    def test_invalid_provider(self):
        """Test invalid provider for model config."""
        config = StarterKitConfig()
        with pytest.raises(ValueError, match="Unknown provider"):
            config.get_model_config("invalid_provider")
    
    def test_environment_checks(self):
        """Test environment checking methods."""
        config = StarterKitConfig(environment="production")
        assert config.is_production()
        assert not config.is_development()
        
        config = StarterKitConfig(environment="development")
        assert config.is_development()
        assert not config.is_production()
    
    def test_to_dict_excludes_api_keys(self):
        """Test that to_dict excludes API keys."""
        config = StarterKitConfig()
        config_dict = config.to_dict()
        assert "api_keys" not in config_dict


class TestConfigLoading:
    """Test configuration loading functions."""
    
    def test_load_config_with_env_vars(self):
        """Test loading configuration with environment variables."""
        test_env = {
            'ENVIRONMENT': 'production',
            'DEBUG': 'false',
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'LOG_LEVEL': 'WARNING',
            'VOICE_RATE': '150',
            'MIN_CONFIDENCE_THRESHOLD': '0.8'
        }
        
        with patch.dict(os.environ, test_env):
            config = load_config(validate_keys=False)
            assert config.environment == "production"
            assert config.debug is False
            assert config.openai_model == "gpt-3.5-turbo"
            assert config.logging.level == "WARNING"
            assert config.voice.rate == 150
            assert config.agent.min_confidence_threshold == 0.8
    
    def test_load_config_with_missing_keys(self):
        """Test loading configuration with missing API keys."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required API keys"):
                load_config(validate_keys=True)
    
    def test_load_config_validation_disabled(self):
        """Test loading configuration with validation disabled."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config(validate_keys=False)
            assert config is not None
    
    def test_load_config_with_env_file(self):
        """Test loading configuration from .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("ENVIRONMENT=testing\n")
            f.write("DEBUG=false\n")
            f.write("OPENAI_API_KEY=sk-test123\n")
            f.write("ANTHROPIC_API_KEY=sk-ant-test123\n")
            env_file = f.name
        
        try:
            config = load_config(env_file=env_file)
            assert config.environment == "testing"
            assert config.debug is False
            assert config.api_keys.has_openai()
            assert config.api_keys.has_anthropic()
        finally:
            os.unlink(env_file)
    
    def test_invalid_configuration_data(self):
        """Test loading with invalid configuration data."""
        test_env = {
            'LOG_LEVEL': 'INVALID_LEVEL',
            'VOICE_RATE': '999999',  # Too high
            'MIN_CONFIDENCE_THRESHOLD': '2.0',  # Too high
        }
        
        with patch.dict(os.environ, test_env):
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_config(validate_keys=False)


class TestGlobalConfig:
    """Test global configuration management."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        # Reset global instance
        import StarterKit.core.config
        StarterKit.core.config._config_instance = None
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-test123',
            'ANTHROPIC_API_KEY': 'sk-ant-test123'
        }):
            config1 = get_config()
            config2 = get_config()
            assert config1 is config2
    
    def test_reload_config(self):
        """Test reloading configuration."""
        # Reset global instance
        import StarterKit.core.config
        StarterKit.core.config._config_instance = None
        
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'development',
            'OPENAI_API_KEY': 'sk-test123',
            'ANTHROPIC_API_KEY': 'sk-ant-test123'
        }):
            config1 = get_config()
            assert config1.environment == "development"
        
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'OPENAI_API_KEY': 'sk-test123',
            'ANTHROPIC_API_KEY': 'sk-ant-test123'
        }):
            config2 = reload_config()
            assert config2.environment == "production"
            assert config2 is not config1


class TestConfigSummary:
    """Test configuration summary generation."""
    
    def test_get_config_summary(self):
        """Test getting configuration summary."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'development',
            'OPENAI_API_KEY': 'sk-test123',
            'ANTHROPIC_API_KEY': 'sk-ant-test123'
        }):
            config = load_config()
            summary = get_config_summary(config)
            
            assert summary['environment'] == 'development'
            assert summary['api_keys_available']['openai'] is True
            assert summary['api_keys_available']['anthropic'] is True
            assert 'models' in summary
            assert 'directories' in summary
    
    def test_config_summary_no_api_keys(self):
        """Test configuration summary without API keys."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config(validate_keys=False)
            summary = get_config_summary(config)
            
            assert summary['api_keys_available']['openai'] is False
            assert summary['api_keys_available']['anthropic'] is False


# Edge cases and error handling
class TestConfigEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_string_values(self):
        """Test handling of empty string values."""
        test_env = {
            'ENVIRONMENT': '',
            'DEBUG': '',
            'LOG_LEVEL': ''
        }
        
        with patch.dict(os.environ, test_env):
            # Should use defaults for empty strings
            config = load_config(validate_keys=False)
            assert config.environment == "development"  # Default
            assert config.debug is True  # Default
            assert config.logging.level == "INFO"  # Default
    
    def test_boolean_parsing(self):
        """Test boolean value parsing."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('1', False),  # Only 'true' should be True
            ('0', False),
            ('', False)
        ]
        
        for value, expected in test_cases:
            with patch.dict(os.environ, {'DEBUG': value}):
                config = load_config(validate_keys=False)
                assert config.debug is expected
    
    def test_numeric_parsing(self):
        """Test numeric value parsing."""
        test_env = {
            'OPENAI_MAX_TOKENS': '2048',
            'VOICE_RATE': '300',
            'VOICE_VOLUME': '0.5',
            'MAX_RETRY_ATTEMPTS': '5'
        }
        
        with patch.dict(os.environ, test_env):
            config = load_config(validate_keys=False)
            assert config.openai_max_tokens == 2048
            assert config.voice.rate == 300
            assert config.voice.volume == 0.5
            assert config.agent.max_retry_attempts == 5
    
    def test_invalid_numeric_values(self):
        """Test handling of invalid numeric values."""
        test_env = {
            'OPENAI_MAX_TOKENS': 'not_a_number',
            'VOICE_RATE': 'invalid'
        }
        
        with patch.dict(os.environ, test_env):
            with pytest.raises(ValueError):
                load_config(validate_keys=False)