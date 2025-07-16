"""
Configuration management for the Ultimate Agentic StarterKit.

This module handles loading and validation of environment variables, API keys,
and system settings with comprehensive error handling and validation.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator


@dataclass
class APIKeys:
    """Container for API keys with validation."""
    
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate API keys format."""
        if self.openai_api_key and not self.openai_api_key.startswith('sk-'):
            raise ValueError("OpenAI API key must start with 'sk-'")
        
        if self.anthropic_api_key and not self.anthropic_api_key.startswith('sk-ant-'):
            raise ValueError("Anthropic API key must start with 'sk-ant-'")
    
    def has_openai(self) -> bool:
        """Check if OpenAI API key is available."""
        return self.openai_api_key is not None and len(self.openai_api_key) > 10
    
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is available."""
        return self.anthropic_api_key is not None and len(self.anthropic_api_key) > 10
    
    def has_huggingface(self) -> bool:
        """Check if Hugging Face API key is available."""
        return self.huggingface_api_key is not None and len(self.huggingface_api_key) > 10


class LoggingConfig(BaseModel):
    """Configuration for logging system."""
    
    level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    file_path: str = Field(default="logs/starterkit.log")
    structured: bool = Field(default=True)
    rotation: bool = Field(default=True)
    max_size: str = Field(default="10MB")
    backup_count: int = Field(default=5, ge=1)
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Ensure log directory exists."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return v


class VoiceConfig(BaseModel):
    """Configuration for voice alert system."""
    
    enabled: bool = Field(default=True)
    rate: int = Field(default=200, ge=50, le=500)
    volume: float = Field(default=0.7, ge=0.0, le=1.0)
    voice_id: int = Field(default=0, ge=0)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class AgentConfig(BaseModel):
    """Configuration for agent system."""
    
    min_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    max_retry_attempts: int = Field(default=3, ge=1)
    agent_timeout: int = Field(default=300, ge=30)
    task_timeout: int = Field(default=600, ge=60)
    
    @validator('high_confidence_threshold')
    def validate_confidence_thresholds(cls, v, values):
        """Ensure high threshold is greater than min threshold."""
        if 'min_confidence_threshold' in values and v <= values['min_confidence_threshold']:
            raise ValueError("High confidence threshold must be greater than min threshold")
        return v


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    
    async_concurrency_limit: int = Field(default=10, ge=1)
    rate_limit_rpm: int = Field(default=60, ge=1)
    memory_limit_mb: int = Field(default=2048, ge=256)


class StarterKitConfig(BaseModel):
    """Main configuration class for the StarterKit."""
    
    # Environment settings
    environment: str = Field(default="development", pattern=r"^(development|staging|production)$")
    debug: bool = Field(default=True)
    test_mode: bool = Field(default=False)
    
    # API configuration
    openai_model: str = Field(default="gpt-4")
    openai_max_tokens: int = Field(default=4096, ge=1)
    anthropic_model: str = Field(default="claude-3-sonnet-20240229")
    anthropic_max_tokens: int = Field(default=4096, ge=1)
    huggingface_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    
    # Directory paths
    project_work_dir: str = Field(default="./workspace")
    temp_dir: str = Field(default="./temp")
    output_dir: str = Field(default="./output")
    
    # Component configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # API keys (set separately for security)
    api_keys: Optional[APIKeys] = None
    
    def __init__(self, **data):
        """Initialize configuration with API keys."""
        super().__init__(**data)
        self.api_keys = APIKeys(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY')
        )
    
    @validator('project_work_dir', 'temp_dir', 'output_dir')
    def create_directories(cls, v):
        """Ensure directories exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    def validate_api_keys(self) -> List[str]:
        """
        Validate API keys and return list of missing keys.
        
        Returns:
            List of missing API key names
        """
        missing_keys = []
        
        if not self.api_keys.has_openai():
            missing_keys.append("OPENAI_API_KEY")
        
        if not self.api_keys.has_anthropic():
            missing_keys.append("ANTHROPIC_API_KEY")
        
        # HuggingFace is optional for basic functionality
        if not self.api_keys.has_huggingface():
            missing_keys.append("HUGGINGFACE_API_KEY (optional)")
        
        return missing_keys
    
    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """
        Get model configuration for specified provider.
        
        Args:
            provider: AI provider name (openai, anthropic, huggingface)
        
        Returns:
            Dict with model configuration
        """
        if provider == "openai":
            return {
                "model": self.openai_model,
                "max_tokens": self.openai_max_tokens,
                "api_key": self.api_keys.openai_api_key
            }
        elif provider == "anthropic":
            return {
                "model": self.anthropic_model,
                "max_tokens": self.anthropic_max_tokens,
                "api_key": self.api_keys.anthropic_api_key
            }
        elif provider == "huggingface":
            return {
                "model": self.huggingface_model,
                "api_key": self.api_keys.huggingface_api_key
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding API keys)."""
        config_dict = self.dict()
        # Remove API keys from the dictionary for security
        if 'api_keys' in config_dict:
            del config_dict['api_keys']
        return config_dict
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        json_encoders = {
            Path: str
        }


def load_environment_variables(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Optional path to .env file
    """
    if env_file:
        load_dotenv(env_file)
    else:
        # Try to load from standard locations
        env_paths = ['.env', '.env.local', '.env.development']
        for path in env_paths:
            if os.path.exists(path):
                load_dotenv(path)
                break


def load_config(env_file: Optional[str] = None, validate_keys: bool = True) -> StarterKitConfig:
    """
    Load and validate configuration from environment variables.
    
    Args:
        env_file: Optional path to .env file
        validate_keys: Whether to validate API keys
    
    Returns:
        StarterKitConfig: Validated configuration object
    
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    # Load environment variables
    load_environment_variables(env_file)
    
    # Create configuration from environment variables
    config_data = {}
    
    # Environment settings
    config_data['environment'] = os.getenv('ENVIRONMENT', 'development')
    config_data['debug'] = os.getenv('DEBUG', 'true').lower() == 'true'
    config_data['test_mode'] = os.getenv('TEST_MODE', 'false').lower() == 'true'
    
    # API configuration
    config_data['openai_model'] = os.getenv('OPENAI_MODEL', 'gpt-4')
    config_data['openai_max_tokens'] = int(os.getenv('OPENAI_MAX_TOKENS', '4096'))
    config_data['anthropic_model'] = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
    config_data['anthropic_max_tokens'] = int(os.getenv('ANTHROPIC_MAX_TOKENS', '4096'))
    config_data['huggingface_model'] = os.getenv('HUGGINGFACE_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    
    # Directory paths
    config_data['project_work_dir'] = os.getenv('PROJECT_WORK_DIR', './workspace')
    config_data['temp_dir'] = os.getenv('TEMP_DIR', './temp')
    config_data['output_dir'] = os.getenv('OUTPUT_DIR', './output')
    
    # Logging configuration
    config_data['logging'] = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'file_path': os.getenv('LOG_FILE_PATH', 'logs/starterkit.log'),
        'structured': os.getenv('LOG_STRUCTURED', 'true').lower() == 'true',
        'rotation': os.getenv('LOG_ROTATION', 'true').lower() == 'true',
        'max_size': os.getenv('LOG_MAX_SIZE', '10MB'),
        'backup_count': int(os.getenv('LOG_BACKUP_COUNT', '5'))
    }
    
    # Voice configuration
    config_data['voice'] = {
        'enabled': os.getenv('VOICE_ALERTS_ENABLED', 'true').lower() == 'true',
        'rate': int(os.getenv('VOICE_RATE', '200')),
        'volume': float(os.getenv('VOICE_VOLUME', '0.7')),
        'voice_id': int(os.getenv('VOICE_VOICE_ID', '0'))
    }
    
    # Agent configuration
    config_data['agent'] = {
        'min_confidence_threshold': float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.7')),
        'high_confidence_threshold': float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.9')),
        'max_retry_attempts': int(os.getenv('MAX_RETRY_ATTEMPTS', '3')),
        'agent_timeout': int(os.getenv('AGENT_TIMEOUT', '300')),
        'task_timeout': int(os.getenv('TASK_TIMEOUT', '600'))
    }
    
    # Performance configuration
    config_data['performance'] = {
        'async_concurrency_limit': int(os.getenv('ASYNC_CONCURRENCY_LIMIT', '10')),
        'rate_limit_rpm': int(os.getenv('RATE_LIMIT_RPM', '60')),
        'memory_limit_mb': int(os.getenv('MEMORY_LIMIT_MB', '2048'))
    }
    
    try:
        # Create and validate configuration
        config = StarterKitConfig(**config_data)
        
        # Validate API keys if requested
        if validate_keys:
            missing_keys = config.validate_api_keys()
            if missing_keys:
                required_keys = [key for key in missing_keys if "optional" not in key]
                if required_keys:
                    raise ValueError(f"Missing required API keys: {', '.join(required_keys)}")
        
        return config
        
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {str(e)}")


def get_config_summary(config: StarterKitConfig) -> Dict[str, Any]:
    """
    Get a summary of configuration settings for logging/debugging.
    
    Args:
        config: Configuration object
    
    Returns:
        Dict with configuration summary (excluding sensitive data)
    """
    return {
        'environment': config.environment,
        'debug': config.debug,
        'test_mode': config.test_mode,
        'models': {
            'openai': config.openai_model,
            'anthropic': config.anthropic_model,
            'huggingface': config.huggingface_model
        },
        'logging_level': config.logging.level,
        'voice_enabled': config.voice.enabled,
        'directories': {
            'work': config.project_work_dir,
            'temp': config.temp_dir,
            'output': config.output_dir
        },
        'api_keys_available': {
            'openai': config.api_keys.has_openai(),
            'anthropic': config.api_keys.has_anthropic(),
            'huggingface': config.api_keys.has_huggingface()
        }
    }


# Global configuration instance
_config_instance: Optional[StarterKitConfig] = None


def get_config() -> StarterKitConfig:
    """
    Get the global configuration instance.
    
    Returns:
        StarterKitConfig: The global configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


def reload_config(env_file: Optional[str] = None) -> StarterKitConfig:
    """
    Reload the global configuration instance.
    
    Args:
        env_file: Optional path to .env file
    
    Returns:
        StarterKitConfig: The new configuration instance
    """
    global _config_instance
    _config_instance = load_config(env_file)
    return _config_instance