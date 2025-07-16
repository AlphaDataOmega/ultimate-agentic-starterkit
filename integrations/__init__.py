"""
External integrations package for the Ultimate Agentic StarterKit.

This package provides integrations with external services and tools:
- Claude Code VS Code extension
- Git management with validation gates
- Ollama local models
- Integration factory and management
"""

from .claude_code import ClaudeCodeIntegration
from .git_manager import GitManager
from .ollama_client import OllamaClient
from .factory import IntegrationFactory

__all__ = [
    'ClaudeCodeIntegration',
    'GitManager',
    'OllamaClient',
    'IntegrationFactory'
]