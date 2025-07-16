"""
AI Agent Framework for the Ultimate Agentic StarterKit.

This package provides specialized AI agents for parsing, code generation, testing,
and advisory tasks, all coordinated through a common interface.
"""

from agents.base_agent import BaseAgent
from agents.parser_agent import ParserAgent
from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent
from agents.advisor_agent import AdvisorAgent
from agents.factory import AgentFactory, create_agent

__all__ = [
    "BaseAgent",
    "ParserAgent", 
    "CoderAgent",
    "TesterAgent",
    "AdvisorAgent",
    "AgentFactory",
    "create_agent"
]