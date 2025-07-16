"""
Ultimate Agentic StarterKit

A comprehensive framework for building AI-powered agent systems with proper
infrastructure, logging, configuration management, and workflow orchestration.
"""

__version__ = "0.1.0"
__author__ = "AlphaDataOmega"
__email__ = "info@alphadataomega.com"

from .core.models import (
    TaskStatus,
    ConfidenceLevel,
    AgentType,
    ProjectTask,
    ProjectSpecification,
    AgentResult,
    WorkflowState,
)

__all__ = [
    "TaskStatus",
    "ConfidenceLevel", 
    "AgentType",
    "ProjectTask",
    "ProjectSpecification",
    "AgentResult",
    "WorkflowState",
]