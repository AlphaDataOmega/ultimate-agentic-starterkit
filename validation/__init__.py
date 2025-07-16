"""
Validation and Quality Assurance System for Ultimate Agentic StarterKit.

This module provides comprehensive validation and quality assurance systems with
confidence scoring, visual testing integration, and automated quality gates.
"""

from .confidence import ConfidenceScorer, ConfidenceFactors
from .visual_validator import VisualTestingValidator
from .test_runner import TestExecutionFramework
from .quality_gates import QualityGateSystem, QualityGate, QualityGateStatus
from .performance_monitor import PerformanceMonitor
from .validator import ValidationOrchestrator

__all__ = [
    'ConfidenceScorer',
    'ConfidenceFactors',
    'VisualTestingValidator',
    'TestExecutionFramework',
    'QualityGateSystem',
    'QualityGate',
    'QualityGateStatus',
    'PerformanceMonitor',
    'ValidationOrchestrator',
]

__version__ = '1.0.0'