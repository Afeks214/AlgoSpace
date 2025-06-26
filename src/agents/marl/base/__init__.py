"""
Base classes for MARL agents.

This module provides the foundation for all specialized trading agents
in the Main MARL Core system.
"""

from .base_agent import BaseTradeAgent
from .embedders import (
    SharedEmbedder,
    TemporalAttention,
    SynergyEncoder
)
from .heads import (
    PolicyHead,
    ActionHead,
    ConfidenceHead,
    ReasoningHead
)

__all__ = [
    'BaseTradeAgent',
    'SharedEmbedder',
    'TemporalAttention',
    'SynergyEncoder',
    'PolicyHead',
    'ActionHead',
    'ConfidenceHead',
    'ReasoningHead'
]