"""
Agents module for AlgoSpace trading system.

This module contains all agent implementations including:
- SynergyDetector: Hard-coded pattern detection (Gate 1)
- MARL Agents: AI-based trading agents (Gate 2)
"""

from .synergy import SynergyDetector

__all__ = [
    'SynergyDetector'
]