"""
Main MARL Core Module.

This module provides the unified intelligence system for trade decision-making,
implementing a two-gate flow with MC Dropout consensus and integrated risk management.
"""

from .engine import MainMARLCoreComponent

__all__ = ['MainMARLCoreComponent']