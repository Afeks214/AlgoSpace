"""Components module for the AlgoSpace trading system.

This module contains reusable components like the BarGenerator
that can be used across different parts of the system.
"""

from .bar_generator import BarGenerator

__all__ = [
    "BarGenerator",
]