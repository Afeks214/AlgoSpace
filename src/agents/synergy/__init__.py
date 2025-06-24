"""
SynergyDetector module for AlgoSpace trading system.

This module implements the hard-coded strategy pattern detection
that serves as Gate 1 in the two-gate MARL system.
"""

from .detector import SynergyDetector
from .base import Signal, SynergyPattern, BasePatternDetector, BaseSynergyDetector
from .patterns import MLMIPatternDetector, NWRQKPatternDetector, FVGPatternDetector
from .sequence import SignalSequence, CooldownTracker

__all__ = [
    'SynergyDetector',
    'Signal',
    'SynergyPattern',
    'BasePatternDetector',
    'BaseSynergyDetector',
    'MLMIPatternDetector',
    'NWRQKPatternDetector',
    'FVGPatternDetector',
    'SignalSequence',
    'CooldownTracker'
]