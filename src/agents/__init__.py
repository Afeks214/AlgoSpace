"""
Agents module for AlgoSpace trading system.

This module contains all agent implementations including:
- SynergyDetector: Hard-coded pattern detection (Gate 1)
- MARL Agents: AI-based trading agents (Gate 2)
"""

from .synergy import SynergyDetector
from .main_core import MainMARLCoreComponent
from .mrms import MRMSComponent
from .rde import RDEComponent
from .rde.engine import RDEComponent as RDEEngine
from .mrms.engine import MRMSComponent as MRMSEngine
from .main_core.engine import MainMARLCoreComponent as MainMARLCore

__all__ = [
    "SynergyDetector",
    "MainMARLCoreComponent",
    "MRMSComponent",
    "RDEComponent",
    "RDEEngine",
    "MRMSEngine",
    "MainMARLCore",
]