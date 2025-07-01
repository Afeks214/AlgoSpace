"""
AlgoSpace Indicators Module

This module provides a comprehensive collection of technical indicators and market analysis tools
for algorithmic trading and quantitative finance. The module includes:

Core Components:
- BaseIndicator: Abstract base class for all indicators
- IndicatorEngine: Main engine for managing and executing indicators
- IndicatorRegistry: Registry system for indicator management

Market Structure Indicators:
- FVGDetector: Fair Value Gap detection and analysis
- LVNAnalyzer: Low Volume Node analysis with market profile
- MLMICalculator: Multi-Level Market Impact calculator
- MMDFeatureExtractor: Market Microstructure Dynamics feature extraction
- NWRQKCalculator: Neural Weighted Risk-Quantum Kernel calculator

Each indicator follows the BaseIndicator interface and can be used independently
or through the IndicatorEngine for coordinated analysis.
"""

from .base import BaseIndicator, IndicatorRegistry
from .engine import IndicatorEngine
from .fvg import FVGDetector
from .lvn import LVNAnalyzer, MarketProfile
from .mlmi import MLMICalculator, MLMIDataFast
from .mmd import MMDFeatureExtractor
from .nwrqk import NWRQKCalculator

__all__ = [
    # Core components
    "BaseIndicator",
    "IndicatorRegistry", 
    "IndicatorEngine",
    
    # Market structure indicators
    "FVGDetector",
    "LVNAnalyzer",
    "MarketProfile",
    "MLMICalculator",
    "MLMIDataFast",
    "MMDFeatureExtractor",
    "NWRQKCalculator",
]

# Module version
__version__ = "1.0.0"