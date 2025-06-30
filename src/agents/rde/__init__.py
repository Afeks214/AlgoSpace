"""
Regime Detection Engine (RDE) module.

This module provides market regime detection capabilities using a
Transformer + VAE architecture trained on MMD features.
"""

from .engine import RDEComponent

__all__ = ['RDEComponent']