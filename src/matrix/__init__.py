"""
Matrix Assembler Module for AlgoSpace Trading System

This module provides the bridge between raw indicator features and 
AI-ready neural network inputs. It implements three specialized matrix
assemblers that transform point-in-time features into rolling time-series
matrices optimized for different aspects of market analysis.

Components:
- MatrixAssembler30m: Long-term market structure (48 x 8 matrix)
- MatrixAssembler5m: Short-term price action (60 x 7 matrix)  
- MatrixAssemblerRegime: Market regime detection (96 x N matrix)

Each assembler maintains a rolling window of normalized features with
thread-safe access and efficient memory management.
"""

from .base import BaseMatrixAssembler
from .assembler_30m import MatrixAssembler30m
from .assembler_5m import MatrixAssembler5m
from .assembler_regime import MatrixAssemblerRegime

__all__ = [
    'BaseMatrixAssembler',
    'MatrixAssembler30m', 
    'MatrixAssembler5m',
    'MatrixAssemblerRegime'
]