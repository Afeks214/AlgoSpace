"""
MARL training algorithms.
"""

from .mappo_trainer import MAPPOTrainer
from .base_trainer import BaseTrainer

__all__ = [
    'MAPPOTrainer',
    'BaseTrainer'
]