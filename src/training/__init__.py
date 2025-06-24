"""
Training Infrastructure for MARL Agents.

This module provides the complete training pipeline for the Multi-Agent
Reinforcement Learning system, including:
- Multi-agent trading environments
- MAPPO training implementation
- Reward systems and shaping
- Evaluation and monitoring tools
"""

from .environments.trading_env import TradingEnvironment, MultiAgentTradingEnv
from .trainers.mappo_trainer import MAPPOTrainer
from .rewards.reward_functions import RewardCalculator
from .evaluation.evaluator import ModelEvaluator

__all__ = [
    'TradingEnvironment',
    'MultiAgentTradingEnv',
    'MAPPOTrainer',
    'RewardCalculator',
    'ModelEvaluator'
]