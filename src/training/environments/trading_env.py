"""
Multi-Agent Trading Environment for MARL training.

This environment simulates a trading scenario where multiple agents
collaborate to make trading decisions based on different market views
and timeframes.
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

import structlog

logger = structlog.get_logger()


@dataclass
class MarketState:
    """Current market state."""
    timestamp: datetime
    price: float
    volume: float
    spread: float
    volatility: float
    trend: float
    regime: np.ndarray  # 8-dim regime vector


@dataclass
class Position:
    """Trading position."""
    side: str  # 'long', 'short', or 'flat'
    size: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class TradingEnvironment(gym.Env):
    """
    Single-agent view of the trading environment.
    
    This is used as a base for each agent's perspective in the
    multi-agent environment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trading environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config
        self.window_size = config.get('window_size', 100)
        self.features = config.get('features', 8)
        self.max_position_size = config.get('max_position_size', 1.0)
        self.transaction_cost = config.get('transaction_cost', 0.0002)  # 2 bps
        self.slippage = config.get('slippage', 0.0001)  # 1 bp
        
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            'market_matrix': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self.features),
                dtype=np.float32
            ),
            'regime_vector': spaces.Box(
                low=-1,
                high=1,
                shape=(8,),
                dtype=np.float32
            ),
            'position': spaces.Box(
                low=-1,
                high=1,
                shape=(3,),  # [side, size, pnl]
                dtype=np.float32
            ),
            'synergy_active': spaces.Discrete(2)  # 0 or 1
        })
        
        # Action space: [action_type, size, timing]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([2, 1, 5]),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.position = Position('flat', 0.0, 0.0, datetime.now())
        self.market_state = None
        self.data_buffer = None
        
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.position = Position('flat', 0.0, 0.0, datetime.now())
        
        # Initialize with zeros (will be populated by data pipeline)
        observation = {
            'market_matrix': np.zeros((self.window_size, self.features), dtype=np.float32),
            'regime_vector': np.zeros(8, dtype=np.float32),
            'position': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'synergy_active': 0
        }
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute action and return results.
        
        Args:
            action: Agent action [action_type, size, timing]
            
        Returns:
            observation: Next observation
            reward: Step reward
            done: Episode complete flag
            info: Additional information
        """
        # Parse action
        action_type = int(action[0])  # 0: pass, 1: long, 2: short
        size = float(action[1])
        timing = int(action[2])  # Bars to wait before execution
        
        # Execute trade if not passing
        if action_type != 0 and timing == 0:
            self._execute_trade(action_type, size)
        
        # Update market state (placeholder - will be fed by data pipeline)
        self._update_market_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update position PnL
        self._update_position()
        
        # Get next observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'position': self.position,
            'market_state': self.market_state,
            'step': self.current_step
        }
        
        self.current_step += 1
        
        return observation, reward, done, info
    
    def _execute_trade(self, action_type: int, size: float):
        """Execute trading action."""
        current_price = self.market_state.price if self.market_state else 100.0
        
        # Apply slippage
        if action_type == 1:  # Long
            execution_price = current_price * (1 + self.slippage)
            new_side = 'long'
        else:  # Short
            execution_price = current_price * (1 - self.slippage)
            new_side = 'short'
        
        # Close existing position if opposite side
        if self.position.side != 'flat' and self.position.side != new_side:
            self._close_position()
        
        # Open new position
        self.position = Position(
            side=new_side,
            size=min(size, self.max_position_size),
            entry_price=execution_price,
            entry_time=datetime.now()
        )
        
        # Apply transaction cost
        cost = self.position.size * execution_price * self.transaction_cost
        self.position.realized_pnl -= cost
    
    def _close_position(self):
        """Close current position."""
        if self.position.side == 'flat':
            return
        
        current_price = self.market_state.price if self.market_state else 100.0
        
        # Calculate PnL
        if self.position.side == 'long':
            pnl = (current_price - self.position.entry_price) * self.position.size
        else:  # short
            pnl = (self.position.entry_price - current_price) * self.position.size
        
        # Apply transaction cost
        cost = self.position.size * current_price * self.transaction_cost
        
        self.position.realized_pnl += pnl - cost
        self.position = Position('flat', 0.0, 0.0, datetime.now())
    
    def _update_market_state(self):
        """Update market state from data feed."""
        # Placeholder - will be connected to data pipeline
        if self.market_state is None:
            self.market_state = MarketState(
                timestamp=datetime.now(),
                price=100.0,
                volume=1000.0,
                spread=0.01,
                volatility=0.15,
                trend=0.0,
                regime=np.zeros(8)
            )
    
    def _update_position(self):
        """Update position unrealized PnL."""
        if self.position.side == 'flat':
            return
        
        current_price = self.market_state.price
        
        if self.position.side == 'long':
            self.position.unrealized_pnl = (
                (current_price - self.position.entry_price) * self.position.size
            )
        else:  # short
            self.position.unrealized_pnl = (
                (self.position.entry_price - current_price) * self.position.size
            )
    
    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        # Base reward is change in total PnL
        total_pnl = self.position.realized_pnl + self.position.unrealized_pnl
        reward = total_pnl
        
        # Add risk penalty for large positions
        position_penalty = -0.01 * (self.position.size ** 2)
        reward += position_penalty
        
        return reward
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Position encoding
        if self.position.side == 'flat':
            position_encoding = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.position.side == 'long':
            position_encoding = np.array([1.0, self.position.size, self.position.unrealized_pnl], dtype=np.float32)
        else:  # short
            position_encoding = np.array([-1.0, self.position.size, self.position.unrealized_pnl], dtype=np.float32)
        
        observation = {
            'market_matrix': np.zeros((self.window_size, self.features), dtype=np.float32),  # Placeholder
            'regime_vector': self.market_state.regime if self.market_state else np.zeros(8, dtype=np.float32),
            'position': position_encoding,
            'synergy_active': 0  # Placeholder
        }
        
        return observation
    
    def _is_done(self) -> bool:
        """Check if episode is complete."""
        # Episode ends after max steps or if position is liquidated
        max_steps = self.config.get('max_steps', 1000)
        
        if self.current_step >= max_steps:
            return True
        
        # Check for liquidation (e.g., -10% loss)
        total_pnl = self.position.realized_pnl + self.position.unrealized_pnl
        if total_pnl < -0.1:
            return True
        
        return False


class MultiAgentTradingEnv:
    """
    Multi-agent trading environment for MARL training.
    
    Coordinates multiple trading agents with different views and
    responsibilities in the market.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-agent environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config
        self.n_agents = config.get('n_agents', 3)
        
        # Agent-specific configurations
        self.agent_configs = {
            'structure_analyzer': {
                'window_size': 48,
                'features': 8,
                'timeframe': '30m'
            },
            'short_term_tactician': {
                'window_size': 60,
                'features': 7,
                'timeframe': '5m'
            },
            'mid_freq_arbitrageur': {
                'window_size': 100,
                'features': 15,
                'timeframe': 'combined'
            }
        }
        
        # Create individual environments for each agent
        self.agents = {}
        for agent_name, agent_config in self.agent_configs.items():
            env_config = {**config, **agent_config}
            self.agents[agent_name] = TradingEnvironment(env_config)
        
        # Shared market state
        self.market_state = None
        self.synergy_context = None
        
        # Episode tracking
        self.current_step = 0
        self.episode_data = []
        
        logger.info(f"Initialized multi-agent trading environment n_agents={self.n_agents} agents={list(self.agents.keys())}")
    
    def reset(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Reset all agent environments.
        
        Returns:
            Dictionary of observations for each agent
        """
        self.current_step = 0
        self.episode_data = []
        
        observations = {}
        for agent_name, env in self.agents.items():
            observations[agent_name] = env.reset()
        
        return observations
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]]
    ]:
        """
        Execute actions for all agents.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            observations: Next observations for each agent
            rewards: Rewards for each agent
            dones: Done flags for each agent
            infos: Additional info for each agent
        """
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # Execute actions for each agent
        for agent_name, action in actions.items():
            if agent_name in self.agents:
                obs, reward, done, info = self.agents[agent_name].step(action)
                observations[agent_name] = obs
                rewards[agent_name] = reward
                dones[agent_name] = done
                infos[agent_name] = info
        
        # Apply coordination rewards
        coordination_rewards = self._calculate_coordination_rewards(actions, infos)
        for agent_name in rewards:
            rewards[agent_name] += coordination_rewards.get(agent_name, 0.0)
        
        # Store episode data
        self.episode_data.append({
            'step': self.current_step,
            'actions': actions,
            'rewards': rewards,
            'observations': observations
        })
        
        self.current_step += 1
        
        # All agents share the same done flag
        all_done = any(dones.values())
        for agent_name in dones:
            dones[agent_name] = all_done
        
        return observations, rewards, dones, infos
    
    def _calculate_coordination_rewards(
        self,
        actions: Dict[str, np.ndarray],
        infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate rewards for agent coordination.
        
        Args:
            actions: Agent actions
            infos: Agent information
            
        Returns:
            Coordination rewards for each agent
        """
        coordination_rewards = {}
        
        # Extract action types
        action_types = {}
        for agent_name, action in actions.items():
            action_types[agent_name] = int(action[0])
        
        # Reward agreement on direction
        long_count = sum(1 for a in action_types.values() if a == 1)
        short_count = sum(1 for a in action_types.values() if a == 2)
        
        if long_count >= 2 or short_count >= 2:
            # Majority agreement bonus
            agreement_bonus = 0.01
            for agent_name in actions:
                if (action_types[agent_name] == 1 and long_count >= 2) or \
                   (action_types[agent_name] == 2 and short_count >= 2):
                    coordination_rewards[agent_name] = agreement_bonus
                else:
                    coordination_rewards[agent_name] = 0.0
        else:
            # No clear agreement
            for agent_name in actions:
                coordination_rewards[agent_name] = 0.0
        
        return coordination_rewards
    
    def set_market_data(self, market_data: pd.DataFrame):
        """
        Set market data for the environment.
        
        Args:
            market_data: Historical market data
        """
        # This will be connected to the data pipeline
        pass
    
    def set_synergy_context(self, synergy_context: Dict[str, Any]):
        """
        Set synergy detection context.
        
        Args:
            synergy_context: Synergy detection results
        """
        self.synergy_context = synergy_context
        
        # Update agent observations
        for env in self.agents.values():
            env.synergy_context = synergy_context
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the completed episode.
        
        Returns:
            Episode statistics
        """
        if not self.episode_data:
            return {}
        
        # Calculate episode statistics
        total_rewards = {agent: 0.0 for agent in self.agents}
        action_counts = {agent: {'pass': 0, 'long': 0, 'short': 0} for agent in self.agents}
        
        for step_data in self.episode_data:
            for agent_name, reward in step_data['rewards'].items():
                total_rewards[agent_name] += reward
            
            for agent_name, action in step_data['actions'].items():
                action_type = int(action[0])
                if action_type == 0:
                    action_counts[agent_name]['pass'] += 1
                elif action_type == 1:
                    action_counts[agent_name]['long'] += 1
                else:
                    action_counts[agent_name]['short'] += 1
        
        return {
            'total_rewards': total_rewards,
            'action_counts': action_counts,
            'episode_length': len(self.episode_data),
            'average_rewards': {
                agent: total / len(self.episode_data) 
                for agent, total in total_rewards.items()
            }
        }