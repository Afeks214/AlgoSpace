"""Multi-Agent Trading Environment for MARL Training.

This module implements the core trading environment for training our MARL agents
using the MAPPO algorithm. It simulates market conditions and manages agent
interactions with historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import gym
from gym import spaces
import logging

from src.core.events import EventBus, Event


logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Represents the current market state."""
    timestamp: pd.Timestamp
    regime_matrix: np.ndarray  # 96×N
    structure_matrix: np.ndarray  # 48×8
    tactical_matrix: np.ndarray  # 60×7
    portfolio_state: Dict[str, float]
    market_info: Dict[str, Any]


@dataclass
class TradingAction:
    """Represents a trading action from the multi-agent system."""
    direction: str  # 'long', 'short', 'neutral'
    size: float  # Position size [0, 1]
    confidence: float  # Action confidence [0, 1]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class MultiAgentTradingEnv(gym.Env):
    """Multi-Agent Trading Environment for MARL training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trading environment.
        
        Args:
            config: Environment configuration including:
                - data_path: Path to historical data
                - initial_capital: Starting capital
                - transaction_cost: Cost per trade (percentage)
                - max_position_size: Maximum position size
                - episode_length: Number of steps per episode
                - agents: List of agent types
        """
        super().__init__()
        self.config = config
        self.event_bus = EventBus()
        
        # Environment parameters
        self.initial_capital = config.get('initial_capital', 100000)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.max_position_size = config.get('max_position_size', 1.0)
        self.episode_length = config.get('episode_length', 1000)
        
        # Agent configuration
        self.agents = config.get('agents', ['regime', 'structure', 'tactical', 'risk'])
        self.n_agents = len(self.agents)
        
        # State and action spaces for each agent
        self._define_spaces()
        
        # Environment state
        self.current_step = 0
        self.episode_count = 0
        self.portfolio = {
            'cash': self.initial_capital,
            'position': 0.0,
            'entry_price': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0
        }
        
        # Data management
        self.data_buffer = deque(maxlen=96)  # Store historical states
        self.current_price = 0.0
        self.prev_price = 0.0
        
        # Performance tracking
        self.episode_returns = []
        self.episode_sharpe = []
        self.episode_max_drawdown = []
        
        logger.info(f"Initialized MultiAgentTradingEnv with {self.n_agents} agents")
    
    def _define_spaces(self):
        """Define observation and action spaces for each agent."""
        # Observation spaces
        self.observation_spaces = {
            'regime': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(96, 10), dtype=np.float32  # Regime matrix
            ),
            'structure': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(48, 8), dtype=np.float32  # Structure matrix
            ),
            'tactical': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(60, 7), dtype=np.float32  # Tactical matrix
            ),
            'risk': spaces.Dict({
                'matrices': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(204, 25), dtype=np.float32  # Combined matrices
                ),
                'portfolio': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(10,), dtype=np.float32  # Portfolio state
                )
            })
        }
        
        # Action spaces
        self.action_spaces = {
            'regime': spaces.Box(
                low=0, high=1,
                shape=(4,), dtype=np.float32  # Regime probabilities
            ),
            'structure': spaces.Box(
                low=-1, high=1,
                shape=(3,), dtype=np.float32  # Direction, size, confidence
            ),
            'tactical': spaces.Box(
                low=0, high=1,
                shape=(5,), dtype=np.float32  # Action probabilities
            ),
            'risk': spaces.Box(
                low=0, high=1,
                shape=(4,), dtype=np.float32  # Risk actions
            )
        }
        
        # Combined spaces for MARL
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment to initial state.
        
        Returns:
            Initial observations for all agents
        """
        self.current_step = 0
        self.episode_count += 1
        
        # Reset portfolio
        self.portfolio = {
            'cash': self.initial_capital,
            'position': 0.0,
            'entry_price': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0
        }
        
        # Reset data buffer
        self.data_buffer.clear()
        
        # Get initial observations
        obs = self._get_observations()
        
        logger.debug(f"Environment reset for episode {self.episode_count}")
        return obs
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]
    ]:
        """Execute one environment step.
        
        Args:
            actions: Dictionary of actions from each agent
            
        Returns:
            observations: Next observations for all agents
            rewards: Rewards for all agents
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1
        
        # Process actions and execute trade
        trade_decision = self._process_actions(actions)
        trade_result = self._execute_trade(trade_decision)
        
        # Update market state
        self._update_market_state()
        
        # Calculate rewards
        rewards = self._calculate_rewards(trade_result)
        
        # Get new observations
        observations = self._get_observations()
        
        # Check if episode is done
        done = self._is_done()
        
        # Compile info
        info = {
            'trade_result': trade_result,
            'portfolio_value': self._get_portfolio_value(),
            'position': self.portfolio['position'],
            'realized_pnl': self.portfolio['realized_pnl'],
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
        
        return observations, rewards, done, info
    
    def _process_actions(self, actions: Dict[str, np.ndarray]) -> TradingAction:
        """Process multi-agent actions into a trading decision.
        
        Args:
            actions: Raw actions from each agent
            
        Returns:
            Aggregated trading action
        """
        # Extract individual agent decisions
        regime_probs = actions['regime']  # [trending, ranging, volatile, transition]
        structure_output = actions['structure']  # [direction, size, confidence]
        tactical_probs = actions['tactical']  # [enter_long, enter_short, exit, hold, reduce]
        risk_actions = actions['risk']  # [allow, modify, exit, block]
        
        # Determine regime
        regime = np.argmax(regime_probs)
        
        # Structure agent decision
        direction_score = structure_output[0]  # -1 to 1
        position_size = structure_output[1]  # 0 to 1
        structure_confidence = structure_output[2]  # 0 to 1
        
        # Tactical agent decision
        tactical_action = np.argmax(tactical_probs)
        tactical_confidence = tactical_probs[tactical_action]
        
        # Risk agent override
        risk_action = np.argmax(risk_actions)
        
        # Aggregate decisions
        if risk_action == 3:  # Block
            return TradingAction('neutral', 0.0, 0.0)
        
        # Determine direction
        if tactical_action == 0:  # Enter long
            direction = 'long'
        elif tactical_action == 1:  # Enter short
            direction = 'short'
        elif tactical_action == 2:  # Exit
            direction = 'neutral'
            position_size = 0.0
        else:  # Hold or reduce
            direction = 'long' if direction_score > 0 else 'short'
            if tactical_action == 4:  # Reduce
                position_size *= 0.5
        
        # Apply risk modifications
        if risk_action == 1:  # Modify size
            position_size *= 0.7
        elif risk_action == 2:  # Force exit
            direction = 'neutral'
            position_size = 0.0
        
        # Calculate final confidence
        confidence = structure_confidence * tactical_confidence
        
        return TradingAction(
            direction=direction,
            size=min(position_size, self.max_position_size),
            confidence=confidence
        )
    
    def _execute_trade(self, action: TradingAction) -> Dict[str, Any]:
        """Execute the trading action.
        
        Args:
            action: Trading action to execute
            
        Returns:
            Trade execution result
        """
        result = {
            'executed': False,
            'type': None,
            'size': 0.0,
            'price': self.current_price,
            'cost': 0.0,
            'pnl': 0.0
        }
        
        current_position = self.portfolio['position']
        
        # Determine trade type and size
        if action.direction == 'neutral' and current_position != 0:
            # Close position
            result['type'] = 'close'
            result['size'] = abs(current_position)
            result['executed'] = True
        elif action.direction == 'long' and action.size > 0:
            if current_position >= 0:
                # Add to long or open long
                result['type'] = 'long'
                result['size'] = action.size
                result['executed'] = True
            else:
                # Close short and open long
                result['type'] = 'reverse_long'
                result['size'] = abs(current_position) + action.size
                result['executed'] = True
        elif action.direction == 'short' and action.size > 0:
            if current_position <= 0:
                # Add to short or open short
                result['type'] = 'short'
                result['size'] = action.size
                result['executed'] = True
            else:
                # Close long and open short
                result['type'] = 'reverse_short'
                result['size'] = abs(current_position) + action.size
                result['executed'] = True
        
        if result['executed']:
            # Calculate transaction cost
            result['cost'] = result['size'] * self.current_price * self.transaction_cost
            
            # Update portfolio
            self._update_portfolio(result)
            
            # Track trade statistics
            self.portfolio['total_trades'] += 1
        
        return result
    
    def _update_portfolio(self, trade_result: Dict[str, Any]):
        """Update portfolio based on trade execution."""
        if trade_result['type'] == 'close':
            # Calculate PnL
            if self.portfolio['position'] > 0:
                pnl = (self.current_price - self.portfolio['entry_price']) * self.portfolio['position']
            else:
                pnl = (self.portfolio['entry_price'] - self.current_price) * abs(self.portfolio['position'])
            
            self.portfolio['realized_pnl'] += pnl - trade_result['cost']
            self.portfolio['cash'] += pnl - trade_result['cost']
            self.portfolio['position'] = 0.0
            self.portfolio['entry_price'] = 0.0
            
            if pnl > 0:
                self.portfolio['winning_trades'] += 1
                
        elif trade_result['type'] in ['long', 'short']:
            # Open or add to position
            new_size = trade_result['size']
            if trade_result['type'] == 'short':
                new_size = -new_size
            
            # Update average entry price
            if self.portfolio['position'] == 0:
                self.portfolio['entry_price'] = self.current_price
            else:
                total_value = (self.portfolio['position'] * self.portfolio['entry_price'] + 
                             new_size * self.current_price)
                self.portfolio['position'] += new_size
                self.portfolio['entry_price'] = total_value / self.portfolio['position']
            
            self.portfolio['cash'] -= trade_result['cost']
            
        elif trade_result['type'] in ['reverse_long', 'reverse_short']:
            # Close and reverse position
            # First close existing
            if self.portfolio['position'] > 0:
                pnl = (self.current_price - self.portfolio['entry_price']) * self.portfolio['position']
            else:
                pnl = (self.portfolio['entry_price'] - self.current_price) * abs(self.portfolio['position'])
            
            self.portfolio['realized_pnl'] += pnl
            self.portfolio['cash'] += pnl
            
            # Then open new position
            new_position = trade_result['size'] - abs(self.portfolio['position'])
            if trade_result['type'] == 'reverse_short':
                new_position = -new_position
            
            self.portfolio['position'] = new_position
            self.portfolio['entry_price'] = self.current_price
            self.portfolio['cash'] -= trade_result['cost']
    
    def _calculate_rewards(self, trade_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate rewards for each agent.
        
        Args:
            trade_result: Result of trade execution
            
        Returns:
            Dictionary of rewards for each agent
        """
        # Base reward components
        pnl_reward = 0.0
        risk_penalty = 0.0
        execution_reward = 0.0
        
        if trade_result['executed']:
            # PnL-based reward (normalized)
            if 'pnl' in trade_result and trade_result['pnl'] != 0:
                pnl_reward = trade_result['pnl'] / self.initial_capital
            
            # Risk penalty for large positions
            position_ratio = abs(self.portfolio['position']) / self.max_position_size
            risk_penalty = -0.1 * (position_ratio ** 2) if position_ratio > 0.7 else 0
            
            # Execution quality reward
            if trade_result['type'] in ['long', 'short']:
                execution_reward = 0.01  # Small reward for taking action
        
        # Calculate Sharpe-based reward
        sharpe_reward = self._calculate_incremental_sharpe()
        
        # Agent-specific rewards
        rewards = {
            'regime': 0.2 * (pnl_reward + sharpe_reward),  # Focus on stability
            'structure': 0.3 * (pnl_reward + sharpe_reward) + 0.1 * execution_reward,
            'tactical': 0.4 * (pnl_reward + execution_reward) + 0.1 * sharpe_reward,
            'risk': -risk_penalty + 0.2 * sharpe_reward  # Focus on risk management
        }
        
        # Add cooperation bonus if all agents agree
        if self._check_agent_agreement(trade_result):
            cooperation_bonus = 0.05
            for agent in rewards:
                rewards[agent] += cooperation_bonus
        
        return rewards
    
    def _calculate_incremental_sharpe(self) -> float:
        """Calculate incremental Sharpe ratio contribution."""
        if len(self.data_buffer) < 20:
            return 0.0
        
        # Get recent returns
        recent_returns = [s['return'] for s in list(self.data_buffer)[-20:]]
        
        if len(recent_returns) > 1:
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
                return np.clip(sharpe / 10, -0.1, 0.1)  # Normalized reward
        
        return 0.0
    
    def _check_agent_agreement(self, trade_result: Dict[str, Any]) -> bool:
        """Check if agents reached consensus on the trade."""
        # Simplified check - would be more sophisticated in practice
        return trade_result['executed'] and trade_result.get('confidence', 0) > 0.7
    
    def _update_market_state(self):
        """Update market state with new data."""
        # This would typically load new market data
        # For now, simulate price movement
        self.prev_price = self.current_price
        self.current_price *= (1 + np.random.normal(0, 0.001))
        
        # Update unrealized PnL
        if self.portfolio['position'] != 0:
            if self.portfolio['position'] > 0:
                self.portfolio['unrealized_pnl'] = (
                    (self.current_price - self.portfolio['entry_price']) * 
                    self.portfolio['position']
                )
            else:
                self.portfolio['unrealized_pnl'] = (
                    (self.portfolio['entry_price'] - self.current_price) * 
                    abs(self.portfolio['position'])
                )
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations for all agents."""
        # Generate synthetic matrices for training
        # In production, these would come from MatrixAssemblers
        
        observations = {
            'regime': np.random.randn(96, 10).astype(np.float32),
            'structure': np.random.randn(48, 8).astype(np.float32),
            'tactical': np.random.randn(60, 7).astype(np.float32),
            'risk': {
                'matrices': np.random.randn(204, 25).astype(np.float32),
                'portfolio': np.array([
                    self.portfolio['cash'] / self.initial_capital,
                    self.portfolio['position'],
                    self.portfolio['unrealized_pnl'] / self.initial_capital,
                    self.portfolio['realized_pnl'] / self.initial_capital,
                    self.portfolio['total_trades'] / 100,
                    self.portfolio['winning_trades'] / max(1, self.portfolio['total_trades']),
                    self.current_price / 100,  # Normalized price
                    self._get_portfolio_value() / self.initial_capital,
                    self.current_step / self.episode_length,
                    0.0  # Placeholder for additional features
                ], dtype=np.float32)
            }
        }
        
        return observations
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return self.portfolio['cash'] + self.portfolio['unrealized_pnl']
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate current Sharpe ratio."""
        if len(self.data_buffer) < 30:
            return 0.0
        
        returns = [s['return'] for s in self.data_buffer]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                return mean_return / std_return * np.sqrt(252)
        
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode should end."""
        # End conditions
        if self.current_step >= self.episode_length:
            return True
        
        # Stop if portfolio is depleted
        if self._get_portfolio_value() < self.initial_capital * 0.5:
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"\n--- Step {self.current_step} ---")
            print(f"Portfolio Value: ${self._get_portfolio_value():,.2f}")
            print(f"Position: {self.portfolio['position']:.4f}")
            print(f"Cash: ${self.portfolio['cash']:,.2f}")
            print(f"Unrealized PnL: ${self.portfolio['unrealized_pnl']:,.2f}")
            print(f"Realized PnL: ${self.portfolio['realized_pnl']:,.2f}")
            print(f"Total Trades: {self.portfolio['total_trades']}")
            print(f"Win Rate: {self.portfolio['winning_trades'] / max(1, self.portfolio['total_trades']):.2%}")
    
    def close(self):
        """Clean up environment resources."""
        logger.info(f"Closing environment after {self.episode_count} episodes")