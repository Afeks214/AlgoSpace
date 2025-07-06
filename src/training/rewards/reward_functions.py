"""
Reward functions for multi-agent reinforcement learning.

Implements both individual and shared reward components to encourage
effective trading and coordination between agents.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class TradeMetrics:
    """Metrics for a completed trade."""
    entry_price: float
    exit_price: float
    size: float
    direction: str  # 'long' or 'short'
    duration: int  # bars held
    pnl: float
    return_pct: float
    max_drawdown: float
    transaction_costs: float


class AgentRewardFunction(ABC):
    """
    Abstract base class for agent-specific reward functions.
    """
    
    @abstractmethod
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """Calculate reward for a single step."""
        pass
    
    @abstractmethod
    def calculate_episode_bonus(
        self,
        episode_info: Dict[str, Any]
    ) -> float:
        """Calculate bonus reward at episode end."""
        pass


class StructureAnalyzerReward(AgentRewardFunction):
    """
    Reward function for Structure Analyzer agent.
    
    Focuses on:
    - Long-term profitability
    - Trend identification accuracy
    - Risk-adjusted returns
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward function.
        
        Args:
            config: Reward configuration
        """
        self.config = config
        self.sharpe_target = config.get('sharpe_target', 1.5)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.15)
        self.trend_accuracy_weight = config.get('trend_accuracy_weight', 0.2)
        
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate step reward for structure analyzer.
        
        Rewards:
        - Positive P&L with emphasis on consistency
        - Correct trend identification
        - Risk management
        """
        reward = 0.0
        
        # Base P&L reward
        position = state.get('position', {})
        if position.get('side') != 'flat':
            pnl = position.get('unrealized_pnl', 0.0)
            # Scale by position duration to reward holding winners
            duration = info.get('position_duration', 1)
            reward += pnl * (1 + 0.1 * np.log1p(duration))
        
        # Trend accuracy bonus
        if action[0] != 0:  # Not passing
            market_trend = self._calculate_market_trend(state)
            action_direction = 'long' if action[0] == 1 else 'short'
            
            if (market_trend > 0 and action_direction == 'long') or \
               (market_trend < 0 and action_direction == 'short'):
                reward += self.trend_accuracy_weight * abs(market_trend)
            else:
                reward -= self.trend_accuracy_weight * abs(market_trend) * 0.5
        
        # Risk penalty
        if 'portfolio_stats' in info:
            current_drawdown = info['portfolio_stats'].get('drawdown', 0)
            if current_drawdown > self.max_drawdown_limit:
                reward -= 0.5 * (current_drawdown - self.max_drawdown_limit)
        
        return reward
    
    def calculate_episode_bonus(self, episode_info: Dict[str, Any]) -> float:
        """
        Calculate episode-end bonus.
        
        Rewards:
        - Achieving target Sharpe ratio
        - Low maximum drawdown
        - Consistent profitability
        """
        bonus = 0.0
        
        # Sharpe ratio bonus
        sharpe = episode_info.get('sharpe_ratio', 0)
        if sharpe > self.sharpe_target:
            bonus += 1.0 * (sharpe - self.sharpe_target)
        
        # Drawdown bonus
        max_dd = episode_info.get('max_drawdown', 0)
        if max_dd < self.max_drawdown_limit:
            bonus += 0.5 * (self.max_drawdown_limit - max_dd)
        
        # Consistency bonus (low volatility of returns)
        return_volatility = episode_info.get('return_volatility', 1.0)
        bonus += 0.3 / (1 + return_volatility)
        
        return bonus
    
    def _calculate_market_trend(self, state: Dict[str, Any]) -> float:
        """Calculate current market trend strength."""
        market_matrix = state.get('market_matrix', np.zeros((48, 8)))
        if len(market_matrix) < 20:
            return 0.0
        
        # Simple trend: compare recent prices to older prices
        recent_price = np.mean(market_matrix[-5:, 3])  # Recent close prices
        older_price = np.mean(market_matrix[-20:-15, 3])  # Older close prices
        
        if older_price == 0:
            return 0.0
        
        trend = (recent_price - older_price) / older_price
        return np.clip(trend, -0.1, 0.1)  # Clip to reasonable range


class ShortTermTacticianReward(AgentRewardFunction):
    """
    Reward function for Short-term Tactician agent.
    
    Focuses on:
    - Execution quality
    - Entry/exit timing
    - Quick profitable trades
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward function.
        
        Args:
            config: Reward configuration
        """
        self.config = config
        self.execution_quality_weight = config.get('execution_quality_weight', 0.3)
        self.timing_precision_weight = config.get('timing_precision_weight', 0.3)
        self.quick_profit_bonus = config.get('quick_profit_bonus', 0.5)
        self.max_hold_bars = config.get('max_hold_bars', 20)
        
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate step reward for tactical agent.
        
        Rewards:
        - Good entry/exit prices
        - Quick profitable trades
        - Low slippage
        """
        reward = 0.0
        
        # Execution quality reward
        if action[0] != 0:  # Taking action
            execution_quality = self._calculate_execution_quality(state, action)
            reward += self.execution_quality_weight * execution_quality
        
        # Quick profit bonus
        position = state.get('position', {)}
        if position.get('side') != 'flat':
            pnl = position.get('unrealized_pnl', 0.0)
            duration = info.get('position_duration', 0)
            
            if pnl > 0 and duration < self.max_hold_bars:
                # Bonus for quick profits
                speed_factor = 1 - (duration / self.max_hold_bars)
                reward += self.quick_profit_bonus * pnl * speed_factor
            elif duration > self.max_hold_bars:
                # Penalty for holding too long
                reward -= 0.1 * (duration - self.max_hold_bars) / self.max_hold_bars
        
        # Timing precision
        if 'price_after_entry' in info and action[0] != 0:
            # Reward if price moved favorably after entry
            entry_price = info.get('entry_price', 0)
            price_after = info['price_after_entry']
            
            if entry_price > 0:
                if action[0] == 1:  # Long
                    favorable_move = (price_after - entry_price) / entry_price
                else:  # Short
                    favorable_move = (entry_price - price_after) / entry_price
                
                reward += self.timing_precision_weight * favorable_move
        
        return reward
    
    def calculate_episode_bonus(self, episode_info: Dict[str, Any]) -> float:
        """
        Calculate episode-end bonus.
        
        Rewards:
        - High win rate
        - Low average slippage
        - Consistent execution
        """
        bonus = 0.0
        
        # Win rate bonus
        win_rate = episode_info.get('win_rate', 0.5)
        if win_rate > 0.6:
            bonus += 0.5 * (win_rate - 0.6)
        
        # Low slippage bonus
        avg_slippage = episode_info.get('avg_slippage', 0.001)
        if avg_slippage < 0.0005:  # Less than 5 bps
            bonus += 0.3 * (0.0005 - avg_slippage) / 0.0005
        
        # Trade frequency bonus (not too many, not too few)
        trades_per_episode = episode_info.get('num_trades', 0)
        optimal_trades = 10
        trade_diff = abs(trades_per_episode - optimal_trades) / optimal_trades
        bonus += 0.2 * (1 - min(trade_diff, 1))
        
        return bonus
    
    def _calculate_execution_quality(self, state: Dict[str, Any], action: np.ndarray) -> float:
        """Calculate execution quality score."""
        # Simplified execution quality based on current market conditions
        market_matrix = state.get('market_matrix', np.zeros((60, 7)))
        
        if len(market_matrix) < 5:
            return 0.0
        
        # Check recent volatility
        recent_data = market_matrix[-5:]
        volatility = np.std(recent_data[:, 3]) / np.mean(recent_data[:, 3])
        
        # Check spread
        recent_spread = np.mean((recent_data[:, 1] - recent_data[:, 2]) / recent_data[:, 3])
        
        # Lower volatility and tighter spreads = better execution
        quality = 1 / (1 + volatility + recent_spread * 10)
        
        return np.clip(quality, 0, 1)


class MidFrequencyArbitrageurReward(AgentRewardFunction):
    """
    Reward function for Mid-frequency Arbitrageur agent.
    
    Focuses on:
    - Cross-timeframe inefficiency capture
    - Market-neutral profits
    - High efficiency ratio
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward function.
        
        Args:
            config: Reward configuration
        """
        self.config = config
        self.inefficiency_capture_weight = config.get('inefficiency_capture_weight', 0.4)
        self.efficiency_ratio_target = config.get('efficiency_ratio_target', 0.7)
        self.market_neutral_bonus = config.get('market_neutral_bonus', 0.3)
        
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate step reward for arbitrageur.
        
        Rewards:
        - Capturing price inefficiencies
        - Market-neutral profits
        - Low correlation with market
        """
        reward = 0.0
        
        # Inefficiency capture reward
        if 'inefficiency_score' in info and action[0] != 0:
            inefficiency = info['inefficiency_score']
            capture_success = info.get('capture_success', 0)
            reward += self.inefficiency_capture_weight * inefficiency * capture_success
        
        # Market-neutral bonus
        position = state.get('position', {)}
        if position.get('side') != 'flat':
            pnl = position.get('unrealized_pnl', 0.0)
            market_return = info.get('market_return', 0.0)
            
            # Reward profits that are uncorrelated with market
            if abs(market_return) > 0.001:
                correlation = pnl / market_return if market_return != 0 else 0
                if correlation < 0:  # Negative correlation is good
                    reward += self.market_neutral_bonus * abs(pnl)
                else:
                    reward += self.market_neutral_bonus * pnl * (1 - min(correlation, 1))
        
        # Efficiency reward (profit per unit of risk)
        if 'trade_risk' in info and info['trade_risk'] > 0:
            efficiency = pnl / info['trade_risk']
            if efficiency > self.efficiency_ratio_target:
                reward += 0.2 * (efficiency - self.efficiency_ratio_target)
        
        return reward
    
    def calculate_episode_bonus(self, episode_info: Dict[str, Any]) -> float:
        """
        Calculate episode-end bonus.
        
        Rewards:
        - High efficiency ratio
        - Low market correlation
        - Consistent arbitrage profits
        """
        bonus = 0.0
        
        # Efficiency ratio bonus
        efficiency_ratio = episode_info.get('efficiency_ratio', 0)
        if efficiency_ratio > self.efficiency_ratio_target:
            bonus += 0.5 * (efficiency_ratio - self.efficiency_ratio_target)
        
        # Market correlation penalty
        market_correlation = abs(episode_info.get('market_correlation', 0))
        bonus += 0.3 * (1 - market_correlation)
        
        # Consistency bonus
        profit_consistency = episode_info.get('profit_consistency', 0)
        bonus += 0.2 * profit_consistency
        
        return bonus


class RewardCalculator:
    """
    Main reward calculator for multi-agent system.
    
    Combines individual and shared rewards for all agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration
        """
        self.config = config
        self.shared_weight = config.get('shared_weight', 0.3)
        self.individual_weight = config.get('individual_weight', 0.7)
        
        # Initialize agent-specific reward functions
        self.agent_rewards = {
            'structure_analyzer': StructureAnalyzerReward(
                config.get('structure_analyzer', {})
            ),
            'short_term_tactician': ShortTermTacticianReward(
                config.get('short_term_tactician', {})
            ),
            'mid_freq_arbitrageur': MidFrequencyArbitrageurReward(
                config.get('mid_freq_arbitrageur', {})
            )
        }
        
        # Portfolio-level metrics
        self.portfolio_stats = {
            'total_pnl': 0.0,
            'num_trades': 0,
            'winning_trades': 0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_value': 1.0
        }
        
        logger.info(f"Initialized reward calculator shared_weight={self.shared_weight} agents={list(self.agent_rewards.keys()}"))
        )
    
    def calculate_rewards(
        self,
        states: Dict[str, Dict[str, Any]],
        actions: Dict[str, np.ndarray],
        next_states: Dict[str, Dict[str, Any]],
        infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate rewards for all agents.
        
        Args:
            states: Current states for each agent
            actions: Actions taken by each agent
            next_states: Next states for each agent
            infos: Additional information for each agent
            
        Returns:
            Dictionary of rewards for each agent
        """
        rewards = {}
        
        # Calculate individual rewards
        individual_rewards = {}
        for agent_name, reward_fn in self.agent_rewards.items():
            if agent_name in states:
                individual_rewards[agent_name] = reward_fn.calculate_reward(
                    states[agent_name],
                    actions[agent_name],
                    next_states[agent_name],
                    infos.get(agent_name, {})
                )
        
        # Calculate shared rewards
        shared_reward = self._calculate_shared_reward(states, actions, infos)
        
        # Combine rewards
        for agent_name in individual_rewards:
            rewards[agent_name] = (
                self.individual_weight * individual_rewards[agent_name] +
                self.shared_weight * shared_reward
            )
        
        # Update portfolio statistics
        self._update_portfolio_stats(infos)
        
        return rewards
    
    def _calculate_shared_reward(
        self,
        states: Dict[str, Dict[str, Any]],
        actions: Dict[str, np.ndarray],
        infos: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Calculate shared reward component.
        
        Rewards:
        - Portfolio-level performance
        - Agent coordination
        - Risk management
        """
        shared_reward = 0.0
        
        # Portfolio P&L
        portfolio_pnl = sum(
            info.get('position', {}).get('unrealized_pnl', 0)
            for info in infos.values()
        )
        shared_reward += portfolio_pnl
        
        # Coordination bonus
        action_types = [int(action[0]) for action in actions.values()]
        if len(set(action_types)) == 1 and action_types[0] != 0:
            # All agents agree on direction
            shared_reward += 0.1
        
        # Risk management
        total_position_size = sum(
            abs(info.get('position', {)}.get('size', 0))
            for info in infos.values()
        )
        max_position = self.config.get('max_total_position', 2.0)
        
        if total_position_size > max_position:
            # Penalty for excessive risk
            shared_reward -= 0.5 * (total_position_size - max_position)
        
        # Drawdown penalty
        if self.portfolio_stats['current_drawdown'] > 0.1:
            shared_reward -= self.portfolio_stats['current_drawdown'] * 2
        
        return shared_reward
    
    def _update_portfolio_stats(self, infos: Dict[str, Dict[str, Any]]):
        """Update portfolio-level statistics."""
        # Calculate current portfolio value
        current_value = 1.0 + self.portfolio_stats['total_pnl']
        
        # Update peak value and drawdown
        if current_value > self.portfolio_stats['peak_value']:
            self.portfolio_stats['peak_value'] = current_value
            self.portfolio_stats['current_drawdown'] = 0.0
        else:
            drawdown = (self.portfolio_stats['peak_value'] - current_value) / self.portfolio_stats['peak_value']
            self.portfolio_stats['current_drawdown'] = drawdown
            self.portfolio_stats['max_drawdown'] = max(
                self.portfolio_stats['max_drawdown'],
                drawdown
            )
        
        # Update trade statistics
        for info in infos.values():
            if 'trade_completed' in info and info['trade_completed']:
                self.portfolio_stats['num_trades'] += 1
                if info.get('trade_pnl', 0) > 0:
                    self.portfolio_stats['winning_trades'] += 1
                self.portfolio_stats['total_pnl'] += info.get('trade_pnl', 0)
    
    def get_episode_bonuses(self) -> Dict[str, float]:
        """
        Calculate episode-end bonuses for all agents.
        
        Returns:
            Dictionary of bonuses for each agent
        """
        episode_info = {
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self.portfolio_stats['max_drawdown'],
            'win_rate': self._calculate_win_rate(),
            'efficiency_ratio': self._calculate_efficiency_ratio(),
            'return_volatility': self._calculate_return_volatility(),
            'market_correlation': 0.0,  # Placeholder
            'profit_consistency': self._calculate_profit_consistency()
        }
        
        bonuses = {}
        for agent_name, reward_fn in self.agent_rewards.items():
            bonuses[agent_name] = reward_fn.calculate_episode_bonus(episode_info)
        
        return bonuses
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate portfolio Sharpe ratio."""
        # Simplified calculation
        if self.portfolio_stats['num_trades'] == 0:
            return 0.0
        
        avg_return = self.portfolio_stats['total_pnl'] / max(self.portfolio_stats['num_trades'], 1)
        # Placeholder for return std
        return avg_return / 0.02 if avg_return > 0 else 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if self.portfolio_stats['num_trades'] == 0:
            return 0.5
        
        return self.portfolio_stats['winning_trades'] / self.portfolio_stats['num_trades']
    
    def _calculate_efficiency_ratio(self) -> float:
        """Calculate efficiency ratio (profit factor)."""
        # Placeholder implementation
        return 0.7
    
    def _calculate_return_volatility(self) -> float:
        """Calculate return volatility."""
        # Placeholder implementation
        return 0.15
    
    def _calculate_profit_consistency(self) -> float:
        """Calculate profit consistency score."""
        # Placeholder implementation
        return 0.6
    
    def reset_episode_stats(self):
        """Reset episode statistics."""
        self.portfolio_stats = {
            'total_pnl': 0.0,
            'num_trades': 0,
            'winning_trades': 0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_value': 1.0
        }