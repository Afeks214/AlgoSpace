"""Reward Functions for Multi-Agent Trading System.

This module implements sophisticated reward functions for training MARL agents,
including individual rewards, shared rewards, and multi-objective optimization.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of rewards in the system."""
    PNL = "pnl"
    SHARPE = "sharpe"
    RISK_ADJUSTED = "risk_adjusted"
    EXECUTION_QUALITY = "execution_quality"
    COOPERATION = "cooperation"
    REGIME_ACCURACY = "regime_accuracy"
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    TIMING_PRECISION = "timing_precision"
    RISK_COMPLIANCE = "risk_compliance"


@dataclass
class RewardComponents:
    """Components of agent rewards."""
    base_reward: float
    bonus_reward: float
    penalty: float
    shared_reward: float
    total: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'base_reward': self.base_reward,
            'bonus_reward': self.bonus_reward,
            'penalty': self.penalty,
            'shared_reward': self.shared_reward,
            'total': self.total
        }


class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reward function.
        
        Args:
            config: Reward function configuration
        """
        self.config = config
        self.normalize = config.get('normalize', True)
        self.scale_factor = config.get('scale_factor', 1.0)
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any], 
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute reward.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            info: Additional information
            
        Returns:
            Computed reward
        """
        raise NotImplementedError


class MultiAgentRewardSystem:
    """Comprehensive reward system for MARL trading agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-agent reward system.
        
        Args:
            config: Reward system configuration including:
                - agent_weights: Weight for each agent's reward components
                - shared_weight: Weight for shared rewards
                - risk_penalty_weight: Weight for risk penalties
                - cooperation_bonus: Bonus for agent cooperation
                - normalization: Reward normalization settings
        """
        self.config = config
        
        # Reward weights
        self.agent_weights = config.get('agent_weights', {
            'regime': {'accuracy': 0.4, 'stability': 0.3, 'shared': 0.3},
            'structure': {'directional': 0.4, 'confidence': 0.2, 'shared': 0.4},
            'tactical': {'timing': 0.5, 'execution': 0.2, 'shared': 0.3},
            'risk': {'compliance': 0.6, 'protection': 0.2, 'shared': 0.2}
        })
        
        # Global parameters
        self.shared_weight = config.get('shared_weight', 0.3)
        self.risk_penalty_weight = config.get('risk_penalty_weight', 0.2)
        self.cooperation_bonus = config.get('cooperation_bonus', 0.1)
        
        # Normalization settings
        self.normalize_rewards = config.get('normalization', {}).get('enabled', True)
        self.reward_scale = config.get('normalization', {}).get('scale', 1.0)
        self.clip_rewards = config.get('normalization', {}).get('clip', 10.0)
        
        # Performance tracking
        self.episode_stats = {
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_value': 0.0
        }
        
        # Initialize reward functions
        self._initialize_reward_functions()
        
        logger.info("Initialized MultiAgentRewardSystem")
    
    def _initialize_reward_functions(self):
        """Initialize individual reward functions."""
        self.reward_functions = {
            RewardType.PNL: PnLReward(self.config),
            RewardType.SHARPE: SharpeReward(self.config),
            RewardType.RISK_ADJUSTED: RiskAdjustedReward(self.config),
            RewardType.EXECUTION_QUALITY: ExecutionQualityReward(self.config),
            RewardType.COOPERATION: CooperationReward(self.config),
            RewardType.REGIME_ACCURACY: RegimeAccuracyReward(self.config),
            RewardType.DIRECTIONAL_ACCURACY: DirectionalAccuracyReward(self.config),
            RewardType.TIMING_PRECISION: TimingPrecisionReward(self.config),
            RewardType.RISK_COMPLIANCE: RiskComplianceReward(self.config)
        }
    
    def compute_rewards(self, state: Dict[str, Any], actions: Dict[str, np.ndarray],
                       next_state: Dict[str, Any], trade_result: Dict[str, Any],
                       portfolio_state: Dict[str, float]) -> Dict[str, float]:
        """Compute rewards for all agents.
        
        Args:
            state: Current market state
            actions: Actions taken by agents
            next_state: Next market state
            trade_result: Result of trade execution
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary of rewards for each agent
        """
        # Update episode statistics
        self._update_episode_stats(trade_result, portfolio_state)
        
        # Compute individual agent rewards
        agent_rewards = {}
        
        # Regime Agent Reward
        agent_rewards['regime'] = self._compute_regime_reward(
            state, actions['regime'], next_state, trade_result
        )
        
        # Structure Agent Reward
        agent_rewards['structure'] = self._compute_structure_reward(
            state, actions['structure'], next_state, trade_result
        )
        
        # Tactical Agent Reward
        agent_rewards['tactical'] = self._compute_tactical_reward(
            state, actions['tactical'], next_state, trade_result
        )
        
        # Risk Agent Reward
        agent_rewards['risk'] = self._compute_risk_reward(
            state, actions['risk'], next_state, trade_result, portfolio_state
        )
        
        # Compute shared rewards
        shared_reward = self._compute_shared_reward(
            state, actions, next_state, trade_result, portfolio_state
        )
        
        # Add shared component to each agent
        for agent_name in agent_rewards:
            weight = self.agent_weights[agent_name].get('shared', 0.3)
            agent_rewards[agent_name] = (
                (1 - weight) * agent_rewards[agent_name] + 
                weight * shared_reward
            )
        
        # Apply cooperation bonus if agents cooperated well
        if self._check_cooperation(actions, trade_result):
            for agent_name in agent_rewards:
                agent_rewards[agent_name] += self.cooperation_bonus
        
        # Normalize and clip rewards
        if self.normalize_rewards:
            agent_rewards = self._normalize_rewards(agent_rewards)
        
        return agent_rewards
    
    def _compute_regime_reward(self, state: Dict[str, Any], action: np.ndarray,
                             next_state: Dict[str, Any], trade_result: Dict[str, Any]) -> float:
        """Compute reward for regime agent.
        
        Focuses on:
        - Regime classification accuracy
        - Stability of regime predictions
        - Early detection of regime changes
        """
        reward_components = RewardComponents(
            base_reward=0.0,
            bonus_reward=0.0,
            penalty=0.0,
            shared_reward=0.0,
            total=0.0
        )
        
        # Regime accuracy reward
        regime_accuracy = self.reward_functions[RewardType.REGIME_ACCURACY].compute(
            state, {'regime_probs': action}, next_state, trade_result
        )
        reward_components.base_reward = regime_accuracy * 0.4
        
        # Stability bonus (penalize frequent regime changes)
        if 'prev_regime' in state and 'current_regime' in next_state:
            if state['prev_regime'] == next_state['current_regime']:
                reward_components.bonus_reward += 0.1
            else:
                # Only penalize if change was not beneficial
                if trade_result.get('pnl', 0) < 0:
                    reward_components.penalty -= 0.05
        
        # Early detection bonus
        if 'regime_change_detected' in trade_result and trade_result['regime_change_detected']:
            if trade_result.get('early_detection', False):
                reward_components.bonus_reward += 0.2
        
        reward_components.total = (
            reward_components.base_reward + 
            reward_components.bonus_reward + 
            reward_components.penalty
        )
        
        return reward_components.total
    
    def _compute_structure_reward(self, state: Dict[str, Any], action: np.ndarray,
                                 next_state: Dict[str, Any], trade_result: Dict[str, Any]) -> float:
        """Compute reward for structure agent.
        
        Focuses on:
        - Directional bias accuracy
        - Position sizing appropriateness
        - Trend identification quality
        """
        reward_components = RewardComponents(
            base_reward=0.0,
            bonus_reward=0.0,
            penalty=0.0,
            shared_reward=0.0,
            total=0.0
        )
        
        # Directional accuracy
        direction_score = action[0]  # -1 to 1
        position_size = action[1]    # 0 to 1
        confidence = action[2]       # 0 to 1
        
        if trade_result.get('executed', False):
            # Reward based on directional accuracy
            pnl = trade_result.get('pnl', 0)
            if pnl > 0 and direction_score > 0:  # Correct long direction
                reward_components.base_reward = 0.3 * confidence
            elif pnl > 0 and direction_score < 0:  # Correct short direction
                reward_components.base_reward = 0.3 * confidence
            elif pnl < 0:  # Wrong direction
                reward_components.penalty = -0.2 * confidence
            
            # Position sizing reward
            optimal_size = self._calculate_optimal_position_size(state, trade_result)
            size_diff = abs(position_size - optimal_size)
            reward_components.bonus_reward = 0.1 * (1 - size_diff)
        
        # Trend identification quality
        if 'trend_strength' in state:
            trend_agreement = abs(direction_score) * state['trend_strength']
            reward_components.bonus_reward += 0.1 * trend_agreement
        
        reward_components.total = (
            reward_components.base_reward + 
            reward_components.bonus_reward + 
            reward_components.penalty
        )
        
        return reward_components.total
    
    def _compute_tactical_reward(self, state: Dict[str, Any], action: np.ndarray,
                               next_state: Dict[str, Any], trade_result: Dict[str, Any]) -> float:
        """Compute reward for tactical agent.
        
        Focuses on:
        - Entry/exit timing precision
        - Execution quality
        - Short-term price movement capture
        """
        reward_components = RewardComponents(
            base_reward=0.0,
            bonus_reward=0.0,
            penalty=0.0,
            shared_reward=0.0,
            total=0.0
        )
        
        # Timing precision reward
        timing_reward = self.reward_functions[RewardType.TIMING_PRECISION].compute(
            state, {'action_probs': action}, next_state, trade_result
        )
        reward_components.base_reward = timing_reward * 0.5
        
        # Execution quality reward
        if trade_result.get('executed', False):
            execution_reward = self.reward_functions[RewardType.EXECUTION_QUALITY].compute(
                state, {'action': action}, next_state, trade_result
            )
            reward_components.bonus_reward = execution_reward * 0.2
            
            # Short-term capture bonus
            if 'price_movement' in trade_result:
                movement = trade_result['price_movement']
                if abs(movement) > 0.001:  # Significant movement
                    capture_quality = trade_result.get('capture_ratio', 0)
                    reward_components.bonus_reward += 0.1 * capture_quality
        
        # Penalty for excessive trading
        if 'trade_frequency' in state and state['trade_frequency'] > 10:
            reward_components.penalty = -0.05
        
        reward_components.total = (
            reward_components.base_reward + 
            reward_components.bonus_reward + 
            reward_components.penalty
        )
        
        return reward_components.total
    
    def _compute_risk_reward(self, state: Dict[str, Any], action: np.ndarray,
                           next_state: Dict[str, Any], trade_result: Dict[str, Any],
                           portfolio_state: Dict[str, float]) -> float:
        """Compute reward for risk agent.
        
        Focuses on:
        - Risk constraint compliance
        - Drawdown prevention
        - Portfolio protection
        """
        reward_components = RewardComponents(
            base_reward=0.0,
            bonus_reward=0.0,
            penalty=0.0,
            shared_reward=0.0,
            total=0.0
        )
        
        # Risk compliance reward
        compliance_reward = self.reward_functions[RewardType.RISK_COMPLIANCE].compute(
            state, {'risk_action': action}, next_state, 
            {'portfolio': portfolio_state, 'trade': trade_result}
        )
        reward_components.base_reward = compliance_reward * 0.6
        
        # Drawdown prevention
        current_value = portfolio_state.get('total_value', 0)
        if current_value > self.episode_stats['peak_value']:
            self.episode_stats['peak_value'] = current_value
            self.episode_stats['current_drawdown'] = 0
        else:
            drawdown = (self.episode_stats['peak_value'] - current_value) / self.episode_stats['peak_value']
            self.episode_stats['current_drawdown'] = drawdown
            
            # Penalty for excessive drawdown
            if drawdown > 0.1:  # 10% drawdown
                reward_components.penalty = -0.3 * (drawdown - 0.1)
            
            # Bonus for preventing further drawdown
            if 'prev_drawdown' in state and drawdown < state['prev_drawdown']:
                reward_components.bonus_reward = 0.1
        
        # Portfolio protection bonus
        risk_action = np.argmax(action)
        if risk_action in [1, 2]:  # Modify or exit actions
            if trade_result.get('risk_prevented', False):
                reward_components.bonus_reward += 0.2
        
        reward_components.total = (
            reward_components.base_reward + 
            reward_components.bonus_reward + 
            reward_components.penalty
        )
        
        return reward_components.total
    
    def _compute_shared_reward(self, state: Dict[str, Any], actions: Dict[str, np.ndarray],
                             next_state: Dict[str, Any], trade_result: Dict[str, Any],
                             portfolio_state: Dict[str, float]) -> float:
        """Compute shared reward for all agents.
        
        Focuses on:
        - Overall portfolio performance
        - Risk-adjusted returns
        - System-wide objectives
        """
        shared_reward = 0.0
        
        # PnL component
        pnl_reward = self.reward_functions[RewardType.PNL].compute(
            state, actions, next_state, {'trade_result': trade_result, 'portfolio': portfolio_state}
        )
        shared_reward += pnl_reward * 0.4
        
        # Sharpe ratio component
        sharpe_reward = self.reward_functions[RewardType.SHARPE].compute(
            state, actions, next_state, {'portfolio': portfolio_state}
        )
        shared_reward += sharpe_reward * 0.3
        
        # Risk-adjusted return component
        risk_adjusted_reward = self.reward_functions[RewardType.RISK_ADJUSTED].compute(
            state, actions, next_state, {'trade_result': trade_result, 'portfolio': portfolio_state}
        )
        shared_reward += risk_adjusted_reward * 0.3
        
        return shared_reward
    
    def _check_cooperation(self, actions: Dict[str, np.ndarray], 
                         trade_result: Dict[str, Any]) -> bool:
        """Check if agents cooperated well.
        
        Args:
            actions: Agent actions
            trade_result: Trade execution result
            
        Returns:
            Whether agents cooperated effectively
        """
        # Check action consensus
        regime_confidence = np.max(actions['regime'])
        structure_confidence = actions['structure'][2]
        tactical_confidence = np.max(actions['tactical'])
        
        # High confidence from all agents
        if all([regime_confidence > 0.7, structure_confidence > 0.7, tactical_confidence > 0.7]):
            # And successful trade
            if trade_result.get('executed', False) and trade_result.get('pnl', 0) > 0:
                return True
        
        # Risk agent prevented loss
        if np.argmax(actions['risk']) in [1, 2, 3]:  # Modify, exit, or block
            if trade_result.get('risk_prevented', False):
                return True
        
        return False
    
    def _calculate_optimal_position_size(self, state: Dict[str, Any], 
                                       trade_result: Dict[str, Any]) -> float:
        """Calculate optimal position size based on Kelly criterion.
        
        Args:
            state: Market state
            trade_result: Trade result
            
        Returns:
            Optimal position size [0, 1]
        """
        # Simplified Kelly criterion
        win_rate = self.episode_stats['winning_trades'] / max(1, self.episode_stats['total_trades'])
        avg_win = 0.02  # Assumed average win
        avg_loss = 0.01  # Assumed average loss
        
        if win_rate > 0 and avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            # Apply Kelly fraction with safety factor
            optimal_size = max(0, min(1, kelly_fraction * 0.25))
        else:
            optimal_size = 0.1  # Default conservative size
        
        # Adjust for market volatility
        if 'volatility' in state:
            volatility_factor = 1 / (1 + state['volatility'])
            optimal_size *= volatility_factor
        
        return optimal_size
    
    def _update_episode_stats(self, trade_result: Dict[str, Any], 
                            portfolio_state: Dict[str, float]):
        """Update episode statistics.
        
        Args:
            trade_result: Trade execution result
            portfolio_state: Current portfolio state
        """
        if trade_result.get('executed', False):
            self.episode_stats['total_trades'] += 1
            
            pnl = trade_result.get('pnl', 0)
            self.episode_stats['total_pnl'] += pnl
            
            if pnl > 0:
                self.episode_stats['winning_trades'] += 1
        
        # Update drawdown tracking
        current_value = portfolio_state.get('total_value', 0)
        if current_value > self.episode_stats['peak_value']:
            self.episode_stats['peak_value'] = current_value
        
        drawdown = (self.episode_stats['peak_value'] - current_value) / self.episode_stats['peak_value']
        self.episode_stats['max_drawdown'] = max(self.episode_stats['max_drawdown'], drawdown)
    
    def _normalize_rewards(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Normalize and clip rewards.
        
        Args:
            rewards: Raw rewards
            
        Returns:
            Normalized rewards
        """
        # Calculate mean and std
        reward_values = list(rewards.values())
        mean_reward = np.mean(reward_values)
        std_reward = np.std(reward_values) + 1e-8
        
        # Normalize
        normalized = {}
        for agent, reward in rewards.items():
            normalized_reward = (reward - mean_reward) / std_reward
            # Scale and clip
            normalized_reward *= self.reward_scale
            normalized_reward = np.clip(normalized_reward, -self.clip_rewards, self.clip_rewards)
            normalized[agent] = normalized_reward
        
        return normalized
    
    def reset_episode_stats(self):
        """Reset episode statistics."""
        self.episode_stats = {
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_value': 0.0
        }


# Individual Reward Function Implementations

class PnLReward(RewardFunction):
    """Profit and Loss based reward."""
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute PnL reward."""
        trade_result = info.get('trade_result', {})
        portfolio = info.get('portfolio', {})
        
        if trade_result.get('executed', False):
            pnl = trade_result.get('pnl', 0)
            initial_capital = portfolio.get('initial_capital', 100000)
            
            # Normalize by initial capital
            normalized_pnl = pnl / initial_capital
            
            if self.normalize:
                # Apply tanh for bounded output
                return np.tanh(normalized_pnl * self.scale_factor)
            else:
                return normalized_pnl * self.scale_factor
        
        return 0.0


class SharpeReward(RewardFunction):
    """Sharpe ratio based reward."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.returns_buffer = []
        self.buffer_size = config.get('buffer_size', 100)
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute Sharpe-based reward."""
        portfolio = info.get('portfolio', {})
        
        # Calculate return
        current_value = portfolio.get('total_value', 0)
        prev_value = state.get('portfolio_value', current_value)
        
        if prev_value > 0:
            ret = (current_value - prev_value) / prev_value
            self.returns_buffer.append(ret)
            
            # Maintain buffer size
            if len(self.returns_buffer) > self.buffer_size:
                self.returns_buffer.pop(0)
            
            # Calculate Sharpe ratio
            if len(self.returns_buffer) > 10:
                mean_return = np.mean(self.returns_buffer)
                std_return = np.std(self.returns_buffer) + 1e-8
                
                # Annualized Sharpe ratio
                sharpe = mean_return / std_return * np.sqrt(252 * 78)  # 78 5-min bars per day
                
                # Convert to reward
                sharpe_reward = np.tanh(sharpe / 2) * self.scale_factor
                return sharpe_reward
        
        return 0.0


class RiskAdjustedReward(RewardFunction):
    """Risk-adjusted return reward."""
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute risk-adjusted reward."""
        trade_result = info.get('trade_result', {)}
        portfolio = info.get('portfolio', {})
        
        if trade_result.get('executed', False):
            pnl = trade_result.get('pnl', 0)
            position_size = abs(portfolio.get('position', 0))
            max_position = portfolio.get('max_position', 1.0)
            
            # Risk factor based on position size
            risk_factor = position_size / max_position
            
            # Adjust reward by risk taken
            if pnl > 0:
                # Reward efficiency - higher reward for lower risk
                risk_adjusted = pnl * (2 - risk_factor)
            else:
                # Penalize losses more for higher risk
                risk_adjusted = pnl * (1 + risk_factor)
            
            # Normalize
            initial_capital = portfolio.get('initial_capital', 100000)
            normalized = risk_adjusted / initial_capital
            
            return np.tanh(normalized * self.scale_factor)
        
        return 0.0


class ExecutionQualityReward(RewardFunction):
    """Execution quality reward."""
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute execution quality reward."""
        trade_result = info.get('trade_result', {})
        
        if trade_result.get('executed', False):
            # Slippage penalty
            expected_price = trade_result.get('expected_price', 0)
            actual_price = trade_result.get('price', expected_price)
            
            if expected_price > 0:
                slippage = abs(actual_price - expected_price) / expected_price
                slippage_penalty = -slippage * 10
            else:
                slippage_penalty = 0
            
            # Transaction cost consideration
            transaction_cost = trade_result.get('cost', 0)
            cost_penalty = -transaction_cost / 1000  # Normalized
            
            # Timing bonus
            if trade_result.get('good_timing', False):
                timing_bonus = 0.1
            else:
                timing_bonus = 0
            
            total_reward = slippage_penalty + cost_penalty + timing_bonus
            return total_reward * self.scale_factor
        
        return 0.0


class CooperationReward(RewardFunction):
    """Multi-agent cooperation reward."""
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute cooperation reward."""
        # This is computed at system level
        return 0.0


class RegimeAccuracyReward(RewardFunction):
    """Regime classification accuracy reward."""
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute regime accuracy reward."""
        regime_probs = action.get('regime_probs', [])
        
        if len(regime_probs) > 0 and 'true_regime' in next_state:
            predicted_regime = np.argmax(regime_probs)
            true_regime = next_state['true_regime']
            
            if predicted_regime == true_regime:
                # Reward based on confidence
                confidence = regime_probs[predicted_regime]
                return confidence * self.scale_factor
            else:
                # Penalty for wrong prediction weighted by confidence
                confidence = regime_probs[predicted_regime]
                return -0.5 * confidence * self.scale_factor
        
        return 0.0


class DirectionalAccuracyReward(RewardFunction):
    """Directional prediction accuracy reward."""
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute directional accuracy reward."""
        direction = action.get('direction', 0)
        
        if 'price_change' in info:
            price_change = info['price_change']
            
            # Correct direction prediction
            if (direction > 0 and price_change > 0) or (direction < 0 and price_change < 0):
                return abs(direction) * self.scale_factor
            # Wrong direction
            elif (direction > 0 and price_change < 0) or (direction < 0 and price_change > 0):
                return -abs(direction) * 0.5 * self.scale_factor
        
        return 0.0


class TimingPrecisionReward(RewardFunction):
    """Entry/exit timing precision reward."""
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute timing precision reward."""
        action_probs = action.get('action_probs', [])
        trade_result = info.get('trade_result', {)}
        
        if trade_result.get('executed', False) and len(action_probs) > 0:
            action_taken = trade_result.get('action_type', 'hold')
            
            # Map action types to indices
            action_map = {
                'enter_long': 0,
                'enter_short': 1, 
                'exit': 2,
                'hold': 3,
                'reduce': 4
            }
            
            if action_taken in action_map:
                action_idx = action_map[action_taken]
                confidence = action_probs[action_idx] if action_idx < len(action_probs) else 0
                
                # Reward based on outcome
                if trade_result.get('good_timing', False):
                    return confidence * self.scale_factor
                else:
                    return -confidence * 0.3 * self.scale_factor
        
        return 0.0


class RiskComplianceReward(RewardFunction):
    """Risk constraint compliance reward."""
    
    def compute(self, state: Dict[str, Any], action: Dict[str, Any],
                next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute risk compliance reward."""
        portfolio = info.get('portfolio', {)}
        risk_action = action.get('risk_action', [])
        
        if len(risk_action) > 0:
            action_type = np.argmax(risk_action)
            
            # Check risk constraints
            position = abs(portfolio.get('position', 0))
            max_position = portfolio.get('max_position', 1.0)
            drawdown = portfolio.get('current_drawdown', 0)
            max_drawdown = 0.15  # 15% max drawdown
            
            violations = 0
            if position > max_position:
                violations += 1
            if drawdown > max_drawdown:
                violations += 1
            
            # Reward for preventing violations
            if action_type in [1, 2, 3] and violations > 0:  # Risk mitigation actions
                return 0.5 * self.scale_factor
            # Penalty for not acting when needed
            elif action_type == 0 and violations > 0:  # Allow action when risky
                return -0.3 * violations * self.scale_factor
            # Reward for allowing good trades
            elif action_type == 0 and violations == 0:
                if info.get('trade_result', {)}.get('pnl', 0) > 0:
                    return 0.2 * self.scale_factor
        
        return 0.0