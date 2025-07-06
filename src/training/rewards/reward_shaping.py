"""
Reward shaping for improved learning.

Provides additional reward signals to guide agent learning
without changing the optimal policy.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
from collections import deque

import structlog

logger = structlog.get_logger()


class RewardShaper:
    """
    Reward shaping to accelerate learning.
    
    Implements potential-based reward shaping and other
    techniques to guide exploration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward shaper.
        
        Args:
            config: Shaping configuration
        """
        self.config = config
        self.gamma = config.get('gamma', 0.99)
        
        # Shaping methods
        self.use_potential_shaping = config.get('use_potential_shaping', True)
        self.use_curiosity_bonus = config.get('use_curiosity_bonus', True)
        self.use_exploration_bonus = config.get('use_exploration_bonus', True)
        
        # Potential function parameters
        self.potential_scale = config.get('potential_scale', 0.1)
        
        # Curiosity parameters
        self.curiosity_weight = config.get('curiosity_weight', 0.05)
        self.state_visit_counts = {}
        self.state_history = deque(maxlen=10000)
        
        # Exploration parameters
        self.exploration_weight = config.get('exploration_weight', 0.1)
        self.exploration_decay = config.get('exploration_decay', 0.999)
        self.exploration_epsilon = 1.0
        
        # Cooperation bonus
        self.cooperation_weight = config.get('cooperation_weight', 0.1)
        
        logger.info(f"Initialized reward shaper potential_shaping={self.use_potential_shaping} curiosity_bonus={self.use_curiosity_bonus} exploration_bonus={self.use_exploration_bonus}")
    
    def shape_rewards(
        self,
        rewards: Dict[str, float],
        states: Dict[str, Dict[str, Any]],
        next_states: Dict[str, Dict[str, Any]],
        actions: Dict[str, np.ndarray],
        episode_step: int
    ) -> Dict[str, float]:
        """
        Apply reward shaping to agent rewards.
        
        Args:
            rewards: Original rewards
            states: Current states
            next_states: Next states
            actions: Actions taken
            episode_step: Current step in episode
            
        Returns:
            Shaped rewards
        """
        shaped_rewards = rewards.copy()
        
        # Apply potential-based shaping
        if self.use_potential_shaping:
            potential_bonuses = self._calculate_potential_bonuses(states, next_states)
            for agent, bonus in potential_bonuses.items():
                if agent in shaped_rewards:
                    shaped_rewards[agent] += self.potential_scale * bonus
        
        # Apply curiosity bonus
        if self.use_curiosity_bonus:
            curiosity_bonuses = self._calculate_curiosity_bonuses(states)
            for agent, bonus in curiosity_bonuses.items():
                if agent in shaped_rewards:
                    shaped_rewards[agent] += self.curiosity_weight * bonus
        
        # Apply exploration bonus
        if self.use_exploration_bonus:
            exploration_bonuses = self._calculate_exploration_bonuses(actions, episode_step)
            for agent, bonus in exploration_bonuses.items():
                if agent in shaped_rewards:
                    shaped_rewards[agent] += self.exploration_weight * bonus
        
        # Apply cooperation bonus
        cooperation_bonus = self._calculate_cooperation_bonus(actions)
        for agent in shaped_rewards:
            shaped_rewards[agent] += self.cooperation_weight * cooperation_bonus
        
        return shaped_rewards
    
    def _calculate_potential_bonuses(
        self,
        states: Dict[str, Dict[str, Any]],
        next_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate potential-based shaping bonuses.
        
        F(s,a,s') = γΦ(s') - Φ(s)
        
        where Φ is the potential function.
        """
        bonuses = {}
        
        for agent in states:
            if agent in next_states:
                # Calculate potentials
                current_potential = self._potential_function(states[agent])
                next_potential = self._potential_function(next_states[agent])
                
                # Potential-based bonus
                bonus = self.gamma * next_potential - current_potential
                bonuses[agent] = bonus
        
        return bonuses
    
    def _potential_function(self, state: Dict[str, Any]) -> float:
        """
        Potential function for a state.
        
        Higher potential for states closer to profitable positions.
        """
        potential = 0.0
        
        # Position-based potential
        position = state.get('position', {})
        if isinstance(position, dict):
            # Potential based on unrealized P&L
            unrealized_pnl = position.get('unrealized_pnl', 0.0)
            potential += np.tanh(unrealized_pnl * 10)  # Bounded between -1 and 1
            
            # Potential for having a position (encourages taking action)
            if position.get('side', 'flat') != 'flat':
                potential += 0.2
        
        # Market state potential
        if 'synergy_active' in state and state['synergy_active']:
            # Higher potential when synergy is detected
            potential += 0.5
        
        # Regime-based potential
        regime_vector = state.get('regime_vector', np.zeros(8))
        if isinstance(regime_vector, np.ndarray) and len(regime_vector) >= 2:
            # Higher potential in trending regimes
            trend_strength = abs(regime_vector[0])  # Assuming first component is trend
            potential += 0.3 * trend_strength
        
        return potential
    
    def _calculate_curiosity_bonuses(
        self,
        states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate curiosity bonuses for visiting novel states.
        """
        bonuses = {}
        
        for agent, state in states.items():
            # Create state hash
            state_hash = self._hash_state(state)
            
            # Count visits
            if state_hash not in self.state_visit_counts:
                self.state_visit_counts[state_hash] = 0
            
            self.state_visit_counts[state_hash] += 1
            visits = self.state_visit_counts[state_hash]
            
            # Curiosity bonus inversely proportional to visit count
            curiosity = 1.0 / np.sqrt(visits)
            bonuses[agent] = curiosity
            
            # Add to history
            self.state_history.append(state_hash)
        
        # Clean up old states
        if len(self.state_visit_counts) > 50000:
            self._cleanup_state_counts()
        
        return bonuses
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create a hash representation of a state."""
        # Simplified hashing - in practice, use more sophisticated method
        key_components = []
        
        # Position info
        position = state.get('position', {)}
        if isinstance(position, dict):
            key_components.append(position.get('side', 'flat'))
            key_components.append(str(round(position.get('size', 0), 2)))
        
        # Synergy status
        key_components.append(str(state.get('synergy_active', 0)))
        
        # Discretized market state
        if 'market_matrix' in state and hasattr(state['market_matrix'], 'shape'):
            # Use recent price level
            try:
                recent_price = np.mean(state['market_matrix'][-5:, 3])
                price_bucket = int(recent_price / 10) * 10  # Bucket by 10
                key_components.append(str(price_bucket))
            except:
                pass
        
        return "_".join(key_components)
    
    def _cleanup_state_counts(self):
        """Remove rarely visited states to manage memory."""
        # Keep only frequently visited states
        threshold = 2
        self.state_visit_counts = {
            state: count
            for state, count in self.state_visit_counts.items()
            if count >= threshold
        }
    
    def _calculate_exploration_bonuses(
        self,
        actions: Dict[str, np.ndarray],
        episode_step: int
    ) -> Dict[str, float]:
        """
        Calculate exploration bonuses to encourage diverse actions.
        """
        bonuses = {}
        
        for agent, action in actions.items():
            # Bonus for non-zero actions (not passing)
            if action[0] != 0:
                bonuses[agent] = self.exploration_epsilon
            else:
                bonuses[agent] = 0.0
            
            # Additional bonus for using timing delays
            if action[2] > 0:  # Timing component
                bonuses[agent] += 0.1 * self.exploration_epsilon
        
        # Decay exploration over time
        if episode_step % 100 == 0:
            self.exploration_epsilon *= self.exploration_decay
            self.exploration_epsilon = max(self.exploration_epsilon, 0.01)
        
        return bonuses
    
    def _calculate_cooperation_bonus(self, actions: Dict[str, np.ndarray]) -> float:
        """
        Calculate bonus for agent cooperation.
        """
        action_types = [int(action[0]) for action in actions.values()]
        
        # Count action agreement
        long_count = sum(1 for a in action_types if a == 1)
        short_count = sum(1 for a in action_types if a == 2)
        pass_count = sum(1 for a in action_types if a == 0)
        
        total_agents = len(action_types)
        
        # Strong agreement bonus
        if long_count >= total_agents - 1 or short_count >= total_agents - 1:
            return 1.0
        # Moderate agreement
        elif long_count >= total_agents // 2 or short_count >= total_agents // 2:
            return 0.5
        # Disagreement penalty
        elif long_count > 0 and short_count > 0:
            return -0.5
        else:
            return 0.0
    
    def add_curriculum_bonus(
        self,
        rewards: Dict[str, float],
        difficulty_level: int,
        success_rate: float
    ) -> Dict[str, float]:
        """
        Add curriculum learning bonuses.
        
        Args:
            rewards: Current rewards
            difficulty_level: Current curriculum difficulty (0-10)
            success_rate: Recent success rate (0-1)
            
        Returns:
            Rewards with curriculum bonuses
        """
        curriculum_bonus = 0.0
        
        # Bonus for succeeding at higher difficulties
        if success_rate > 0.7:
            curriculum_bonus = 0.1 * difficulty_level
        # Penalty for failing at lower difficulties
        elif success_rate < 0.3 and difficulty_level < 5:
            curriculum_bonus = -0.05 * (5 - difficulty_level)
        
        # Apply bonus to all agents
        shaped_rewards = rewards.copy()
        for agent in shaped_rewards:
            shaped_rewards[agent] += curriculum_bonus
        
        return shaped_rewards
    
    def get_shaping_stats(self) -> Dict[str, Any]:
        """Get statistics about reward shaping."""
        return {
            'exploration_epsilon': self.exploration_epsilon,
            'unique_states_visited': len(self.state_visit_counts),
            'total_state_visits': sum(self.state_visit_counts.values()),
            'avg_visits_per_state': np.mean(list(self.state_visit_counts.values())) if self.state_visit_counts else 0
        }
    
    def reset_episode(self):
        """Reset episode-specific tracking."""
        # Optionally reset some tracking between episodes
        pass
    
    def update_shaping_parameters(self, metrics: Dict[str, float]):
        """
        Adapt shaping parameters based on training progress.
        
        Args:
            metrics: Current training metrics
        """
        # Reduce exploration if learning is progressing well
        if metrics.get('average_reward', 0) > 0:
            self.exploration_weight *= 0.99
        
        # Reduce curiosity bonus over time
        if metrics.get('episodes_completed', 0) % 100 == 0:
            self.curiosity_weight *= 0.95
            self.curiosity_weight = max(self.curiosity_weight, 0.01)