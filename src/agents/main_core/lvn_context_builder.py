"""
LVN Context Builder for preparing data for the LVN Embedder.

This module handles the transformation of raw LVN indicator data into
rich context objects suitable for the advanced LVN embedder.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging

from .lvn_embedder import LVNContext

logger = logging.getLogger(__name__)


class LVNContextBuilder:
    """
    Builds rich LVN context from indicator data and market state.
    
    Handles data transformation, feature engineering, and context preparation
    for the LVN embedder.
    """
    
    def __init__(
        self,
        max_levels: int = 10,
        price_history_length: int = 20,
        interaction_lookback: int = 50,
        device: Optional[torch.device] = None
    ):
        self.max_levels = max_levels
        self.price_history_length = price_history_length
        self.interaction_lookback = interaction_lookback
        self.device = device or torch.device('cpu')
        
        # Price history buffer
        self.price_history = deque(maxlen=price_history_length)
        
        # Interaction tracking
        self.interaction_buffer = deque(maxlen=interaction_lookback)
        
        # LVN history for temporal analysis
        self.lvn_history = deque(maxlen=100)
        
    def build_context(
        self,
        lvn_data: Dict[str, Any],
        market_state: Dict[str, Any],
        additional_features: Optional[Dict[str, Any]] = None
    ) -> LVNContext:
        """
        Build LVN context from raw data.
        
        Args:
            lvn_data: Dictionary containing LVN indicator outputs
            market_state: Current market state information
            additional_features: Optional additional features
            
        Returns:
            LVNContext object ready for embedder
        """
        # Extract LVN levels
        all_levels = lvn_data.get('all_levels', [])
        current_price = market_state.get('current_price', 0.0)
        
        # Update price history
        self.price_history.append(current_price)
        
        # Filter and sort levels by relevance
        relevant_levels = self._filter_relevant_levels(
            all_levels,
            current_price,
            max_levels=self.max_levels
        )
        
        # Extract level features
        if relevant_levels:
            prices = torch.tensor(
                [level['price'] for level in relevant_levels],
                dtype=torch.float32,
                device=self.device
            )
            strengths = torch.tensor(
                [level['strength'] for level in relevant_levels],
                dtype=torch.float32,
                device=self.device
            )
            distances = torch.tensor(
                [level['distance'] for level in relevant_levels],
                dtype=torch.float32,
                device=self.device
            )
            volumes = torch.tensor(
                [level.get('volume', 0.0) for level in relevant_levels],
                dtype=torch.float32,
                device=self.device
            )
        else:
            # No levels found - create dummy data
            prices = torch.zeros(1, dtype=torch.float32, device=self.device)
            strengths = torch.zeros(1, dtype=torch.float32, device=self.device)
            distances = torch.ones(1, dtype=torch.float32, device=self.device) * 1e6
            volumes = torch.zeros(1, dtype=torch.float32, device=self.device)
            
        # Prepare price history tensor
        price_history_tensor = torch.tensor(
            list(self.price_history),
            dtype=torch.float32,
            device=self.device
        )
        
        # Pad price history if needed
        if len(price_history_tensor) < self.price_history_length:
            pad_size = self.price_history_length - len(price_history_tensor)
            price_history_tensor = torch.cat([
                torch.full((pad_size,), current_price, device=self.device),
                price_history_tensor
            ])
            
        # Build interaction history
        interaction_history = self._build_interaction_history(
            relevant_levels,
            current_price
        )
        
        # Create context
        context = LVNContext(
            prices=prices,
            strengths=strengths,
            distances=distances,
            volumes=volumes,
            current_price=current_price,
            price_history=price_history_tensor,
            interaction_history=interaction_history
        )
        
        # Update histories
        self._update_histories(relevant_levels, current_price)
        
        return context
        
    def _filter_relevant_levels(
        self,
        all_levels: List[Dict[str, Any]],
        current_price: float,
        max_levels: int
    ) -> List[Dict[str, Any]]:
        """
        Filter and prioritize relevant LVN levels.
        
        Prioritizes by:
        1. Proximity to current price
        2. Strength score
        3. Recent interaction history
        """
        if not all_levels:
            return []
            
        # Score each level
        scored_levels = []
        for level in all_levels:
            score = self._calculate_relevance_score(level, current_price)
            scored_levels.append((score, level))
            
        # Sort by score (descending)
        scored_levels.sort(key=lambda x: x[0], reverse=True)
        
        # Take top levels
        relevant_levels = [level for _, level in scored_levels[:max_levels]]
        
        # Sort by price for consistency
        relevant_levels.sort(key=lambda x: x['price'])
        
        return relevant_levels
        
    def _calculate_relevance_score(
        self,
        level: Dict[str, Any],
        current_price: float
    ) -> float:
        """Calculate relevance score for an LVN level."""
        # Base score from strength
        score = level['strength'] / 100.0  # Normalize to [0, 1]
        
        # Distance penalty (closer is better)
        distance_ratio = abs(level['price'] - current_price) / current_price
        distance_score = 1.0 / (1.0 + distance_ratio * 10)  # Decay factor
        score *= distance_score
        
        # Recent interaction bonus
        interaction_bonus = self._get_interaction_bonus(level['price'])
        score += interaction_bonus * 0.2
        
        # Direction bias (levels in price direction get bonus)
        if self.price_history and len(self.price_history) > 1:
            price_direction = self.price_history[-1] - self.price_history[-2]
            level_direction = level['price'] - current_price
            
            if np.sign(price_direction) == np.sign(level_direction):
                score *= 1.1  # 10% bonus for levels in price direction
                
        return score
        
    def _get_interaction_bonus(self, lvn_price: float) -> float:
        """Get bonus score based on recent interactions."""
        if not self.interaction_buffer:
            return 0.0
            
        interaction_count = 0
        for interaction in self.interaction_buffer:
            if abs(interaction['lvn_price'] - lvn_price) < 0.5:  # Price tolerance
                interaction_count += 1
                
        # Normalize by buffer size
        return min(interaction_count / 10.0, 1.0)
        
    def _build_interaction_history(
        self,
        levels: List[Dict[str, Any]],
        current_price: float
    ) -> Dict[str, Any]:
        """Build interaction history features."""
        history = {
            'total_tests': 0,
            'total_bounces': 0,
            'total_breaks': 0,
            'recent_interactions': []
        }
        
        # Analyze recent interactions
        for interaction in list(self.interaction_buffer)[-20:]:
            history['total_tests'] += 1
            
            if interaction['type'] == 'bounce':
                history['total_bounces'] += 1
            elif interaction['type'] == 'break':
                history['total_breaks'] += 1
                
            # Find closest level
            min_distance = float('inf')
            closest_level = None
            
            for level in levels:
                distance = abs(level['price'] - interaction['lvn_price'])
                if distance < min_distance:
                    min_distance = distance
                    closest_level = level
                    
            if closest_level and min_distance < 1.0:
                history['recent_interactions'].append({
                    'level_price': closest_level['price'],
                    'interaction_type': interaction['type'],
                    'time_ago': len(self.interaction_buffer) - self.interaction_buffer.index(interaction)
                })
                
        return history
        
    def _update_histories(
        self,
        levels: List[Dict[str, Any]],
        current_price: float
    ):
        """Update internal histories with new data."""
        # Update LVN history
        self.lvn_history.append({
            'timestamp': len(self.lvn_history),
            'levels': levels,
            'current_price': current_price
        })
        
        # Check for new interactions
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2]
            curr_price = self.price_history[-1]
            
            for level in levels:
                lvn_price = level['price']
                
                # Check if price crossed the level
                if (prev_price < lvn_price <= curr_price) or (prev_price > lvn_price >= curr_price):
                    # Determine interaction type
                    if abs(curr_price - lvn_price) > abs(prev_price - lvn_price):
                        interaction_type = 'break'
                    else:
                        interaction_type = 'bounce'
                        
                    self.interaction_buffer.append({
                        'lvn_price': lvn_price,
                        'type': interaction_type,
                        'strength': level['strength'],
                        'timestamp': len(self.interaction_buffer)
                    })
                    
    def batch_build_context(
        self,
        lvn_data_batch: List[Dict[str, Any]],
        market_state_batch: List[Dict[str, Any]]
    ) -> LVNContext:
        """
        Build context for a batch of samples.
        
        Args:
            lvn_data_batch: List of LVN data dictionaries
            market_state_batch: List of market state dictionaries
            
        Returns:
            Batched LVNContext
        """
        contexts = []
        
        for lvn_data, market_state in zip(lvn_data_batch, market_state_batch):
            context = self.build_context(lvn_data, market_state)
            contexts.append(context)
            
        # Stack contexts
        batched_context = LVNContext(
            prices=torch.stack([c.prices for c in contexts]),
            strengths=torch.stack([c.strengths for c in contexts]),
            distances=torch.stack([c.distances for c in contexts]),
            volumes=torch.stack([c.volumes for c in contexts]),
            current_price=np.mean([c.current_price for c in contexts]),
            price_history=torch.stack([c.price_history for c in contexts]),
            interaction_history=None  # Complex to batch, handled separately
        )
        
        return batched_context
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about context building."""
        stats = {
            'price_history_length': len(self.price_history),
            'interaction_buffer_size': len(self.interaction_buffer),
            'lvn_history_size': len(self.lvn_history)
        }
        
        if self.interaction_buffer:
            interaction_types = [i['type'] for i in self.interaction_buffer]
            stats['bounce_rate'] = interaction_types.count('bounce') / len(interaction_types)
            stats['break_rate'] = interaction_types.count('break') / len(interaction_types)
            
        if self.lvn_history:
            recent_levels = [len(h['levels']) for h in list(self.lvn_history)[-10:]]
            stats['avg_levels_per_update'] = np.mean(recent_levels)
            
        return stats