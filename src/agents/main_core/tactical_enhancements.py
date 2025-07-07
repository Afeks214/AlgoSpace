"""
Tactical Embedder Enhancements for advanced pattern recognition and microstructure analysis.

This module provides enhancement modules that can be integrated into the existing
TacticalEmbedder to add pattern recognition, microstructure analysis, FVG detection,
and execution quality prediction capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class TacticalEnhancementConfig:
    """Configuration for tactical enhancements."""
    enable_pattern_recognition: bool = True
    enable_microstructure: bool = True
    enable_fvg_detection: bool = True
    enable_execution_quality: bool = True
    pattern_lookback: int = 50
    microstructure_depth: int = 20
    fvg_threshold: float = 0.5
    hidden_dim: int = 128


class PatternRecognitionModule(nn.Module):
    """
    Advanced pattern recognition for price action patterns.
    
    Identifies: Pin bars, engulfing patterns, doji, hammer/shooting star,
    three-bar patterns, momentum shifts.
    """
    
    def __init__(self, hidden_dim: int = 128, pattern_lookback: int = 50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pattern_lookback = pattern_lookback
        
        # Pattern feature extractors
        self.candle_encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=1),  # OHLCV features
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Multi-scale pattern detection
        self.pattern_convs = nn.ModuleList([
            nn.Conv1d(64, 32, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]  # Different pattern lengths
        ])
        
        # Pattern classification heads
        self.pattern_heads = nn.ModuleDict({
            'pin_bar': nn.Linear(128, 2),  # bullish/bearish
            'engulfing': nn.Linear(128, 2),  # bullish/bearish
            'doji': nn.Linear(128, 1),  # probability
            'hammer': nn.Linear(128, 2),  # hammer/shooting star
            'three_bar': nn.Linear(128, 3),  # morning/evening star, three soldiers
            'momentum': nn.Linear(128, 3)  # increasing/decreasing/neutral
        })
        
        # Pattern importance weighting
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, price_data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect patterns in price data.
        
        Args:
            price_data: [batch, seq_len, 5] OHLCV data
            
        Returns:
            pattern_features: [batch, hidden_dim]
            pattern_detections: Dictionary of pattern probabilities
        """
        batch_size, seq_len, _ = price_data.shape
        
        # Transpose for conv1d
        x = price_data.transpose(1, 2)  # [batch, 5, seq_len]
        
        # Extract candle features
        candle_features = self.candle_encoder(x)  # [batch, 64, seq_len]
        
        # Multi-scale pattern detection
        pattern_features = []
        for conv in self.pattern_convs:
            features = conv(candle_features)
            features = F.max_pool1d(features, kernel_size=features.size(2))
            pattern_features.append(features.squeeze(-1))
            
        # Combine multi-scale features
        combined_features = torch.cat(pattern_features, dim=1)  # [batch, 128]
        
        # Detect specific patterns
        pattern_detections = {}
        for pattern_name, head in self.pattern_heads.items():
            pattern_detections[pattern_name] = torch.sigmoid(head(combined_features))
            
        # Self-attention over patterns
        pattern_tensor = combined_features.unsqueeze(0)  # [1, batch, hidden_dim]
        attended_features, _ = self.pattern_attention(
            pattern_tensor, pattern_tensor, pattern_tensor
        )
        attended_features = attended_features.squeeze(0)
        
        # Final projection
        output_features = self.output_proj(attended_features + combined_features)
        
        return output_features, pattern_detections


class MicrostructureAnalysisModule(nn.Module):
    """
    Analyzes market microstructure for liquidity and order flow insights.
    
    Features: Bid-ask spread dynamics, volume imbalance, trade size distribution,
    order book pressure, liquidity consumption.
    """
    
    def __init__(self, hidden_dim: int = 128, depth_levels: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth_levels = depth_levels
        
        # Spread dynamics analyzer
        self.spread_encoder = nn.LSTM(
            input_size=3,  # bid, ask, spread
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Volume imbalance calculator
        self.volume_imbalance = nn.Sequential(
            nn.Linear(4, 64),  # buy_vol, sell_vol, total_vol, imbalance
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32)
        )
        
        # Order book pressure analyzer
        self.book_pressure = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),  # bid/ask depths
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Trade size distribution analyzer
        self.trade_size_encoder = nn.Sequential(
            nn.Linear(10, 32),  # histogram bins
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32)
        )
        
        # Liquidity consumption tracker
        self.liquidity_lstm = nn.LSTM(
            input_size=5,  # consumed liquidity features
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(192, hidden_dim),  # 64 + 32 + 64 + 32
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        spread_data: torch.Tensor,
        volume_data: torch.Tensor,
        book_data: torch.Tensor,
        trade_sizes: torch.Tensor,
        liquidity_data: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Analyze market microstructure.
        
        Args:
            spread_data: [batch, seq_len, 3] bid/ask/spread
            volume_data: [batch, 4] volume features
            book_data: [batch, 2, depth_levels] order book
            trade_sizes: [batch, 10] trade size histogram
            liquidity_data: [batch, seq_len, 5] liquidity consumption
            
        Returns:
            microstructure_features: [batch, hidden_dim]
            metrics: Dictionary of microstructure metrics
        """
        # Analyze spread dynamics
        spread_out, (h_n, _) = self.spread_encoder(spread_data)
        spread_features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, 64]
        
        # Calculate volume imbalance
        volume_features = self.volume_imbalance(volume_data)  # [batch, 32]
        
        # Analyze order book pressure
        book_features = self.book_pressure(book_data).squeeze(-1)  # [batch, 64]
        
        # Analyze trade size distribution
        trade_features = self.trade_size_encoder(trade_sizes)  # [batch, 32]
        
        # Track liquidity consumption
        liquidity_out, (h_n, _) = self.liquidity_lstm(liquidity_data)
        liquidity_features = h_n[-1]  # [batch, 32]
        
        # Combine all features
        combined = torch.cat([
            spread_features,
            volume_features,
            book_features,
            trade_features,
            liquidity_features
        ], dim=1)
        
        # Fuse features
        microstructure_features = self.fusion_layer(combined)
        
        # Calculate metrics
        metrics = {
            'avg_spread': spread_data[:, -1, 2].mean(dim=0),
            'volume_imbalance': volume_data[:, 3],
            'book_pressure': (book_data[:, 0] - book_data[:, 1]).sum(dim=1) / book_data.sum(dim=(1, 2)),
            'liquidity_consumption_rate': liquidity_data[:, -1, 0]
        }
        
        return microstructure_features, metrics


class FVGDetectionModule(nn.Module):
    """
    Fair Value Gap (FVG) detection and analysis.
    
    Identifies price inefficiencies and imbalances that create trading opportunities.
    """
    
    def __init__(self, hidden_dim: int = 64, threshold: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        
        # Gap detection network
        self.gap_detector = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1)
        )
        
        # Gap classification
        self.gap_classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # bullish_fvg, bearish_fvg, no_fvg
        )
        
        # Gap quality assessment
        self.quality_assessor = nn.Sequential(
            nn.Linear(32 + 5, 64),  # gap features + context
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # quality score [0, 1]
        )
        
        # Temporal tracking
        self.gap_tracker = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
    def forward(
        self,
        price_data: torch.Tensor,
        context_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect and analyze FVGs.
        
        Args:
            price_data: [batch, seq_len, 5] OHLCV data
            context_features: [batch, context_dim] additional context
            
        Returns:
            fvg_features: [batch, hidden_dim]
            fvg_detections: Dictionary with FVG information
        """
        batch_size, seq_len, _ = price_data.shape
        
        # Detect gaps
        x = price_data.transpose(1, 2)  # [batch, 5, seq_len]
        gap_features = self.gap_detector(x)  # [batch, 32, seq_len]
        
        # Classify gaps at each time step
        gap_features_flat = gap_features.transpose(1, 2).reshape(-1, 32)
        gap_classes = self.gap_classifier(gap_features_flat)
        gap_probs = F.softmax(gap_classes, dim=1).view(batch_size, seq_len, 3)
        
        # Find significant gaps
        bullish_fvg = gap_probs[:, :, 0] > self.threshold
        bearish_fvg = gap_probs[:, :, 1] > self.threshold
        
        # Assess gap quality
        if context_features is not None:
            context_expanded = context_features.unsqueeze(1).expand(-1, seq_len, -1)
            quality_input = torch.cat([
                gap_features.transpose(1, 2),
                context_expanded[:, :, :5]  # Use first 5 context features
            ], dim=2).reshape(-1, 37)
        else:
            # Use last price bar as context
            last_bar = price_data[:, -1:, :].expand(-1, seq_len, -1)
            quality_input = torch.cat([
                gap_features.transpose(1, 2),
                last_bar
            ], dim=2).reshape(-1, 37)
            
        gap_quality = self.quality_assessor(quality_input).view(batch_size, seq_len)
        
        # Track gaps over time
        gap_features_seq = gap_features.transpose(1, 2)  # [batch, seq_len, 32]
        tracked_features, (h_n, _) = self.gap_tracker(gap_features_seq)
        
        # Extract final features
        fvg_features = h_n[-1]  # [batch, hidden_dim]
        
        # Compile detections
        fvg_detections = {
            'bullish_fvg_prob': gap_probs[:, -1, 0],
            'bearish_fvg_prob': gap_probs[:, -1, 1],
            'gap_quality': gap_quality[:, -1],
            'active_bullish_gaps': bullish_fvg.sum(dim=1).float(),
            'active_bearish_gaps': bearish_fvg.sum(dim=1).float(),
            'strongest_gap_quality': gap_quality.max(dim=1)[0]
        }
        
        return fvg_features, fvg_detections


class ExecutionQualityPredictor(nn.Module):
    """
    Predicts execution quality metrics for tactical decisions.
    
    Estimates: Expected slippage, fill probability, adverse selection risk,
    optimal execution timing.
    """
    
    def __init__(self, hidden_dim: int = 64, input_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Market condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )
        
        # Slippage predictor
        self.slippage_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # mean and std of expected slippage
        )
        
        # Fill probability estimator
        self.fill_prob_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # probabilities for different order types
            nn.Sigmoid()
        )
        
        # Adverse selection risk
        self.adverse_selection_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # risk score [0, 1]
        )
        
        # Optimal timing predictor
        self.timing_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # scores for next 5 time periods
            nn.Softmax(dim=1)
        )
        
    def forward(
        self,
        market_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict execution quality metrics.
        
        Args:
            market_features: [batch, input_dim] combined market features
            
        Returns:
            Dictionary of execution quality predictions
        """
        # Encode market conditions
        encoded = self.condition_encoder(market_features)
        
        # Predict slippage
        slippage_params = self.slippage_head(encoded)
        slippage_mean = slippage_params[:, 0]
        slippage_std = F.softplus(slippage_params[:, 1]) + 1e-6
        
        # Predict fill probabilities
        fill_probs = self.fill_prob_head(encoded)
        
        # Predict adverse selection risk
        adverse_risk = self.adverse_selection_head(encoded).squeeze(1)
        
        # Predict optimal timing
        timing_scores = self.timing_head(encoded)
        
        return {
            'expected_slippage_mean': slippage_mean,
            'expected_slippage_std': slippage_std,
            'fill_prob_market': fill_probs[:, 0],
            'fill_prob_limit': fill_probs[:, 1],
            'fill_prob_stop': fill_probs[:, 2],
            'adverse_selection_risk': adverse_risk,
            'optimal_timing_scores': timing_scores,
            'recommended_delay': torch.argmax(timing_scores, dim=1)
        }


class TacticalEnhancementIntegrator(nn.Module):
    """
    Integrates all tactical enhancements into a unified module.
    """
    
    def __init__(self, config: TacticalEnhancementConfig):
        super().__init__()
        self.config = config
        
        # Initialize enhancement modules
        if config.enable_pattern_recognition:
            self.pattern_module = PatternRecognitionModule(
                hidden_dim=config.hidden_dim,
                pattern_lookback=config.pattern_lookback
            )
            
        if config.enable_microstructure:
            self.microstructure_module = MicrostructureAnalysisModule(
                hidden_dim=config.hidden_dim,
                depth_levels=config.microstructure_depth
            )
            
        if config.enable_fvg_detection:
            self.fvg_module = FVGDetectionModule(
                hidden_dim=config.hidden_dim // 2,
                threshold=config.fvg_threshold
            )
            
        if config.enable_execution_quality:
            self.execution_module = ExecutionQualityPredictor(
                hidden_dim=config.hidden_dim // 2,
                input_dim=config.hidden_dim * 2
            )
            
        # Feature dimension calculation
        feature_dim = 0
        if config.enable_pattern_recognition:
            feature_dim += config.hidden_dim
        if config.enable_microstructure:
            feature_dim += config.hidden_dim
        if config.enable_fvg_detection:
            feature_dim += config.hidden_dim // 2
            
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(
        self,
        price_data: torch.Tensor,
        microstructure_data: Optional[Dict[str, torch.Tensor]] = None,
        base_features: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Process all tactical enhancements.
        
        Args:
            price_data: OHLCV price data
            microstructure_data: Optional microstructure inputs
            base_features: Optional base tactical features
            
        Returns:
            Dictionary with enhanced features and predictions
        """
        features = []
        results = {}
        
        # Pattern recognition
        if self.config.enable_pattern_recognition:
            pattern_features, pattern_detections = self.pattern_module(price_data)
            features.append(pattern_features)
            results['patterns'] = pattern_detections
            
        # Microstructure analysis
        if self.config.enable_microstructure and microstructure_data is not None:
            micro_features, micro_metrics = self.microstructure_module(
                microstructure_data['spread'],
                microstructure_data['volume'],
                microstructure_data['book'],
                microstructure_data['trade_sizes'],
                microstructure_data['liquidity']
            )
            features.append(micro_features)
            results['microstructure'] = micro_metrics
            
        # FVG detection
        if self.config.enable_fvg_detection:
            fvg_features, fvg_detections = self.fvg_module(price_data, base_features)
            features.append(fvg_features)
            results['fvg'] = fvg_detections
            
        # Combine features
        if features:
            combined_features = torch.cat(features, dim=1)
            enhanced_features = self.fusion_layer(combined_features)
            
            # Execution quality prediction
            if self.config.enable_execution_quality:
                if base_features is not None:
                    exec_input = torch.cat([enhanced_features, base_features], dim=1)
                else:
                    exec_input = torch.cat([enhanced_features, enhanced_features], dim=1)
                    
                execution_predictions = self.execution_module(exec_input)
                results['execution'] = execution_predictions
                
            results['enhanced_features'] = enhanced_features
        else:
            results['enhanced_features'] = base_features
            
        return results