"""
IndicatorEngine Component - Central Feature Calculation Engine

This module implements the system's computational powerhouse that transforms raw OHLCV bars
into the rich set of technical indicators, market profile features, and regime detection inputs.
It serves as the single source of truth for all calculated features.

Based on Master PRD - IndicatorEngine Component v1.0
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np
import pandas as pd
import copy

from ..core.kernel import ComponentBase, AlgoSpaceKernel
from ..core.events import EventType, Event, BarData
from ..data.validators import BarValidator

# Import existing indicators
from .mlmi import MLMICalculator
from .nwrqk import NWRQKCalculator
from .fvg import FVGDetector
from .lvn import LVNAnalyzer
from .mmd import MMDFeatureExtractor


class IndicatorEngine(ComponentBase):
    """
    Central Feature Calculation Engine
    
    Transforms raw OHLCV bars into comprehensive technical indicators and features.
    Maintains dual-path processing for 5-minute and 30-minute timeframes with
    Heiken Ashi conversion and centralized Feature Store management.
    
    Key Features:
    - Dual timeframe processing (5min FVG, 30min all others)
    - Heiken Ashi conversion for 30min indicators
    - Centralized Feature Store with atomic updates
    - Single INDICATORS_READY event emission
    - Default parameter enforcement (DIR-DATA-02)
    - Advanced feature engineering including LVN strength scores
    - MMD feature vector with 7 reference distributions
    - Interaction features between indicators
    """
    
    def __init__(self, name: str, kernel: AlgoSpaceKernel):
        """
        Initialize IndicatorEngine
        
        Args:
            name: Component name
            kernel: System kernel instance
        """
        super().__init__(name, kernel)
        
        # Configuration
        self.symbol = self.config.primary_symbol
        self.timeframes = self.config.timeframes
        self.indicators_config = self.config.get_section('indicators')
        
        # Feature Store - Single source of truth
        self.feature_store: Dict[str, Any] = {}
        self._feature_store_lock = asyncio.Lock()
        
        # History buffers with fixed sizes for memory management
        self.history_5m: deque = deque(maxlen=100)  # Last 100 5-min bars
        self.history_30m: deque = deque(maxlen=100)  # Last 100 30-min bars
        self.ha_history_30m: deque = deque(maxlen=100)  # Heiken Ashi 30-min bars
        
        # Volume profile for LVN (rolling 20 bars)
        self.volume_profile_buffer: deque = deque(maxlen=20)
        
        # FVG tracking
        self.active_fvgs: List[Dict[str, Any]] = []
        self.fvg_max_age = 50  # Remove FVGs older than 50 bars
        
        # Update tracking for synchronized emission
        self.last_update_5min: Optional[datetime] = None
        self.last_update_30min: Optional[datetime] = None
        self.has_30min_data = False
        
        # Statistics
        self.calculations_5min = 0
        self.calculations_30min = 0
        self.events_emitted = 0
        
        # Initialize bar validator
        self.bar_validator = BarValidator()
        
        # MMD Reference Distributions - 7 market archetypes
        self.mmd_reference_distributions = self._load_reference_distributions()
        
        # LVN interaction history for strength calculation
        self.lvn_interaction_history: Dict[float, List[Dict[str, Any]]] = {}
        
        # Initialize indicator instances with default parameters
        self._initialize_indicators()
        
        self.logger.info("IndicatorEngine initialized",
                        symbol=self.symbol,
                        timeframes=self.timeframes,
                        indicators=list(self.indicators_config.keys()),
                        feature_store_size=len(self.feature_store))
    
    def _initialize_indicators(self) -> None:
        """Initialize all indicator instances with default parameters"""
        try:
            # Default parameters (DIR-DATA-02)
            default_configs = {
                'mlmi': {
                    'k_neighbors': 5,
                    'trend_length': 14,
                    'max_history_length': 100
                },
                'nwrqk': {
                    'bandwidth': 46,
                    'alpha': 8,
                    'max_history_length': 100
                },
                'fvg': {
                    'threshold': 0.001,  # 0.1% minimum gap size
                    'max_age': 50
                },
                'lvn': {
                    'lookback_periods': 20,
                    'strength_threshold': 0.7,  # 70% below POC = LVN
                    'max_history_length': 100
                },
                'mmd': {
                    'signature_degree': 3,
                    'max_history_length': 100
                }
            }
            
            # Create indicator instances
            self.mlmi = MLMICalculator(default_configs['mlmi'], self.event_bus)
            self.nwrqk = NWRQKCalculator(default_configs['nwrqk'], self.event_bus)
            self.fvg = FVGDetector(default_configs['fvg'], self.event_bus)
            self.lvn = LVNAnalyzer(default_configs['lvn'], self.event_bus)
            self.mmd = MMDFeatureExtractor(default_configs['mmd'], self.event_bus)
            
            # Initialize Feature Store with default values
            self._initialize_feature_store()
            
            self.logger.info("All indicators initialized with default parameters")
            
        except Exception as e:
            self.logger.error("Failed to initialize indicators", error=str(e))
            raise
    
    def _initialize_feature_store(self) -> None:
        """Initialize Feature Store with default values"""
        self.feature_store = {
            # 30-minute features (Heiken Ashi based)
            'mlmi_value': 0.0,
            'mlmi_signal': 0,
            'nwrqk_value': 0.0,
            'nwrqk_slope': 0.0,
            'nwrqk_signal': 0,
            'lvn_nearest_price': 0.0,
            'lvn_nearest_strength': 0.0,
            'lvn_distance_points': 0.0,
            
            # 5-minute features (Standard candles)
            'fvg_bullish_active': False,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 0.0,
            'fvg_age': 0,
            'fvg_mitigation_signal': False,
            
            # Regime features - MMD feature vector
            'mmd_features': np.zeros(13),  # 13-dimensional feature vector
            
            # Interaction features
            'mlmi_minus_nwrqk': 0.0,
            'mlmi_div_nwrqk': 1.0,
            
            # Metadata
            'last_update_5min': None,
            'last_update_30min': None,
            'calculation_status': 'initialized',
            'feature_count': 0
        }
    
    def _load_reference_distributions(self) -> List[np.ndarray]:
        """
        Load 7 pre-defined reference distributions for MMD calculation
        
        Returns:
            List of 7 reference distribution arrays representing market archetypes:
            1. Strong Trending Up
            2. Strong Trending Down
            3. Ranging/Consolidation
            4. High Volatility
            5. Low Volatility
            6. Breakout Pattern
            7. Mean Reversion Pattern
        """
        # Generate synthetic reference distributions for market archetypes
        n_samples = 100
        n_features = 4  # returns, log_returns, range, volatility
        
        distributions = []
        
        # 1. Strong Trending Up
        trend_up = np.zeros((n_samples, n_features))
        trend_up[:, 0] = np.linspace(0.001, 0.005, n_samples) + np.random.normal(0, 0.0001, n_samples)
        trend_up[:, 1] = np.log(1 + trend_up[:, 0])
        trend_up[:, 2] = np.abs(np.random.normal(0.002, 0.0005, n_samples))
        trend_up[:, 3] = np.full(n_samples, 0.015)
        distributions.append(trend_up)
        
        # 2. Strong Trending Down
        trend_down = np.zeros((n_samples, n_features))
        trend_down[:, 0] = np.linspace(-0.001, -0.005, n_samples) + np.random.normal(0, 0.0001, n_samples)
        trend_down[:, 1] = np.log(1 + trend_down[:, 0])
        trend_down[:, 2] = np.abs(np.random.normal(0.002, 0.0005, n_samples))
        trend_down[:, 3] = np.full(n_samples, 0.015)
        distributions.append(trend_down)
        
        # 3. Ranging/Consolidation
        ranging = np.zeros((n_samples, n_features))
        ranging[:, 0] = np.random.normal(0, 0.001, n_samples)
        ranging[:, 1] = np.log(1 + ranging[:, 0])
        ranging[:, 2] = np.abs(np.random.normal(0.001, 0.0002, n_samples))
        ranging[:, 3] = np.full(n_samples, 0.008)
        distributions.append(ranging)
        
        # 4. High Volatility
        high_vol = np.zeros((n_samples, n_features))
        high_vol[:, 0] = np.random.normal(0, 0.01, n_samples)
        high_vol[:, 1] = np.log(1 + high_vol[:, 0])
        high_vol[:, 2] = np.abs(np.random.normal(0.01, 0.003, n_samples))
        high_vol[:, 3] = np.full(n_samples, 0.03)
        distributions.append(high_vol)
        
        # 5. Low Volatility
        low_vol = np.zeros((n_samples, n_features))
        low_vol[:, 0] = np.random.normal(0, 0.0005, n_samples)
        low_vol[:, 1] = np.log(1 + low_vol[:, 0])
        low_vol[:, 2] = np.abs(np.random.normal(0.0005, 0.0001, n_samples))
        low_vol[:, 3] = np.full(n_samples, 0.005)
        distributions.append(low_vol)
        
        # 6. Breakout Pattern
        breakout = np.zeros((n_samples, n_features))
        breakout[:50, 0] = np.random.normal(0, 0.001, 50)
        breakout[50:, 0] = np.random.normal(0.003, 0.001, 50)
        breakout[:, 1] = np.log(1 + breakout[:, 0])
        breakout[:, 2] = np.abs(np.random.normal(0.003, 0.001, n_samples))
        breakout[:, 3] = np.concatenate([np.full(50, 0.008), np.full(50, 0.02)])
        distributions.append(breakout)
        
        # 7. Mean Reversion Pattern
        mean_rev = np.zeros((n_samples, n_features))
        mean_rev[:, 0] = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.003 + np.random.normal(0, 0.0005, n_samples)
        mean_rev[:, 1] = np.log(1 + mean_rev[:, 0])
        mean_rev[:, 2] = np.abs(np.random.normal(0.002, 0.0005, n_samples))
        mean_rev[:, 3] = np.full(n_samples, 0.012)
        distributions.append(mean_rev)
        
        return distributions
    
    async def start(self) -> None:
        """Start the IndicatorEngine component"""
        await super().start()
        
        # Subscribe to bar events from BarGenerator
        self.subscribe_to_event(EventType.NEW_5MIN_BAR, self._on_5min_bar)
        self.subscribe_to_event(EventType.NEW_30MIN_BAR, self._on_30min_bar)
        
        self.logger.info("IndicatorEngine started - subscribed to bar events")
    
    async def stop(self) -> None:
        """Stop the IndicatorEngine component"""
        # Log final statistics
        self.logger.info("IndicatorEngine stopping",
                        calculations_5min=self.calculations_5min,
                        calculations_30min=self.calculations_30min,
                        events_emitted=self.events_emitted,
                        final_feature_count=len(self.feature_store))
        
        await super().stop()
    
    def _on_5min_bar(self, event: Event) -> None:
        """
        Process 5-minute bar for FVG detection
        
        Args:
            event: NEW_5MIN_BAR event containing BarData
        """
        try:
            bar_data: BarData = event.payload
            
            # Validate bar data
            if not self._validate_bar_data(bar_data, '5min'):
                return
            
            # Update history buffer
            self.history_5m.append(bar_data)
            
            # Calculate 5-minute features (FVG on standard candles)
            start_time = datetime.now()
            features_5min = self._calculate_5min_features(bar_data)
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Verify performance requirement (<50ms)
            if calc_time > 50:
                self.logger.warning("5-min calculation exceeded 50ms threshold",
                                   calculation_time_ms=calc_time)
            
            # Update Feature Store atomically
            asyncio.create_task(self._update_feature_store_5min(features_5min, bar_data.timestamp))
            
            self.calculations_5min += 1
            
            self.logger.debug("5-min features calculated",
                            timestamp=bar_data.timestamp.isoformat(),
                            calc_time_ms=calc_time,
                            features=list(features_5min.keys()))
            
        except Exception as e:
            self.logger.error("Error processing 5-min bar",
                            bar_data=bar_data,
                            error=str(e))
    
    def _on_30min_bar(self, event: Event) -> None:
        """
        Process 30-minute bar for all major indicators
        
        Args:
            event: NEW_30MIN_BAR event containing BarData
        """
        try:
            bar_data: BarData = event.payload
            
            # Validate bar data
            if not self._validate_bar_data(bar_data, '30min'):
                return
            
            # Update history buffers
            self.history_30m.append(bar_data)
            self.volume_profile_buffer.append(bar_data)
            
            # Convert to Heiken Ashi and update HA history
            ha_bar = self._convert_to_heiken_ashi(bar_data)
            self.ha_history_30m.append(ha_bar)
            
            # Calculate 30-minute features (all on Heiken Ashi)
            start_time = datetime.now()
            features_30min = self._calculate_30min_features(bar_data, ha_bar)
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Verify performance requirement (<100ms)
            if calc_time > 100:
                self.logger.warning("30-min calculation exceeded 100ms threshold",
                                   calculation_time_ms=calc_time)
            
            # Update Feature Store atomically
            asyncio.create_task(self._update_feature_store_30min(features_30min, bar_data.timestamp))
            
            self.calculations_30min += 1
            self.has_30min_data = True
            
            self.logger.debug("30-min features calculated",
                            timestamp=bar_data.timestamp.isoformat(),
                            calc_time_ms=calc_time,
                            features=list(features_30min.keys()))
            
        except Exception as e:
            self.logger.error("Error processing 30-min bar",
                            bar_data=bar_data,
                            error=str(e))
    
    def _validate_bar_data(self, bar_data: BarData, timeframe: str) -> bool:
        """
        Validate incoming bar data
        
        Args:
            bar_data: Bar data to validate
            timeframe: Timeframe string for logging
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check symbol match first
            if bar_data.symbol != self.symbol:
                self.logger.debug(f"Bar for different symbol ignored",
                                expected=self.symbol,
                                received=bar_data.symbol)
                return False
            
            # Use BarValidator for comprehensive validation
            is_valid, validation_errors = self.bar_validator.validate_bar(bar_data)
            
            if not is_valid:
                self.logger.warning(f"Invalid {timeframe} bar detected",
                                  bar_data=bar_data,
                                  errors=validation_errors)
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating {timeframe} bar", error=str(e))
            return False
    
    def _convert_to_heiken_ashi(self, bar_data: BarData) -> Dict[str, float]:
        """
        Convert standard OHLCV bar to Heiken Ashi
        
        Args:
            bar_data: Standard OHLCV bar
            
        Returns:
            Heiken Ashi OHLC dictionary
        """
        # HA Close = (O + H + L + C) / 4
        ha_close = (bar_data.open + bar_data.high + bar_data.low + bar_data.close) / 4
        
        # HA Open calculation
        if len(self.ha_history_30m) == 0:
            # First bar: HA_Open = (Open + Close) / 2
            ha_open = (bar_data.open + bar_data.close) / 2
        else:
            # HA_Open = (Previous_HA_Open + Previous_HA_Close) / 2
            prev_ha = self.ha_history_30m[-1]
            ha_open = (prev_ha['open'] + prev_ha['close']) / 2
        
        # HA High = max(High, HA_Open, HA_Close)
        ha_high = max(bar_data.high, ha_open, ha_close)
        
        # HA Low = min(Low, HA_Open, HA_Close)
        ha_low = min(bar_data.low, ha_open, ha_close)
        
        return {
            'open': ha_open,
            'high': ha_high,
            'low': ha_low,
            'close': ha_close,
            'volume': bar_data.volume,
            'timestamp': bar_data.timestamp
        }
    
    def _calculate_5min_features(self, bar_data: BarData) -> Dict[str, Any]:  # noqa: F841
        """
        Calculate 5-minute features (FVG detection on standard candles)
        
        Args:
            bar_data: Current 5-minute bar
            
        Returns:
            Dictionary of calculated features
        """
        features = {}
        
        try:
            # Need at least 3 bars for FVG detection
            if len(self.history_5m) >= 3:
                fvg_results = self._detect_fvg()
                features.update(fvg_results)
            else:
                # Insufficient data
                features.update({
                    'fvg_bullish_active': False,
                    'fvg_bearish_active': False,
                    'fvg_nearest_level': 0.0,
                    'fvg_age': 0,
                    'fvg_mitigation_signal': False
                })
            
            return features
            
        except Exception as e:
            self.logger.error("Error calculating 5-min features", error=str(e))
            return {
                'fvg_bullish_active': False,
                'fvg_bearish_active': False,
                'fvg_nearest_level': 0.0,
                'fvg_age': 0,
                'fvg_mitigation_signal': False
            }
    
    def _calculate_30min_features(self, bar_data: BarData, ha_bar: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate 30-minute features (all indicators on Heiken Ashi)
        
        Args:
            bar_data: Standard OHLCV bar for volume analysis
            ha_bar: Heiken Ashi converted bar
            
        Returns:
            Dictionary of calculated features
        """
        features = {}
        
        try:
            # Calculate MLMI (on HA data)
            if len(self.ha_history_30m) >= 15:  # Need at least 15 bars for RSI + k-NN
                mlmi_results = self._calculate_mlmi(ha_bar)
                features.update(mlmi_results)
            else:
                features.update({'mlmi_value': 0.0, 'mlmi_signal': 0})
            
            # Calculate NW-RQK (on HA data)
            if len(self.ha_history_30m) >= 46:  # Need bandwidth worth of data
                nwrqk_results = self._calculate_nwrqk(ha_bar)
                features.update(nwrqk_results)
            else:
                features.update({'nwrqk_value': 0.0, 'nwrqk_slope': 0.0, 'nwrqk_signal': 0})
            
            # Calculate LVN with enhanced strength score (on standard bar data for volume profile)
            if len(self.volume_profile_buffer) >= 20:  # Need full volume profile window
                lvn_results = self._calculate_enhanced_lvn(bar_data)
                features.update(lvn_results)
            else:
                features.update({
                    'lvn_nearest_price': 0.0,
                    'lvn_nearest_strength': 0.0,
                    'lvn_distance_points': 0.0
                })
            
            # Calculate MMD features with 7 reference distributions
            if len(self.ha_history_30m) >= 10:  # Need minimum data for path signatures
                mmd_results = self._calculate_enhanced_mmd(bar_data)
                features.update(mmd_results)
            else:
                features.update({'mmd_features': np.zeros(13)})
            
            return features
            
        except Exception as e:
            self.logger.error("Error calculating 30-min features", error=str(e))
            return {
                'mlmi_value': 0.0,
                'mlmi_signal': 0,
                'nwrqk_value': 0.0,
                'nwrqk_slope': 0.0,
                'nwrqk_signal': 0,
                'lvn_nearest_price': 0.0,
                'lvn_nearest_strength': 0.0,
                'lvn_distance_points': 0.0,
                'mmd_features': np.zeros(13)
            }
    
    def _detect_fvg(self) -> Dict[str, Any]:
        """
        Detect Fair Value Gaps using the FVG detector
        
        Returns:
            Dictionary of FVG features
        """
        try:
            # Convert deque to list for FVG detector
            bars_list = list(self.history_5m)
            
            # Use the existing FVG detector
            current_bar = bars_list[-1]
            fvg_result = self.fvg.calculate_5m(current_bar)
            
            # Update active FVGs tracking
            self._update_fvg_tracking(bars_list)
            
            # Calculate nearest FVG level and age
            nearest_fvg = self._find_nearest_fvg(current_bar.close)
            
            return {
                'fvg_bullish_active': fvg_result.get('fvg_bullish_active', False),
                'fvg_bearish_active': fvg_result.get('fvg_bearish_active', False),
                'fvg_nearest_level': nearest_fvg.get('level', 0.0),
                'fvg_age': nearest_fvg.get('age', 0),
                'fvg_mitigation_signal': fvg_result.get('fvg_mitigation_signal', False)
            }
            
        except Exception as e:
            self.logger.error("Error in FVG detection", error=str(e))
            return {
                'fvg_bullish_active': False,
                'fvg_bearish_active': False,
                'fvg_nearest_level': 0.0,
                'fvg_age': 0,
                'fvg_mitigation_signal': False
            }
    
    def _calculate_mlmi(self, ha_bar: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate MLMI using the MLMI calculator on Heiken Ashi data
        
        Args:
            ha_bar: Current Heiken Ashi bar
            
        Returns:
            Dictionary of MLMI features
        """
        try:
            # Create a BarData object from HA data for consistency
            ha_bar_data = BarData(
                symbol=self.symbol,
                timestamp=ha_bar['timestamp'],
                open=ha_bar['open'],
                high=ha_bar['high'],
                low=ha_bar['low'],
                close=ha_bar['close'],
                volume=int(ha_bar['volume']),
                timeframe=30
            )
            
            # Use the existing MLMI calculator
            mlmi_result = self.mlmi.calculate_30m(ha_bar_data)
            
            return {
                'mlmi_value': mlmi_result.get('mlmi_value', 0.0),
                'mlmi_signal': mlmi_result.get('mlmi_signal', 0)
            }
            
        except Exception as e:
            self.logger.error("Error calculating MLMI", error=str(e))
            return {'mlmi_value': 0.0, 'mlmi_signal': 0}
    
    def _calculate_nwrqk(self, ha_bar: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate NW-RQK using the NW-RQK calculator on Heiken Ashi data
        
        Args:
            ha_bar: Current Heiken Ashi bar
            
        Returns:
            Dictionary of NW-RQK features including slope
        """
        try:
            # Create a BarData object from HA data
            ha_bar_data = BarData(
                symbol=self.symbol,
                timestamp=ha_bar['timestamp'],
                open=ha_bar['open'],
                high=ha_bar['high'],
                low=ha_bar['low'],
                close=ha_bar['close'],
                volume=int(ha_bar['volume']),
                timeframe=30
            )
            
            # Use the existing NW-RQK calculator
            nwrqk_result = self.nwrqk.calculate_30m(ha_bar_data)
            
            # Calculate slope of NW-RQK line
            nwrqk_slope = 0.0
            if len(self.ha_history_30m) >= 2:
                prev_ha_bar_data = BarData(
                    symbol=self.symbol,
                    timestamp=self.ha_history_30m[-2]['timestamp'],
                    open=self.ha_history_30m[-2]['open'],
                    high=self.ha_history_30m[-2]['high'],
                    low=self.ha_history_30m[-2]['low'],
                    close=self.ha_history_30m[-2]['close'],
                    volume=int(self.ha_history_30m[-2]['volume']),
                    timeframe=30
                )
                prev_nwrqk = self.nwrqk.calculate_30m(prev_ha_bar_data)
                current_value = nwrqk_result.get('nwrqk_value', 0.0)
                prev_value = prev_nwrqk.get('nwrqk_value', 0.0)
                if prev_value != 0:
                    nwrqk_slope = (current_value - prev_value) / prev_value
            
            return {
                'nwrqk_value': nwrqk_result.get('nwrqk_value', 0.0),
                'nwrqk_slope': nwrqk_slope,
                'nwrqk_signal': nwrqk_result.get('nwrqk_signal', 0)
            }
            
        except Exception as e:
            self.logger.error("Error calculating NW-RQK", error=str(e))
            return {'nwrqk_value': 0.0, 'nwrqk_slope': 0.0, 'nwrqk_signal': 0}
    
    def _calculate_enhanced_lvn(self, bar_data: BarData) -> Dict[str, Any]:
        """
        Calculate LVN with enhanced strength score based on historical interactions
        
        Args:
            bar_data: Current standard OHLCV bar
            
        Returns:
            Dictionary of LVN features with strength score
        """
        try:
            # Use the LVN analyzer with original bar data for volume profile
            lvn_result = self.lvn.calculate_30m(bar_data)
            
            nearest_lvn_price = lvn_result.get('nearest_lvn_price', 0.0)
            base_strength = lvn_result.get('nearest_lvn_strength', 0.0)
            
            # Calculate enhanced strength score based on historical interactions
            enhanced_strength = self._calculate_lvn_strength_score(
                nearest_lvn_price, 
                base_strength,
                bar_data.close
            )
            
            return {
                'lvn_nearest_price': nearest_lvn_price,
                'lvn_nearest_strength': enhanced_strength,
                'lvn_distance_points': lvn_result.get('distance_to_nearest_lvn', 0.0)
            }
            
        except Exception as e:
            self.logger.error("Error calculating enhanced LVN", error=str(e))
            return {
                'lvn_nearest_price': 0.0,
                'lvn_nearest_strength': 0.0,
                'lvn_distance_points': 0.0
            }
    
    def _calculate_lvn_strength_score(self, lvn_price: float, base_strength: float, current_price: float) -> float:
        """
        Calculate LVN strength score based on historical price action
        
        Args:
            lvn_price: The identified LVN price level
            base_strength: The base strength from volume profile
            current_price: Current market price
            
        Returns:
            Enhanced strength score (0-1)
        """
        if lvn_price == 0.0:
            return 0.0
        
        # Initialize interaction tracking for new LVN levels
        if lvn_price not in self.lvn_interaction_history:
            self.lvn_interaction_history[lvn_price] = []
        
        # Check if price is testing the LVN level (within 0.25% range)
        test_threshold = lvn_price * 0.0025
        is_testing = abs(current_price - lvn_price) <= test_threshold
        
        if is_testing:
            # Record the interaction
            interaction = {
                'timestamp': datetime.now(),
                'test_price': current_price,
                'approached_from': 'above' if current_price > lvn_price else 'below',
                'volume': self.history_30m[-1].volume if self.history_30m else 0
            }
            self.lvn_interaction_history[lvn_price].append(interaction)
        
        # Calculate strength factors
        interactions = self.lvn_interaction_history[lvn_price]
        
        # Factor 1: Number of tests (more tests = stronger level)
        test_count_factor = min(len(interactions) / 10.0, 1.0)  # Normalize to 0-1
        
        # Factor 2: Rejection strength (look at price movement after test)
        rejection_factor = 0.0
        if len(interactions) >= 2:
            strong_rejections = 0
            for i in range(len(interactions) - 1):
                # Check if price moved away significantly after test
                time_diff = (interactions[i+1]['timestamp'] - interactions[i]['timestamp']).total_seconds()
                if time_diff < 7200:  # Within 2 hours
                    price_diff = abs(interactions[i+1]['test_price'] - lvn_price)
                    if price_diff > test_threshold * 2:
                        strong_rejections += 1
            rejection_factor = min(strong_rejections / 5.0, 1.0)
        
        # Factor 3: Recency (recent interactions are more relevant)
        recency_factor = 0.0
        if interactions:
            latest_interaction = interactions[-1]
            hours_since = (datetime.now() - latest_interaction['timestamp']).total_seconds() / 3600
            recency_factor = max(0, 1 - (hours_since / 168))  # Decay over 1 week
        
        # Combine factors with weights
        enhanced_strength = (
            base_strength * 0.4 +
            test_count_factor * 0.3 +
            rejection_factor * 0.2 +
            recency_factor * 0.1
        )
        
        # Clean up old interactions (older than 7 days)
        cutoff_time = datetime.now().timestamp() - (7 * 24 * 3600)
        self.lvn_interaction_history[lvn_price] = [
            i for i in interactions 
            if i['timestamp'].timestamp() > cutoff_time
        ]
        
        return min(enhanced_strength, 1.0)
    
    def _calculate_enhanced_mmd(self, bar_data: BarData) -> Dict[str, Any]:
        """
        Calculate MMD features with 7 reference distributions
        
        Args:
            bar_data: Current standard OHLCV bar
            
        Returns:
            Dictionary with enhanced MMD feature vector
        """
        try:
            # Get base MMD features from the MMD engine
            mmd_result = self.mmd.calculate_30m(bar_data)
            base_features = mmd_result.get('mmd_features', np.zeros(13))
            
            # Extract the current window data for MMD calculation
            if len(self.history_30m) < 100:
                return {'mmd_features': base_features}
            
            # Prepare current window data
            df = pd.DataFrame([{
                'close': b.close, 'high': b.high, 'low': b.low, 'volume': b.volume
            } for b in list(self.history_30m)[-100:]])
            
            # Calculate features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close'].shift(1)
            df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252 * 48)
            df = df.dropna()
            
            current_data = df[['returns', 'log_returns', 'range', 'volatility']].values
            
            # Normalize current data
            mean = np.mean(current_data, axis=0)
            std = np.std(current_data, axis=0)
            std[std < 1e-8] = 1e-8
            current_normalized = (current_data - mean) / std
            
            # Calculate MMD scores against 7 reference distributions
            mmd_scores = []
            for ref_dist in self.mmd_reference_distributions:
                # Normalize reference distribution
                ref_normalized = (ref_dist - mean) / std
                
                # Use the MMD compute function from the mmd module
                from .mmd import compute_mmd
                sigma = 1.0  # Use standard sigma
                mmd_score = compute_mmd(current_normalized, ref_normalized, sigma)
                mmd_scores.append(mmd_score)
            
            # Create enhanced feature vector
            # First 7 elements: MMD scores against reference distributions
            # Next 6 elements: Statistical features from base_features
            enhanced_features = np.zeros(13)
            enhanced_features[:7] = mmd_scores
            enhanced_features[7:] = base_features[7:]  # Keep statistical features
            
            return {'mmd_features': enhanced_features}
            
        except Exception as e:
            self.logger.error("Error calculating enhanced MMD", error=str(e))
            return {'mmd_features': np.zeros(13)}
    
    def _update_fvg_tracking(self, bars: List[BarData]) -> None:
        """Update FVG tracking list with new detections and mitigations"""
        try:
            # Age existing FVGs and remove expired ones
            current_bar_index = len(bars) - 1
            self.active_fvgs = [
                fvg for fvg in self.active_fvgs
                if current_bar_index - fvg['creation_bar'] <= self.fvg_max_age
            ]
            
            # Check for mitigations
            current_price = bars[-1].close
            for fvg in self.active_fvgs:
                if not fvg.get('mitigated', False):
                    if fvg['type'] == 'bullish' and current_price <= fvg['lower_bound']:
                        fvg['mitigated'] = True
                        fvg['mitigation_bar'] = current_bar_index
                    elif fvg['type'] == 'bearish' and current_price >= fvg['upper_bound']:
                        fvg['mitigated'] = True
                        fvg['mitigation_bar'] = current_bar_index
            
        except Exception as e:
            self.logger.error("Error updating FVG tracking", error=str(e))
    
    def _find_nearest_fvg(self, current_price: float) -> Dict[str, Any]:
        """Find the nearest active FVG to current price"""
        try:
            if not self.active_fvgs:
                return {'level': 0.0, 'age': 0}
            
            nearest_fvg = None
            min_distance = float('inf')
            
            for fvg in self.active_fvgs:
                if not fvg.get('mitigated', False):
                    # Calculate distance to FVG center
                    fvg_center = (fvg['upper_bound'] + fvg['lower_bound']) / 2
                    distance = abs(current_price - fvg_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_fvg = fvg
            
            if nearest_fvg:
                return {
                    'level': (nearest_fvg['upper_bound'] + nearest_fvg['lower_bound']) / 2,
                    'age': len(self.history_5m) - nearest_fvg['creation_bar']
                }
            
            return {'level': 0.0, 'age': 0}
            
        except Exception as e:
            self.logger.error("Error finding nearest FVG", error=str(e))
            return {'level': 0.0, 'age': 0}
    
    def _calculate_interaction_features(self) -> None:
        """Calculate interaction features between indicators"""
        try:
            # MLMI minus NWRQK
            mlmi_value = self.feature_store.get('mlmi_value', 0.0)
            nwrqk_value = self.feature_store.get('nwrqk_value', 0.0)
            
            self.feature_store['mlmi_minus_nwrqk'] = mlmi_value - nwrqk_value
            
            # MLMI divided by NWRQK (with zero protection)
            if abs(nwrqk_value) > 1e-8:
                self.feature_store['mlmi_div_nwrqk'] = mlmi_value / nwrqk_value
            else:
                self.feature_store['mlmi_div_nwrqk'] = 1.0 if mlmi_value > 0 else -1.0 if mlmi_value < 0 else 0.0
            
        except Exception as e:
            self.logger.error("Error calculating interaction features", error=str(e))
    
    async def _update_feature_store_5min(self, features: Dict[str, Any], timestamp: datetime) -> None:
        """
        Update Feature Store with 5-minute features atomically
        
        Args:
            features: Calculated 5-minute features
            timestamp: Update timestamp
        """
        async with self._feature_store_lock:
            try:
                # Update 5-minute features
                for key, value in features.items():
                    self.feature_store[key] = value
                
                # Update metadata
                self.feature_store['last_update_5min'] = timestamp
                self.last_update_5min = timestamp
                
                # Check if we should emit INDICATORS_READY
                await self._check_and_emit_indicators_ready()
                
            except Exception as e:
                self.logger.error("Error updating Feature Store (5min)", error=str(e))
    
    async def _update_feature_store_30min(self, features: Dict[str, Any], timestamp: datetime) -> None:
        """
        Update Feature Store with 30-minute features atomically
        
        Args:
            features: Calculated 30-minute features
            timestamp: Update timestamp
        """
        async with self._feature_store_lock:
            try:
                # Update 30-minute features
                for key, value in features.items():
                    self.feature_store[key] = value
                
                # Calculate interaction features after updating base features
                self._calculate_interaction_features()
                
                # Update metadata
                self.feature_store['last_update_30min'] = timestamp
                self.last_update_30min = timestamp
                
                # Always emit after 30-minute update (PRD requirement)
                await self._emit_indicators_ready()
                
            except Exception as e:
                self.logger.error("Error updating Feature Store (30min)", error=str(e))
    
    async def _check_and_emit_indicators_ready(self) -> None:
        """
        Check conditions and emit INDICATORS_READY event
        
        Per PRD: Emit when 5-min update completes AND 30-min features exist
        """
        try:
            # Only emit if we have 30-minute data (ensures complete feature set)
            if self.has_30min_data and self.last_update_5min:
                await self._emit_indicators_ready()
                
        except Exception as e:
            self.logger.error("Error checking indicators ready conditions", error=str(e))
    
    async def _emit_indicators_ready(self) -> None:
        """Emit INDICATORS_READY event with complete Feature Store"""
        try:
            # Create deep copy of Feature Store for event payload
            feature_store_copy = copy.deepcopy(self.feature_store)
            
            # Add metadata
            feature_store_copy['feature_count'] = len([k for k in feature_store_copy.keys() 
                                                     if not k.startswith('last_update')])
            feature_store_copy['calculation_status'] = 'complete'
            feature_store_copy['emission_timestamp'] = datetime.now()
            
            # Emit event
            self.publish_event(EventType.INDICATORS_READY, feature_store_copy)
            
            self.events_emitted += 1
            
            self.logger.info("INDICATORS_READY emitted",
                           feature_count=feature_store_copy['feature_count'],
                           last_5min=self.last_update_5min.isoformat() if self.last_update_5min else None,
                           last_30min=self.last_update_30min.isoformat() if self.last_update_30min else None)
            
        except Exception as e:
            self.logger.error("Error emitting INDICATORS_READY event", error=str(e))
    
    def get_current_features(self) -> Dict[str, Any]:
        """
        Get current Feature Store contents
        
        Returns:
            Copy of current Feature Store
        """
        return copy.deepcopy(self.feature_store)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of current feature status
        
        Returns:
            Feature summary dictionary
        """
        return {
            'total_features': len(self.feature_store),
            'calculations_5min': self.calculations_5min,
            'calculations_30min': self.calculations_30min,
            'events_emitted': self.events_emitted,
            'has_30min_data': self.has_30min_data,
            'last_update_5min': self.last_update_5min,
            'last_update_30min': self.last_update_30min,
            'active_fvgs': len(self.active_fvgs),
            'history_sizes': {
                '5min': len(self.history_5m),
                '30min': len(self.history_30m),
                'ha_30min': len(self.ha_history_30m)
            }
        }