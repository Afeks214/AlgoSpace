"""Simplified synergy detector without torch dependencies"""
from typing import Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SynergyDetector:
    """
    Simple synergy detector that works without torch dependencies.
    This is a production-ready fallback version.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.time_window_bars = config.get('time_window_bars', 10)
        self.required_signals = config.get('required_signals', 3)
        self.last_synergy_time = 0
        self.cooldown_bars = config.get('cooldown_bars', 5)
        
    def detect_synergy(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect synergy from market data.
        
        Args:
            market_data: Dictionary containing market features
            
        Returns:
            Synergy detection result or None
        """
        try:
            # Extract basic features
            mlmi_value = market_data.get('mlmi_value', 50)
            nwrqk_slope = market_data.get('nwrqk_slope', 0.0)
            fvg_active = market_data.get('fvg_bullish_active', False) or market_data.get('fvg_bearish_active', False)
            
            # Simple synergy logic
            signals = []
            
            # MLMI signal
            if abs(mlmi_value - 50) > 20:  # Significant deviation from neutral
                signals.append('mlmi')
            
            # NWRQK signal
            if abs(nwrqk_slope) > 0.5:  # Significant slope
                signals.append('nwrqk')
            
            # FVG signal
            if fvg_active:
                signals.append('fvg')
            
            # Check if we have enough signals
            if len(signals) >= self.required_signals:
                # Determine direction
                direction = 'bullish' if mlmi_value > 50 else 'bearish'
                
                return {
                    'direction': direction,
                    'strength': min(len(signals) / self.required_signals, 1.0),
                    'signals': signals,
                    'timestamp': market_data.get('timestamp', 0),
                    'confidence': 0.7 + (len(signals) - self.required_signals) * 0.1
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in synergy detection: {e}")
            return None
    
    def is_ready(self) -> bool:
        """Check if detector is ready"""
        return True