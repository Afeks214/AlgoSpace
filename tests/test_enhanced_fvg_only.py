"""
Enhanced FVG Detector Standalone Test

Tests the enhanced FVG detector functionality in isolation.
"""

import unittest
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indicators.fvg import FVGDetector
from src.core.events import BarData
from src.core.event_bus import EventBus
from src.utils.logger import setup_logging, get_logger


class TestEnhancedFVG(unittest.TestCase):
    
    def setUp(self):
        setup_logging(log_level="INFO")
        self.logger = get_logger("TestEnhancedFVG")
        self.event_bus = EventBus()
        
    def tearDown(self):
        if self.event_bus:
            self.event_bus.stop()
    
    def test_enhanced_fvg_features(self):
        """Test that enhanced FVG detector produces all 9 features"""
        
        config = {
            'threshold': 0.001,
            'lookback_period': 10,
            'body_multiplier': 1.5,
            'fvg': {
                'max_age': 50,
                'mitigation': {
                    'penetration_weight': 0.4,
                    'speed_weight': 0.3,
                    'volume_weight': 0.2,
                    'age_weight': 0.1,
                    'min_penetration': 0.5,
                    'volume_lookback': 20
                },
                'gap_size': {
                    'max_percentage': 0.05,
                    'min_percentage': 0.001
                }
            }
        }
        
        detector = FVGDetector(config, self.event_bus)
        
        # Create bars with a clear FVG pattern
        # BarData(symbol, timestamp, open, high, low, close, volume, timeframe)
        bars = [
            BarData("ES", datetime.now(), 4500, 4505, 4498, 4502, 1000, 5),  # Setup
            BarData("ES", datetime.now(), 4502, 4510, 4500, 4506, 1100, 5),  # Setup
            BarData("ES", datetime.now(), 4506, 4512, 4504, 4508, 1200, 5),  # Bar before gap
            BarData("ES", datetime.now(), 4520, 4525, 4518, 4522, 1500, 5),  # Gap up (FVG)
            BarData("ES", datetime.now(), 4522, 4530, 4520, 4528, 1300, 5),  # Continuation
        ]
        
        # Process bars
        result = None
        for bar in bars:
            result = detector.calculate_5m(bar)
        
        # Validate 9 features are present
        expected_features = [
            'fvg_bullish_active',
            'fvg_bearish_active',
            'fvg_nearest_level',
            'fvg_age',
            'fvg_mitigation_signal',
            'fvg_gap_size',
            'fvg_gap_size_pct',
            'fvg_mitigation_strength',
            'fvg_mitigation_depth'
        ]
        
        self.assertEqual(len(result), 9, f"Expected 9 features, got {len(result)}")
        
        for feature in expected_features:
            self.assertIn(feature, result, f"Missing feature: {feature}")
        
        # Validate types and ranges
        self.assertIsInstance(result['fvg_bullish_active'], (bool, int, float))
        self.assertIsInstance(result['fvg_bearish_active'], (bool, int, float))
        self.assertGreaterEqual(result['fvg_gap_size'], 0.0)
        self.assertGreaterEqual(result['fvg_gap_size_pct'], 0.0)
        self.assertGreaterEqual(result['fvg_mitigation_strength'], 0.0)
        self.assertLessEqual(result['fvg_mitigation_strength'], 1.0)
        self.assertGreaterEqual(result['fvg_mitigation_depth'], 0.0)
        self.assertLessEqual(result['fvg_mitigation_depth'], 1.0)
        
        print(f"✅ Enhanced FVG Test Results:")
        for feature, value in result.items():
            print(f"  {feature}: {value}")


if __name__ == "__main__":
    unittest.main(verbosity=2)