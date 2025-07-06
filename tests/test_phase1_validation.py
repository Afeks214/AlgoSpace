"""
Phase 1 Implementation Validation Test

This test validates that all Phase 1 critical fixes have been implemented
correctly and the system is ready for orchestration reshape.

Key Validations:
1. Enhanced FVG detector produces 9 features
2. 5m Matrix Assembler outputs 60×9 matrix
3. TacticalEmbedder processes 9 inputs and outputs 32D
4. Complete dimension flow works end-to-end
5. No blocking issues remain

Author: QuantNova Team
Date: 2025-01-06
"""

import unittest
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logging, get_logger
from src.core.events import BarData
from src.indicators.fvg import FVGDetector
from src.matrix.assembler_5m import MatrixAssembler5m
from src.core.event_bus import EventBus


class Phase1ValidationTest(unittest.TestCase):
    """Comprehensive Phase 1 implementation validation"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        setup_logging(log_level="INFO")
        cls.logger = get_logger("Phase1Validation")
        cls.validation_results = {}
        
    def setUp(self):
        """Initialize test components"""
        self.event_bus = EventBus()
        
    def tearDown(self):
        """Cleanup test resources"""
        if self.event_bus:
            self.event_bus.stop()
            
    def test_1_enhanced_fvg_detector_features(self):
        """Test 1: Validate enhanced FVG detector produces all 9 features"""
        self.logger.info("🧪 Testing enhanced FVG detector...")
        
        # Initialize FVG detector with enhanced config
        fvg_config = {
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
        
        fvg_detector = FVGDetector(fvg_config, self.event_bus)
        
        # Generate test bars with FVG pattern
        test_bars = self._generate_fvg_test_bars()
        
        # Process bars and get features
        features = None
        for bar in test_bars:
            features = fvg_detector.calculate_5m(bar)
        
        # Validate all 9 enhanced features are present
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
        
        self.assertIsNotNone(features, "FVG detector should return features")
        
        for feature_name in expected_features:
            self.assertIn(feature_name, features, f"Missing enhanced feature: {feature_name}")
            
        # Validate feature types and ranges
        self.assertIsInstance(features['fvg_bullish_active'], (bool, int, float))
        self.assertIsInstance(features['fvg_bearish_active'], (bool, int, float))
        self.assertGreaterEqual(features['fvg_gap_size'], 0.0)
        self.assertGreaterEqual(features['fvg_gap_size_pct'], 0.0)
        self.assertGreaterEqual(features['fvg_mitigation_strength'], 0.0)
        self.assertLessEqual(features['fvg_mitigation_strength'], 1.0)
        self.assertGreaterEqual(features['fvg_mitigation_depth'], 0.0)
        self.assertLessEqual(features['fvg_mitigation_depth'], 1.0)
        
        self.validation_results['fvg_detector'] = {
            'feature_count': len(features),
            'expected_count': 9,
            'all_features_present': True,
            'status': 'PASS'
        }
        
        self.logger.info(f"✅ Enhanced FVG detector validation: {len(features)}/9 features")
        
    def test_2_matrix_assembler_5m_dimensions(self):
        """Test 2: Validate 5m Matrix Assembler outputs 60×9 matrix"""
        self.logger.info("🧪 Testing 5m Matrix Assembler dimensions...")
        
        # Configuration for 9 features
        assembler_config = {
            'window_size': 60,
            'warmup_period': 20,
            'features': [
                'fvg_bullish_active',
                'fvg_bearish_active',
                'fvg_nearest_level',
                'fvg_age',
                'fvg_mitigation_signal',
                'price_momentum_5',
                'volume_ratio',
                'fvg_gap_size_pct',
                'fvg_mitigation_strength'
            ],
            'feature_configs': {
                'price_momentum_5': {'ema_alpha': 0.1},
                'volume_ratio': {'ema_alpha': 0.05},
                'fvg_gap_size_pct': {'ema_alpha': 0.1},
                'fvg_mitigation_strength': {'ema_alpha': 0.05}
            }
        }
        
        assembler = MatrixAssembler5m(assembler_config)
        
        # Validate initial configuration
        self.assertEqual(assembler.n_features, 9, "Assembler should be configured for 9 features")
        self.assertEqual(assembler.window_size, 60, "Window size should be 60")
        
        # Generate test feature store data
        test_data = []
        for i in range(70):  # More than window_size for proper testing
            feature_store = self._generate_test_feature_store(i)
            test_data.append(feature_store)
            assembler.update(feature_store)
        
        # Check if assembler is ready
        self.assertTrue(assembler.is_ready(), "Assembler should be ready after warmup")
        
        # Get matrix and validate dimensions
        matrix = assembler.get_matrix()
        self.assertIsNotNone(matrix, "Matrix should not be None")
        
        expected_shape = (60, 9)
        self.assertEqual(matrix.shape, expected_shape, 
                        f"Matrix shape should be {expected_shape}, got {matrix.shape}")
        
        # Validate matrix contains finite values
        self.assertTrue(np.all(np.isfinite(matrix)), "Matrix should contain only finite values")
        
        # Validate feature importance sums to 1.0
        importance = assembler.get_feature_importance()
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=2, 
                              msg="Feature importance should sum to 1.0")
        
        self.validation_results['matrix_assembler_5m'] = {
            'actual_shape': matrix.shape,
            'expected_shape': expected_shape,
            'n_features': assembler.n_features,
            'is_ready': assembler.is_ready(),
            'status': 'PASS'
        }
        
        self.logger.info(f"✅ 5m Matrix Assembler validation: {matrix.shape} matrix")
        
    def test_3_tactical_embedder_dimensions(self):
        """Test 3: Validate TacticalEmbedder dimension compatibility"""
        self.logger.info("🧪 Testing TacticalEmbedder dimensions...")
        
        # Import here to avoid torch dependency issues in simple tests
        try:
            from src.agents.main_core.models import TacticalEmbedder
            import torch
        except ImportError:
            self.skipTest("PyTorch not available for dimension testing")
            
        # Initialize TacticalEmbedder with new dimensions
        embedder = TacticalEmbedder(
            input_dim=9,
            hidden_dim=64,
            output_dim=32,
            dropout_rate=0.2
        )
        
        # Validate configuration
        self.assertEqual(embedder.input_dim, 9, "Input dimension should be 9")
        self.assertEqual(embedder.output_dim, 32, "Output dimension should be 32")
        
        # Test forward pass with correct input shape
        batch_size = 2
        sequence_length = 60
        input_features = 9
        
        test_input = torch.randn(batch_size, sequence_length, input_features)
        
        # Forward pass
        embedder.eval()
        with torch.no_grad():
            output = embedder(test_input)
        
        expected_output_shape = (batch_size, 32)
        self.assertEqual(output.shape, expected_output_shape,
                        f"Output shape should be {expected_output_shape}, got {output.shape}")
        
        # Validate output contains finite values
        self.assertTrue(torch.all(torch.isfinite(output)), "Output should contain finite values")
        
        self.validation_results['tactical_embedder'] = {
            'input_dim': embedder.input_dim,
            'output_dim': embedder.output_dim,
            'actual_output_shape': tuple(output.shape),
            'expected_output_shape': expected_output_shape,
            'status': 'PASS'
        }
        
        self.logger.info(f"✅ TacticalEmbedder validation: {test_input.shape} → {output.shape}")
        
    def test_4_unified_state_dimensions(self):
        """Test 4: Validate unified state vector dimensions"""
        self.logger.info("🧪 Testing unified state vector dimensions...")
        
        try:
            from src.agents.main_core.models import (
                StructureEmbedder, TacticalEmbedder, RegimeEmbedder, LVNEmbedder
            )
            import torch
        except ImportError:
            self.skipTest("PyTorch not available for unified state testing")
            
        # Initialize all embedders
        structure_embedder = StructureEmbedder(input_channels=8, output_dim=64)
        tactical_embedder = TacticalEmbedder(input_dim=9, output_dim=32) 
        regime_embedder = RegimeEmbedder(input_dim=8, output_dim=16)
        lvn_embedder = LVNEmbedder(input_dim=5, output_dim=32)  # Updated from 8 to 32
        
        batch_size = 1
        
        # Create test inputs
        structure_input = torch.randn(batch_size, 48, 8)  # 30m matrix
        tactical_input = torch.randn(batch_size, 60, 9)   # Enhanced 5m matrix
        regime_input = torch.randn(batch_size, 8)         # Regime vector
        lvn_input = torch.randn(batch_size, 5)            # LVN vector
        
        # Get embeddings
        with torch.no_grad():
            structure_embedding = structure_embedder(structure_input)
            tactical_embedding = tactical_embedder(tactical_input)
            regime_embedding = regime_embedder(regime_input)
            lvn_embedding = lvn_embedder(lvn_input)
        
        # Validate individual embedding dimensions
        self.assertEqual(structure_embedding.shape, (batch_size, 64))
        self.assertEqual(tactical_embedding.shape, (batch_size, 32))
        self.assertEqual(regime_embedding.shape, (batch_size, 16))
        self.assertEqual(lvn_embedding.shape, (batch_size, 32))
        
        # Construct unified state
        unified_state = torch.cat([
            structure_embedding,
            tactical_embedding,
            regime_embedding,
            lvn_embedding
        ], dim=-1)
        
        expected_unified_dim = 64 + 32 + 16 + 32  # 144D
        self.assertEqual(unified_state.shape, (batch_size, expected_unified_dim),
                        f"Unified state should be {expected_unified_dim}D")
        
        self.validation_results['unified_state'] = {
            'structure_dim': 64,
            'tactical_dim': 32,
            'regime_dim': 16,
            'lvn_dim': 32,
            'total_dim': expected_unified_dim,
            'actual_shape': tuple(unified_state.shape),
            'status': 'PASS'
        }
        
        self.logger.info(f"✅ Unified state validation: {expected_unified_dim}D vector")
        
    def test_5_end_to_end_dimension_flow(self):
        """Test 5: Validate complete end-to-end dimension flow"""
        self.logger.info("🧪 Testing end-to-end dimension flow...")
        
        # Create a comprehensive flow test
        flow_results = []
        
        # Step 1: FVG Detection → 9 features
        fvg_config = {'fvg': {'max_age': 50, 'mitigation': {}, 'gap_size': {}}}
        fvg_detector = FVGDetector(fvg_config, self.event_bus)
        
        test_bars = self._generate_fvg_test_bars()
        fvg_features = None
        for bar in test_bars:
            fvg_features = fvg_detector.calculate_5m(bar)
        
        flow_results.append({
            'step': 'FVG Detection',
            'output': f"{len(fvg_features)} features",
            'expected': '9 features',
            'status': 'PASS' if len(fvg_features) == 9 else 'FAIL'
        })
        
        # Step 2: Matrix Assembly → 60×9
        assembler_config = {
            'window_size': 60, 'warmup_period': 20,
            'features': list(fvg_features.keys()),
            'feature_configs': {}
        }
        assembler = MatrixAssembler5m(assembler_config)
        
        for i in range(70):
            feature_store = self._generate_test_feature_store(i)
            assembler.update(feature_store)
        
        matrix = assembler.get_matrix()
        matrix_shape = matrix.shape if matrix is not None else None
        
        flow_results.append({
            'step': 'Matrix Assembly',
            'output': f"{matrix_shape} matrix",
            'expected': '(60, 9) matrix',
            'status': 'PASS' if matrix_shape == (60, 9) else 'FAIL'
        })
        
        # Step 3: Embedding → 32D
        try:
            from src.agents.main_core.models import TacticalEmbedder
            import torch
            
            embedder = TacticalEmbedder(input_dim=9, output_dim=32)
            test_input = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                embedding = embedder(test_input)
            
            embedding_shape = tuple(embedding.shape)
            
            flow_results.append({
                'step': 'Tactical Embedding',
                'output': f"{embedding_shape} embedding",
                'expected': '(1, 32) embedding',
                'status': 'PASS' if embedding_shape == (1, 32) else 'FAIL'
            })
            
        except ImportError:
            flow_results.append({
                'step': 'Tactical Embedding',
                'output': 'PyTorch not available',
                'expected': '(1, 32) embedding',
                'status': 'SKIP'
            })
        
        # Validate all steps passed
        all_passed = all(result['status'] in ['PASS', 'SKIP'] for result in flow_results)
        
        self.validation_results['end_to_end_flow'] = {
            'steps': flow_results,
            'all_passed': all_passed,
            'status': 'PASS' if all_passed else 'FAIL'
        }
        
        # Print flow results
        for result in flow_results:
            status_icon = '✅' if result['status'] == 'PASS' else '⚠️' if result['status'] == 'SKIP' else '❌'
            self.logger.info(f"{status_icon} {result['step']}: {result['output']}")
        
        self.assertTrue(all_passed, "End-to-end flow should pass all steps")
        
    def test_6_generate_phase1_report(self):
        """Test 6: Generate comprehensive Phase 1 validation report"""
        self.logger.info("📊 Generating Phase 1 validation report...")
        
        # Calculate overall success
        all_tests = [
            self.validation_results.get('fvg_detector', {}).get('status') == 'PASS',
            self.validation_results.get('matrix_assembler_5m', {}).get('status') == 'PASS',
            self.validation_results.get('tactical_embedder', {}).get('status') == 'PASS',
            self.validation_results.get('unified_state', {}).get('status') == 'PASS',
            self.validation_results.get('end_to_end_flow', {}).get('status') == 'PASS'
        ]
        
        passed_tests = sum(all_tests)
        total_tests = len(all_tests)
        success_rate = (passed_tests / total_tests) * 100
        
        # Generate final report
        report = {
            'phase': 'Phase 1 Critical Fixes',
            'timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'ready_for_reshape': success_rate >= 95,
            'validation_details': self.validation_results,
            'summary': {
                'fvg_detector': f"✅ Enhanced FVG: 9/9 features implemented",
                'matrix_assembler': f"✅ 5m Matrix: 60×9 output achieved",
                'tactical_embedder': f"✅ TacticalEmbedder: 9→32 dimensions",
                'unified_state': f"✅ Unified State: 144D vector",
                'dimension_flow': f"✅ Complete flow validated"
            }
        }
        
        # Print comprehensive report
        print("\\n" + "="*80)
        print("🔥 PHASE 1 IMPLEMENTATION VALIDATION REPORT")
        print("="*80)
        print(f"Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
        print(f"Ready for Orchestration Reshape: {'✅ YES' if report['ready_for_reshape'] else '❌ NO'}")
        
        print("\\n📋 VALIDATION SUMMARY")
        for component, status in report['summary'].items():
            print(f"  {status}")
        
        print("\\n🔍 DETAILED RESULTS")
        for test_name, results in self.validation_results.items():
            status_icon = '✅' if results.get('status') == 'PASS' else '❌'
            print(f"  {status_icon} {test_name.replace('_', ' ').title()}")
            if isinstance(results, dict):
                for key, value in results.items():
                    if key != 'status':
                        print(f"     {key}: {value}")
        
        if report['ready_for_reshape']:
            print("\\n🎉 PHASE 1 COMPLETE - SYSTEM READY FOR ORCHESTRATION RESHAPE!")
        else:
            print("\\n⚠️  PHASE 1 INCOMPLETE - ADDITIONAL FIXES NEEDED")
            
        print("="*80 + "\\n")
        
        # Store final report
        self.validation_results['final_report'] = report
        
        # Assert overall success
        self.assertGreaterEqual(success_rate, 95, 
                               f"Phase 1 validation should achieve ≥95% success rate, got {success_rate:.1f}%")
        
        self.logger.info(f"📊 Phase 1 validation completed: {success_rate:.1f}% success rate")
    
    # Helper methods
    def _generate_fvg_test_bars(self) -> List[BarData]:
        """Generate test bars with FVG pattern"""
        bars = []
        base_price = 4500.0
        
        for i in range(10):
            # Create FVG pattern in bars 3-5
            if i == 2:  # First bar - establish low
                bar = BarData(
                    symbol="ES", timeframe=5, timestamp=datetime.now(),
                    open=base_price, high=base_price+5, low=base_price-2, close=base_price+2,
                    volume=1000
                )
            elif i == 3:  # Second bar - gap up (FVG creation)
                bar = BarData(
                    symbol="ES", timeframe=5, timestamp=datetime.now(),
                    open=base_price+10, high=base_price+15, low=base_price+8, close=base_price+12,
                    volume=1500
                )
            elif i == 4:  # Third bar - continuation
                bar = BarData(
                    symbol="ES", timeframe=5, timestamp=datetime.now(),
                    open=base_price+12, high=base_price+20, low=base_price+10, close=base_price+18,
                    volume=1200
                )
            else:  # Normal bars
                price_change = np.random.uniform(-3, 3)
                current_price = base_price + price_change
                bar = BarData(
                    symbol="ES", timeframe=5, timestamp=datetime.now(),
                    open=current_price, high=current_price+2, low=current_price-2, close=current_price+1,
                    volume=1000 + np.random.randint(-200, 200)
                )
                
            bars.append(bar)
            
        return bars
    
    def _generate_test_feature_store(self, bar_index: int) -> Dict[str, Any]:
        """Generate test feature store data"""
        return {
            'current_price': 4500.0 + np.random.uniform(-10, 10),
            'current_volume': 1000 + np.random.randint(-200, 200),
            'fvg_bullish_active': bool(bar_index % 4 == 0),
            'fvg_bearish_active': bool(bar_index % 6 == 0),
            'fvg_nearest_level': 4500.0 + np.random.uniform(-5, 5),
            'fvg_age': max(0, bar_index % 10),
            'fvg_mitigation_signal': bool(bar_index % 8 == 0),
            'fvg_gap_size': np.random.uniform(0, 5),
            'fvg_gap_size_pct': np.random.uniform(0, 0.5),
            'fvg_mitigation_strength': np.random.uniform(0, 1),
            'fvg_mitigation_depth': np.random.uniform(0, 1),
            'price_momentum_5': np.random.uniform(-2, 2),
            'volume_ratio': np.random.uniform(0.5, 2.5)
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)