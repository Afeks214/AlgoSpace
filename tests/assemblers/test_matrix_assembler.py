"""
Unit tests for refactored Matrix Assembler components

Tests configuration-driven initialization, robust error handling,
and missing feature graceful degradation.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import Dict, Any, List

from src.matrix.base import BaseMatrixAssembler
from src.matrix.assembler_30m import MatrixAssembler30m
from src.matrix.assembler_5m import MatrixAssembler5m
from src.matrix.assembler_regime import MatrixAssemblerRegime
from src.core.events import Event, EventType


class ConcreteMatrixAssembler(BaseMatrixAssembler):
    """Concrete implementation for testing base class"""
    
    def extract_features(self, feature_store: Dict[str, Any]) -> List[float]:
        # Return None to trigger safe extraction
        return None
    
    def preprocess_features(self, raw_features: List[float], feature_store: Dict[str, Any]) -> np.ndarray:
        # Simple preprocessing - just convert to numpy array
        return np.array(raw_features, dtype=np.float32)


@pytest.fixture
def mock_kernel():
    """Create a mock kernel with event bus"""
    kernel = Mock()
    kernel.get_event_bus = Mock(return_value=Mock())
    kernel.config = {
        'matrix_assemblers': {
            '30m': {
                'window_size': 48,
                'features': ['mlmi_value', 'nwrqk_value']
            }
        }
    }
    return kernel


@pytest.fixture
def basic_config(mock_kernel):
    """Basic configuration for testing"""
    return {
        'name': 'TestAssembler',
        'window_size': 10,
        'features': ['feature_1', 'feature_2', 'feature_3'],
        'kernel': mock_kernel,
        'warmup_period': 5
    }


@pytest.fixture
def complete_feature_store():
    """Complete feature store with all features"""
    return {
        'feature_1': 1.0,
        'feature_2': 2.0,
        'feature_3': 3.0,
        'mlmi_value': 50.0,
        'nwrqk_value': 15000.0,
        'current_price': 15000.0,
        'timestamp': datetime.now()
    }


@pytest.fixture
def incomplete_feature_store():
    """Feature store missing some features"""
    return {
        'feature_1': 1.0,
        # feature_2 is missing
        'feature_3': 3.0,
        'current_price': 15000.0,
        'timestamp': datetime.now()
    }


class TestBaseMatrixAssembler:
    """Test the base matrix assembler class"""
    
    def test_assembler_initializes_from_config(self, basic_config):
        """Test that assembler correctly initializes from configuration"""
        assembler = ConcreteMatrixAssembler(basic_config)
        
        assert assembler.name == 'TestAssembler'
        assert assembler.window_size == 10
        assert assembler.feature_names == ['feature_1', 'feature_2', 'feature_3']
        assert assembler.n_features == 3
        assert assembler.matrix.shape == (10, 3)
        assert assembler._warmup_period == 5
        
    def test_config_validation(self, mock_kernel):
        """Test configuration validation"""
        # Missing window_size
        with pytest.raises(ValueError, match="window_size is required"):
            ConcreteMatrixAssembler({
                'features': ['f1', 'f2'],
                'kernel': mock_kernel
            })
        
        # Missing features
        with pytest.raises(ValueError, match="features list is required"):
            ConcreteMatrixAssembler({
                'window_size': 10,
                'kernel': mock_kernel
            })
        
        # Missing kernel
        with pytest.raises(ValueError, match="Kernel reference is required"):
            ConcreteMatrixAssembler({
                'window_size': 10,
                'features': ['f1', 'f2']
            })
    
    def test_safe_feature_extraction(self, basic_config, complete_feature_store):
        """Test safe feature extraction with all features present"""
        assembler = ConcreteMatrixAssembler(basic_config)
        
        # Extract features safely
        features = assembler._extract_features_safely(complete_feature_store)
        
        assert features is not None
        assert len(features) == 3
        assert features == [1.0, 2.0, 3.0]
        assert len(assembler.missing_feature_warnings) == 0
    
    def test_assembler_handles_missing_feature_gracefully(self, basic_config, incomplete_feature_store, caplog):
        """Test that assembler handles missing features without crashing"""
        assembler = ConcreteMatrixAssembler(basic_config)
        
        # Set up logging capture
        with caplog.at_level(logging.WARNING):
            # Extract features with missing feature_2
            features = assembler._extract_features_safely(incomplete_feature_store)
        
        # Verify extraction succeeded with default value
        assert features is not None
        assert len(features) == 3
        assert features == [1.0, 0.0, 3.0]  # feature_2 defaults to 0.0
        
        # Verify warning was logged
        assert "Feature 'feature_2' not found in Feature Store" in caplog.text
        assert "Using default value 0.0" in caplog.text
        
        # Verify missing feature tracking
        assert 'feature_2' in assembler.missing_feature_warnings
        assert assembler.missing_feature_warnings['feature_2'] == 1
    
    def test_missing_feature_warning_throttling(self, basic_config, incomplete_feature_store):
        """Test that missing feature warnings are throttled"""
        assembler = ConcreteMatrixAssembler(basic_config)
        
        # Mock logger to count warnings
        warning_count = 0
        original_warning = assembler.logger.warning
        
        def count_warnings(msg):
            nonlocal warning_count
            if "not found in Feature Store" in msg:
                warning_count += 1
            original_warning(msg)
        
        assembler.logger.warning = count_warnings
        
        # Extract features multiple times
        for i in range(10):
            assembler._extract_features_safely(incomplete_feature_store)
        
        # Should only warn 5 times (first 5 occurrences)
        assert warning_count == 5
        assert assembler.missing_feature_warnings['feature_2'] == 10
    
    def test_non_numeric_feature_handling(self, basic_config, caplog):
        """Test handling of non-numeric feature values"""
        assembler = ConcreteMatrixAssembler(basic_config)
        
        feature_store = {
            'feature_1': 1.0,
            'feature_2': 'not_a_number',  # Invalid
            'feature_3': None  # Invalid
        }
        
        with caplog.at_level(logging.WARNING):
            features = assembler._extract_features_safely(feature_store)
        
        assert features == [1.0, 0.0, 0.0]
        assert "has non-numeric value" in caplog.text
    
    def test_matrix_update_with_missing_features(self, basic_config, incomplete_feature_store):
        """Test full matrix update flow with missing features"""
        assembler = ConcreteMatrixAssembler(basic_config)
        
        # Update matrix
        assembler._update_matrix(incomplete_feature_store)
        
        # Verify update succeeded
        assert assembler.n_updates == 1
        assert not assembler.is_full
        
        # Check that the matrix contains expected values
        first_row = assembler.matrix[0]
        assert first_row[0] == 1.0
        assert first_row[1] == 0.0  # Missing feature defaulted to 0
        assert first_row[2] == 3.0
    
    def test_validate_matrix_with_missing_features(self, basic_config, incomplete_feature_store):
        """Test matrix validation identifies frequently missing features"""
        assembler = ConcreteMatrixAssembler(basic_config)
        
        # Update matrix many times with missing feature
        for _ in range(10):
            assembler._update_matrix(incomplete_feature_store)
        
        # Validate matrix
        is_valid, issues = assembler.validate_matrix()
        
        # Should identify critical missing features
        assert not is_valid
        assert any("Critical features frequently missing" in issue for issue in issues)
        assert any("feature_2" in issue for issue in issues)


class TestMatrixAssembler30m:
    """Test the 30-minute matrix assembler"""
    
    def test_30m_assembler_config_driven(self, mock_kernel):
        """Test 30m assembler uses configuration"""
        config = {
            'name': 'Test30m',
            'window_size': 24,  # Different from default
            'features': ['mlmi_value', 'nwrqk_value'],  # Subset of features
            'kernel': mock_kernel
        }
        
        assembler = MatrixAssembler30m(config)
        
        assert assembler.window_size == 24
        assert assembler.n_features == 2
        assert assembler.feature_names == ['mlmi_value', 'nwrqk_value']
    
    def test_30m_feature_extraction_with_missing(self, mock_kernel):
        """Test 30m assembler handles missing features"""
        config = {
            'name': 'Test30m',
            'window_size': 48,
            'features': ['mlmi_value', 'nwrqk_value', 'lvn_nearest_strength'],
            'kernel': mock_kernel
        }
        
        assembler = MatrixAssembler30m(config)
        
        # Feature store missing lvn_nearest_strength
        feature_store = {
            'mlmi_value': 75.0,
            'nwrqk_value': 15100.0,
            'current_price': 15000.0
        }
        
        # Should not crash
        features = assembler.extract_features(feature_store)
        
        # Should use safe extraction
        assert features is None  # Triggers safe extraction
    
    def test_30m_preprocessing_robustness(self, mock_kernel):
        """Test 30m preprocessing handles edge cases"""
        config = {
            'name': 'Test30m',
            'window_size': 48,
            'features': ['mlmi_value', 'nwrqk_value', 'time_hour_sin'],
            'kernel': mock_kernel
        }
        
        assembler = MatrixAssembler30m(config)
        assembler.current_price = 15000.0
        
        # Test with out-of-range values
        raw_features = [
            150.0,  # mlmi_value > 100
            0.0,    # nwrqk_value = 0
            25.0    # hour > 24
        ]
        
        processed = assembler.preprocess_features(raw_features, {})
        
        # Should handle gracefully
        assert np.all(np.isfinite(processed))
        assert processed.shape == (3,)


class TestMatrixAssembler5m:
    """Test the 5-minute matrix assembler"""
    
    def test_5m_assembler_config_driven(self, mock_kernel):
        """Test 5m assembler uses configuration"""
        config = {
            'name': 'Test5m',
            'window_size': 30,  # Different from default
            'features': ['fvg_bullish_active', 'fvg_bearish_active', 'price_momentum_5'],
            'kernel': mock_kernel
        }
        
        assembler = MatrixAssembler5m(config)
        
        assert assembler.window_size == 30
        assert assembler.n_features == 3
        assert assembler.feature_names == ['fvg_bullish_active', 'fvg_bearish_active', 'price_momentum_5']
    
    def test_5m_custom_feature_calculation(self, mock_kernel):
        """Test 5m custom feature calculations"""
        config = {
            'name': 'Test5m',
            'window_size': 60,
            'features': ['price_momentum_5', 'volume_ratio'],
            'kernel': mock_kernel
        }
        
        assembler = MatrixAssembler5m(config)
        
        # Populate price history
        for price in [14900, 14920, 14940, 14960, 14980, 15000]:
            assembler.price_history.append(price)
        
        # Set volume EMA
        assembler.volume_ema = 1000.0
        
        feature_store = {
            'current_price': 15000.0,
            'current_volume': 1500.0
        }
        
        features = assembler.extract_features(feature_store)
        
        assert features is not None
        assert len(features) == 2
        
        # Check momentum calculation
        # Note: extract_features appends current_price to price_history,
        # which pushes out the oldest price due to maxlen=6
        # So momentum is calculated from 14920 to 15000, not 14900 to 15000
        momentum = features[0]
        expected_momentum = ((15000 - 14920) / 14920) * 100
        assert abs(momentum - expected_momentum) < 0.01
        
        # Check volume ratio
        # Note: extract_features updates volume_ema before calculating ratio
        # EMA update: 1000 + 0.02 * (1500 - 1000) = 1010
        volume_ratio = features[1]
        expected_ratio = 1500.0 / 1010.0
        assert abs(volume_ratio - expected_ratio) < 0.001


class TestMatrixAssemblerRegime:
    """Test the regime detection matrix assembler"""
    
    def test_regime_assembler_dynamic_mmd(self, mock_kernel):
        """Test regime assembler handles dynamic MMD dimensions"""
        config = {
            'name': 'TestRegime',
            'window_size': 96,
            'features': ['mmd_features', 'volatility_30'],
            'kernel': mock_kernel
        }
        
        assembler = MatrixAssemblerRegime(config)
        
        # Initially MMD dimension is unknown
        assert assembler.mmd_dimension == -1
        
        feature_store = {
            'mmd_features': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # 5 dimensions
            'current_price': 15000.0
        }
        
        features = assembler.extract_features(feature_store)
        
        # Should handle dynamic dimension
        assert features is not None
        assert len(features) == 6  # 5 MMD + 1 volatility
        assert assembler.mmd_dimension == 5
    
    def test_regime_custom_calculations(self, mock_kernel):
        """Test regime assembler custom calculations"""
        config = {
            'name': 'TestRegime',
            'window_size': 96,
            'features': ['volatility_30', 'volume_profile_skew', 'price_acceleration'],
            'kernel': mock_kernel
        }
        
        assembler = MatrixAssemblerRegime(config)
        
        # Populate price history for volatility
        prices = [15000 + i * 10 for i in range(20)]
        for price in prices:
            assembler.price_history.append(price)
        
        # Populate volume history for skew
        volumes = [1000 + i * 50 for i in range(20)]
        for volume in volumes:
            assembler.volume_history.append(volume)
        
        feature_store = {}
        features = assembler.extract_features(feature_store)
        
        assert features is not None
        assert len(features) == 3
        
        # Check that calculations produced values
        volatility, skew, acceleration = features
        assert volatility > 0  # Should have calculated volatility
        assert isinstance(skew, float)
        assert isinstance(acceleration, float)


class TestIntegration:
    """Integration tests for the refactored system"""
    
    def test_kernel_assembler_integration(self, mock_kernel):
        """Test that kernel properly instantiates assemblers with config"""
        from src.core.kernel import AlgoSpaceKernel
        
        # Create a more complete mock kernel
        kernel = AlgoSpaceKernel()
        kernel.config = {
            'data_handler': {'type': 'backtest'},
            'execution': {},
            'risk_management': {},
            'agents': {},
            'models': {},
            'matrix_assemblers': {
                '30m': {
                    'window_size': 48,
                    'features': ['mlmi_value', 'nwrqk_value']
                },
                '5m': {
                    'window_size': 60,
                    'features': ['fvg_bullish_active', 'price_momentum_5']
                },
                'regime': {
                    'window_size': 96,
                    'features': ['mmd_features', 'volatility_30']
                }
            }
        }
        
        # Mock all the required components
        with patch('src.core.kernel.BacktestDataHandler'), \
             patch('src.core.kernel.BarGenerator'), \
             patch('src.core.kernel.IndicatorEngine'), \
             patch('src.core.kernel.MatrixAssembler30m') as Mock30m, \
             patch('src.core.kernel.MatrixAssembler5m') as Mock5m, \
             patch('src.core.kernel.MatrixAssemblerRegime') as MockRegime, \
             patch('src.core.kernel.SynergyDetector'), \
             patch('src.core.kernel.RDEComponent'), \
             patch('src.core.kernel.MRMSComponent'), \
             patch('src.core.kernel.MainMARLCoreComponent'), \
             patch('src.core.kernel.BacktestExecutionHandler'):
            
            # Call the instantiation method
            kernel._instantiate_components()
            
            # Verify assemblers were called with correct config
            Mock30m.assert_called_once()
            config_30m = Mock30m.call_args[0][0]
            assert config_30m['window_size'] == 48
            assert config_30m['features'] == ['mlmi_value', 'nwrqk_value']
            assert config_30m['kernel'] == kernel
            
            Mock5m.assert_called_once()
            config_5m = Mock5m.call_args[0][0]
            assert config_5m['window_size'] == 60
            assert config_5m['features'] == ['fvg_bullish_active', 'price_momentum_5']
            
            MockRegime.assert_called_once()
            config_regime = MockRegime.call_args[0][0]
            assert config_regime['window_size'] == 96
            assert config_regime['features'] == ['mmd_features', 'volatility_30']
    
    def test_event_flow_with_missing_features(self, mock_kernel):
        """Test event flow handles missing features gracefully"""
        config = {
            'name': 'TestAssembler',
            'window_size': 10,
            'features': ['feature_1', 'feature_2', 'feature_3'],
            'kernel': mock_kernel
        }
        
        assembler = ConcreteMatrixAssembler(config)
        
        # Create event with incomplete feature store
        event = Event(
            event_type=EventType.INDICATORS_READY,
            timestamp=datetime.now(),
            source='test',
            payload={
                'feature_1': 1.0,
                # feature_2 missing
                'feature_3': 3.0,
                'emission_timestamp': datetime.now()
            }
        )
        
        # Process event
        assembler._on_indicators_ready(event)
        
        # Should have processed successfully
        assert assembler.n_updates == 1
        assert assembler.missing_feature_warnings.get('feature_2', 0) > 0
    
    def test_performance_with_missing_features(self, basic_config):
        """Test that missing features don't significantly impact performance"""
        assembler = ConcreteMatrixAssembler(basic_config)
        
        # Time with complete features
        complete_store = {f'feature_{i}': float(i) for i in range(1, 4)}
        
        import time
        start = time.time()
        for _ in range(1000):
            assembler._extract_features_safely(complete_store)
        complete_time = time.time() - start
        
        # Time with missing features
        incomplete_store = {'feature_1': 1.0, 'feature_3': 3.0}
        
        start = time.time()
        for _ in range(1000):
            assembler._extract_features_safely(incomplete_store)
        incomplete_time = time.time() - start
        
        # Both should complete quickly (under 100ms for 1000 iterations)
        assert complete_time < 0.1, f"Complete extraction too slow: {complete_time}s"
        assert incomplete_time < 0.1, f"Incomplete extraction too slow: {incomplete_time}s"
        
        # Log the performance for informational purposes
        print(f"Complete: {complete_time:.4f}s, Incomplete: {incomplete_time:.4f}s, "
              f"Ratio: {incomplete_time/complete_time:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])