# tests/core/test_kernel.py
"""
Unit tests for the AlgoSpaceKernel class.
"""
import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

from src.core.kernel import AlgoSpaceKernel


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock configuration file for testing."""
    config_data = {
        'data_handler': {
            'type': 'backtest',
            'backtest_file': 'data/historical/ES - 5 min.csv',
            'replay_speed': 1.0,
            'config': {
                'lookback_days': 60,
                'cache_enabled': True,
                'cache_path': './data/cache/'
            }
        },
        'execution': {
            'order_type': 'limit',
            'slippage_ticks': 1,
            'commission_per_contract': 2.5
        },
        'risk_management': {
            'max_position_size': 100000,
            'max_daily_loss': 5000,
            'max_drawdown_percent': 10,
            'stop_loss_percent': 2.0,
            'position_sizing_method': 'kelly'
        },
        'agents': {
            'agent_30m': {'enabled': True, 'model_path': './models/agent_30m.pth'},
            'agent_5m': {'enabled': True, 'model_path': './models/agent_5m.pth'},
            'agent_regime': {'enabled': True, 'model_path': './models/agent_regime.pth'},
            'agent_risk': {'enabled': True, 'model_path': './models/agent_risk.pth'}
        },
        'models': {
            'rde_path': './models/hybrid_regime_engine.pth',
            'mrms_path': './models/m_rms_model.pth'
        }
    }
    
    config_path = tmp_path / "settings.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return str(config_path)


@pytest.fixture
def mock_components():
    """Create mock component classes."""
    mocks = {
        'BacktestDataHandler': Mock(return_value=Mock()),
        'LiveDataHandler': Mock(return_value=Mock()),
        'BarGenerator': Mock(return_value=Mock()),
        'IndicatorEngine': Mock(return_value=Mock()),
        'MatrixAssembler30m': Mock(return_value=Mock()),
        'MatrixAssembler5m': Mock(return_value=Mock()),
        'MatrixAssemblerRegime': Mock(return_value=Mock()),
        'RDEComponent': Mock(return_value=Mock()),
        'MRMSComponent': Mock(return_value=Mock()),
        'MainMARLCoreComponent': Mock(return_value=Mock()),
        'SynergyDetector': Mock(return_value=Mock()),
        'BacktestExecutionHandler': Mock(return_value=Mock()),
        'LiveExecutionHandler': Mock(return_value=Mock()),
    }
    
    # Add required methods to mocked components
    for component_name, component_mock in mocks.items():
        instance = component_mock.return_value
        # Create mocks with proper __name__ attributes
        instance.on_new_tick = Mock(__name__='on_new_tick')
        instance.on_new_bar = Mock(__name__='on_new_bar')
        instance.on_indicators_ready = Mock(__name__='on_indicators_ready')
        instance.check_synergy = Mock(__name__='check_synergy')
        instance.initiate_qualification = Mock(__name__='initiate_qualification')
        instance.execute_trade = Mock(__name__='execute_trade')
        instance.record_outcome = Mock(__name__='record_outcome')
        instance.load_model = Mock(__name__='load_model')
        instance.load_models = Mock(__name__='load_models')
        instance._handle_indicators_ready = Mock(__name__='_handle_indicators_ready')
    
    return mocks


def test_kernel_initialization_in_backtest_mode(mock_config_file, mock_components, caplog):
    """Test that the kernel can be initialized successfully in backtest mode."""
    # Setup (Arrange)
    # Patch all external component imports
    with patch('src.core.kernel.BacktestDataHandler', mock_components['BacktestDataHandler']), \
         patch('src.core.kernel.LiveDataHandler', mock_components['LiveDataHandler']), \
         patch('src.core.kernel.BarGenerator', mock_components['BarGenerator']), \
         patch('src.core.kernel.IndicatorEngine', mock_components['IndicatorEngine']), \
         patch('src.core.kernel.MatrixAssembler30m', mock_components['MatrixAssembler30m']), \
         patch('src.core.kernel.MatrixAssembler5m', mock_components['MatrixAssembler5m']), \
         patch('src.core.kernel.MatrixAssemblerRegime', mock_components['MatrixAssemblerRegime']), \
         patch('src.core.kernel.RDEComponent', mock_components['RDEComponent']), \
         patch('src.core.kernel.MRMSComponent', mock_components['MRMSComponent']), \
         patch('src.core.kernel.MainMARLCoreComponent', mock_components['MainMARLCoreComponent']), \
         patch('src.core.kernel.SynergyDetector', mock_components['SynergyDetector']), \
         patch('src.core.kernel.BacktestExecutionHandler', mock_components['BacktestExecutionHandler']), \
         patch('src.core.kernel.LiveExecutionHandler', mock_components['LiveExecutionHandler']):
        
        # Create logs directory to prevent FileNotFoundError
        Path('logs').mkdir(exist_ok=True)
        
        # Execution (Act)
        kernel = AlgoSpaceKernel(config_path=mock_config_file)
        
        # Clear any log records from kernel creation
        caplog.clear()
        
        # Initialize the kernel
        with caplog.at_level(logging.INFO):
            kernel.initialize()
        
        # Verification (Assert)
        # 1. Verify initialization completed without errors
        assert kernel.config is not None
        assert kernel.components is not None
        assert len(kernel.components) > 0
        
        # 2. Verify BacktestDataHandler was created, not LiveDataHandler
        assert mock_components['BacktestDataHandler'].called
        assert not mock_components['LiveDataHandler'].called
        assert 'data_handler' in kernel.components
        
        # 3. Verify BacktestExecutionHandler was created, not LiveExecutionHandler
        assert mock_components['BacktestExecutionHandler'].called
        assert not mock_components['LiveExecutionHandler'].called
        assert 'execution_handler' in kernel.components
        
        # 4. Verify all expected components were instantiated
        expected_components = [
            'data_handler',
            'bar_generator',
            'indicator_engine',
            'matrix_30m',
            'matrix_5m',
            'matrix_regime',
            'synergy_detector',
            'rde',
            'm_rms',
            'main_marl_core',
            'execution_handler'
        ]
        
        for component_name in expected_components:
            assert component_name in kernel.components, f"Component {component_name} not found in kernel"
        
        # 5. Verify event subscriptions were set up correctly
        event_bus = kernel.event_bus
        
        # Check data flow events
        assert 'NEW_TICK' in event_bus.subscribers
        assert len(event_bus.subscribers['NEW_TICK']) > 0
        
        assert 'NEW_5MIN_BAR' in event_bus.subscribers
        assert 'NEW_30MIN_BAR' in event_bus.subscribers
        
        # Check indicator events
        assert 'INDICATORS_READY' in event_bus.subscribers
        # Should have multiple subscribers for INDICATORS_READY
        assert len(event_bus.subscribers['INDICATORS_READY']) >= 4  # 3 matrices + synergy detector
        
        # Check decision flow events
        assert 'SYNERGY_DETECTED' in event_bus.subscribers
        assert 'EXECUTE_TRADE' in event_bus.subscribers
        
        # Check feedback events
        assert 'TRADE_CLOSED' in event_bus.subscribers
        
        # Check system events
        assert 'SYSTEM_ERROR' in event_bus.subscribers
        assert 'SHUTDOWN_REQUEST' in event_bus.subscribers
        
        # 6. Verify models were attempted to be loaded
        rde_instance = kernel.components['rde']
        assert hasattr(rde_instance, 'load_model')
        
        mrms_instance = kernel.components['m_rms']
        assert hasattr(mrms_instance, 'load_model')
        
        marl_instance = kernel.components['main_marl_core']
        assert hasattr(marl_instance, 'load_models')
        
        # 7. Verify initialization log messages
        log_messages = [record.message for record in caplog.records]
        assert any("AlgoSpace System Initialization Starting" in msg for msg in log_messages)
        assert any("AlgoSpace Initialization Complete. System is READY." in msg for msg in log_messages)
        assert any("Instantiating components for data handler type: backtest" in msg for msg in log_messages)
        
        # 8. Verify kernel status
        status = kernel.get_status()
        assert status['mode'] == 'backtest'
        assert not status['running']  # Should not be running yet
        assert len(status['components']) == len(expected_components)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after tests."""
    yield
    # Clean up logs directory if empty
    logs_path = Path('logs')
    if logs_path.exists() and not any(logs_path.iterdir()):
        logs_path.rmdir()


def test_kernel_handles_missing_config_file():
    """Test that kernel raises appropriate error when config file is missing."""
    with pytest.raises(FileNotFoundError):
        kernel = AlgoSpaceKernel(config_path="non_existent_config.yaml")
        kernel.initialize()


def test_kernel_get_component():
    """Test the get_component method."""
    # Create mock components with proper methods
    mock_components = {}
    for component_name in ['BacktestDataHandler', 'BarGenerator', 'IndicatorEngine',
                           'MatrixAssembler30m', 'MatrixAssembler5m', 'MatrixAssemblerRegime',
                           'SynergyDetector', 'RDEComponent', 'MRMSComponent',
                           'MainMARLCoreComponent', 'BacktestExecutionHandler']:
        mock = Mock(return_value=Mock())
        instance = mock.return_value
        instance.on_new_tick = Mock(__name__='on_new_tick')
        instance.on_new_bar = Mock(__name__='on_new_bar')
        instance.on_indicators_ready = Mock(__name__='on_indicators_ready')
        instance._handle_indicators_ready = Mock(__name__='_handle_indicators_ready')
        instance.initiate_qualification = Mock(__name__='initiate_qualification')
        instance.execute_trade = Mock(__name__='execute_trade')
        instance.record_outcome = Mock(__name__='record_outcome')
        instance.load_model = Mock(__name__='load_model')
        instance.load_models = Mock(__name__='load_models')
        mock_components[component_name] = mock
    
    # Mock all components to prevent real instantiation
    with patch('src.core.kernel.BacktestDataHandler', mock_components['BacktestDataHandler']), \
         patch('src.core.kernel.BarGenerator', mock_components['BarGenerator']), \
         patch('src.core.kernel.IndicatorEngine', mock_components['IndicatorEngine']), \
         patch('src.core.kernel.MatrixAssembler30m', mock_components['MatrixAssembler30m']), \
         patch('src.core.kernel.MatrixAssembler5m', mock_components['MatrixAssembler5m']), \
         patch('src.core.kernel.MatrixAssemblerRegime', mock_components['MatrixAssemblerRegime']), \
         patch('src.core.kernel.SynergyDetector', mock_components['SynergyDetector']), \
         patch('src.core.kernel.RDEComponent', mock_components['RDEComponent']), \
         patch('src.core.kernel.MRMSComponent', mock_components['MRMSComponent']), \
         patch('src.core.kernel.MainMARLCoreComponent', mock_components['MainMARLCoreComponent']), \
         patch('src.core.kernel.BacktestExecutionHandler', mock_components['BacktestExecutionHandler']):
        
        # Create a minimal config
        config_data = {
            'data_handler': {'type': 'backtest', 'backtest_file': 'test.csv'},
            'execution': {'order_type': 'limit'},
            'risk_management': {'max_position_size': 100000},
            'agents': {},
            'models': {},
            'matrix_assemblers': {}  # Add empty matrix_assemblers config
        }
        
        with patch('src.core.config.load_config', return_value=config_data):
            kernel = AlgoSpaceKernel()
            
            # Create logs directory
            Path('logs').mkdir(exist_ok=True)
            
            # Test before initialization
            assert kernel.get_component('data_handler') is None
            
            # Initialize (with minimal components)
            kernel.initialize()
            
            # Test after initialization
            data_handler = kernel.get_component('data_handler')
            assert data_handler is not None
            
            # Test non-existent component
            assert kernel.get_component('non_existent') is None


def test_kernel_shutdown_before_run():
    """Test that shutdown can be called safely before run."""
    with patch('src.core.config.load_config', return_value={}):
        kernel = AlgoSpaceKernel()
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Should not raise any errors
        kernel.shutdown()
        
        # Kernel should not be marked as running
        assert not kernel.running