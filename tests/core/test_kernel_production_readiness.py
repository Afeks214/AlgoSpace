"""
Production readiness tests for the AlgoSpace kernel and event bus.
These tests verify production-grade reliability, performance, and error handling.
"""
import pytest
import time
import threading
import gc
import psutil
import os
import yaml
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from queue import Queue
import weakref
from datetime import datetime, timedelta

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.kernel import AlgoSpaceKernel
from src.core.event_bus import EventBus
from src.matrix.assembler_30m import MatrixAssembler30m
from src.matrix.assembler_5m import MatrixAssembler5m
from src.matrix.assembler_regime import MatrixAssemblerRegime


class TestKernelProductionReadiness:
    """Test suite for kernel production readiness verification."""
    
    @pytest.fixture
    def production_config(self, tmp_path):
        """Create a production-grade configuration file."""
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
            'matrix_assemblers': {
                '30m': {
                    'window_size': 48,
                    'features': ['mlmi_value', 'mlmi_signal', 'nwrqk_value', 'nwrqk_slope']
                },
                '5m': {
                    'window_size': 60,
                    'features': ['fvg_bullish_active', 'fvg_bearish_active']
                },
                'regime': {
                    'window_size': 96,
                    'features': ['mmd_features', 'volatility_30']
                }
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
            },
            'rde': {'input_dim': 155, 'd_model': 256, 'latent_dim': 8},
            'm_rms': {'synergy_dim': 30, 'account_dim': 10},
            'main_core': {'device': 'cpu'}
        }
        
        config_path = tmp_path / "production_settings.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        return str(config_path)
    
    @pytest.fixture
    def mock_production_components(self):
        """Create production-grade mock components."""
        mocks = {}
        component_names = [
            'BacktestDataHandler', 'LiveDataHandler', 'BarGenerator', 
            'IndicatorEngine', 'MatrixAssembler30m', 'MatrixAssembler5m', 
            'MatrixAssemblerRegime', 'RDEComponent', 'MRMSComponent', 
            'MainMARLCoreComponent', 'SynergyDetector', 
            'BacktestExecutionHandler', 'LiveExecutionHandler'
        ]
        
        for name in component_names:
            mock = Mock(return_value=Mock())
            instance = mock.return_value
            
            # Add all required methods with proper __name__ attributes
            for method_name in ['on_new_tick', 'on_new_bar', 'on_indicators_ready', 
                               '_handle_indicators_ready', 'initiate_qualification', 
                               'execute_trade', 'record_outcome', 'load_model', 
                               'load_models', 'start_stream', 'stop_stream', 
                               'close_all_positions', 'save_state']:
                method = Mock(__name__=method_name)
                setattr(instance, method_name, method)
            
            mocks[name] = mock
        
        return mocks
    
    def test_kernel_initialization_error_handling(self, production_config, mock_production_components):
        """Test kernel initialization with various error conditions."""
        Path('logs').mkdir(exist_ok=True)
        
        # Test 1: Invalid configuration file
        with pytest.raises(Exception):
            kernel = AlgoSpaceKernel(config_path="non_existent.yaml")
            kernel.initialize()
        
        # Test 2: Component instantiation failure
        with patch('src.core.kernel.BacktestDataHandler', side_effect=Exception("DataHandler failed")):
            kernel = AlgoSpaceKernel(config_path=production_config)
            with pytest.raises(Exception):
                kernel.initialize()
        
        # Test 3: Configuration loading failure
        with patch('src.core.config.load_config', side_effect=Exception("Config load failed")):
            kernel = AlgoSpaceKernel(config_path=production_config)
            with pytest.raises(Exception):
                kernel.initialize()
        
        # Test 4: Partial component failure should not prevent other components
        failing_mock = Mock(side_effect=Exception("Component failed"))
        
        with patch('src.core.kernel.BacktestDataHandler', mock_production_components['BacktestDataHandler']), \
             patch('src.core.kernel.BarGenerator', failing_mock), \
             patch('src.core.kernel.IndicatorEngine', mock_production_components['IndicatorEngine']), \
             patch('src.core.kernel.MatrixAssembler30m', mock_production_components['MatrixAssembler30m']), \
             patch('src.core.kernel.MatrixAssembler5m', mock_production_components['MatrixAssembler5m']), \
             patch('src.core.kernel.MatrixAssemblerRegime', mock_production_components['MatrixAssemblerRegime']), \
             patch('src.core.kernel.SynergyDetector', mock_production_components['SynergyDetector']), \
             patch('src.core.kernel.RDEComponent', mock_production_components['RDEComponent']), \
             patch('src.core.kernel.MRMSComponent', mock_production_components['MRMSComponent']), \
             patch('src.core.kernel.MainMARLCoreComponent', mock_production_components['MainMARLCoreComponent']), \
             patch('src.core.kernel.BacktestExecutionHandler', mock_production_components['BacktestExecutionHandler']):
            
            kernel = AlgoSpaceKernel(config_path=production_config)
            
            # Should handle component failure gracefully
            # Note: Current implementation will raise exception, but in production
            # we might want to continue with available components
            with pytest.raises(Exception):
                kernel.initialize()
    
    def test_kernel_memory_leak_detection(self, production_config, mock_production_components):
        """Test for memory leaks during long-running operations."""
        Path('logs').mkdir(exist_ok=True)
        
        # Patch all components
        with patch('src.core.kernel.BacktestDataHandler', mock_production_components['BacktestDataHandler']), \
             patch('src.core.kernel.BarGenerator', mock_production_components['BarGenerator']), \
             patch('src.core.kernel.IndicatorEngine', mock_production_components['IndicatorEngine']), \
             patch('src.core.kernel.MatrixAssembler30m', mock_production_components['MatrixAssembler30m']), \
             patch('src.core.kernel.MatrixAssembler5m', mock_production_components['MatrixAssembler5m']), \
             patch('src.core.kernel.MatrixAssemblerRegime', mock_production_components['MatrixAssemblerRegime']), \
             patch('src.core.kernel.SynergyDetector', mock_production_components['SynergyDetector']), \
             patch('src.core.kernel.RDEComponent', mock_production_components['RDEComponent']), \
             patch('src.core.kernel.MRMSComponent', mock_production_components['MRMSComponent']), \
             patch('src.core.kernel.MainMARLCoreComponent', mock_production_components['MainMARLCoreComponent']), \
             patch('src.core.kernel.BacktestExecutionHandler', mock_production_components['BacktestExecutionHandler']):
            
            # Get baseline memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and destroy kernels multiple times
            kernels = []
            for i in range(100):
                kernel = AlgoSpaceKernel(config_path=production_config)
                kernel.initialize()
                kernels.append(kernel)
                
                # Simulate some operations
                status = kernel.get_status()
                component = kernel.get_component('data_handler')
                
                if i % 10 == 0:
                    # Force garbage collection
                    gc.collect()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - initial_memory
                    
                    # Memory growth should be reasonable (less than 100MB for 100 iterations)
                    assert memory_growth < 100, f"Memory growth too high: {memory_growth:.2f} MB"
            
            # Cleanup
            for kernel in kernels:
                kernel.shutdown()
            
            # Final memory check
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024
            total_growth = final_memory - initial_memory
            
            # Final memory growth should be minimal
            assert total_growth < 50, f"Total memory growth too high: {total_growth:.2f} MB"
    
    def test_kernel_component_registration_completeness(self, production_config, mock_production_components):
        """Test that all components are properly registered and accessible."""
        Path('logs').mkdir(exist_ok=True)
        
        with patch('src.core.kernel.BacktestDataHandler', mock_production_components['BacktestDataHandler']), \
             patch('src.core.kernel.BarGenerator', mock_production_components['BarGenerator']), \
             patch('src.core.kernel.IndicatorEngine', mock_production_components['IndicatorEngine']), \
             patch('src.core.kernel.MatrixAssembler30m', mock_production_components['MatrixAssembler30m']), \
             patch('src.core.kernel.MatrixAssembler5m', mock_production_components['MatrixAssembler5m']), \
             patch('src.core.kernel.MatrixAssemblerRegime', mock_production_components['MatrixAssemblerRegime']), \
             patch('src.core.kernel.SynergyDetector', mock_production_components['SynergyDetector']), \
             patch('src.core.kernel.RDEComponent', mock_production_components['RDEComponent']), \
             patch('src.core.kernel.MRMSComponent', mock_production_components['MRMSComponent']), \
             patch('src.core.kernel.MainMARLCoreComponent', mock_production_components['MainMARLCoreComponent']), \
             patch('src.core.kernel.BacktestExecutionHandler', mock_production_components['BacktestExecutionHandler']):
            
            kernel = AlgoSpaceKernel(config_path=production_config)
            kernel.initialize()
            
            # Verify all expected components are registered
            expected_components = [
                'data_handler', 'bar_generator', 'indicator_engine',
                'matrix_30m', 'matrix_5m', 'matrix_regime',
                'synergy_detector', 'rde', 'm_rms', 'main_marl_core',
                'execution_handler'
            ]
            
            for component_name in expected_components:
                component = kernel.get_component(component_name)
                assert component is not None, f"Component {component_name} not registered"
                
                # Verify component has expected interface
                if hasattr(component, 'on_new_tick'):
                    assert callable(component.on_new_tick)
                if hasattr(component, 'on_new_bar'):
                    assert callable(component.on_new_bar)
    
    def test_kernel_graceful_shutdown(self, production_config, mock_production_components):
        """Test graceful shutdown procedures."""
        Path('logs').mkdir(exist_ok=True)
        
        with patch('src.core.kernel.BacktestDataHandler', mock_production_components['BacktestDataHandler']), \
             patch('src.core.kernel.BarGenerator', mock_production_components['BarGenerator']), \
             patch('src.core.kernel.IndicatorEngine', mock_production_components['IndicatorEngine']), \
             patch('src.core.kernel.MatrixAssembler30m', mock_production_components['MatrixAssembler30m']), \
             patch('src.core.kernel.MatrixAssembler5m', mock_production_components['MatrixAssembler5m']), \
             patch('src.core.kernel.MatrixAssemblerRegime', mock_production_components['MatrixAssemblerRegime']), \
             patch('src.core.kernel.SynergyDetector', mock_production_components['SynergyDetector']), \
             patch('src.core.kernel.RDEComponent', mock_production_components['RDEComponent']), \
             patch('src.core.kernel.MRMSComponent', mock_production_components['MRMSComponent']), \
             patch('src.core.kernel.MainMARLCoreComponent', mock_production_components['MainMARLCoreComponent']), \
             patch('src.core.kernel.BacktestExecutionHandler', mock_production_components['BacktestExecutionHandler']):
            
            kernel = AlgoSpaceKernel(config_path=production_config)
            kernel.initialize()
            
            # Simulate running state
            kernel.running = True
            
            # Test graceful shutdown
            kernel.shutdown()
            
            # Verify shutdown procedures were called
            data_handler = kernel.get_component('data_handler')
            execution_handler = kernel.get_component('execution_handler')
            
            # Verify data stream was stopped
            data_handler.stop_stream.assert_called_once()
            
            # Verify positions were closed
            execution_handler.close_all_positions.assert_called_once()
            
            # Verify component states were saved
            for component_name in kernel.components:
                component = kernel.get_component(component_name)
                if hasattr(component, 'save_state'):
                    component.save_state.assert_called_once()
            
            # Verify kernel is no longer running
            assert not kernel.running
    
    def test_kernel_system_error_handling(self, production_config, mock_production_components):
        """Test system error handling and recovery."""
        Path('logs').mkdir(exist_ok=True)
        
        with patch('src.core.kernel.BacktestDataHandler', mock_production_components['BacktestDataHandler']), \
             patch('src.core.kernel.BarGenerator', mock_production_components['BarGenerator']), \
             patch('src.core.kernel.IndicatorEngine', mock_production_components['IndicatorEngine']), \
             patch('src.core.kernel.MatrixAssembler30m', mock_production_components['MatrixAssembler30m']), \
             patch('src.core.kernel.MatrixAssembler5m', mock_production_components['MatrixAssembler5m']), \
             patch('src.core.kernel.MatrixAssemblerRegime', mock_production_components['MatrixAssemblerRegime']), \
             patch('src.core.kernel.SynergyDetector', mock_production_components['SynergyDetector']), \
             patch('src.core.kernel.RDEComponent', mock_production_components['RDEComponent']), \
             patch('src.core.kernel.MRMSComponent', mock_production_components['MRMSComponent']), \
             patch('src.core.kernel.MainMARLCoreComponent', mock_production_components['MainMARLCoreComponent']), \
             patch('src.core.kernel.BacktestExecutionHandler', mock_production_components['BacktestExecutionHandler']):
            
            kernel = AlgoSpaceKernel(config_path=production_config)
            kernel.initialize()
            
            # Test non-critical error handling
            non_critical_error = {'error': 'test_error', 'critical': False}
            kernel._handle_system_error(non_critical_error)
            
            # Kernel should still be in initialized state
            assert kernel.components is not None
            
            # Test critical error handling
            kernel.running = True
            critical_error = {'error': 'critical_test_error', 'critical': True}
            kernel._handle_system_error(critical_error)
            
            # Kernel should be shut down
            assert not kernel.running


class TestEventBusProductionReadiness:
    """Test suite for event bus production readiness verification."""
    
    def test_event_bus_thread_safety(self):
        """Test event bus thread safety under concurrent operations."""
        event_bus = EventBus()
        
        # Shared state for testing
        received_events = []
        lock = threading.Lock()
        
        def handler(payload):
            with lock:
                received_events.append(payload)
        
        # Subscribe multiple handlers
        for i in range(10):
            event_bus.subscribe(f'TEST_EVENT_{i}', handler)
        
        # Publish events from multiple threads
        def publish_events(thread_id):
            for i in range(100):
                event_bus.publish(f'TEST_EVENT_{i % 10}', f'payload_{thread_id}_{i}')
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=publish_events, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no race conditions occurred
        assert len(received_events) == 0  # Events are queued, not immediately processed
        assert event_bus.event_queue.qsize() == 500  # 5 threads * 100 events (each event type queued once)
    
    def test_event_bus_high_throughput_stress(self):
        """Test event bus performance under high throughput (10,000 events/second)."""
        event_bus = EventBus()
        
        # Performance tracking
        events_processed = []
        processing_times = []
        
        def fast_handler(payload):
            start_time = time.time()
            # Simulate minimal processing
            result = payload['value'] * 2
            end_time = time.time()
            
            events_processed.append(result)
            processing_times.append(end_time - start_time)
        
        # Subscribe handler
        event_bus.subscribe('HIGH_THROUGHPUT_EVENT', fast_handler)
        
        # Start event dispatcher in separate thread
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Publish 10,000 events as fast as possible
        start_time = time.time()
        
        for i in range(10000):
            event_bus.publish('HIGH_THROUGHPUT_EVENT', {'value': i)}
        
        # Wait for all events to be processed
        while event_bus.event_queue.qsize() > 0:
            time.sleep(0.01)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stop the dispatcher
        event_bus.stop()
        
        # Verify performance metrics
        assert len(events_processed) == 10000
        assert total_time < 2.0  # Should process 10k events in under 2 seconds
        
        # Verify event processing latency
        avg_processing_time = sum(processing_times) / len(processing_times)
        assert avg_processing_time < 0.001  # Average processing time should be under 1ms
        
        # Verify no events were lost
        assert len(set(events_processed)) == 10000  # All events should be unique
    
    def test_event_bus_ordering_under_load(self):
        """Test that event ordering is maintained under high load."""
        event_bus = EventBus()
        
        # Track event order
        received_order = []
        
        def order_tracking_handler(payload):
            received_order.append(payload)
        
        event_bus.subscribe('ORDER_TEST', order_tracking_handler)
        
        # Start dispatcher
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Publish events in order
        expected_order = list(range(1000))
        for i in expected_order:
            event_bus.publish('ORDER_TEST', i)
        
        # Wait for processing
        while event_bus.event_queue.qsize() > 0:
            time.sleep(0.01)
        
        # Stop dispatcher
        event_bus.stop()
        
        # Verify order was maintained
        assert received_order == expected_order
    
    def test_event_bus_subscriber_management_during_runtime(self):
        """Test adding/removing subscribers during runtime."""
        event_bus = EventBus()
        
        # Start dispatcher
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Track handler calls
        handler1_calls = []
        handler2_calls = []
        
        def handler1(payload):
            handler1_calls.append(payload)
        
        def handler2(payload):
            handler2_calls.append(payload)
        
        # Subscribe first handler
        event_bus.subscribe('RUNTIME_TEST', handler1)
        
        # Publish some events
        for i in range(10):
            event_bus.publish('RUNTIME_TEST', f'batch1_{i}')
        
        # Wait for processing
        time.sleep(0.1)
        
        # Add second handler during runtime
        event_bus.subscribe('RUNTIME_TEST', handler2)
        
        # Publish more events
        for i in range(10):
            event_bus.publish('RUNTIME_TEST', f'batch2_{i}')
        
        # Wait for processing
        time.sleep(0.1)
        
        # Remove first handler
        event_bus.unsubscribe('RUNTIME_TEST', handler1)
        
        # Publish final batch
        for i in range(10):
            event_bus.publish('RUNTIME_TEST', f'batch3_{i}')
        
        # Wait for processing
        time.sleep(0.1)
        
        # Stop dispatcher
        event_bus.stop()
        
        # Verify handler behavior
        assert len(handler1_calls) == 20  # First 2 batches
        assert len(handler2_calls) == 20  # Last 2 batches
        
        # Verify correct events were received
        assert all('batch1_' in call or 'batch2_' in call for call in handler1_calls)
        assert all('batch2_' in call or 'batch3_' in call for call in handler2_calls)
    
    def test_event_bus_error_handling_in_handlers(self):
        """Test event bus continues processing when handlers fail."""
        event_bus = EventBus()
        
        # Track successful processing
        successful_events = []
        
        def failing_handler(payload):
            if payload.get('fail', False):
                raise Exception(f"Handler failed for {payload}")
            successful_events.append(payload)
        
        def stable_handler(payload):
            # Always succeeds
            successful_events.append(f"stable_{payload}")
        
        # Subscribe both handlers
        event_bus.subscribe('ERROR_TEST', failing_handler)
        event_bus.subscribe('ERROR_TEST', stable_handler)
        
        # Start dispatcher
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Publish mix of successful and failing events
        test_events = [
            {'id': 1, 'fail': False},
            {'id': 2, 'fail': True},   # This should fail
            {'id': 3, 'fail': False},
            {'id': 4, 'fail': True},   # This should fail
            {'id': 5, 'fail': False}
        ]
        
        for event in test_events:
            event_bus.publish('ERROR_TEST', event)
        
        # Wait for processing
        time.sleep(0.2)
        
        # Stop dispatcher
        event_bus.stop()
        
        # Verify that stable handler processed all events
        stable_events = [e for e in successful_events if str(e).startswith('stable_')]
        assert len(stable_events) == 5  # All events should be processed by stable handler
        
        # Verify that failing handler only processed non-failing events
        failing_handler_events = [e for e in successful_events if not str(e).startswith('stable_')]
        assert len(failing_handler_events) == 3  # Only non-failing events
    
    def test_event_bus_memory_usage_under_load(self):
        """Test event bus memory usage patterns under sustained load."""
        event_bus = EventBus()
        
        # Track memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simple handler
        def memory_test_handler(payload):
            # Minimal processing to avoid memory accumulation in handler
            pass
        
        event_bus.subscribe('MEMORY_TEST', memory_test_handler)
        
        # Start dispatcher
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Publish events in batches and monitor memory
        for batch in range(100):
            # Publish 100 events
            for i in range(100):
                event_bus.publish('MEMORY_TEST', f'batch_{batch}_event_{i}')
            
            # Wait for processing
            while event_bus.event_queue.qsize() > 0:
                time.sleep(0.01)
            
            # Check memory every 10 batches
            if batch % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be minimal (less than 50MB for sustained load)
                assert memory_growth < 50, f"Memory growth too high: {memory_growth:.2f} MB at batch {batch}"
        
        # Stop dispatcher
        event_bus.stop()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        # Total memory growth should be reasonable
        assert total_growth < 100, f"Total memory growth too high: {total_growth:.2f} MB"
    
    def test_event_bus_graceful_shutdown(self):
        """Test event bus graceful shutdown procedures."""
        event_bus = EventBus()
        
        # Track shutdown behavior
        events_processed = []
        
        def shutdown_test_handler(payload):
            events_processed.append(payload)
            time.sleep(0.01)  # Simulate some processing time
        
        event_bus.subscribe('SHUTDOWN_TEST', shutdown_test_handler)
        
        # Start dispatcher
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Publish some events
        for i in range(100):
            event_bus.publish('SHUTDOWN_TEST', i)
        
        # Let some events process
        time.sleep(0.1)
        
        # Initiate shutdown
        event_bus.stop()
        
        # Wait for dispatcher to finish
        dispatcher_thread.join(timeout=5.0)
        
        # Verify shutdown was clean
        assert not event_bus._running
        assert not dispatcher_thread.is_alive()
        
        # Some events should have been processed
        assert len(events_processed) > 0
        
        # Verify no events can be published after shutdown
        initial_processed_count = len(events_processed)
        event_bus.publish('SHUTDOWN_TEST', 'after_shutdown')
        
        # Give time for any erroneous processing
        time.sleep(0.1)
        
        # Should not have processed the post-shutdown event
        assert len(events_processed) == initial_processed_count


class TestComponentIntegration:
    """Test component integration and event flow."""
    
    def test_complete_event_flow(self):
        """Test the complete event flow: NEW_TICK → NEW_5MIN_BAR → INDICATORS_READY → SYNERGY_DETECTED."""
        event_bus = EventBus()
        
        # Track event flow
        event_flow = []
        
        def tick_handler(payload):
            event_flow.append(('NEW_TICK', payload))
            # Simulate bar generation
            event_bus.publish('NEW_5MIN_BAR', {'bar_data': 'test_bar')}
        
        def bar_handler(payload):
            event_flow.append(('NEW_5MIN_BAR', payload))
            # Simulate indicator calculation
            event_bus.publish('INDICATORS_READY', {'indicators': 'test_indicators'})
        
        def indicators_handler(payload):
            event_flow.append(('INDICATORS_READY', payload))
            # Simulate synergy detection
            event_bus.publish('SYNERGY_DETECTED', {'synergy': 'test_synergy'})
        
        def synergy_handler(payload):
            event_flow.append(('SYNERGY_DETECTED', payload))
        
        # Subscribe handlers
        event_bus.subscribe('NEW_TICK', tick_handler)
        event_bus.subscribe('NEW_5MIN_BAR', bar_handler)
        event_bus.subscribe('INDICATORS_READY', indicators_handler)
        event_bus.subscribe('SYNERGY_DETECTED', synergy_handler)
        
        # Start dispatcher
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Trigger the flow
        event_bus.publish('NEW_TICK', {'tick_data': 'test_tick'})
        
        # Wait for processing
        time.sleep(0.1)
        
        # Stop dispatcher
        event_bus.stop()
        
        # Verify complete flow
        assert len(event_flow) == 4
        assert event_flow[0][0] == 'NEW_TICK'
        assert event_flow[1][0] == 'NEW_5MIN_BAR'
        assert event_flow[2][0] == 'INDICATORS_READY'
        assert event_flow[3][0] == 'SYNERGY_DETECTED'
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies in event flow."""
        event_bus = EventBus()
        
        # Track recursive calls
        call_count = {'A': 0, 'B': 0, 'C': 0}
        
        def handler_a(payload):
            call_count['A'] += 1
            if call_count['A'] <= 5:  # Limit recursion
                event_bus.publish('EVENT_B', payload)
        
        def handler_b(payload):
            call_count['B'] += 1
            if call_count['B'] <= 5:  # Limit recursion
                event_bus.publish('EVENT_C', payload)
        
        def handler_c(payload):
            call_count['C'] += 1
            if call_count['C'] <= 5:  # Limit recursion
                event_bus.publish('EVENT_A', payload)  # Creates circular dependency
        
        # Subscribe handlers
        event_bus.subscribe('EVENT_A', handler_a)
        event_bus.subscribe('EVENT_B', handler_b)
        event_bus.subscribe('EVENT_C', handler_c)
        
        # Start dispatcher
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Trigger the circular flow
        event_bus.publish('EVENT_A', {'data': 'test'})
        
        # Wait for processing
        time.sleep(0.2)
        
        # Stop dispatcher
        event_bus.stop()
        
        # Verify circular dependency was handled (calls should be limited)
        # Note: Due to event queuing, there might be one extra call
        assert call_count['A'] >= 5 and call_count['A'] <= 6
        assert call_count['B'] >= 5 and call_count['B'] <= 6
        assert call_count['C'] >= 5 and call_count['C'] <= 6
    
    def test_event_latency_measurement(self):
        """Test event latency measurement (should be <1ms)."""
        event_bus = EventBus()
        
        # Track latencies
        latencies = []
        
        def latency_handler(payload):
            receive_time = time.time()
            send_time = payload['send_time']
            latency = (receive_time - send_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        
        event_bus.subscribe('LATENCY_TEST', latency_handler)
        
        # Start dispatcher
        dispatcher_thread = threading.Thread(target=event_bus.dispatch_forever)
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Publish events with timestamps
        for i in range(1000):
            send_time = time.time()
            event_bus.publish('LATENCY_TEST', {'send_time': send_time, 'event_id': i})
            time.sleep(0.001)  # Small delay between sends
        
        # Wait for processing
        time.sleep(0.5)
        
        # Stop dispatcher
        event_bus.stop()
        
        # Analyze latencies
        assert len(latencies) == 1000
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # Verify latency requirements
        assert avg_latency < 1.0, f"Average latency too high: {avg_latency:.3f}ms"
        assert max_latency < 10.0, f"Maximum latency too high: {max_latency:.3f}ms"
        assert min_latency >= 0, f"Negative latency detected: {min_latency:.3f}ms"
        
        # Verify 95th percentile
        sorted_latencies = sorted(latencies)
        p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))]
        assert p95_latency < 2.0, f"95th percentile latency too high: {p95_latency:.3f}ms"


class TestMatrixAssemblerProductionReadiness:
    """Test suite for Matrix Assembler production readiness verification."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Create a mock kernel for testing."""
        kernel = Mock()
        kernel.get_event_bus.return_value = Mock()
        return kernel
    
    @pytest.fixture
    def assembler_30m_config(self, mock_kernel):
        """Create configuration for 30m assembler."""
        return {
            'name': 'MatrixAssembler30m',
            'kernel': mock_kernel,
            'window_size': 48,
            'features': [
                'mlmi_value', 'mlmi_signal', 'nwrqk_value', 'nwrqk_slope',
                'lvn_distance_points', 'lvn_nearest_strength', 'time_hour_sin', 'time_hour_cos'
            ],
            'warmup_period': 20,
            'feature_configs': {
                'mlmi_value': {'ema_alpha': 0.01, 'warmup_samples': 50},
                'nwrqk_slope': {'ema_alpha': 0.02, 'warmup_samples': 100}
            }
        }
    
    @pytest.fixture
    def assembler_5m_config(self, mock_kernel):
        """Create configuration for 5m assembler."""
        return {
            'name': 'MatrixAssembler5m',
            'kernel': mock_kernel,
            'window_size': 60,
            'features': [
                'fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
                'fvg_age', 'fvg_mitigation_signal', 'fvg_gap_size_pct',
                'fvg_mitigation_strength', 'price_momentum_5', 'volume_ratio'
            ],
            'warmup_period': 25
        }
    
    @pytest.fixture
    def assembler_regime_config(self, mock_kernel):
        """Create configuration for regime assembler."""
        return {
            'name': 'MatrixAssemblerRegime',
            'kernel': mock_kernel,
            'window_size': 96,
            'features': [
                'mmd_features', 'volatility_30', 'volume_profile_skew', 'price_acceleration'
            ],
            'warmup_period': 30
        }
    
    def test_matrix_assembler_30m_construction_and_features(self, assembler_30m_config):
        """Test Matrix Assembler 30m correct 48×8 matrix construction and feature extraction."""
        assembler = MatrixAssembler30m(assembler_30m_config)
        
        # Verify matrix dimensions
        assert assembler.matrix.shape == (48, 8), f"Expected 48×8 matrix, got {assembler.matrix.shape}"
        assert assembler.window_size == 48
        assert assembler.n_features == 8
        
        # Test feature extraction with comprehensive feature store
        feature_store = {
            'mlmi_value': 75.5,
            'mlmi_signal': 1.0,
            'nwrqk_value': 4125.50,
            'nwrqk_slope': 0.05,
            'lvn_distance_points': 25.0,
            'lvn_nearest_strength': 85.0,
            'timestamp': datetime.now(),
            'current_price': 4125.50,
            'close': 4125.50
        }
        
        # Test feature extraction
        features = assembler.extract_features(feature_store)
        assert features is not None
        assert len(features) == 8
        
        # Test feature validation
        assert assembler.validate_features(features)
        
        # Test specific feature values
        assert 0 <= features[0] <= 100  # mlmi_value
        assert features[1] in [-1, 0, 1]  # mlmi_signal
        assert features[4] >= 0  # lvn_distance_points
        assert 0 <= features[5] <= 100  # lvn_nearest_strength
        
        # Test preprocessing
        processed = assembler.preprocess_features(features, feature_store)
        assert processed.dtype == np.float32
        assert len(processed) == 8
        assert np.all(np.isfinite(processed))
        assert np.all(processed >= -3.0) and np.all(processed <= 3.0)
        
    def test_matrix_assembler_30m_normalization_and_threading(self, assembler_30m_config):
        """Test Matrix Assembler 30m normalization consistency and thread safety."""
        assembler = MatrixAssembler30m(assembler_30m_config)
        
        # Test normalization consistency
        feature_store = {
            'mlmi_value': 50.0,
            'mlmi_signal': 0.0,
            'nwrqk_value': 4100.0,
            'nwrqk_slope': 0.0,
            'lvn_distance_points': 10.0,
            'lvn_nearest_strength': 50.0,
            'timestamp': datetime.now(),
            'current_price': 4100.0
        }
        
        # Process same features multiple times
        results = []
        for _ in range(10):
            features = assembler.extract_features(feature_store)
            processed = assembler.preprocess_features(features, feature_store)
            results.append(processed.copy())
        
        # Check normalization consistency
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], atol=1e-6), "Normalization inconsistent"
        
        # Test thread safety with concurrent updates
        def update_worker(worker_id):
            local_results = []
            for i in range(100):
                test_store = feature_store.copy()
                test_store['mlmi_value'] = 50.0 + (worker_id * 10) + i
                test_store['current_price'] = 4100.0 + i
                
                features = assembler.extract_features(test_store)
                if features:
                    processed = assembler.preprocess_features(features, test_store)
                    local_results.append(processed)
            return local_results
        
        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(update_worker, i) for i in range(4)]
            thread_results = [future.result() for future in as_completed(futures)]
        
        # Verify all threads completed successfully
        assert all(len(result) == 100 for result in thread_results)
        
        # Verify no race conditions in normalization
        for thread_result in thread_results:
            for processed in thread_result:
                assert np.all(np.isfinite(processed))
                assert np.all(processed >= -3.0) and np.all(processed <= 3.0)
    
    def test_matrix_assembler_5m_construction_and_fvg_features(self, assembler_5m_config):
        """Test Matrix Assembler 5m correct 60×9 matrix construction and FVG features."""
        assembler = MatrixAssembler5m(assembler_5m_config)
        
        # Verify matrix dimensions
        assert assembler.matrix.shape == (60, 9), f"Expected 60×9 matrix, got {assembler.matrix.shape}"
        assert assembler.window_size == 60
        assert assembler.n_features == 9
        
        # Test comprehensive FVG feature extraction
        feature_store = {
            'fvg_bullish_active': 1.0,
            'fvg_bearish_active': 0.0,
            'fvg_nearest_level': 4127.25,
            'fvg_age': 5.0,
            'fvg_mitigation_signal': 0.0,
            'fvg_gap_size_pct': 0.15,
            'fvg_mitigation_strength': 0.75,
            'current_price': 4125.50,
            'current_volume': 1500,
            'volume': 1500,
            'close': 4125.50
        }
        
        # Test feature extraction
        features = assembler.extract_features(feature_store)
        assert features is not None
        assert len(features) == 9
        
        # Test FVG-specific features
        assert features[0] in [0.0, 1.0]  # fvg_bullish_active
        assert features[1] in [0.0, 1.0]  # fvg_bearish_active
        assert features[3] >= 0  # fvg_age
        assert features[4] in [0.0, 1.0]  # fvg_mitigation_signal
        
        # Test new FVG features
        if len(features) > 5:
            assert features[5] >= 0  # fvg_gap_size_pct
            assert 0 <= features[6] <= 1.0  # fvg_mitigation_strength
        
        # Test preprocessing
        processed = assembler.preprocess_features(features, feature_store)
        assert processed.dtype == np.float32
        assert len(processed) == 9
        assert np.all(np.isfinite(processed))
        assert np.all(processed >= -2.0) and np.all(processed <= 2.0)
        
    def test_matrix_assembler_5m_ema_smoothing_and_performance(self, assembler_5m_config):
        """Test Matrix Assembler 5m EMA smoothing and performance requirements."""
        assembler = MatrixAssembler5m(assembler_5m_config)
        
        # Test EMA smoothing for volume
        base_volume = 1000
        feature_store = {
            'fvg_bullish_active': 0.0,
            'fvg_bearish_active': 0.0,
            'fvg_nearest_level': 4125.0,
            'fvg_age': 0.0,
            'fvg_mitigation_signal': 0.0,
            'fvg_gap_size_pct': 0.0,
            'fvg_mitigation_strength': 0.0,
            'current_price': 4125.0,
            'current_volume': base_volume,
            'volume': base_volume
        }
        
        # Build volume history for EMA
        volumes = []
        for i in range(50):
            feature_store['current_volume'] = base_volume + (i * 100)
            _ = assembler.extract_features(feature_store)  # Process to update EMA
            volumes.append(assembler.volume_ema)
        
        # Verify EMA smoothing behavior
        assert len(volumes) == 50
        assert volumes[0] == base_volume  # First value should be initial
        assert volumes[-1] > volumes[0]  # Should trend upward
        
        # Test performance requirement (<100μs)
        performance_times = []
        
        for _ in range(1000):
            start_time = time.time()
            matrix = assembler.get_matrix()
            end_time = time.time()
            
            if matrix is not None:
                performance_times.append((end_time - start_time) * 1000000)  # Convert to microseconds
        
        if performance_times:
            avg_time = sum(performance_times) / len(performance_times)
            max_time = max(performance_times)
            
            # Performance requirements
            assert avg_time < 100, f"Average get_matrix() time {avg_time:.2f}μs exceeds 100μs"
            assert max_time < 500, f"Maximum get_matrix() time {max_time:.2f}μs exceeds 500μs"
    
    def test_matrix_assembler_regime_mmd_features(self, assembler_regime_config):
        """Test Regime Matrix Assembler 96×N matrix with MMD features."""
        assembler = MatrixAssemblerRegime(assembler_regime_config)
        
        # Verify matrix dimensions
        assert assembler.matrix.shape == (96, 4), f"Expected 96×4 matrix, got {assembler.matrix.shape}"
        assert assembler.window_size == 96
        assert assembler.n_features == 4
        
        # Test MMD features handling
        mmd_features = np.random.randn(32).astype(np.float32)  # 32-dimensional MMD features
        feature_store = {
            'mmd_features': mmd_features,
            'volatility_30': 1.5,
            'volume_profile_skew': 0.25,
            'price_acceleration': 0.1,
            'current_price': 4125.0,
            'current_volume': 1200,
            'close': 4125.0,
            'volume': 1200
        }
        
        # Test feature extraction with MMD array
        features = assembler.extract_features(feature_store)
        assert features is not None
        # Should have 32 MMD features + 3 other features = 35 total
        expected_features = len(mmd_features) + 3
        assert len(features) == expected_features
        
        # Test MMD feature accuracy
        for i in range(len(mmd_features)):
            assert abs(features[i] - mmd_features[i]) < 1e-6
        
        # Test preprocessing
        processed = assembler.preprocess_features(features, feature_store)
        assert processed.dtype == np.float32
        assert len(processed) == expected_features
        assert np.all(np.isfinite(processed))
        
        # Test custom feature calculations
        volatility = assembler._calculate_volatility()
        skew = assembler._calculate_volume_skew()
        acceleration = assembler._calculate_price_acceleration()
        
        assert volatility >= 0
        assert -3.0 <= skew <= 3.0
        assert -5.0 <= acceleration <= 5.0
    
    def test_matrix_assembler_regime_memory_efficiency(self, assembler_regime_config):
        """Test Regime Matrix Assembler memory efficiency with large windows."""
        assembler = MatrixAssemblerRegime(assembler_regime_config)
        
        # Track memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate large MMD feature arrays
        for i in range(1000):
            # Large MMD features (simulate real regime detection features)
            mmd_features = np.random.randn(64).astype(np.float32)
            feature_store = {
                'mmd_features': mmd_features,
                'volatility_30': 1.0 + (i * 0.01),
                'volume_profile_skew': np.random.uniform(-1, 1),
                'price_acceleration': np.random.uniform(-0.5, 0.5),
                'current_price': 4125.0 + (i * 0.25),
                'current_volume': 1000 + (i * 2),
                'close': 4125.0 + (i * 0.25),
                'volume': 1000 + (i * 2)
            }
            
            features = assembler.extract_features(feature_store)
            if features:
                _ = assembler.preprocess_features(features, feature_store)  # Process for memory tracking
                # Simulate matrix update
                assembler.n_updates += 1
                assembler.current_index = (assembler.current_index + 1) % assembler.window_size
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable for 1000 updates
        assert memory_growth < 50, f"Memory growth too high: {memory_growth:.2f} MB"
        
        # Test matrix retrieval efficiency
        start_time = time.time()
        matrix = assembler.get_matrix()
        end_time = time.time()
        
        if matrix is not None:
            retrieval_time = (end_time - start_time) * 1000  # milliseconds
            assert retrieval_time < 10, f"Matrix retrieval too slow: {retrieval_time:.2f}ms"
    
    def test_matrix_assembler_integration_flow(self, assembler_30m_config, assembler_5m_config, assembler_regime_config):
        """Test complete flow: Feature Store → Assemblers → Matrices."""
        # Create all assemblers
        assembler_30m = MatrixAssembler30m(assembler_30m_config)
        assembler_5m = MatrixAssembler5m(assembler_5m_config)
        assembler_regime = MatrixAssemblerRegime(assembler_regime_config)
        
        # Comprehensive feature store
        feature_store = {
            # 30m features
            'mlmi_value': 65.0,
            'mlmi_signal': 1.0,
            'nwrqk_value': 4125.0,
            'nwrqk_slope': 0.025,
            'lvn_distance_points': 15.0,
            'lvn_nearest_strength': 75.0,
            'timestamp': datetime.now(),
            
            # 5m features
            'fvg_bullish_active': 1.0,
            'fvg_bearish_active': 0.0,
            'fvg_nearest_level': 4127.0,
            'fvg_age': 3.0,
            'fvg_mitigation_signal': 0.0,
            'fvg_gap_size_pct': 0.12,
            'fvg_mitigation_strength': 0.8,
            
            # Regime features
            'mmd_features': np.random.randn(32).astype(np.float32),
            'volatility_30': 1.2,
            'volume_profile_skew': 0.15,
            'price_acceleration': 0.05,
            
            # Common features
            'current_price': 4125.0,
            'current_volume': 1350,
            'close': 4125.0,
            'volume': 1350
        }
        
        # Test all assemblers can process the same feature store
        features_30m = assembler_30m.extract_features(feature_store)
        features_5m = assembler_5m.extract_features(feature_store)
        features_regime = assembler_regime.extract_features(feature_store)
        
        assert features_30m is not None
        assert features_5m is not None
        assert features_regime is not None
        
        # Test preprocessing
        processed_30m = assembler_30m.preprocess_features(features_30m, feature_store)
        processed_5m = assembler_5m.preprocess_features(features_5m, feature_store)
        processed_regime = assembler_regime.preprocess_features(features_regime, feature_store)
        
        assert np.all(np.isfinite(processed_30m))
        assert np.all(np.isfinite(processed_5m))
        assert np.all(np.isfinite(processed_regime))
        
        # Test PyTorch tensor compatibility
        if TORCH_AVAILABLE:
            tensor_30m = torch.from_numpy(processed_30m)
            tensor_5m = torch.from_numpy(processed_5m)
            tensor_regime = torch.from_numpy(processed_regime)
            
            assert tensor_30m.dtype == torch.float32
            assert tensor_5m.dtype == torch.float32
            assert tensor_regime.dtype == torch.float32
            
            # Test dimensions for neural network input
            assert tensor_30m.shape == (8,)
            assert tensor_5m.shape == (9,)
            assert tensor_regime.shape == (35,)  # 32 MMD + 3 other features
        else:
            # At least verify the shapes are correct for tensor conversion
            assert processed_30m.shape == (8,)
            assert processed_5m.shape == (9,)
            assert processed_regime.shape == (35,)  # 32 MMD + 3 other features
    
    def test_matrix_assembler_edge_cases_missing_data(self, assembler_30m_config):
        """Test edge cases with missing/delayed indicator data."""
        assembler = MatrixAssembler30m(assembler_30m_config)
        
        # Test with missing features
        incomplete_store = {
            'mlmi_value': 50.0,
            'current_price': 4125.0,
            # Missing: mlmi_signal, nwrqk_value, nwrqk_slope, etc.
        }
        
        features = assembler.extract_features(incomplete_store)
        assert features is not None
        assert len(features) == 8  # Should use defaults for missing features
        
        # Test with invalid data types
        invalid_store = {
            'mlmi_value': 'invalid',
            'mlmi_signal': None,
            'nwrqk_value': float('inf'),
            'nwrqk_slope': float('nan'),
            'current_price': 4125.0
        }
        
        features = assembler.extract_features(invalid_store)
        assert features is not None
        assert len(features) == 8
        
        # Test preprocessing with invalid features
        processed = assembler.preprocess_features(features, invalid_store)
        assert np.all(np.isfinite(processed))
        
        # Test validation with edge cases
        edge_features = [
            float('inf'),  # Invalid
            2.0,           # Invalid mlmi_signal
            -1000.0,       # Valid but extreme
            float('nan'),  # Invalid
            -50.0,         # Valid
            150.0,         # Invalid range
            12.5,          # Valid hour
            12.5           # Valid hour
        ]
        
        # Should handle invalid features gracefully
        validation_result = assembler.validate_features(edge_features)
        assert isinstance(validation_result, bool)
    
    def test_matrix_assembler_end_to_end_latency(self, assembler_30m_config, assembler_5m_config):
        """Test end-to-end latency from feature store to matrix output."""
        assembler_30m = MatrixAssembler30m(assembler_30m_config)
        assembler_5m = MatrixAssembler5m(assembler_5m_config)
        
        # Warm up assemblers
        warmup_store = {
            'mlmi_value': 50.0, 'mlmi_signal': 0.0, 'nwrqk_value': 4125.0, 'nwrqk_slope': 0.0,
            'lvn_distance_points': 10.0, 'lvn_nearest_strength': 50.0, 'timestamp': datetime.now(),
            'fvg_bullish_active': 0.0, 'fvg_bearish_active': 0.0, 'fvg_nearest_level': 4125.0,
            'fvg_age': 0.0, 'fvg_mitigation_signal': 0.0, 'fvg_gap_size_pct': 0.0,
            'fvg_mitigation_strength': 0.0, 'current_price': 4125.0, 'current_volume': 1000
        }
        
        for _ in range(50):
            assembler_30m.extract_features(warmup_store)
            assembler_5m.extract_features(warmup_store)
        
        # Measure end-to-end latency
        latencies_30m = []
        latencies_5m = []
        
        for i in range(100):
            test_store = warmup_store.copy()
            test_store['mlmi_value'] = 50.0 + i
            test_store['current_price'] = 4125.0 + i
            
            # 30m assembler latency
            start_time = time.time()
            features_30m = assembler_30m.extract_features(test_store)
            if features_30m:
                _ = assembler_30m.preprocess_features(features_30m, test_store)
            end_time = time.time()
            latencies_30m.append((end_time - start_time) * 1000000)  # microseconds
            
            # 5m assembler latency
            start_time = time.time()
            features_5m = assembler_5m.extract_features(test_store)
            if features_5m:
                _ = assembler_5m.preprocess_features(features_5m, test_store)
            end_time = time.time()
            latencies_5m.append((end_time - start_time) * 1000000)  # microseconds
        
        # Verify latency requirements
        avg_latency_30m = sum(latencies_30m) / len(latencies_30m)
        avg_latency_5m = sum(latencies_5m) / len(latencies_5m)
        max_latency_30m = max(latencies_30m)
        max_latency_5m = max(latencies_5m)
        
        # Performance assertions
        assert avg_latency_30m < 200, f"30m average latency too high: {avg_latency_30m:.2f}μs"
        assert avg_latency_5m < 150, f"5m average latency too high: {avg_latency_5m:.2f}μs"
        assert max_latency_30m < 1000, f"30m max latency too high: {max_latency_30m:.2f}μs"
        assert max_latency_5m < 800, f"5m max latency too high: {max_latency_5m:.2f}μs"


if __name__ == '__main__':
    # Run with: python -m pytest tests/core/test_kernel_production_readiness.py -v
    pytest.main([__file__, '-v'])