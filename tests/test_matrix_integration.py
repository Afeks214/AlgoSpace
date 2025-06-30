"""
Integration tests for Matrix Assemblers with the complete data pipeline.

This test suite verifies that matrix assemblers correctly integrate with:
- DataHandler (tick generation)
- BarGenerator (bar creation)
- IndicatorEngine (feature calculation)
- Event system (proper flow)
"""

import asyncio
import unittest
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import time
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.kernel import AlgoSpaceKernel
from src.core.events import EventType, Event
from src.data.handlers import create_data_handler
from src.data.bar_generator import BarGenerator
from src.indicators.engine import IndicatorEngine
from src.matrix import MatrixAssembler30m, MatrixAssembler5m, MatrixAssemblerRegime
from src.utils.logger import setup_logging, get_logger


class MatrixIntegrationMonitor:
    """Monitor to track matrix assembler behavior during integration tests."""
    
    def __init__(self):
        self.logger = get_logger("MatrixMonitor")
        
        # Event tracking
        self.event_counts = {
            EventType.NEW_TICK: 0,
            EventType.NEW_5MIN_BAR: 0,
            EventType.NEW_30MIN_BAR: 0,
            EventType.INDICATORS_READY: 0
        }
        
        # Matrix update tracking
        self.matrix_updates = {
            '30m': [],
            '5m': [],
            'regime': []
        }
        
        # Performance tracking
        self.update_latencies = []
        self.matrix_snapshots = []
        
        # Error tracking
        self.errors = []
    
    def track_event(self, event: Event) -> None:
        """Track system events."""
        if event.event_type in self.event_counts:
            self.event_counts[event.event_type] += 1
    
    def track_matrix_update(self, assembler_name: str, matrix_stats: Dict[str, Any]) -> None:
        """Track matrix updates."""
        update_info = {
            'timestamp': datetime.now(),
            'stats': matrix_stats
        }
        self.matrix_updates[assembler_name].append(update_info)
    
    def track_performance(self, operation: str, duration_ms: float) -> None:
        """Track performance metrics."""
        self.update_latencies.append({
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': datetime.now()
        })
    
    def capture_matrix_snapshot(self, assembler_name: str, matrix: np.ndarray) -> None:
        """Capture matrix snapshot for analysis."""
        snapshot = {
            'assembler': assembler_name,
            'timestamp': datetime.now(),
            'shape': matrix.shape,
            'mean': float(np.mean(matrix)),
            'std': float(np.std(matrix)),
            'min': float(np.min(matrix)),
            'max': float(np.max(matrix)),
            'sample': matrix[-5:, :3] if matrix.shape[0] >= 5 else matrix  # Last 5 rows, first 3 cols
        }
        self.matrix_snapshots.append(snapshot)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary."""
        return {
            'event_counts': self.event_counts,
            'matrix_update_counts': {
                name: len(updates) for name, updates in self.matrix_updates.items()
            },
            'performance': {
                'avg_latency_ms': np.mean([l['duration_ms'] for l in self.update_latencies]) if self.update_latencies else 0,
                'max_latency_ms': np.max([l['duration_ms'] for l in self.update_latencies]) if self.update_latencies else 0
            },
            'errors': len(self.errors),
            'snapshots_captured': len(self.matrix_snapshots)
        }


class TestMatrixIntegration(unittest.TestCase):
    """Integration tests for matrix assemblers."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        setup_logging()
        cls.logger = get_logger("TestMatrixIntegration")
    
    def setUp(self):
        """Set up for each test."""
        self.kernel = AlgoSpaceKernel()
        self.monitor = MatrixIntegrationMonitor()
        
        # Components will be created in individual tests
        self.data_handler = None
        self.bar_generator = None
        self.indicator_engine = None
        self.matrix_30m = None
        self.matrix_5m = None
        self.matrix_regime = None
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'kernel'):
            # Synchronous cleanup
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.kernel.shutdown())
            loop.close()
    
    def setup_pipeline(self):
        """Set up complete data pipeline with matrix assemblers."""
        # Create data pipeline components
        self.data_handler = create_data_handler(self.kernel)
        self.bar_generator = BarGenerator("BarGenerator", self.kernel)
        self.indicator_engine = IndicatorEngine("IndicatorEngine", self.kernel)
        
        # Create matrix assemblers
        self.matrix_30m = MatrixAssembler30m("MatrixAssembler30m", self.kernel)
        self.matrix_5m = MatrixAssembler5m("MatrixAssembler5m", self.kernel)
        self.matrix_regime = MatrixAssemblerRegime("MatrixAssemblerRegime", self.kernel)
        
        # Register components with kernel
        self.kernel.register_component("DataHandler", self.data_handler)
        self.kernel.register_component("BarGenerator", self.bar_generator, ["DataHandler"])
        self.kernel.register_component("IndicatorEngine", self.indicator_engine, ["BarGenerator"])
        self.kernel.register_component("MatrixAssembler30m", self.matrix_30m, ["IndicatorEngine"])
        self.kernel.register_component("MatrixAssembler5m", self.matrix_5m, ["IndicatorEngine"])
        self.kernel.register_component("MatrixAssemblerRegime", self.matrix_regime, ["IndicatorEngine"])
        
        # Set up monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Set up event monitoring."""
        event_bus = self.kernel.get_event_bus()
        
        # Monitor all events
        for event_type in [EventType.NEW_TICK, EventType.NEW_5MIN_BAR, 
                          EventType.NEW_30MIN_BAR, EventType.INDICATORS_READY]:
            event_bus.subscribe(event_type, self.monitor.track_event)
        
        # Monitor matrix updates by wrapping update methods
        self._wrap_matrix_updates()
    
    def _wrap_matrix_updates(self):
        """Wrap matrix update methods for monitoring."""
        # Wrap 30m assembler
        original_30m = self.matrix_30m._update_matrix
        def monitored_30m(feature_store):
            start = time.perf_counter()
            result = original_30m(feature_store)
            duration = (time.perf_counter() - start) * 1000
            self.monitor.track_performance("matrix_30m_update", duration)
            self.monitor.track_matrix_update('30m', self.matrix_30m.get_statistics())
            return result
        self.matrix_30m._update_matrix = monitored_30m
        
        # Similar for 5m and regime
        original_5m = self.matrix_5m._update_matrix
        def monitored_5m(feature_store):
            start = time.perf_counter()
            result = original_5m(feature_store)
            duration = (time.perf_counter() - start) * 1000
            self.monitor.track_performance("matrix_5m_update", duration)
            self.monitor.track_matrix_update('5m', self.matrix_5m.get_statistics())
            return result
        self.matrix_5m._update_matrix = monitored_5m
        
        original_regime = self.matrix_regime._update_matrix
        def monitored_regime(feature_store):
            start = time.perf_counter()
            result = original_regime(feature_store)
            duration = (time.perf_counter() - start) * 1000
            self.monitor.track_performance("matrix_regime_update", duration)
            self.monitor.track_matrix_update('regime', self.matrix_regime.get_statistics())
            return result
        self.matrix_regime._update_matrix = monitored_regime
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline from ticks to matrices."""
        self.setup_pipeline()
        
        # Run the system
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            # Start system
            await self.kernel.start()
            
            # Let it run for a bit
            await asyncio.sleep(5)  # 5 seconds of processing
            
            # Capture matrix snapshots
            for assembler, name in [(self.matrix_30m, '30m'),
                                   (self.matrix_5m, '5m'),
                                   (self.matrix_regime, 'regime')]:
                if assembler.is_ready():
                    matrix = assembler.get_matrix()
                    if matrix is not None:
                        self.monitor.capture_matrix_snapshot(name, matrix)
        
        loop.run_until_complete(run_test())
        
        # Get summary
        summary = self.monitor.get_summary()
        
        # Verify pipeline worked
        self.assertGreater(summary['event_counts'][EventType.NEW_TICK], 0,
                          "No ticks generated")
        self.assertGreater(summary['event_counts'][EventType.INDICATORS_READY], 0,
                          "No indicators calculated")
        
        # Verify matrix updates
        self.assertGreater(summary['matrix_update_counts']['30m'], 0,
                          "No 30m matrix updates")
        self.assertGreater(summary['matrix_update_counts']['5m'], 0,
                          "No 5m matrix updates")
        
        # Verify performance
        self.assertLess(summary['performance']['avg_latency_ms'], 1.0,
                       "Average latency exceeds 1ms requirement")
        
        # Log results
        self.logger.info(f"Integration test summary: {summary}")
    
    def test_matrix_readiness_progression(self):
        """Test how matrices progress from not ready to ready state."""
        self.setup_pipeline()
        
        # Track readiness over time
        readiness_timeline = []
        
        def track_readiness():
            status = {
                'timestamp': datetime.now(),
                '30m_ready': self.matrix_30m.is_ready(),
                '5m_ready': self.matrix_5m.is_ready(),
                'regime_ready': self.matrix_regime.is_ready(),
                '30m_updates': self.matrix_30m.n_updates,
                '5m_updates': self.matrix_5m.n_updates,
                'regime_updates': self.matrix_regime.n_updates
            }
            readiness_timeline.append(status)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            await self.kernel.start()
            
            # Track readiness progression
            for i in range(20):  # 20 checks over 10 seconds
                track_readiness()
                await asyncio.sleep(0.5)
            
            # Final check
            track_readiness()
        
        loop.run_until_complete(run_test())
        
        # Analyze progression
        initial_status = readiness_timeline[0]
        final_status = readiness_timeline[-1]
        
        # Initially not ready
        self.assertFalse(initial_status['30m_ready'])
        self.assertFalse(initial_status['5m_ready'])
        
        # Eventually ready (5m should be ready first due to shorter warmup)
        self.assertTrue(final_status['5m_ready'])
        
        # 5m should have more updates than 30m
        self.assertGreater(final_status['5m_updates'], final_status['30m_updates'])
        
        self.logger.info(f"Readiness progression: {len(readiness_timeline)} snapshots")
        self.logger.info(f"Final state - 30m: {final_status['30m_updates']} updates, "
                        f"5m: {final_status['5m_updates']} updates")
    
    def test_matrix_content_validation(self):
        """Test that matrix content is valid and properly normalized."""
        self.setup_pipeline()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        validation_results = {}
        
        async def run_test():
            await self.kernel.start()
            
            # Let system warm up
            await asyncio.sleep(10)
            
            # Validate each matrix
            for assembler, name in [(self.matrix_30m, '30m'),
                                   (self.matrix_5m, '5m'),
                                   (self.matrix_regime, 'regime')]:
                if assembler.is_ready():
                    matrix = assembler.get_matrix()
                    if matrix is not None:
                        # Validate matrix
                        is_valid, issues = assembler.validate_matrix()
                        
                        validation_results[name] = {
                            'valid': is_valid,
                            'issues': issues,
                            'shape': matrix.shape,
                            'finite_ratio': np.sum(np.isfinite(matrix)) / matrix.size,
                            'value_range': (float(np.min(matrix)), float(np.max(matrix))),
                            'features': assembler.get_feature_names()
                        }
        
        loop.run_until_complete(run_test())
        
        # Check validation results
        for name, result in validation_results.items():
            self.assertTrue(result['valid'], 
                           f"{name} matrix validation failed: {result['issues']}")
            self.assertEqual(result['finite_ratio'], 1.0,
                           f"{name} matrix contains non-finite values")
            
            # Check normalized range
            min_val, max_val = result['value_range']
            self.assertGreaterEqual(min_val, -3.0,
                                   f"{name} matrix has values below -3")
            self.assertLessEqual(max_val, 3.0,
                                f"{name} matrix has values above 3")
        
        self.logger.info(f"Matrix validation results: {validation_results}")
    
    def test_concurrent_access_stress(self):
        """Stress test concurrent access to matrices."""
        self.setup_pipeline()
        
        access_results = []
        access_errors = []
        
        def reader_thread(assembler, name, iterations=1000):
            """Continuously read from matrix."""
            try:
                for i in range(iterations):
                    matrix = assembler.get_matrix()
                    if matrix is not None:
                        # Perform some computation to ensure data is accessed
                        result = {
                            'name': name,
                            'iteration': i,
                            'mean': float(np.mean(matrix)),
                            'shape': matrix.shape
                        }
                        access_results.append(result)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                access_errors.append((name, str(e)))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            await self.kernel.start()
            
            # Let matrices get some data
            await asyncio.sleep(5)
            
            # Start concurrent readers
            import threading
            threads = []
            
            for assembler, name in [(self.matrix_30m, '30m'),
                                   (self.matrix_5m, '5m'),
                                   (self.matrix_regime, 'regime')]:
                # Start 3 reader threads per assembler
                for i in range(3):
                    t = threading.Thread(
                        target=reader_thread,
                        args=(assembler, f"{name}_reader_{i}")
                    )
                    threads.append(t)
                    t.start()
            
            # Let readers run while system continues updating
            await asyncio.sleep(5)
            
            # Wait for threads
            for t in threads:
                t.join(timeout=10)
        
        loop.run_until_complete(run_test())
        
        # Verify no errors
        self.assertEqual(len(access_errors), 0,
                        f"Concurrent access errors: {access_errors}")
        
        # Verify successful reads
        self.assertGreater(len(access_results), 0,
                          "No successful matrix reads")
        
        # Check consistency
        by_reader = {}
        for result in access_results:
            name = result['name']
            if name not in by_reader:
                by_reader[name] = []
            by_reader[name].append(result['mean'])
        
        # Means should be relatively stable for each reader
        for reader, means in by_reader.items():
            if len(means) > 10:
                std = np.std(means)
                self.assertLess(std, 1.0,
                               f"High variance in {reader} reads: std={std}")
        
        self.logger.info(f"Concurrent access test: {len(access_results)} successful reads")


if __name__ == '__main__':
    unittest.main()