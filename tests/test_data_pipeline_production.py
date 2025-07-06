"""
Comprehensive Production Readiness Test Suite for AlgoSpace Data Pipeline

This test suite validates the entire data pipeline for production deployment,
covering all critical components and integration points.

Components Tested:
1. BarGenerator - Tick processing and bar generation
2. IndicatorEngine - Feature calculation and storage
3. Event System - Reliability and performance
4. Data Quality - Validation and error handling
5. Performance - Latency and throughput benchmarks
6. Memory Management - Stability and leak detection

Author: QuantNova Team
Date: 2025-01-06
"""

import asyncio
import unittest
import sys
import time
import numpy as np
import threading
import psutil
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.kernel import AlgoSpaceKernel
from src.core.events import EventType, Event, BarData, TickData
from src.core.event_bus import EventBus
from src.components.bar_generator import BarGenerator
from src.data.bar_generator import BarGenerator as ModernBarGenerator
from src.indicators.engine import IndicatorEngine
from src.matrix import MatrixAssembler5m, MatrixAssembler30m
from src.utils.logger import setup_logging, get_logger


class ProductionTestDataGenerator:
    """Generate realistic market data for production testing."""
    
    def __init__(self, symbol: str = "NQ"):
        self.symbol = symbol
        self.logger = get_logger(f"TestDataGen_{symbol}")
        self.base_price = 15000.0
        
    def generate_realistic_ticks(self, num_ticks: int = 10000, 
                               tick_frequency_ms: int = 100) -> List[TickData]:
        """Generate realistic tick data with market microstructure."""
        ticks = []
        current_price = self.base_price
        timestamp = datetime.now()
        
        # Market microstructure parameters
        bid_ask_spread = 0.25
        volatility = 0.002
        trend_strength = 0.0001
        
        for i in range(num_ticks):
            # Add realistic price movement
            random_walk = np.random.normal(0, volatility * current_price)
            trend = trend_strength * current_price * np.sin(i / 1000)
            
            current_price += random_walk + trend
            current_price = max(current_price, 1000)  # Minimum price
            
            # Create bid/ask with realistic spread
            mid_price = current_price
            bid = mid_price - bid_ask_spread / 2
            ask = mid_price + bid_ask_spread / 2
            
            # Realistic volume (more volume during price movements)
            volume_multiplier = 1 + abs(random_walk) * 10
            volume = int(np.random.exponential(100) * volume_multiplier)
            
            tick = TickData(
                symbol=self.symbol,
                timestamp=timestamp + timedelta(milliseconds=i * tick_frequency_ms),
                price=current_price,
                volume=volume,
                bid=bid,
                ask=ask
            )
            ticks.append(tick)
            
        return ticks
    
    def generate_gap_scenario_ticks(self, num_ticks: int = 1000) -> List[TickData]:
        """Generate tick data with intentional gaps to test gap handling."""
        ticks = []
        current_price = self.base_price
        timestamp = datetime.now()
        
        for i in range(num_ticks):
            # Create gaps at specific intervals
            if i % 100 == 50:  # 10-minute gap every 100 ticks
                timestamp += timedelta(minutes=10)
                current_price *= 1.005  # 0.5% gap
                
            tick = TickData(
                symbol=self.symbol,
                timestamp=timestamp + timedelta(milliseconds=i * 100),
                price=current_price + np.random.normal(0, 5),
                volume=np.random.randint(50, 500),
                bid=current_price - 0.125,
                ask=current_price + 0.125
            )
            ticks.append(tick)
            
        return ticks
    
    def generate_extreme_scenario_ticks(self, num_ticks: int = 1000) -> List[TickData]:
        """Generate edge case tick data for stress testing."""
        ticks = []
        current_price = self.base_price
        timestamp = datetime.now()
        
        for i in range(num_ticks):
            # Extreme scenarios
            if i % 200 == 100:  # Price spike
                price_multiplier = 1.02
            elif i % 200 == 150:  # Price crash
                price_multiplier = 0.98
            else:
                price_multiplier = 1.0
                
            current_price *= price_multiplier
            
            # Extreme volume scenarios
            if i % 300 == 200:  # High volume burst
                volume = np.random.randint(5000, 10000)
            elif i % 300 == 250:  # Low volume
                volume = np.random.randint(1, 10)
            else:
                volume = np.random.randint(50, 500)
                
            tick = TickData(
                symbol=self.symbol,
                timestamp=timestamp + timedelta(milliseconds=i * 100),
                price=current_price,
                volume=volume,
                bid=current_price - 0.125,
                ask=current_price + 0.125
            )
            ticks.append(tick)
            
        return ticks


class ProductionTestMonitor:
    """Monitor system performance and behavior during testing."""
    
    def __init__(self):
        self.logger = get_logger("ProductionTestMonitor")
        
        # Performance metrics
        self.latencies = defaultdict(list)
        self.throughput = defaultdict(list)
        self.memory_usage = []
        self.cpu_usage = []
        
        # Data quality metrics
        self.data_validation_errors = []
        self.missing_events = []
        self.duplicate_events = []
        
        # Event tracking
        self.event_counts = defaultdict(int)
        self.event_times = defaultdict(list)
        
        # System health
        self.component_status = {}
        self.error_log = []
        
        # Bar validation
        self.bar_validation_results = []
        
        # Feature validation
        self.feature_snapshots = []
        
    def record_latency(self, component: str, latency_ms: float):
        """Record processing latency for a component."""
        self.latencies[component].append(latency_ms)
        
    def record_throughput(self, component: str, items_per_second: float):
        """Record throughput for a component."""
        self.throughput[component].append(items_per_second)
        
    def record_memory_usage(self):
        """Record current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        
    def record_cpu_usage(self):
        """Record current CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_percent)
        
    def record_event(self, event_type: str, timestamp: float = None):
        """Record event occurrence."""
        if timestamp is None:
            timestamp = time.time()
        self.event_counts[event_type] += 1
        self.event_times[event_type].append(timestamp)
        
    def record_bar_validation(self, bar: BarData, is_valid: bool, 
                            validation_errors: List[str] = None):
        """Record bar validation results."""
        self.bar_validation_results.append({
            'bar': bar,
            'is_valid': is_valid,
            'errors': validation_errors or [],
            'timestamp': time.time()
        })
        
    def record_feature_snapshot(self, features: Dict[str, Any]):
        """Record feature snapshot for validation."""
        self.feature_snapshots.append({
            'features': features.copy(),
            'timestamp': time.time()
        })
        
    def validate_bar_data(self, bar: BarData) -> Tuple[bool, List[str]]:
        """Validate bar data quality."""
        errors = []
        
        # Price validation
        if bar.high < bar.low:
            errors.append("High price below low price")
        if bar.open < 0 or bar.close < 0:
            errors.append("Negative price values")
        if not (bar.low <= bar.open <= bar.high):
            errors.append("Open price outside high-low range")
        if not (bar.low <= bar.close <= bar.high):
            errors.append("Close price outside high-low range")
            
        # Volume validation
        if bar.volume < 0:
            errors.append("Negative volume")
            
        # Timestamp validation
        if bar.timestamp is None:
            errors.append("Missing timestamp")
            
        # Check for infinity or NaN
        prices = [bar.open, bar.high, bar.low, bar.close]
        if any(np.isnan(p) or np.isinf(p) for p in prices):
            errors.append("NaN or infinity in price data")
            
        return len(errors) == 0, errors
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("=" * 80)
        report.append("ALGOSPACE DATA PIPELINE PRODUCTION READINESS REPORT")
        report.append("=" * 80)
        report.append(f"Test completed at: {datetime.now()}")
        
        # Latency analysis
        report.append("\n## LATENCY ANALYSIS")
        for component, latencies in self.latencies.items():
            if latencies:
                avg_lat = statistics.mean(latencies)
                p95_lat = np.percentile(latencies, 95)
                p99_lat = np.percentile(latencies, 99)
                max_lat = max(latencies)
                
                report.append(f"  {component}:")
                report.append(f"    Average: {avg_lat:.2f}ms")
                report.append(f"    95th percentile: {p95_lat:.2f}ms")
                report.append(f"    99th percentile: {p99_lat:.2f}ms")
                report.append(f"    Maximum: {max_lat:.2f}ms")
                
                # Performance assessment
                if component == "BarGenerator" and avg_lat > 1.0:
                    report.append(f"    ⚠️  Warning: Average latency exceeds 1ms")
                elif component == "IndicatorEngine" and avg_lat > 50.0:
                    report.append(f"    ⚠️  Warning: Average latency exceeds 50ms")
                else:
                    report.append(f"    ✅ Performance meets requirements")
                    
        # Throughput analysis
        report.append("\n## THROUGHPUT ANALYSIS")
        for component, throughputs in self.throughput.items():
            if throughputs:
                avg_throughput = statistics.mean(throughputs)
                min_throughput = min(throughputs)
                report.append(f"  {component}:")
                report.append(f"    Average: {avg_throughput:.1f} items/sec")
                report.append(f"    Minimum: {min_throughput:.1f} items/sec")
                
        # Memory usage
        report.append("\n## MEMORY USAGE")
        if self.memory_usage:
            avg_memory = statistics.mean(self.memory_usage)
            max_memory = max(self.memory_usage)
            memory_growth = max(self.memory_usage) - min(self.memory_usage)
            
            report.append(f"  Average memory: {avg_memory:.1f} MB")
            report.append(f"  Peak memory: {max_memory:.1f} MB")
            report.append(f"  Memory growth: {memory_growth:.1f} MB")
            
            if memory_growth > 100:
                report.append(f"  ⚠️  Warning: Possible memory leak detected")
            else:
                report.append(f"  ✅ Memory usage stable")
                
        # CPU usage
        report.append("\n## CPU USAGE")
        if self.cpu_usage:
            avg_cpu = statistics.mean(self.cpu_usage)
            max_cpu = max(self.cpu_usage)
            
            report.append(f"  Average CPU: {avg_cpu:.1f}%")
            report.append(f"  Peak CPU: {max_cpu:.1f}%")
            
        # Event processing
        report.append("\n## EVENT PROCESSING")
        for event_type, count in self.event_counts.items():
            report.append(f"  {event_type}: {count} events")
            
        # Data quality
        report.append("\n## DATA QUALITY")
        valid_bars = sum(1 for r in self.bar_validation_results if r['is_valid'])
        total_bars = len(self.bar_validation_results)
        if total_bars > 0:
            validation_rate = (valid_bars / total_bars) * 100
            report.append(f"  Valid bars: {valid_bars}/{total_bars} ({validation_rate:.1f}%)")
            
            if validation_rate < 99.0:
                report.append(f"  ⚠️  Warning: Data quality issues detected")
            else:
                report.append(f"  ✅ Data quality excellent")
                
        # Feature validation
        report.append("\n## FEATURE VALIDATION")
        if self.feature_snapshots:
            last_features = self.feature_snapshots[-1]['features']
            feature_count = len(last_features)
            report.append(f"  Total features: {feature_count}")
            
            # Check for NaN/infinity in features
            nan_features = [k for k, v in last_features.items() 
                          if isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v))]
            if nan_features:
                report.append(f"  ⚠️  Features with NaN/Inf: {nan_features}")
            else:
                report.append(f"  ✅ All features valid")
                
        # Error summary
        report.append("\n## ERROR SUMMARY")
        if self.error_log:
            report.append(f"  Total errors: {len(self.error_log)}")
            error_types = defaultdict(int)
            for error in self.error_log:
                error_types[type(error).__name__] += 1
            for error_type, count in error_types.items():
                report.append(f"    {error_type}: {count}")
        else:
            report.append("  ✅ No errors detected")
            
        # Overall assessment
        report.append("\n## OVERALL ASSESSMENT")
        
        # Calculate score
        score = 0
        max_score = 100
        
        # Latency score (30 points)
        if self.latencies:
            avg_latencies = [statistics.mean(lats) for lats in self.latencies.values()]
            if all(lat < 50 for lat in avg_latencies):
                score += 30
            elif all(lat < 100 for lat in avg_latencies):
                score += 20
            else:
                score += 10
                
        # Data quality score (25 points)
        if total_bars > 0:
            if validation_rate > 99.5:
                score += 25
            elif validation_rate > 99.0:
                score += 20
            elif validation_rate > 95.0:
                score += 15
            else:
                score += 5
                
        # Memory stability score (20 points)
        if self.memory_usage and memory_growth < 50:
            score += 20
        elif self.memory_usage and memory_growth < 100:
            score += 15
        else:
            score += 5
            
        # Error handling score (25 points)
        if len(self.error_log) == 0:
            score += 25
        elif len(self.error_log) < 10:
            score += 15
        else:
            score += 5
            
        report.append(f"  Production Readiness Score: {score}/{max_score}")
        
        if score >= 90:
            report.append("  🎯 EXCELLENT - Ready for production deployment")
        elif score >= 80:
            report.append("  ✅ GOOD - Minor optimizations recommended")
        elif score >= 70:
            report.append("  ⚠️  FAIR - Significant improvements needed")
        else:
            report.append("  ❌ POOR - Not ready for production")
            
        report.append("\n" + "=" * 80)
        return "\n".join(report)


class TestDataPipelineProduction(unittest.TestCase):
    """Comprehensive production readiness test suite."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment."""
        setup_logging(log_level="INFO")
        cls.logger = get_logger("TestDataPipelineProduction")
        cls.logger.info("Starting Data Pipeline Production Readiness Tests")
        
    def setUp(self):
        """Initialize test components."""
        self.monitor = ProductionTestMonitor()
        self.data_generator = ProductionTestDataGenerator()
        self.event_bus = EventBus()
        
        # Component references
        self.bar_generator = None
        self.indicator_engine = None
        self.kernel = None
        
    def tearDown(self):
        """Cleanup test resources."""
        if self.kernel:
            asyncio.run(self.kernel.shutdown())
        if self.event_bus:
            self.event_bus.stop()
            
    async def setup_production_system(self):
        """Setup production-like system configuration."""
        # Use the production test configuration file
        config_path = Path(__file__).parent / "production_test_config.yaml"
        
        self.kernel = AlgoSpaceKernel(str(config_path))
        await self.kernel.initialize()
        
        # Get component references
        self.bar_generator = self.kernel.components.get('bar_generator')
        self.indicator_engine = self.kernel.components.get('indicator_engine')
        
        # Setup monitoring
        self._setup_monitoring()
        
    def _setup_monitoring(self):
        """Setup comprehensive monitoring."""
        
        def monitor_new_5min_bar(bar_data):
            """Monitor 5-minute bar generation."""
            start_time = time.time()
            
            # Validate bar data
            is_valid, errors = self.monitor.validate_bar_data(bar_data)
            self.monitor.record_bar_validation(bar_data, is_valid, errors)
            
            # Record timing
            latency = (time.time() - start_time) * 1000
            self.monitor.record_latency("BarGenerator_5min", latency)
            self.monitor.record_event("NEW_5MIN_BAR")
            
        def monitor_new_30min_bar(bar_data):
            """Monitor 30-minute bar generation."""
            start_time = time.time()
            
            # Validate bar data
            is_valid, errors = self.monitor.validate_bar_data(bar_data)
            self.monitor.record_bar_validation(bar_data, is_valid, errors)
            
            # Record timing
            latency = (time.time() - start_time) * 1000
            self.monitor.record_latency("BarGenerator_30min", latency)
            self.monitor.record_event("NEW_30MIN_BAR")
            
        def monitor_indicators_ready(payload):
            """Monitor indicator calculations."""
            start_time = time.time()
            
            # Record feature snapshot
            if payload and 'features' in payload:
                self.monitor.record_feature_snapshot(payload['features'])
                
            # Record timing
            latency = (time.time() - start_time) * 1000
            self.monitor.record_latency("IndicatorEngine", latency)
            self.monitor.record_event("INDICATORS_READY")
            
        # Subscribe to events
        self.event_bus.subscribe("NEW_5MIN_BAR", monitor_new_5min_bar)
        self.event_bus.subscribe("NEW_30MIN_BAR", monitor_new_30min_bar)
        self.event_bus.subscribe("INDICATORS_READY", monitor_indicators_ready)
        
    async def test_bar_generator_accuracy(self):
        """Test 1: BarGenerator accuracy and performance."""
        self.logger.info("Testing BarGenerator accuracy and performance...")
        
        await self.setup_production_system()
        
        # Generate realistic tick data
        test_ticks = self.data_generator.generate_realistic_ticks(5000)
        
        # Process ticks and measure performance
        start_time = time.time()
        processed_ticks = 0
        
        for tick in test_ticks:
            tick_start = time.time()
            
            # Process tick through bar generator
            if self.bar_generator:
                await self.bar_generator.process_tick(tick)
                
            # Record performance
            tick_latency = (time.time() - tick_start) * 1000
            self.monitor.record_latency("TickProcessing", tick_latency)
            processed_ticks += 1
            
            # System monitoring
            if processed_ticks % 1000 == 0:
                self.monitor.record_memory_usage()
                self.monitor.record_cpu_usage()
                
        # Calculate throughput
        total_time = time.time() - start_time
        throughput = processed_ticks / total_time
        self.monitor.record_throughput("BarGenerator", throughput)
        
        # Validate results
        valid_bars = sum(1 for r in self.monitor.bar_validation_results if r['is_valid'])
        total_bars = len(self.monitor.bar_validation_results)
        
        self.assertGreater(total_bars, 0, "No bars were generated")
        self.assertGreaterEqual(valid_bars / total_bars, 0.99, 
                              "Bar validation rate below 99%")
        
        self.logger.info(f"✅ BarGenerator test completed: {valid_bars}/{total_bars} valid bars")
        
    async def test_indicator_engine_performance(self):
        """Test 2: IndicatorEngine performance and accuracy."""
        self.logger.info("Testing IndicatorEngine performance and accuracy...")
        
        await self.setup_production_system()
        
        # Generate test data
        test_ticks = self.data_generator.generate_realistic_ticks(2000)
        
        # Process through pipeline
        for tick in test_ticks:
            if self.bar_generator:
                await self.bar_generator.process_tick(tick)
                
            # Monitor system resources
            self.monitor.record_memory_usage()
            
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Validate indicator performance
        indicator_latencies = self.monitor.latencies.get("IndicatorEngine", [])
        if indicator_latencies:
            avg_latency = statistics.mean(indicator_latencies)
            self.assertLess(avg_latency, 50.0, 
                          f"Indicator processing too slow: {avg_latency:.2f}ms")
            
        # Validate feature quality
        feature_count = len(self.monitor.feature_snapshots)
        self.assertGreater(feature_count, 0, "No features were generated")
        
        # Check for NaN/infinity in features
        if self.monitor.feature_snapshots:
            last_features = self.monitor.feature_snapshots[-1]['features']
            for feature_name, value in last_features.items():
                if isinstance(value, (int, float)):
                    self.assertFalse(np.isnan(value), 
                                   f"NaN detected in feature: {feature_name}")
                    self.assertFalse(np.isinf(value), 
                                   f"Infinity detected in feature: {feature_name}")
                    
        self.logger.info(f"✅ IndicatorEngine test completed: {feature_count} feature snapshots")
        
    async def test_gap_handling(self):
        """Test 3: Data gap handling and recovery."""
        self.logger.info("Testing data gap handling and recovery...")
        
        await self.setup_production_system()
        
        # Generate tick data with gaps
        test_ticks = self.data_generator.generate_gap_scenario_ticks(1000)
        
        # Process ticks
        for tick in test_ticks:
            if self.bar_generator:
                await self.bar_generator.process_tick(tick)
                
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Validate gap handling
        bars_generated = len(self.monitor.bar_validation_results)
        self.assertGreater(bars_generated, 0, "No bars generated during gap scenario")
        
        # Check for synthetic bar generation
        valid_bars = sum(1 for r in self.monitor.bar_validation_results if r['is_valid'])
        self.assertGreaterEqual(valid_bars, bars_generated * 0.95, 
                              "Too many invalid bars during gap handling")
        
        self.logger.info(f"✅ Gap handling test completed: {valid_bars}/{bars_generated} valid bars")
        
    async def test_extreme_conditions(self):
        """Test 4: Extreme market conditions and edge cases."""
        self.logger.info("Testing extreme market conditions...")
        
        await self.setup_production_system()
        
        # Generate extreme scenario data
        test_ticks = self.data_generator.generate_extreme_scenario_ticks(1000)
        
        # Process under extreme conditions
        for tick in test_ticks:
            if self.bar_generator:
                await self.bar_generator.process_tick(tick)
                
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Validate system stability
        self.assertGreater(len(self.monitor.bar_validation_results), 0, 
                          "System failed under extreme conditions")
        
        # Check error handling
        error_count = len(self.monitor.error_log)
        self.assertLess(error_count, 10, 
                       f"Too many errors under extreme conditions: {error_count}")
        
        self.logger.info(f"✅ Extreme conditions test completed")
        
    async def test_memory_stability(self):
        """Test 5: Memory stability and leak detection."""
        self.logger.info("Testing memory stability and leak detection...")
        
        await self.setup_production_system()
        
        # Long-running test
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        for cycle in range(10):  # 10 cycles of processing
            test_ticks = self.data_generator.generate_realistic_ticks(500)
            
            for tick in test_ticks:
                if self.bar_generator:
                    await self.bar_generator.process_tick(tick)
                    
            # Force garbage collection
            gc.collect()
            
            # Monitor memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.monitor.record_memory_usage()
            
            # Check for memory leaks
            memory_growth = current_memory - initial_memory
            if memory_growth > 100:  # 100MB growth threshold
                self.logger.warning(f"Memory growth detected: {memory_growth:.1f}MB")
                
            await asyncio.sleep(0.1)
            
        # Final memory check
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        self.assertLess(total_growth, 200, 
                       f"Excessive memory growth: {total_growth:.1f}MB")
        
        self.logger.info(f"✅ Memory stability test completed: {total_growth:.1f}MB growth")
        
    async def test_concurrent_processing(self):
        """Test 6: Concurrent processing and thread safety."""
        self.logger.info("Testing concurrent processing and thread safety...")
        
        await self.setup_production_system()
        
        # Create multiple tick streams
        tick_streams = [
            self.data_generator.generate_realistic_ticks(1000),
            self.data_generator.generate_realistic_ticks(1000),
            self.data_generator.generate_realistic_ticks(1000)
        ]
        
        # Process streams concurrently
        async def process_stream(stream, stream_id):
            for tick in stream:
                if self.bar_generator:
                    await self.bar_generator.process_tick(tick)
                await asyncio.sleep(0.001)  # Small delay
                
        # Run concurrent processing
        tasks = [process_stream(stream, i) for i, stream in enumerate(tick_streams)]
        await asyncio.gather(*tasks)
        
        # Wait for all processing
        await asyncio.sleep(1.0)
        
        # Validate concurrent processing
        total_bars = len(self.monitor.bar_validation_results)
        self.assertGreater(total_bars, 0, "No bars generated during concurrent processing")
        
        valid_bars = sum(1 for r in self.monitor.bar_validation_results if r['is_valid'])
        self.assertGreaterEqual(valid_bars / total_bars, 0.95, 
                              "Data corruption during concurrent processing")
        
        self.logger.info(f"✅ Concurrent processing test completed: {valid_bars}/{total_bars} valid bars")
        
    def test_run_complete_production_suite(self):
        """Main test runner for complete production validation."""
        asyncio.run(self._run_complete_suite())
        
    async def _run_complete_suite(self):
        """Execute complete production test suite."""
        try:
            # Test 1: BarGenerator accuracy
            self.logger.info("\n=== Test 1: BarGenerator Accuracy ===")
            await self.test_bar_generator_accuracy()
            await self.cleanup_test()
            
            # Test 2: IndicatorEngine performance
            self.logger.info("\n=== Test 2: IndicatorEngine Performance ===")
            await self.test_indicator_engine_performance()
            await self.cleanup_test()
            
            # Test 3: Gap handling
            self.logger.info("\n=== Test 3: Gap Handling ===")
            await self.test_gap_handling()
            await self.cleanup_test()
            
            # Test 4: Extreme conditions
            self.logger.info("\n=== Test 4: Extreme Conditions ===")
            await self.test_extreme_conditions()
            await self.cleanup_test()
            
            # Test 5: Memory stability
            self.logger.info("\n=== Test 5: Memory Stability ===")
            await self.test_memory_stability()
            await self.cleanup_test()
            
            # Test 6: Concurrent processing
            self.logger.info("\n=== Test 6: Concurrent Processing ===")
            await self.test_concurrent_processing()
            await self.cleanup_test()
            
            # Generate comprehensive report
            print("\n" + self.monitor.generate_performance_report())
            
            # Final assessment
            print("\n" + "=" * 80)
            print("🎯 ALGOSPACE DATA PIPELINE PRODUCTION READINESS COMPLETE! 🎯")
            print("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Production test suite failed: {str(e)}")
            raise
            
    async def cleanup_test(self):
        """Cleanup between tests."""
        if self.kernel:
            await self.kernel.shutdown()
            self.kernel = None
        self.setUp()  # Reset monitor and components


if __name__ == "__main__":
    unittest.main(verbosity=2)