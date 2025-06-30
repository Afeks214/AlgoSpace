"""
End-to-End Testing Suite for AlgoSpace Trading System

This module provides comprehensive testing of the complete data pipeline
from tick data through to indicator calculations, ensuring all components
work together correctly with real ES historical data.

Test Coverage:
- Data loading and validation
- Tick to bar conversion
- Indicator calculations
- Event flow verification
- Performance benchmarking
- Error handling scenarios
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.kernel import AlgoSpaceKernel
from src.core.events import EventType, Event
from src.utils.logger import setup_logging, get_logger
from src.data.handlers import create_data_handler
from src.data.bar_generator import BarGenerator
from src.indicators.engine import IndicatorEngine


class TestMonitor:
    """
    Test monitoring component that tracks system events and validates behavior
    """
    
    def __init__(self):
        """Initialize test monitor"""
        self.logger = get_logger("TestMonitor")
        
        # Event counters
        self.event_counts = {
            EventType.NEW_TICK: 0,
            EventType.NEW_5MIN_BAR: 0,
            EventType.NEW_30MIN_BAR: 0,
            EventType.INDICATORS_READY: 0,
            EventType.SYSTEM_ERROR: 0,
            EventType.BACKTEST_COMPLETE: 0
        }
        
        # Event timing
        self.event_timings: Dict[str, List[float]] = {
            'tick_to_5min_bar': [],
            '5min_bar_to_indicators': [],
            '30min_bar_to_indicators': [],
            'tick_processing': []
        }
        
        # Data validation
        self.tick_prices: List[float] = []
        self.bar_5min_closes: List[float] = []
        self.bar_30min_closes: List[float] = []
        self.indicator_snapshots: List[Dict[str, Any]] = []
        
        # Error tracking
        self.errors: List[Dict[str, Any]] = []
        
        # Timing references
        self.last_tick_time: Optional[float] = None
        self.last_5min_bar_time: Optional[float] = None
        self.last_30min_bar_time: Optional[float] = None
        
        # Test results
        self.test_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
    
    def track_event(self, event: Event) -> None:
        """Track event for analysis"""
        event_type = event.event_type
        
        # Count events
        if event_type in self.event_counts:
            self.event_counts[event_type] += 1
        
        # Track timing
        current_time = time.time()
        
        if event_type == EventType.NEW_TICK:
            self._track_tick(event, current_time)
        elif event_type == EventType.NEW_5MIN_BAR:
            self._track_5min_bar(event, current_time)
        elif event_type == EventType.NEW_30MIN_BAR:
            self._track_30min_bar(event, current_time)
        elif event_type == EventType.INDICATORS_READY:
            self._track_indicators(event, current_time)
        elif event_type == EventType.SYSTEM_ERROR:
            self._track_error(event)
    
    def _track_tick(self, event: Event, current_time: float) -> None:
        """Track tick event"""
        tick_data = event.payload
        self.tick_prices.append(tick_data.price)
        
        if self.last_tick_time:
            processing_time = (current_time - self.last_tick_time) * 1000  # ms
            self.event_timings['tick_processing'].append(processing_time)
        
        self.last_tick_time = current_time
    
    def _track_5min_bar(self, event: Event, current_time: float) -> None:
        """Track 5-minute bar event"""
        bar_data = event.payload
        self.bar_5min_closes.append(bar_data.close)
        
        if self.last_tick_time:
            latency = (current_time - self.last_tick_time) * 1000  # ms
            self.event_timings['tick_to_5min_bar'].append(latency)
        
        self.last_5min_bar_time = current_time
    
    def _track_30min_bar(self, event: Event, current_time: float) -> None:
        """Track 30-minute bar event"""
        bar_data = event.payload
        self.bar_30min_closes.append(bar_data.close)
        
        self.last_30min_bar_time = current_time
    
    def _track_indicators(self, event: Event, current_time: float) -> None:
        """Track indicators ready event"""
        features = event.payload
        
        # Store snapshot
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'feature_count': features.get('feature_count', 0),
            'mlmi_value': features.get('mlmi_value', 0),
            'mlmi_signal': features.get('mlmi_signal', 0),
            'nwrqk_value': features.get('nwrqk_value', 0),
            'fvg_bullish_active': features.get('fvg_bullish_active', False),
            'lvn_nearest_price': features.get('lvn_nearest_price', 0)
        }
        self.indicator_snapshots.append(snapshot)
        
        # Track timing
        if self.last_5min_bar_time:
            latency = (current_time - self.last_5min_bar_time) * 1000  # ms
            self.event_timings['5min_bar_to_indicators'].append(latency)
        
        if self.last_30min_bar_time:
            latency = (current_time - self.last_30min_bar_time) * 1000  # ms
            self.event_timings['30min_bar_to_indicators'].append(latency)
    
    def _track_error(self, event: Event) -> None:
        """Track error event"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'component': event.payload.get('component', 'Unknown'),
            'error': event.payload.get('error', 'Unknown error'),
            'context': event.payload.get('context', {})
        }
        self.errors.append(error_info)
        self.logger.error("System error detected", **error_info)
    
    def run_validations(self) -> None:
        """Run all validation tests"""
        self.logger.info("Running validation tests...")
        
        # Test 1: Event Flow Validation
        self._validate_event_flow()
        
        # Test 2: Data Integrity Validation
        self._validate_data_integrity()
        
        # Test 3: Performance Validation
        self._validate_performance()
        
        # Test 4: Indicator Calculation Validation
        self._validate_indicators()
        
        # Test 5: Error Handling Validation
        self._validate_error_handling()
    
    def _validate_event_flow(self) -> None:
        """Validate event flow is correct"""
        # Check tick events generated
        if self.event_counts[EventType.NEW_TICK] > 0:
            self.test_results['passed'].append("✓ Tick events generated")
        else:
            self.test_results['failed'].append("✗ No tick events generated")
        
        # Check bar generation
        if self.event_counts[EventType.NEW_5MIN_BAR] > 0:
            self.test_results['passed'].append("✓ 5-minute bars generated")
        else:
            self.test_results['failed'].append("✗ No 5-minute bars generated")
        
        if self.event_counts[EventType.NEW_30MIN_BAR] > 0:
            self.test_results['passed'].append("✓ 30-minute bars generated")
        else:
            self.test_results['failed'].append("✗ No 30-minute bars generated")
        
        # Check indicator calculations
        if self.event_counts[EventType.INDICATORS_READY] > 0:
            self.test_results['passed'].append("✓ Indicators calculated")
        else:
            self.test_results['failed'].append("✗ No indicators calculated")
        
        # Check event ratios
        if self.event_counts[EventType.NEW_TICK] > 0 and self.event_counts[EventType.NEW_5MIN_BAR] > 0:
            tick_to_bar_ratio = self.event_counts[EventType.NEW_TICK] / self.event_counts[EventType.NEW_5MIN_BAR]
            if 1 <= tick_to_bar_ratio <= 1000:  # Reasonable range
                self.test_results['passed'].append(f"✓ Tick to 5min bar ratio reasonable: {tick_to_bar_ratio:.1f}")
            else:
                self.test_results['warnings'].append(f"⚠ Unusual tick to bar ratio: {tick_to_bar_ratio:.1f}")
    
    def _validate_data_integrity(self) -> None:
        """Validate data integrity throughout pipeline"""
        # Check price continuity
        if len(self.tick_prices) > 1:
            max_price_jump = max(abs(self.tick_prices[i] - self.tick_prices[i-1]) 
                               for i in range(1, len(self.tick_prices)))
            
            if max_price_jump < 100:  # Reasonable for ES futures
                self.test_results['passed'].append(f"✓ Price continuity maintained (max jump: {max_price_jump:.2f})")
            else:
                self.test_results['warnings'].append(f"⚠ Large price jump detected: {max_price_jump:.2f}")
        
        # Check bar consistency
        if len(self.bar_5min_closes) > 0:
            min_price = min(self.bar_5min_closes)
            max_price = max(self.bar_5min_closes)
            price_range = max_price - min_price
            
            self.test_results['passed'].append(
                f"✓ Bar price range: {min_price:.2f} - {max_price:.2f} (range: {price_range:.2f})"
            )
    
    def _validate_performance(self) -> None:
        """Validate system performance metrics"""
        # Tick processing speed
        if self.event_timings['tick_processing']:
            avg_tick_time = sum(self.event_timings['tick_processing']) / len(self.event_timings['tick_processing'])
            max_tick_time = max(self.event_timings['tick_processing'])
            
            if avg_tick_time < 1.0:  # Less than 1ms average
                self.test_results['passed'].append(f"✓ Tick processing fast (avg: {avg_tick_time:.2f}ms)")
            else:
                self.test_results['warnings'].append(f"⚠ Slow tick processing (avg: {avg_tick_time:.2f}ms)")
            
            if max_tick_time < 10.0:  # Less than 10ms max
                self.test_results['passed'].append(f"✓ Max tick processing acceptable ({max_tick_time:.2f}ms)")
            else:
                self.test_results['warnings'].append(f"⚠ Slow tick processing spike ({max_tick_time:.2f}ms)")
        
        # Bar generation latency
        if self.event_timings['tick_to_5min_bar']:
            avg_bar_latency = sum(self.event_timings['tick_to_5min_bar']) / len(self.event_timings['tick_to_5min_bar'])
            
            if avg_bar_latency < 5.0:  # Less than 5ms
                self.test_results['passed'].append(f"✓ Bar generation fast (avg: {avg_bar_latency:.2f}ms)")
            else:
                self.test_results['warnings'].append(f"⚠ Slow bar generation (avg: {avg_bar_latency:.2f}ms)")
        
        # Indicator calculation speed
        if self.event_timings['5min_bar_to_indicators']:
            avg_indicator_time = sum(self.event_timings['5min_bar_to_indicators']) / len(self.event_timings['5min_bar_to_indicators'])
            
            if avg_indicator_time < 50.0:  # Less than 50ms (PRD requirement)
                self.test_results['passed'].append(f"✓ 5min indicator calculation within PRD limit ({avg_indicator_time:.2f}ms < 50ms)")
            else:
                self.test_results['failed'].append(f"✗ 5min indicator calculation exceeds PRD limit ({avg_indicator_time:.2f}ms > 50ms)")
    
    def _validate_indicators(self) -> None:
        """Validate indicator calculations"""
        if not self.indicator_snapshots:
            self.test_results['failed'].append("✗ No indicator snapshots captured")
            return
        
        # Check indicator values are reasonable
        mlmi_values = [s['mlmi_value'] for s in self.indicator_snapshots if s['mlmi_value'] > 0]
        if mlmi_values:
            if all(0 <= v <= 100 for v in mlmi_values):
                self.test_results['passed'].append("✓ MLMI values in valid range [0-100]")
            else:
                self.test_results['failed'].append("✗ MLMI values out of range")
        
        # Check signal generation
        signals = [s['mlmi_signal'] for s in self.indicator_snapshots]
        unique_signals = set(signals)
        if len(unique_signals) > 1:
            self.test_results['passed'].append(f"✓ Multiple signal states generated: {unique_signals}")
        else:
            self.test_results['warnings'].append(f"⚠ Limited signal variation: {unique_signals}")
        
        # Check feature completeness
        last_snapshot = self.indicator_snapshots[-1]
        expected_features = ['mlmi_value', 'nwrqk_value', 'fvg_bullish_active', 'lvn_nearest_price']
        missing_features = [f for f in expected_features if f not in last_snapshot or last_snapshot[f] is None]
        
        if not missing_features:
            self.test_results['passed'].append("✓ All expected features present")
        else:
            self.test_results['failed'].append(f"✗ Missing features: {missing_features}")
    
    def _validate_error_handling(self) -> None:
        """Validate error handling"""
        if self.event_counts[EventType.SYSTEM_ERROR] == 0:
            self.test_results['passed'].append("✓ No system errors detected")
        else:
            self.test_results['warnings'].append(
                f"⚠ {self.event_counts[EventType.SYSTEM_ERROR]} system errors detected"
            )
            for error in self.errors[:3]:  # Show first 3 errors
                self.logger.warning(f"  Error: {error['component']} - {error['error']}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        # Calculate statistics
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed'])
        pass_rate = (len(self.test_results['passed']) / total_tests * 100) if total_tests > 0 else 0
        
        # Performance summary
        performance_summary = {}
        for metric, timings in self.event_timings.items():
            if timings:
                performance_summary[metric] = {
                    'avg_ms': sum(timings) / len(timings),
                    'max_ms': max(timings),
                    'min_ms': min(timings),
                    'count': len(timings)
                }
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed': len(self.test_results['passed']),
                'failed': len(self.test_results['failed']),
                'warnings': len(self.test_results['warnings']),
                'pass_rate': f"{pass_rate:.1f}%"
            },
            'event_counts': self.event_counts,
            'performance_metrics': performance_summary,
            'data_summary': {
                'total_ticks': len(self.tick_prices),
                'total_5min_bars': len(self.bar_5min_closes),
                'total_30min_bars': len(self.bar_30min_closes),
                'total_indicator_updates': len(self.indicator_snapshots)
            },
            'test_results': self.test_results,
            'errors': self.errors
        }
        
        return report


class EndToEndTester:
    """
    Main end-to-end testing orchestrator
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize tester"""
        setup_logging()
        self.logger = get_logger("EndToEndTester")
        
        # Create test kernel and monitor
        self.kernel = AlgoSpaceKernel(config_path)
        self.monitor = TestMonitor()
        
        # Test configuration
        self.test_duration = 60  # seconds
        self.start_time = None
    
    def setup_components(self) -> None:
        """Setup all system components with monitoring"""
        self.logger.info("Setting up components for testing...")
        
        # Create components
        self.data_handler = create_data_handler(self.kernel)
        self.bar_generator = BarGenerator("BarGenerator", self.kernel)
        self.indicator_engine = IndicatorEngine("IndicatorEngine", self.kernel)
        
        # Register components
        self.kernel.register_component("DataHandler", self.data_handler)
        self.kernel.register_component("BarGenerator", self.bar_generator, ["DataHandler"])
        self.kernel.register_component("IndicatorEngine", self.indicator_engine, ["BarGenerator"])
        
        # Setup monitoring
        self._setup_monitoring()
        
        self.logger.info("Components setup complete")
    
    def _setup_monitoring(self) -> None:
        """Setup event monitoring"""
        event_bus = self.kernel.get_event_bus()
        
        # Monitor all critical events
        for event_type in [EventType.NEW_TICK, EventType.NEW_5MIN_BAR, 
                          EventType.NEW_30MIN_BAR, EventType.INDICATORS_READY,
                          EventType.SYSTEM_ERROR, EventType.BACKTEST_COMPLETE]:
            event_bus.subscribe(event_type, self.monitor.track_event)
        
        # Special handling for backtest complete
        event_bus.subscribe(EventType.BACKTEST_COMPLETE, self._on_backtest_complete)
    
    def _on_backtest_complete(self, event: Event) -> None:
        """Handle backtest completion"""
        self.logger.info("Backtest completed - running validations")
        asyncio.create_task(self._complete_test())
    
    async def _complete_test(self) -> None:
        """Complete the test and generate report"""
        # Run validations
        self.monitor.run_validations()
        
        # Generate report
        report = self.monitor.generate_report()
        
        # Save report
        self._save_report(report)
        
        # Print summary
        self._print_summary(report)
        
        # Shutdown
        await self.kernel.shutdown()
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save test report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"tests/reports/test_report_{timestamp}.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Test report saved to: {report_path}")
    
    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print test summary to console"""
        print("\n" + "="*80)
        print("                    END-TO-END TEST RESULTS")
        print("="*80)
        
        # Test Summary
        summary = report['test_summary']
        print(f"\nTest Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']} ✓")
        print(f"  Failed: {summary['failed']} ✗")
        print(f"  Warnings: {summary['warnings']} ⚠")
        print(f"  Pass Rate: {summary['pass_rate']}")
        
        # Event Counts
        print(f"\nEvent Processing:")
        for event_type, count in report['event_counts'].items():
            if count > 0:
                print(f"  {event_type.value}: {count:,}")
        
        # Performance Summary
        print(f"\nPerformance Metrics:")
        for metric, stats in report['performance_metrics'].items():
            print(f"  {metric}:")
            print(f"    Average: {stats['avg_ms']:.2f}ms")
            print(f"    Max: {stats['max_ms']:.2f}ms")
        
        # Test Results
        print(f"\nTest Results:")
        print("  Passed:")
        for result in report['test_results']['passed'][:5]:  # First 5
            print(f"    {result}")
        
        if report['test_results']['failed']:
            print("  Failed:")
            for result in report['test_results']['failed']:
                print(f"    {result}")
        
        if report['test_results']['warnings']:
            print("  Warnings:")
            for result in report['test_results']['warnings'][:3]:  # First 3
                print(f"    {result}")
        
        print("\n" + "="*80)
    
    async def run_test(self) -> None:
        """Run the end-to-end test"""
        try:
            self.logger.info("="*60)
            self.logger.info("    Starting End-to-End Test")
            self.logger.info("="*60)
            
            self.start_time = datetime.now()
            
            # Setup components
            self.setup_components()
            
            # Start the system
            await self.kernel.start()
            
        except Exception as e:
            self.logger.error(f"Test failed with error: {e}")
            raise


async def main():
    """Main test runner"""
    # Create tester
    tester = EndToEndTester()
    
    # Run test
    await tester.run_test()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())