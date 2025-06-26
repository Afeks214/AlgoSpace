#!/usr/bin/env python3
"""
Performance Benchmarking Suite for AlgoSpace System
"""

import time
import sys
import psutil
import os
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
import memory_profiler

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.kernel import SystemKernel
from src.core.events import EventType, Event
from src.core.config import get_config
from src.data.handlers import create_data_handler
from src.data.bar_generator import BarGenerator
from src.indicators.engine import IndicatorEngine
from src.matrix import MatrixAssembler30m, MatrixAssembler5m, MatrixAssemblerRegime


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.kernel = SystemKernel()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': {},
            'resource_usage': {},
            'component_performance': {}
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': round(psutil.virtual_memory().total / 1024**3, 2),
            'memory_available_gb': round(psutil.virtual_memory().available / 1024**3, 2),
            'platform': sys.platform,
            'python_version': sys.version.split()[0]
        }
    
    def benchmark_matrix_assemblers(self) -> Dict[str, Any]:
        """Benchmark matrix assembler performance."""
        print("Benchmarking Matrix Assemblers...")
        
        # Create assemblers
        assembler_30m = MatrixAssembler30m("Bench30m", self.kernel)
        assembler_5m = MatrixAssembler5m("Bench5m", self.kernel)
        assembler_regime = MatrixAssemblerRegime("BenchRegime", self.kernel)
        
        # Test data
        test_features = {
            'current_price': 100.0,
            'current_volume': 1000,
            'mlmi_value': 65.0,
            'mlmi_signal': 1,
            'nwrqk_value': 102.0,
            'nwrqk_slope': 0.5,
            'lvn_distance_points': 5.0,
            'lvn_nearest_strength': 85.0,
            'fvg_bullish_active': True,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 99.5,
            'fvg_age': 5,
            'fvg_mitigation_signal': False,
            'mmd_features': [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1],
            'timestamp': datetime.now()
        }
        
        # Warm up
        for _ in range(100):
            assembler_30m._update_matrix(test_features)
            assembler_5m._update_matrix(test_features)
            assembler_regime._update_matrix(test_features)
        
        results = {}
        
        # Test 30m assembler
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            assembler_30m._update_matrix(test_features)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        results['assembler_30m'] = {
            'update_time_avg_ms': np.mean(times),
            'update_time_max_ms': np.max(times),
            'update_time_p95_ms': np.percentile(times, 95),
            'update_time_p99_ms': np.percentile(times, 99)
        }
        
        # Test 5m assembler
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            assembler_5m._update_matrix(test_features)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        results['assembler_5m'] = {
            'update_time_avg_ms': np.mean(times),
            'update_time_max_ms': np.max(times),
            'update_time_p95_ms': np.percentile(times, 95),
            'update_time_p99_ms': np.percentile(times, 99)
        }
        
        # Test regime assembler
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            assembler_regime._update_matrix(test_features)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        results['assembler_regime'] = {
            'update_time_avg_ms': np.mean(times),
            'update_time_max_ms': np.max(times),
            'update_time_p95_ms': np.percentile(times, 95),
            'update_time_p99_ms': np.percentile(times, 99)
        }
        
        # Test matrix access speed
        access_times = []
        for _ in range(10000):
            start = time.perf_counter()
            matrix = assembler_30m.get_matrix()
            access_times.append((time.perf_counter() - start) * 1000000)  # μs
        
        results['matrix_access'] = {
            'access_time_avg_us': np.mean(access_times),
            'access_time_max_us': np.max(access_times),
            'access_time_p95_us': np.percentile(access_times, 95)
        }
        
        return results
    
    def benchmark_indicators(self) -> Dict[str, Any]:
        """Benchmark indicator calculation performance."""
        print("Benchmarking Indicator Engine...")
        
        # Create indicator engine
        indicator_engine = IndicatorEngine("BenchIndicators", self.kernel)
        
        # Create test bar data
        test_bar = type('BarData', (), {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000,
            'symbol': 'ES',
            'timeframe': '5m'
        })()
        
        # Warm up with 100 bars
        for i in range(100):
            test_bar.close = 100.0 + i * 0.1
            indicator_engine._on_5min_bar(test_bar)
        
        results = {}
        
        # Test 5-minute indicator calculation
        times = []
        for i in range(1000):
            start = time.perf_counter()
            test_bar.close = 100.0 + i * 0.01
            indicator_engine._on_5min_bar(test_bar)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        results['indicators_5m'] = {
            'calculation_time_avg_ms': np.mean(times),
            'calculation_time_max_ms': np.max(times),
            'calculation_time_p95_ms': np.percentile(times, 95),
            'calculation_time_p99_ms': np.percentile(times, 99)
        }
        
        # Test 30-minute calculation
        test_bar.timeframe = '30m'
        times = []
        for i in range(100):
            start = time.perf_counter()
            test_bar.close = 100.0 + i * 0.1
            indicator_engine._on_30min_bar(test_bar)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        results['indicators_30m'] = {
            'calculation_time_avg_ms': np.mean(times),
            'calculation_time_max_ms': np.max(times),
            'calculation_time_p95_ms': np.percentile(times, 95),
            'calculation_time_p99_ms': np.percentile(times, 99)
        }
        
        return results
    
    def benchmark_event_system(self) -> Dict[str, Any]:
        """Benchmark event system performance."""
        print("Benchmarking Event System...")
        
        event_bus = self.kernel.get_event_bus()
        
        # Event handling metrics
        received_events = []
        
        def event_handler(event):
            received_events.append(time.perf_counter())
        
        event_bus.subscribe(EventType.NEW_TICK, event_handler)
        
        # Publish events and measure latency
        publish_times = []
        for i in range(10000):
            start = time.perf_counter()
            
            # Create tick data
            tick_data = type('TickData', (), {
                'symbol': 'ES',
                'price': 100.0 + i * 0.01,
                'volume': 1,
                'timestamp': datetime.now()
            })()
            
            event = event_bus.create_event(EventType.NEW_TICK, tick_data, source="Benchmark")
            event_bus.publish(event)
            
            publish_times.append((time.perf_counter() - start) * 1000000)  # μs
        
        # Allow time for event processing
        time.sleep(0.1)
        
        return {
            'event_publish_avg_us': np.mean(publish_times),
            'event_publish_max_us': np.max(publish_times),
            'event_publish_p95_us': np.percentile(publish_times, 95),
            'events_published': len(publish_times),
            'events_received': len(received_events)
        }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("Benchmarking Memory Usage...")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024**2  # MB
        
        # Create components
        assembler_30m = MatrixAssembler30m("MemBench30m", self.kernel)
        assembler_5m = MatrixAssembler5m("MemBench5m", self.kernel)
        assembler_regime = MatrixAssemblerRegime("MemBenchRegime", self.kernel)
        indicator_engine = IndicatorEngine("MemBenchIndicators", self.kernel)
        
        component_memory = process.memory_info().rss / 1024**2 - initial_memory
        
        # Load test data
        test_features = {
            'current_price': 100.0,
            'current_volume': 1000,
            'mlmi_value': 65.0,
            'mlmi_signal': 1,
            'nwrqk_value': 102.0,
            'nwrqk_slope': 0.5,
            'lvn_distance_points': 5.0,
            'lvn_nearest_strength': 85.0,
            'fvg_bullish_active': True,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 99.5,
            'fvg_age': 5,
            'fvg_mitigation_signal': False,
            'mmd_features': [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1],
            'timestamp': datetime.now()
        }
        
        # Simulate sustained load
        memory_samples = []
        for i in range(10000):
            assembler_30m._update_matrix(test_features)
            assembler_5m._update_matrix(test_features)
            assembler_regime._update_matrix(test_features)
            
            if i % 1000 == 0:
                memory_samples.append(process.memory_info().rss / 1024**2)
        
        final_memory = process.memory_info().rss / 1024**2
        data_processing_memory = final_memory - initial_memory - component_memory
        
        return {
            'initial_memory_mb': initial_memory,
            'component_overhead_mb': component_memory,
            'data_processing_memory_mb': data_processing_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': final_memory - initial_memory,
            'memory_samples': memory_samples,
            'peak_memory_mb': max(memory_samples) if memory_samples else final_memory
        }
    
    def benchmark_concurrent_access(self) -> Dict[str, Any]:
        """Benchmark concurrent access performance."""
        print("Benchmarking Concurrent Access...")
        
        assembler = MatrixAssembler30m("ConcurrentBench", self.kernel)
        
        # Make assembler ready
        test_features = {
            'current_price': 100.0,
            'mlmi_value': 50.0,
            'mlmi_signal': 0,
            'nwrqk_value': 100.0,
            'nwrqk_slope': 0.0,
            'lvn_distance_points': 5.0,
            'lvn_nearest_strength': 70.0,
            'timestamp': datetime.now()
        }
        
        for _ in range(50):
            assembler._update_matrix(test_features)
        
        # Concurrent access test
        read_times = []
        write_times = []
        errors = []
        
        def reader_thread():
            """Continuously read matrix."""
            thread_read_times = []
            try:
                for _ in range(1000):
                    start = time.perf_counter()
                    matrix = assembler.get_matrix()
                    thread_read_times.append((time.perf_counter() - start) * 1000000)  # μs
                    time.sleep(0.0001)  # 0.1ms
            except Exception as e:
                errors.append(f"Reader error: {e}")
            read_times.extend(thread_read_times)
        
        def writer_thread():
            """Continuously update matrix."""
            thread_write_times = []
            try:
                for i in range(500):
                    start = time.perf_counter()
                    test_features['current_price'] = 100.0 + i * 0.01
                    assembler._update_matrix(test_features)
                    thread_write_times.append((time.perf_counter() - start) * 1000000)  # μs
                    time.sleep(0.0002)  # 0.2ms
            except Exception as e:
                errors.append(f"Writer error: {e}")
            write_times.extend(thread_write_times)
        
        # Start threads
        threads = []
        for _ in range(3):  # 3 readers
            threads.append(threading.Thread(target=reader_thread))
        for _ in range(2):  # 2 writers
            threads.append(threading.Thread(target=writer_thread))
        
        start_time = time.perf_counter()
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        total_time = time.perf_counter() - start_time
        
        return {
            'total_test_time_s': total_time,
            'read_operations': len(read_times),
            'write_operations': len(write_times),
            'read_time_avg_us': np.mean(read_times) if read_times else 0,
            'read_time_max_us': np.max(read_times) if read_times else 0,
            'write_time_avg_us': np.mean(write_times) if write_times else 0,
            'write_time_max_us': np.max(write_times) if write_times else 0,
            'errors': errors,
            'throughput_reads_per_sec': len(read_times) / total_time if total_time > 0 else 0,
            'throughput_writes_per_sec': len(write_times) / total_time if total_time > 0 else 0
        }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("=" * 80)
        print("    AlgoSpace Performance Benchmark Suite")
        print("=" * 80)
        
        start_time = time.perf_counter()
        
        # Run benchmarks
        self.results['benchmarks']['matrix_assemblers'] = self.benchmark_matrix_assemblers()
        self.results['benchmarks']['indicators'] = self.benchmark_indicators()
        self.results['benchmarks']['event_system'] = self.benchmark_event_system()
        self.results['benchmarks']['memory_usage'] = self.benchmark_memory_usage()
        self.results['benchmarks']['concurrent_access'] = self.benchmark_concurrent_access()
        
        total_time = time.perf_counter() - start_time
        self.results['benchmark_duration_s'] = total_time
        
        # Final resource usage
        process = psutil.Process(os.getpid())
        self.results['resource_usage'] = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024**2,
            'memory_percent': process.memory_percent(),
            'threads': process.num_threads()
        }
        
        return self.results
    
    def generate_report(self) -> None:
        """Generate and save performance report."""
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"performance_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📊 Performance Report saved to: {report_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print performance summary."""
        print("\n" + "=" * 80)
        print("                    PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # System info
        sys_info = self.results['system_info']
        print(f"\n🖥️  System Information:")
        print(f"   CPU Cores: {sys_info['cpu_count']}")
        print(f"   Memory: {sys_info['memory_total_gb']} GB total, {sys_info['memory_available_gb']} GB available")
        print(f"   Platform: {sys_info['platform']}")
        print(f"   Python: {sys_info['python_version']}")
        
        # Matrix assemblers
        matrix_bench = self.results['benchmarks']['matrix_assemblers']
        print(f"\n⚡ Matrix Assembler Performance:")
        print(f"   30m Update Avg: {matrix_bench['assembler_30m']['update_time_avg_ms']:.3f} ms")
        print(f"   5m Update Avg: {matrix_bench['assembler_5m']['update_time_avg_ms']:.3f} ms")
        print(f"   Regime Update Avg: {matrix_bench['assembler_regime']['update_time_avg_ms']:.3f} ms")
        print(f"   Matrix Access Avg: {matrix_bench['matrix_access']['access_time_avg_us']:.1f} μs")
        
        # Indicators
        indicator_bench = self.results['benchmarks']['indicators']
        print(f"\n📈 Indicator Performance:")
        print(f"   5m Calculation Avg: {indicator_bench['indicators_5m']['calculation_time_avg_ms']:.3f} ms")
        print(f"   30m Calculation Avg: {indicator_bench['indicators_30m']['calculation_time_avg_ms']:.3f} ms")
        
        # Event system
        event_bench = self.results['benchmarks']['event_system']
        print(f"\n🔄 Event System Performance:")
        print(f"   Event Publish Avg: {event_bench['event_publish_avg_us']:.1f} μs")
        print(f"   Events Published: {event_bench['events_published']:,}")
        print(f"   Events Received: {event_bench['events_received']:,}")
        
        # Memory usage
        memory_bench = self.results['benchmarks']['memory_usage']
        print(f"\n💾 Memory Usage:")
        print(f"   Initial: {memory_bench['initial_memory_mb']:.1f} MB")
        print(f"   Component Overhead: {memory_bench['component_overhead_mb']:.1f} MB")
        print(f"   Data Processing: {memory_bench['data_processing_memory_mb']:.1f} MB")
        print(f"   Peak Memory: {memory_bench['peak_memory_mb']:.1f} MB")
        print(f"   Total Growth: {memory_bench['memory_growth_mb']:.1f} MB")
        
        # Concurrent access
        concurrent_bench = self.results['benchmarks']['concurrent_access']
        print(f"\n🔄 Concurrent Access Performance:")
        print(f"   Read Throughput: {concurrent_bench['throughput_reads_per_sec']:.0f} ops/sec")
        print(f"   Write Throughput: {concurrent_bench['throughput_writes_per_sec']:.0f} ops/sec")
        print(f"   Read Latency Avg: {concurrent_bench['read_time_avg_us']:.1f} μs")
        print(f"   Write Latency Avg: {concurrent_bench['write_time_avg_us']:.1f} μs")
        
        # Overall assessment
        print(f"\n🎯 Performance Assessment:")
        
        # Check PRD requirements
        indicators_5m_time = indicator_bench['indicators_5m']['calculation_time_avg_ms']
        if indicators_5m_time <= 50:
            print(f"   ✅ 5m Indicators: {indicators_5m_time:.1f}ms ≤ 50ms PRD requirement")
        else:
            print(f"   ❌ 5m Indicators: {indicators_5m_time:.1f}ms > 50ms PRD requirement")
        
        matrix_update_time = max(
            matrix_bench['assembler_30m']['update_time_avg_ms'],
            matrix_bench['assembler_5m']['update_time_avg_ms'],
            matrix_bench['assembler_regime']['update_time_avg_ms']
        )
        if matrix_update_time <= 1.0:
            print(f"   ✅ Matrix Updates: {matrix_update_time:.3f}ms ≤ 1ms target")
        else:
            print(f"   ⚠️  Matrix Updates: {matrix_update_time:.3f}ms > 1ms target")
        
        matrix_access_time = matrix_bench['matrix_access']['access_time_avg_us']
        if matrix_access_time <= 100:
            print(f"   ✅ Matrix Access: {matrix_access_time:.1f}μs ≤ 100μs target")
        else:
            print(f"   ⚠️  Matrix Access: {matrix_access_time:.1f}μs > 100μs target")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.generate_report()