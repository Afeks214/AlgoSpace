"""
Performance benchmark for SynergyDetector.

This script validates that the SynergyDetector meets the <1ms 
processing time requirement per INDICATORS_READY event.
"""

import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

from src.agents.synergy import SynergyDetector
from src.core.events import EventType, Event


class MockKernel:
    """Mock kernel for benchmarking."""
    
    def __init__(self):
        self.config = {
            'synergy_detector': {
                'time_window': 10,
                'mlmi_threshold': 0.5,
                'nwrqk_threshold': 0.3,
                'fvg_min_size': 0.001,
                'cooldown_bars': 5
            }
        }
        self.event_bus = MockEventBus()


class MockEventBus:
    """Mock event bus for benchmarking."""
    
    def subscribe(self, event_type, callback):
        pass
    
    def publish(self, event):
        pass
    
    def create_event(self, event_type, payload, source):
        return Event(event_type, datetime.now(), payload, source)


def generate_test_features(timestamp: datetime, pattern_step: int = 0) -> Dict[str, Any]:
    """
    Generate test feature data.
    
    Args:
        timestamp: Current timestamp
        pattern_step: Which step of a pattern to generate (0=none, 1=mlmi, 2=nwrqk, 3=fvg)
    """
    base_features = {
        'timestamp': timestamp,
        'current_price': 5150.0 + random.uniform(-10, 10),
        'volatility_30': 12.5 + random.uniform(-2, 2),
        'volume_ratio': 1.0 + random.uniform(-0.3, 0.3),
        'volume_momentum_30': random.uniform(-0.5, 0.5),
        'lvn_nearest_price': 5145.0,
        'lvn_nearest_strength': 85.0 + random.uniform(-10, 10),
        'lvn_distance_points': 5.0 + random.uniform(-2, 2),
        
        # Default: no signals
        'mlmi_signal': 0,
        'mlmi_value': 50 + random.uniform(-10, 10),
        'nwrqk_signal': 0,
        'nwrqk_slope': random.uniform(-0.2, 0.2),
        'nwrqk_value': 100.0 + random.uniform(-5, 5),
        'fvg_mitigation_signal': False,
        'fvg_bullish_mitigated': False,
        'fvg_bearish_mitigated': False
    }
    
    # Add pattern signals based on step
    if pattern_step == 1:  # MLMI signal
        base_features['mlmi_signal'] = 1
        base_features['mlmi_value'] = 75.0  # Strong bullish
    elif pattern_step == 2:  # NW-RQK signal
        base_features['nwrqk_signal'] = 1
        base_features['nwrqk_slope'] = 0.5  # Strong slope
    elif pattern_step == 3:  # FVG mitigation
        base_features['fvg_mitigation_signal'] = True
        base_features['fvg_bullish_mitigated'] = True
        base_features['fvg_bullish_size'] = 10.0
        base_features['fvg_bullish_level'] = base_features['current_price'] - 5
    
    return base_features


def benchmark_single_event_processing(detector: SynergyDetector, iterations: int = 10000) -> List[float]:
    """
    Benchmark processing of single events without patterns.
    
    Returns:
        List of processing times in milliseconds
    """
    print(f"\nBenchmarking single event processing ({iterations} iterations)...")
    
    processing_times = []
    base_time = datetime.now()
    
    for i in range(iterations):
        # Generate features with no signals
        features = generate_test_features(base_time + timedelta(seconds=i*5))
        
        # Measure processing time
        start = time.perf_counter()
        detector.process_features(features, features['timestamp'])
        end = time.perf_counter()
        
        processing_times.append((end - start) * 1000)  # Convert to ms
    
    return processing_times


def benchmark_pattern_detection(detector: SynergyDetector, iterations: int = 1000) -> List[float]:
    """
    Benchmark processing when detecting patterns.
    
    Returns:
        List of processing times in milliseconds
    """
    print(f"\nBenchmarking pattern detection ({iterations} iterations)...")
    
    processing_times = []
    base_time = datetime.now()
    
    for i in range(iterations):
        # Generate a complete TYPE_1 pattern sequence
        pattern_times = []
        
        # Step 1: MLMI signal
        features1 = generate_test_features(base_time + timedelta(minutes=i*20), pattern_step=1)
        start = time.perf_counter()
        detector.process_features(features1, features1['timestamp'])
        pattern_times.append((time.perf_counter() - start) * 1000)
        
        # Step 2: NW-RQK signal
        features2 = generate_test_features(base_time + timedelta(minutes=i*20+5), pattern_step=2)
        start = time.perf_counter()
        detector.process_features(features2, features2['timestamp'])
        pattern_times.append((time.perf_counter() - start) * 1000)
        
        # Step 3: FVG mitigation (completes pattern)
        features3 = generate_test_features(base_time + timedelta(minutes=i*20+10), pattern_step=3)
        start = time.perf_counter()
        result = detector.process_features(features3, features3['timestamp'])
        pattern_times.append((time.perf_counter() - start) * 1000)
        
        processing_times.extend(pattern_times)
        
        # Reset for next iteration
        detector.sequence.reset()
        detector.cooldown.last_synergy_time = None
    
    return processing_times


def benchmark_worst_case(detector: SynergyDetector, iterations: int = 1000) -> List[float]:
    """
    Benchmark worst-case scenarios (all signals active).
    
    Returns:
        List of processing times in milliseconds
    """
    print(f"\nBenchmarking worst-case scenarios ({iterations} iterations)...")
    
    processing_times = []
    base_time = datetime.now()
    
    for i in range(iterations):
        # Generate features with all signals active
        features = generate_test_features(base_time + timedelta(seconds=i*5))
        features['mlmi_signal'] = 1
        features['mlmi_value'] = 75.0
        features['nwrqk_signal'] = 1
        features['nwrqk_slope'] = 0.5
        features['fvg_mitigation_signal'] = True
        features['fvg_bullish_mitigated'] = True
        features['fvg_bullish_size'] = 10.0
        
        # Measure processing time
        start = time.perf_counter()
        detector.process_features(features, features['timestamp'])
        end = time.perf_counter()
        
        processing_times.append((end - start) * 1000)
        
        # Reset sequence to avoid cooldowns
        detector.sequence.reset()
    
    return processing_times


def print_statistics(times: List[float], scenario: str):
    """Print statistics for processing times."""
    if not times:
        return
        
    print(f"\n{scenario} Statistics:")
    print(f"  Samples: {len(times)}")
    print(f"  Mean: {statistics.mean(times):.3f} ms")
    print(f"  Median: {statistics.median(times):.3f} ms")
    print(f"  Std Dev: {statistics.stdev(times):.3f} ms")
    print(f"  Min: {min(times):.3f} ms")
    print(f"  Max: {max(times):.3f} ms")
    
    # Calculate percentiles
    sorted_times = sorted(times)
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    
    print(f"  95th percentile: {p95:.3f} ms")
    print(f"  99th percentile: {p99:.3f} ms")
    
    # Check against 1ms requirement
    under_1ms = sum(1 for t in times if t < 1.0)
    percentage = (under_1ms / len(times)) * 100
    
    print(f"\nPerformance vs 1ms requirement:")
    print(f"  Under 1ms: {under_1ms}/{len(times)} ({percentage:.1f}%)")
    
    if percentage >= 99:
        print("  ✅ PASSES <1ms requirement (99%+ under 1ms)")
    else:
        print("  ❌ FAILS <1ms requirement")


def main():
    """Run the benchmark suite."""
    print("="*60)
    print("SynergyDetector Performance Benchmark")
    print("="*60)
    
    # Create detector
    kernel = MockKernel()
    detector = SynergyDetector('SynergyDetector', kernel)
    
    # Warm up
    print("\nWarming up...")
    for _ in range(100):
        features = generate_test_features(datetime.now())
        detector.process_features(features, features['timestamp'])
    
    # Run benchmarks
    single_times = benchmark_single_event_processing(detector, iterations=10000)
    pattern_times = benchmark_pattern_detection(detector, iterations=1000)
    worst_case_times = benchmark_worst_case(detector, iterations=1000)
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print_statistics(single_times, "Single Event Processing")
    print_statistics(pattern_times, "Pattern Detection")
    print_statistics(worst_case_times, "Worst Case (All Signals)")
    
    # Overall summary
    all_times = single_times + pattern_times + worst_case_times
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print_statistics(all_times, "All Scenarios Combined")
    
    # Component-level metrics
    print("\n" + "="*60)
    print("COMPONENT METRICS")
    print("="*60)
    
    status = detector.get_status()
    perf_metrics = status['performance_metrics']
    
    print(f"\nSynergyDetector Metrics:")
    print(f"  Events processed: {perf_metrics['events_processed']}")
    print(f"  Signals detected: {perf_metrics['signals_detected']}")
    print(f"  Synergies detected: {perf_metrics['synergies_detected']}")
    print(f"  Avg processing time: {perf_metrics['avg_processing_time_ms']:.3f} ms")
    print(f"  Max processing time: {perf_metrics['max_processing_time_ms']:.3f} ms")
    
    print("\nPattern Detector Metrics:")
    for name, metrics in status['pattern_detectors'].items():
        print(f"  {name.upper()}:")
        print(f"    Total detections: {metrics['total_detections']}")
        print(f"    Avg time: {metrics['avg_detection_time_ms']:.3f} ms")
        print(f"    Max time: {metrics['max_detection_time_ms']:.3f} ms")


if __name__ == '__main__':
    main()