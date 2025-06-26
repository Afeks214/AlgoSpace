#!/usr/bin/env python3
"""
Simple Performance Test for AlgoSpace Components
"""

import time
import sys
import psutil
import os
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.kernel import SystemKernel
from src.matrix import MatrixAssembler30m, MatrixAssembler5m, MatrixAssemblerRegime


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': round(psutil.virtual_memory().total / 1024**3, 2),
        'memory_available_gb': round(psutil.virtual_memory().available / 1024**3, 2),
        'platform': sys.platform,
        'python_version': sys.version.split()[0]
    }


def test_matrix_performance():
    """Test matrix assembler performance."""
    print("Testing Matrix Assembler Performance...")
    
    kernel = SystemKernel()
    
    # Create assemblers
    assembler_30m = MatrixAssembler30m("Perf30m", kernel)
    assembler_5m = MatrixAssembler5m("Perf5m", kernel)
    assembler_regime = MatrixAssemblerRegime("PerfRegime", kernel)
    
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
        # Skip regime assembler due to configuration issue
    
    results = {}
    
    # Test 30m assembler update performance
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        assembler_30m._update_matrix(test_features)
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    results['assembler_30m_update'] = {
        'avg_ms': np.mean(times),
        'max_ms': np.max(times),
        'min_ms': np.min(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99)
    }
    
    # Test 5m assembler update performance
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        assembler_5m._update_matrix(test_features)
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    results['assembler_5m_update'] = {
        'avg_ms': np.mean(times),
        'max_ms': np.max(times),
        'min_ms': np.min(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99)
    }
    
    # Test matrix access performance
    access_times = []
    for _ in range(10000):
        start = time.perf_counter()
        matrix = assembler_30m.get_matrix()
        access_times.append((time.perf_counter() - start) * 1000000)  # μs
    
    results['matrix_access'] = {
        'avg_us': np.mean(access_times),
        'max_us': np.max(access_times),
        'min_us': np.min(access_times),
        'p95_us': np.percentile(access_times, 95)
    }
    
    return results


def test_memory_usage():
    """Test memory usage patterns."""
    print("Testing Memory Usage...")
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024**2  # MB
    
    kernel = SystemKernel()
    
    # Create assemblers
    assembler_30m = MatrixAssembler30m("Mem30m", kernel)
    assembler_5m = MatrixAssembler5m("Mem5m", kernel)
    
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
        
        if i % 1000 == 0:
            memory_samples.append(process.memory_info().rss / 1024**2)
    
    final_memory = process.memory_info().rss / 1024**2
    
    return {
        'initial_memory_mb': initial_memory,
        'component_overhead_mb': component_memory,
        'final_memory_mb': final_memory,
        'memory_growth_mb': final_memory - initial_memory,
        'memory_samples': memory_samples,
        'peak_memory_mb': max(memory_samples) if memory_samples else final_memory
    }


def test_data_throughput():
    """Test data processing throughput."""
    print("Testing Data Throughput...")
    
    kernel = SystemKernel()
    assembler = MatrixAssembler30m("Throughput", kernel)
    
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
    
    # Warm up
    for _ in range(100):
        assembler._update_matrix(test_features)
    
    # Test sustained throughput
    start_time = time.perf_counter()
    update_count = 10000
    
    for i in range(update_count):
        test_features['current_price'] = 100.0 + (i % 100) * 0.01
        assembler._update_matrix(test_features)
    
    total_time = time.perf_counter() - start_time
    throughput = update_count / total_time
    
    return {
        'updates_processed': update_count,
        'total_time_s': total_time,
        'throughput_updates_per_sec': throughput,
        'avg_update_time_us': (total_time / update_count) * 1000000
    }


def analyze_validation_reports():
    """Analyze existing validation reports for trends."""
    print("Analyzing Validation Reports...")
    
    reports = []
    for report_file in Path('.').glob('validation_report_*.json'):
        try:
            with open(report_file) as f:
                report = json.load(f)
                reports.append(report)
        except Exception as e:
            print(f"Error reading {report_file}: {e}")
    
    if not reports:
        return {'error': 'No validation reports found'}
    
    # Sort by timestamp
    reports.sort(key=lambda r: r.get('timestamp', ''))
    
    analysis = {
        'report_count': len(reports),
        'latest_report': reports[-1]['timestamp'] if reports else None,
        'system_growth': {}
    }
    
    if len(reports) >= 2:
        first = reports[0]
        latest = reports[-1]
        
        # Track code growth
        first_lines = sum(
            comp.get('lines', 0) 
            for comp in first.get('component_stats', {}).values()
        )
        latest_lines = sum(
            comp.get('lines', 0) 
            for comp in latest.get('component_stats', {}).values()
        )
        
        analysis['system_growth'] = {
            'lines_of_code': {
                'first': first_lines,
                'latest': latest_lines,
                'growth': latest_lines - first_lines,
                'growth_pct': ((latest_lines - first_lines) / first_lines * 100) if first_lines > 0 else 0
            },
            'total_functions': {
                'first': first.get('summary', {}).get('total_functions', 0),
                'latest': latest.get('summary', {}).get('total_functions', 0)
            },
            'total_classes': {
                'first': first.get('summary', {}).get('total_classes', 0),
                'latest': latest.get('summary', {}).get('total_classes', 0)
            }
        }
    
    return analysis


def main():
    """Run all performance tests."""
    print("=" * 80)
    print("    AlgoSpace Performance Analysis")
    print("=" * 80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': get_system_info(),
        'performance_tests': {},
        'validation_analysis': {}
    }
    
    # Run tests
    try:
        results['performance_tests']['matrix_performance'] = test_matrix_performance()
        results['performance_tests']['memory_usage'] = test_memory_usage()
        results['performance_tests']['data_throughput'] = test_data_throughput()
        results['validation_analysis'] = analyze_validation_reports()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"performance_analysis_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Performance Analysis saved to: {report_file}")
        
        # Print summary
        print_performance_summary(results)
        
    except Exception as e:
        print(f"Error during performance testing: {e}")
        import traceback
        traceback.print_exc()


def print_performance_summary(results: Dict[str, Any]):
    """Print performance summary."""
    print("\n" + "=" * 80)
    print("                    PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # System info
    sys_info = results['system_info']
    print(f"\n🖥️  System Information:")
    print(f"   CPU Cores: {sys_info['cpu_count']}")
    print(f"   Memory: {sys_info['memory_total_gb']} GB total, {sys_info['memory_available_gb']} GB available")
    print(f"   Platform: {sys_info['platform']}")
    print(f"   Python: {sys_info['python_version']}")
    
    # Matrix performance
    matrix_perf = results['performance_tests']['matrix_performance']
    print(f"\n⚡ Matrix Assembler Performance:")
    print(f"   30m Update Average: {matrix_perf['assembler_30m_update']['avg_ms']:.3f} ms")
    print(f"   30m Update P95: {matrix_perf['assembler_30m_update']['p95_ms']:.3f} ms")
    print(f"   5m Update Average: {matrix_perf['assembler_5m_update']['avg_ms']:.3f} ms")
    print(f"   5m Update P95: {matrix_perf['assembler_5m_update']['p95_ms']:.3f} ms")
    print(f"   Matrix Access Average: {matrix_perf['matrix_access']['avg_us']:.1f} μs")
    print(f"   Matrix Access P95: {matrix_perf['matrix_access']['p95_us']:.1f} μs")
    
    # Memory usage
    memory_usage = results['performance_tests']['memory_usage']
    print(f"\n💾 Memory Usage:")
    print(f"   Component Overhead: {memory_usage['component_overhead_mb']:.1f} MB")
    print(f"   Peak Memory: {memory_usage['peak_memory_mb']:.1f} MB")
    print(f"   Memory Growth: {memory_usage['memory_growth_mb']:.1f} MB")
    
    # Throughput
    throughput = results['performance_tests']['data_throughput']
    print(f"\n📊 Data Processing Throughput:")
    print(f"   Updates per Second: {throughput['throughput_updates_per_sec']:.0f}")
    print(f"   Average Update Time: {throughput['avg_update_time_us']:.1f} μs")
    
    # Validation analysis
    validation = results['validation_analysis']
    if 'system_growth' in validation:
        growth = validation['system_growth']
        print(f"\n📈 System Evolution:")
        print(f"   Lines of Code Growth: {growth['lines_of_code']['growth']:+,} ({growth['lines_of_code']['growth_pct']:+.1f}%)")
        print(f"   Current LOC: {growth['lines_of_code']['latest']:,}")
        print(f"   Function Count: {growth['total_functions']['latest']}")
        print(f"   Class Count: {growth['total_classes']['latest']}")
    
    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    
    # Check critical timings
    avg_30m_time = matrix_perf['assembler_30m_update']['avg_ms']
    avg_5m_time = matrix_perf['assembler_5m_update']['avg_ms']
    avg_access_time = matrix_perf['matrix_access']['avg_us']
    
    if avg_30m_time <= 1.0:
        print(f"   ✅ 30m Matrix Updates: {avg_30m_time:.3f}ms ≤ 1ms target")
    else:
        print(f"   ⚠️  30m Matrix Updates: {avg_30m_time:.3f}ms > 1ms target")
    
    if avg_5m_time <= 1.0:
        print(f"   ✅ 5m Matrix Updates: {avg_5m_time:.3f}ms ≤ 1ms target")
    else:
        print(f"   ⚠️  5m Matrix Updates: {avg_5m_time:.3f}ms > 1ms target")
    
    if avg_access_time <= 100:
        print(f"   ✅ Matrix Access: {avg_access_time:.1f}μs ≤ 100μs target")
    else:
        print(f"   ⚠️  Matrix Access: {avg_access_time:.1f}μs > 100μs target")
    
    throughput_rate = throughput['throughput_updates_per_sec']
    if throughput_rate >= 1000:
        print(f"   ✅ Throughput: {throughput_rate:.0f} updates/sec ≥ 1000 target")
    else:
        print(f"   ⚠️  Throughput: {throughput_rate:.0f} updates/sec < 1000 target")
    
    peak_memory = memory_usage['peak_memory_mb']
    if peak_memory <= 500:
        print(f"   ✅ Memory Usage: {peak_memory:.1f}MB ≤ 500MB target")
    else:
        print(f"   ⚠️  Memory Usage: {peak_memory:.1f}MB > 500MB target")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()