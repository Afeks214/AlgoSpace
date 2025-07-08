"""
Verification script for synergy detector deployment.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.synergy.detector import SynergyDetector
from src.agents.synergy.base import Signal
from src.core.kernel import AlgoSpaceKernel

logger = logging.getLogger(__name__)


def verify_synergy_patterns(detector: SynergyDetector) -> bool:
    """Verify all 4 synergy patterns are properly detected."""
    print("\n🔍 Verifying Synergy Pattern Detection...")
    
    patterns = {
        'TYPE_1': ['mlmi', 'nwrqk', 'fvg'],
        'TYPE_2': ['mlmi', 'fvg', 'nwrqk'],
        'TYPE_3': ['nwrqk', 'fvg', 'mlmi'],
        'TYPE_4': ['nwrqk', 'mlmi', 'fvg']
    }
    
    all_passed = True
    
    for pattern_type, signal_sequence in patterns.items():
        print(f"\n  Testing {pattern_type}: {' → '.join(signal_sequence)}")
        
        # Reset detector
        detector.sequence.reset()
        detector.cooldown.reset()
        
        # Build pattern
        base_time = datetime.now()
        for i, signal_type in enumerate(signal_sequence):
            signal = Signal(
                signal_type=signal_type,
                direction=1,
                timestamp=base_time + timedelta(minutes=i*3),
                value=70.0 if signal_type == 'mlmi' else 1.0,
                strength=0.8
            )
            detector.sequence.add_signal(signal)
        
        # Check detection
        synergy = detector._check_and_create_synergy()
        
        if synergy and synergy.synergy_type == pattern_type:
            print(f"    ✅ {pattern_type} detected correctly")
        else:
            print(f"    ❌ {pattern_type} detection FAILED!")
            all_passed = False
    
    return all_passed


def verify_performance(detector: SynergyDetector) -> bool:
    """Verify performance meets <1ms requirement."""
    print("\n⚡ Verifying Performance Requirements...")
    
    # Create test features
    features = {
        'timestamp': datetime.now(),
        'current_price': 5000.0,
        'mlmi_signal': 0,
        'mlmi_value': 50.0,
        'nwrqk_signal': 0,
        'nwrqk_slope': 0.0,
        'fvg_mitigation_signal': False,
        'volatility_30': 0.15,
        'volume_ratio': 1.0
    }
    
    # Warm up
    for _ in range(100):
        detector.process_features(features, features['timestamp'])
    
    # Measure
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        detector.process_features(features, features['timestamp'])
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    print(f"  Average processing time: {avg_time:.3f}ms")
    print(f"  Maximum processing time: {max_time:.3f}ms")
    
    if avg_time < 1.0:
        print("  ✅ Performance requirement met")
        return True
    else:
        print("  ❌ Performance requirement FAILED!")
        return False


def verify_cooldown_and_timewindow(detector: SynergyDetector) -> bool:
    """Verify cooldown and time window enforcement."""
    print("\n⏱️ Verifying Time Controls...")
    
    # Test time window
    print("\n  Testing 10-bar time window...")
    detector.sequence.reset()
    
    base_time = datetime.now()
    
    # Add first signal
    signal1 = Signal('mlmi', 1, base_time, 75.0, 0.8)
    added = detector.sequence.add_signal(signal1)
    assert added, "First signal should be added"
    
    # Try to add signal after time window
    late_signal = Signal('nwrqk', 1, base_time + timedelta(minutes=55), 1.0, 0.8)
    added = detector.sequence.add_signal(late_signal)
    
    if not added:
        print("    ✅ Time window properly enforced")
    else:
        print("    ❌ Time window enforcement FAILED!")
        return False
    
    # Test cooldown
    print("\n  Testing 5-bar cooldown...")
    detector.cooldown.start_cooldown(base_time)
    
    # Check cooldown status
    in_cooldown = detector.cooldown.is_in_cooldown()
    remaining = detector.cooldown.get_remaining_bars()
    
    if in_cooldown and remaining == 5:
        print("    ✅ Cooldown properly initialized")
    else:
        print("    ❌ Cooldown initialization FAILED!")
        return False
    
    # Update time and check again
    detector.cooldown.update(base_time + timedelta(minutes=30))
    in_cooldown = detector.cooldown.is_in_cooldown()
    
    if not in_cooldown:
        print("    ✅ Cooldown properly expired")
        return True
    else:
        print("    ❌ Cooldown expiration FAILED!")
        return False


def main():
    """Main verification script."""
    print("="*60)
    print("SynergyDetector Implementation Verification")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create mock kernel
    kernel = AlgoSpaceKernel()
    
    # Create detector
    detector = SynergyDetector('VerificationDetector', kernel)
    
    # Run verifications
    results = {
        'patterns': verify_synergy_patterns(detector),
        'performance': verify_performance(detector),
        'time_controls': verify_cooldown_and_timewindow(detector)
    }
    
    # Get status
    print("\n📊 Component Status:")
    status = detector.get_status()
    print(f"  Events processed: {status['performance_metrics']['events_processed']}")
    print(f"  Signals detected: {status['performance_metrics']['signals_detected']}")
    print(f"  Synergies detected: {status['performance_metrics']['synergies_detected']}")
    print(f"  Average processing time: {status['performance_metrics']['avg_processing_time_ms']:.3f}ms")
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.capitalize()}: {status}")
    
    if all_passed:
        print("\n✅ ALL VERIFICATIONS PASSED - SynergyDetector is production ready!")
    else:
        print("\n❌ SOME VERIFICATIONS FAILED - Please fix issues before deployment!")
        sys.exit(1)


if __name__ == '__main__':
    main()