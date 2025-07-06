#!/usr/bin/env python3
"""
TERMINAL 3: Quick Validation and Testing Framework
Real-time verification of critical fixes for AlgoSpace production readiness
"""

import time
import torch
import numpy as np
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.append('/home/QuantNova/AlgoSpace-4')

class Terminal3Validator:
    """Real-time testing and validation for Terminal 3"""
    
    def __init__(self):
        self.results = {
            'rde_performance': {'target': 5.0, 'current': None, 'status': 'UNKNOWN'},
            'mrms_format': {'target': 'PASS', 'current': None, 'status': 'UNKNOWN'},
            'fvg_features': {'target': 9, 'current': None, 'status': 'UNKNOWN'},
            'overall_score': {'target': 95.0, 'current': 73.3, 'status': 'NEEDS_FIX'}
        }
        self.test_count = 0
        
    def print_header(self):
        """Print Terminal 3 header"""
        print("=" * 80)
        print("🔬 TERMINAL 3: TESTING & BENCHMARKING STATION")
        print("=" * 80)
        print(f"🕐 Session Start: {datetime.now().strftime('%H:%M:%S')}")
        print("📊 Real-time validation of critical fixes")
        print("🎯 Target: 73.3/100 → 95/100 production score")
        print("=" * 80)
    
    def quick_rde_test(self):
        """Test RDE performance after optimization"""
        print("\n🧠 TESTING RDE PERFORMANCE...")
        
        try:
            from src.agents.rde.engine import RDEComponent
            
            # Create RDE with production config
            rde_config = {
                'input_dim': 155,
                'd_model': 256,
                'latent_dim': 8,
                'n_heads': 8,
                'n_layers': 6,
                'sequence_length': 96
            }
            
            rde = RDEComponent(rde_config)
            input_data = np.random.randn(96, 155).astype(np.float32)
            
            # Warm up (5 runs)
            print("   🔥 Warming up RDE...")
            for _ in range(5):
                _ = rde.get_regime_vector(input_data)
            
            # Performance test (100 runs)
            print("   ⚡ Running performance test (100 iterations)...")
            times = []
            
            for i in range(100):
                start = time.perf_counter()
                regime_vector = rde.get_regime_vector(input_data)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                
                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"      Progress: {i+1}/100 ({(i+1):.0%})")
            
            # Calculate metrics
            avg_time = np.mean(times)
            p95_time = np.percentile(times, 95)
            min_time = np.min(times)
            max_time = np.max(times)
            
            # Update results
            self.results['rde_performance']['current'] = avg_time
            self.results['rde_performance']['status'] = 'PASS' if avg_time < 5.0 else 'FAIL'
            
            # Report
            print(f"   ✅ Average Time: {avg_time:.3f}ms (Target: <5ms)")
            print(f"   📊 P95 Time: {p95_time:.3f}ms")
            print(f"   🏃 Min Time: {min_time:.3f}ms")
            print(f"   🐌 Max Time: {max_time:.3f}ms")
            print(f"   🎯 Status: {'✅ PASS' if avg_time < 5.0 else '❌ FAIL'}")
            print(f"   🔢 Output Shape: {regime_vector.shape}")
            
            return avg_time < 5.0
            
        except Exception as e:
            print(f"   ❌ RDE Test Failed: {e}")
            self.results['rde_performance']['status'] = 'ERROR'
            return False
    
    def quick_mrms_test(self):
        """Test M-RMS output format consistency"""
        print("\n💰 TESTING M-RMS OUTPUT FORMAT...")
        
        try:
            from src.agents.mrms.engine import MRMSComponent
            
            # Create M-RMS
            mrms_config = {
                'synergy_dim': 30,
                'account_dim': 10,
                'hidden_dim': 128,
                'num_agents': 3
            }
            
            mrms = MRMSComponent(mrms_config)
            
            # Test with mock data
            synergy_vector = torch.randn(1, 30)
            account_state = torch.randn(1, 10)
            
            print("   🔄 Testing output format...")
            
            # Get risk proposal
            start = time.perf_counter()
            risk_proposal = mrms.model(synergy_vector, account_state)
            inference_time = (time.perf_counter() - start) * 1000
            
            # Check output format
            required_fields = ['position_size', 'stop_loss', 'take_profit', 'risk_amount', 'confidence']
            
            if isinstance(risk_proposal, dict):
                missing_fields = [f for f in required_fields if f not in risk_proposal]
                format_ok = len(missing_fields) == 0
                
                self.results['mrms_format']['current'] = 'DICT_FORMAT'
                self.results['mrms_format']['status'] = 'PASS' if format_ok else 'MISSING_FIELDS'
                
                print(f"   ✅ Output Type: Dictionary")
                print(f"   📋 Fields Present: {list(risk_proposal.keys())}")
                print(f"   ❓ Missing Fields: {missing_fields if missing_fields else 'None'}")
                
            elif hasattr(risk_proposal, 'shape'):
                # Tensor output
                self.results['mrms_format']['current'] = f'TENSOR_{list(risk_proposal.shape)}'
                self.results['mrms_format']['status'] = 'TENSOR_FORMAT'
                
                print(f"   📊 Output Type: Tensor")
                print(f"   🔢 Shape: {risk_proposal.shape}")
                
            else:
                self.results['mrms_format']['current'] = type(risk_proposal).__name__
                self.results['mrms_format']['status'] = 'UNKNOWN_FORMAT'
                
                print(f"   ❓ Output Type: {type(risk_proposal)}")
            
            print(f"   ⚡ Inference Time: {inference_time:.3f}ms (Target: <10ms)")
            print(f"   🎯 Status: {'✅ PASS' if inference_time < 10.0 else '❌ FAIL'}")
            
            return inference_time < 10.0
            
        except Exception as e:
            print(f"   ❌ M-RMS Test Failed: {e}")
            self.results['mrms_format']['status'] = 'ERROR'
            return False
    
    def quick_fvg_test(self):
        """Test Enhanced FVG implementation"""
        print("\n📈 TESTING ENHANCED FVG...")
        
        try:
            # Try to import Enhanced FVG
            try:
                from src.indicators.fvg import EnhancedFVGDetector
                fvg_available = True
            except ImportError as e:
                print(f"   ❌ Import Failed: {e}")
                self.results['fvg_features']['status'] = 'NOT_IMPLEMENTED'
                return False
            
            # Test FVG features
            detector = EnhancedFVGDetector({})
            
            # Create test bar data
            from src.core.events import BarData
            from datetime import datetime
            
            test_bar = BarData(
                timestamp=datetime.now(),
                open=5000.0,
                high=5020.0,
                low=4980.0,
                close=5010.0,
                volume=1500
            )
            
            print("   🔍 Testing feature calculation...")
            features = detector.calculate_5m(test_bar)
            
            # Check for required 9 features
            required_features = [
                'fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
                'fvg_age', 'fvg_mitigation_signal', 'fvg_gap_size',
                'fvg_gap_size_pct', 'fvg_mitigation_strength', 'fvg_mitigation_depth'
            ]
            
            present_features = [f for f in required_features if f in features]
            missing_features = [f for f in required_features if f not in features]
            
            self.results['fvg_features']['current'] = len(present_features)
            self.results['fvg_features']['status'] = 'PASS' if len(missing_features) == 0 else 'INCOMPLETE'
            
            print(f"   ✅ Features Found: {len(present_features)}/9")
            print(f"   📋 Present: {present_features}")
            print(f"   ❓ Missing: {missing_features if missing_features else 'None'}")
            print(f"   🎯 Status: {'✅ COMPLETE' if len(missing_features) == 0 else '❌ INCOMPLETE'}")
            
            return len(missing_features) == 0
            
        except Exception as e:
            print(f"   ❌ FVG Test Failed: {e}")
            self.results['fvg_features']['status'] = 'ERROR'
            return False
    
    def benchmark_full_system(self):
        """Run full system benchmark"""
        print("\n🚀 FULL SYSTEM BENCHMARK...")
        
        try:
            # Run the comprehensive test
            import subprocess
            result = subprocess.run([
                'python', 'test_pytorch_complete.py'
            ], capture_output=True, text=True, cwd='/home/QuantNova/AlgoSpace-4')
            
            if result.returncode == 0:
                # Parse output for score
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'TRUE PRODUCTION READINESS SCORE:' in line:
                        score_str = line.split(':')[1].strip().split('/')[0]
                        try:
                            score = float(score_str)
                            self.results['overall_score']['current'] = score
                            self.results['overall_score']['status'] = 'PASS' if score >= 95.0 else 'IMPROVING'
                            print(f"   🎯 Current Score: {score:.1f}/100")
                            break
                        except ValueError:
                            pass
                
                print("   ✅ Full system benchmark completed")
                return True
            else:
                print(f"   ❌ Benchmark failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Benchmark error: {e}")
            return False
    
    def print_status_summary(self):
        """Print current status summary"""
        print("\n" + "=" * 60)
        print("📊 TERMINAL 3 STATUS SUMMARY")
        print("=" * 60)
        
        for component, data in self.results.items():
            status_icon = {
                'PASS': '✅',
                'FAIL': '❌', 
                'ERROR': '💥',
                'UNKNOWN': '❓',
                'IMPROVING': '🔄',
                'NEEDS_FIX': '⚠️',
                'NOT_IMPLEMENTED': '🚧',
                'INCOMPLETE': '⚠️'
            }.get(data['status'], '❓')
            
            print(f"{status_icon} {component.upper()}: {data['current']} (Target: {data['target']})")
        
        print("=" * 60)
        self.test_count += 1
        print(f"🔬 Test Cycle: #{self.test_count}")
        print(f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}")
    
    def run_continuous_validation(self):
        """Main validation loop for Terminal 3"""
        self.print_header()
        
        while True:
            try:
                print(f"\n🔄 VALIDATION CYCLE #{self.test_count + 1}")
                print("-" * 50)
                
                # Run all quick tests
                self.quick_rde_test()
                self.quick_mrms_test() 
                self.quick_fvg_test()
                
                # Print summary
                self.print_status_summary()
                
                # Ask for next action
                print("\n🎮 TERMINAL 3 OPTIONS:")
                print("1. Run another cycle (Enter)")
                print("2. Full benchmark (b)")
                print("3. RDE only (r)")
                print("4. M-RMS only (m)")
                print("5. FVG only (f)")
                print("6. Exit (q)")
                
                choice = input("\nChoice: ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == 'b':
                    self.benchmark_full_system()
                elif choice == 'r':
                    self.quick_rde_test()
                elif choice == 'm':
                    self.quick_mrms_test()
                elif choice == 'f':
                    self.quick_fvg_test()
                # Default: continue with full cycle
                
            except KeyboardInterrupt:
                print("\n\n🛑 Terminal 3 stopped by user")
                break
            except Exception as e:
                print(f"\n💥 Validation error: {e}")
                time.sleep(2)
        
        print("\n✅ Terminal 3 session completed")


def main():
    """Terminal 3 main entry point"""
    validator = Terminal3Validator()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        validator.run_continuous_validation()
    else:
        # Single run mode
        validator.print_header()
        validator.quick_rde_test()
        validator.quick_mrms_test()
        validator.quick_fvg_test()
        validator.print_status_summary()


if __name__ == '__main__':
    main()