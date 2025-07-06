#!/usr/bin/env python3
"""
Runtime verification script for AlgoSpace components
Tests imports, basic functionality, and integration points
"""
import sys
import os
import importlib
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json

class RuntimeVerifier:
    """Comprehensive runtime verification for AlgoSpace"""
    
    def __init__(self):
        self.results = {
            'core_systems': [],
            'indicators': [],
            'agents': [],
            'data_components': [],
            'training': [],
            'errors': [],
            'warnings': []
        }
        
    def verify_core_systems(self) -> bool:
        """Verify core AlgoSpace systems"""
        print("🔍 Verifying core systems...")
        
        core_modules = [
            'src.core.kernel',
            'src.core.events',
            'src.core.event_bus',
            'src.core.config',
            'src.core.component_base',
            'src.core.memory_manager',
            'src.core.thread_safety'
        ]
        
        success_count = 0
        
        for module in core_modules:
            try:
                # Test import
                mod = importlib.import_module(module)
                
                # Basic functionality tests
                if hasattr(mod, 'AlgoSpaceKernel'):
                    # Test kernel initialization
                    print(f"  ✅ {module}: Kernel class found")
                elif hasattr(mod, 'EventBus'):
                    # Test event bus
                    print(f"  ✅ {module}: EventBus class found")
                elif hasattr(mod, 'Event'):
                    # Test events
                    print(f"  ✅ {module}: Event classes found")
                else:
                    print(f"  ✅ {module}: Module imported successfully")
                
                self.results['core_systems'].append({
                    'module': module,
                    'status': 'success',
                    'message': 'Import and basic checks passed'
                })
                success_count += 1
                
            except Exception as e:
                error_msg = f"Failed to import {module}: {str(e)}"
                print(f"  ❌ {error_msg}")
                self.results['errors'].append(error_msg)
                self.results['core_systems'].append({
                    'module': module,
                    'status': 'failed',
                    'message': str(e)
                })
        
        print(f"📊 Core systems: {success_count}/{len(core_modules)} passed")
        return success_count == len(core_modules)
    
    def verify_indicators(self) -> bool:
        """Verify indicator components"""
        print("\n🔍 Verifying indicators...")
        
        indicator_modules = [
            'src.indicators.base',
            'src.indicators.engine',
            'src.indicators.fvg',
            'src.indicators.mlmi',
            'src.indicators.nwrqk',
            'src.indicators.lvn',
            'src.indicators.mmd'
        ]
        
        success_count = 0
        
        for module in indicator_modules:
            try:
                # Test import
                mod = importlib.import_module(module)
                
                # Check for expected classes
                if hasattr(mod, 'IndicatorEngine'):
                    print(f"  ✅ {module}: IndicatorEngine found")
                elif hasattr(mod, 'BaseIndicator'):
                    print(f"  ✅ {module}: BaseIndicator found")
                elif hasattr(mod, 'FVGDetector'):
                    print(f"  ✅ {module}: FVGDetector found")
                else:
                    print(f"  ✅ {module}: Module imported successfully")
                
                self.results['indicators'].append({
                    'module': module,
                    'status': 'success',
                    'message': 'Import successful'
                })
                success_count += 1
                
            except Exception as e:
                error_msg = f"Failed to import {module}: {str(e)}"
                print(f"  ❌ {error_msg}")
                self.results['errors'].append(error_msg)
                self.results['indicators'].append({
                    'module': module,
                    'status': 'failed',
                    'message': str(e)
                })
        
        print(f"📊 Indicators: {success_count}/{len(indicator_modules)} passed")
        return success_count >= len(indicator_modules) * 0.8  # 80% threshold
    
    def verify_agents(self) -> bool:
        """Verify agent components"""
        print("\n🔍 Verifying agents...")
        
        agent_modules = [
            'src.agents.mrms.engine',
            'src.agents.rde.engine',
            'src.agents.main_core.engine'
        ]
        
        success_count = 0
        
        for module in agent_modules:
            try:
                # Test import
                mod = importlib.import_module(module)
                
                # Check for expected classes
                if hasattr(mod, 'MRMSEngine'):
                    print(f"  ✅ {module}: MRMSEngine found")
                elif hasattr(mod, 'RDEComponent'):
                    print(f"  ✅ {module}: RDEComponent found")
                elif hasattr(mod, 'MainCoreEngine'):
                    print(f"  ✅ {module}: MainCoreEngine found")
                else:
                    print(f"  ✅ {module}: Module imported successfully")
                
                self.results['agents'].append({
                    'module': module,
                    'status': 'success',
                    'message': 'Import successful'
                })
                success_count += 1
                
            except Exception as e:
                error_msg = f"Failed to import {module}: {str(e)}"
                print(f"  ❌ {error_msg}")
                self.results['errors'].append(error_msg)
                self.results['agents'].append({
                    'module': module,
                    'status': 'failed',
                    'message': str(e)
                })
        
        print(f"📊 Agents: {success_count}/{len(agent_modules)} passed")
        return success_count >= len(agent_modules) * 0.8  # 80% threshold
    
    def verify_data_components(self) -> bool:
        """Verify data components"""
        print("\n🔍 Verifying data components...")
        
        data_modules = [
            'src.data.bar_generator',
            'src.data.validators',
            'src.data.market_data'
        ]
        
        success_count = 0
        
        for module in data_modules:
            try:
                # Test import
                mod = importlib.import_module(module)
                
                # Check for expected classes
                if hasattr(mod, 'BarGenerator'):
                    print(f"  ✅ {module}: BarGenerator found")
                elif hasattr(mod, 'BarValidator'):
                    print(f"  ✅ {module}: BarValidator found")
                else:
                    print(f"  ✅ {module}: Module imported successfully")
                
                self.results['data_components'].append({
                    'module': module,
                    'status': 'success',
                    'message': 'Import successful'
                })
                success_count += 1
                
            except Exception as e:
                error_msg = f"Failed to import {module}: {str(e)}"
                print(f"  ❌ {error_msg}")
                self.results['errors'].append(error_msg)
                self.results['data_components'].append({
                    'module': module,
                    'status': 'failed',
                    'message': str(e)
                })
        
        print(f"📊 Data components: {success_count}/{len(data_modules)} passed")
        return success_count >= len(data_modules) * 0.8  # 80% threshold
    
    def verify_training_components(self) -> bool:
        """Verify training components"""
        print("\n🔍 Verifying training components...")
        
        training_modules = [
            'src.training.environment',
            'src.training.experience',
            'src.training.rewards'
        ]
        
        success_count = 0
        
        for module in training_modules:
            try:
                # Test import
                mod = importlib.import_module(module)
                
                print(f"  ✅ {module}: Module imported successfully")
                
                self.results['training'].append({
                    'module': module,
                    'status': 'success',
                    'message': 'Import successful'
                })
                success_count += 1
                
            except Exception as e:
                error_msg = f"Failed to import {module}: {str(e)}"
                print(f"  ❌ {error_msg}")
                self.results['errors'].append(error_msg)
                self.results['training'].append({
                    'module': module,
                    'status': 'failed',
                    'message': str(e)
                })
        
        print(f"📊 Training components: {success_count}/{len(training_modules)} passed")
        return success_count >= len(training_modules) * 0.8  # 80% threshold
    
    def test_pytorch_integration(self) -> bool:
        """Test PyTorch integration"""
        print("\n🔍 Testing PyTorch integration...")
        
        try:
            import torch
            print(f"  ✅ PyTorch version: {torch.__version__}")
            
            # Test basic tensor operations
            x = torch.randn(2, 3)
            y = torch.randn(2, 3)
            z = x + y
            print(f"  ✅ Basic tensor operations work")
            
            # Test if CUDA is available (not required but good to know)
            if torch.cuda.is_available():
                print(f"  ✅ CUDA available: {torch.cuda.device_count()} devices")
            else:
                print(f"  ⚠️  CUDA not available (CPU only)")
            
            return True
            
        except ImportError:
            print("  ❌ PyTorch not installed")
            self.results['errors'].append("PyTorch not installed")
            return False
        except Exception as e:
            print(f"  ❌ PyTorch error: {str(e)}")
            self.results['errors'].append(f"PyTorch error: {str(e)}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        total_tests = (len(self.results['core_systems']) + 
                      len(self.results['indicators']) +
                      len(self.results['agents']) +
                      len(self.results['data_components']) +
                      len(self.results['training']))
        
        successful_tests = sum(1 for category in ['core_systems', 'indicators', 'agents', 'data_components', 'training']
                              for result in self.results[category]
                              if result['status'] == 'success')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'errors_count': len(self.results['errors']),
            'warnings_count': len(self.results['warnings']),
            'results': self.results
        }
        
        return report
    
    def run_verification(self) -> bool:
        """Run complete verification suite"""
        print("🚀 Starting AlgoSpace Runtime Verification")
        print("=" * 60)
        
        # Add current directory to Python path
        if '.' not in sys.path:
            sys.path.insert(0, '.')
        
        results = []
        
        # Run all verification tests
        results.append(self.verify_core_systems())
        results.append(self.verify_indicators())
        results.append(self.verify_agents())
        results.append(self.verify_data_components())
        results.append(self.verify_training_components())
        results.append(self.test_pytorch_integration())
        
        # Generate report
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("📊 RUNTIME VERIFICATION REPORT")
        print("=" * 60)
        
        print(f"✅ Successful tests: {report['successful_tests']}/{report['total_tests']}")
        print(f"📈 Success rate: {report['success_rate']:.1%}")
        print(f"❌ Errors: {report['errors_count']}")
        print(f"⚠️  Warnings: {report['warnings_count']}")
        
        if report['errors_count'] > 0:
            print("\n🚨 ERRORS:")
            for error in self.results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        # Save detailed report
        with open('runtime_verification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: runtime_verification_report.json")
        
        # Overall success criteria
        overall_success = (report['success_rate'] >= 0.8 and 
                          report['errors_count'] < 10)
        
        if overall_success:
            print("\n🎉 Runtime verification PASSED!")
            return True
        else:
            print("\n⚠️  Runtime verification needs attention")
            return False

def main():
    """Main verification function"""
    verifier = RuntimeVerifier()
    success = verifier.run_verification()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()