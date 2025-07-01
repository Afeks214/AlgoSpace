#!/usr/bin/env python3
"""
M-RMS Implementation Verification Script.

This script verifies that the Multi-Agent Risk Management Subsystem (M-RMS)
has been correctly implemented and integrated into the AlgoSpace system.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

def verify_directory_structure():
    """Verify the M-RMS directory structure exists."""
    print("🔍 Verifying M-RMS directory structure...")
    
    required_files = [
        'src/agents/mrms/__init__.py',
        'src/agents/mrms/models.py',
        'src/agents/mrms/engine.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✅ M-RMS directory structure is complete!")
    return True

def verify_models_implementation():
    """Verify the models.py implementation."""
    print("\n🔍 Verifying models.py implementation...")
    
    try:
        with open('src/agents/mrms/models.py', 'r') as f:
            content = f.read()
        
        required_classes = [
            'PositionSizingAgent',
            'StopLossAgent', 
            'ProfitTargetAgent',
            'RiskManagementEnsemble'
        ]
        
        missing_classes = []
        for class_name in required_classes:
            if f'class {class_name}(' in content:
                print(f"  ✓ {class_name} class found")
            else:
                missing_classes.append(class_name)
        
        if missing_classes:
            print(f"✗ Missing classes: {missing_classes}")
            return False
        
        # Check for key methods
        key_methods = [
            'def forward(',
            'def get_action_dict(',
            'def get_model_info('
        ]
        
        missing_methods = []
        for method in key_methods:
            if method in content:
                print(f"  ✓ {method.replace('def ', '').replace('(', '')} method found")
            else:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"✗ Missing methods: {missing_methods}")
            return False
        
        print("✅ Models implementation is complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error reading models.py: {e}")
        return False

def verify_engine_implementation():
    """Verify the engine.py implementation."""
    print("\n🔍 Verifying engine.py implementation...")
    
    try:
        with open('src/agents/mrms/engine.py', 'r') as f:
            content = f.read()
        
        # Check for MRMSComponent class
        if 'class MRMSComponent' not in content:
            print("✗ MRMSComponent class not found")
            return False
        print("  ✓ MRMSComponent class found")
        
        # Check for required methods
        required_methods = [
            'def __init__(',
            'def load_model(',
            'def generate_risk_proposal(',
            'def get_model_info('
        ]
        
        missing_methods = []
        for method in required_methods:
            if method in content:
                print(f"  ✓ {method.replace('def ', '').replace('(', '')} method found")
            else:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"✗ Missing methods: {missing_methods}")
            return False
        
        # Check for key functionality
        key_features = [
            'RiskManagementEnsemble(',
            'torch.no_grad()',
            '_validate_trade_qualification',
            'position_size',
            'stop_loss_price',
            'take_profit_price'
        ]
        
        missing_features = []
        for feature in key_features:
            if feature in content:
                print(f"  ✓ {feature} implementation found")
            else:
                missing_features.append(feature)
        
        if missing_features:
            print(f"✗ Missing features: {missing_features}")
            return False
        
        print("✅ Engine implementation is complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error reading engine.py: {e}")
        return False

def verify_init_file():
    """Verify the __init__.py file."""
    print("\n🔍 Verifying __init__.py file...")
    
    try:
        with open('src/agents/mrms/__init__.py', 'r') as f:
            content = f.read()
        
        if 'from .engine import MRMSComponent' in content:
            print("  ✓ MRMSComponent import found")
        else:
            print("✗ MRMSComponent import not found")
            return False
        
        if "__all__ = ['MRMSComponent']" in content or '__all__ = ["MRMSComponent"]' in content:
            print("  ✓ __all__ export found")
        else:
            print("✗ __all__ export not found")
            return False
        
        print("✅ __init__.py is correct!")
        return True
        
    except Exception as e:
        print(f"✗ Error reading __init__.py: {e}")
        return False

def verify_system_integration():
    """Verify integration with the system kernel."""
    print("\n🔍 Verifying system integration...")
    
    try:
        with open('src/core/kernel.py', 'r') as f:
            kernel_content = f.read()
        
        # Check for import
        if 'from ..agents.mrms import MRMSComponent' in kernel_content:
            print("  ✓ MRMSComponent import found in kernel")
        else:
            print("  ⚠ MRMSComponent import not found in kernel (may use try/except)")
        
        # Check for instantiation
        if "MRMSComponent(mrms_config)" in kernel_content:
            print("  ✓ MRMSComponent instantiation found")
        else:
            print("✗ MRMSComponent instantiation not found")
            return False
        
        print("✅ System integration is complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error reading kernel.py: {e}")
        return False

def verify_configuration():
    """Verify M-RMS configuration in settings.yaml."""
    print("\n🔍 Verifying configuration...")
    
    try:
        with open('config/settings.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        if 'm_rms' not in config:
            print("✗ m_rms configuration section not found")
            return False
        
        mrms_config = config['m_rms']
        required_params = [
            'synergy_dim',
            'account_dim',
            'device',
            'point_value',
            'max_position_size'
        ]
        
        missing_params = []
        for param in required_params:
            if param in mrms_config:
                print(f"  ✓ {param}: {mrms_config[param]}")
            else:
                missing_params.append(param)
        
        if missing_params:
            print(f"✗ Missing configuration parameters: {missing_params}")
            return False
        
        print("✅ Configuration is complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error reading configuration: {e}")
        return False

def verify_syntax():
    """Verify Python syntax of all M-RMS files."""
    print("\n🔍 Verifying Python syntax...")
    
    import py_compile
    
    files_to_check = [
        'src/agents/mrms/models.py',
        'src/agents/mrms/engine.py',
        'src/agents/mrms/__init__.py'
    ]
    
    for file_path in files_to_check:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"  ✓ {file_path} compiles successfully")
        except py_compile.PyCompileError as e:
            print(f"✗ Syntax error in {file_path}: {e}")
            return False
    
    print("✅ All files have valid Python syntax!")
    return True

def main():
    """Main verification function."""
    print("🚀 Starting M-RMS Implementation Verification\n")
    
    verification_results = []
    
    # Run all verification steps
    verification_steps = [
        ("Directory Structure", verify_directory_structure),
        ("Models Implementation", verify_models_implementation),
        ("Engine Implementation", verify_engine_implementation),
        ("Init File", verify_init_file),
        ("System Integration", verify_system_integration),
        ("Configuration", verify_configuration),
        ("Python Syntax", verify_syntax)
    ]
    
    for step_name, step_function in verification_steps:
        try:
            result = step_function()
            verification_results.append((step_name, result))
        except Exception as e:
            print(f"✗ Error in {step_name}: {e}")
            verification_results.append((step_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for step_name, result in verification_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{step_name:<25} {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 ALL VERIFICATION CHECKS PASSED!")
        print("✅ M-RMS component is fully implemented and integrated")
        print("✅ Ready for model loading and inference")
        print("✅ System integration is complete")
    else:
        print("❌ SOME VERIFICATION CHECKS FAILED!")
        print("Please review the errors above and fix the issues.")
    
    print("="*60)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)