"""
Structural validation tests for the M-RMS component.

These tests verify the code structure and organization without requiring
PyTorch or other dependencies to be installed.
"""

import ast
import os


def test_mrms_module_structure():
    """Verify the M-RMS module has the correct file structure."""
    mrms_path = 'src/agents/mrms'
    
    # Check directory exists
    assert os.path.exists(mrms_path), "M-RMS module directory not found"
    assert os.path.isdir(mrms_path), "M-RMS path is not a directory"
    
    # Check required files exist
    required_files = ['__init__.py', 'models.py', 'engine.py']
    for filename in required_files:
        filepath = os.path.join(mrms_path, filename)
        assert os.path.exists(filepath), f"Required file {filename} not found"
        assert os.path.isfile(filepath), f"{filename} is not a file"
    
    print("✅ M-RMS module structure verified")


def test_models_py_contains_nn_modules():
    """Verify models.py contains all required neural network classes."""
    models_path = 'src/agents/mrms/models.py'
    
    with open(models_path, 'r') as f:
        content = f.read()
    
    # Parse the AST
    tree = ast.parse(content)
    
    # Check that required classes exist
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    required_classes = ['PositionSizingAgent', 'StopLossAgent', 'ProfitTargetAgent', 'RiskManagementEnsemble']
    for cls in required_classes:
        assert cls in class_names, f"Required class {cls} not found in models.py"
    
    # Check for forbidden elements (should be pure models)
    forbidden_keywords = ['train', 'optimizer', 'DataLoader', 'save_model', 'load_model']
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_keywords, f"Found forbidden keyword: {node.id}"
    
    print("✅ models.py contains only NN module definitions")


def test_engine_py_has_required_methods():
    """Verify engine.py has all required methods."""
    engine_path = 'src/agents/mrms/engine.py'
    
    with open(engine_path, 'r') as f:
        content = f.read()
    
    # Parse the AST
    tree = ast.parse(content)
    
    # Find MRMSComponent class
    mrms_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'MRMSComponent':
            mrms_class = node
            break
    
    assert mrms_class is not None, "MRMSComponent class not found"
    
    # Check required methods
    method_names = [n.name for n in mrms_class.body if isinstance(n, ast.FunctionDef)]
    required_methods = ['__init__', 'load_model', 'generate_risk_proposal', 'get_model_info']
    
    for method in required_methods:
        assert method in method_names, f"Required method {method} not found"
    
    # Check method signatures
    for node in mrms_class.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == '__init__':
                # Should have self and config parameters
                assert len(node.args.args) >= 2, "__init__ should have self and config parameters"
                assert node.args.args[1].arg == 'config', "__init__ should accept config parameter"
            
            elif node.name == 'load_model':
                # Should have self and model_path parameters
                assert len(node.args.args) >= 2, "load_model should have self and model_path parameters"
                assert node.args.args[1].arg == 'model_path', "load_model should accept model_path parameter"
            
            elif node.name == 'generate_risk_proposal':
                # Should have self and trade_qualification parameters
                assert len(node.args.args) >= 2, "generate_risk_proposal should have self and trade_qualification parameters"
                assert node.args.args[1].arg == 'trade_qualification', "generate_risk_proposal should accept trade_qualification parameter"
    
    print("✅ engine.py has all required methods with correct signatures")


def test_init_py_exports():
    """Verify __init__.py exports the correct components."""
    init_path = 'src/agents/mrms/__init__.py'
    
    with open(init_path, 'r') as f:
        content = f.read()
    
    # Check for MRMSComponent import
    assert 'from .engine import MRMSComponent' in content, "MRMSComponent not imported in __init__.py"
    
    # Parse to check __all__
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    # Check that MRMSComponent is in __all__
                    if isinstance(node.value, ast.List):
                        exports = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
                        assert 'MRMSComponent' in exports, "MRMSComponent not in __all__"
    
    print("✅ __init__.py correctly exports MRMSComponent")


def test_kernel_integration():
    """Verify kernel.py correctly integrates M-RMS."""
    kernel_path = 'src/core/kernel.py'
    
    with open(kernel_path, 'r') as f:
        content = f.read()
    
    # Check for correct import
    assert 'from ..agents.mrms import MRMSComponent' in content, "MRMSComponent not imported correctly"
    
    # Check for instantiation
    assert 'MRMSComponent(mrms_config)' in content, "MRMSComponent not instantiated correctly"
    
    # Check for model loading
    assert 'load_model(model_path)' in content, "load_model not called in kernel"
    
    # Ensure no references to old RiskManagementSubsystem
    assert 'RiskManagementSubsystem' not in content, "Old RiskManagementSubsystem class still referenced"
    
    print("✅ Kernel correctly integrates the new MRMSComponent")


def test_config_yaml_has_mrms_section():
    """Verify settings.yaml has M-RMS configuration."""
    config_path = 'config/settings.yaml'
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check for M-RMS section
    assert 'm_rms:' in content, "M-RMS configuration section not found"
    
    # Check for required M-RMS parameters
    required_params = ['synergy_dim:', 'account_dim:', 'max_position_size:', 'point_value:']
    for param in required_params:
        assert param in content, f"Required M-RMS parameter {param} not found in config"
    
    # Check for models section
    assert 'models:' in content, "Models section not found"
    assert 'mrms_path:' in content, "M-RMS model path not configured"
    
    print("✅ Configuration file has proper M-RMS settings")


def test_risk_proposal_structure():
    """Verify the generate_risk_proposal method returns correct structure."""
    engine_path = 'src/agents/mrms/engine.py'
    
    with open(engine_path, 'r') as f:
        content = f.read()
    
    # Check that risk proposal includes all required fields
    required_fields = [
        "'position_size':",
        "'stop_loss_price':",
        "'take_profit_price':",
        "'risk_amount':",
        "'reward_amount':",
        "'risk_reward_ratio':",
        "'sl_atr_multiplier':",
        "'confidence_score':",
        "'risk_metrics':"
    ]
    
    for field in required_fields:
        assert field in content, f"Risk proposal missing required field: {field}"
    
    print("✅ Risk proposal structure is complete")


if __name__ == '__main__':
    print("Running M-RMS structural validation tests...\n")
    
    # Run all tests
    test_mrms_module_structure()
    test_models_py_contains_nn_modules()
    test_engine_py_has_required_methods()
    test_init_py_exports()
    test_kernel_integration()
    test_config_yaml_has_mrms_section()
    test_risk_proposal_structure()
    
    print("\n✅ All M-RMS structural validation tests passed!")
    print("Note: Full unit tests with mocks would require PyTorch to be installed")