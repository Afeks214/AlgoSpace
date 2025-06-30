"""
Structural validation tests for the RDE component.

These tests verify the code structure and organization without requiring
PyTorch or other dependencies to be installed.
"""

import ast
import os


def test_rde_module_structure():
    """Verify the RDE module has the correct file structure."""
    rde_path = 'src/agents/rde'
    
    # Check directory exists
    assert os.path.exists(rde_path), "RDE module directory not found"
    assert os.path.isdir(rde_path), "RDE path is not a directory"
    
    # Check required files exist
    required_files = ['__init__.py', 'model.py', 'engine.py']
    for filename in required_files:
        filepath = os.path.join(rde_path, filename)
        assert os.path.exists(filepath), f"Required file {filename} not found"
        assert os.path.isfile(filepath), f"{filename} is not a file"
    
    print("✅ RDE module structure verified")


def test_model_py_contains_only_nn_modules():
    """Verify model.py contains only neural network definitions."""
    model_path = 'src/agents/rde/model.py'
    
    with open(model_path, 'r') as f:
        content = f.read()
    
    # Parse the AST
    tree = ast.parse(content)
    
    # Check for forbidden elements
    forbidden_keywords = ['train', 'optimizer', 'DataLoader', 'save', 'load']
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_keywords, f"Found forbidden keyword: {node.id}"
    
    # Check that required classes exist
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    required_classes = ['RegimeDetectionEngine', 'TransformerEncoder', 'VAEHead', 'Decoder']
    for cls in required_classes:
        assert cls in class_names, f"Required class {cls} not found in model.py"
    
    print("✅ model.py contains only NN module definitions")


def test_engine_py_has_required_methods():
    """Verify engine.py has all required methods."""
    engine_path = 'src/agents/rde/engine.py'
    
    with open(engine_path, 'r') as f:
        content = f.read()
    
    # Parse the AST
    tree = ast.parse(content)
    
    # Find RDEComponent class
    rde_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'RDEComponent':
            rde_class = node
            break
    
    assert rde_class is not None, "RDEComponent class not found"
    
    # Check required methods
    method_names = [n.name for n in rde_class.body if isinstance(n, ast.FunctionDef)]
    required_methods = ['__init__', 'load_model', 'get_regime_vector']
    
    for method in required_methods:
        assert method in method_names, f"Required method {method} not found"
    
    # Check method signatures
    for node in rde_class.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == '__init__':
                # Should have self and config parameters
                assert len(node.args.args) >= 2, "__init__ should have self and config parameters"
                assert node.args.args[1].arg == 'config', "__init__ should accept config parameter"
            
            elif node.name == 'load_model':
                # Should have self and model_path parameters
                assert len(node.args.args) >= 2, "load_model should have self and model_path parameters"
                assert node.args.args[1].arg == 'model_path', "load_model should accept model_path parameter"
            
            elif node.name == 'get_regime_vector':
                # Should have self and mmd_matrix parameters
                assert len(node.args.args) >= 2, "get_regime_vector should have self and mmd_matrix parameters"
                assert node.args.args[1].arg == 'mmd_matrix', "get_regime_vector should accept mmd_matrix parameter"
    
    print("✅ engine.py has all required methods with correct signatures")


def test_init_py_exports():
    """Verify __init__.py exports the correct components."""
    init_path = 'src/agents/rde/__init__.py'
    
    with open(init_path, 'r') as f:
        content = f.read()
    
    # Check for RDEComponent import
    assert 'from .engine import RDEComponent' in content, "RDEComponent not imported in __init__.py"
    
    # Parse to check __all__
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    # Check that RDEComponent is in __all__
                    if isinstance(node.value, ast.List):
                        exports = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
                        assert 'RDEComponent' in exports, "RDEComponent not in __all__"
    
    print("✅ __init__.py correctly exports RDEComponent")


def test_kernel_integration():
    """Verify kernel.py correctly integrates RDE."""
    kernel_path = 'src/core/kernel.py'
    
    with open(kernel_path, 'r') as f:
        content = f.read()
    
    # Check for correct import
    assert 'from ..agents.rde import RDEComponent' in content, "RDEComponent not imported correctly"
    
    # Check for instantiation
    assert 'RDEComponent(rde_config)' in content, "RDEComponent not instantiated correctly"
    
    # Check for model loading
    assert "load_model(model_path)" in content, "load_model not called in kernel"
    
    # Ensure no references to old classes
    assert 'RegimeDetectionEngine' not in content.replace('RDEComponent', ''), "Old RegimeDetectionEngine class still referenced"
    
    print("✅ Kernel correctly integrates the new RDEComponent")


def test_config_yaml_has_rde_section():
    """Verify settings.yaml has RDE configuration."""
    config_path = 'config/settings.yaml'
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check for RDE section
    assert 'rde:' in content, "RDE configuration section not found"
    
    # Check for required RDE parameters
    required_params = ['input_dim:', 'd_model:', 'latent_dim:', 'n_heads:', 'n_layers:']
    for param in required_params:
        assert param in content, f"Required RDE parameter {param} not found in config"
    
    # Check for models section
    assert 'models:' in content, "Models section not found"
    assert 'rde_path:' in content, "RDE model path not configured"
    
    print("✅ Configuration file has proper RDE settings")


if __name__ == '__main__':
    # Run all tests
    test_rde_module_structure()
    test_model_py_contains_only_nn_modules()
    test_engine_py_has_required_methods()
    test_init_py_exports()
    test_kernel_integration()
    test_config_yaml_has_rde_section()
    
    print("\n✅ All structural validation tests passed!")
    print("Note: Full unit tests with mocks are in test_rde_engine.py (requires PyTorch)")