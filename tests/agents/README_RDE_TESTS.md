# RDE Component Test Suite

This directory contains comprehensive tests for the Regime Detection Engine (RDE) component.

## Test Files

### 1. `test_rde_engine.py`
Complete unit test suite using pytest and mocks. Requires PyTorch to be installed.

**Test Cases:**
- `test_rde_component_initialization`: Verifies component creation
- `test_load_model_successfully`: Tests model loading functionality
- `test_load_model_with_different_checkpoint_formats`: Tests various checkpoint formats
- `test_load_model_file_not_found`: Tests error handling for missing files
- `test_get_regime_vector_interface`: Tests main inference method
- `test_get_regime_vector_different_sequence_lengths`: Tests flexibility with sequence lengths
- `test_rde_handles_incorrect_input_shape`: Tests input validation
- `test_get_regime_vector_without_loaded_model`: Tests error when model not loaded
- `test_get_model_info`: Tests metadata retrieval
- `test_validate_config`: Tests configuration validation
- `test_full_inference_pipeline`: Integration test of complete pipeline
- `test_device_handling`: Tests CPU/GPU device configuration

### 2. `test_rde_engine_structure.py`
Structural validation tests that can run without PyTorch installed.

**Test Cases:**
- `test_rde_module_structure`: Verifies file structure
- `test_model_py_contains_only_nn_modules`: Ensures clean separation of concerns
- `test_engine_py_has_required_methods`: Validates API surface
- `test_init_py_exports`: Checks module exports
- `test_kernel_integration`: Verifies system integration
- `test_config_yaml_has_rde_section`: Validates configuration

## Running the Tests

### With PyTorch installed:
```bash
pytest tests/agents/test_rde_engine.py -v
```

### Without PyTorch (structural tests only):
```bash
python tests/agents/test_rde_engine_structure.py
```

## Test Coverage

The test suite covers:
- ✅ Component initialization and configuration
- ✅ Model loading with various checkpoint formats
- ✅ Inference pipeline with proper tensor handling
- ✅ Input validation and error handling
- ✅ Integration with the kernel
- ✅ Configuration management
- ✅ Device handling (CPU/GPU)
- ✅ Edge cases and error conditions

## Key Validations

1. **Separation of Concerns**: `model.py` contains only neural network definitions
2. **Clean API**: `engine.py` provides a simple, robust interface
3. **Proper Integration**: Kernel correctly instantiates and loads the component
4. **Error Handling**: Graceful failures with informative error messages
5. **Type Safety**: Correct input/output types and shapes