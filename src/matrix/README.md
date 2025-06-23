# Matrix Assembler Module

The Matrix Assembler module provides the critical bridge between raw indicator features and AI-ready neural network inputs. It implements three specialized matrix assemblers that transform point-in-time features into rolling time-series matrices optimized for different aspects of market analysis.

## Architecture Overview

```
INDICATORS_READY Event
         ↓
    Feature Store
         ↓
┌────────────────────┐
│  Matrix Assemblers │
├────────────────────┤
│ • MatrixAssembler30m  │ → 48×8 matrix (Long-term structure)
│ • MatrixAssembler5m   │ → 60×7 matrix (Short-term tactics)
│ • MatrixAssemblerRegime │ → 96×N matrix (Market regime)
└────────────────────┘
         ↓
   Neural Network
   Ready Matrices
```

## Components

### 1. BaseMatrixAssembler

**Purpose**: Abstract base class providing common functionality for all matrix assemblers.

**Key Features**:
- Thread-safe circular buffer implementation
- Rolling normalization with exponential moving averages
- Performance monitoring and error tracking
- Configurable warm-up periods
- Automatic event subscription

**Thread Safety**: Uses `threading.RLock()` for all matrix operations, ensuring safe concurrent access.

### 2. MatrixAssembler30m - Long-term Structure

**Purpose**: Creates a 48×8 matrix capturing 24 hours of market structure using 30-minute bars.

**Features (8 dimensions)**:
1. `mlmi_value` - Machine Learning Market Index (scaled to [-1, 1])
2. `mlmi_signal` - Crossover signal (-1, 0, 1)
3. `nwrqk_value` - Nadaraya-Watson value (% from current price)
4. `nwrqk_slope` - Rate of change (z-score normalized)
5. `lvn_distance_points` - Distance to LVN (exponential decay)
6. `lvn_nearest_strength` - LVN strength (scaled to [0, 1])
7. `time_hour_sin` - Cyclical hour encoding (sin component)
8. `time_hour_cos` - Cyclical hour encoding (cos component)

**Matrix Shape**: `(48, 8)` - 24 hours of 30-minute bars

### 3. MatrixAssembler5m - Short-term Tactics

**Purpose**: Creates a 60×7 matrix capturing 5 hours of price action using 5-minute bars.

**Features (7 dimensions)**:
1. `fvg_bullish_active` - Binary flag for active bullish FVG
2. `fvg_bearish_active` - Binary flag for active bearish FVG
3. `fvg_nearest_level` - Distance to nearest FVG (% normalized)
4. `fvg_age` - Age with exponential decay
5. `fvg_mitigation_signal` - Binary mitigation flag
6. `price_momentum_5` - 5-bar momentum (% scaled)
7. `volume_ratio` - Volume ratio (log-transformed)

**Matrix Shape**: `(60, 7)` - 5 hours of 5-minute bars

### 4. MatrixAssemblerRegime - Market Regime

**Purpose**: Creates a 96×N matrix capturing 48 hours of market behavior for regime detection.

**Features (N dimensions)**:
- `mmd_features[0..7]` - Market Microstructure Dynamics features
- `volatility_30` - 30-period volatility (z-score normalized)
- `volume_profile_skew` - Volume distribution skewness
- `price_acceleration` - Second derivative of price movement

**Matrix Shape**: `(96, N)` where N = MMD dimension + 3

## Usage Example

```python
from src.core.kernel import SystemKernel
from src.matrix import MatrixAssembler30m, MatrixAssembler5m, MatrixAssemblerRegime

# Initialize system
kernel = SystemKernel()

# Create assemblers
assembler_30m = MatrixAssembler30m("Structure", kernel)
assembler_5m = MatrixAssembler5m("Tactical", kernel)  
assembler_regime = MatrixAssemblerRegime("Regime", kernel)

# Register with kernel (automatic event subscription)
kernel.register_component("MatrixAssembler30m", assembler_30m, ["IndicatorEngine"])
kernel.register_component("MatrixAssembler5m", assembler_5m, ["IndicatorEngine"])
kernel.register_component("MatrixAssemblerRegime", assembler_regime, ["IndicatorEngine"])

# Start system - assemblers will automatically update on INDICATORS_READY events
await kernel.start()

# Access matrices (after warm-up period)
if assembler_30m.is_ready():
    matrix_30m = assembler_30m.get_matrix()  # Shape: (48, 8)
    print(f"30m matrix shape: {matrix_30m.shape}")

if assembler_5m.is_ready():
    matrix_5m = assembler_5m.get_matrix()    # Shape: (60, 7)
    print(f"5m matrix shape: {matrix_5m.shape}")

if assembler_regime.is_ready():
    matrix_regime = assembler_regime.get_matrix()  # Shape: (96, N)
    print(f"Regime matrix shape: {matrix_regime.shape}")
```

## Configuration

Matrix assemblers are configured in `config/settings.yaml`:

```yaml
matrix_assemblers:
  30m:
    window_size: 48
    warmup_period: 48
    features:
      - mlmi_value
      - mlmi_signal
      # ... other features
    feature_configs:
      mlmi_value:
        ema_alpha: 0.02  # EMA adaptation rate
      nwrqk_slope:
        ema_alpha: 0.05
        
  5m:
    window_size: 60
    warmup_period: 20
    # ... similar structure
    
  regime:
    window_size: 96
    warmup_period: 30
    # ... similar structure
```

## Normalization Methods

The module includes comprehensive normalization utilities:

### Core Functions
- `z_score_normalize()` - Standard z-score with clipping
- `min_max_scale()` - Min-max scaling to target range
- `cyclical_encode()` - Sin/cos encoding for periodic features
- `percentage_from_price()` - Price distance as percentage
- `exponential_decay()` - Age-based decay for time-sensitive features
- `log_transform()` - Safe logarithmic transformation

### RollingNormalizer Class
Maintains online statistics for adaptive normalization:
- Exponential moving averages for mean/variance
- Approximate percentile tracking
- Multiple normalization methods (z-score, min-max, robust)

## Performance Characteristics

### Latency Requirements (from PRD)
- **Update Latency**: <1ms per INDICATORS_READY event ✓
- **Access Latency**: <100μs for get_matrix() call ✓
- **Memory Usage**: <50MB total for all assemblers ✓

### Memory Management
- Fixed-size numpy arrays (no dynamic allocation)
- Circular buffer prevents memory growth
- Thread-safe access with minimal locking overhead
- Float32 precision for neural network efficiency

## Thread Safety

All matrix assemblers are fully thread-safe:
- **Read-Write Locks**: Protect matrix updates and access
- **Atomic Operations**: All matrix updates are atomic
- **Concurrent Access**: Multiple readers supported
- **No Race Conditions**: Proper synchronization throughout

## Error Handling

### Data Validation
- Input feature validation with type checking
- Range validation for expected feature bounds
- Non-finite value detection and handling
- Missing data interpolation

### Robustness Features
- Graceful degradation on missing features
- Automatic recovery from temporary data issues
- Comprehensive logging for debugging
- Performance monitoring and alerting

## Monitoring and Diagnostics

### Statistics Available
```python
stats = assembler.get_statistics()
# Returns:
# {
#     'window_size': 48,
#     'n_features': 8,
#     'n_updates': 1250,
#     'is_ready': True,
#     'performance': {
#         'avg_latency_ms': 0.15,
#         'max_latency_ms': 0.89
#     },
#     'matrix_stats': {
#         'shape': (48, 8),
#         'mean': 0.023,
#         'std': 0.456
#     }
# }
```

### Validation
```python
is_valid, issues = assembler.validate_matrix()
if not is_valid:
    print(f"Matrix issues detected: {issues}")
```

## Integration with MARL Agents

Matrix assemblers provide the input data for MARL agents:

```python
# For neural network input
def get_agent_input(assembler):
    matrix = assembler.get_matrix()
    if matrix is not None:
        # Convert to PyTorch tensor
        import torch
        return torch.tensor(matrix, dtype=torch.float32)
    return None

# Agent forward pass
structure_input = get_agent_input(assembler_30m)
tactical_input = get_agent_input(assembler_5m)
regime_input = get_agent_input(assembler_regime)
```

## Testing

Comprehensive test suite available:
- `tests/test_matrix_assemblers.py` - Unit tests
- `tests/test_matrix_integration.py` - Integration tests
- `test_matrix_simple.py` - Simple validation script

Run tests:
```bash
python test_matrix_simple.py
python -m unittest tests.test_matrix_assemblers
```

## Future Enhancements

1. **Dynamic Feature Selection**: Runtime feature enable/disable
2. **Multiple Timeframes**: Support for additional timeframes
3. **Compression**: Optional matrix compression for memory efficiency
4. **Persistence**: Optional matrix state persistence
5. **Distributed Mode**: Multi-process matrix assembly

## Implementation Notes

### Design Decisions
- **Circular Buffers**: Efficient memory usage with O(1) updates
- **Float32 Precision**: Balance between accuracy and performance
- **Copy-on-Access**: Return matrix copies to prevent external modification
- **Event-Driven**: Automatic updates via event subscription
- **Configurable**: Flexible configuration for different strategies

### Performance Optimizations
- Pre-allocated numpy arrays
- Vectorized operations
- Minimal data copying
- Efficient normalization caching
- Thread-local storage for temporary calculations

This module represents a robust, production-ready implementation that bridges the gap between raw market data and AI-ready inputs for the AlgoSpace trading system.