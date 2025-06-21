# Product Requirements Document (PRD): MatrixAssembler Component

**Document Version**: 2.0  
**Date**: 2025-06-19  
**Status**: Refined  
**Component**: MatrixAssembler  

## 1.0 Overview

### 1.1 Purpose
The MatrixAssembler serves as the critical data structuring bridge between the feature-rich IndicatorEngine and the neural network components of the AI agents. It transforms the comprehensive feature store data into time-series matrices (N×F format) that serve as direct inputs for LSTM/Transformer embedders, enabling the AI agents to process temporal patterns and relationships in the market data.

### 1.2 Scope

**In Scope:**
- Multi-instance matrix management (30min, 5min, Regime streams)
- Rolling window time-series matrix construction
- Feature selection and filtering per agent type
- Real-time matrix updates with optimal performance
- Memory-efficient matrix operations
- Data quality validation and completeness checking
- Thread-safe matrix access for concurrent AI agent requests

**Out of Scope:**
- Feature calculation or generation (handled by IndicatorEngine)
- Neural network inference or model operations (handled by AI agents)
- Historical data persistence (beyond operational rolling windows)
- Feature engineering or transformation logic

### 1.3 Architectural Position
The MatrixAssembler sits between feature generation and AI processing:
IndicatorEngine → **MatrixAssembler** → Main MARL Core (AI Agents) → Decision Making

## 2.0 Functional Requirements

### FR-MA-01: Multi-Instance Architecture
**Requirement**: The system MUST support multiple independent MatrixAssembler instances, each configured for specific agent requirements.

**Specification**:
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from collections import deque
import threading
from datetime import datetime

@dataclass
class MatrixConfig:
    """Configuration for a specific MatrixAssembler instance"""
    name: str
    lookback_window: int  # N: number of historical time steps
    feature_keys: List[str]  # F: list of feature names to extract
    update_frequency: str  # "5min", "30min", "on_demand"
    data_validation: bool = True
    missing_value_strategy: str = "forward_fill"  # "forward_fill", "zero_fill", "interpolate"
    
class MatrixAssemblerManager:
    """Manages multiple MatrixAssembler instances"""
    
    def __init__(self, configs: List[MatrixConfig], event_bus):
        self.event_bus = event_bus
        self.assemblers = {}
        
        # Create assembler instances
        for config in configs:
            self.assemblers[config.name] = MatrixAssembler(config, event_bus)
            
        # Default configurations for standard agents
        self._setup_default_assemblers()
        
    def _setup_default_assemblers(self):
        """Setup standard assembler configurations"""
        
        # 30-minute agent matrix
        config_30m = MatrixConfig(
            name="agent_30m",
            lookback_window=100,
            feature_keys=[
                "30min_ha_close", "30min_ha_trend_numeric",
                "30min_mlmi_value", "30min_mlmi_wma", "30min_mlmi_strength",
                "30min_nwrqk_value", "30min_nwrqk_slope", "30min_nwrqk_strength",
                "30min_nearest_lvn_distance", "30min_nearest_lvn_strength"
            ],
            update_frequency="30min"
        )
        
        # 5-minute agent matrix
        config_5m = MatrixConfig(
            name="agent_5m", 
            lookback_window=200,
            feature_keys=[
                "5min_ha_close", "5min_ha_trend_numeric",
                "5min_active_fvg_count", "5min_nearest_fvg_distance",
                "5min_fvg_bias_numeric", "5min_price_momentum"
            ],
            update_frequency="5min"
        )
        
        # Regime detection matrix
        config_regime = MatrixConfig(
            name="regime_detector",
            lookback_window=150,
            feature_keys=[
                "30min_ha_close", "30min_mlmi_value", "30min_nwrqk_value",
                "30min_volume_profile_strength", "30min_market_structure_score",
                "macro_sentiment_score", "volatility_regime_indicator"
            ],
            update_frequency="30min"
        )
        
        # Create assemblers if not already existing
        for config in [config_30m, config_5m, config_regime]:
            if config.name not in self.assemblers:
                self.assemblers[config.name] = MatrixAssembler(config, self.event_bus)
```

### FR-MA-02: Rolling Window Matrix Management
**Requirement**: Each MatrixAssembler MUST maintain a fixed-size rolling window matrix with efficient append/remove operations.

**Matrix Implementation**:
```python
class RollingMatrix:
    """High-performance rolling window matrix implementation"""
    
    def __init__(self, lookback_window: int, feature_count: int):
        self.N = lookback_window  # Number of time steps
        self.F = feature_count    # Number of features
        
        # Pre-allocate matrix for performance
        self.matrix = np.full((self.N, self.F), np.nan, dtype=np.float32)
        self.timestamps = np.full(self.N, None, dtype=object)
        
        # Current position in circular buffer
        self.current_pos = 0
        self.is_full = False
        
        # Thread safety
        self._lock = threading.RLock()
        
    def append_row(self, feature_vector: np.ndarray, timestamp: datetime) -> None:
        """Append new feature vector to matrix"""
        
        with self._lock:
            # Validate input
            if len(feature_vector) != self.F:
                raise ValueError(f"Feature vector size {len(feature_vector)} != {self.F}")
            
            # Add new row at current position
            self.matrix[self.current_pos] = feature_vector
            self.timestamps[self.current_pos] = timestamp
            
            # Update position (circular buffer)
            self.current_pos = (self.current_pos + 1) % self.N
            
            if not self.is_full and self.current_pos == 0:
                self.is_full = True
                
    def get_matrix(self) -> np.ndarray:
        """Get current matrix in correct temporal order"""
        
        with self._lock:
            if not self.is_full:
                # Return only filled portion
                valid_rows = self.current_pos
                if valid_rows == 0:
                    return np.array([]).reshape(0, self.F)
                return self.matrix[:valid_rows].copy()
            else:
                # Reorder circular buffer to correct temporal sequence
                if self.current_pos == 0:
                    return self.matrix.copy()
                else:
                    return np.vstack([
                        self.matrix[self.current_pos:],
                        self.matrix[:self.current_pos]
                    ])
                    
    def get_latest_row(self) -> Optional[np.ndarray]:
        """Get the most recent feature vector"""
        
        with self._lock:
            if not self.is_full and self.current_pos == 0:
                return None
            
            latest_pos = (self.current_pos - 1) % self.N
            return self.matrix[latest_pos].copy()
            
    def get_completeness_ratio(self) -> float:
        """Get ratio of non-NaN values in matrix"""
        
        with self._lock:
            if not self.is_full and self.current_pos == 0:
                return 0.0
                
            valid_matrix = self.get_matrix()
            total_elements = valid_matrix.size
            valid_elements = np.sum(~np.isnan(valid_matrix))
            
            return valid_elements / total_elements if total_elements > 0 else 0.0
```

### FR-MA-03: Feature Selection and Extraction
**Requirement**: Each assembler MUST extract only its configured features from the comprehensive feature store.

**Feature Extraction Logic**:
```python
class FeatureExtractor:
    """Handles feature selection and preprocessing"""
    
    def __init__(self, feature_keys: List[str], validation_enabled: bool = True):
        self.feature_keys = feature_keys
        self.validation_enabled = validation_enabled
        self.feature_stats = {}  # For normalization/validation
        
    def extract_features(self, feature_store_data: Dict[str, Any]) -> np.ndarray:
        """Extract configured features from feature store"""
        
        feature_vector = []
        missing_features = []
        
        for feature_key in self.feature_keys:
            if feature_key in feature_store_data:
                value = feature_store_data[feature_key]
                
                # Convert to numeric if needed
                numeric_value = self._convert_to_numeric(value, feature_key)
                
                # Validate value
                if self.validation_enabled:
                    numeric_value = self._validate_feature_value(numeric_value, feature_key)
                
                feature_vector.append(numeric_value)
            else:
                missing_features.append(feature_key)
                feature_vector.append(np.nan)  # Will be handled by missing value strategy
                
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
        return np.array(feature_vector, dtype=np.float32)
        
    def _convert_to_numeric(self, value: Any, feature_key: str) -> float:
        """Convert feature value to numeric representation"""
        
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Handle categorical features
            if 'trend' in feature_key.lower():
                trend_map = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}
                return trend_map.get(value.lower(), 0.0)
            elif 'signal' in feature_key.lower():
                signal_map = {'buy': 1.0, 'sell': -1.0, 'hold': 0.0, 'neutral': 0.0}
                return signal_map.get(value.lower(), 0.0)
            elif 'bias' in feature_key.lower():
                bias_map = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}
                return bias_map.get(value.lower(), 0.0)
            else:
                # Try to parse as float
                try:
                    return float(value)
                except ValueError:
                    logger.warning(f"Cannot convert {feature_key}='{value}' to numeric")
                    return 0.0
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        else:
            logger.warning(f"Unknown type for {feature_key}: {type(value)}")
            return 0.0
            
    def _validate_feature_value(self, value: float, feature_key: str) -> float:
        """Validate and potentially clip feature values"""
        
        if np.isnan(value) or np.isinf(value):
            return np.nan
            
        # Update running statistics for this feature
        if feature_key not in self.feature_stats:
            self.feature_stats[feature_key] = {
                'min': value, 'max': value, 'count': 1, 'sum': value, 'sum_sq': value**2
            }
        else:
            stats = self.feature_stats[feature_key]
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['count'] += 1
            stats['sum'] += value
            stats['sum_sq'] += value**2
            
        # Detect extreme outliers (beyond 5 standard deviations)
        stats = self.feature_stats[feature_key]
        if stats['count'] > 10:  # Need minimum samples for statistics
            mean = stats['sum'] / stats['count']
            variance = (stats['sum_sq'] / stats['count']) - mean**2
            std = np.sqrt(max(0, variance))
            
            if abs(value - mean) > 5 * std:
                logger.warning(f"Extreme outlier detected for {feature_key}: {value}")
                # Clip to 3 standard deviations
                return mean + (3 * std * np.sign(value - mean))
                
        return value
```

### FR-MA-04: Missing Value Handling
**Requirement**: The system MUST implement configurable strategies for handling missing or invalid feature values.

**Missing Value Strategies**:
```python
class MissingValueHandler:
    """Handles various missing value strategies"""
    
    def __init__(self, strategy: str = "forward_fill"):
        self.strategy = strategy
        self.last_valid_values = {}
        
    def handle_missing_values(self, feature_vector: np.ndarray, 
                            feature_keys: List[str], 
                            timestamp: datetime) -> np.ndarray:
        """Apply missing value strategy to feature vector"""
        
        processed_vector = feature_vector.copy()
        
        for i, (value, key) in enumerate(zip(feature_vector, feature_keys)):
            if np.isnan(value):
                if self.strategy == "forward_fill":
                    processed_vector[i] = self._forward_fill(key, value)
                elif self.strategy == "zero_fill":
                    processed_vector[i] = 0.0
                elif self.strategy == "interpolate":
                    processed_vector[i] = self._interpolate(key, value, timestamp)
                elif self.strategy == "mean_fill":
                    processed_vector[i] = self._mean_fill(key)
                else:
                    processed_vector[i] = 0.0  # Default fallback
            else:
                # Update last valid value for forward fill
                self.last_valid_values[key] = value
                
        return processed_vector
        
    def _forward_fill(self, feature_key: str, missing_value: float) -> float:
        """Forward fill with last valid value"""
        return self.last_valid_values.get(feature_key, 0.0)
        
    def _mean_fill(self, feature_key: str) -> float:
        """Fill with historical mean (if available)"""
        # This would require maintaining running statistics
        # For now, fallback to forward fill
        return self.last_valid_values.get(feature_key, 0.0)
```

### FR-MA-05: Event-Driven Matrix Updates
**Requirement**: MatrixAssembler instances MUST respond to INDICATORS_READY events and update their matrices accordingly.

**Event Processing**:
```python
class MatrixAssembler:
    """Core MatrixAssembler implementation"""
    
    def __init__(self, config: MatrixConfig, event_bus):
        self.config = config
        self.event_bus = event_bus
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(config.feature_keys, config.data_validation)
        self.missing_value_handler = MissingValueHandler(config.missing_value_strategy)
        self.rolling_matrix = RollingMatrix(config.lookback_window, len(config.feature_keys))
        
        # Performance metrics
        self.metrics = MatrixAssemblerMetrics()
        
        # Subscribe to events
        self.event_bus.subscribe("INDICATORS_READY", self.on_indicators_ready)
        
    async def on_indicators_ready(self, event_type: str, payload: Dict[str, Any]):
        """Handle INDICATORS_READY event"""
        
        start_time = time.perf_counter()
        
        try:
            # Check if this update is relevant for our frequency
            if not self._should_process_update(payload):
                return
                
            # Extract features
            feature_vector = self.feature_extractor.extract_features(payload)
            
            # Handle missing values
            processed_vector = self.missing_value_handler.handle_missing_values(
                feature_vector, self.config.feature_keys, datetime.now()
            )
            
            # Update matrix
            self.rolling_matrix.append_row(processed_vector, datetime.now())
            
            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self.metrics.update_processing_time(processing_time)
            self.metrics.matrices_updated += 1
            
            # Quality check
            completeness = self.rolling_matrix.get_completeness_ratio()
            self.metrics.data_completeness = completeness
            
            if completeness < 0.8:  # Alert threshold
                logger.warning(f"Low data completeness for {self.config.name}: {completeness:.2%}")
                
        except Exception as e:
            logger.error(f"Error updating matrix {self.config.name}: {e}")
            self.metrics.error_count += 1
            
    def _should_process_update(self, payload: Dict[str, Any]) -> bool:
        """Determine if this update should trigger matrix update"""
        
        if self.config.update_frequency == "on_demand":
            return True
            
        # Check if any of our required features were updated
        required_timeframes = set()
        for feature_key in self.config.feature_keys:
            if feature_key.startswith("5min_"):
                required_timeframes.add("5min")
            elif feature_key.startswith("30min_"):
                required_timeframes.add("30min")
                
        # Check if payload contains updates for our timeframes
        for timeframe in required_timeframes:
            if f"{timeframe}_" in str(payload.keys()):
                return True
                
        return False
        
    def get_matrix(self) -> np.ndarray:
        """Get current matrix for AI agent consumption"""
        
        matrix = self.rolling_matrix.get_matrix()
        
        # Update access metrics
        self.metrics.matrix_requests += 1
        self.metrics.last_access_time = datetime.now()
        
        return matrix
        
    def get_matrix_info(self) -> Dict[str, Any]:
        """Get matrix metadata and quality information"""
        
        matrix = self.rolling_matrix.get_matrix()
        
        return {
            'name': self.config.name,
            'shape': matrix.shape,
            'completeness_ratio': self.rolling_matrix.get_completeness_ratio(),
            'feature_count': len(self.config.feature_keys),
            'feature_keys': self.config.feature_keys,
            'last_update': self.metrics.last_update_time,
            'data_quality_score': self._calculate_quality_score(matrix)
        }
        
    def _calculate_quality_score(self, matrix: np.ndarray) -> float:
        """Calculate overall data quality score"""
        
        if matrix.size == 0:
            return 0.0
            
        # Factors affecting quality:
        # 1. Completeness (% non-NaN values)
        completeness = np.sum(~np.isnan(matrix)) / matrix.size
        
        # 2. Consistency (low variance in feature ranges)
        feature_ranges = []
        for col in range(matrix.shape[1]):
            col_data = matrix[:, col]
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 1:
                feature_ranges.append(np.std(valid_data))
                
        consistency = 1.0 / (1.0 + np.mean(feature_ranges)) if feature_ranges else 0.5
        
        # 3. Recency (how recent is the data)
        recency = 1.0  # Assume real-time data is always recent
        
        # Combined quality score
        quality_score = (completeness * 0.5) + (consistency * 0.3) + (recency * 0.2)
        
        return min(1.0, max(0.0, quality_score))
```

### FR-MA-06: Performance Monitoring and Metrics
**Requirement**: Each MatrixAssembler MUST provide comprehensive performance and quality metrics.

**Metrics Implementation**:
```python
@dataclass
class MatrixAssemblerMetrics:
    """Performance and quality metrics for MatrixAssembler"""
    
    # Processing metrics
    matrices_updated: int = 0
    matrix_requests: int = 0
    error_count: int = 0
    
    # Performance metrics
    average_processing_time_ms: float = 0.0
    peak_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Quality metrics
    data_completeness: float = 0.0
    feature_correlation_score: float = 0.0
    outlier_detection_count: int = 0
    
    # Timestamps
    last_update_time: Optional[datetime] = None
    last_access_time: Optional[datetime] = None
    
    def update_processing_time(self, time_ms: float):
        """Update processing time statistics"""
        self.peak_processing_time_ms = max(self.peak_processing_time_ms, time_ms)
        
        # Exponential moving average
        alpha = 0.1
        self.average_processing_time_ms = (
            alpha * time_ms + (1 - alpha) * self.average_processing_time_ms
        )
        
        self.last_update_time = datetime.now()
```

## 3.0 Interface Specifications

### 3.1 Configuration Interface
```yaml
matrix_assembler:
  instances:
    - name: "agent_30m"
      lookback_window: 100
      update_frequency: "30min"
      feature_keys:
        - "30min_ha_close"
        - "30min_mlmi_value"
        - "30min_nwrqk_value"
        - "30min_nearest_lvn_strength"
      missing_value_strategy: "forward_fill"
      data_validation: true
      
    - name: "agent_5m"
      lookback_window: 200
      update_frequency: "5min"
      feature_keys:
        - "5min_ha_close"
        - "5min_active_fvg_count"
        - "5min_nearest_fvg_distance"
      missing_value_strategy: "forward_fill"
      
    - name: "regime_detector"
      lookback_window: 150
      update_frequency: "30min"
      feature_keys:
        - "30min_mlmi_value"
        - "30min_nwrqk_slope"
        - "30min_volume_profile_strength"
      missing_value_strategy: "interpolate"
      
  performance:
    max_processing_time_ms: 10
    memory_limit_mb: 200
    quality_threshold: 0.8
    alert_on_low_quality: true
```

### 3.2 API Interface
```python
class MatrixAssemblerAPI:
    """Public API for MatrixAssembler access"""
    
    def get_matrix(self, assembler_name: str) -> np.ndarray:
        """Get matrix for specific assembler"""
        
    def get_all_matrices(self) -> Dict[str, np.ndarray]:
        """Get all assembler matrices"""
        
    def get_matrix_info(self, assembler_name: str) -> Dict[str, Any]:
        """Get matrix metadata and quality info"""
        
    def get_metrics(self, assembler_name: str) -> MatrixAssemblerMetrics:
        """Get performance metrics"""
        
    def validate_matrix_quality(self, assembler_name: str) -> bool:
        """Check if matrix meets quality requirements"""
```

### 3.3 Event Interface

**Subscribed Events**:
- **INDICATORS_READY**: Primary trigger for matrix updates

**Published Events** (Optional):
- **MATRIX_UPDATED**: When matrix is updated
- **MATRIX_QUALITY_ALERT**: When quality falls below threshold

## 4.0 Dependencies & Interactions

### 4.1 Upstream Dependencies
- **IndicatorEngine**: Source of INDICATORS_READY events
- **Event Bus**: Event subscription mechanism
- **Configuration System**: Matrix assembler configurations

### 4.2 Downstream Dependencies
- **Main MARL Core**: Primary consumer of assembled matrices
- **AI Agents**: Direct consumers of N×F matrices
- **Performance Monitor**: Consumer of quality metrics

## 5.0 Non-Functional Requirements

### 5.1 Performance
- **NFR-MA-01**: Matrix updates MUST complete in under 10ms (95th percentile)
- **NFR-MA-02**: Matrix access MUST be sub-millisecond (thread-safe)
- **NFR-MA-03**: Memory usage MUST remain constant (no memory leaks)
- **NFR-MA-04**: Support concurrent access from multiple AI agents

### 5.2 Data Quality
- **NFR-MA-05**: Data completeness MUST exceed 80% for production use
- **NFR-MA-06**: Missing value handling MUST preserve temporal relationships
- **NFR-MA-07**: Feature extraction MUST be deterministic and reproducible

### 5.3 Reliability
- **NFR-MA-08**: Component MUST handle malformed feature data gracefully
- **NFR-MA-09**: Matrix operations MUST be atomic and thread-safe
- **NFR-MA-10**: System MUST continue operating if individual assemblers fail

## 6.0 Testing Requirements

### 6.1 Unit Tests
```python
def test_rolling_matrix_operations():
    """Test rolling window matrix operations"""
    matrix = RollingMatrix(lookback_window=5, feature_count=3)
    
    # Test appending and ordering
    for i in range(10):
        feature_vector = np.array([i, i*2, i*3], dtype=np.float32)
        matrix.append_row(feature_vector, datetime.now())
    
    result = matrix.get_matrix()
    assert result.shape == (5, 3)
    assert np.array_equal(result[-1], np.array([9, 18, 27]))  # Latest row

def test_feature_extraction():
    """Test feature extraction with various data types"""
    extractor = FeatureExtractor(['price', 'trend', 'volume'])
    
    test_data = {
        'price': 5100.25,
        'trend': 'bullish',
        'volume': 1500,
        'extra_feature': 999  # Should be ignored
    }
    
    result = extractor.extract_features(test_data)
    expected = np.array([5100.25, 1.0, 1500.0])
    np.testing.assert_array_equal(result, expected)

def test_missing_value_handling():
    """Test various missing value strategies"""
    handler = MissingValueHandler("forward_fill")
    
    # First vector with missing values
    vector1 = np.array([100.0, np.nan, 200.0])
    result1 = handler.handle_missing_values(vector1, ['a', 'b', 'c'], datetime.now())
    
    # Second vector should forward fill
    vector2 = np.array([np.nan, np.nan, 300.0])
    result2 = handler.handle_missing_values(vector2, ['a', 'b', 'c'], datetime.now())
    
    assert result2[0] == 100.0  # Forward filled from vector1
```

### 6.2 Integration Tests
- End-to-end event processing from IndicatorEngine
- Multi-assembler coordination and independence
- Thread safety under concurrent access
- Memory usage stability over extended periods

### 6.3 Performance Tests
```python
async def test_matrix_update_performance():
    """Test matrix update performance under load"""
    assembler = MatrixAssembler(test_config, mock_event_bus)
    
    update_times = []
    for _ in range(1000):
        start = time.perf_counter()
        await assembler.on_indicators_ready("INDICATORS_READY", mock_feature_data)
        end = time.perf_counter()
        update_times.append((end - start) * 1000)  # Convert to ms
    
    p95_time = np.percentile(update_times, 95)
    assert p95_time < 10.0, f"95th percentile update time too high: {p95_time}ms"
```

## 7.0 Future Enhancements

### 7.1 V2.0 Features
- **Dynamic Feature Selection**: ML-based feature importance ranking
- **Adaptive Window Sizes**: Dynamic lookback window optimization
- **Feature Normalization**: Automatic feature scaling and normalization
- **Matrix Compression**: Compressed storage for memory efficiency
- **Real-time Quality Monitoring**: Advanced data quality analytics

### 7.2 Advanced Capabilities
- **Feature Engineering**: Automated feature generation and selection
- **Temporal Alignment**: Intelligent handling of multi-timeframe synchronization
- **Distributed Matrices**: Support for distributed matrix operations
- **GPU Acceleration**: CUDA-based matrix operations for high-frequency data

This PRD establishes the foundation for a high-performance, reliable data structuring system that provides AI agents with properly formatted, high-quality temporal data while maintaining the flexibility to support various agent architectures and requirements.