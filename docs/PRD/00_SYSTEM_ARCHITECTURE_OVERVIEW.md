# AlgoSpace Trading System - Architecture Overview

## System Vision
AlgoSpace is a sophisticated multi-agent reinforcement learning (MARL) trading system designed for ES futures. The system uses an event-driven architecture where specialized AI agents collaborate to identify market opportunities, assess risk, and execute trades with minimal human intervention.

## Core Design Principles

### 1. Event-Driven Architecture
- **Central Event Bus**: All components communicate via standardized events
- **Loose Coupling**: Components are independent and can be developed/tested in isolation
- **Asynchronous Processing**: Non-blocking event flow for high-performance operation

### 2. Multi-Agent Reinforcement Learning (MARL)
- **Specialized Agents**: Each agent masters a specific aspect of trading
- **Collaborative Decision Making**: Agents work together to make final decisions
- **Continuous Learning**: System adapts to changing market conditions

### 3. Modular Component Design
- **Single Responsibility**: Each component has one clear purpose
- **Abstraction Layers**: Live/backtest modes use same interfaces
- **Configuration-Driven**: Behavior controlled via settings.yaml

## System Components

### Data Flow Chain
```
DataHandler → BarGenerator → IndicatorEngine → MatrixAssembler → AI Agents → ExecutionHandler
```

### Component Hierarchy

#### 1. **System Kernel & Orchestration**
- **Role**: System coordinator and entry point
- **Responsibilities**: Component initialization, event bus setup, graceful shutdown
- **Key Features**: Dependency injection, configuration management

#### 2. **DataHandler Component**
- **Role**: Market data abstraction layer
- **Input**: Live feed (Rithmic API) or historical CSV files
- **Output**: Standardized TickData events
- **Key Features**: Connection resilience, data normalization

#### 3. **BarGenerator Component**
- **Role**: Time-series aggregation engine
- **Input**: NEW_TICK events
- **Output**: NEW_5MIN_BAR, NEW_30MIN_BAR events
- **Key Features**: Concurrent multi-timeframe processing, gap handling

#### 4. **IndicatorEngine Component**
- **Role**: Technical analysis computation center
- **Input**: NEW_BAR events
- **Output**: INDICATORS_READY events with Feature Store
- **Key Features**: Heiken Ashi conversion, FVG/MLMI/NW-RQK/LVN calculations

#### 5. **MatrixAssembler Component**
- **Role**: Data structure preparation for ML models
- **Input**: INDICATORS_READY events
- **Output**: N×F matrices for AI agents
- **Key Features**: Rolling window management, feature selection

#### 6. **Regime Detection Engine (RDE)**
- **Role**: Market state classification using unsupervised learning
- **Architecture**: Transformer + VAE hybrid
- **Output**: Continuous regime vector
- **Key Features**: Unsupervised learning, latent space mapping

#### 7. **Risk Management Sub-system (M-RMS)**
- **Role**: Position sizing and risk assessment
- **Algorithm**: MAPPO-based reinforcement learning
- **Output**: Risk_Proposal objects
- **Key Features**: Sortino ratio optimization, rule-based constraints

#### 8. **Main MARL Core**
- **Role**: Central decision-making engine
- **Process**: Two-stage decision (Opportunity → Execution)
- **Output**: EXECUTE_TRADE events
- **Key Features**: MC Dropout confidence gating, ensemble decision making

#### 9. **ExecutionHandler Component**
- **Role**: Order management and trade execution
- **Input**: EXECUTE_TRADE events
- **Output**: TRADE_CLOSED events
- **Key Features**: Bracket orders, slippage simulation, position tracking

## AI/ML Architecture

### Agent Composition
Each "agent" consists of three parts:
1. **Pipeline (MatrixAssembler)**: Data preparation
2. **Senses (NN Embedder)**: Feature extraction via LSTM/Transformer
3. **Shared Brain (MAPPO Policy)**: Decision contribution

### Training Strategy: "Divide and Conquer"
1. **Phase 1**: Train RDE (unsupervised)
2. **Phase 2**: Train M-RMS (isolated RL environment)
3. **Phase 3**: Train Main MARL Core (with frozen expert models)

### Model Optimization
- **Training**: PyTorch with Ray RLlib
- **Production**: TensorRT conversion for sub-20ms inference
- **Validation**: MC Dropout for confidence estimation

## Event Flow Diagram

```
DataHandler
    ↓ NEW_TICK
BarGenerator
    ↓ NEW_5MIN_BAR, NEW_30MIN_BAR
IndicatorEngine
    ↓ INDICATORS_READY
MatrixAssembler (×3 instances)
    ↓ Matrix ready
Main MARL Core
    ├── RDE → Regime_Vector
    ├── M-RMS → Risk_Proposal
    └── DecisionGate → EXECUTE_TRADE
ExecutionHandler
    ↓ TRADE_CLOSED
Performance Tracking
```

## Configuration Management

### settings.yaml Structure
```yaml
system:
  mode: "live" | "backtest"
  
data_handler:
  live_settings:
    provider: "rithmic"
  backtest_settings:
    file_path: "data/ES_historical.csv"

marl_core:
  mc_dropout:
    iterations: 30
    confidence_threshold: 0.8
  
rde:
  latent_dim: 8
  lookback_window: 100
  
risk_management:
  max_position_size: 5
  max_daily_drawdown: 0.05
```

## Performance Requirements

### Latency Targets
- DataHandler processing: < 5ms
- BarGenerator update: < 100μs
- IndicatorEngine calculation: < 100ms
- Main MARL decision cycle: < 20ms
- ExecutionHandler order placement: < 5ms

### Throughput Targets
- Support 1000+ ticks/second sustained
- Handle multiple timeframes simultaneously
- Process complex indicators in real-time

## Technology Stack

### Core Technologies
- **Language**: Python 3.9+
- **ML Framework**: PyTorch + Ray RLlib
- **Data Processing**: NumPy, Pandas
- **Event System**: Custom implementation
- **Configuration**: PyYAML

### Production Optimization
- **Inference**: NVIDIA TensorRT
- **Monitoring**: Prometheus + Grafana
- **Containerization**: Docker
- **Orchestration**: Docker Compose

### Development Tools
- **Testing**: pytest, pytest-asyncio
- **Code Quality**: black, flake8, mypy
- **Documentation**: Sphinx
- **Version Control**: Git with LFS for models

## Deployment Modes

### Backtesting Mode
- Historical data simulation
- Realistic slippage modeling
- Comprehensive performance metrics
- Walk-forward analysis support

### Live Trading Mode
- Real-time Rithmic API connection
- Sub-second decision latency
- Robust error handling
- Automatic reconnection logic

## Future Enhancements

### V2.0 Roadmap
- Multi-asset support (NQ, YM, RTY)
- Additional timeframes (1min, 15min, 1hour)
- Advanced order types (iceberg, TWAP)
- Dynamic regime adaptation
- Distributed computing support

### Research Directions
- Attention mechanisms in RDE
- Online learning capabilities
- Alternative reward functions
- Multi-market correlation analysis

## Success Metrics

### Performance Targets
- **Sharpe Ratio**: > 2.0
- **Maximum Drawdown**: < 15%
- **Win Rate**: > 55%
- **Profit Factor**: > 1.5

### System Reliability
- **Uptime**: > 99.9%
- **Order Execution**: < 100ms latency
- **Data Processing**: < 1% packet loss
- **Error Recovery**: < 30s reconnection time

This architecture provides a robust foundation for a production-grade algorithmic trading system that can adapt to changing market conditions while maintaining strict risk controls and performance requirements.