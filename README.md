# AlgoSpace Trading System

An advanced automated trading system using Multi-Agent Reinforcement Learning (MARL) for futures trading.

## Overview

AlgoSpace is a sophisticated algorithmic trading platform that combines multiple AI agents, advanced indicators, and robust risk management to trade futures contracts. The system uses a multi-timeframe approach with specialized agents for different trading aspects.

## Key Features

- **Multi-Agent System**: Specialized agents for 30m trends, 5m execution, regime detection, and risk management
- **Advanced Indicators**: MLMI, NWRQK, FVG, LVN implementations
- **Event-Driven Architecture**: Async event bus for real-time processing
- **Multi-Timeframe Analysis**: Simultaneous 5-minute and 30-minute analysis
- **Risk Management**: Multi-level risk controls and position sizing
- **Backtesting Engine**: Comprehensive historical testing capabilities

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AlgoSpace
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials and settings
   ```

5. **Run the system**
   ```bash
   python src/main.py
   ```

## Project Structure

```
algospace/
├── src/                 # Source code
│   ├── core/           # Event system and kernel
│   ├── data/           # Data handlers
│   ├── indicators/     # Technical indicators
│   ├── agents/         # MARL agents
│   ├── execution/      # Order execution
│   └── utils/          # Utilities
├── config/             # Configuration files
├── models/             # Trained ML models
├── data/               # Historical data
├── tests/              # Test suite
└── docs/               # Documentation and PRDs
```

## Development Status

### Phase 1: Foundation (Current)
- [x] Project structure setup
- [x] Environment configuration
- [ ] Core event system
- [ ] System kernel

### Phase 2: Data Pipeline
- [ ] Abstract data handler
- [ ] Backtest data handler
- [ ] Bar generator

### Phase 3: Indicators
- [ ] MLMI implementation
- [ ] NWRQK implementation
- [ ] FVG implementation
- [ ] LVN implementation

### Phase 4: MARL Framework
- [ ] Agent base structure
- [ ] 30m trend agent
- [ ] 5m execution agent
- [ ] Regime detection agent
- [ ] Risk management agent

### Phase 5: Execution & Risk
- [ ] Execution handler
- [ ] Order management
- [ ] Risk management system

## Configuration

Main configuration is in `config/settings.yaml`. Key settings include:
- Trading symbols and timeframes
- Indicator parameters
- Agent configurations
- Risk management limits
- Backtesting parameters

## Documentation

Detailed documentation is available in the `docs/` directory:
- System architecture PRDs
- Component specifications
- Research materials

## License

Proprietary - All rights reserved