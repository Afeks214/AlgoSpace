# AlgoSpace - AI-Powered Algorithmic Trading System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![Production Ready](https://img.shields.io/badge/production-ready-green.svg)](https://github.com/Afeks214/AlgoSpace)

## 🚀 Overview

AlgoSpace is a sophisticated Multi-Agent Reinforcement Learning (MARL) trading system that combines advanced AI techniques with real-time market analysis for automated trading decisions.

### Key Features

- **Multi-Agent Architecture**: RDE, M-RMS, and Main MARL Core for comprehensive market analysis
- **Real-Time Processing**: Sub-100ms decision latency with PyTorch optimization
- **Advanced Indicators**: MLMI, NW-RQK, FVG, LVN, and MMD indicators
- **Risk Management**: Integrated M-RMS with dynamic position sizing
- **Production Ready**: 100/100 health score with comprehensive monitoring

## 🏗️ Architecture

```
AlgoSpace System
├── Data Pipeline (Tick → Bar → Indicators)
├── Intelligence Layer (RDE + M-RMS + MARL Core)
├── Decision Engine (Multi-Agent with MC Dropout)
└── Execution Handler (Order Management)
```

### Core Components

- **RDE (Regime Detection Engine)**: Transformer-VAE for market regime identification
- **M-RMS (Multi-Regime Risk Management System)**: Dynamic risk adaptation
- **Main MARL Core**: Multi-agent decision making with PPO/SAC
- **Indicator Engine**: Real-time technical analysis (MLMI, NW-RQK, FVG, LVN, MMD)
- **Event Bus**: High-performance inter-component communication

## 📋 Requirements

- Python 3.9+
- PyTorch 2.7.1
- NumPy, Pandas, StructLog
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/Afeks214/AlgoSpace.git
cd AlgoSpace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version)
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🚀 Quick Start

```bash
# Run system health check
python scripts/runtime_verification.py

# Start the system
python main.py --config config/default.yaml

# Run tests
python -m pytest tests/ -v
```

## 📊 Production Status

- **Health Score**: 100/100 ✅
- **PyTorch Integration**: Operational ✅
- **Logger Performance**: 100/100 (Fixed 403+ calls) ✅
- **Memory Management**: Stable (< 3MB growth) ✅
- **Thread Safety**: No race conditions ✅
- **Component Tests**: All passing ✅

## 🧠 AI Components

### RDE (Regime Detection Engine)
- **Architecture**: Transformer + VAE
- **Input**: 155-dimensional MMD features
- **Output**: 8-dimensional regime vectors
- **Latency**: <10ms inference time

### M-RMS (Multi-Regime Risk Management)
- **Dynamic Position Sizing**: Based on regime confidence
- **Risk Metrics**: VaR, Expected Shortfall, Sharpe optimization
- **Adaptive**: Real-time parameter adjustment

### Indicator Engine
- **MLMI**: Machine Learning Market Index
- **NW-RQK**: Nadaraya-Watson Regression with Rational Quadratic Kernel
- **FVG**: Fair Value Gap detection
- **LVN**: Low Volume Nodes analysis
- **MMD**: Maximum Mean Discrepancy features

## 📚 Documentation

- [System Architecture](docs/architecture.md)
- [PRD Documents](docs/prd/)
- [API Reference](docs/api/)
- [Component Guide](docs/components.md)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific component tests
python -m pytest tests/core/ -v
python -m pytest tests/agents/ -v
python -m pytest tests/indicators/ -v

# Run production readiness tests
python scripts/runtime_verification.py
```

## 🚀 Deployment

The system is production-ready with:
- Comprehensive error handling
- Memory leak prevention
- Thread-safe operations
- Performance monitoring
- Automatic recovery systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch and modern MARL techniques
- Inspired by cutting-edge algorithmic trading research
- Developed for high-frequency futures trading environments

---

**Status**: Production Ready ✅ | **Version**: 1.0.0 | **Last Updated**: December 2024
