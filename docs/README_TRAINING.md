# 🚀 AlgoSpace Training Guide

**Complete training readiness and execution guide for the AlgoSpace MARL system**

## 🎯 Quick Start

### 1. Run Training Readiness Enablement
```bash
# Option A: Run the standalone script
python training_readiness_enablement.py

# Option B: Use the Jupyter notebook (recommended for Colab)
jupyter notebook notebooks/00_Training_Readiness_Enablement.ipynb
```

### 2. Follow the 3-Phase Training Plan
1. **RDE Training** (4-6 GPU hours) → `notebooks/Regime_Agent_Training.ipynb`
2. **M-RMS Training** (3-4 GPU hours) → `notebooks/train_mrms_agent.ipynb`  
3. **Main MARL Core** (8-10 GPU hours) → `notebooks/MARL_Training_Master_Colab.ipynb`

---

## 📋 Training Readiness Checklist

The training readiness system will verify:

### ✅ **Essential Requirements**
- [ ] **Dependencies**: PyTorch, NumPy, Pandas, Matplotlib
- [ ] **Data Files**: ES futures data (30-min and 5-min)
- [ ] **Preprocessing**: Data pipeline execution
- [ ] **Output Validation**: Training sequences created
- [ ] **GPU Check**: Hardware verification (optional but recommended)

### 📁 **Required Data Files**
- `ES  30 min  New.csv` - ES futures 30-minute data
- `ES  5 min.csv` - ES futures 5-minute data

### 🎯 **Expected Outputs**
- `./processed_data/sequences_train.npy` - Training sequences
- `./processed_data/sequences_val.npy` - Validation sequences
- `./processed_data/sequences_test.npy` - Test sequences
- `training_readiness_confirmed.json` - Readiness confirmation

---

## 🏗️ Architecture Overview

```
┌─────────────────────┐
│   ES Futures Data   │
│   (30min + 5min)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   RDE Training      │     │  M-RMS Training     │
│ (Regime Detection)  │     │ (Risk Management)   │
│                     │     │                     │
│ • Transformer+VAE   │     │ • 3 Sub-agents      │
│ • Unsupervised      │     │ • Sortino Optimize  │
│ • 4-6 GPU hours     │     │ • 3-4 GPU hours     │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │ Main MARL Core Train  │
           │  (Shared Policy)      │
           │                       │
           │ • Uses RDE & M-RMS    │
           │ • Two-Gate Decision   │
           │ • 8-10 GPU hours      │
           └───────────────────────┘
```

## 🔧 Training Environment Setup

### Google Colab (Recommended)
```python
# Enable GPU runtime: Runtime → Change runtime type → GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
```

### Local Environment
```bash
# Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## 📊 Training Data Requirements

### Data Format
- **Sequence Length**: 96 timesteps (represents ~48 hours of 30-min data)
- **Features**: 12 multi-modal features per timestep
- **Sample Count**: 1,000+ training sequences (more = better)

### Data Pipeline
1. **ES Futures Loading**: Read CSV files with OHLCV data
2. **Feature Engineering**: Create technical indicators and market features
3. **Sequence Creation**: Generate 96-timestep windows
4. **Train/Val/Test Split**: 70/15/15 split ratio
5. **Normalization**: Feature scaling for neural network training

## 🎯 Training Phases

### Phase 1: RDE Training
**Objective**: Train Transformer+VAE for regime detection

```python
# Key parameters
model_config = {
    'input_dim': 155,
    'hidden_dim': 256, 
    'latent_dim': 8,
    'num_heads': 8,
    'num_layers': 3
}
```

**Expected Output**: 8-dimensional regime vectors

### Phase 2: M-RMS Training  
**Objective**: Train risk management ensemble

```python
# 3 Sub-agents
agents = {
    'position_sizing': 'Optimal position size',
    'stop_loss': 'Dynamic stop loss levels', 
    'profit_target': 'Adaptive profit targets'
}
```

**Optimization**: Sortino ratio maximization

### Phase 3: Main MARL Core
**Objective**: Train shared policy with expert systems

```python
# Architecture components
components = {
    'SharedPolicy': 'Unified decision making',
    'DecisionGate': 'MC Dropout + Risk integration',
    'ExpertSystems': 'RDE + M-RMS integration'
}
```

**Result**: Production-ready trading system

## 📈 Performance Targets

### Training Metrics
- **RDE**: Stable latent representations, low reconstruction loss
- **M-RMS**: Sortino ratio > 1.5, consistent risk parameters
- **Main Core**: Validation accuracy > 60%, agent consensus > 70%

### Backtesting Targets
- **Sharpe Ratio**: > 1.5
- **Maximum Drawdown**: < 15%
- **Win Rate**: > 55%
- **Profit Factor**: > 1.3

## 🔍 Troubleshooting

### Common Issues

**❌ "Files not found"**
- Ensure ES futures CSV files are in the correct location
- Use the file upload feature in Colab
- Check file naming matches exactly: `ES  30 min  New.csv`

**❌ "PyTorch not installed"**
```bash
pip install torch torchvision
# Or in Colab:
!pip install torch torchvision
```

**❌ "Out of memory" during training**
- Reduce batch size in training notebooks
- Use gradient accumulation for effective larger batches
- Switch to CPU training (slower but functional)

**❌ "Preprocessing failed"**
- Check CSV file format and columns
- Verify data has numerical values
- Use fallback synthetic data for testing

### Performance Optimization

**🚀 Speed up training:**
- Use GPU runtime in Colab
- Increase batch size if memory allows  
- Enable mixed precision training
- Use larger learning rates with proper scheduling

**🎯 Improve results:**
- Increase training data volume
- Add more sophisticated features
- Tune hyperparameters systematically
- Use ensemble methods

## 📝 Training Logs

The system automatically creates:
- `training_readiness_confirmed.json` - Readiness status
- `./processed_data/data_preparation_metadata.json` - Data info
- `./models/` - Model checkpoints during training
- `./logs/training/` - Training progress logs

## 🚀 Next Steps After Training

1. **Model Evaluation**: Run backtesting notebooks
2. **Performance Analysis**: Analyze training metrics
3. **Hyperparameter Tuning**: Optimize based on results
4. **Production Deployment**: Prepare for live trading

---

## 🎉 Success Criteria

**✅ Training Complete When:**
- All 3 phases finished successfully
- Model checkpoints saved
- Validation metrics meet targets
- Integration tests pass
- Backtesting shows positive results

**🏁 Ready for Production When:**
- Consistent positive performance
- Risk management validated
- System integration tested
- Documentation complete

---

*Total Training Time: ~21-26 GPU hours*  
*Architecture: Single shared policy with expert system support*  
*Result: Production-ready AlgoSpace MARL trading system* 🎯