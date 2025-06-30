# Multi-Agent Risk Management Subsystem (M-RMS)

## Overview

The M-RMS is a sophisticated neural network ensemble trained with reinforcement learning to provide intelligent risk management decisions for trading opportunities. It consists of three specialized sub-agents that work together to determine optimal position sizing, stop loss placement, and profit targets.

## Architecture

### Sub-Agents

1. **PositionSizingAgent**: Determines the optimal number of contracts (0-5) based on:
   - Current market conditions
   - Account state and performance metrics
   - Risk tolerance parameters

2. **StopLossAgent**: Calculates dynamic stop loss placement using:
   - ATR-based multipliers (0.5x - 3.0x)
   - Market volatility adaptation
   - Risk minimization strategies

3. **ProfitTargetAgent**: Sets intelligent profit targets with:
   - Risk-reward ratios (1:1 - 5:1)
   - Win rate optimization
   - Market regime considerations

### Ensemble Coordination

The `RiskManagementEnsemble` coordinates all three sub-agents and includes a value function head for reinforcement learning optimization. The ensemble processes:
- **Synergy vectors** (30 features): Market conditions, indicators, and trade setup information
- **Account state vectors** (10 features): Balance, drawdown, performance metrics

## Integration

### Configuration (settings.yaml)

```yaml
m_rms:
  synergy_dim: 30         # Synergy feature vector dimension
  account_dim: 10         # Account state vector dimension  
  device: cpu             # Computing device
  point_value: 5.0        # Dollar value per point (MES)
  max_position_size: 5    # Maximum contracts per trade
```

### Usage Example

```python
from src.agents.mrms import MRMSComponent

# Initialize component
mrms = MRMSComponent(config['m_rms'])

# Load pre-trained model
mrms.load_model('./models/m_rms_model.pth')

# Generate risk proposal
trade_qualification = {
    'synergy_vector': np.array([...]),      # 30 features
    'account_state_vector': np.array([...]), # 10 features
    'entry_price': 4500.0,
    'direction': 'LONG',
    'atr': 10.0,
    'symbol': 'ES',
    'timestamp': pd.Timestamp.now()
}

risk_proposal = mrms.generate_risk_proposal(trade_qualification)
```

### Risk Proposal Output

```python
{
    'position_size': 3,              # Number of contracts
    'stop_loss_price': 4475.50,      # Calculated stop loss
    'take_profit_price': 4537.50,    # Calculated take profit
    'risk_amount': 375.00,           # Dollar risk
    'reward_amount': 937.50,         # Potential reward
    'risk_reward_ratio': 2.50,       # R:R ratio
    'sl_atr_multiplier': 2.45,       # Stop loss distance in ATRs
    'confidence_score': 0.782,       # Model confidence (0-1)
    'risk_metrics': {
        'sl_distance_points': 24.50,
        'tp_distance_points': 37.50,
        'risk_per_contract': 125.00,
        'max_position_allowed': 5,
        'position_utilization': 0.6
    }
}
```

## Training

The M-RMS was trained using:
- **Algorithm**: Multi-Agent PPO (Proximal Policy Optimization)
- **Environment**: Custom Gymnasium environment simulating TopStep-like evaluation rules
- **Reward Function**: Sortino ratio optimization with position sizing penalties
- **Data**: Historical synergy events with full trade simulation

## Files

- `models.py`: Pure PyTorch neural network definitions
- `engine.py`: High-level component interface
- `__init__.py`: Module exports

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Python >= 3.8