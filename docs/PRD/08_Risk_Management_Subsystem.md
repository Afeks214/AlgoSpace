# Product Requirements Document (PRD): Risk Management Sub-system (M-RMS)

**Document Version**: 2.0  
**Date**: 2025-06-19  
**Status**: Refined  
**Component**: Risk Management Sub-system (M-RMS)  

## 1.0 Overview

### 1.1 Purpose
The Risk Management Sub-system (M-RMS) serves as the system's intelligent "Chief Risk Officer" - a specialized MARL agent that optimizes position sizing, stop-loss placement, and take-profit targets based on trading opportunities and current account state. Unlike traditional rule-based risk management, the M-RMS uses reinforcement learning to develop sophisticated, context-aware risk policies that maximize risk-adjusted returns while preserving capital.

### 1.2 Scope

**In Scope:**
- MAPPO-based reinforcement learning agent for risk optimization
- Hybrid discrete/continuous action space for trade planning
- Sophisticated reward function based on Sortino ratio optimization
- Real-time risk proposal generation with sub-10ms latency
- Dedicated neural network embedders for synergy and account state processing
- Comprehensive training simulation environment
- Model optimization and TensorRT acceleration
- Rule-based constraint enforcement and penalty systems

**Out of Scope:**
- Trading opportunity detection (handled by Main MARL Core)
- Final execution decisions (handled by DecisionGate)
- Trade execution (handled by ExecutionHandler)
- Market data processing or feature generation
- Portfolio-level risk management across multiple strategies

### 1.3 Architectural Position
The M-RMS operates as a specialized agent within the MARL ecosystem:
Main MARL Core → **Risk Management Sub-system** → DecisionGate → ExecutionHandler

## 2.0 Functional Requirements

### FR-RMS-01: MAPPO-Based Agent Architecture
**Requirement**: The M-RMS MUST implement a complete MAPPO agent with specialized neural network architectures for risk management.

**Specification**:
```python
import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
import numpy as np

@dataclass
class SynergyVector:
    """Input vector containing trading opportunity details"""
    indicator_values: np.ndarray  # Raw indicator values that created the synergy
    time_since_signal: float      # Hours since signal formation
    signal_strength: float        # Confidence/strength of the signal
    timeframe: str               # "5min" or "30min"
    direction: str               # "LONG" or "SHORT"
    market_regime: np.ndarray    # Regime vector from RDE
    volatility_state: float     # Current market volatility level

@dataclass
class AccountState:
    """Current account health and risk metrics"""
    total_equity: float
    available_margin: float
    current_drawdown: float
    daily_pnl: float
    win_rate_30d: float
    avg_hold_time: float
    open_positions: int
    max_adverse_excursion: float
    consecutive_losses: int
    volatility_30d: float
    
@dataclass
class RiskProposal:
    """Output proposal for trade execution"""
    position_size: int
    stop_loss_price: float
    take_profit_price: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    confidence_score: float
    raw_actions: Dict[str, float]
    constraint_violations: List[str]

class RiskManagementModel(TorchModelV2, nn.Module):
    """Neural network model for the M-RMS agent"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.config = model_config.get("custom_model_config", {})
        
        # Input dimensions
        self.synergy_dim = self.config.get("synergy_dim", 32)
        self.account_dim = self.config.get("account_dim", 16)
        self.regime_dim = self.config.get("regime_dim", 8)
        
        # Synergy embedder (LSTM for temporal processing)
        self.synergy_embedder = nn.LSTM(
            input_size=self.synergy_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Account state processor
        self.account_processor = nn.Sequential(
            nn.Linear(self.account_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32)
        )
        
        # Regime context processor
        self.regime_processor = nn.Sequential(
            nn.Linear(self.regime_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Combined state processor
        combined_dim = 64 + 32 + 16  # synergy + account + regime
        self.state_processor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Action heads for hybrid action space
        # Discrete: position size (1-5 contracts)
        self.position_size_head = nn.Linear(64, 5)
        
        # Continuous: stop loss ATR multiplier (0.5-3.5)
        self.sl_atr_head = nn.Linear(64, 1)
        
        # Continuous: risk-reward ratio (1.0-5.0)
        self.rr_ratio_head = nn.Linear(64, 1)
        
        # Value function head
        self.value_head = nn.Linear(64, 1)
        
        # Internal state for recurrent processing
        self._last_hidden_state = None
        
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the risk management model"""
        
        obs = input_dict["obs"]
        
        # Split observation into components
        synergy_data = obs[:, :self.synergy_dim]
        account_data = obs[:, self.synergy_dim:self.synergy_dim + self.account_dim]
        regime_data = obs[:, -self.regime_dim:]
        
        # Process synergy data through LSTM
        # Reshape for LSTM: [batch, seq_len, features]
        synergy_reshaped = synergy_data.unsqueeze(1)  # Add sequence dimension
        synergy_embedded, hidden = self.synergy_embedder(synergy_reshaped)
        synergy_features = synergy_embedded[:, -1, :]  # Take last output
        
        # Process account state
        account_features = self.account_processor(account_data)
        
        # Process regime context
        regime_features = self.regime_processor(regime_data)
        
        # Combine all features
        combined_features = torch.cat([synergy_features, account_features, regime_features], dim=1)
        
        # Final state processing
        state_representation = self.state_processor(combined_features)
        
        # Generate action logits
        position_logits = self.position_size_head(state_representation)
        sl_atr_raw = self.sl_atr_head(state_representation)
        rr_ratio_raw = self.rr_ratio_head(state_representation)
        
        # Apply activation functions for continuous actions
        sl_atr_action = torch.sigmoid(sl_atr_raw) * 3.0 + 0.5  # [0.5, 3.5]
        rr_ratio_action = torch.sigmoid(rr_ratio_raw) * 4.0 + 1.0  # [1.0, 5.0]
        
        # Concatenate all action outputs
        action_logits = torch.cat([
            position_logits,
            sl_atr_action,
            rr_ratio_action
        ], dim=1)
        
        # Store value function output
        self._value_out = self.value_head(state_representation)
        
        return action_logits, state
        
    def value_function(self):
        """Return value function output"""
        return self._value_out.squeeze(-1)

class RiskManagementAgent:
    """Main M-RMS agent implementing MAPPO algorithm"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.current_price = None
        self.current_atr = None
        
        # Initialize Ray RLlib PPO algorithm
        self.ppo_config = (
            PPOConfig()
            .training(
                model={
                    "custom_model": "risk_management_model",
                    "custom_model_config": config.get("model_config", {})
                },
                lr=config.get("learning_rate", 3e-4),
                gamma=config.get("gamma", 0.95),
                lambda_=config.get("lambda", 0.9),
                kl_coeff=config.get("kl_coeff", 0.2),
                train_batch_size=config.get("train_batch_size", 4000),
                sgd_minibatch_size=config.get("sgd_minibatch_size", 128),
                num_sgd_iter=config.get("num_sgd_iter", 10)
            )
            .environment(
                env="risk_management_env",
                env_config=config.get("env_config", {})
            )
            .rollouts(
                num_rollout_workers=config.get("num_workers", 4),
                rollout_fragment_length=config.get("rollout_length", 200)
            )
        )
        
        # Create algorithm instance
        self.algorithm = self.ppo_config.build()
        
        # Load trained model if available
        if "model_checkpoint" in config:
            self.algorithm.restore(config["model_checkpoint"])
            
    def get_risk_plan(self, synergy_vector: SynergyVector, 
                     account_state: AccountState,
                     current_price: float,
                     current_atr: float) -> RiskProposal:
        """Generate risk management proposal for trading opportunity"""
        
        self.current_price = current_price
        self.current_atr = current_atr
        
        # Prepare observation
        obs = self._prepare_observation(synergy_vector, account_state)
        
        # Get action from trained policy
        action = self.algorithm.compute_single_action(obs)
        
        # Convert action to risk proposal
        proposal = self._action_to_proposal(action, synergy_vector, account_state)
        
        # Validate against constraints
        proposal = self._apply_constraints(proposal, account_state)
        
        return proposal
        
    def _prepare_observation(self, synergy_vector: SynergyVector, 
                           account_state: AccountState) -> np.ndarray:
        """Convert inputs to neural network observation format"""
        
        # Normalize synergy features
        synergy_features = np.concatenate([
            synergy_vector.indicator_values,
            [synergy_vector.time_since_signal / 24.0],  # Normalize hours
            [synergy_vector.signal_strength],
            [1.0 if synergy_vector.timeframe == "30min" else 0.0],
            [1.0 if synergy_vector.direction == "LONG" else -1.0],
            [synergy_vector.volatility_state]
        ])
        
        # Normalize account features
        account_features = np.array([
            account_state.current_drawdown,
            account_state.daily_pnl / max(account_state.total_equity, 1.0),
            account_state.win_rate_30d,
            account_state.avg_hold_time / 24.0,  # Normalize hours
            min(account_state.open_positions / 5.0, 1.0),  # Cap at 5 positions
            account_state.max_adverse_excursion,
            min(account_state.consecutive_losses / 10.0, 1.0),  # Cap at 10
            account_state.volatility_30d,
            account_state.available_margin / max(account_state.total_equity, 1.0)
        ])
        
        # Combine with regime vector
        observation = np.concatenate([
            synergy_features,
            account_features,
            synergy_vector.market_regime
        ])
        
        return observation.astype(np.float32)
        
    def _action_to_proposal(self, action: np.ndarray, 
                          synergy_vector: SynergyVector,
                          account_state: AccountState) -> RiskProposal:
        """Convert raw action values to concrete risk proposal"""
        
        # Extract action components
        position_size_idx = int(action[0])
        sl_atr_multiplier = float(action[1])
        rr_ratio = float(action[2])
        
        # Convert discrete position size (1-5 contracts)
        position_size = position_size_idx + 1
        
        # Calculate stop loss price
        if synergy_vector.direction == "LONG":
            stop_loss_price = self.current_price - (sl_atr_multiplier * self.current_atr)
            take_profit_price = self.current_price + (rr_ratio * sl_atr_multiplier * self.current_atr)
        else:  # SHORT
            stop_loss_price = self.current_price + (sl_atr_multiplier * self.current_atr)
            take_profit_price = self.current_price - (rr_ratio * sl_atr_multiplier * self.current_atr)
            
        # Calculate risk and reward amounts
        risk_per_contract = abs(self.current_price - stop_loss_price)
        reward_per_contract = abs(take_profit_price - self.current_price)
        
        risk_amount = risk_per_contract * position_size
        reward_amount = reward_per_contract * position_size
        
        # Calculate confidence score based on signal strength and account state
        confidence_score = self._calculate_confidence(synergy_vector, account_state, position_size)
        
        return RiskProposal(
            position_size=position_size,
            stop_loss_price=round(stop_loss_price, 2),
            take_profit_price=round(take_profit_price, 2),
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=rr_ratio,
            confidence_score=confidence_score,
            raw_actions={
                "position_size": position_size,
                "sl_atr_multiplier": sl_atr_multiplier,
                "rr_ratio": rr_ratio
            },
            constraint_violations=[]
        )
        
    def _apply_constraints(self, proposal: RiskProposal, 
                         account_state: AccountState) -> RiskProposal:
        """Apply hard constraints and rules validation"""
        
        violations = []
        
        # Maximum position size constraint
        max_position_size = self.config.get("max_position_size", 5)
        if proposal.position_size > max_position_size:
            proposal.position_size = max_position_size
            violations.append(f"Position size capped at {max_position_size}")
            
        # Maximum risk per trade (% of account)
        max_risk_pct = self.config.get("max_risk_per_trade", 0.02)
        max_risk_amount = account_state.total_equity * max_risk_pct
        
        if proposal.risk_amount > max_risk_amount:
            # Scale down position size
            scale_factor = max_risk_amount / proposal.risk_amount
            proposal.position_size = max(1, int(proposal.position_size * scale_factor))
            proposal.risk_amount = proposal.risk_amount * scale_factor
            proposal.reward_amount = proposal.reward_amount * scale_factor
            violations.append(f"Position sized scaled for max risk: {max_risk_pct*100:.1f}%")
            
        # Maximum daily drawdown check
        max_daily_dd = self.config.get("max_daily_drawdown", 0.05)
        potential_dd = (account_state.daily_pnl + proposal.risk_amount) / account_state.total_equity
        
        if potential_dd < -max_daily_dd:
            # Reduce position size or reject
            safe_risk = account_state.total_equity * max_daily_dd + account_state.daily_pnl
            if safe_risk > 0:
                scale_factor = safe_risk / proposal.risk_amount
                proposal.position_size = max(1, int(proposal.position_size * scale_factor))
                violations.append("Position reduced for daily drawdown limit")
            else:
                proposal.position_size = 0
                violations.append("Trade rejected: daily drawdown limit exceeded")
                
        # Available margin check
        required_margin = proposal.position_size * self.current_price * 0.05  # 5% margin requirement
        if required_margin > account_state.available_margin:
            max_contracts = int(account_state.available_margin / (self.current_price * 0.05))
            proposal.position_size = max(0, max_contracts)
            violations.append("Position size reduced for margin requirements")
            
        proposal.constraint_violations = violations
        return proposal
        
    def _calculate_confidence(self, synergy_vector: SynergyVector,
                            account_state: AccountState,
                            position_size: int) -> float:
        """Calculate confidence score for the risk proposal"""
        
        # Base confidence from signal strength
        confidence = synergy_vector.signal_strength
        
        # Adjust for account state
        if account_state.current_drawdown > 0.1:  # In drawdown
            confidence *= 0.8
        elif account_state.win_rate_30d > 0.6:  # Recent good performance
            confidence *= 1.1
            
        # Adjust for position size (larger positions = lower confidence)
        size_penalty = 1.0 - (position_size - 1) * 0.1
        confidence *= size_penalty
        
        # Adjust for market regime volatility
        if synergy_vector.volatility_state > 0.8:  # High volatility
            confidence *= 0.9
            
        return min(1.0, max(0.0, confidence))
```

### FR-RMS-02: Training Environment and Simulation
**Requirement**: The M-RMS MUST have a comprehensive training environment that simulates realistic trading scenarios.

**Specification**:
```python
import gym
from gym import spaces
from typing import Tuple
import pandas as pd

class RiskManagementEnvironment(gym.Env):
    """Training environment for the Risk Management Sub-system"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.historical_data = self._load_historical_data(config["data_path"])
        self.synergy_events = self._load_synergy_events(config["synergy_path"])
        
        # Environment state
        self.current_step = 0
        self.account = VirtualAccount(
            initial_balance=config.get("initial_balance", 100000),
            max_drawdown=config.get("max_drawdown", 0.20)
        )
        
        # Define observation and action spaces
        obs_dim = config.get("synergy_dim", 32) + config.get("account_dim", 16) + config.get("regime_dim", 8)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Hybrid action space: [position_size(discrete), sl_atr(continuous), rr_ratio(continuous)]
        self.action_space = spaces.Box(
            low=np.array([0, 0.5, 1.0]),
            high=np.array([4, 3.5, 5.0]),
            dtype=np.float32
        )
        
        # Training metrics
        self.episode_metrics = {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sortino_ratio": 0.0,
            "num_trades": 0,
            "win_rate": 0.0
        }
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        
        # Reset account
        self.account.reset()
        
        # Select random starting point
        self.current_step = np.random.randint(
            1000, len(self.synergy_events) - 1000
        )
        
        # Reset metrics
        self.episode_metrics = {key: 0.0 for key in self.episode_metrics}
        
        return self._get_observation()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step"""
        
        # Get current synergy event
        synergy_event = self.synergy_events.iloc[self.current_step]
        
        # Convert action to risk proposal
        proposal = self._action_to_proposal(action, synergy_event)
        
        # Simulate trade execution
        trade_result = self._simulate_trade(proposal, synergy_event)
        
        # Update account state
        self.account.process_trade(trade_result)
        
        # Calculate reward
        reward = self._calculate_reward(trade_result, proposal)
        
        # Update metrics
        self._update_metrics(trade_result)
        
        # Check if episode is done
        done = self._check_episode_end()
        
        # Move to next step
        self.current_step += 1
        
        # Prepare next observation
        next_obs = self._get_observation()
        
        info = {
            "trade_result": trade_result,
            "account_state": self.account.get_state(),
            "episode_metrics": self.episode_metrics.copy()
        }
        
        return next_obs, reward, done, info
        
    def _calculate_reward(self, trade_result: Dict, proposal: RiskProposal) -> float:
        """Calculate reward based on Sortino ratio and rule compliance"""
        
        # Base reward from trade PnL impact on Sortino ratio
        account_returns = self.account.get_returns_series()
        
        if len(account_returns) > 30:  # Need sufficient history
            sortino_before = self._calculate_sortino_ratio(account_returns[:-1])
            sortino_after = self._calculate_sortino_ratio(account_returns)
            sortino_improvement = sortino_after - sortino_before
            
            # Scale the improvement for reasonable reward magnitude
            base_reward = sortino_improvement * 10.0
        else:
            # Simple PnL-based reward for early episodes
            base_reward = trade_result["pnl_net"] / 1000.0  # Normalize
            
        # Penalty for constraint violations
        violation_penalty = -5.0 * len(proposal.constraint_violations)
        
        # Penalty for excessive risk
        risk_penalty = 0.0
        if proposal.risk_amount / self.account.total_equity > 0.03:  # >3% risk
            risk_penalty = -2.0
            
        # Bonus for good risk-reward ratios
        rr_bonus = 0.0
        if proposal.risk_reward_ratio >= 2.0:
            rr_bonus = 0.5
            
        # Penalty for account rule violations
        rule_penalty = 0.0
        if self.account.current_drawdown > self.config.get("max_drawdown", 0.20):
            rule_penalty = -10.0  # Large penalty for breaking risk rules
            
        total_reward = base_reward + violation_penalty + risk_penalty + rr_bonus + rule_penalty
        
        return float(total_reward)
        
    def _simulate_trade(self, proposal: RiskProposal, synergy_event: pd.Series) -> Dict:
        """Simulate trade execution and outcome"""
        
        if proposal.position_size == 0:
            return {
                "executed": False,
                "pnl_gross": 0.0,
                "pnl_net": 0.0,
                "exit_reason": "rejected"
            }
            
        # Get market data for trade simulation
        entry_time = synergy_event["timestamp"]
        entry_price = synergy_event["entry_price"]
        
        # Find exit point (SL or TP hit)
        future_data = self.historical_data[
            self.historical_data.index > entry_time
        ].head(100)  # Look ahead up to 100 bars
        
        direction = synergy_event["direction"]
        sl_price = proposal.stop_loss_price
        tp_price = proposal.take_profit_price
        
        exit_price = None
        exit_reason = "timeout"
        
        for timestamp, bar in future_data.iterrows():
            if direction == "LONG":
                if bar["low"] <= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                    break
                elif bar["high"] >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                    break
            else:  # SHORT
                if bar["high"] >= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                    break
                elif bar["low"] <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                    break
                    
        # If no exit found, use timeout exit
        if exit_price is None:
            exit_price = future_data.iloc[-1]["close"]
            
        # Calculate PnL
        if direction == "LONG":
            pnl_gross = (exit_price - entry_price) * proposal.position_size
        else:
            pnl_gross = (entry_price - exit_price) * proposal.position_size
            
        # Subtract commissions and slippage
        commission = proposal.position_size * 2.5 * 2  # Round trip
        slippage = proposal.position_size * 0.25 * 2  # 1 tick slippage each way
        pnl_net = pnl_gross - commission - slippage
        
        return {
            "executed": True,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": proposal.position_size,
            "direction": direction,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "exit_reason": exit_reason,
            "commission": commission,
            "slippage": slippage
        }
        
    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio for return series"""
        
        if len(returns) < 10:
            return 0.0
            
        excess_returns = returns - target_return
        mean_return = np.mean(excess_returns)
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 5.0  # High ratio if no downside
            
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return 5.0
            
        sortino_ratio = mean_return / downside_deviation
        return sortino_ratio

class VirtualAccount:
    """Virtual trading account for M-RMS training"""
    
    def __init__(self, initial_balance: float, max_drawdown: float = 0.20):
        self.initial_balance = initial_balance
        self.max_drawdown = max_drawdown
        self.reset()
        
    def reset(self):
        """Reset account to initial state"""
        self.total_equity = self.initial_balance
        self.cash_balance = self.initial_balance
        self.peak_equity = self.initial_balance
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.trade_history = []
        self.returns_series = []
        
    def process_trade(self, trade_result: Dict):
        """Process completed trade and update account"""
        
        if not trade_result["executed"]:
            return
            
        # Update equity
        pnl_net = trade_result["pnl_net"]
        self.total_equity += pnl_net
        self.cash_balance += pnl_net
        self.daily_pnl += pnl_net
        
        # Update peak and drawdown
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.total_equity) / self.peak_equity
            
        # Store trade
        self.trade_history.append(trade_result)
        
        # Calculate return for this trade
        if len(self.trade_history) > 0:
            trade_return = pnl_net / self.initial_balance
            self.returns_series.append(trade_return)
            
    def get_state(self) -> AccountState:
        """Get current account state for observation"""
        
        # Calculate statistics from recent trades
        recent_trades = self.trade_history[-30:] if len(self.trade_history) >= 30 else self.trade_history
        
        if recent_trades:
            wins = [t for t in recent_trades if t["pnl_net"] > 0]
            win_rate = len(wins) / len(recent_trades)
            
            # Calculate consecutive losses
            consecutive_losses = 0
            for trade in reversed(recent_trades):
                if trade["pnl_net"] < 0:
                    consecutive_losses += 1
                else:
                    break
                    
            # Calculate average hold time (mock for simulation)
            avg_hold_time = 4.0  # Assume 4 hours average
            
            # Calculate MAE from recent trades
            mae_values = [abs(t.get("max_adverse_excursion", 0)) for t in recent_trades]
            max_adverse_excursion = max(mae_values) if mae_values else 0.0
        else:
            win_rate = 0.0
            consecutive_losses = 0
            avg_hold_time = 0.0
            max_adverse_excursion = 0.0
            
        # Calculate 30-day volatility
        if len(self.returns_series) >= 10:
            volatility_30d = np.std(self.returns_series[-30:])
        else:
            volatility_30d = 0.0
            
        return AccountState(
            total_equity=self.total_equity,
            available_margin=self.cash_balance * 0.8,  # 80% of cash available for margin
            current_drawdown=self.current_drawdown,
            daily_pnl=self.daily_pnl,
            win_rate_30d=win_rate,
            avg_hold_time=avg_hold_time,
            open_positions=0,  # Simplified: no overlapping positions in training
            max_adverse_excursion=max_adverse_excursion,
            consecutive_losses=consecutive_losses,
            volatility_30d=volatility_30d
        )
        
    def get_returns_series(self) -> np.ndarray:
        """Get returns series for Sortino calculation"""
        return np.array(self.returns_series)
```

### FR-RMS-03: Production Inference and Optimization
**Requirement**: The M-RMS MUST provide optimized production inference with TensorRT acceleration.

**Specification**:
```python
class ProductionRiskManager:
    """Production-optimized risk management system"""
    
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = self._load_model(model_path)
        
        # Performance metrics
        self.inference_stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "error_count": 0,
            "avg_latency_ms": 0.0
        }
        
        # Risk constraints
        self.risk_constraints = RiskConstraints(config.get("constraints", {}))
        
    def _load_model(self, model_path: str):
        """Load and optimize model for production"""
        
        if model_path.endswith('.engine'):
            # Load TensorRT engine
            return self._load_tensorrt_model(model_path)
        else:
            # Load PyTorch model
            return self._load_pytorch_model(model_path)
            
    def generate_risk_proposal(self, synergy_vector: SynergyVector,
                             account_state: AccountState,
                             current_price: float,
                             current_atr: float) -> RiskProposal:
        """Generate optimized risk proposal with constraints"""
        
        start_time = time.perf_counter()
        
        try:
            # Prepare model input
            obs = self._prepare_observation(synergy_vector, account_state)
            
            # Model inference
            with torch.no_grad():
                if hasattr(self.model, 'compute_single_action'):
                    # Ray RLlib model
                    action = self.model.compute_single_action(obs)
                else:
                    # Raw PyTorch model
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    action_logits, _ = self.model({"obs": obs_tensor}, [], None)
                    action = self._sample_action(action_logits)
                    
            # Convert to risk proposal
            proposal = self._create_proposal(action, synergy_vector, current_price, current_atr)
            
            # Apply production constraints
            proposal = self.risk_constraints.apply_constraints(proposal, account_state)
            
            # Update performance metrics
            inference_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(inference_time)
            
            return proposal
            
        except Exception as e:
            logger.error(f"Risk management inference error: {e}")
            self.inference_stats["error_count"] += 1
            
            # Return conservative fallback proposal
            return self._create_fallback_proposal(synergy_vector, current_price, current_atr)
            
    def _create_fallback_proposal(self, synergy_vector: SynergyVector,
                                current_price: float,
                                current_atr: float) -> RiskProposal:
        """Create conservative fallback proposal on errors"""
        
        # Conservative parameters
        position_size = 1  # Minimum position
        sl_distance = 2.0 * current_atr  # 2 ATR stop loss
        rr_ratio = 2.0  # 1:2 risk reward
        
        if synergy_vector.direction == "LONG":
            stop_loss_price = current_price - sl_distance
            take_profit_price = current_price + (sl_distance * rr_ratio)
        else:
            stop_loss_price = current_price + sl_distance
            take_profit_price = current_price - (sl_distance * rr_ratio)
            
        return RiskProposal(
            position_size=position_size,
            stop_loss_price=round(stop_loss_price, 2),
            take_profit_price=round(take_profit_price, 2),
            risk_amount=sl_distance * position_size,
            reward_amount=sl_distance * rr_ratio * position_size,
            risk_reward_ratio=rr_ratio,
            confidence_score=0.3,  # Low confidence for fallback
            raw_actions={
                "position_size": position_size,
                "sl_atr_multiplier": 2.0,
                "rr_ratio": rr_ratio
            },
            constraint_violations=["fallback_proposal"]
        )
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get inference performance statistics"""
        return self.inference_stats.copy()

class RiskConstraints:
    """Production risk constraint enforcement"""
    
    def __init__(self, config: Dict):
        self.max_position_size = config.get("max_position_size", 5)
        self.max_risk_per_trade = config.get("max_risk_per_trade", 0.02)
        self.max_daily_risk = config.get("max_daily_risk", 0.05)
        self.max_account_drawdown = config.get("max_account_drawdown", 0.15)
        self.min_rr_ratio = config.get("min_rr_ratio", 1.0)
        self.max_open_positions = config.get("max_open_positions", 3)
        
    def apply_constraints(self, proposal: RiskProposal, 
                        account_state: AccountState) -> RiskProposal:
        """Apply all production risk constraints"""
        
        violations = []
        
        # Position size constraints
        if proposal.position_size > self.max_position_size:
            proposal.position_size = self.max_position_size
            violations.append(f"Position size capped at {self.max_position_size}")
            
        # Risk per trade constraint
        max_risk_amount = account_state.total_equity * self.max_risk_per_trade
        if proposal.risk_amount > max_risk_amount:
            scale_factor = max_risk_amount / proposal.risk_amount
            proposal.position_size = max(1, int(proposal.position_size * scale_factor))
            violations.append("Position reduced for per-trade risk limit")
            
        # Daily risk constraint
        daily_risk_used = abs(account_state.daily_pnl) if account_state.daily_pnl < 0 else 0
        remaining_daily_risk = (account_state.total_equity * self.max_daily_risk) - daily_risk_used
        
        if proposal.risk_amount > remaining_daily_risk:
            if remaining_daily_risk > 0:
                scale_factor = remaining_daily_risk / proposal.risk_amount
                proposal.position_size = max(1, int(proposal.position_size * scale_factor))
                violations.append("Position reduced for daily risk limit")
            else:
                proposal.position_size = 0
                violations.append("Trade rejected: daily risk limit exceeded")
                
        # Account drawdown constraint
        if account_state.current_drawdown > self.max_account_drawdown:
            proposal.position_size = 0
            violations.append("Trade rejected: account drawdown limit exceeded")
            
        # Minimum risk-reward ratio
        if proposal.risk_reward_ratio < self.min_rr_ratio:
            proposal.position_size = 0
            violations.append(f"Trade rejected: R:R ratio below minimum {self.min_rr_ratio}")
            
        # Maximum open positions
        if account_state.open_positions >= self.max_open_positions:
            proposal.position_size = 0
            violations.append("Trade rejected: maximum open positions reached")
            
        proposal.constraint_violations.extend(violations)
        return proposal
```

## 3.0 Interface Specifications

### 3.1 Configuration Interface
```yaml
risk_management:
  model:
    synergy_dim: 32
    account_dim: 16
    regime_dim: 8
    lstm_hidden_size: 64
    lstm_layers: 2
    
  training:
    algorithm: "MAPPO"
    learning_rate: 0.0003
    gamma: 0.95
    lambda: 0.9
    train_batch_size: 4000
    num_workers: 4
    num_epochs: 1000
    
  constraints:
    max_position_size: 5
    max_risk_per_trade: 0.02
    max_daily_risk: 0.05
    max_account_drawdown: 0.15
    min_rr_ratio: 1.0
    max_open_positions: 3
    
  reward:
    sortino_weight: 1.0
    violation_penalty: 5.0
    rr_bonus_threshold: 2.0
    
  deployment:
    model_path: "models/risk_model.pth"
    enable_tensorrt: true
    fallback_enabled: true
    max_inference_latency_ms: 10
```

### 3.2 API Interface
```python
class RiskManagementAPI:
    def get_risk_plan(self, synergy_vector: SynergyVector, 
                     account_state: AccountState,
                     current_price: float, 
                     current_atr: float) -> RiskProposal
    
    def validate_proposal(self, proposal: RiskProposal, 
                         account_state: AccountState) -> bool
    
    def get_performance_metrics(self) -> Dict[str, float]
    
    def update_constraints(self, new_constraints: Dict) -> None
```

## 4.0 Dependencies & Interactions

### 4.1 Upstream Dependencies
- **Main MARL Core**: Source of synergy vectors and trading opportunities
- **Account Management**: Current account state and position information
- **Market Data**: Current price and ATR for calculation

### 4.2 Downstream Dependencies
- **DecisionGate**: Consumer of risk proposals for final execution decisions
- **Performance Monitor**: Consumer of risk metrics and constraint violations
- **ExecutionHandler**: Consumer of validated risk proposals

## 5.0 Non-Functional Requirements

### 5.1 Performance
- **NFR-RMS-01**: Risk proposal generation MUST complete under 10ms (95th percentile)
- **NFR-RMS-02**: Model MUST support concurrent inference requests
- **NFR-RMS-03**: Memory usage MUST remain stable during continuous operation
- **NFR-RMS-04**: TensorRT optimization MUST achieve 3x+ speedup over PyTorch

### 5.2 Accuracy and Robustness
- **NFR-RMS-05**: Model MUST achieve Sortino ratio > 1.5 in validation
- **NFR-RMS-06**: Constraint violations MUST be prevented 100% of the time
- **NFR-RMS-07**: Model MUST provide graceful degradation on inference errors
- **NFR-RMS-08**: Training MUST converge within 1000 episodes

### 5.3 Risk Controls
- **NFR-RMS-09**: All risk limits MUST be enforced at the code level
- **NFR-RMS-10**: Constraint violations MUST trigger immediate alerts
- **NFR-RMS-11**: Model outputs MUST be validated before use

## 6.0 Testing Requirements

### 6.1 Unit Tests
- Individual constraint validation functions
- Action space conversion and validation
- Reward function calculation accuracy
- Observation preprocessing and normalization

### 6.2 Integration Tests
- End-to-end training pipeline validation
- Production inference integration
- Constraint enforcement under edge cases
- Fallback proposal generation

### 6.3 Backtesting Validation
```python
def test_risk_management_performance():
    """Validate M-RMS performance on historical data"""
    risk_manager = ProductionRiskManager(model_path, config)
    
    # Load historical synergy events
    test_data = load_historical_synergies()
    
    account = VirtualAccount(100000)
    sortino_ratios = []
    
    for synergy in test_data:
        proposal = risk_manager.generate_risk_proposal(
            synergy.vector, account.get_state(), 
            synergy.price, synergy.atr
        )
        
        # Simulate trade outcome
        trade_result = simulate_trade(proposal, synergy)
        account.process_trade(trade_result)
        
        # Calculate rolling Sortino ratio
        if len(account.returns_series) >= 30:
            sortino = calculate_sortino_ratio(account.returns_series[-30:])
            sortino_ratios.append(sortino)
    
    # Validate performance
    avg_sortino = np.mean(sortino_ratios)
    max_dd = account.current_drawdown
    
    assert avg_sortino > 1.5, f"Sortino ratio too low: {avg_sortino}"
    assert max_dd < 0.15, f"Maximum drawdown exceeded: {max_dd}"
```

## 7.0 Future Enhancements

### 7.1 V2.0 Features
- **Dynamic Exit Strategies**: ML-optimized trailing stops and exit rules
- **Multi-Asset Support**: Portfolio-level risk management across instruments
- **Regime-Adaptive Models**: Different risk models for different market regimes
- **Online Learning**: Continuous model adaptation based on recent performance
- **Advanced Metrics**: Kelly criterion and other sophisticated position sizing

### 7.2 Research Directions
- **Hierarchical Risk Management**: Multi-level risk controls (trade, daily, monthly)
- **Adversarial Training**: Robustness against market regime changes
- **Uncertainty Quantification**: Confidence intervals for risk proposals
- **Causal Risk Modeling**: Understanding risk factor relationships

This PRD establishes the foundation for an advanced, learning-based risk management system that goes beyond traditional rule-based approaches to provide intelligent, adaptive position sizing and risk control while maintaining strict safety constraints for capital preservation.