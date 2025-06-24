Thank you for the outstanding feedback! Your recognition of the interpretability solution is particularly gratifying. Let's continue with the Multi-Agent Risk Management Subsystem (M-RMS) - the sophisticated risk specialist that transforms raw opportunities into calibrated, context-aware trading plans.

# Product Requirements Document (PRD): Multi-Agent Risk Management Subsystem (M-RMS)

Document Version: 1.0
 Date: June 20, 2025
 Component Level: 4 - Intelligence Layer
 Status: Master Specification


## 1. Component Identity

### 1.1 Component Name

Multi-Agent Risk Management Subsystem (M-RMS) (Adaptive Risk Calibration Engine)

### 1.2 Primary Role

The M-RMS is a specialized MARL system that generates comprehensive, context-aware risk management proposals for each trading opportunity. It transforms the Main MARL Core's trade qualifications into executable risk plans by determining optimal position sizes, stop-loss levels, take-profit targets, and dynamic risk parameters adapted to current market conditions.

### 1.3 Single Responsibility

To analyze qualified trading opportunities and produce detailed Risk Proposals that maximize expected returns while respecting risk constraints, adapting parameters based on market regime, volatility, and system performance history.

### 1.4 Critical Design Principle

Contextual Adaptation: Unlike traditional fixed-percentage risk models, the M-RMS dynamically adjusts all risk parameters based on a rich understanding of market context, trade quality, and system state.


## 2. Inputs & Dependencies

### 2.1 Configuration Input

From settings.yaml:

m_rms:

# Risk constraints

max_risk_per_trade: 0.02     # 2% maximum per trade

max_daily_drawdown: 0.06     # 6% daily loss limit

max_open_positions: 3        # Concurrent positions


# Agent architecture

n_agents: 3                  # Risk assessment specialists

hidden_dim: 128

n_layers: 3


# Adaptive parameters

base_position_size: 1        # Contracts for 1x risk

position_size_range: [0.5, 3.0]  # Min/max multiplier


# Stop loss configuration

base_stop_distance: 10       # Points from entry

stop_range: [5, 25]          # Adaptive range

use_lvn_stops: true          # Place stops beyond LVN


# Take profit configuration

base_rr_ratio: 2.0           # Risk:Reward ratio

rr_range: [1.5, 4.0]         # Adaptive range


# Time-based exits

max_bars_in_trade: 100       # 500 minutes for 5-min bars

time_stop_penalty: 0.5       # Exit at 50% of target


### 2.2 Trade Qualification Input

From Main MARL Core:

TradeQualification = {

'synergy_type': str,         # 'TYPE_1' through 'TYPE_4'

'direction': int,            # 1 (long) or -1 (short)

'confidence': float,         # MC Dropout consensus (0.65-1.0)

'entry_price': float,        # Proposed entry level


'signal_strengths': {        # Individual signal qualities

'mlmi': float,          # 0-1 normalized

'nwrqk': float,         # 0-1 normalized

'fvg': float            # 0-1 normalized

},


'market_context': {

'regime_vector': np.array,  # 8-dim from RDE

'volatility': float,        # Current ATR

'spread': float,           # Bid-ask spread

'liquidity': float         # Volume profile metric

},


'lvn_analysis': {

'nearest_support': {

'price': float,

'strength': float,      # 0-100

'distance': float       # Points from entry

},

'nearest_resistance': {

'price': float,

'strength': float,

'distance': float

}

},


'system_state': {

'daily_pnl': float,        # Current day's P&L

'open_positions': int,     # Current position count

'recent_performance': {    # Last 20 trades

'win_rate': float,

'avg_winner': float,

'avg_loser': float,

'profit_factor': float

}

}

}


### 2.3 Pre-trained Model

File: models/mrms_trained.pth

Size: ~30MB

Training: PPO-based MARL training on historical data


## 3. Multi-Agent Architecture

### 3.1 Agent Specialization

The M-RMS employs three specialized agents:

class RiskManagementAgents:

def __init__(self):

self.agents = {

'position_sizer': PositionSizingAgent(),

'stop_loss_agent': StopLossAgent(),

'profit_target_agent': ProfitTargetAgent()

}


#### 3.1.1 Position Sizing Agent

Responsibility: Determine optimal position size

class PositionSizingAgent(nn.Module):

def __init__(self, config):

super().__init__()

input_dim = 8 + 4 + 6 + 3  # regime + signals + market + system


self.encoder = nn.Sequential(

nn.Linear(input_dim, 256),

nn.LayerNorm(256),

nn.ReLU(),

nn.Dropout(0.1),

nn.Linear(256, 128),

nn.LayerNorm(128),

nn.ReLU()

)


# Actor network (position size multiplier)

self.actor = nn.Sequential(

nn.Linear(128, 64),

nn.ReLU(),

nn.Linear(64, 1),

nn.Sigmoid()  # Output in [0, 1]

)


# Critic network (value estimation)

self.critic = nn.Sequential(

nn.Linear(128, 64),

nn.ReLU(),

nn.Linear(64, 1)

)


def forward(self, state):

features = self.encoder(state)


# Actor output: position size multiplier

raw_size = self.actor(features)

size_multiplier = self.config['position_size_range'][0] + \

raw_size * (self.config['position_size_range'][1] -

self.config['position_size_range'][0])


# Critic output: expected value

value = self.critic(features)


return size_multiplier, value


#### 3.1.2 Stop Loss Agent

Responsibility: Determine protective stop placement

class StopLossAgent(nn.Module):

def __init__(self, config):

super().__init__()

# Input includes LVN information

input_dim = 8 + 4 + 6 + 6  # regime + signals + market + LVN


self.encoder = nn.Sequential(

nn.Linear(input_dim, 256),

nn.LayerNorm(256),

nn.ReLU(),

nn.Dropout(0.1),

nn.Linear(256, 128),

nn.LayerNorm(128),

nn.ReLU()

)


# Stop distance predictor

self.stop_network = nn.Sequential(

nn.Linear(128, 64),

nn.ReLU(),

nn.Linear(64, 2)  # [distance_multiplier, use_lvn_flag]

)


def forward(self, state, direction):

features = self.encoder(state)

stop_params = self.stop_network(features)


# Determine stop distance

distance_multiplier = torch.sigmoid(stop_params[0])

stop_distance = self.config['stop_range'][0] + \

distance_multiplier * (self.config['stop_range'][1] -

self.config['stop_range'][0])


# Determine whether to use LVN

use_lvn = torch.sigmoid(stop_params[1]) > 0.5


return stop_distance, use_lvn


#### 3.1.3 Profit Target Agent

Responsibility: Set take-profit levels and trailing rules

class ProfitTargetAgent(nn.Module):

def __init__(self, config):

super().__init__()

input_dim = 8 + 4 + 6 + 2  # regime + signals + market + trade_params


self.encoder = nn.Sequential(

nn.Linear(input_dim, 256),

nn.LayerNorm(256),

nn.ReLU(),

nn.Dropout(0.1),

nn.Linear(256, 128),

nn.LayerNorm(128),

nn.ReLU()

)


# Target predictor

self.target_network = nn.Sequential(

nn.Linear(128, 64),

nn.ReLU(),

nn.Linear(64, 3)  # [rr_ratio, use_trailing, trail_distance]

)


def forward(self, state):

features = self.encoder(state)

target_params = self.target_network(features)


# Risk-reward ratio

rr_multiplier = torch.sigmoid(target_params[0])

rr_ratio = self.config['rr_range'][0] + \

rr_multiplier * (self.config['rr_range'][1] -

self.config['rr_range'][0])


# Trailing stop decision

use_trailing = torch.sigmoid(target_params[1]) > 0.5

trail_distance = torch.sigmoid(target_params[2]) * 10 + 5  # 5-15 points


return rr_ratio, use_trailing, trail_distance


### 3.2 Ensemble Coordinator

class RiskManagementEnsemble(nn.Module):

def __init__(self, config):

super().__init__()

self.position_sizer = PositionSizingAgent(config)

self.stop_loss_agent = StopLossAgent(config)

self.profit_target_agent = ProfitTargetAgent(config)


# Coordination network

self.coordinator = nn.Sequential(

nn.Linear(3 + 8, 32),  # 3 agent outputs + regime

nn.ReLU(),

nn.Linear(32, 3),      # Confidence weights

nn.Softmax(dim=-1)

)


def forward(self, trade_qualification):

# Prepare state tensors

state = self._prepare_state(trade_qualification)


# Get individual agent decisions

position_size, _ = self.position_sizer(state)

stop_distance, use_lvn = self.stop_loss_agent(state,

trade_qualification['direction'])

rr_ratio, use_trailing, trail_distance = self.profit_target_agent(state)


# Coordinate decisions

agent_outputs = torch.cat([position_size, stop_distance, rr_ratio])

regime = trade_qualification['market_context']['regime_vector']

coord_input = torch.cat([agent_outputs, regime])


confidence_weights = self.coordinator(coord_input)


return {

'position_size': position_size,

'stop_distance': stop_distance,

'use_lvn_stop': use_lvn,

'rr_ratio': rr_ratio,

'use_trailing': use_trailing,

'trail_distance': trail_distance,

'confidence_weights': confidence_weights

}



## 4. Risk Proposal Generation

### 4.1 Complete Risk Proposal Structure

RiskProposal = {

'entry_plan': {

'order_type': 'MARKET',

'entry_price': float,      # From qualification

'position_size': int,      # Contracts

'direction': int           # 1 or -1

},


'stop_loss_plan': {

'initial_stop': float,     # Price level

'stop_distance': float,    # Points from entry

'placement_rule': str,     # 'FIXED' or 'LVN_ADJUSTED'

'lvn_buffer': float        # Extra points beyond LVN

},


'take_profit_plan': {

'target_price': float,     # Primary target

'rr_ratio': float,         # Risk:Reward achieved

'scaling_out': {           # Optional partial exits

'level_1': {'price': float, 'percent': 0.5},

'level_2': {'price': float, 'percent': 0.3},

'level_3': {'price': float, 'percent': 0.2}

}

},


'dynamic_management': {

'use_trailing_stop': bool,

'trail_activation': float,  # Price level to activate

'trail_distance': float,    # Points to trail

'time_stop': {

'max_bars': int,        # Maximum time in trade

'reduction_percent': 0.5 # Exit at X% of target

}

},


'risk_metrics': {

'dollar_risk': float,       # $ at risk

'percent_risk': float,      # % of account

'max_loss': float,          # Worst case scenario

'expected_value': float,    # Statistical expectation

'sharpe_contribution': float # Impact on portfolio Sharpe

},


'confidence_scores': {

'position_size_confidence': float,  # 0-1

'stop_placement_confidence': float, # 0-1

'target_confidence': float,         # 0-1

'overall_confidence': float         # Weighted average

}

}


### 4.2 Risk Calculation Logic

def generate_risk_proposal(self, trade_qualification: Dict) -> Dict:

"""Generate complete risk management proposal"""


# 1. Run ensemble forward pass

decisions = self.ensemble(trade_qualification)


# 2. Calculate position size

base_contracts = self.config['base_position_size']

position_size = int(base_contracts * decisions['position_size'])


# 3. Determine stop loss

if decisions['use_lvn_stop']:

stop_price = self._calculate_lvn_stop(

trade_qualification,

decisions['stop_distance']

)

else:

stop_price = self._calculate_fixed_stop(

trade_qualification['entry_price'],

trade_qualification['direction'],

decisions['stop_distance']

)


# 4. Calculate take profit

risk_points = abs(trade_qualification['entry_price'] - stop_price)

reward_points = risk_points * decisions['rr_ratio']


if trade_qualification['direction'] == 1:  # Long

target_price = trade_qualification['entry_price'] + reward_points

else:  # Short

target_price = trade_qualification['entry_price'] - reward_points


# 5. Build complete proposal

proposal = self._build_proposal(

trade_qualification,

position_size,

stop_price,

target_price,

decisions

)


# 6. Validate against constraints

validated_proposal = self._validate_proposal(proposal)


return validated_proposal


### 4.3 LVN-Aware Stop Placement

def _calculate_lvn_stop(self, qualification: Dict, base_distance: float) -> float:

"""Place stops beyond significant LVN levels"""


direction = qualification['direction']

entry_price = qualification['entry_price']


if direction == 1:  # Long trade

# Look for support LVN below entry

lvn = qualification['lvn_analysis']['nearest_support']

if lvn['strength'] > 70:  # Strong LVN

# Place stop below LVN with buffer

stop_price = lvn['price'] - self.config['lvn_buffer']

else:

# Use base distance if no strong LVN

stop_price = entry_price - base_distance

else:  # Short trade

# Look for resistance LVN above entry

lvn = qualification['lvn_analysis']['nearest_resistance']

if lvn['strength'] > 70:

stop_price = lvn['price'] + self.config['lvn_buffer']

else:

stop_price = entry_price + base_distance


return stop_price



## 5. Constraint Validation

### 5.1 Risk Limit Enforcement

def _validate_proposal(self, proposal: Dict) -> Dict:

"""Ensure proposal respects all risk constraints"""


# 1. Check position risk

dollar_risk = self._calculate_dollar_risk(proposal)

max_allowed = self.account_balance * self.config['max_risk_per_trade']


if dollar_risk > max_allowed:

# Scale down position size

scale_factor = max_allowed / dollar_risk

proposal = self._scale_position(proposal, scale_factor)


# 2. Check daily drawdown limit

potential_loss = self.daily_pnl - dollar_risk

if potential_loss < -self.account_balance * self.config['max_daily_drawdown']:

# Reject trade

proposal['rejected'] = True

proposal['rejection_reason'] = 'Daily loss limit would be exceeded'


# 3. Check position count

if self.open_positions >= self.config['max_open_positions']:

proposal['rejected'] = True

proposal['rejection_reason'] = 'Maximum positions already open'


return proposal


### 5.2 Dynamic Adjustment Logic

def _apply_performance_adjustment(self, base_size: float) -> float:

"""Adjust position size based on recent performance"""


recent = self.system_state['recent_performance']


# Increase size after winning streaks

if recent['win_rate'] > 0.65 and recent['profit_factor'] > 2.0:

multiplier = 1.2

# Decrease size after losing streaks

elif recent['win_rate'] < 0.35 or recent['profit_factor'] < 0.8:

multiplier = 0.7

else:

multiplier = 1.0


# Apply regime-based adjustment

regime = self.market_context['regime_vector']

volatility_dim = regime[1]  # Volatility dimension


if volatility_dim > 0.5:  # High volatility

multiplier *= 0.8


return base_size * multiplier



## 6. Performance & Training

### 6.1 Training Methodology

Reward Function:

def calculate_reward(self, proposal: Dict, trade_result: Dict) -> float:

"""Multi-objective reward for training"""


# Primary reward: Risk-adjusted returns

sharpe_contribution = trade_result['pnl'] / trade_result['risk_taken']


# Secondary rewards

risk_efficiency = 1.0 - (trade_result['max_drawdown'] / trade_result['risk_taken'])

time_efficiency = 1.0 - (trade_result['bars_in_trade'] / self.config['max_bars_in_trade'])


# Penalty for constraint violations

violation_penalty = -1.0 if trade_result['violated_constraints'] else 0.0


# Combined reward

reward = (

0.6 * sharpe_contribution +

0.2 * risk_efficiency +

0.1 * time_efficiency +

0.1 * violation_penalty

)


return reward


### 6.2 Performance Metrics

# Tracked for each agent

{

'position_sizer_metrics': {

'avg_position_size': 1.8,

'size_stability': 0.85,      # Consistency

'risk_efficiency': 0.92       # Actual vs planned risk

},

'stop_agent_metrics': {

'stop_hit_rate': 0.35,

'avg_stop_distance': 12.3,

'lvn_usage_rate': 0.78

},

'target_agent_metrics': {

'target_hit_rate': 0.62,

'avg_rr_achieved': 2.1,

'trailing_success_rate': 0.71

}

}



## 7. Output Events

### 7.1 Risk Proposal Event

Event Name: RISK_PROPOSAL_READY
 Triggered by: M-RMS after processing trade qualification
 Consumed by: DecisionGate (final validation)

### 7.2 Risk Alerts

# Optional risk warnings

RISK_ALERTS = {

'HIGH_VOLATILITY_WARNING': 'Volatility exceeds normal range',

'CORRELATION_WARNING': 'High correlation with existing positions',

'REGIME_TRANSITION_WARNING': 'Market regime appears unstable',

'DRAWDOWN_WARNING': 'Approaching daily loss limit'

}



## 8. Integration Requirements

### 8.1 Upstream Dependencies

Trade qualifications from Main MARL Core

Market context including regime vectors

Current system state and performance metrics

### 8.2 Downstream Integration

Risk proposals to DecisionGate

Executed parameters to ExecutionHandler

Results feedback for continuous learning


## 9. Operational Requirements

### 9.1 Performance Requirements

Inference Latency: <10ms per proposal

Memory Usage: <500MB including models

CPU/GPU: CPU sufficient for inference

### 9.2 Reliability Requirements

Failsafe Defaults: Revert to conservative parameters on error

Constraint Guarantee: Never violate hard risk limits

Audit Trail: Log all decisions with full context


## 10. What This Component Does NOT Do

Does NOT make entry/exit timing decisions

Does NOT identify trading opportunities

Does NOT execute trades directly

Does NOT analyze market data

Does NOT override system risk limits

Does NOT manage existing positions

Does NOT calculate P&L


This completes the M-RMS PRD. The system provides sophisticated, adaptive risk management that goes far beyond simple percentage-based position sizing, incorporating market context, system performance, and multiple risk dimensions to optimize each trading opportunity.

The multi-agent architecture ensures that different aspects of risk are specialized and optimized independently while maintaining overall coherence through the ensemble coordinator.

Ready to proceed to the Main MARL Core - the central intelligence that orchestrates everything?

