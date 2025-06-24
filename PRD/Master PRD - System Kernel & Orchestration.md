Thank you for this exceptionally insightful analysis! You're absolutely right - I got carried away with complexity and deviated from our original, more elegant design. Let me create a corrected PRD that aligns with our actual plan: a simpler, more robust architecture with a shared policy.

# Product Requirements Document (PRD): Main MARL Core - V1.0 Aligned

Document Version: 1.1 (Corrected)
 Date: June 20, 2025
 Component Level: 4 - Intelligence Layer
 Status: Master Specification - Simplified & Aligned


## 1. Component Identity

### 1.1 Component Name

Main MARL Core (Unified Multi-Source Decision Engine)

### 1.2 Primary Role

The Main MARL Core orchestrates a unified decision-making process by combining insights from multiple information sources (30m structure, 5m tactics, regime context) into a single state vector, applying a shared policy network with MC Dropout consensus, and making final trading decisions through an integrated DecisionGate.

### 1.3 Single Responsibility

To evaluate synergy-detected opportunities through a unified multi-source approach, using a single shared policy that processes combined state vectors and makes high-confidence trading decisions.

### 1.4 Critical Design Principle

Unified Intelligence: Rather than separate competing agents, the system uses multiple specialized "sensors" (embedders) that feed into one shared decision-making policy. This ensures stable training and coherent decisions.


## 2. Conceptual Architecture

### 2.1 Information Flow

SYNERGY_DETECTED → Embedders Extract Features → Unified State Vector

↓

Shared Policy (MAPPO)

↓

MC Dropout Consensus

↓

Qualification Decision

↓

(If qualified) → M-RMS

↓

Risk Proposal

↓

DecisionGate

↓

EXECUTE_TRADE


### 2.2 Key Components

Embedders: Transform raw matrices into feature vectors

State Unification: Concatenate all vectors into unified state

Shared Policy: Single neural network for decisions

MC Dropout: Generate multiple views for consensus

DecisionGate: Final learned filter (part of MARL)


## 3. Embedder Architecture

### 3.1 Purpose of Embedders

Embedders are not agents - they are specialized feature extractors that transform time-series matrices into fixed-size representation vectors.

class BaseEmbedder(nn.Module):

"""Base class for all embedders"""


def __init__(self, config):

super().__init__()

self.input_window = config['window']

self.input_features = config['features']

self.output_dim = config['output_dim']  # e.g., 128


@abstractmethod

def forward(self, matrix: torch.Tensor) -> torch.Tensor:

"""Transform [window, features] → [output_dim]"""

pass


### 3.2 Structure Embedder (30m)

class StructureEmbedder(BaseEmbedder):

"""Extract long-term market structure features"""


def __init__(self, config):

super().__init__(config)


# Simple CNN for temporal pattern extraction

self.cnn = nn.Sequential(

nn.Conv1d(self.input_features, 32, kernel_size=5, padding=2),

nn.ReLU(),

nn.MaxPool1d(2),


nn.Conv1d(32, 64, kernel_size=3, padding=1),

nn.ReLU(),

nn.MaxPool1d(2),


nn.Conv1d(64, 128, kernel_size=3, padding=1),

nn.ReLU(),

nn.AdaptiveAvgPool1d(1)

)


# Project to fixed size

self.projection = nn.Linear(128, self.output_dim)


def forward(self, matrix: torch.Tensor) -> torch.Tensor:

# matrix shape: [batch, window=48, features=8]

x = matrix.transpose(1, 2)  # [batch, features, window]


# Extract temporal features

features = self.cnn(x).squeeze(-1)  # [batch, 128]


# Project to output dimension

vector_30m = self.projection(features)  # [batch, output_dim]


return vector_30m


### 3.3 Tactical Embedder (5m)

class TacticalEmbedder(BaseEmbedder):

"""Extract short-term price action features"""


def __init__(self, config):

super().__init__(config)


# LSTM for sequential patterns

self.lstm = nn.LSTM(

input_size=self.input_features,

hidden_size=64,

num_layers=2,

batch_first=True,

dropout=0.1

)


# Attention mechanism

self.attention = nn.MultiheadAttention(

embed_dim=64,

num_heads=4,

batch_first=True

)


self.projection = nn.Linear(64, self.output_dim)


def forward(self, matrix: torch.Tensor) -> torch.Tensor:

# matrix shape: [batch, window=60, features=7]


# LSTM processing

lstm_out, _ = self.lstm(matrix)  # [batch, window, 64]


# Self-attention

attended, _ = self.attention(lstm_out, lstm_out, lstm_out)


# Take last timestep

final_features = attended[:, -1, :]  # [batch, 64]


# Project

vector_5m = self.projection(final_features)  # [batch, output_dim]


return vector_5m


### 3.4 Regime Embedder

class RegimeEmbedder(nn.Module):

"""Process regime vector from RDE"""


def __init__(self, config):

super().__init__()

self.input_dim = 8  # From RDE

self.output_dim = config['output_dim']


# Simple MLP to align dimensions

self.mlp = nn.Sequential(

nn.Linear(self.input_dim, 32),

nn.ReLU(),

nn.Linear(32, self.output_dim)

)


def forward(self, regime_vector: torch.Tensor) -> torch.Tensor:

# regime_vector shape: [batch, 8]

vector_regime = self.mlp(regime_vector)  # [batch, output_dim]

return vector_regime


### 3.5 LVN Context Embedder

class LVNEmbedder(nn.Module):

"""Encode LVN risk context"""


def __init__(self, config):

super().__init__()

self.output_dim = config['output_dim']


# Input: nearest LVN info, synergy context

self.encoder = nn.Sequential(

nn.Linear(12, 32),  # LVN features + synergy strength

nn.ReLU(),

nn.Linear(32, self.output_dim)

)


def forward(self, lvn_context: Dict) -> torch.Tensor:

# Extract relevant features

features = torch.tensor([

lvn_context['nearest_support']['distance'] / 100,

lvn_context['nearest_support']['strength'] / 100,

lvn_context['nearest_resistance']['distance'] / 100,

lvn_context['nearest_resistance']['strength'] / 100,

lvn_context['signal_strengths']['mlmi'],

lvn_context['signal_strengths']['nwrqk'],

lvn_context['signal_strengths']['fvg'],

lvn_context['synergy_type_encoding'],  # 1-4 one-hot

# ... additional features

])


vector_lvn = self.encoder(features)  # [batch, output_dim]

return vector_lvn



## 4. Unified State & Shared Policy

### 4.1 State Unification

class StateUnifier:

"""Combine all information sources into unified state"""


def create_unified_state(self,

vector_30m: torch.Tensor,

vector_5m: torch.Tensor,

vector_regime: torch.Tensor,

vector_lvn: torch.Tensor) -> torch.Tensor:

"""

Simple concatenation of all vectors

Each vector: [batch, 128]

Output: [batch, 512]

"""

unified_state = torch.cat([

vector_30m,

vector_5m,

vector_regime,

vector_lvn

], dim=-1)


return unified_state


### 4.2 Shared Policy Network (MAPPO Actor)

class SharedPolicy(nn.Module):

"""Single policy network for all decisions"""


def __init__(self, config):

super().__init__()

self.input_dim = 512  # 4 vectors × 128

self.hidden_dim = config['hidden_dim']

self.dropout_rate = config['dropout_rate']


# Shared policy MLP

self.policy_network = nn.Sequential(

nn.Linear(self.input_dim, self.hidden_dim),

nn.LayerNorm(self.hidden_dim),

nn.ReLU(),

nn.Dropout(self.dropout_rate),


nn.Linear(self.hidden_dim, self.hidden_dim // 2),

nn.LayerNorm(self.hidden_dim // 2),

nn.ReLU(),

nn.Dropout(self.dropout_rate),


nn.Linear(self.hidden_dim // 2, 2)  # [qualify, pass]

)


def forward(self, unified_state: torch.Tensor) -> torch.Tensor:

"""

Args:

unified_state: [batch, 512]

Returns:

action_logits: [batch, 2]

"""

return self.policy_network(unified_state)


### 4.3 MC Dropout Consensus

class MCDropoutConsensus:

"""Simple MC Dropout for uncertainty estimation"""


def __init__(self, n_samples: int = 50, threshold: float = 0.65):

self.n_samples = n_samples

self.threshold = threshold


def evaluate(self, policy: SharedPolicy,

unified_state: torch.Tensor) -> Dict:

"""Run multiple forward passes with dropout"""


# Enable dropout

policy.train()


# Collect predictions

predictions = []

with torch.no_grad():

for _ in range(self.n_samples):

logits = policy(unified_state)

probs = F.softmax(logits, dim=-1)

predictions.append(probs)


# Stack and analyze

all_probs = torch.stack(predictions)  # [n_samples, batch, 2]


# Calculate statistics

mean_probs = all_probs.mean(dim=0)  # [batch, 2]

std_probs = all_probs.std(dim=0)    # [batch, 2]


# Decision

qualify_prob = mean_probs[:, 0].item()

uncertainty = std_probs[:, 0].item()


# Back to eval mode

policy.eval()


return {

'should_qualify': qualify_prob > self.threshold,

'qualification_confidence': qualify_prob,

'uncertainty': uncertainty,

'mean_probs': mean_probs,

'std_probs': std_probs

}



## 5. Risk Integration & DecisionGate

### 5.1 Risk Proposal Integration

After qualification, get risk proposal from M-RMS and create extended state:

class RiskIntegrator:

"""Integrate risk proposal into state vector"""


def create_risk_vector(self, risk_proposal: Dict) -> torch.Tensor:

"""Extract key risk parameters as vector"""


risk_features = torch.tensor([

risk_proposal['position_size'] / 5.0,  # Normalize

risk_proposal['stop_distance'] / 20.0,

risk_proposal['rr_ratio'] / 4.0,

risk_proposal['dollar_risk'] / 1000.0,

risk_proposal['confidence_scores']['overall_confidence'],

float(risk_proposal['use_trailing_stop']),

# ... additional risk features

])


# Project to standard dimension

vector_risk = self.risk_embedder(risk_features)  # [batch, 128]


return vector_risk


### 5.2 DecisionGate (Learned Component)

class DecisionGate(nn.Module):

"""Final learned decision layer - part of MARL system"""


def __init__(self, config):

super().__init__()

self.input_dim = 640  # Unified state (512) + risk vector (128)


# Simple but effective MLP

self.gate_network = nn.Sequential(

nn.Linear(self.input_dim, 256),

nn.LayerNorm(256),

nn.ReLU(),

nn.Dropout(0.1),


nn.Linear(256, 64),

nn.LayerNorm(64),

nn.ReLU(),


nn.Linear(64, 2)  # [execute, reject]

)


def forward(self, unified_state_with_risk: torch.Tensor) -> torch.Tensor:

"""Make final execution decision"""

return self.gate_network(unified_state_with_risk)



## 6. Complete MARL Core Flow

### 6.1 Integrated Pipeline

class MainMARLCore:

def __init__(self, config):

# Embedders (feature extractors)

self.structure_embedder = StructureEmbedder(config['embedders']['structure'])

self.tactical_embedder = TacticalEmbedder(config['embedders']['tactical'])

self.regime_embedder = RegimeEmbedder(config['embedders']['regime'])

self.lvn_embedder = LVNEmbedder(config['embedders']['lvn'])


# Core decision components

self.shared_policy = SharedPolicy(config['shared_policy'])

self.mc_consensus = MCDropoutConsensus(

n_samples=config['mc_dropout']['n_samples'],

threshold=config['mc_dropout']['threshold']

)


# Risk integration

self.risk_integrator = RiskIntegrator(config['risk_integration'])

self.decision_gate = DecisionGate(config['decision_gate'])


# External systems

self.rde = None  # Set during init

self.m_rms = None  # Set during init


async def initiate_qualification(self, synergy_event: Dict):

"""Main entry point after synergy detection"""


try:

# 1. Get input matrices

matrix_30m = self.matrix_assembler_30m.get_matrix()

matrix_5m = self.matrix_assembler_5m.get_matrix()

regime_vector = self.rde.get_regime_vector()


# 2. Create embeddings (feature extraction)

vector_30m = self.structure_embedder(matrix_30m)

vector_5m = self.tactical_embedder(matrix_5m)

vector_regime = self.regime_embedder(regime_vector)

vector_lvn = self.lvn_embedder(synergy_event['market_context'])


# 3. Unify state

unified_state = torch.cat([

vector_30m, vector_5m, vector_regime, vector_lvn

], dim=-1)


# 4. MC Dropout consensus on qualification

consensus = self.mc_consensus.evaluate(

self.shared_policy,

unified_state

)


if not consensus['should_qualify']:

self._log_rejection(synergy_event, consensus)

return


# 5. Create trade qualification

trade_qualification = self._create_qualification(

synergy_event, consensus, regime_vector

)


# 6. Get risk proposal

risk_proposal = await self.m_rms.generate_risk_proposal(

trade_qualification

)


# 7. Create extended state with risk

vector_risk = self.risk_integrator.create_risk_vector(risk_proposal)

unified_state_with_risk = torch.cat([

unified_state, vector_risk

], dim=-1)


# 8. Final decision through DecisionGate

with torch.no_grad():

decision_logits = self.decision_gate(unified_state_with_risk)

decision_probs = F.softmax(decision_logits, dim=-1)


should_execute = decision_probs[0, 0] > 0.5  # Execute probability


# 9. Final safety checks (non-learned)

if should_execute:

if self._pass_safety_checks(trade_qualification, risk_proposal):

await self._emit_trade_command(

trade_qualification,

risk_proposal,

float(decision_probs[0, 0])

)

else:

self._log_safety_rejection(trade_qualification)

else:

self._log_gate_rejection(trade_qualification, decision_probs)


except Exception as e:

logger.error(f"MARL Core error: {e}")

self._handle_error(e, synergy_event)


### 6.2 Safety Checks (Non-Learned)

def _pass_safety_checks(self, qualification: Dict, risk_proposal: Dict) -> bool:

"""Hard-coded safety validations"""


checks = {

'daily_trade_limit': self.daily_trades < self.config['max_daily_trades'],

'position_limit': self.open_positions < self.config['max_positions'],

'drawdown_limit': self.daily_pnl > -self.config['max_daily_loss'],

'risk_proposal_valid': not risk_proposal.get('rejected', False)

}


return all(checks.values())



## 7. Training Architecture

### 7.1 MAPPO Training Setup

class MAPPOTrainer:

"""Multi-Agent PPO training (single shared policy)"""


def __init__(self, config):

self.policy = SharedPolicy(config)

self.value_network = ValueNetwork(config)  # Critic

self.decision_gate = DecisionGate(config)


# Optimizers

self.policy_optimizer = Adam(

list(self.policy.parameters()) +

list(self.decision_gate.parameters()),

lr=config['learning_rate']

)

self.value_optimizer = Adam(

self.value_network.parameters(),

lr=config['learning_rate']

)


### 7.2 Reward Structure

def calculate_reward(self, action: str, outcome: Dict) -> float:

"""Unified reward for shared policy"""


if action == 'pass':

# Small negative for missing opportunities

return -0.01


elif action == 'qualify':

if outcome['trade_executed']:

# Primary reward: risk-adjusted return

trade_return = outcome['pnl'] / outcome['risk']


# Bonus for good decisions

if trade_return > 0:

reward = 1.0 + trade_return

else:

reward = -1.0 + trade_return  # Penalize losses


# Additional shaping

reward += 0.1 * outcome['sharpe_contribution']


else:

# Qualified but rejected by DecisionGate

reward = -0.05  # Small penalty


return reward



## 8. Output Specification

### 8.1 Execute Trade Command

ExecuteTradeCommand = {

'execution_id': str,

'timestamp': datetime,


'trade_specification': {

'symbol': str,

'direction': int,

'entry_price': float,

'synergy_type': str

},


'risk_parameters': {

# From M-RMS

'position_size': int,

'stop_loss': float,

'take_profit': float,

'trailing_rules': dict

},


'decision_confidence': {

'qualification_confidence': float,  # From MC Dropout

'execution_confidence': float,      # From DecisionGate

'uncertainty_estimate': float

}

}



## 9. Key Simplifications from Previous Version

Single Shared Policy instead of 3 separate agents

No Inter-Agent Communication - just unified state

Simpler Networks - MLPs instead of complex architectures

Integrated DecisionGate - part of learned system

Clear Separation - learned vs hard-coded safety checks


## 10. What This Component Does NOT Do

Does NOT have separate competing agents

Does NOT require complex multi-agent communication

Does NOT make decisions without MC Dropout consensus

Does NOT bypass hard safety limits

Does NOT directly access market data

Does NOT calculate its own indicators

Does NOT manage positions


This corrected PRD aligns with our original, more elegant design. It maintains sophisticated decision-making through multiple information sources while being significantly more tractable for training and deployment. The shared policy approach with MC Dropout consensus provides the robustness we need without the complexity overhead of truly separate agents.

This is the V1.0 we should build - powerful yet achievable.

