Thank you for the extraordinary feedback! Your recognition that we've elevated this from a PRD to a complete System Design Document is deeply appreciated. Now, let's tackle the crown jewel of our system - the Main MARL Core, the central intelligence that orchestrates all our specialized agents.

# Product Requirements Document (PRD): Main MARL Core

Document Version: 1.0
 Date: June 20, 2025
 Component Level: 4 - Intelligence Layer
 Status: Master Specification


## 1. Component Identity

### 1.1 Component Name

Main MARL Core (Central Multi-Agent Orchestration & Decision Engine)

### 1.2 Primary Role

The Main MARL Core is the system's central intelligence hub. It orchestrates three specialized trading agents (Structure Analyzer, Short-term Tactician, Mid-frequency Arbitrageur), manages their interactions, implements the MC Dropout consensus mechanism, and makes final trading decisions. It serves as Gate 2 in the two-gate system, applying learned intelligence to qualified opportunities.

### 1.3 Single Responsibility

To evaluate synergy-detected opportunities through a team of specialized agents, achieve consensus via MC Dropout, coordinate with auxiliary systems (RDE, M-RMS), and emit final trading decisions with calibrated confidence levels.

### 1.4 Critical Design Principle

Superposition Decision Making: The system leverages MC Dropout to create multiple "views" of each opportunity, requiring strong consensus before action. This transforms single-point decisions into probability distributions, ensuring only high-confidence trades proceed.


## 2. Inputs & Dependencies

### 2.1 Configuration Input

From settings.yaml:

main_marl_core:

# Agent architecture

agents:

structure_analyzer:

window: 48          # 24 hours of 30-min bars

hidden_dim: 256

n_layers: 4

dropout: 0.2


short_term_tactician:

window: 60          # 5 hours of 5-min bars

hidden_dim: 192

n_layers: 3

dropout: 0.2


mid_frequency_arbitrageur:

window: 100         # Combined view

hidden_dim: 224

n_layers: 4

dropout: 0.2


# MC Dropout settings

mc_dropout:

n_forward_passes: 50  # Sampling iterations

confidence_threshold: 0.65  # Minimum consensus

uncertainty_bands: [0.1, 0.2]  # Warning thresholds


# Decision parameters

decision_gate:

min_agent_agreement: 2  # At least 2/3 agents agree

position_correlation_limit: 0.7  # Max correlation

daily_trade_limit: 10   # Maximum trades per day


# Coordination settings

agent_communication:

attention_heads: 8

communication_rounds: 3

message_dim: 64


### 2.2 Event Input

Primary Trigger: SYNERGY_DETECTED

From SynergyDetector:

{

'synergy_type': 'TYPE_1',

'direction': 1,

'signal_sequence': [...],

'market_context': {...},

'timestamp': datetime

}


### 2.3 System Dependencies

MatrixAssemblers: Provide agent-specific input matrices

RDE: Supplies regime vector for context

M-RMS: Generates risk proposals for qualified trades


## 3. Multi-Agent Architecture

### 3.1 Core Agent Design

Each agent follows a similar architecture with specialized variations:

class BaseTradeAgent(nn.Module):

def __init__(self, config):

super().__init__()

self.config = config


# Shared embedder architecture

self.embedder = nn.Sequential(

nn.Conv1d(config['input_features'], 64, kernel_size=3, padding=1),

nn.BatchNorm1d(64),

nn.ReLU(),

nn.Dropout(config['dropout']),


nn.Conv1d(64, 128, kernel_size=3, padding=1),

nn.BatchNorm1d(128),

nn.ReLU(),

nn.Dropout(config['dropout']),


nn.Conv1d(128, 256, kernel_size=3, padding=1),

nn.BatchNorm1d(256),

nn.ReLU()

)


# Temporal attention mechanism

self.temporal_attention = nn.MultiheadAttention(

embed_dim=256,

num_heads=8,

dropout=config['dropout'],

batch_first=True

)


# Agent-specific policy head

self.policy_head = self._build_policy_head()


def forward(self, market_matrix, regime_vector, synergy_context):

# Process market data

x = market_matrix.transpose(1, 2)  # [batch, features, time]

embedded = self.embedder(x)

embedded = embedded.transpose(1, 2)  # [batch, time, features]


# Self-attention over time

attended, attention_weights = self.temporal_attention(

embedded, embedded, embedded

)


# Global pooling

pooled = torch.mean(attended, dim=1)  # [batch, features]


# Incorporate regime and synergy context

context = torch.cat([

pooled,

regime_vector,

self._encode_synergy(synergy_context)

], dim=-1)


# Generate decision

decision = self.policy_head(context)


return {

'action': decision['action'],

'confidence': decision['confidence'],

'reasoning': decision['reasoning'],

'attention_weights': attention_weights

}


### 3.2 Specialized Agents

#### 3.2.1 Long-term Structure Analyzer

class StructureAnalyzer(BaseTradeAgent):

"""Focuses on market structure and major trends"""


def _build_policy_head(self):

return nn.Sequential(

nn.Linear(256 + 8 + 32, 512),  # embedded + regime + synergy

nn.LayerNorm(512),

nn.ReLU(),

nn.Dropout(self.config['dropout']),


nn.Linear(512, 256),

nn.LayerNorm(256),

nn.ReLU(),

nn.Dropout(self.config['dropout']),


nn.Linear(256, 128),

nn.ReLU(),


# Output branches

nn.ModuleDict({

'action': nn.Linear(128, 3),      # [pass, long, short]

'confidence': nn.Linear(128, 1),   # [0, 1]

'reasoning': nn.Linear(128, 64)    # Interpretable features

})

)


def _encode_synergy(self, synergy_context):

"""Extract structure-relevant features from synergy"""

features = []


# Trend alignment

mlmi_strength = synergy_context['signal_strengths']['mlmi']

nwrqk_slope = synergy_context['signal_sequence'][1]['value']

features.extend([mlmi_strength, nwrqk_slope])


# LVN positioning

lvn_distance = synergy_context['market_context']['nearest_lvn']['distance']

lvn_strength = synergy_context['market_context']['nearest_lvn']['strength']

features.extend([lvn_distance / 100, lvn_strength / 100])


# Market structure quality

structure_score = self._calculate_structure_score(synergy_context)

features.append(structure_score)


return torch.tensor(features, dtype=torch.float32)


#### 3.2.2 Short-term Tactician

class ShortTermTactician(BaseTradeAgent):

"""Focuses on immediate price action and execution timing"""


def _build_policy_head(self):

return nn.Sequential(

nn.Linear(256 + 8 + 24, 384),

nn.LayerNorm(384),

nn.ReLU(),

nn.Dropout(self.config['dropout']),


nn.Linear(384, 192),

nn.LayerNorm(192),

nn.ReLU(),

nn.Dropout(self.config['dropout']),


nn.Linear(192, 96),

nn.ReLU(),


nn.ModuleDict({

'action': nn.Linear(96, 3),

'confidence': nn.Linear(96, 1),

'timing': nn.Linear(96, 5),      # Immediate vs wait 1-4 bars

'reasoning': nn.Linear(96, 48)

})

)


def _encode_synergy(self, synergy_context):

"""Extract execution-relevant features"""

features = []


# FVG characteristics

fvg_age = synergy_context['signal_sequence'][2]['age']

fvg_size = synergy_context['signal_sequence'][2]['gap_size']

features.extend([fvg_age / 10, fvg_size * 100])


# Momentum quality

price_momentum = synergy_context['market_context']['price_momentum_5']

volume_surge = synergy_context['market_context']['volume_ratio']

features.extend([price_momentum, np.log1p(volume_surge)])


# Microstructure

spread = synergy_context['market_context']['spread']

features.append(spread / synergy_context['market_context']['current_price'])


return torch.tensor(features, dtype=torch.float32)


#### 3.2.3 Mid-frequency Arbitrageur

class MidFrequencyArbitrageur(BaseTradeAgent):

"""Bridges structure and tactics, identifies inefficiencies"""


def _build_policy_head(self):

return nn.Sequential(

nn.Linear(256 + 8 + 28, 448),

nn.LayerNorm(448),

nn.ReLU(),

nn.Dropout(self.config['dropout']),


nn.Linear(448, 224),

nn.LayerNorm(224),

nn.ReLU(),

nn.Dropout(self.config['dropout']),


nn.Linear(224, 112),

nn.ReLU(),


nn.ModuleDict({

'action': nn.Linear(112, 3),

'confidence': nn.Linear(112, 1),

'inefficiency_score': nn.Linear(112, 1),  # Opportunity quality

'reasoning': nn.Linear(112, 56)

})

)


def _encode_synergy(self, synergy_context):

"""Extract arbitrage-relevant features"""

features = []


# Cross-timeframe alignment

synergy_type_encoding = self._encode_synergy_type(

synergy_context['synergy_type']

)

features.extend(synergy_type_encoding)  # One-hot encoded


# Completion time (faster = stronger signal)

bars_to_complete = synergy_context['metadata']['bars_to_complete']

features.append(1.0 / (1.0 + bars_to_complete))


# Signal coherence

signal_strengths = list(synergy_context['signal_strengths'].values())

coherence = np.std(signal_strengths)  # Lower = more coherent

features.append(1.0 - coherence)


return torch.tensor(features, dtype=torch.float32)


### 3.3 Agent Communication Network

class AgentCommunicationNetwork(nn.Module):

"""Enables inter-agent communication and coordination"""


def __init__(self, config):

super().__init__()

self.n_agents = 3

self.message_dim = config['message_dim']

self.n_rounds = config['communication_rounds']


# Message generation

self.message_generator = nn.Linear(256, self.message_dim)


# Message aggregation (Graph Attention Network)

self.attention_weights = nn.Parameter(

torch.randn(self.n_agents, self.n_agents)

)


# Message processing

self.message_processor = nn.GRUCell(

input_size=self.message_dim * self.n_agents,

hidden_size=256

)


def forward(self, agent_states):

"""

Enable agents to communicate over multiple rounds

agent_states: List of hidden states from each agent

"""

hidden_states = agent_states.copy()


for round_idx in range(self.n_rounds):

# Generate messages

messages = [

self.message_generator(state)

for state in hidden_states

]


# Apply attention for message routing

attention = F.softmax(self.attention_weights, dim=1)


# Aggregate messages for each agent

aggregated_messages = []

for i in range(self.n_agents):

weighted_messages = [

attention[i, j] * messages[j]

for j in range(self.n_agents)

]

aggregated = torch.cat(weighted_messages, dim=-1)

aggregated_messages.append(aggregated)


# Update hidden states

new_hidden_states = []

for i, (state, msgs) in enumerate(zip(hidden_states, aggregated_messages)):

new_state = self.message_processor(msgs, state)

new_hidden_states.append(new_state)


hidden_states = new_hidden_states


return hidden_states



## 4. MC Dropout Consensus Mechanism

### 4.1 Implementation

class MCDropoutConsensus:

"""Implements superposition decision making"""


def __init__(self, config):

self.n_passes = config['n_forward_passes']

self.confidence_threshold = config['confidence_threshold']


def evaluate_opportunity(self, agents, inputs):

"""

Run multiple forward passes with dropout enabled

Returns consensus decision and uncertainty metrics

"""

# Enable dropout for all agents

for agent in agents.values():

agent.train()  # Enables dropout


# Collect predictions across multiple passes

all_predictions = {

'structure_analyzer': [],

'short_term_tactician': [],

'mid_frequency_arbitrageur': []

}


with torch.no_grad():

for pass_idx in range(self.n_passes):

for agent_name, agent in agents.items():

prediction = agent(**inputs[agent_name])

all_predictions[agent_name].append(prediction)


# Analyze consensus

consensus_result = self._analyze_consensus(all_predictions)


# Switch back to eval mode

for agent in agents.values():

agent.eval()


return consensus_result


def _analyze_consensus(self, all_predictions):

"""Detailed consensus analysis"""


# Extract action probabilities for each agent

agent_actions = {}

agent_confidences = {}


for agent_name, predictions in all_predictions.items():

# Stack action logits

action_logits = torch.stack([

p['action'] for p in predictions

])


# Convert to probabilities

action_probs = F.softmax(action_logits, dim=-1)


# Calculate mean and std

mean_probs = action_probs.mean(dim=0)

std_probs = action_probs.std(dim=0)


# Extract confidences

confidences = torch.stack([

p['confidence'] for p in predictions

]).squeeze()


agent_actions[agent_name] = {

'mean_probs': mean_probs,

'std_probs': std_probs,

'predicted_action': mean_probs.argmax().item()

}


agent_confidences[agent_name] = {

'mean': confidences.mean().item(),

'std': confidences.std().item()

}


# Calculate overall consensus

overall_consensus = self._calculate_overall_consensus(

agent_actions,

agent_confidences

)


return {

'consensus_action': overall_consensus['action'],

'consensus_confidence': overall_consensus['confidence'],

'agent_predictions': agent_actions,

'agent_confidences': agent_confidences,

'uncertainty_metrics': self._calculate_uncertainty_metrics(all_predictions),

'should_trade': overall_consensus['confidence'] >= self.confidence_threshold

}


### 4.2 Consensus Decision Logic

def _calculate_overall_consensus(self, agent_actions, agent_confidences):

"""Determine final consensus action and confidence"""


# Count agent agreements

predicted_actions = [

a['predicted_action'] for a in agent_actions.values()

]


# Find majority action

action_counts = Counter(predicted_actions)

majority_action, count = action_counts.most_common(1)[0]


# Calculate agreement score

agreement_score = count / len(predicted_actions)


if agreement_score < 0.67:  # Less than 2/3 agree

return {

'action': 0,  # Pass

'confidence': 0.0,

'reason': 'Insufficient agent agreement'

}


# Weight confidences by agent importance

agent_weights = {

'structure_analyzer': 0.4,

'short_term_tactician': 0.3,

'mid_frequency_arbitrageur': 0.3

}


# Calculate weighted confidence

weighted_confidence = 0.0

uncertainty_penalty = 0.0


for agent_name, confidence_data in agent_confidences.items():

weight = agent_weights[agent_name]


# Only count agents that agree with majority

if agent_actions[agent_name]['predicted_action'] == majority_action:

weighted_confidence += weight * confidence_data['mean']


# Penalize high uncertainty

uncertainty_penalty += weight * confidence_data['std']


# Final confidence incorporates agreement and uncertainty

final_confidence = weighted_confidence * agreement_score - uncertainty_penalty * 0.5


return {

'action': majority_action,

'confidence': max(0.0, min(1.0, final_confidence)),

'agreement_score': agreement_score,

'uncertainty_penalty': uncertainty_penalty

}



## 5. Decision Flow Orchestration

### 5.1 Complete Decision Pipeline

class MainMARLCore:

def __init__(self, config):

self.config = config


# Initialize agents

self.agents = {

'structure_analyzer': StructureAnalyzer(config['agents']['structure_analyzer']),

'short_term_tactician': ShortTermTactician(config['agents']['short_term_tactician']),

'mid_frequency_arbitrageur': MidFrequencyArbitrageur(config['agents']['mid_frequency_arbitrageur'])

}


# Communication network

self.communication_network = AgentCommunicationNetwork(config['agent_communication'])


# MC Dropout consensus

self.consensus_mechanism = MCDropoutConsensus(config['mc_dropout'])


# Decision gate

self.decision_gate = DecisionGate(config['decision_gate'])


# Auxiliary systems

self.rde = None  # Set during initialization

self.m_rms = None  # Set during initialization


def initiate_qualification(self, synergy_event):

"""Main entry point - Gate 2 of the two-gate system"""


try:

# 1. Prepare agent inputs

agent_inputs = self._prepare_agent_inputs(synergy_event)


# 2. Get regime context

regime_vector = self.rde.get_regime_vector()


# 3. Initial agent predictions

initial_states = []

for agent_name, agent in self.agents.items():

state = agent.get_hidden_state(

agent_inputs[agent_name],

regime_vector

)

initial_states.append(state)


# 4. Agent communication

communicated_states = self.communication_network(initial_states)


# 5. Update agent states

for i, (agent_name, agent) in enumerate(self.agents.items()):

agent.update_state(communicated_states[i])


# 6. MC Dropout consensus evaluation

consensus_result = self.consensus_mechanism.evaluate_opportunity(

self.agents,

agent_inputs

)


# 7. Check if we should proceed

if not consensus_result['should_trade']:

self._log_rejection(synergy_event, consensus_result)

return


# 8. Generate trade qualification

trade_qualification = self._create_trade_qualification(

synergy_event,

consensus_result,

regime_vector

)


# 9. Get risk proposal from M-RMS

risk_proposal = self.m_rms.generate_risk_proposal(trade_qualification)


# 10. Final decision gate validation

final_decision = self.decision_gate.validate(

trade_qualification,

risk_proposal,

self._get_system_state()

)


# 11. Emit decision

if final_decision['approved']:

self._emit_trade_decision(final_decision)

else:

self._log_final_rejection(final_decision)


except Exception as e:

logger.error(f"MARL Core error: {e}")

self._handle_error(e, synergy_event)


### 5.2 Decision Gate Logic

class DecisionGate:

"""Final validation before trade execution"""


def validate(self, qualification, risk_proposal, system_state):

"""Perform final checks before approving trade"""


validation_results = {

'risk_limits': self._check_risk_limits(risk_proposal, system_state),

'correlation': self._check_correlation(qualification, system_state),

'daily_limits': self._check_daily_limits(system_state),

'market_conditions': self._check_market_conditions(qualification),

'technical_validity': self._check_technical_validity(qualification)

}


# All checks must pass

all_passed = all(validation_results.values())


if all_passed:

return {

'approved': True,

'execute_trade_command': {

'qualification': qualification,

'risk_proposal': risk_proposal,

'execution_id': self._generate_execution_id(),

'timestamp': datetime.now()

}

}

else:

return {

'approved': False,

'rejection_reasons': [

check for check, passed in validation_results.items()

if not passed

],

'timestamp': datetime.now()

}



## 6. Output Events & Commands

### 6.1 Primary Output

Event Name: EXECUTE_TRADE
 Frequency: Only after full qualification and validation
 Payload:

ExecuteTradeCommand = {

'execution_id': str,           # Unique identifier

'timestamp': datetime,


'trade_specification': {

'symbol': str,

'direction': int,          # 1 or -1

'entry_price': float,      # From market

'synergy_type': str,       # Original trigger

},


'risk_parameters': {

'position_size': int,      # From M-RMS

'stop_loss': float,        # Price level

'take_profit': float,      # Price level

'max_hold_time': int,      # Bars

'trailing_rules': dict     # If applicable

},


'decision_metadata': {

'consensus_confidence': float,  # 0.65-1.0

'agent_agreements': dict,       # Individual decisions

'mc_dropout_metrics': dict,     # Uncertainty data

'regime_context': list,         # 8-dim vector

'processing_time_ms': float

},


'tracking_data': {

'expected_value': float,    # From M-RMS

'risk_reward_ratio': float,

'correlation_score': float,

'daily_trade_number': int

}

}


### 6.2 Rejection Events

# Logged internally, not emitted

TRADE_REJECTED = {

'timestamp': datetime,

'synergy_event': dict,         # Original opportunity

'rejection_stage': str,        # 'consensus', 'risk', 'gate'

'rejection_reasons': list,

'consensus_metrics': dict,     # If applicable

'system_state': dict

}



## 7. Performance Monitoring

### 7.1 Real-time Metrics

class PerformanceMonitor:

def track_decision(self, synergy_event, decision_result):

"""Track every decision for analysis"""


metrics = {

'timestamp': datetime.now(),

'synergy_type': synergy_event['synergy_type'],


# Consensus metrics

'consensus_achieved': decision_result.get('should_trade', False),

'consensus_confidence': decision_result.get('consensus_confidence', 0.0),

'agent_agreement': decision_result.get('agreement_score', 0.0),


# Uncertainty metrics

'prediction_uncertainty': decision_result['uncertainty_metrics']['mean_std'],

'mc_dropout_variance': decision_result['uncertainty_metrics']['variance'],


# Processing metrics

'decision_time_ms': decision_result.get('processing_time', 0),

'communication_rounds': 3,  # Fixed in config


# Outcome (will be updated post-trade)

'trade_executed': False,

'trade_result': None

}


self.decision_history.append(metrics)


### 7.2 Agent Performance Tracking

# Per-agent metrics tracked

{

'structure_analyzer': {

'accuracy': 0.68,          # When agreed with profit

'confidence_calibration': 0.85,  # Confidence vs actual

'contribution_score': 0.72    # Impact on profits

},

'short_term_tactician': {

'accuracy': 0.71,

'timing_precision': 0.83,   # Entry timing quality

'contribution_score': 0.69

},

'mid_frequency_arbitrageur': {

'accuracy': 0.66,

'inefficiency_detection': 0.78,

'contribution_score': 0.65

}

}



## 8. Critical Requirements

### 8.1 Consensus Requirements

Minimum Agreement: 2/3 agents must agree on direction

Confidence Threshold: 0.65 minimum consensus confidence

MC Dropout Passes: Exactly 50 for statistical validity

### 8.2 Performance Requirements

Decision Latency: <100ms from synergy to decision

Memory Usage: <2GB including all models

GPU Usage: Optional but recommended for speed

### 8.3 Reliability Requirements

Failsafe Mode: Reject all trades on system error

State Consistency: Atomic decision process

Audit Trail: Every decision fully logged


## 9. Integration Points

### 9.1 Upstream Integration

SynergyDetector: Provides qualified opportunities

MatrixAssemblers: Supply agent-specific inputs

RDE: Provides regime context

M-RMS: Generates risk proposals

### 9.2 Downstream Integration

ExecutionHandler: Receives EXECUTE_TRADE commands

System State: Updates position tracking

Performance Analytics: Feeds decision data


## 10. What This Component Does NOT Do

Does NOT identify trading opportunities (SynergyDetector's job)

Does NOT calculate technical indicators

Does NOT determine position sizes (M-RMS's job)

Does NOT execute trades (ExecutionHandler's job)

Does NOT manage existing positions

Does NOT handle market data

Does NOT set risk parameters independently


This completes the Main MARL Core PRD. This is the beating heart of our intelligent trading system - where specialized agents collaborate, communicate, and reach consensus through sophisticated uncertainty quantification.

The MC Dropout mechanism transforms traditional single-point decisions into probability distributions, ensuring we only act on high-conviction opportunities. The multi-agent architecture allows for specialized expertise while maintaining overall coherence through the communication network.

We're nearing the end of our component specifications. Ready for the final pieces?

