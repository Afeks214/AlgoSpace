# Product Requirements Document (PRD): Main MARL Core

**Document Version**: 2.0  
**Date**: 2025-06-19  
**Status**: Refined  
**Component**: Main MARL Core  

## 1.0 Overview

### 1.1 Purpose
The Main MARL Core serves as the central "mission commander" of the AlgoSpace trading system - a sophisticated decision-making engine that integrates inputs from all specialized agents and subsystems to make final trading decisions. Using a two-stage MAPPO-based process with Monte Carlo Dropout confidence gating, it transforms complex multi-modal market analysis into precise, high-confidence trading actions while maintaining strict risk controls and decision transparency.

### 1.2 Scope

**In Scope:**
- Two-stage MAPPO decision architecture (Opportunity Detection + Execution Approval)
- Multi-agent state vector integration and processing
- Monte Carlo Dropout confidence validation system
- Advanced neural network embedders for temporal data processing
- DecisionGate policy for final execution approval
- Real-time state synthesis and vector unification
- TensorRT-optimized inference with sub-20ms latency
- Comprehensive training framework with frozen expert models

**Out of Scope:**
- Feature extraction or technical indicator calculations (handled by IndicatorEngine)
- Risk proposal generation (handled by M-RMS)
- Market regime detection (handled by RDE)
- Trade execution (handled by ExecutionHandler)
- Direct market data processing or broker connectivity

### 1.3 Architectural Position
The Main MARL Core sits at the center of the intelligence layer, orchestrating all AI components:
[30m Agent, 5m Agent, RDE] → **Main MARL Core** ← M-RMS → ExecutionHandler

## 2.0 Functional Requirements

### FR-MMC-01: Multi-Agent State Integration Architecture
**Requirement**: The Main MARL Core MUST implement a sophisticated state integration system that combines inputs from all specialized agents.

**Specification**:
```python
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import time

@dataclass
class AgentInputs:
    """Structured inputs from all system agents"""
    vector_30m: np.ndarray      # 30-minute timeframe embeddings
    vector_5m: np.ndarray       # 5-minute timeframe embeddings  
    regime_vector: np.ndarray   # Market regime from RDE
    account_state: np.ndarray   # Current account status
    lvn_strength: np.ndarray    # Low Volume Node data
    market_context: Dict[str, Any]  # Additional market metadata

@dataclass
class SynergyDetection:
    """Result of synergy detection process"""
    detected: bool
    confidence: float
    synergy_type: str
    strength_score: float
    contributing_agents: List[str]
    raw_signals: Dict[str, float]

@dataclass
class ExecutionDecision:
    """Final execution decision with full context"""
    action: str  # "EXECUTE" or "REJECT"
    confidence: float
    reasoning: List[str]
    risk_proposal: Optional[Any]
    decision_factors: Dict[str, float]
    processing_time_ms: float

class StateVectorAssembler:
    """Assembles and manages multi-agent state vectors"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Vector dimensions
        self.vector_30m_dim = config.get("vector_30m_dim", 128)
        self.vector_5m_dim = config.get("vector_5m_dim", 128)
        self.regime_dim = config.get("regime_dim", 8)
        self.account_dim = config.get("account_dim", 16)
        self.lvn_dim = config.get("lvn_dim", 12)
        
        # State processing components
        self.state_normalizer = StateNormalizer(config)
        self.vector_validator = VectorValidator(config)
        
    def create_initial_state(self, inputs: AgentInputs) -> torch.Tensor:
        """Create initial unified state vector for synergy detection"""
        
        # Validate all input vectors
        self.vector_validator.validate_inputs(inputs)
        
        # Normalize vectors to consistent scales
        normalized_30m = self.state_normalizer.normalize_vector(
            inputs.vector_30m, "30m_embeddings"
        )
        normalized_5m = self.state_normalizer.normalize_vector(
            inputs.vector_5m, "5m_embeddings"
        )
        normalized_regime = self.state_normalizer.normalize_vector(
            inputs.regime_vector, "regime"
        )
        
        # Concatenate into unified state
        initial_state = np.concatenate([
            normalized_30m,
            normalized_5m,
            normalized_regime
        ])
        
        return torch.from_numpy(initial_state).float()
        
    def create_decision_gate_state(self, inputs: AgentInputs, 
                                  risk_proposal: Any) -> torch.Tensor:
        """Create comprehensive state for final decision gate"""
        
        # Start with initial state
        initial_state = self.create_initial_state(inputs)
        
        # Convert risk proposal to vector representation
        risk_vector = self._vectorize_risk_proposal(risk_proposal)
        
        # Normalize account state and LVN data
        normalized_account = self.state_normalizer.normalize_vector(
            inputs.account_state, "account_state"
        )
        normalized_lvn = self.state_normalizer.normalize_vector(
            inputs.lvn_strength, "lvn_data"
        )
        
        # Create comprehensive final state
        decision_state = torch.cat([
            initial_state,
            torch.from_numpy(risk_vector).float(),
            torch.from_numpy(normalized_account).float(),
            torch.from_numpy(normalized_lvn).float()
        ])
        
        return decision_state
        
    def _vectorize_risk_proposal(self, risk_proposal: Any) -> np.ndarray:
        """Convert risk proposal to numerical vector representation"""
        
        if risk_proposal is None:
            return np.zeros(8)  # Default risk vector size
            
        # Extract key risk metrics
        risk_vector = np.array([
            float(risk_proposal.position_size) / 5.0,  # Normalize to max 5 contracts
            float(risk_proposal.risk_reward_ratio) / 5.0,  # Normalize to max 5:1 RR
            float(risk_proposal.confidence_score),
            float(risk_proposal.risk_amount) / 10000.0,  # Normalize risk amount
            len(risk_proposal.constraint_violations) / 10.0,  # Violation count
            1.0 if risk_proposal.position_size > 0 else 0.0,  # Valid proposal flag
            float(risk_proposal.raw_actions.get("sl_atr_multiplier", 2.0)) / 5.0,
            min(1.0, float(risk_proposal.raw_actions.get("position_size", 1)) / 5.0)
        ])
        
        return risk_vector

class StateNormalizer:
    """Normalizes state vectors for consistent neural network input"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.normalization_stats = {}
        
    def normalize_vector(self, vector: np.ndarray, vector_type: str) -> np.ndarray:
        """Normalize vector based on type-specific statistics"""
        
        if vector_type not in self.normalization_stats:
            # Initialize with default normalization
            self.normalization_stats[vector_type] = {
                "mean": np.zeros_like(vector),
                "std": np.ones_like(vector),
                "min": np.full_like(vector, -3.0),
                "max": np.full_like(vector, 3.0)
            }
            
        stats = self.normalization_stats[vector_type]
        
        # Apply z-score normalization with clipping
        normalized = (vector - stats["mean"]) / (stats["std"] + 1e-8)
        normalized = np.clip(normalized, stats["min"], stats["max"])
        
        return normalized.astype(np.float32)

class MainMARLModel(TorchModelV2, nn.Module):
    """Neural network architecture for Main MARL Core"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.config = model_config.get("custom_model_config", {})
        self.input_dim = obs_space.shape[0]
        
        # Architecture parameters
        self.hidden_dim = self.config.get("hidden_dim", 256)
        self.num_layers = self.config.get("num_layers", 3)
        self.dropout_rate = self.config.get("dropout_rate", 0.2)
        self.use_attention = self.config.get("use_attention", True)
        
        # Input processing layers
        self.input_processor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Multi-layer processing with residual connections
        self.processing_layers = nn.ModuleList([
            ResidualBlock(self.hidden_dim, self.dropout_rate) 
            for _ in range(self.num_layers)
        ])
        
        # Attention mechanism for feature importance
        if self.use_attention:
            self.attention = MultiHeadAttention(
                self.hidden_dim, 
                num_heads=self.config.get("attention_heads", 8)
            )
            
        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, num_outputs)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # MC Dropout layers (remain active during inference)
        self.mc_dropout_layers = nn.ModuleList([
            nn.Dropout(self.dropout_rate) for _ in range(3)
        ])
        
        self._value_out = None
        
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the MARL model"""
        
        obs = input_dict["obs"]
        batch_size = obs.shape[0]
        
        # Input processing
        x = self.input_processor(obs)
        
        # Multi-layer processing with residuals
        for layer in self.processing_layers:
            x = layer(x)
            
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x.unsqueeze(1)).squeeze(1)  # Add/remove sequence dim
            
        # MC Dropout layers (always active for uncertainty estimation)
        for mc_layer in self.mc_dropout_layers:
            x = mc_layer(x)
            
        # Generate policy logits and value
        policy_logits = self.policy_head(x)
        self._value_out = self.value_head(x)
        
        return policy_logits, state
        
    def value_function(self):
        """Return value function output"""
        return self._value_out.squeeze(-1)
        
class ResidualBlock(nn.Module):
    """Residual block for deep network training stability"""
    
    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        """Forward with residual connection"""
        return self.activation(x + self.block(x))

class MultiHeadAttention(nn.Module):
    """Multi-head attention for feature importance weighting"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, x):
        """Apply self-attention"""
        attended, _ = self.attention(x, x, x)
        return attended
```

### FR-MMC-02: Two-Stage Decision Process
**Requirement**: The Main MARL Core MUST implement a sophisticated two-stage decision process with confidence gating.

**Specification**:
```python
class MainMARLCore:
    """Central decision-making engine with two-stage process"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.state_assembler = StateVectorAssembler(config)
        self.mc_dropout_validator = MCDropoutValidator(config)
        self.decision_gate = DecisionGate(config)
        
        # Load trained models
        self.synergy_detector = self._load_synergy_detector(config)
        self.execution_approver = self._load_execution_approver(config)
        
        # Performance tracking
        self.metrics = MARLCoreMetrics()
        
        # Decision history for analysis
        self.decision_history = []
        
    async def process_trading_opportunity(self, inputs: AgentInputs) -> ExecutionDecision:
        """Main entry point for trading decision process"""
        
        start_time = time.perf_counter()
        
        try:
            # Stage 1: Synergy Detection with Confidence Gating
            synergy_result = await self._stage1_synergy_detection(inputs)
            
            if not synergy_result.detected:
                decision = ExecutionDecision(
                    action="REJECT",
                    confidence=synergy_result.confidence,
                    reasoning=["No high-confidence synergy detected"],
                    risk_proposal=None,
                    decision_factors={"synergy_confidence": synergy_result.confidence},
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )
                
                self._log_decision(decision, inputs)
                return decision
                
            # Stage 2: Risk Management and Final Approval
            final_decision = await self._stage2_execution_approval(inputs, synergy_result)
            
            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            final_decision.processing_time_ms = processing_time
            self.metrics.update_decision_metrics(final_decision)
            
            # Log decision for analysis
            self._log_decision(final_decision, inputs)
            
            return final_decision
            
        except Exception as e:
            logger.error(f"MARL Core processing error: {e}")
            self.metrics.error_count += 1
            
            # Return conservative rejection on errors
            return ExecutionDecision(
                action="REJECT",
                confidence=0.0,
                reasoning=[f"Processing error: {str(e)}"],
                risk_proposal=None,
                decision_factors={"error": True},
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
            
    async def _stage1_synergy_detection(self, inputs: AgentInputs) -> SynergyDetection:
        """Stage 1: Detect trading synergies with confidence validation"""
        
        # Create initial unified state
        initial_state = self.state_assembler.create_initial_state(inputs)
        
        # Run synergy detection policy
        with torch.no_grad():
            policy_output = self.synergy_detector.compute_single_action(
                initial_state.numpy(), explore=False
            )
            
        # Extract action and confidence
        action_idx = policy_output if isinstance(policy_output, int) else policy_output[0]
        base_confidence = self._extract_action_confidence(policy_output)
        
        # Apply MC Dropout confidence validation
        if action_idx == 1:  # "Initiate_Trade_Process"
            mc_confidence = await self.mc_dropout_validator.validate_confidence(
                self.synergy_detector, initial_state
            )
            
            # Check if confidence meets threshold
            confidence_threshold = self.config.get("confidence_threshold", 0.8)
            final_confidence = min(base_confidence, mc_confidence)
            
            if final_confidence >= confidence_threshold:
                return SynergyDetection(
                    detected=True,
                    confidence=final_confidence,
                    synergy_type=self._identify_synergy_type(inputs),
                    strength_score=self._calculate_synergy_strength(inputs),
                    contributing_agents=self._identify_contributing_agents(inputs),
                    raw_signals=self._extract_raw_signals(inputs)
                )
                
        return SynergyDetection(
            detected=False,
            confidence=base_confidence,
            synergy_type="none",
            strength_score=0.0,
            contributing_agents=[],
            raw_signals={}
        )
        
    async def _stage2_execution_approval(self, inputs: AgentInputs, 
                                       synergy: SynergyDetection) -> ExecutionDecision:
        """Stage 2: Risk management and final execution approval"""
        
        # Request risk proposal from M-RMS
        risk_proposal = await self._request_risk_proposal(inputs, synergy)
        
        if risk_proposal is None or risk_proposal.position_size == 0:
            return ExecutionDecision(
                action="REJECT",
                confidence=0.0,
                reasoning=["Risk management rejected proposal"],
                risk_proposal=risk_proposal,
                decision_factors={"risk_rejected": True}
            )
            
        # Create comprehensive decision state
        decision_state = self.state_assembler.create_decision_gate_state(
            inputs, risk_proposal
        )
        
        # Run decision gate policy
        gate_decision = await self.decision_gate.make_final_decision(
            decision_state, synergy, risk_proposal
        )
        
        # Compile decision factors
        decision_factors = {
            "synergy_confidence": synergy.confidence,
            "synergy_strength": synergy.strength_score,
            "risk_confidence": risk_proposal.confidence_score,
            "position_size": risk_proposal.position_size,
            "risk_reward_ratio": risk_proposal.risk_reward_ratio,
            "constraint_violations": len(risk_proposal.constraint_violations),
            "gate_confidence": gate_decision.confidence
        }
        
        return ExecutionDecision(
            action=gate_decision.action,
            confidence=gate_decision.confidence,
            reasoning=gate_decision.reasoning,
            risk_proposal=risk_proposal,
            decision_factors=decision_factors
        )
        
    def _identify_synergy_type(self, inputs: AgentInputs) -> str:
        """Identify the type of synergy detected"""
        
        # Analyze contributing timeframes and regime
        regime_state = inputs.regime_vector
        
        # Simple heuristic for synergy classification
        if np.mean(inputs.vector_30m[-10:]) > 0.5:  # Strong 30m signal
            if np.mean(inputs.vector_5m[-10:]) > 0.5:  # Confirmed by 5m
                return "multi_timeframe_bullish"
            else:
                return "30m_bullish"
        elif np.mean(inputs.vector_30m[-10:]) < -0.5:  # Strong 30m bearish
            if np.mean(inputs.vector_5m[-10:]) < -0.5:  # Confirmed by 5m
                return "multi_timeframe_bearish"
            else:
                return "30m_bearish"
        else:
            return "regime_change"
            
    def _calculate_synergy_strength(self, inputs: AgentInputs) -> float:
        """Calculate overall strength of detected synergy"""
        
        # Combine signals from different timeframes
        tf_30m_strength = abs(np.mean(inputs.vector_30m[-5:]))  # Recent strength
        tf_5m_strength = abs(np.mean(inputs.vector_5m[-5:]))
        regime_clarity = 1.0 - np.std(inputs.regime_vector)  # Lower std = clearer regime
        
        # Weighted combination
        strength = (
            tf_30m_strength * 0.4 +
            tf_5m_strength * 0.3 +
            regime_clarity * 0.3
        )
        
        return min(1.0, max(0.0, strength))
        
    async def _request_risk_proposal(self, inputs: AgentInputs, 
                                   synergy: SynergyDetection) -> Any:
        """Request risk proposal from M-RMS"""
        
        # Convert inputs to M-RMS format
        synergy_vector = self._create_synergy_vector(inputs, synergy)
        account_state = self._create_account_state(inputs)
        
        # Call M-RMS (assuming it's injected as dependency)
        if hasattr(self, 'risk_manager'):
            return await self.risk_manager.get_risk_plan(
                synergy_vector, account_state
            )
        else:
            logger.warning("Risk manager not available, using fallback")
            return self._create_fallback_risk_proposal()

class MCDropoutValidator:
    """Monte Carlo Dropout confidence validation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_samples = config.get("mc_dropout_samples", 30)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        
    async def validate_confidence(self, model, state: torch.Tensor) -> float:
        """Validate decision confidence using MC Dropout"""
        
        # Ensure model is in training mode for dropout
        model.train()
        
        predictions = []
        
        # Run multiple forward passes with dropout active
        with torch.no_grad():
            for _ in range(self.num_samples):
                # Forward pass with dropout active
                if hasattr(model, 'model'):
                    # Ray RLlib model
                    logits, _ = model.model({"obs": state.unsqueeze(0)}, [], None)
                    probs = torch.softmax(logits, dim=-1)
                else:
                    # Direct PyTorch model
                    logits = model(state.unsqueeze(0))
                    probs = torch.softmax(logits, dim=-1)
                    
                predictions.append(probs.cpu().numpy())
                
        # Calculate confidence statistics
        predictions = np.array(predictions)  # [num_samples, batch_size, num_actions]
        
        # Mean prediction and variance
        mean_probs = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        # Confidence based on consistency across samples
        # High confidence = low variance in predictions
        max_prob = np.max(mean_probs)
        prediction_variance = np.max(variance)
        
        # Confidence score: high if predictions are consistent and confident
        confidence = max_prob * (1.0 - prediction_variance * 10.0)  # Scale variance
        confidence = min(1.0, max(0.0, confidence))
        
        return float(confidence)

class DecisionGate:
    """Final decision gate for execution approval"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.decision_model = self._load_decision_model(config)
        
    async def make_final_decision(self, decision_state: torch.Tensor,
                                synergy: SynergyDetection,
                                risk_proposal: Any) -> ExecutionDecision:
        """Make final execution decision"""
        
        with torch.no_grad():
            # Forward pass through decision gate
            if hasattr(self.decision_model, 'compute_single_action'):
                action = self.decision_model.compute_single_action(
                    decision_state.numpy(), explore=False
                )
            else:
                logits = self.decision_model(decision_state.unsqueeze(0))
                action = torch.argmax(logits, dim=-1).item()
                
        # Convert action to decision
        action_name = "EXECUTE" if action == 1 else "REJECT"
        
        # Calculate confidence based on various factors
        confidence = self._calculate_decision_confidence(
            synergy, risk_proposal, decision_state
        )
        
        # Generate reasoning
        reasoning = self._generate_decision_reasoning(
            action_name, synergy, risk_proposal
        )
        
        return ExecutionDecision(
            action=action_name,
            confidence=confidence,
            reasoning=reasoning,
            risk_proposal=risk_proposal,
            decision_factors={}
        )
        
    def _calculate_decision_confidence(self, synergy: SynergyDetection,
                                     risk_proposal: Any,
                                     decision_state: torch.Tensor) -> float:
        """Calculate overall decision confidence"""
        
        # Combine multiple confidence sources
        synergy_conf = synergy.confidence
        risk_conf = risk_proposal.confidence_score if risk_proposal else 0.0
        
        # Penalty for constraint violations
        violation_penalty = len(risk_proposal.constraint_violations) * 0.1 if risk_proposal else 0.0
        
        # Combined confidence
        confidence = (synergy_conf * 0.6 + risk_conf * 0.4) - violation_penalty
        
        return min(1.0, max(0.0, confidence))
        
    def _generate_decision_reasoning(self, action: str,
                                   synergy: SynergyDetection,
                                   risk_proposal: Any) -> List[str]:
        """Generate human-readable decision reasoning"""
        
        reasoning = []
        
        if action == "EXECUTE":
            reasoning.append(f"High-confidence {synergy.synergy_type} detected")
            reasoning.append(f"Synergy strength: {synergy.strength_score:.2f}")
            if risk_proposal:
                reasoning.append(f"Risk-reward ratio: {risk_proposal.risk_reward_ratio:.1f}:1")
                reasoning.append(f"Position size: {risk_proposal.position_size} contracts")
        else:
            if synergy.confidence < 0.8:
                reasoning.append("Insufficient synergy confidence")
            if risk_proposal and risk_proposal.position_size == 0:
                reasoning.append("Risk management rejected proposal")
            if risk_proposal and len(risk_proposal.constraint_violations) > 0:
                reasoning.append("Risk constraint violations detected")
                
        return reasoning

@dataclass
class MARLCoreMetrics:
    """Performance metrics for Main MARL Core"""
    
    decisions_processed: int = 0
    synergies_detected: int = 0
    trades_approved: int = 0
    average_confidence: float = 0.0
    average_processing_time_ms: float = 0.0
    error_count: int = 0
    confidence_rejections: int = 0
    risk_rejections: int = 0
    
    def update_decision_metrics(self, decision: ExecutionDecision):
        """Update metrics with new decision"""
        
        self.decisions_processed += 1
        
        # Update running averages
        alpha = 0.1  # Exponential moving average factor
        self.average_confidence = (
            alpha * decision.confidence + 
            (1 - alpha) * self.average_confidence
        )
        self.average_processing_time_ms = (
            alpha * decision.processing_time_ms +
            (1 - alpha) * self.average_processing_time_ms
        )
        
        # Count specific decision types
        if decision.action == "EXECUTE":
            self.trades_approved += 1
        
        if "confidence" in decision.reasoning[0].lower():
            self.confidence_rejections += 1
        elif "risk" in decision.reasoning[0].lower():
            self.risk_rejections += 1
```

### FR-MMC-03: Event Processing and Integration
**Requirement**: The Main MARL Core MUST integrate seamlessly with the event-driven system architecture.

**Specification**:
```python
class MARLCoreEventProcessor:
    """Event processing integration for Main MARL Core"""
    
    def __init__(self, core: MainMARLCore, event_bus):
        self.core = core
        self.event_bus = event_bus
        
        # Subscribe to required events
        self.event_bus.subscribe("INDICATORS_READY", self.on_indicators_ready)
        self.event_bus.subscribe("MATRIX_UPDATED", self.on_matrix_updated)
        
        # State management
        self.current_inputs = AgentInputs(
            vector_30m=np.zeros(128),
            vector_5m=np.zeros(128),
            regime_vector=np.zeros(8),
            account_state=np.zeros(16),
            lvn_strength=np.zeros(12),
            market_context={}
        )
        
        self.ready_components = set()
        self.decision_pending = False
        
    async def on_indicators_ready(self, event_type: str, payload: Dict[str, Any]):
        """Process INDICATORS_READY event"""
        
        # Update regime vector if available
        if "regime_vector" in payload:
            self.current_inputs.regime_vector = payload["regime_vector"]
            self.ready_components.add("regime")
            
        # Update LVN strength data
        if "lvn_data" in payload:
            self.current_inputs.lvn_strength = payload["lvn_data"]
            self.ready_components.add("lvn")
            
        # Update market context
        self.current_inputs.market_context.update(payload.get("market_context", {}))
        
        # Check if we have enough data for decision
        await self._check_decision_readiness()
        
    async def on_matrix_updated(self, event_type: str, payload: Dict[str, Any]):
        """Process MATRIX_UPDATED event from MatrixAssemblers"""
        
        matrix_name = payload.get("matrix_name", "")
        matrix_data = payload.get("matrix_data", np.array([]))
        
        if matrix_name == "agent_30m":
            # Extract latest embedding from 30m agent matrix
            if len(matrix_data) > 0:
                self.current_inputs.vector_30m = matrix_data[-1]  # Latest row
                self.ready_components.add("30m")
                
        elif matrix_name == "agent_5m":
            # Extract latest embedding from 5m agent matrix
            if len(matrix_data) > 0:
                self.current_inputs.vector_5m = matrix_data[-1]  # Latest row
                self.ready_components.add("5m")
                
        # Check if we have enough data for decision
        await self._check_decision_readiness()
        
    async def _check_decision_readiness(self):
        """Check if we have sufficient data to make a trading decision"""
        
        required_components = {"30m", "5m", "regime"}
        
        if (required_components.issubset(self.ready_components) and 
            not self.decision_pending):
            
            self.decision_pending = True
            
            try:
                # Process trading opportunity
                decision = await self.core.process_trading_opportunity(self.current_inputs)
                
                # Emit execution event if approved
                if decision.action == "EXECUTE" and decision.risk_proposal:
                    await self.event_bus.publish(
                        event_type="EXECUTE_TRADE",
                        payload=decision.risk_proposal,
                        priority="HIGH",
                        timestamp=datetime.now(),
                        metadata={
                            "confidence": decision.confidence,
                            "reasoning": decision.reasoning,
                            "processing_time_ms": decision.processing_time_ms
                        }
                    )
                    
                    logger.info(f"Trade execution approved: {decision.reasoning}")
                    
                else:
                    logger.info(f"Trade rejected: {decision.reasoning}")
                    
                # Emit decision event for monitoring
                await self.event_bus.publish(
                    event_type="TRADING_DECISION",
                    payload=decision,
                    priority="MEDIUM",
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"Decision processing error: {e}")
                
            finally:
                self.decision_pending = False
                
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for monitoring"""
        
        return {
            "ready_components": list(self.ready_components),
            "decision_pending": self.decision_pending,
            "inputs_valid": self._validate_current_inputs(),
            "metrics": self.core.metrics.__dict__,
            "last_decision_time": self.core.decision_history[-1].get("timestamp") if self.core.decision_history else None
        }
        
    def _validate_current_inputs(self) -> bool:
        """Validate current inputs are reasonable"""
        
        try:
            # Check for NaN or infinite values
            for field_name, field_value in self.current_inputs.__dict__.items():
                if isinstance(field_value, np.ndarray):
                    if np.any(np.isnan(field_value)) or np.any(np.isinf(field_value)):
                        return False
                        
            return True
            
        except Exception:
            return False
```

## 3.0 Interface Specifications

### 3.1 Configuration Interface
```yaml
main_marl_core:
  model:
    hidden_dim: 256
    num_layers: 3
    dropout_rate: 0.2
    attention_heads: 8
    use_attention: true
    
  decision_process:
    confidence_threshold: 0.8
    mc_dropout_samples: 30
    max_processing_time_ms: 20
    
  vector_dimensions:
    vector_30m_dim: 128
    vector_5m_dim: 128
    regime_dim: 8
    account_dim: 16
    lvn_dim: 12
    
  training:
    algorithm: "MAPPO"
    learning_rate: 0.0003
    gamma: 0.99
    lambda: 0.95
    train_batch_size: 8000
    
  optimization:
    enable_tensorrt: true
    fp16_precision: true
    batch_inference: false
    
  monitoring:
    log_all_decisions: true
    decision_history_size: 1000
    performance_metrics_interval: 60
```

### 3.2 Event Interface

**Subscribed Events**:
- **INDICATORS_READY**: Feature data from IndicatorEngine
- **MATRIX_UPDATED**: State vectors from MatrixAssemblers

**Published Events**:
- **EXECUTE_TRADE**: Approved trades with risk proposals
- **TRADING_DECISION**: All decisions for monitoring
- **MARL_CORE_STATUS**: System health and performance

### 3.3 API Interface
```python
class MARLCoreAPI:
    def process_trading_opportunity(self, inputs: AgentInputs) -> ExecutionDecision
    def get_performance_metrics(self) -> MARLCoreMetrics
    def get_decision_history(self, limit: int = 100) -> List[Dict]
    def update_confidence_threshold(self, threshold: float) -> None
    def get_system_status(self) -> Dict[str, Any]
    def force_decision_cycle(self) -> ExecutionDecision
```

## 4.0 Dependencies & Interactions

### 4.1 Upstream Dependencies
- **MatrixAssemblers**: Source of processed state vectors
- **IndicatorEngine**: Source of feature data and LVN information
- **Regime Detection Engine**: Market regime classification
- **Account Management**: Current account state and position data

### 4.2 Internal Dependencies
- **Risk Management Sub-system**: Risk proposal generation
- **Event Bus**: Event processing and communication
- **Configuration System**: Model parameters and thresholds

### 4.3 Downstream Dependencies
- **ExecutionHandler**: Consumer of approved trade executions
- **Performance Monitor**: Consumer of decision metrics
- **Research Tools**: Consumer of decision analysis data

## 5.0 Non-Functional Requirements

### 5.1 Performance
- **NFR-MMC-01**: Complete decision cycle MUST finish under 20ms (95th percentile)
- **NFR-MMC-02**: MC Dropout validation MUST complete under 10ms
- **NFR-MMC-03**: Memory usage MUST remain stable during continuous operation
- **NFR-MMC-04**: Support 1000+ decision cycles per hour

### 5.2 Accuracy and Reliability
- **NFR-MMC-05**: Decision confidence calibration MUST be accurate (±5%)
- **NFR-MMC-06**: System MUST handle component failures gracefully
- **NFR-MMC-07**: All decisions MUST be fully auditable and explainable
- **NFR-MMC-08**: Error recovery MUST complete within 1 second

### 5.3 Monitoring and Observability
- **NFR-MMC-09**: All decisions MUST be logged with full context
- **NFR-MMC-10**: Performance metrics MUST be updated in real-time
- **NFR-MMC-11**: System health MUST be continuously monitored

## 6.0 Testing Requirements

### 6.1 Unit Tests
- State vector assembly and normalization
- MC Dropout confidence calculation
- Decision gate logic and reasoning
- Event processing and integration

### 6.2 Integration Tests
- End-to-end decision processing pipeline
- Multi-agent state integration
- Event bus integration and timing
- Error handling and recovery scenarios

### 6.3 Performance Tests
```python
async def test_decision_cycle_latency():
    """Test complete decision cycle meets latency requirements"""
    core = MainMARLCore(config)
    
    # Create test inputs
    inputs = create_test_agent_inputs()
    
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        decision = await core.process_trading_opportunity(inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    p95_latency = np.percentile(latencies, 95)
    assert p95_latency < 20.0, f"Decision latency too high: {p95_latency}ms"

def test_mc_dropout_accuracy():
    """Test MC Dropout confidence calibration"""
    validator = MCDropoutValidator(config)
    model = create_test_model()
    
    # Test on known confident/uncertain cases
    confident_state = create_confident_test_state()
    uncertain_state = create_uncertain_test_state()
    
    conf_confident = await validator.validate_confidence(model, confident_state)
    conf_uncertain = await validator.validate_confidence(model, uncertain_state)
    
    assert conf_confident > 0.8, "Should be confident on clear signals"
    assert conf_uncertain < 0.6, "Should be uncertain on unclear signals"
```

## 7.0 Future Enhancements

### 7.1 V2.0 Features
- **Hierarchical Decision Making**: Multi-level decision trees for complex scenarios
- **Dynamic Confidence Thresholds**: Regime-adaptive confidence requirements
- **Advanced Ensembling**: Multiple model consensus for critical decisions
- **Causal Decision Analysis**: Understanding decision factor relationships
- **Real-time Model Updates**: Online learning and adaptation

### 7.2 Research Directions
- **Explainable AI**: Advanced decision explanation and visualization
- **Meta-Learning**: Learning to learn from new market conditions
- **Adversarial Robustness**: Protection against adversarial market conditions
- **Multi-Objective Optimization**: Balancing multiple performance criteria
- **Attention Mechanisms**: Understanding what the model focuses on

This PRD establishes the foundation for a sophisticated, multi-agent decision-making system that can integrate complex market analysis from multiple sources into precise, confident trading decisions while maintaining the transparency and reliability required for professional algorithmic trading operations.