"""
Main MARL Core Engine Component.

This module provides the high-level orchestration for the unified
intelligence system, implementing the two-gate decision flow with
MC Dropout consensus and integrated risk management.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

from .models import (
    StructureEmbedder,
    TacticalEmbedder,
    RegimeEmbedder,
    LVNEmbedder,
    SharedPolicy,
    DecisionGate,
    MCDropoutEvaluator
)

logger = logging.getLogger(__name__)


class MainMARLCoreComponent:
    """
    High-level orchestrator for the Main MARL Core.
    
    This component implements the two-gate decision flow:
    1. Gate 1: Unified state evaluation with MC Dropout consensus
    2. Gate 2: Final decision with risk proposal integration
    
    The system processes synergy events through specialized embedders,
    evaluates them with a shared policy, and makes final trading decisions.
    """
    
    def __init__(self, config: Dict[str, Any], components: Dict[str, Any]):
        """
        Initialize the Main MARL Core component.
        
        Args:
            config: Configuration dictionary containing:
                - embedders: Configuration for each embedder
                - shared_policy: SharedPolicy network config
                - mc_dropout: MC Dropout evaluation config
                - decision_gate: DecisionGate config
                - device: Computing device
            components: Dictionary of system components (rde, m_rms, etc.)
        """
        self.config = config
        self.components = components
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Initialize embedders
        embedder_config = config.get('embedders', {})
        
        self.structure_embedder = StructureEmbedder(
            input_channels=8,
            output_dim=embedder_config.get('structure', {}).get('output_dim', 64),
            dropout_rate=embedder_config.get('structure', {}).get('dropout', 0.2)
        ).to(self.device)
        
        self.tactical_embedder = TacticalEmbedder(
            input_dim=7,
            hidden_dim=embedder_config.get('tactical', {}).get('hidden_dim', 64),
            output_dim=embedder_config.get('tactical', {}).get('output_dim', 48),
            dropout_rate=embedder_config.get('tactical', {}).get('dropout', 0.2)
        ).to(self.device)
        
        self.regime_embedder = RegimeEmbedder(
            input_dim=8,
            output_dim=embedder_config.get('regime', {}).get('output_dim', 16),
            hidden_dim=embedder_config.get('regime', {}).get('hidden_dim', 32)
        ).to(self.device)
        
        self.lvn_embedder = LVNEmbedder(
            input_dim=embedder_config.get('lvn', {}).get('input_dim', 5),
            output_dim=embedder_config.get('lvn', {}).get('output_dim', 8),
            hidden_dim=embedder_config.get('lvn', {}).get('hidden_dim', 16)
        ).to(self.device)
        
        # Calculate unified state dimension
        unified_dim = (
            embedder_config.get('structure', {}).get('output_dim', 64) +
            embedder_config.get('tactical', {}).get('output_dim', 48) +
            embedder_config.get('regime', {}).get('output_dim', 16) +
            embedder_config.get('lvn', {}).get('output_dim', 8)
        )
        
        # Initialize shared policy
        policy_config = config.get('shared_policy', {})
        self.shared_policy = SharedPolicy(
            input_dim=unified_dim,
            hidden_dims=policy_config.get('hidden_dims', [256, 128, 64]),
            dropout_rate=policy_config.get('dropout', 0.2),
            action_dim=2
        ).to(self.device)
        
        # Initialize decision gate
        gate_config = config.get('decision_gate', {})
        self.decision_gate = DecisionGate(
            input_dim=unified_dim + 8,  # Unified state + risk vector
            hidden_dim=gate_config.get('hidden_dim', 64),
            dropout_rate=gate_config.get('dropout', 0.1)
        ).to(self.device)
        
        # Initialize MC Dropout evaluator
        mc_config = config.get('mc_dropout', {})
        self.mc_evaluator = MCDropoutEvaluator(
            n_passes=mc_config.get('n_passes', 50)
        )
        self.confidence_threshold = mc_config.get('confidence_threshold', 0.65)
        
        # Set all models to eval mode by default
        self.eval_mode()
        
        # Track model loading status
        self.models_loaded = False
        
        # Performance tracking
        self.decision_count = 0
        self.execution_count = 0
        
        logger.info(
            f"Main MARL Core initialized",
            unified_dim=unified_dim,
            device=self.device
        )
        
    def load_models(self) -> None:
        """
        Load pre-trained weights for all neural network components.
        
        Loads weights from paths specified in configuration.
        """
        model_paths = self.config.get('model_paths', {})
        
        # Load embedder weights
        embedder_models = {
            'structure_embedder': self.structure_embedder,
            'tactical_embedder': self.tactical_embedder,
            'regime_embedder': self.regime_embedder,
            'lvn_embedder': self.lvn_embedder
        }
        
        for name, model in embedder_models.items():
            path = model_paths.get(name)
            if path and Path(path).exists():
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    logger.info(f"Loaded {name} from {path}")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
            else:
                logger.warning(f"No model path found for {name}")
        
        # Load shared policy
        policy_path = model_paths.get('shared_policy')
        if policy_path and Path(policy_path).exists():
            try:
                checkpoint = torch.load(policy_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.shared_policy.load_state_dict(checkpoint['state_dict'])
                else:
                    self.shared_policy.load_state_dict(checkpoint)
                logger.info(f"Loaded shared policy from {policy_path}")
            except Exception as e:
                logger.error(f"Failed to load shared policy: {e}")
        
        # Load decision gate
        gate_path = model_paths.get('decision_gate')
        if gate_path and Path(gate_path).exists():
            try:
                checkpoint = torch.load(gate_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.decision_gate.load_state_dict(checkpoint['state_dict'])
                else:
                    self.decision_gate.load_state_dict(checkpoint)
                logger.info(f"Loaded decision gate from {gate_path}")
            except Exception as e:
                logger.error(f"Failed to load decision gate: {e}")
        
        self.models_loaded = True
        self.eval_mode()
        
    def _prepare_unified_state(self, synergy_event: Dict[str, Any]) -> torch.Tensor:
        """
        Prepare the unified state vector from all data sources.
        
        Args:
            synergy_event: Synergy detection event data
            
        Returns:
            Unified state tensor ready for policy evaluation
        """
        try:
            # Get matrix assembler components
            matrix_30m = self.components.get('matrix_30m')
            matrix_5m = self.components.get('matrix_5m')
            rde = self.components.get('rde')
            
            if not all([matrix_30m, matrix_5m, rde]):
                raise RuntimeError("Required components not available")
            
            # Get data from matrix assemblers
            structure_matrix = matrix_30m.get_matrix()  # [48, 8]
            tactical_matrix = matrix_5m.get_matrix()    # [60, 7]
            
            # Get regime vector
            regime_vector = rde.get_regime_vector()     # [8]
            
            # Extract LVN features from synergy context
            lvn_features = self._extract_lvn_features(synergy_event)  # [5]
            
            # Convert to tensors and add batch dimension
            structure_tensor = torch.tensor(
                structure_matrix, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            tactical_tensor = torch.tensor(
                tactical_matrix, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            regime_tensor = torch.tensor(
                regime_vector, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            lvn_tensor = torch.tensor(
                lvn_features, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Process through embedders
            with torch.no_grad():
                structure_embedded = self.structure_embedder(structure_tensor)
                tactical_embedded = self.tactical_embedder(tactical_tensor)
                regime_embedded = self.regime_embedder(regime_tensor)
                lvn_embedded = self.lvn_embedder(lvn_tensor)
            
            # Concatenate all embeddings
            unified_state = torch.cat([
                structure_embedded,
                tactical_embedded,
                regime_embedded,
                lvn_embedded
            ], dim=-1)
            
            return unified_state
            
        except Exception as e:
            logger.error(f"Failed to prepare unified state: {e}")
            raise
    
    def _extract_lvn_features(self, synergy_event: Dict[str, Any]) -> np.ndarray:
        """
        Extract LVN-related features from synergy event.
        
        Args:
            synergy_event: Synergy detection event
            
        Returns:
            NumPy array of LVN features [5]
        """
        market_context = synergy_event.get('market_context', {})
        lvn_context = market_context.get('nearest_lvn', {})
        
        # Extract features with defaults
        lvn_price = lvn_context.get('price', 0.0)
        lvn_strength = lvn_context.get('strength', 0.0) / 100.0  # Normalize
        lvn_distance = lvn_context.get('distance', 100.0)
        current_price = market_context.get('current_price', 0.0)
        
        # Calculate relative position
        if current_price > 0 and lvn_price > 0:
            relative_position = (current_price - lvn_price) / current_price
        else:
            relative_position = 0.0
        
        # Normalize distance
        distance_normalized = min(abs(lvn_distance) / 50.0, 1.0)
        
        # Create feature vector
        features = np.array([
            lvn_strength,
            distance_normalized,
            relative_position,
            1.0 if current_price > lvn_price else 0.0,  # Above LVN indicator
            lvn_strength * (1 - distance_normalized)     # Combined score
        ], dtype=np.float32)
        
        return features
    
    def _run_mc_dropout_consensus(self, unified_state: torch.Tensor) -> Dict[str, Any]:
        """
        Run MC Dropout consensus evaluation on the unified state.
        
        Args:
            unified_state: Prepared unified state tensor
            
        Returns:
            Consensus results dictionary
        """
        return self.mc_evaluator.evaluate(
            self.shared_policy,
            unified_state,
            self.confidence_threshold
        )
    
    def _vectorize_risk_proposal(self, risk_proposal: Dict[str, Any]) -> np.ndarray:
        """
        Convert M-RMS risk proposal to vector format.
        
        Args:
            risk_proposal: Risk proposal from M-RMS
            
        Returns:
            Risk vector [8]
        """
        # Extract key risk metrics
        position_size = risk_proposal.get('position_size', 0)
        sl_atr_multiplier = risk_proposal.get('sl_atr_multiplier', 1.5)
        risk_reward_ratio = risk_proposal.get('risk_reward_ratio', 2.0)
        confidence_score = risk_proposal.get('confidence_score', 0.5)
        
        risk_metrics = risk_proposal.get('risk_metrics', {})
        position_utilization = risk_metrics.get('position_utilization', 0.0)
        
        # Normalize position size (0-5 contracts to 0-1)
        position_size_norm = position_size / 5.0
        
        # Create risk vector
        risk_vector = np.array([
            position_size_norm,
            sl_atr_multiplier / 3.0,      # Normalize (0.5-3.0 to ~0.17-1.0)
            risk_reward_ratio / 5.0,      # Normalize (1-5 to 0.2-1.0)
            confidence_score,
            position_utilization,
            1.0 if position_size > 0 else 0.0,  # Trade active indicator
            min(position_size / 3.0, 1.0),      # Conservative size indicator
            0.5  # Reserved for future use
        ], dtype=np.float32)
        
        return risk_vector
    
    def initiate_qualification(self, synergy_event: Dict[str, Any]) -> None:
        """
        Main entry point for processing synergy events.
        
        This method orchestrates the entire two-gate decision flow:
        1. Prepare unified state from all data sources
        2. Run MC Dropout consensus evaluation
        3. Conditionally call M-RMS for risk proposal
        4. Make final EXECUTE/REJECT decision
        5. Emit trading command if approved
        
        Args:
            synergy_event: SYNERGY_DETECTED event data
        """
        start_time = datetime.now()
        self.decision_count += 1
        
        try:
            logger.info(
                f"Processing synergy event",
                synergy_type=synergy_event.get('synergy_type'),
                direction=synergy_event.get('direction')
            )
            
            # Step 1: Prepare unified state
            unified_state = self._prepare_unified_state(synergy_event)
            
            # Step 2: Run MC Dropout consensus
            consensus_result = self._run_mc_dropout_consensus(unified_state)
            
            # Log consensus results
            logger.info(
                f"MC Dropout consensus",
                confidence=consensus_result['confidence'].item(),
                should_proceed=consensus_result['should_proceed'].item(),
                uncertainty=consensus_result['uncertainty_metrics']
            )
            
            # Step 3: Check if consensus is met
            if not consensus_result['should_proceed'].item():
                logger.info(
                    f"Trade rejected at Gate 1",
                    reason="Insufficient consensus confidence",
                    confidence=consensus_result['confidence'].item()
                )
                self._emit_rejection(synergy_event, 'gate1', consensus_result)
                return
            
            # Step 4: Generate trade qualification
            trade_qualification = self._create_trade_qualification(
                synergy_event,
                consensus_result
            )
            
            # Step 5: Get risk proposal from M-RMS
            m_rms = self.components.get('m_rms')
            if not m_rms:
                raise RuntimeError("M-RMS component not available")
            
            risk_proposal = m_rms.generate_risk_proposal(trade_qualification)
            
            # Step 6: Vectorize risk proposal
            risk_vector = self._vectorize_risk_proposal(risk_proposal)
            risk_tensor = torch.tensor(
                risk_vector, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Step 7: Assemble final gate state
            final_gate_state = torch.cat([unified_state, risk_tensor], dim=-1)
            
            # Step 8: Run decision gate
            with torch.no_grad():
                gate_decision = self.decision_gate(final_gate_state)
            
            execute_prob = gate_decision['execute_probability'].item()
            should_execute = execute_prob > 0.5
            
            logger.info(
                f"Decision Gate result",
                execute_probability=execute_prob,
                should_execute=should_execute
            )
            
            # Step 9: Make final decision
            if should_execute:
                # Emit EXECUTE_TRADE command
                self._emit_trade_command(
                    synergy_event,
                    consensus_result,
                    risk_proposal,
                    gate_decision,
                    start_time
                )
                self.execution_count += 1
            else:
                # Log rejection at Gate 2
                logger.info(
                    f"Trade rejected at Gate 2",
                    reason="Failed final validation",
                    execute_probability=execute_prob
                )
                self._emit_rejection(
                    synergy_event, 
                    'gate2', 
                    gate_decision,
                    risk_proposal
                )
                
        except Exception as e:
            logger.error(f"Error in trade qualification: {e}", exc_info=True)
            self._handle_error(synergy_event, e)
    
    def _create_trade_qualification(
        self,
        synergy_event: Dict[str, Any],
        consensus_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create trade qualification for M-RMS.
        
        Args:
            synergy_event: Original synergy event
            consensus_result: MC Dropout consensus results
            
        Returns:
            Trade qualification dictionary
        """
        # Get market context
        market_context = synergy_event.get('market_context', {})
        
        # Prepare synergy vector (30 features as expected by M-RMS)
        synergy_vector = self._create_synergy_vector(synergy_event)
        
        # Prepare account state vector (10 features)
        account_vector = self._create_account_vector()
        
        return {
            'synergy_vector': synergy_vector,
            'account_state_vector': account_vector,
            'entry_price': market_context.get('current_price', 0.0),
            'direction': 'LONG' if synergy_event.get('direction', 1) > 0 else 'SHORT',
            'atr': market_context.get('atr', 10.0),
            'symbol': synergy_event.get('symbol', 'ES'),
            'timestamp': synergy_event.get('timestamp', datetime.now()),
            'consensus_confidence': consensus_result['confidence'].item(),
            'synergy_type': synergy_event.get('synergy_type', 'TYPE_1')
        }
    
    def _create_synergy_vector(self, synergy_event: Dict[str, Any]) -> np.ndarray:
        """
        Create 30-feature synergy vector for M-RMS.
        
        Args:
            synergy_event: Synergy event data
            
        Returns:
            Synergy feature vector [30]
        """
        features = []
        
        # Synergy type encoding (4 features)
        synergy_type = synergy_event.get('synergy_type', 'TYPE_1')
        type_encoding = [0.0] * 4
        type_map = {'TYPE_1': 0, 'TYPE_2': 1, 'TYPE_3': 2, 'TYPE_4': 3}
        if synergy_type in type_map:
            type_encoding[type_map[synergy_type]] = 1.0
        features.extend(type_encoding)
        
        # Direction and strength (3 features)
        direction = synergy_event.get('direction', 1)
        signal_strengths = synergy_event.get('signal_strengths', {})
        features.extend([
            1.0 if direction > 0 else 0.0,
            abs(direction),
            signal_strengths.get('overall', 0.5)
        ])
        
        # Market context features (10 features)
        market_context = synergy_event.get('market_context', {})
        features.extend([
            market_context.get('volatility', 10.0) / 50.0,
            market_context.get('volume_ratio', 1.0) / 2.0,
            market_context.get('price_momentum_5', 0.0) / 10.0,
            market_context.get('rsi', 50.0) / 100.0,
            market_context.get('spread', 0.25) / 1.0,
            0.5,  # Placeholder
            0.5,  # Placeholder
            0.5,  # Placeholder
            0.5,  # Placeholder
            0.5   # Placeholder
        ])
        
        # Signal sequence features (10 features)
        signal_sequence = synergy_event.get('signal_sequence', [])
        for i in range(3):  # MLMI, NWRQK, FVG
            if i < len(signal_sequence):
                signal = signal_sequence[i]
                features.extend([
                    signal.get('value', 0.0) / 100.0,
                    signal.get('strength', 0.5),
                    1.0 if signal.get('confirmed', False) else 0.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        features.append(0.5)  # Placeholder for 10th feature
        
        # Timing features (3 features)
        metadata = synergy_event.get('metadata', {})
        features.extend([
            metadata.get('bars_to_complete', 5) / 10.0,
            1.0,  # Placeholder
            0.5   # Placeholder
        ])
        
        # Ensure exactly 30 features
        features = features[:30]  # Trim if over
        while len(features) < 30:  # Pad if under
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _create_account_vector(self) -> np.ndarray:
        """
        Create account state vector for M-RMS.
        
        Returns:
            Account state vector [10]
        """
        # TODO: Get actual account state from system
        # For now, return reasonable defaults
        return np.array([
            1.0,   # Balance ratio (current/initial)
            0.0,   # Current drawdown
            0.0,   # Max drawdown
            0.5,   # Win rate
            1.0,   # Profit factor
            0.0,   # Sortino ratio
            0.1,   # Trade count / 100
            0.0,   # Recent performance
            0.5,   # Reserved
            0.5    # Reserved
        ], dtype=np.float32)
    
    def _emit_trade_command(
        self,
        synergy_event: Dict[str, Any],
        consensus_result: Dict[str, Any],
        risk_proposal: Dict[str, Any],
        gate_decision: Dict[str, Any],
        start_time: datetime
    ) -> None:
        """
        Emit EXECUTE_TRADE command to the system.
        
        Args:
            synergy_event: Original synergy event
            consensus_result: MC Dropout results
            risk_proposal: M-RMS risk proposal
            gate_decision: Decision gate results
            start_time: Processing start time
        """
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        execute_command = {
            'event_type': 'EXECUTE_TRADE',
            'execution_id': self._generate_execution_id(),
            'timestamp': datetime.now(),
            
            'trade_specification': {
                'symbol': synergy_event.get('symbol', 'ES'),
                'direction': synergy_event.get('direction'),
                'entry_price': risk_proposal.get('entry_price'),
                'synergy_type': synergy_event.get('synergy_type')
            },
            
            'risk_parameters': {
                'position_size': risk_proposal.get('position_size'),
                'stop_loss': risk_proposal.get('stop_loss_price'),
                'take_profit': risk_proposal.get('take_profit_price'),
                'risk_amount': risk_proposal.get('risk_amount'),
                'reward_amount': risk_proposal.get('reward_amount')
            },
            
            'decision_metadata': {
                'consensus_confidence': consensus_result['confidence'].item(),
                'mc_dropout_metrics': consensus_result['uncertainty_metrics'],
                'execute_probability': gate_decision['execute_probability'].item(),
                'processing_time_ms': processing_time
            },
            
            'tracking_data': {
                'risk_reward_ratio': risk_proposal.get('risk_reward_ratio'),
                'expected_value': risk_proposal.get('reward_amount', 0) * 0.6 - 
                                 risk_proposal.get('risk_amount', 0) * 0.4,
                'decision_number': self.decision_count,
                'execution_number': self.execution_count
            }
        }
        
        # Emit through event bus
        event_bus = self.components.get('kernel').event_bus
        event_bus.emit('EXECUTE_TRADE', execute_command)
        
        logger.info(
            f"Trade command emitted",
            execution_id=execute_command['execution_id'],
            position_size=risk_proposal.get('position_size'),
            risk_reward=risk_proposal.get('risk_reward_ratio')
        )
    
    def _emit_rejection(
        self,
        synergy_event: Dict[str, Any],
        stage: str,
        decision_data: Dict[str, Any],
        risk_proposal: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log trade rejection for analysis.
        
        Args:
            synergy_event: Original synergy event
            stage: Rejection stage ('gate1' or 'gate2')
            decision_data: Decision data at rejection
            risk_proposal: Risk proposal if available
        """
        rejection_log = {
            'timestamp': datetime.now(),
            'synergy_event': synergy_event,
            'rejection_stage': stage,
            'decision_data': decision_data,
            'risk_proposal': risk_proposal,
            'decision_count': self.decision_count
        }
        
        # Log internally (not emitted as event)
        logger.info(f"Trade rejected", **rejection_log)
    
    def _handle_error(self, synergy_event: Dict[str, Any], error: Exception) -> None:
        """
        Handle errors during processing.
        
        Args:
            synergy_event: Event being processed
            error: Exception that occurred
        """
        error_data = {
            'timestamp': datetime.now(),
            'synergy_event': synergy_event,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'decision_count': self.decision_count
        }
        
        logger.error(f"Processing error", **error_data)
        
        # Emit system error if critical
        if isinstance(error, (RuntimeError, ValueError)):
            event_bus = self.components.get('kernel').event_bus
            event_bus.emit('SYSTEM_ERROR', {
                'component': 'main_marl_core',
                'error': error_data,
                'critical': True
            })
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"EXEC_{timestamp}_{self.execution_count:04d}"
    
    def eval_mode(self) -> None:
        """Set all models to evaluation mode."""
        self.structure_embedder.eval()
        self.tactical_embedder.eval()
        self.regime_embedder.eval()
        self.lvn_embedder.eval()
        self.shared_policy.eval()
        self.decision_gate.eval()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'decision_count': self.decision_count,
            'execution_count': self.execution_count,
            'execution_rate': (
                self.execution_count / self.decision_count 
                if self.decision_count > 0 else 0.0
            ),
            'models_loaded': self.models_loaded,
            'device': str(self.device)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MainMARLCoreComponent("
            f"decisions={self.decision_count}, "
            f"executions={self.execution_count}, "
            f"models_loaded={self.models_loaded})"
        )