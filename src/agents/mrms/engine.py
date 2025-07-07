"""
Multi-Agent Risk Management Subsystem (M-RMS) Engine Component.

This module provides the high-level interface for the M-RMS, handling
model loading, inference, and risk proposal generation.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch

from .models import RiskManagementEnsemble
from .communication import MRMSCommunicationLSTM, RiskMemory

logger = logging.getLogger(__name__)


class MRMSComponent:
    """
    High-level component interface for the Multi-Agent Risk Management Subsystem.
    
    This class provides a simple interface for the rest of the AlgoSpace system
    to interact with the complex M-RMS neural network ensemble without needing
    to understand its internal workings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the M-RMS component.
        
        Args:
            config: M-RMS specific configuration dictionary containing:
                - synergy_dim: Dimension of synergy feature vector (default: 30)
                - account_dim: Dimension of account state vector (default: 10)
                - device: Computing device ('cpu' or 'cuda')
                - point_value: Dollar value per point for the instrument
                - max_position_size: Maximum allowed position size
                - Other model architecture parameters
        """
        self.config = config
        
        # Extract configuration
        self.synergy_dim = config.get('synergy_dim', 30)
        self.account_dim = config.get('account_dim', 10)
        self.device = torch.device(config.get('device', 'cpu'))
        self.point_value = config.get('point_value', 5.0)  # MES default
        self.max_position_size = config.get('max_position_size', 5)
        
        # Initialize the ensemble model
        self.model = RiskManagementEnsemble(
            synergy_dim=self.synergy_dim,
            account_dim=self.account_dim,
            hidden_dim=config.get('hidden_dim', 128),
            position_agent_hidden=config.get('position_agent_hidden', 128),
            sl_agent_hidden=config.get('sl_agent_hidden', 64),
            pt_agent_hidden=config.get('pt_agent_hidden', 64),
            dropout_rate=config.get('dropout_rate', 0.2)
        ).to(self.device)
        
        # Set to evaluation mode by default
        self.model.eval()
        self.model_loaded = False
        
        # Initialize communication layer if configured
        if 'communication' in config:
            self.communication_lstm = MRMSCommunicationLSTM(
                config['communication']
            ).to(self.device)
            self.communication_lstm.eval()
            logger.info("MRMS Communication LSTM initialized")
        else:
            self.communication_lstm = None
            
        # Track recent outcomes
        self.recent_outcomes = []
        self.max_outcome_history = config.get('max_outcome_history', 20)
        
        logger.info(f"M-RMS Component initialized on device: {self.device}")
        
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained weights from a saved model file.
        
        Args:
            model_path: Path to the .pth file containing model weights
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            RuntimeError: If loading fails due to architecture mismatch
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load checkpoint with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats (allow missing keys for development)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                # Assume the checkpoint is the state dict itself
                self.model.load_state_dict(checkpoint, strict=False)
            
            # Ensure model is in evaluation mode
            self.model.eval()
            self.model_loaded = True
            
            # Log additional info if available
            if isinstance(checkpoint, dict):
                training_iterations = checkpoint.get('training_iterations', 'unknown')
                final_reward = checkpoint.get('final_reward_mean', 'unknown')
                logger.info(f"Loaded M-RMS model from: {model_path}")
                logger.info(f"Training iterations: {training_iterations}")
                logger.info(f"Final reward mean: {final_reward}")
            else:
                logger.info(f"Loaded M-RMS model weights from: {model_path}")
                
            # Load communication weights if available
            if self.communication_lstm is not None:
                comm_path = str(model_path).replace('.pth', '_comm.pth')
                if Path(comm_path).exists():
                    try:
                        comm_state = torch.load(comm_path, map_location=self.device)
                        self.communication_lstm.load_state_dict(comm_state)
                        logger.info(f"MRMS Communication weights loaded from {comm_path}")
                    except Exception as e:
                        logger.warning(f"Could not load communication weights: {e}")
                
        except Exception as e:
            logger.error(f"Failed to load M-RMS model: {e}")
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    def generate_risk_proposal(self, trade_qualification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive risk proposal for a qualified trade opportunity.
        
        This is the primary public method for inference. It processes the trade
        qualification data and returns a detailed risk management proposal.
        
        Args:
            trade_qualification: Dictionary containing:
                - synergy_vector: Numpy array of synergy features [30]
                - account_state_vector: Numpy array of account state [10]
                - entry_price: Proposed entry price
                - direction: Trade direction ('LONG' or 'SHORT')
                - atr: Current Average True Range
                - symbol: Trading symbol
                - timestamp: Trade timestamp
                
        Returns:
            RiskProposal dictionary containing:
                - position_size: Number of contracts (0-5)
                - stop_loss_price: Calculated stop loss price
                - take_profit_price: Calculated take profit price
                - risk_amount: Dollar risk for the trade
                - reward_amount: Potential dollar reward
                - risk_reward_ratio: R:R ratio
                - sl_atr_multiplier: Stop loss distance in ATR units
                - confidence_score: Model confidence (0-1)
                - risk_metrics: Additional risk analytics
                
        Raises:
            RuntimeError: If model weights haven't been loaded
            ValueError: If input validation fails
        """
        if not self.model_loaded:
            raise RuntimeError("Model weights not loaded. Call load_model() first.")
        
        # Validate inputs
        self._validate_trade_qualification(trade_qualification)
        
        # Extract inputs
        synergy_vector = trade_qualification['synergy_vector']
        account_vector = trade_qualification['account_state_vector']
        entry_price = trade_qualification['entry_price']
        direction = trade_qualification['direction']
        atr = trade_qualification['atr']
        
        # Convert to tensors
        synergy_tensor = torch.tensor(synergy_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        account_tensor = torch.tensor(account_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            actions = self.model.get_action_dict(synergy_tensor, account_tensor)
            
            # Get raw outputs for confidence calculation
            outputs = self.model(synergy_tensor, account_tensor)
            
        # Extract action values
        position_size = int(actions['position_size'].cpu().item())
        sl_multiplier = float(actions['sl_atr_multiplier'].cpu().item())
        rr_ratio = float(actions['rr_ratio'].cpu().item())
        
        # Get value estimate for confidence
        value_estimate = outputs.get('value', torch.tensor(0.5))
        
        # Calculate stop loss and take profit prices
        sl_distance = sl_multiplier * atr
        tp_distance = sl_distance * rr_ratio
        
        if direction == 'LONG':
            stop_loss_price = entry_price - sl_distance
            take_profit_price = entry_price + tp_distance
        else:  # SHORT
            stop_loss_price = entry_price + sl_distance
            take_profit_price = entry_price - tp_distance
        
        # Calculate risk and reward amounts
        risk_per_contract = sl_distance * self.point_value
        reward_per_contract = tp_distance * self.point_value
        
        risk_amount = risk_per_contract * position_size if position_size > 0 else 0
        reward_amount = reward_per_contract * position_size if position_size > 0 else 0
        
        # Calculate confidence score from position logits
        position_logits = outputs['position_logits']
        position_probs = torch.softmax(position_logits, dim=-1)
        confidence_score = float(position_probs[0, position_size].cpu().item())
        
        # Process through communication LSTM if available
        mu_risk = None
        sigma_risk = None
        adapted_position_size = position_size
        
        if self.communication_lstm is not None:
            # Create risk vector
            risk_vector = torch.tensor([
                position_size / self.max_position_size,
                sl_multiplier / 3.0,  # Normalize
                rr_ratio / 5.0,  # Normalize  
                value_estimate.item() if hasattr(value_estimate, 'item') else 0.5
            ], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get recent outcome vector
            recent_outcome = self._get_recent_outcome_vector()
            
            # Process through communication LSTM
            mu_risk, sigma_risk = self.communication_lstm(
                risk_vector,
                recent_outcome,
                update_memory=False  # Only update after trade completes
            )
            
            # Adapt position size based on uncertainty
            adapted_position_size = self._adapt_position_size(
                position_size, sigma_risk[0].mean().item()
            )
        
        # Build comprehensive risk proposal with strict type enforcement
        risk_proposal = {
            # Core trade parameters (guaranteed types)
            'position_size': int(position_size),
            'stop_loss_price': float(round(stop_loss_price, 2)),
            'take_profit_price': float(round(take_profit_price, 2)),
            'risk_amount': float(round(risk_amount, 2)),
            'reward_amount': float(round(reward_amount, 2)),
            'risk_reward_ratio': float(round(rr_ratio, 2)),
            'sl_atr_multiplier': float(round(sl_multiplier, 3)),
            'confidence_score': float(round(confidence_score, 3)),
            
            # Trade direction and entry
            'trade_direction': str(direction),
            'entry_price': float(trade_qualification['entry_price']),
            'atr_value': float(trade_qualification['atr']),
            
            # Risk metrics
            'risk_metrics': {
                'sl_distance_points': float(round(sl_distance, 2)),
                'tp_distance_points': float(round(tp_distance, 2)),
                'risk_per_contract': float(round(risk_per_contract, 2)),
                'reward_per_contract': float(round(reward_per_contract, 2)),
                'max_position_allowed': int(self.max_position_size),
                'position_utilization': float(position_size / self.max_position_size if self.max_position_size > 0 else 0.0),
                'risk_percentage': float(round((risk_amount / 100000) * 100, 2)) if risk_amount > 0 else 0.0  # Assume 100k account
            },
            
            # Model diagnostics
            'model_outputs': {
                'raw_sl_multiplier': float(sl_multiplier),
                'raw_rr_ratio': float(rr_ratio),
                'position_probabilities': position_probs[0].cpu().numpy().tolist()
            },
            
            # Metadata
            'metadata': {
                'timestamp': trade_qualification.get('timestamp', ''),
                'symbol': trade_qualification.get('symbol', ''),
                'model_version': '1.0',
                'point_value': float(self.point_value),
                'currency': 'USD'
            },
            
            # Validation flags
            'validation': {
                'is_valid': bool(position_size >= 0 and confidence_score > 0),
                'risk_acceptable': bool(risk_amount <= 5000),  # Max $5k risk
                'position_within_limits': bool(position_size <= self.max_position_size),
                'rr_ratio_acceptable': bool(rr_ratio >= 1.0)
            }
        }
        
        # Add temporal context if available
        if mu_risk is not None and sigma_risk is not None:
            risk_proposal['risk_embedding'] = mu_risk.cpu().numpy()
            risk_proposal['risk_uncertainty'] = sigma_risk.cpu().numpy()
            risk_proposal['adapted_position_size'] = int(adapted_position_size)
            risk_proposal['risk_metrics']['uncertainty_mean'] = float(sigma_risk.mean().item())
        
        # Validate output format
        self._validate_risk_proposal(risk_proposal)
        
        # Log the proposal
        logger.info(f"Generated risk proposal: size={position_size}, "
                   f"SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, "
                   f"R:R={rr_ratio:.2f}, confidence={confidence_score:.3f}")
        
        return risk_proposal
    
    def _validate_trade_qualification(self, trade_qual: Dict[str, Any]) -> None:
        """
        Validate the trade qualification inputs.
        
        Args:
            trade_qual: Trade qualification dictionary to validate
            
        Raises:
            ValueError: If validation fails
        """
        required_fields = ['synergy_vector', 'account_state_vector', 
                          'entry_price', 'direction', 'atr']
        
        for field in required_fields:
            if field not in trade_qual:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate synergy vector
        synergy_vector = trade_qual['synergy_vector']
        if not isinstance(synergy_vector, np.ndarray):
            raise ValueError("synergy_vector must be a numpy array")
        if synergy_vector.shape != (self.synergy_dim,):
            raise ValueError(f"synergy_vector must have shape ({self.synergy_dim},), "
                           f"got {synergy_vector.shape}")
        
        # Validate account vector
        account_vector = trade_qual['account_state_vector']
        if not isinstance(account_vector, np.ndarray):
            raise ValueError("account_state_vector must be a numpy array")
        if account_vector.shape != (self.account_dim,):
            raise ValueError(f"account_state_vector must have shape ({self.account_dim},), "
                           f"got {account_vector.shape}")
        
        # Validate direction
        if trade_qual['direction'] not in ['LONG', 'SHORT']:
            raise ValueError("direction must be either 'LONG' or 'SHORT'")
        
        # Validate numeric fields
        if trade_qual['entry_price'] <= 0:
            raise ValueError("entry_price must be positive")
        if trade_qual['atr'] <= 0:
            raise ValueError("atr must be positive")
    
    def _validate_risk_proposal(self, proposal: Dict[str, Any]) -> None:
        """
        Validate risk proposal output format and types.
        
        Args:
            proposal: Risk proposal dictionary to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Required top-level fields
        required_fields = [
            'position_size', 'stop_loss_price', 'take_profit_price',
            'risk_amount', 'reward_amount', 'risk_reward_ratio',
            'sl_atr_multiplier', 'confidence_score', 'trade_direction',
            'entry_price', 'atr_value', 'risk_metrics', 'model_outputs',
            'metadata', 'validation'
        ]
        
        for field in required_fields:
            if field not in proposal:
                raise ValueError(f"Missing required field in risk proposal: {field}")
        
        # Type validation
        type_checks = [
            ('position_size', int),
            ('stop_loss_price', (int, float)),
            ('take_profit_price', (int, float)),
            ('risk_amount', (int, float)),
            ('reward_amount', (int, float)),
            ('risk_reward_ratio', (int, float)),
            ('confidence_score', (int, float)),
            ('trade_direction', str),
        ]
        
        for field, expected_type in type_checks:
            if not isinstance(proposal[field], expected_type):
                raise ValueError(f"Field '{field}' must be {expected_type}, got {type(proposal[field])}")
        
        # Range validation
        if not (0 <= proposal['position_size'] <= self.max_position_size):
            raise ValueError(f"Position size {proposal['position_size']} outside valid range [0, {self.max_position_size}]")
        
        if not (0.0 <= proposal['confidence_score'] <= 1.0):
            raise ValueError(f"Confidence score {proposal['confidence_score']} outside valid range [0.0, 1.0]")
        
        # Validate nested dictionaries
        if not isinstance(proposal['risk_metrics'], dict):
            raise ValueError("risk_metrics must be a dictionary")
        
        if not isinstance(proposal['validation'], dict):
            raise ValueError("validation must be a dictionary")
        
        # Ensure all validation flags are boolean
        for key, value in proposal['validation'].items():
            if not isinstance(value, bool):
                raise ValueError(f"Validation flag '{key}' must be boolean, got {type(value)}")
        
        logger.debug("Risk proposal validation passed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information and status
        """
        info = self.model.get_model_info()
        info.update({
            'model_loaded': self.model_loaded,
            'device': str(self.device),
            'point_value': self.point_value,
            'max_position_size': self.max_position_size
        })
        return info
    
    def _get_recent_outcome_vector(self) -> torch.Tensor:
        """Get recent outcome vector for communication LSTM."""
        if not self.recent_outcomes:
            # No history yet, return neutral vector
            return torch.zeros(1, 3, dtype=torch.float32).to(self.device)
            
        # Calculate recent performance metrics
        recent_stops = sum(1 for o in self.recent_outcomes[-5:] if o.get('hit_stop', False))
        recent_targets = sum(1 for o in self.recent_outcomes[-5:] if o.get('hit_target', False))
        recent_pnl = sum(o.get('pnl', 0) for o in self.recent_outcomes[-5:]) / 100.0  # Normalize
        
        return torch.tensor([
            recent_stops / 5.0,
            recent_targets / 5.0,
            np.clip(recent_pnl, -1, 1)
        ], dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _adapt_position_size(self, base_size: int, uncertainty: float) -> int:
        """Adapt position size based on uncertainty level."""
        if self.communication_lstm is None:
            return base_size
            
        # Use communication LSTM's adaptation logic
        adapted_size = self.communication_lstm._adapt_risk_parameters(
            float(base_size), uncertainty
        )
        
        # Ensure integer and within limits
        return int(min(max(0, round(adapted_size)), self.max_position_size))
    
    def update_trade_outcome(self, trade_outcome: Dict[str, Any]) -> None:
        """Update communication layer with trade outcome.
        
        Args:
            trade_outcome: Dictionary containing:
                - hit_stop: bool
                - hit_target: bool  
                - pnl: float
                - position_size: int
                - sl_distance: float
                - tp_distance: float
        """
        # Add to recent outcomes
        self.recent_outcomes.append(trade_outcome)
        
        # Trim to max history
        if len(self.recent_outcomes) > self.max_outcome_history:
            self.recent_outcomes = self.recent_outcomes[-self.max_outcome_history:]
            
        # Update communication LSTM if available
        if self.communication_lstm is not None:
            # Create risk vector from trade
            risk_vector = torch.tensor([[
                trade_outcome.get('position_size', 0) / self.max_position_size,
                trade_outcome.get('sl_distance', 0) / 50.0,
                trade_outcome.get('tp_distance', 0) / 100.0,
                0.5  # Default confidence
            ]], dtype=torch.float32).to(self.device)
            
            # Create outcome vector
            outcome_vector = torch.tensor([[
                float(trade_outcome.get('hit_stop', False)),
                float(trade_outcome.get('hit_target', False)),
                np.clip(trade_outcome.get('pnl', 0) / 100.0, -1, 1)
            ]], dtype=torch.float32).to(self.device)
            
            # Update memory
            self.communication_lstm._update_memory(risk_vector, outcome_vector)
            
            logger.info(f"Updated MRMS communication with trade outcome: PnL={trade_outcome.get('pnl', 0)}")
    
    def __repr__(self) -> str:
        """String representation of the M-RMS component."""
        return (f"MRMSComponent(synergy_dim={self.synergy_dim}, "
                f"account_dim={self.account_dim}, "
                f"model_loaded={self.model_loaded}, "
                f"device={self.device})")


# Alias for backward compatibility
MRMSEngine = MRMSComponent