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
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the checkpoint is the state dict itself
                self.model.load_state_dict(checkpoint)
            
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
        
        # Build comprehensive risk proposal
        risk_proposal = {
            'position_size': position_size,
            'stop_loss_price': round(stop_loss_price, 2),
            'take_profit_price': round(take_profit_price, 2),
            'risk_amount': round(risk_amount, 2),
            'reward_amount': round(reward_amount, 2),
            'risk_reward_ratio': round(rr_ratio, 2),
            'sl_atr_multiplier': round(sl_multiplier, 3),
            'confidence_score': round(confidence_score, 3),
            'risk_metrics': {
                'sl_distance_points': round(sl_distance, 2),
                'tp_distance_points': round(tp_distance, 2),
                'risk_per_contract': round(risk_per_contract, 2),
                'max_position_allowed': self.max_position_size,
                'position_utilization': position_size / self.max_position_size if self.max_position_size > 0 else 0
            },
            'model_outputs': {
                'raw_sl_multiplier': sl_multiplier,
                'raw_rr_ratio': rr_ratio,
                'position_probabilities': position_probs[0].cpu().numpy().tolist()
            }
        }
        
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
    
    def __repr__(self) -> str:
        """String representation of the M-RMS component."""
        return (f"MRMSComponent(synergy_dim={self.synergy_dim}, "
                f"account_dim={self.account_dim}, "
                f"model_loaded={self.model_loaded}, "
                f"device={self.device})")