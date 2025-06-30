"""
Multi-Agent Risk Management Subsystem (M-RMS) Neural Network Models.

This module contains the core neural network architectures for the M-RMS,
including three specialized sub-agents and their ensemble coordinator.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionSizingAgent(nn.Module):
    """
    Sub-agent responsible for intelligent position sizing decisions.
    
    This agent analyzes market conditions and account state to determine
    the optimal number of contracts to trade (0-5).
    
    Args:
        input_dim: Dimension of the combined state vector (default: 40)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout_rate: Dropout probability for regularization (default: 0.2)
    """
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Deep neural network for position sizing
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 6)  # 6 position size options (0-5 contracts)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning position size logits.
        
        Args:
            state: Combined state tensor [batch_size, input_dim]
            
        Returns:
            Logits for position size selection [batch_size, 6]
        """
        return self.network(state)


class StopLossAgent(nn.Module):
    """
    Sub-agent responsible for dynamic stop loss placement.
    
    This agent determines the optimal stop loss distance as a multiple
    of the Average True Range (ATR), adapting to market volatility.
    
    Args:
        input_dim: Dimension of the combined state vector (default: 40)
        hidden_dim: Hidden layer dimension (default: 64)
        dropout_rate: Dropout probability for regularization (default: 0.2)
        min_multiplier: Minimum ATR multiplier (default: 0.5)
        max_multiplier: Maximum ATR multiplier (default: 3.0)
    """
    
    def __init__(
        self, 
        input_dim: int = 40, 
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
        min_multiplier: float = 0.5,
        max_multiplier: float = 3.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        
        # Neural network for stop loss determination
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # ATR multiplier
            nn.Sigmoid()  # Ensure output in [0, 1]
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning stop loss ATR multiplier.
        
        Args:
            state: Combined state tensor [batch_size, input_dim]
            
        Returns:
            Stop loss ATR multiplier scaled to [min_multiplier, max_multiplier]
        """
        raw_output = self.network(state)
        # Scale sigmoid output to desired range
        scaled_output = self.min_multiplier + (self.max_multiplier - self.min_multiplier) * raw_output
        return scaled_output


class ProfitTargetAgent(nn.Module):
    """
    Sub-agent responsible for profit target placement.
    
    This agent determines the optimal risk-reward ratio for trades,
    balancing profit potential with win rate optimization.
    
    Args:
        input_dim: Dimension of the combined state vector (default: 40)
        hidden_dim: Hidden layer dimension (default: 64)
        dropout_rate: Dropout probability for regularization (default: 0.2)
        min_rr: Minimum risk-reward ratio (default: 1.0)
        max_rr: Maximum risk-reward ratio (default: 5.0)
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
        min_rr: float = 1.0,
        max_rr: float = 5.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_rr = min_rr
        self.max_rr = max_rr
        
        # Neural network for profit target determination
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Risk-reward ratio
            nn.Sigmoid()  # Ensure output in [0, 1]
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning risk-reward ratio.
        
        Args:
            state: Combined state tensor [batch_size, input_dim]
            
        Returns:
            Risk-reward ratio scaled to [min_rr, max_rr]
        """
        raw_output = self.network(state)
        # Scale sigmoid output to desired range
        scaled_output = self.min_rr + (self.max_rr - self.min_rr) * raw_output
        return scaled_output


class RiskManagementEnsemble(nn.Module):
    """
    Ensemble coordinator for the three risk management sub-agents.
    
    This class orchestrates the three specialized sub-agents (position sizing,
    stop loss, and profit target) to produce comprehensive risk management
    decisions for each trade opportunity.
    
    Args:
        synergy_dim: Dimension of synergy feature vector (default: 30)
        account_dim: Dimension of account state vector (default: 10)
        hidden_dim: Hidden dimension for value head (default: 128)
        position_agent_hidden: Hidden dimension for position sizing agent
        sl_agent_hidden: Hidden dimension for stop loss agent
        pt_agent_hidden: Hidden dimension for profit target agent
        dropout_rate: Dropout rate for all sub-agents
    """
    
    def __init__(
        self,
        synergy_dim: int = 30,
        account_dim: int = 10,
        hidden_dim: int = 128,
        position_agent_hidden: int = 128,
        sl_agent_hidden: int = 64,
        pt_agent_hidden: int = 64,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        # Calculate total input dimension
        self.synergy_dim = synergy_dim
        self.account_dim = account_dim
        self.input_dim = synergy_dim + account_dim
        
        # Initialize sub-agents
        self.position_agent = PositionSizingAgent(
            input_dim=self.input_dim,
            hidden_dim=position_agent_hidden,
            dropout_rate=dropout_rate
        )
        
        self.stop_loss_agent = StopLossAgent(
            input_dim=self.input_dim,
            hidden_dim=sl_agent_hidden,
            dropout_rate=dropout_rate
        )
        
        self.profit_target_agent = ProfitTargetAgent(
            input_dim=self.input_dim,
            hidden_dim=pt_agent_hidden,
            dropout_rate=dropout_rate
        )
        
        # Value function head for RL training
        self.value_head = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self, 
        synergy_vector: torch.Tensor, 
        account_vector: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ensemble.
        
        Args:
            synergy_vector: Synergy features [batch_size, synergy_dim]
            account_vector: Account state features [batch_size, account_dim]
            
        Returns:
            Dictionary containing:
                - position_logits: Logits for position sizing [batch_size, 6]
                - sl_multiplier: Stop loss ATR multiplier [batch_size, 1]
                - rr_ratio: Risk-reward ratio [batch_size, 1]
                - value: Value function output [batch_size]
        """
        # Concatenate state vectors
        combined_state = torch.cat([synergy_vector, account_vector], dim=-1)
        
        # Get outputs from each sub-agent
        position_logits = self.position_agent(combined_state)
        sl_multiplier = self.stop_loss_agent(combined_state)
        rr_ratio = self.profit_target_agent(combined_state)
        
        # Compute value
        value = self.value_head(combined_state).squeeze(-1)
        
        return {
            'position_logits': position_logits,
            'sl_multiplier': sl_multiplier,
            'rr_ratio': rr_ratio,
            'value': value
        }
    
    def get_action_dict(
        self, 
        synergy_vector: torch.Tensor, 
        account_vector: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get discrete actions for inference (no gradients).
        
        Args:
            synergy_vector: Synergy features [batch_size, synergy_dim]
            account_vector: Account state features [batch_size, account_dim]
            
        Returns:
            Dictionary containing:
                - position_size: Selected position size [batch_size]
                - sl_atr_multiplier: Stop loss ATR multiplier [batch_size, 1]
                - rr_ratio: Risk-reward ratio [batch_size, 1]
        """
        with torch.no_grad():
            outputs = self.forward(synergy_vector, account_vector)
            
            # Convert position logits to discrete action
            position_size = torch.argmax(outputs['position_logits'], dim=-1)
            
            return {
                'position_size': position_size,
                'sl_atr_multiplier': outputs['sl_multiplier'],
                'rr_ratio': outputs['rr_ratio']
            }
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing model architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Multi-Agent Risk Management Ensemble',
            'sub_agents': ['PositionSizingAgent', 'StopLossAgent', 'ProfitTargetAgent'],
            'input_dim': self.input_dim,
            'synergy_dim': self.synergy_dim,
            'account_dim': self.account_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'position_size_options': 6,
            'sl_multiplier_range': [self.stop_loss_agent.min_multiplier, 
                                   self.stop_loss_agent.max_multiplier],
            'rr_ratio_range': [self.profit_target_agent.min_rr, 
                              self.profit_target_agent.max_rr]
        }