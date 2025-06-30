"""
Unit tests for Main MARL Core Component.

This test suite validates the functionality of the MainMARLCoreComponent,
ensuring correct implementation of the unified intelligence architecture
with two-gate decision flow.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.main_core.engine import MainMARLCoreComponent
from src.agents.main_core.models import (
    StructureEmbedder,
    TacticalEmbedder,
    RegimeEmbedder,
    LVNEmbedder,
    SharedPolicy,
    DecisionGate,
    MCDropoutEvaluator
)


class TestMainMARLCoreComponent:
    """Test suite for Main MARL Core Component."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            'device': 'cpu',
            'embedders': {
                'structure': {'output_dim': 64, 'dropout': 0.2},
                'tactical': {'hidden_dim': 64, 'output_dim': 48, 'dropout': 0.2},
                'regime': {'output_dim': 16, 'hidden_dim': 32},
                'lvn': {'input_dim': 5, 'output_dim': 8, 'hidden_dim': 16}
            },
            'shared_policy': {
                'hidden_dims': [256, 128, 64],
                'dropout': 0.2
            },
            'decision_gate': {
                'hidden_dim': 64,
                'dropout': 0.1
            },
            'mc_dropout': {
                'n_passes': 10,
                'confidence_threshold': 0.8
            }
        }
    
    @pytest.fixture
    def mock_components(self):
        """Create mock system components."""
        # Mock matrix assemblers
        matrix_30m = Mock()
        matrix_30m.get_matrix.return_value = np.random.randn(48, 8)
        
        matrix_5m = Mock()
        matrix_5m.get_matrix.return_value = np.random.randn(60, 7)
        
        # Mock RDE
        rde = Mock()
        rde.get_regime_vector.return_value = np.random.randn(8)
        
        # Mock M-RMS
        m_rms = Mock()
        m_rms.generate_risk_proposal.return_value = {
            'position_size': 2,
            'sl_atr_multiplier': 1.5,
            'risk_reward_ratio': 2.0,
            'confidence_score': 0.85,
            'risk_metrics': {'position_utilization': 0.4},
            'entry_price': 4500.0,
            'stop_loss_price': 4480.0,
            'take_profit_price': 4540.0,
            'risk_amount': 1000.0,
            'reward_amount': 2000.0
        }
        
        # Mock kernel with event bus
        kernel = Mock()
        event_bus = Mock()
        kernel.event_bus = event_bus
        
        return {
            'matrix_30m': matrix_30m,
            'matrix_5m': matrix_5m,
            'rde': rde,
            'm_rms': m_rms,
            'kernel': kernel
        }
    
    @pytest.fixture
    def synergy_event(self):
        """Create a sample synergy event."""
        return {
            'event_type': 'SYNERGY_DETECTED',
            'synergy_type': 'TYPE_1',
            'direction': 1,
            'symbol': 'ES',
            'timestamp': datetime.now(),
            'market_context': {
                'current_price': 4500.0,
                'atr': 10.0,
                'volatility': 15.0,
                'volume_ratio': 1.2,
                'price_momentum_5': 2.5,
                'rsi': 55.0,
                'spread': 0.25,
                'nearest_lvn': {
                    'price': 4495.0,
                    'strength': 75.0,
                    'distance': 5.0
                }
            },
            'signal_strengths': {
                'overall': 0.75
            },
            'metadata': {
                'bars_to_complete': 3
            }
        }
    
    def test_component_initialization_and_model_loading(self, mock_config, mock_components):
        """Test that the MainMARLCoreComponent initializes all its sub-models correctly."""
        component = MainMARLCoreComponent(mock_config, mock_components)
        
        # Assert all embedders are created with correct types
        assert isinstance(component.structure_embedder, StructureEmbedder)
        assert isinstance(component.tactical_embedder, TacticalEmbedder)
        assert isinstance(component.regime_embedder, RegimeEmbedder)
        assert isinstance(component.lvn_embedder, LVNEmbedder)
        
        # Assert SharedPolicy is created
        assert isinstance(component.shared_policy, SharedPolicy)
        
        # Assert DecisionGate is created
        assert isinstance(component.decision_gate, DecisionGate)
        
        # Assert MC Dropout evaluator is created
        assert isinstance(component.mc_evaluator, MCDropoutEvaluator)
        assert component.mc_evaluator.n_passes == 10
        assert component.confidence_threshold == 0.8
        
        # Verify dimensions match configuration
        assert component.structure_embedder.output_dim == 64
        assert component.tactical_embedder.output_dim == 48
        # Check the Linear layer before LayerNorm (index -3 instead of -2)
        assert component.regime_embedder.mlp[-3].out_features == 16
        assert component.lvn_embedder.mlp[-3].out_features == 8
        
        # Verify unified dimension calculation
        expected_unified_dim = 64 + 48 + 16 + 8  # 136
        assert component.shared_policy.input_dim == expected_unified_dim
        assert component.decision_gate.decision_network[0].in_features == expected_unified_dim + 8
    
    def test_unified_state_vector_creation(self, mock_config, mock_components, synergy_event):
        """Test the correct assembly of the main state vector."""
        component = MainMARLCoreComponent(mock_config, mock_components)
        
        # Mock embedder outputs with known sizes
        with patch.object(component.structure_embedder, 'forward') as mock_structure:
            with patch.object(component.tactical_embedder, 'forward') as mock_tactical:
                with patch.object(component.regime_embedder, 'forward') as mock_regime:
                    with patch.object(component.lvn_embedder, 'forward') as mock_lvn:
                        # Set return values with correct dimensions
                        mock_structure.return_value = torch.randn(1, 64)
                        mock_tactical.return_value = torch.randn(1, 48)
                        mock_regime.return_value = torch.randn(1, 16)
                        mock_lvn.return_value = torch.randn(1, 8)
                        
                        # Call the method
                        unified_state = component._prepare_unified_state(synergy_event)
                        
                        # Assert unified state has correct shape
                        assert unified_state.shape == (1, 136)  # 64 + 48 + 16 + 8 = 136
                        
                        # Verify all embedders were called
                        assert mock_structure.called
                        assert mock_tactical.called
                        assert mock_regime.called
                        assert mock_lvn.called
    
    def test_mc_dropout_consensus_logic(self, mock_config, mock_components):
        """Test MC Dropout mechanism correctly evaluates policy confidence."""
        component = MainMARLCoreComponent(mock_config, mock_components)
        
        # Test case 1: High consensus (90%) - should qualify
        with patch.object(component.shared_policy, 'forward') as mock_forward:
            # Make policy return consistent high confidence for "Initiate" action
            mock_forward.return_value = {
                'action_logits': torch.tensor([[2.0, -2.0]]),  # Strong preference for action 0
                'action_probs': torch.tensor([[0.9, 0.1]])
            }
            
            unified_state = torch.randn(1, 136)
            result = component._run_mc_dropout_consensus(unified_state)
            
            # Assert high confidence leads to qualification
            assert result['confidence'].item() >= 0.8
            assert result['should_proceed'].item() == True
            assert result['predicted_action'].item() == 0  # Initiate action
        
        # Test case 2: Low consensus (70%) - should not qualify
        with patch.object(component.shared_policy, 'forward') as mock_forward:
            # Make policy return lower confidence
            mock_forward.return_value = {
                'action_logits': torch.tensor([[0.8, -0.8]]),  # Weaker preference
                'action_probs': torch.tensor([[0.7, 0.3]])
            }
            
            unified_state = torch.randn(1, 136)
            result = component._run_mc_dropout_consensus(unified_state)
            
            # Assert low confidence leads to rejection
            assert result['confidence'].item() < 0.8
            assert result['should_proceed'].item() == False
    
    def test_two_gate_flow_with_successful_qualification(
        self, mock_config, mock_components, synergy_event
    ):
        """Test the happy path of the entire decision flow."""
        component = MainMARLCoreComponent(mock_config, mock_components)
        
        # Mock MC Dropout to return should_qualify = True
        with patch.object(component, '_run_mc_dropout_consensus') as mock_mc_dropout:
            mock_mc_dropout.return_value = {
                'predicted_action': torch.tensor([0]),
                'mean_probs': torch.tensor([[0.85, 0.15]]),
                'std_probs': torch.tensor([[0.05, 0.05]]),
                'confidence': torch.tensor([0.85]),
                'entropy': torch.tensor([0.3]),
                'should_proceed': torch.tensor([True]),
                'uncertainty_metrics': {
                    'mean_std': 0.05,
                    'max_std': 0.05,
                    'entropy': 0.3
                }
            }
            
            # Mock DecisionGate to return EXECUTE
            with patch.object(component.decision_gate, 'forward') as mock_gate:
                mock_gate.return_value = {
                    'decision_logits': torch.tensor([[1.0, -1.0]]),
                    'decision_probs': torch.tensor([[0.7, 0.3]]),
                    'execute_probability': torch.tensor([0.7])
                }
                
                # Mock unified state preparation
                with patch.object(component, '_prepare_unified_state') as mock_prepare:
                    mock_prepare.return_value = torch.randn(1, 136)
                    
                    # Call the main method
                    component.initiate_qualification(synergy_event)
                    
                    # Assert M-RMS was called
                    assert mock_components['m_rms'].generate_risk_proposal.called
                    
                    # Assert DecisionGate was called
                    assert mock_gate.called
                    
                    # Assert EXECUTE_TRADE event was published
                    event_bus = mock_components['kernel'].event_bus
                    assert event_bus.emit.called
                    
                    # Verify the event type
                    call_args = event_bus.emit.call_args
                    assert call_args[0][0] == 'EXECUTE_TRADE'
                    
                    # Verify execution count increased
                    assert component.execution_count == 1
    
    def test_two_gate_flow_with_failed_qualification(
        self, mock_config, mock_components, synergy_event
    ):
        """Test that the flow stops correctly at the first gate."""
        component = MainMARLCoreComponent(mock_config, mock_components)
        
        # Mock MC Dropout to return should_qualify = False
        with patch.object(component, '_run_mc_dropout_consensus') as mock_mc_dropout:
            mock_mc_dropout.return_value = {
                'predicted_action': torch.tensor([1]),  # Do_Nothing action
                'mean_probs': torch.tensor([[0.3, 0.7]]),
                'std_probs': torch.tensor([[0.1, 0.1]]),
                'confidence': torch.tensor([0.7]),
                'entropy': torch.tensor([0.6]),
                'should_proceed': torch.tensor([False]),
                'uncertainty_metrics': {
                    'mean_std': 0.1,
                    'max_std': 0.1,
                    'entropy': 0.6
                }
            }
            
            # Mock unified state preparation
            with patch.object(component, '_prepare_unified_state') as mock_prepare:
                mock_prepare.return_value = torch.randn(1, 136)
                
                # Mock DecisionGate (should not be called)
                with patch.object(component.decision_gate, 'forward') as mock_gate:
                    
                    # Call the main method
                    component.initiate_qualification(synergy_event)
                    
                    # Assert M-RMS was NOT called
                    assert not mock_components['m_rms'].generate_risk_proposal.called
                    
                    # Assert DecisionGate was NOT called
                    assert not mock_gate.called
                    
                    # Assert NO EXECUTE_TRADE event was published
                    event_bus = mock_components['kernel'].event_bus
                    emit_calls = [call for call in event_bus.emit.call_args_list 
                                  if call[0][0] == 'EXECUTE_TRADE']
                    assert len(emit_calls) == 0
                    
                    # Verify execution count did not increase
                    assert component.execution_count == 0


def test_embedder_output_dimensions():
    """Test that all embedders produce outputs with expected dimensions."""
    # Test StructureEmbedder
    structure_embedder = StructureEmbedder(input_channels=8, output_dim=64)
    input_structure = torch.randn(2, 48, 8)  # batch_size=2
    output_structure = structure_embedder(input_structure)
    assert output_structure.shape == (2, 64)
    
    # Test TacticalEmbedder
    tactical_embedder = TacticalEmbedder(input_dim=7, output_dim=48)
    input_tactical = torch.randn(2, 60, 7)
    output_tactical = tactical_embedder(input_tactical)
    assert output_tactical.shape == (2, 48)
    
    # Test RegimeEmbedder
    regime_embedder = RegimeEmbedder(input_dim=8, output_dim=16)
    input_regime = torch.randn(2, 8)
    output_regime = regime_embedder(input_regime)
    assert output_regime.shape == (2, 16)
    
    # Test LVNEmbedder
    lvn_embedder = LVNEmbedder(input_dim=5, output_dim=8)
    input_lvn = torch.randn(2, 5)
    output_lvn = lvn_embedder(input_lvn)
    assert output_lvn.shape == (2, 8)


def test_shared_policy_mc_dropout_modes():
    """Test SharedPolicy MC Dropout enable/disable functionality."""
    policy = SharedPolicy(input_dim=136, dropout_rate=0.2)
    
    # Test enable MC Dropout
    policy.enable_mc_dropout()
    assert policy.training == True
    
    # Test disable MC Dropout
    policy.disable_mc_dropout()
    assert policy.training == False


def test_decision_gate_output_format():
    """Test DecisionGate output format and probabilities."""
    gate = DecisionGate(input_dim=144)
    input_state = torch.randn(3, 144)  # batch_size=3
    
    output = gate(input_state)
    
    # Check output structure
    assert 'decision_logits' in output
    assert 'decision_probs' in output
    assert 'execute_probability' in output
    
    # Check shapes
    assert output['decision_logits'].shape == (3, 2)
    assert output['decision_probs'].shape == (3, 2)
    assert output['execute_probability'].shape == (3,)
    
    # Check probabilities sum to 1
    assert torch.allclose(output['decision_probs'].sum(dim=1), torch.ones(3))
    
    # Check execute_probability matches first column of decision_probs
    assert torch.allclose(output['execute_probability'], output['decision_probs'][:, 0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])