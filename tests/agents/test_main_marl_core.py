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
    MCDropoutEvaluator,
    SharedPolicyNetwork,
    MCDropoutConsensus,
    PolicyOutput
)
from src.agents.main_core.decision_gate import (
    DecisionGate as AdvancedDecisionGate,
    RiskProposalEncoder,
    AdaptiveDecisionGate
)
from src.training.marl_training import MAPPOTrainer, ExperienceBuffer, calculate_trading_reward


class TestMainMARLCoreComponent:
    """Test suite for Main MARL Core Component."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            'device': 'cpu',
            'embedders': {
                'structure': {
                    'output_dim': 64, 
                    'dropout': 0.2,
                    'd_model': 128,
                    'n_heads': 4,
                    'n_layers': 3,
                    'd_ff': 512,
                    'max_seq_len': 48
                },
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
                        # Structure embedder now returns (mu, sigma)
                        mock_structure.return_value = (torch.randn(1, 64), torch.randn(1, 64))
                        # Tactical embedder now also returns (mu, sigma)
                        mock_tactical.return_value = (torch.randn(1, 48), torch.randn(1, 48))
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
    # Test StructureEmbedder with dual outputs
    structure_embedder = StructureEmbedder(
        input_channels=8, 
        output_dim=64,
        d_model=128,
        n_heads=4,
        n_layers=3
    )
    input_structure = torch.randn(2, 48, 8)  # batch_size=2
    mu_structure, sigma_structure = structure_embedder(input_structure)
    
    # Check shapes
    assert mu_structure.shape == (2, 64), f"Expected (2, 64), got {mu_structure.shape}"
    assert sigma_structure.shape == (2, 64), f"Expected (2, 64), got {sigma_structure.shape}"
    
    # Check uncertainty is positive
    assert torch.all(sigma_structure > 0), "Uncertainty must be positive"
    
    # Test attention weights return
    mu, sigma, attention_weights = structure_embedder(
        input_structure, 
        return_attention_weights=True
    )
    assert attention_weights.shape == (2, 48), "Attention weights shape mismatch"
    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(2)), "Attention weights must sum to 1"
    
    # Test gradient flow
    loss = mu_structure.mean() + sigma_structure.mean()
    loss.backward()
    
    for name, param in structure_embedder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    # Test TacticalEmbedder (now returns dual outputs)
    tactical_embedder = TacticalEmbedder(input_dim=7, output_dim=48)
    input_tactical = torch.randn(2, 60, 7)
    mu_tactical, sigma_tactical = tactical_embedder(input_tactical)
    assert mu_tactical.shape == (2, 48)
    assert sigma_tactical.shape == (2, 48)
    assert torch.all(sigma_tactical > 0), "Tactical uncertainty must be positive"
    
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


def test_tactical_embedder_advanced():
    """Test advanced tactical embedder with all features."""
    # Initialize embedder
    tactical_embedder = TacticalEmbedder(
        input_dim=7,
        hidden_dim=128,
        output_dim=48,
        n_layers=3,
        attention_scales=[5, 15, 30]
    )
    
    # Test batch
    batch_size = 4
    input_tactical = torch.randn(batch_size, 60, 7)
    
    # Test 1: Basic forward pass
    mu, sigma = tactical_embedder(input_tactical)
    assert mu.shape == (batch_size, 48), f"Expected {(batch_size, 48)}, got {mu.shape}"
    assert sigma.shape == (batch_size, 48), f"Expected {(batch_size, 48)}, got {sigma.shape}"
    assert torch.all(sigma > 0), "Uncertainty must be positive"
    
    # Test 2: Attention weights
    mu, sigma, attention_weights = tactical_embedder(
        input_tactical, 
        return_attention_weights=True
    )
    assert attention_weights.shape == (batch_size, 60)
    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(batch_size))
    
    # Test 3: All states return
    mu, sigma, lstm_states = tactical_embedder(
        input_tactical,
        return_all_states=True
    )
    assert len(lstm_states) == 3, "Should have 3 LSTM layer outputs"
    
    # Test 4: MC Dropout predictions
    mc_mean, mc_std = tactical_embedder.get_mc_predictions(input_tactical, n_samples=10)
    assert mc_mean.shape == (batch_size, 48)
    assert mc_std.shape == (batch_size, 48)
    assert torch.all(mc_std > 0), "MC std should be positive"
    
    # Test 5: Gradient flow
    loss = mu.mean() + sigma.mean()
    loss.backward()
    
    for name, param in tactical_embedder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    # Test 6: Momentum signal extraction
    momentum_signal = tactical_embedder.extract_momentum_signal(input_tactical)
    assert momentum_signal.shape == (batch_size, 60)
    assert torch.allclose(momentum_signal.sum(dim=1), torch.ones(batch_size))
    
    # Test 7: Optimized forward pass
    tactical_embedder.enable_inference_mode()
    mu_opt, sigma_opt = tactical_embedder.forward_optimized(input_tactical)
    assert mu_opt.shape == (batch_size, 48)
    assert sigma_opt.shape == (batch_size, 48)


def test_momentum_analyzer():
    """Test momentum analyzer functionality."""
    from src.agents.main_core.models import MomentumAnalyzer
    
    analyzer = MomentumAnalyzer(window_size=50)
    
    # Create test data
    batch_size = 2
    embeddings = torch.randn(batch_size, 48)
    attention_weights = torch.softmax(torch.randn(batch_size, 60), dim=1)
    
    # Create mock LSTM states
    lstm_states = [
        torch.randn(batch_size, 60, 256),  # Layer 1
        torch.randn(batch_size, 60, 256),  # Layer 2
        torch.randn(batch_size, 60, 256)   # Layer 3
    ]
    
    # Test analysis
    metrics = analyzer.analyze(embeddings, attention_weights, lstm_states)
    
    # Check required metrics
    required_keys = [
        'momentum_clarity', 'recent_focus_ratio', 'attention_peak_strength',
        'momentum_stability', 'directional_consistency', 'patterns'
    ]
    
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
    
    # Check metric ranges
    assert 0 <= metrics['momentum_clarity'] <= 1
    assert metrics['recent_focus_ratio'] >= 0
    assert 0 <= metrics['attention_peak_strength'] <= 1
    assert 0 <= metrics['momentum_stability'] <= 1
    assert 0 <= metrics['directional_consistency'] <= 1
    assert isinstance(metrics['patterns'], list)
    
    # Test pattern detection
    patterns = metrics['patterns']
    valid_patterns = ['acceleration', 'deceleration', 'reversal', 'continuation']
    for pattern in patterns:
        assert pattern in valid_patterns, f"Invalid pattern detected: {pattern}"


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


class TestAdvancedSharedPolicyNetwork:
    """Comprehensive test suite for Advanced Shared Policy Network."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'state_dim': 512,
            'hidden_dim': 512,
            'n_heads': 8,
            'n_layers': 6,
            'dropout_rate': 0.2
        }
        
    @pytest.fixture
    def shared_policy(self, config):
        """Initialize shared policy for testing."""
        return SharedPolicyNetwork(**config)
        
    def test_shared_policy_forward(self, shared_policy):
        """Test shared policy network forward pass."""
        batch_size = 4
        unified_state = torch.randn(batch_size, 512)
        
        # Forward pass
        output = shared_policy(unified_state)
        
        # Check output type
        assert isinstance(output, PolicyOutput)
        
        # Check output shapes
        assert output.action_logits.shape == (batch_size, 2)
        assert output.value_estimate.shape == (batch_size, 1)
        assert output.confidence.shape == (batch_size, 1)
        
        # Check confidence is between 0 and 1
        assert torch.all(output.confidence >= 0) and torch.all(output.confidence <= 1)
        
    def test_shared_policy_with_attention(self, shared_policy):
        """Test shared policy with attention weights return."""
        unified_state = torch.randn(2, 512)
        
        output = shared_policy(unified_state, return_attention=True)
        
        assert output.attention_weights is not None
        assert output.attention_weights.dim() == 3  # [n_layers, batch, seq_len]
        
    def test_shared_policy_state_segmentation(self, shared_policy):
        """Test unified state segmentation."""
        unified_state = torch.randn(1, 512)
        
        segments = shared_policy._segment_unified_state(unified_state)
        
        assert len(segments) == 5  # structure, tactical, regime, lvn, extended
        assert segments[0].shape == (1, 64)   # Structure
        assert segments[1].shape == (1, 48)   # Tactical
        assert segments[2].shape == (1, 16)   # Regime
        assert segments[3].shape == (1, 8)    # LVN
        assert segments[4].shape == (1, 376)  # Extended features
        
    def test_action_probs_method(self, shared_policy):
        """Test get_action_probs method."""
        unified_state = torch.randn(3, 512)
        
        probs = shared_policy.get_action_probs(unified_state)
        
        assert probs.shape == (3, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3))
        assert torch.all(probs >= 0) and torch.all(probs <= 1)


class TestMCDropoutConsensus:
    """Test suite for MC Dropout consensus mechanism."""
    
    @pytest.fixture
    def mc_consensus(self):
        """Initialize MC Dropout consensus."""
        return MCDropoutConsensus(n_samples=10, confidence_threshold=0.65)
        
    @pytest.fixture
    def policy(self):
        """Simple policy for testing."""
        return SharedPolicyNetwork(state_dim=512, n_layers=2)
        
    def test_mc_dropout_evaluation(self, mc_consensus, policy):
        """Test MC Dropout evaluation."""
        unified_state = torch.randn(2, 512)
        
        results = mc_consensus.evaluate(policy, unified_state)
        
        # Check required keys
        expected_keys = [
            'action_probs', 'action_std', 'consensus_score', 'entropy',
            'value_mean', 'value_std', 'confidence_mean', 'confidence_std',
            'should_qualify', 'qualify_prob'
        ]
        
        for key in expected_keys:
            assert key in results
            
        # Check shapes
        assert results['action_probs'].shape == (2, 2)
        assert results['should_qualify'].shape == (2,)
        assert results['qualify_prob'].shape == (2,)
        
        # Check probability constraints
        assert torch.allclose(results['action_probs'].sum(dim=-1), torch.ones(2))
        assert torch.all(results['consensus_score'] >= -1) and torch.all(results['consensus_score'] <= 1)
        
    def test_detailed_analysis(self, mc_consensus, policy):
        """Test detailed uncertainty analysis."""
        unified_state = torch.randn(1, 512)
        
        results = mc_consensus.evaluate(policy, unified_state, detailed_analysis=True)
        
        # Check additional metrics
        detailed_keys = [
            'predictive_uncertainty', 'aleatoric_uncertainty', 
            'epistemic_uncertainty', 'sample_diversity'
        ]
        
        for key in detailed_keys:
            assert key in results
            
    def test_uncertainty_calculations(self, mc_consensus):
        """Test uncertainty calculation methods."""
        # Mock probability tensor
        probs = torch.rand(10, 2, 2)  # [n_samples, batch, 2]
        probs = F.softmax(probs, dim=-1)
        
        # Test uncertainty calculations
        predictive = mc_consensus._calculate_predictive_uncertainty(probs)
        aleatoric = mc_consensus._calculate_aleatoric_uncertainty(probs)
        epistemic = mc_consensus._calculate_epistemic_uncertainty(probs)
        
        assert predictive.shape == (2,)
        assert aleatoric.shape == (2,)
        assert epistemic.shape == (2,)
        
        # Epistemic should be difference between predictive and aleatoric
        assert torch.allclose(epistemic, predictive - aleatoric, atol=1e-5)


class TestAdvancedDecisionGate:
    """Test suite for Advanced Decision Gate."""
    
    @pytest.fixture
    def config(self):
        """Decision gate configuration."""
        return {
            'input_dim': 640,
            'hidden_dim': 256,
            'risk_threshold': 0.3,
            'confidence_threshold': 0.7
        }
        
    @pytest.fixture
    def decision_gate(self, config):
        """Initialize decision gate."""
        return AdvancedDecisionGate(config)
        
    @pytest.fixture
    def risk_proposal(self):
        """Mock risk proposal."""
        return {
            'position_size': 2,
            'position_size_pct': 0.02,
            'leverage': 1.0,
            'dollar_risk': 200,
            'portfolio_heat': 0.06,
            'stop_loss_distance': 20,
            'stop_loss_atr_multiple': 1.5,
            'use_trailing_stop': True,
            'take_profit_distance': 60,
            'risk_reward_ratio': 3.0,
            'expected_return': 600,
            'risk_metrics': {
                'portfolio_risk_score': 0.4,
                'correlation_risk': 0.2,
                'concentration_risk': 0.1,
                'market_risk_multiplier': 1.2
            },
            'confidence_scores': {
                'overall_confidence': 0.75,
                'sl_confidence': 0.8,
                'tp_confidence': 0.7,
                'size_confidence': 0.8
            }
        }
        
    @pytest.fixture
    def mc_consensus(self):
        """Mock MC consensus results."""
        return {
            'should_qualify': torch.tensor([True]),
            'qualify_prob': torch.tensor([0.8])
        }
        
    def test_decision_gate_forward(self, decision_gate, risk_proposal, mc_consensus):
        """Test decision gate forward pass."""
        unified_state = torch.randn(1, 512)
        
        results = decision_gate(unified_state, risk_proposal, mc_consensus)
        
        # Check required keys
        expected_keys = [
            'decision_logits', 'decision_probs', 'execute_prob', 'confidence',
            'should_execute', 'risk_adjusted', 'decision_factors'
        ]
        
        for key in expected_keys:
            assert key in results
            
        # Check shapes
        assert results['decision_logits'].shape == (1, 2)
        assert results['decision_probs'].shape == (1, 2)
        assert results['execute_prob'].shape == (1,)
        assert results['confidence'].shape == (1, 1)
        assert results['should_execute'].shape == (1,)
        
        # Check constraints
        assert torch.allclose(results['decision_probs'].sum(dim=-1), torch.ones(1))
        assert torch.all(results['confidence'] >= 0) and torch.all(results['confidence'] <= 1)
        
    def test_risk_proposal_encoder(self, config, risk_proposal):
        """Test risk proposal encoder."""
        encoder = RiskProposalEncoder(config)
        
        risk_vector = encoder(risk_proposal)
        
        assert risk_vector.shape == (1, 128)
        assert not torch.isnan(risk_vector).any()
        
    def test_risk_adjustment(self, decision_gate, risk_proposal, mc_consensus):
        """Test risk-based decision adjustment."""
        unified_state = torch.randn(1, 512)
        
        # Test with high risk
        high_risk_proposal = risk_proposal.copy()
        high_risk_proposal['risk_metrics']['portfolio_risk_score'] = 0.9
        
        results_high_risk = decision_gate(unified_state, high_risk_proposal, mc_consensus)
        
        # Test with low risk
        low_risk_proposal = risk_proposal.copy()
        low_risk_proposal['risk_metrics']['portfolio_risk_score'] = 0.1
        
        results_low_risk = decision_gate(unified_state, low_risk_proposal, mc_consensus)
        
        # High risk should be less likely to execute
        assert results_low_risk['execute_prob'] >= results_high_risk['execute_prob']


class TestMAPPOTraining:
    """Test suite for MAPPO training implementation."""
    
    @pytest.fixture
    def config(self):
        """Training configuration."""
        return {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'value_clip': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'batch_size': 64,
            'n_epochs': 4,
            'n_minibatches': 2,
            'training_steps': 1000
        }
        
    @pytest.fixture
    def trainer(self, config):
        """Initialize MAPPO trainer."""
        return MAPPOTrainer(config)
        
    def test_gae_computation(self, trainer):
        """Test GAE computation."""
        batch_size, seq_len = 2, 5
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len + 1)
        dones = torch.zeros(batch_size, seq_len)
        
        advantages, returns = trainer.compute_gae(rewards, values, dones)
        
        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()
        
    def test_experience_buffer(self):
        """Test experience buffer functionality."""
        buffer = ExperienceBuffer(capacity=100, state_dim=512)
        
        # Add experiences
        for i in range(10):
            buffer.add(
                state=torch.randn(512),
                action=i % 2,
                reward=np.random.randn(),
                done=i == 9,
                log_prob=np.random.randn(),
                value=np.random.randn()
            )
            
        assert buffer.size == 10
        
        # Sample batch
        batch = buffer.sample(5)
        assert batch['states'].shape == (5, 512)
        assert batch['actions'].shape == (5,)
        
        # Get all experiences
        all_exp = buffer.get_all()
        assert all_exp['states'].shape == (10, 512)
        
        # Clear buffer
        buffer.clear()
        assert buffer.size == 0
        
    def test_trading_reward_calculation(self):
        """Test trading reward calculation function."""
        config = {
            'miss_penalty': -0.02,
            'correct_pass_reward': 0.01,
            'false_qualify_penalty': -0.05
        }
        
        # Test passing on unprofitable opportunity
        outcome_unprofitable = {'was_profitable': False}
        reward = calculate_trading_reward('pass', outcome_unprofitable, config)
        assert reward == config['correct_pass_reward']
        
        # Test passing on profitable opportunity
        outcome_profitable = {'was_profitable': True}
        reward = calculate_trading_reward('pass', outcome_profitable, config)
        assert reward == config['miss_penalty']
        
        # Test qualifying and executing profitable trade
        outcome_executed = {
            'trade_executed': True,
            'pnl': 100,
            'risk_taken': 50
        }
        reward = calculate_trading_reward('qualify', outcome_executed, config)
        assert reward > 0  # Should be positive for profitable trade


class TestProductionValidation:
    """Production validation tests."""
    
    def test_gradient_flow(self):
        """Test gradient flow through entire system."""
        policy = SharedPolicyNetwork(state_dim=512, n_layers=2)
        
        # Forward pass
        state = torch.randn(2, 512, requires_grad=True)
        output = policy(state)
        
        # Compute loss
        target_actions = torch.tensor([0, 1])
        loss = F.cross_entropy(output.action_logits, target_actions)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert state.grad is not None
        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                
    def test_memory_efficiency(self):
        """Test memory efficiency of models."""
        # Test with batch processing
        batch_sizes = [1, 4, 16, 64]
        
        policy = SharedPolicyNetwork(state_dim=512)
        
        for batch_size in batch_sizes:
            state = torch.randn(batch_size, 512)
            
            # Forward pass should not explode memory
            with torch.no_grad():
                output = policy(state)
                
            assert output.action_logits.shape == (batch_size, 2)
            
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        policy = SharedPolicyNetwork(state_dim=512)
        
        # Test with extreme values
        extreme_state = torch.full((1, 512), 1000.0)
        output = policy(extreme_state)
        
        # Check for NaN/Inf
        assert not torch.isnan(output.action_logits).any()
        assert not torch.isinf(output.action_logits).any()
        assert not torch.isnan(output.confidence).any()
        
        # Test with zeros
        zero_state = torch.zeros(1, 512)
        output = policy(zero_state)
        
        assert not torch.isnan(output.action_logits).any()
        assert not torch.isnan(output.confidence).any()
        
    def test_consistency_across_runs(self):
        """Test model consistency across multiple runs."""
        torch.manual_seed(42)
        
        policy = SharedPolicyNetwork(state_dim=512)
        state = torch.randn(1, 512)
        
        # Run multiple times
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                output = policy(state)
                outputs.append(output.action_logits)
                
        # All outputs should be identical (deterministic)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i])
            
    def test_device_compatibility(self):
        """Test CPU/GPU device compatibility."""
        devices = ['cpu']
        
        # Add CUDA if available
        if torch.cuda.is_available():
            devices.append('cuda')
            
        for device in devices:
            policy = SharedPolicyNetwork(state_dim=512).to(device)
            state = torch.randn(1, 512).to(device)
            
            output = policy(state)
            
            assert output.action_logits.device.type == device
            assert output.confidence.device.type == device


class TestIntegrationFlow:
    """Integration tests for complete decision flow."""
    
    def test_end_to_end_decision_flow(self):
        """Test complete decision flow from state to execution decision."""
        # Setup components
        policy = SharedPolicyNetwork(state_dim=512)
        mc_consensus = MCDropoutConsensus(n_samples=5)  # Reduced for testing
        decision_gate = AdvancedDecisionGate({
            'input_dim': 640,
            'hidden_dim': 256,
            'risk_threshold': 0.3,
            'confidence_threshold': 0.7
        })
        
        # Create test data
        unified_state = torch.randn(1, 512)
        risk_proposal = {
            'position_size': 2,
            'position_size_pct': 0.02,
            'leverage': 1.0,
            'dollar_risk': 200,
            'portfolio_heat': 0.06,
            'stop_loss_distance': 20,
            'stop_loss_atr_multiple': 1.5,
            'use_trailing_stop': True,
            'take_profit_distance': 60,
            'risk_reward_ratio': 3.0,
            'expected_return': 600,
            'risk_metrics': {
                'portfolio_risk_score': 0.4,
                'correlation_risk': 0.2,
                'concentration_risk': 0.1,
                'market_risk_multiplier': 1.2
            },
            'confidence_scores': {
                'overall_confidence': 0.75,
                'sl_confidence': 0.8,
                'tp_confidence': 0.7,
                'size_confidence': 0.8
            }
        }
        
        # Step 1: MC Dropout consensus
        consensus_results = mc_consensus.evaluate(policy, unified_state)
        
        # Step 2: Decision gate
        final_decision = decision_gate(unified_state, risk_proposal, consensus_results)
        
        # Verify flow completion
        assert 'should_execute' in final_decision
        assert isinstance(final_decision['should_execute'].item(), bool)
        
    def test_uncertainty_propagation(self):
        """Test uncertainty propagation through the system."""
        policy = SharedPolicyNetwork(state_dim=512)
        mc_consensus = MCDropoutConsensus(n_samples=10)
        
        # High uncertainty state (extreme values)
        high_uncertainty_state = torch.randn(1, 512) * 10
        
        # Low uncertainty state (near zero)
        low_uncertainty_state = torch.randn(1, 512) * 0.1
        
        # Evaluate both
        high_results = mc_consensus.evaluate(policy, high_uncertainty_state, detailed_analysis=True)
        low_results = mc_consensus.evaluate(policy, low_uncertainty_state, detailed_analysis=True)
        
        # High uncertainty should have higher entropy
        assert high_results['entropy'].item() >= low_results['entropy'].item()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])