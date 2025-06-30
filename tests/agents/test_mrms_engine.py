"""
Comprehensive unit tests for the M-RMS Engine Component.

This test suite validates the functionality and robustness of the MRMSComponent,
ensuring correct initialization, model loading, inference, and error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch


class TestMRMSComponent:
    """Test suite for the M-RMS Component."""
    
    @pytest.fixture
    def base_config(self):
        """Provide a base configuration for M-RMS tests."""
        return {
            'synergy_dim': 30,
            'account_dim': 10,
            'device': 'cpu',
            'point_value': 5.0,
            'max_position_size': 5,
            'hidden_dim': 128,
            'position_agent_hidden': 128,
            'sl_agent_hidden': 64,
            'pt_agent_hidden': 64,
            'dropout_rate': 0.2
        }
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock RiskManagementEnsemble model."""
        mock = MagicMock()
        
        # Mock the sub-agents
        mock.position_agent = MagicMock()
        mock.stop_loss_agent = MagicMock()
        mock.stop_loss_agent.min_multiplier = 0.5
        mock.stop_loss_agent.max_multiplier = 3.0
        mock.profit_target_agent = MagicMock()
        mock.profit_target_agent.min_rr = 1.0
        mock.profit_target_agent.max_rr = 5.0
        
        # Mock model methods
        mock.eval = MagicMock()
        mock.to = MagicMock(return_value=mock)
        mock.parameters = MagicMock(return_value=[])
        
        return mock
    
    @pytest.fixture
    def valid_trade_qualification(self):
        """Provide a valid trade qualification dictionary."""
        return {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(10).astype(np.float32),
            'entry_price': 4500.0,
            'direction': 'LONG',
            'atr': 10.0,
            'symbol': 'ES',
            'timestamp': '2024-01-01 10:00:00'
        }
    
    def test_mrms_component_initialization(self, base_config):
        """Test that MRMSComponent can be created successfully."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            # Import here to trigger the patch
            from src.agents.mrms.engine import MRMSComponent
            
            # Configure the mock
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            MockEnsemble.return_value = mock_model
            
            # Initialize component
            mrms = MRMSComponent(base_config)
            
            # Verify initialization
            assert mrms is not None
            assert mrms.synergy_dim == 30
            assert mrms.account_dim == 10
            assert mrms.point_value == 5.0
            assert mrms.max_position_size == 5
            assert mrms.model_loaded == False
            
            # Verify model creation
            MockEnsemble.assert_called_once_with(
                synergy_dim=30,
                account_dim=10,
                hidden_dim=128,
                position_agent_hidden=128,
                sl_agent_hidden=64,
                pt_agent_hidden=64,
                dropout_rate=0.2
            )
            
            # Verify model is set to eval mode
            mock_model.eval.assert_called_once()
    
    def test_generate_risk_proposal_interface(self, base_config, valid_trade_qualification):
        """Test that the main inference method works with correct input/output structure."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            with patch('src.agents.mrms.engine.torch') as mock_torch:
                from src.agents.mrms.engine import MRMSComponent
                
                # Setup mock model
                mock_model = MagicMock()
                mock_model.eval = MagicMock()
                mock_model.to = MagicMock(return_value=mock_model)
                MockEnsemble.return_value = mock_model
                
                # Mock model outputs
                mock_actions = {
                    'position_size': MagicMock(cpu=lambda: MagicMock(item=lambda: 3)),
                    'sl_atr_multiplier': MagicMock(cpu=lambda: MagicMock(item=lambda: 1.5)),
                    'rr_ratio': MagicMock(cpu=lambda: MagicMock(item=lambda: 2.0))
                }
                
                mock_outputs = {
                    'position_logits': MagicMock(),
                    'sl_multiplier': MagicMock(),
                    'rr_ratio': MagicMock(),
                    'value': MagicMock()
                }
                
                # Setup softmax mock
                mock_probs = MagicMock()
                mock_probs.__getitem__ = MagicMock(return_value=MagicMock(
                    cpu=lambda: MagicMock(item=lambda: 0.85)
                ))
                mock_torch.softmax = MagicMock(return_value=mock_probs)
                
                mock_model.get_action_dict = MagicMock(return_value=mock_actions)
                mock_model.__call__ = MagicMock(return_value=mock_outputs)
                
                # Initialize and load model
                mrms = MRMSComponent(base_config)
                mrms.model_loaded = True  # Bypass load_model for this test
                
                # Generate risk proposal
                proposal = mrms.generate_risk_proposal(valid_trade_qualification)
                
                # Verify output structure
                assert isinstance(proposal, dict)
                
                # Check required top-level keys
                required_keys = [
                    'position_size', 'stop_loss_price', 'take_profit_price',
                    'risk_amount', 'reward_amount', 'risk_reward_ratio',
                    'sl_atr_multiplier', 'confidence_score', 'risk_metrics'
                ]
                
                for key in required_keys:
                    assert key in proposal, f"Missing required key: {key}"
                
                # Verify risk_metrics sub-structure
                assert 'risk_metrics' in proposal
                risk_metrics = proposal['risk_metrics']
                assert 'sl_distance_points' in risk_metrics
                assert 'tp_distance_points' in risk_metrics
                assert 'risk_per_contract' in risk_metrics
                assert 'max_position_allowed' in risk_metrics
                assert 'position_utilization' in risk_metrics
                
                # Verify model was called with no_grad context
                mock_torch.no_grad.assert_called()
    
    def test_risk_proposal_calculation_logic(self, base_config, valid_trade_qualification):
        """Test that risk calculations are mathematically correct."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            with patch('src.agents.mrms.engine.torch') as mock_torch:
                from src.agents.mrms.engine import MRMSComponent
                
                # Setup mock model
                mock_model = MagicMock()
                mock_model.eval = MagicMock()
                mock_model.to = MagicMock(return_value=mock_model)
                MockEnsemble.return_value = mock_model
                
                # Fixed model outputs for testing calculations
                position_size = 2
                sl_multiplier = 1.5
                rr_ratio = 2.5
                
                mock_actions = {
                    'position_size': MagicMock(cpu=lambda: MagicMock(item=lambda: position_size)),
                    'sl_atr_multiplier': MagicMock(cpu=lambda: MagicMock(item=lambda: sl_multiplier)),
                    'rr_ratio': MagicMock(cpu=lambda: MagicMock(item=lambda: rr_ratio))
                }
                
                mock_outputs = {
                    'position_logits': MagicMock(),
                }
                
                # Mock softmax for confidence
                mock_probs = MagicMock()
                mock_probs.__getitem__ = MagicMock(return_value=MagicMock(
                    cpu=lambda: MagicMock(item=lambda: 0.75)
                ))
                mock_torch.softmax = MagicMock(return_value=mock_probs)
                
                mock_model.get_action_dict = MagicMock(return_value=mock_actions)
                mock_model.__call__ = MagicMock(return_value=mock_outputs)
                
                # Initialize component
                mrms = MRMSComponent(base_config)
                mrms.model_loaded = True
                
                # Test LONG trade calculations
                proposal = mrms.generate_risk_proposal(valid_trade_qualification)
                
                # Expected calculations
                entry_price = 4500.0
                atr = 10.0
                sl_distance = sl_multiplier * atr  # 1.5 * 10 = 15
                tp_distance = sl_distance * rr_ratio  # 15 * 2.5 = 37.5
                
                expected_sl = entry_price - sl_distance  # 4500 - 15 = 4485
                expected_tp = entry_price + tp_distance  # 4500 + 37.5 = 4537.5
                
                risk_per_contract = sl_distance * 5.0  # 15 * 5 = 75
                expected_risk = risk_per_contract * position_size  # 75 * 2 = 150
                expected_reward = tp_distance * 5.0 * position_size  # 37.5 * 5 * 2 = 375
                
                # Verify calculations
                assert proposal['position_size'] == position_size
                assert proposal['stop_loss_price'] == round(expected_sl, 2)
                assert proposal['take_profit_price'] == round(expected_tp, 2)
                assert proposal['risk_amount'] == round(expected_risk, 2)
                assert proposal['reward_amount'] == round(expected_reward, 2)
                assert proposal['risk_reward_ratio'] == round(rr_ratio, 2)
                assert proposal['sl_atr_multiplier'] == round(sl_multiplier, 3)
                
                # Test SHORT trade calculations
                valid_trade_qualification['direction'] = 'SHORT'
                proposal_short = mrms.generate_risk_proposal(valid_trade_qualification)
                
                expected_sl_short = entry_price + sl_distance  # 4500 + 15 = 4515
                expected_tp_short = entry_price - tp_distance  # 4500 - 37.5 = 4462.5
                
                assert proposal_short['stop_loss_price'] == round(expected_sl_short, 2)
                assert proposal_short['take_profit_price'] == round(expected_tp_short, 2)
    
    def test_mrms_handles_invalid_input(self, base_config):
        """Test that component fails gracefully with invalid input."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            from src.agents.mrms.engine import MRMSComponent
            
            # Setup mock
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            MockEnsemble.return_value = mock_model
            
            # Initialize component
            mrms = MRMSComponent(base_config)
            mrms.model_loaded = True
            
            # Test missing required field
            invalid_trade_qual = {
                'synergy_vector': np.random.randn(30),
                'account_state_vector': np.random.randn(10),
                # Missing 'entry_price'
                'direction': 'LONG',
                'atr': 10.0
            }
            
            with pytest.raises(ValueError) as exc_info:
                mrms.generate_risk_proposal(invalid_trade_qual)
            
            assert "Missing required field: entry_price" in str(exc_info.value)
            
            # Test invalid synergy vector shape
            invalid_trade_qual2 = {
                'synergy_vector': np.random.randn(20),  # Wrong shape
                'account_state_vector': np.random.randn(10),
                'entry_price': 4500.0,
                'direction': 'LONG',
                'atr': 10.0
            }
            
            with pytest.raises(ValueError) as exc_info:
                mrms.generate_risk_proposal(invalid_trade_qual2)
            
            assert "synergy_vector must have shape (30,)" in str(exc_info.value)
            
            # Test invalid direction
            invalid_trade_qual3 = {
                'synergy_vector': np.random.randn(30),
                'account_state_vector': np.random.randn(10),
                'entry_price': 4500.0,
                'direction': 'INVALID',  # Invalid direction
                'atr': 10.0
            }
            
            with pytest.raises(ValueError) as exc_info:
                mrms.generate_risk_proposal(invalid_trade_qual3)
            
            assert "direction must be either 'LONG' or 'SHORT'" in str(exc_info.value)
    
    def test_model_not_loaded_error(self, base_config, valid_trade_qualification):
        """Test that inference fails properly when model is not loaded."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            from src.agents.mrms.engine import MRMSComponent
            
            # Setup mock
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            MockEnsemble.return_value = mock_model
            
            # Initialize component without loading model
            mrms = MRMSComponent(base_config)
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                mrms.generate_risk_proposal(valid_trade_qualification)
            
            assert "Model weights not loaded" in str(exc_info.value)
    
    def test_load_model_functionality(self, base_config, tmp_path):
        """Test model loading functionality."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            with patch('src.agents.mrms.engine.torch.load') as mock_load:
                from src.agents.mrms.engine import MRMSComponent
                
                # Setup mock model
                mock_model = MagicMock()
                mock_model.eval = MagicMock()
                mock_model.to = MagicMock(return_value=mock_model)
                mock_model.load_state_dict = MagicMock()
                MockEnsemble.return_value = mock_model
                
                # Create a dummy model file
                model_path = tmp_path / "test_model.pth"
                model_path.touch()
                
                # Mock different checkpoint formats
                # Test 1: model_state_dict format
                mock_checkpoint = {'model_state_dict': {'dummy': 'weights'}}
                mock_load.return_value = mock_checkpoint
                
                mrms = MRMSComponent(base_config)
                mrms.load_model(str(model_path))
                
                assert mrms.model_loaded == True
                mock_model.load_state_dict.assert_called_with({'dummy': 'weights'})
                
                # Test 2: state_dict format
                mock_model.load_state_dict.reset_mock()
                mock_checkpoint = {'state_dict': {'other': 'weights'}}
                mock_load.return_value = mock_checkpoint
                
                mrms2 = MRMSComponent(base_config)
                mrms2.load_model(str(model_path))
                
                mock_model.load_state_dict.assert_called_with({'other': 'weights'})
                
                # Test 3: Direct state dict
                mock_model.load_state_dict.reset_mock()
                mock_checkpoint = {'direct': 'weights'}
                mock_load.return_value = mock_checkpoint
                
                mrms3 = MRMSComponent(base_config)
                mrms3.load_model(str(model_path))
                
                mock_model.load_state_dict.assert_called_with({'direct': 'weights'})
    
    def test_load_model_file_not_found(self, base_config):
        """Test that loading non-existent model file raises appropriate error."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            from src.agents.mrms.engine import MRMSComponent
            
            # Setup mock
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            MockEnsemble.return_value = mock_model
            
            mrms = MRMSComponent(base_config)
            
            with pytest.raises(FileNotFoundError) as exc_info:
                mrms.load_model("/non/existent/path.pth")
            
            assert "Model file not found" in str(exc_info.value)
    
    def test_get_model_info(self, base_config):
        """Test the get_model_info method."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            from src.agents.mrms.engine import MRMSComponent
            
            # Setup mock model with get_model_info
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.get_model_info = MagicMock(return_value={
                'architecture': 'Multi-Agent Risk Management Ensemble',
                'sub_agents': ['PositionSizingAgent', 'StopLossAgent', 'ProfitTargetAgent'],
                'total_parameters': 50000
            })
            MockEnsemble.return_value = mock_model
            
            mrms = MRMSComponent(base_config)
            info = mrms.get_model_info()
            
            assert 'architecture' in info
            assert 'model_loaded' in info
            assert 'device' in info
            assert info['model_loaded'] == False
            assert info['point_value'] == 5.0
            assert info['max_position_size'] == 5
    
    def test_position_size_zero_handling(self, base_config, valid_trade_qualification):
        """Test that zero position size is handled correctly."""
        with patch('src.agents.mrms.engine.RiskManagementEnsemble') as MockEnsemble:
            with patch('src.agents.mrms.engine.torch') as mock_torch:
                from src.agents.mrms.engine import MRMSComponent
                
                # Setup mock model to return position_size = 0
                mock_model = MagicMock()
                mock_model.eval = MagicMock()
                mock_model.to = MagicMock(return_value=mock_model)
                MockEnsemble.return_value = mock_model
                
                mock_actions = {
                    'position_size': MagicMock(cpu=lambda: MagicMock(item=lambda: 0)),
                    'sl_atr_multiplier': MagicMock(cpu=lambda: MagicMock(item=lambda: 1.5)),
                    'rr_ratio': MagicMock(cpu=lambda: MagicMock(item=lambda: 2.0))
                }
                
                mock_outputs = {'position_logits': MagicMock()}
                
                mock_probs = MagicMock()
                mock_probs.__getitem__ = MagicMock(return_value=MagicMock(
                    cpu=lambda: MagicMock(item=lambda: 0.6)
                ))
                mock_torch.softmax = MagicMock(return_value=mock_probs)
                
                mock_model.get_action_dict = MagicMock(return_value=mock_actions)
                mock_model.__call__ = MagicMock(return_value=mock_outputs)
                
                mrms = MRMSComponent(base_config)
                mrms.model_loaded = True
                
                proposal = mrms.generate_risk_proposal(valid_trade_qualification)
                
                # When position_size is 0, risk and reward should be 0
                assert proposal['position_size'] == 0
                assert proposal['risk_amount'] == 0
                assert proposal['reward_amount'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])