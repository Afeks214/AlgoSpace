"""
Comprehensive tests for the Multi-Agent Risk Management Subsystem (M-RMS) component.

This test suite validates the MRMSComponent class functionality including:
- Component initialization and configuration
- Model loading from checkpoint files
- Risk proposal generation interface and output structure
- Input validation and error handling
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.agents.mrms.engine import MRMSComponent
from src.agents.mrms.models import RiskManagementEnsemble


class TestMRMSComponentInitialization:
    """Test MRMSComponent initialization and configuration."""
    
    def test_mrms_component_initialization(self):
        """Verify that the MRMSComponent class can be instantiated correctly."""
        # Create a mock config dictionary
        config = {
            'synergy_dim': 30,
            'account_dim': 10,
            'device': 'cpu',
            'point_value': 5.0,
            'max_position_size': 5,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.1
        }
        
        # Instantiate MRMSComponent
        mrms_component = MRMSComponent(config)
        
        # Verify initialization
        assert mrms_component is not None
        assert mrms_component.synergy_dim == 30
        assert mrms_component.account_dim == 10
        assert mrms_component.device == torch.device('cpu')
        assert mrms_component.point_value == 5.0
        assert mrms_component.max_position_size == 5
        assert mrms_component.model_loaded == False
        
        # Verify that internal RiskManagementEnsemble model is initialized
        assert isinstance(mrms_component.model, RiskManagementEnsemble)
        assert not mrms_component.model.training  # Should be in eval mode
    
    def test_mrms_component_initialization_with_defaults(self):
        """Test MRMSComponent initialization with minimal config (using defaults)."""
        # Minimal config
        config = {}
        
        # Instantiate MRMSComponent
        mrms_component = MRMSComponent(config)
        
        # Verify defaults are applied
        assert mrms_component.synergy_dim == 30     # default
        assert mrms_component.account_dim == 10     # default
        assert mrms_component.device == torch.device('cpu')  # default
        assert mrms_component.point_value == 5.0    # default (MES)
        assert mrms_component.max_position_size == 5  # default
        assert mrms_component.model_loaded == False
    
    def test_mrms_component_cuda_initialization(self):
        """Test MRMSComponent initialization with CUDA device (if available)."""
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        mrms_component = MRMSComponent(config)
        
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert mrms_component.device == expected_device


class TestMRMSModelLoading:
    """Test M-RMS model loading functionality."""
    
    @pytest.fixture
    def mrms_component(self):
        """Create an MRMSComponent instance for testing."""
        config = {
            'synergy_dim': 30,
            'account_dim': 10,
            'device': 'cpu',
            'point_value': 5.0,
            'max_position_size': 5,
            'hidden_dims': [64, 32],  # Smaller for testing
            'dropout': 0.1
        }
        return MRMSComponent(config)
    
    def test_mrms_loads_model_correctly(self, mrms_component):
        """Test the load_model() method with a dummy PyTorch model."""
        # Create a dummy model with the same architecture
        dummy_model = RiskManagementEnsemble(
            synergy_dim=30,
            account_dim=10,
            hidden_dims=[64, 32],
            dropout=0.1
        )
        
        # Save dummy model state dict to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(dummy_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Test loading the model
            mrms_component.load_model(temp_path)
            
            # Verify model was loaded successfully
            assert mrms_component.model_loaded == True
            
            # Verify model is in evaluation mode
            assert not mrms_component.model.training
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def test_mrms_loads_model_with_checkpoint_format(self, mrms_component):
        """Test loading model from checkpoint with metadata."""
        # Create a dummy model
        dummy_model = RiskManagementEnsemble(
            synergy_dim=30,
            account_dim=10,
            hidden_dims=[64, 32],
            dropout=0.1
        )
        
        # Create checkpoint with metadata
        checkpoint = {
            'model_state_dict': dummy_model.state_dict(),
            'training_iterations': 50000,
            'final_reward_mean': 125.5,
            'optimizer_state_dict': {'test': 'data'}
        }
        
        # Save checkpoint to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(checkpoint, temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Test loading the checkpoint
            mrms_component.load_model(temp_path)
            
            # Verify model was loaded successfully
            assert mrms_component.model_loaded == True
            assert not mrms_component.model.training
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def test_mrms_load_model_file_not_found(self, mrms_component):
        """Test that load_model raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            mrms_component.load_model("/nonexistent/path/model.pth")
    
    def test_mrms_load_model_invalid_weights(self, mrms_component):
        """Test that load_model handles invalid model weights gracefully."""
        # Create a model with different architecture
        wrong_model = torch.nn.Linear(10, 5)
        
        # Save wrong model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(wrong_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Test loading incompatible weights should raise RuntimeError
            with pytest.raises(RuntimeError):
                mrms_component.load_model(temp_path)
                
        finally:
            # Clean up temporary file
            os.unlink(temp_path)


class TestMRMSRiskProposal:
    """Test M-RMS risk proposal generation functionality."""
    
    @pytest.fixture
    def loaded_mrms_component(self):
        """Create a loaded MRMSComponent for risk proposal testing."""
        config = {
            'synergy_dim': 30,
            'account_dim': 10,
            'device': 'cpu',
            'point_value': 5.0,
            'max_position_size': 5,
            'hidden_dims': [64, 32],
            'dropout': 0.0  # No dropout for deterministic testing
        }
        
        mrms_component = MRMSComponent(config)
        
        # Create and load a dummy model
        dummy_model = RiskManagementEnsemble(
            synergy_dim=30,
            account_dim=10,
            hidden_dims=[64, 32],
            dropout=0.0
        )
        
        # Save and load dummy model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(dummy_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            mrms_component.load_model(temp_path)
        finally:
            os.unlink(temp_path)
        
        return mrms_component
    
    @pytest.fixture
    def valid_trade_qualification(self):
        """Create a valid TradeQualification dictionary for testing."""
        return {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(10).astype(np.float32),
            'entry_price': 5250.75,
            'direction': 'LONG',
            'atr': 25.5,
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
    
    def test_generate_risk_proposal_interface(self, loaded_mrms_component, valid_trade_qualification):
        """Test generate_risk_proposal method interface and output structure."""
        # Call generate_risk_proposal
        risk_proposal = loaded_mrms_component.generate_risk_proposal(valid_trade_qualification)
        
        # Assert that returned value is a dictionary
        assert isinstance(risk_proposal, dict)
        
        # Assert that dictionary contains required top-level keys
        required_keys = [
            'entry_plan', 'stop_loss_plan', 'risk_metrics'
        ]
        
        for key in required_keys:
            assert key in risk_proposal, f"Missing required key: {key}"
        
        # Verify entry_plan structure
        entry_plan = risk_proposal['entry_plan']
        assert isinstance(entry_plan, dict)
        assert 'position_size' in entry_plan
        assert 'entry_price' in entry_plan
        assert 'direction' in entry_plan
        
        # Verify stop_loss_plan structure
        stop_loss_plan = risk_proposal['stop_loss_plan']
        assert isinstance(stop_loss_plan, dict)
        assert 'stop_loss_price' in stop_loss_plan
        assert 'sl_atr_multiplier' in stop_loss_plan
        
        # Verify risk_metrics structure
        risk_metrics = risk_proposal['risk_metrics']
        assert isinstance(risk_metrics, dict)
        assert 'risk_amount' in risk_metrics
        assert 'confidence_score' in risk_metrics
    
    def test_generate_risk_proposal_value_types(self, loaded_mrms_component, valid_trade_qualification):
        """Test that generate_risk_proposal returns correct value types."""
        risk_proposal = loaded_mrms_component.generate_risk_proposal(valid_trade_qualification)
        
        # Check entry plan types
        entry_plan = risk_proposal['entry_plan']
        assert isinstance(entry_plan['position_size'], (int, float))
        assert isinstance(entry_plan['entry_price'], (int, float))
        assert isinstance(entry_plan['direction'], str)
        
        # Check stop loss plan types
        stop_loss_plan = risk_proposal['stop_loss_plan']
        assert isinstance(stop_loss_plan['stop_loss_price'], (int, float))
        assert isinstance(stop_loss_plan['sl_atr_multiplier'], (int, float))
        
        # Check risk metrics types
        risk_metrics = risk_proposal['risk_metrics']
        assert isinstance(risk_metrics['risk_amount'], (int, float))
        assert isinstance(risk_metrics['confidence_score'], (int, float))
        
        # Check value ranges
        assert 0 <= entry_plan['position_size'] <= 5  # max_position_size
        assert 0 <= risk_metrics['confidence_score'] <= 1  # confidence should be 0-1
        assert stop_loss_plan['sl_atr_multiplier'] > 0  # ATR multiplier should be positive
    
    def test_generate_risk_proposal_long_trade(self, loaded_mrms_component):
        """Test risk proposal generation for LONG trades."""
        trade_qualification = {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(10).astype(np.float32),
            'entry_price': 5250.0,
            'direction': 'LONG',
            'atr': 20.0,
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
        
        risk_proposal = loaded_mrms_component.generate_risk_proposal(trade_qualification)
        
        # For LONG trades, stop loss should be below entry price
        entry_price = risk_proposal['entry_plan']['entry_price']
        stop_loss_price = risk_proposal['stop_loss_plan']['stop_loss_price']
        
        assert stop_loss_price < entry_price, "Stop loss should be below entry for LONG trades"
    
    def test_generate_risk_proposal_short_trade(self, loaded_mrms_component):
        """Test risk proposal generation for SHORT trades."""
        trade_qualification = {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(10).astype(np.float32),
            'entry_price': 5250.0,
            'direction': 'SHORT',
            'atr': 20.0,
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
        
        risk_proposal = loaded_mrms_component.generate_risk_proposal(trade_qualification)
        
        # For SHORT trades, stop loss should be above entry price
        entry_price = risk_proposal['entry_plan']['entry_price']
        stop_loss_price = risk_proposal['stop_loss_plan']['stop_loss_price']
        
        assert stop_loss_price > entry_price, "Stop loss should be above entry for SHORT trades"


class TestMRMSInputValidation:
    """Test M-RMS input validation and error handling."""
    
    @pytest.fixture
    def loaded_mrms_component(self):
        """Create a loaded MRMSComponent for validation testing."""
        config = {
            'synergy_dim': 30,
            'account_dim': 10,
            'device': 'cpu'
        }
        
        mrms_component = MRMSComponent(config)
        
        # Create and load a dummy model
        dummy_model = RiskManagementEnsemble(
            synergy_dim=30,
            account_dim=10
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(dummy_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            mrms_component.load_model(temp_path)
        finally:
            os.unlink(temp_path)
        
        return mrms_component
    
    def test_generate_risk_proposal_model_not_loaded(self):
        """Test that generate_risk_proposal raises RuntimeError when model not loaded."""
        config = {'synergy_dim': 30, 'account_dim': 10}
        mrms_component = MRMSComponent(config)
        
        trade_qualification = {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(10).astype(np.float32),
            'entry_price': 5250.0,
            'direction': 'LONG',
            'atr': 20.0,
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
        
        with pytest.raises(RuntimeError, match="Model weights not loaded"):
            mrms_component.generate_risk_proposal(trade_qualification)
    
    def test_generate_risk_proposal_invalid_synergy_vector(self, loaded_mrms_component):
        """Test validation of synergy vector."""
        # Wrong size synergy vector
        trade_qualification = {
            'synergy_vector': np.random.randn(20).astype(np.float32),  # Should be 30
            'account_state_vector': np.random.randn(10).astype(np.float32),
            'entry_price': 5250.0,
            'direction': 'LONG',
            'atr': 20.0,
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
        
        with pytest.raises(ValueError):
            loaded_mrms_component.generate_risk_proposal(trade_qualification)
    
    def test_generate_risk_proposal_invalid_account_vector(self, loaded_mrms_component):
        """Test validation of account state vector."""
        # Wrong size account vector
        trade_qualification = {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(5).astype(np.float32),  # Should be 10
            'entry_price': 5250.0,
            'direction': 'LONG',
            'atr': 20.0,
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
        
        with pytest.raises(ValueError):
            loaded_mrms_component.generate_risk_proposal(trade_qualification)
    
    def test_generate_risk_proposal_missing_required_fields(self, loaded_mrms_component):
        """Test validation of required fields."""
        # Missing required fields
        incomplete_qualification = {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(10).astype(np.float32),
            # Missing: entry_price, direction, atr, symbol, timestamp
        }
        
        with pytest.raises(ValueError):
            loaded_mrms_component.generate_risk_proposal(incomplete_qualification)
    
    def test_generate_risk_proposal_invalid_direction(self, loaded_mrms_component):
        """Test validation of trade direction."""
        trade_qualification = {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(10).astype(np.float32),
            'entry_price': 5250.0,
            'direction': 'INVALID',  # Should be 'LONG' or 'SHORT'
            'atr': 20.0,
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
        
        with pytest.raises(ValueError):
            loaded_mrms_component.generate_risk_proposal(trade_qualification)


class TestMRMSEdgeCases:
    """Test M-RMS component edge cases and robustness."""
    
    @pytest.fixture
    def loaded_mrms_component(self):
        """Create a loaded MRMSComponent for edge case testing."""
        config = {
            'synergy_dim': 30,
            'account_dim': 10,
            'device': 'cpu',
            'point_value': 5.0,
            'max_position_size': 3,  # Smaller for testing
            'hidden_dims': [32, 16],
            'dropout': 0.0
        }
        
        mrms_component = MRMSComponent(config)
        
        dummy_model = RiskManagementEnsemble(
            synergy_dim=30,
            account_dim=10,
            hidden_dims=[32, 16],
            dropout=0.0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(dummy_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            mrms_component.load_model(temp_path)
        finally:
            os.unlink(temp_path)
        
        return mrms_component
    
    def test_extreme_input_values(self, loaded_mrms_component):
        """Test M-RMS with extreme input values."""
        # Test with very large synergy values
        extreme_qualification = {
            'synergy_vector': np.full(30, 1000.0, dtype=np.float32),
            'account_state_vector': np.full(10, -1000.0, dtype=np.float32),
            'entry_price': 9999.99,
            'direction': 'LONG',
            'atr': 100.0,
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
        
        # Should not crash and return valid structure
        risk_proposal = loaded_mrms_component.generate_risk_proposal(extreme_qualification)
        assert isinstance(risk_proposal, dict)
        assert 'entry_plan' in risk_proposal
        assert 'stop_loss_plan' in risk_proposal
        assert 'risk_metrics' in risk_proposal
    
    def test_zero_atr_handling(self, loaded_mrms_component):
        """Test M-RMS handling of zero ATR."""
        trade_qualification = {
            'synergy_vector': np.random.randn(30).astype(np.float32),
            'account_state_vector': np.random.randn(10).astype(np.float32),
            'entry_price': 5250.0,
            'direction': 'LONG',
            'atr': 0.0,  # Zero ATR
            'symbol': 'MES',
            'timestamp': datetime.now()
        }
        
        # Should handle gracefully (might use default ATR or reject)
        try:
            risk_proposal = loaded_mrms_component.generate_risk_proposal(trade_qualification)
            # If it succeeds, verify structure
            assert isinstance(risk_proposal, dict)
        except ValueError:
            # Or it might reject zero ATR, which is also valid
            pass
    
    def test_deterministic_output(self, loaded_mrms_component):
        """Test that M-RMS produces deterministic output for same input."""
        # Create fixed input
        fixed_qualification = {
            'synergy_vector': np.random.RandomState(42).randn(30).astype(np.float32),
            'account_state_vector': np.random.RandomState(123).randn(10).astype(np.float32),
            'entry_price': 5250.0,
            'direction': 'LONG',
            'atr': 25.0,
            'symbol': 'MES',
            'timestamp': datetime(2024, 1, 1, 12, 0, 0)
        }
        
        # Get risk proposal twice
        proposal_1 = loaded_mrms_component.generate_risk_proposal(fixed_qualification)
        proposal_2 = loaded_mrms_component.generate_risk_proposal(fixed_qualification)
        
        # Key values should be identical (model in eval mode, no dropout)
        assert proposal_1['entry_plan']['position_size'] == proposal_2['entry_plan']['position_size']
        assert abs(proposal_1['stop_loss_plan']['stop_loss_price'] - 
                  proposal_2['stop_loss_plan']['stop_loss_price']) < 1e-6
        assert abs(proposal_1['risk_metrics']['confidence_score'] - 
                  proposal_2['risk_metrics']['confidence_score']) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])