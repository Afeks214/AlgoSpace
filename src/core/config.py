# src/core/config.py
"""
Handles loading and validation of the system's YAML configuration file.
"""
import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required fields."""
    pass


def load_config(path: str = 'config/settings.yaml') -> Dict[str, Any]:
    """
    Loads the YAML configuration file and injects credentials from environment variables.

    Args:
        path: The path to the settings.yaml file.

    Returns:
        The complete configuration dictionary.
    
    Raises:
        FileNotFoundError: If the config file is not found.
        ConfigurationError: If essential configuration is missing or invalid.
    """
    config_path = Path(path)
    logger.info(f"Loading configuration from: {config_path.absolute()}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path.absolute()}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML configuration: {e}")

    # Validate basic structure
    required_sections = ['data_handler', 'execution', 'risk_management', 'agents', 'models']
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ConfigurationError(f"Missing required configuration sections: {missing_sections}")

    # Inject sensitive credentials from environment variables
    _inject_credentials(config)
    
    # Validate configuration
    _validate_config(config)
    
    logger.info("Configuration loaded and validated successfully")
    return config


def _inject_credentials(config: Dict[str, Any]) -> None:
    """
    Injects sensitive credentials from environment variables.
    
    Args:
        config: The configuration dictionary to modify.
    
    Raises:
        ConfigurationError: If required credentials are missing.
    """
    # Rithmic credentials for live trading
    if config.get('data_handler', {}).get('type') == 'rithmic':
        rithmic_creds = {
            'user': os.environ.get('RITHMIC_USER'),
            'password': os.environ.get('RITHMIC_PASSWORD'),
            'system': os.environ.get('RITHMIC_SYSTEM'),
            'gateway': os.environ.get('RITHMIC_GATEWAY', 'apis.rithmic.com:443')
        }
        
        missing_creds = [k for k, v in rithmic_creds.items() if not v and k != 'gateway']
        if missing_creds:
            raise ConfigurationError(
                f"Missing Rithmic credentials in environment variables: {missing_creds}"
            )
        
        config['data_handler']['live_settings'] = rithmic_creds
        logger.info("Rithmic credentials injected from environment variables")

    # API keys for any external services
    api_keys = {}
    for key in ['ALPHAVANTAGE_KEY', 'POLYGON_KEY', 'IB_GATEWAY_KEY']:
        if key in os.environ:
            api_keys[key.lower()] = os.environ[key]
    
    if api_keys:
        config.setdefault('api_keys', {}).update(api_keys)
        logger.info(f"Injected {len(api_keys)} API keys from environment variables")


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the configuration for required fields and correct types.
    
    Args:
        config: The configuration dictionary to validate.
    
    Raises:
        ConfigurationError: If validation fails.
    """
    # Validate data_handler configuration
    data_config = config.get('data_handler', {})
    if 'type' not in data_config:
        raise ConfigurationError("data_handler.type is required (rithmic, ib, or backtest)")
    
    if data_config['type'] not in ['rithmic', 'ib', 'backtest']:
        raise ConfigurationError("data_handler.type must be 'rithmic', 'ib', or 'backtest'")
    
    # For backtest mode, validate backtest_file exists
    if data_config['type'] == 'backtest':
        if 'backtest_file' not in data_config:
            raise ConfigurationError("data_handler.backtest_file is required for backtest mode")
    
    # Validate model paths
    models_config = config.get('models', {})
    required_models = ['rde_path', 'mrms_path']
    missing_models = [m for m in required_models if m not in models_config]
    if missing_models:
        raise ConfigurationError(f"Missing required model paths: {missing_models}")
    
    # Validate model files exist
    for model_name, model_path in models_config.items():
        if model_path and not Path(model_path).parent.exists():
            logger.warning(f"Model directory does not exist for {model_name}: {Path(model_path).parent}")
    
    # Validate risk parameters
    risk_config = config.get('risk_management', {})
    if 'max_position_size' not in risk_config:
        config['risk_management']['max_position_size'] = 100000  # Default
        logger.info("Using default max_position_size: 100000")
    
    if 'max_daily_loss' not in risk_config:
        config['risk_management']['max_daily_loss'] = 5000  # Default
        logger.info("Using default max_daily_loss: 5000")
    
    logger.info("Configuration validation completed successfully")


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely retrieves a nested configuration value using dot notation.
    
    Args:
        config: The configuration dictionary.
        path: Dot-separated path to the value (e.g., 'data.contracts.MES.symbol').
        default: Default value if path doesn't exist.
    
    Returns:
        The configuration value or default.
    """
    try:
        value = config
        for key in path.split('.'):
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default