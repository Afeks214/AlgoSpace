import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()


class Config:
    """Configuration manager for the AlgoSpace system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        self._config_path = config_path or self._get_default_config_path()
        self._config_data: Dict[str, Any] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / "config" / "settings.yaml")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self._config_path, 'r') as file:
                # Load YAML with environment variable substitution
                raw_config = yaml.safe_load(file)
                self._config_data = self._substitute_env_vars(raw_config)
            
            logger.info("Configuration loaded successfully", path=self._config_path)
            
        except FileNotFoundError:
            logger.error("Configuration file not found", path=self._config_path)
            raise
        except yaml.YAMLError as e:
            logger.error("Error parsing YAML configuration", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error loading configuration", error=str(e))
            raise
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_var_string(obj)
        else:
            return obj
    
    def _substitute_env_var_string(self, value: str) -> Any:
        """Substitute environment variables in a string"""
        if not value.startswith('${') or not value.endswith('}'):
            return value
        
        # Parse ${VAR_NAME:default_value} format
        var_spec = value[2:-1]  # Remove ${ and }
        
        if ':' in var_spec:
            var_name, default_value = var_spec.split(':', 1)
            env_value = os.getenv(var_name, default_value)
        else:
            var_name = var_spec
            env_value = os.getenv(var_name)
            if env_value is None:
                logger.warning("Environment variable not set", var_name=var_name)
                return value
        
        # Try to convert to appropriate type
        return self._convert_type(env_value)
    
    def _convert_type(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation (e.g., 'system.mode')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        current = self._config_data
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., 'indicators')
            
        Returns:
            Section dictionary or empty dict if not found
        """
        return self.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        current = self._config_data
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    @property
    def system_mode(self) -> str:
        """Get system mode (backtest, paper, live)"""
        return self.get('system.mode', 'backtest')
    
    @property
    def is_backtest(self) -> bool:
        """Check if system is in backtest mode"""
        return self.system_mode == 'backtest'
    
    @property
    def is_live(self) -> bool:
        """Check if system is in live mode"""
        return self.system_mode == 'live'
    
    @property
    def symbols(self) -> list:
        """Get list of trading symbols"""
        return self.get('symbols', ['ES'])
    
    @property
    def primary_symbol(self) -> str:
        """Get primary trading symbol"""
        symbols = self.symbols
        return symbols[0] if symbols else 'ES'
    
    @property
    def timeframes(self) -> list:
        """Get list of timeframes"""
        return self.get('timeframes', [5, 30])
    
    def get_indicator_config(self, indicator_name: str) -> Dict[str, Any]:
        """Get configuration for specific indicator"""
        return self.get(f'indicators.{indicator_name}', {})
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for specific agent"""
        return self.get(f'agents.{agent_name}', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data handler configuration"""
        return self.get_section('data_handler')
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self.get_section('risk_management')
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration"""
        return self.get_section('execution')
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get_section('logging')
    
    def to_dict(self) -> Dict[str, Any]:
        """Return entire configuration as dictionary"""
        return self._config_data.copy()
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"Config(path={self._config_path}, mode={self.system_mode})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"Config(path='{self._config_path}', loaded_keys={list(self._config_data.keys())})"


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern)
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        Configuration instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reload_config() -> None:
    """Reload global configuration"""
    global _config_instance
    
    if _config_instance is not None:
        _config_instance.reload()
    else:
        logger.warning("No configuration instance to reload")