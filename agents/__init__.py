# Multi-Agent Trading System Components

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Central registry for all trading agents"""
    
    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._agent_configs: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, agent_class: type, config: Optional[Dict[str, Any]] = None):
        """Register a new agent type"""
        self._agents[name] = agent_class
        if config:
            self._agent_configs[name] = config
        logger.info(f"Registered agent: {name}")
    
    def create(self, name: str, **kwargs) -> Any:
        """Create an instance of a registered agent"""
        if name not in self._agents:
            raise ValueError(f"Agent {name} not registered")
        
        config = self._agent_configs.get(name, {})
        config.update(kwargs)
        
        return self._agents[name](**config)
    
    def list_agents(self) -> list:
        """List all registered agents"""
        return list(self._agents.keys())


# Global registry instance
agent_registry = AgentRegistry()


# Agent base classes and utilities will be imported here
__all__ = ['agent_registry', 'AgentRegistry']