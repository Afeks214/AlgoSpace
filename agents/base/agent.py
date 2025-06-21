from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all trading agents"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.config = config
        self.state = {}
        self.is_active = False
        self.created_at = datetime.utcnow()
        self.last_action_time = None
        
        logger.info(f"Initialized agent {self.name} with ID {self.agent_id}")
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize agent resources and connections"""
        pass
    
    @abstractmethod
    def observe(self, market_data: Dict[str, Any]) -> None:
        """Process incoming market data"""
        pass
    
    @abstractmethod
    def decide(self) -> Optional[Dict[str, Any]]:
        """Make trading decision based on observations"""
        pass
    
    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading action based on decision"""
        pass
    
    def update_state(self, key: str, value: Any) -> None:
        """Update agent's internal state"""
        self.state[key] = value
        logger.debug(f"Agent {self.name} state updated: {key} = {value}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return self.state.copy()
    
    def activate(self) -> None:
        """Activate the agent"""
        self.is_active = True
        logger.info(f"Agent {self.name} activated")
    
    def deactivate(self) -> None:
        """Deactivate the agent"""
        self.is_active = False
        logger.info(f"Agent {self.name} deactivated")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'last_action_time': self.last_action_time.isoformat() if self.last_action_time else None
        }


class TradingAgent(BaseAgent):
    """Base class for trading-specific agents"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.positions = {}
        self.orders = []
        self.pnl = 0.0
        self.risk_limits = config.get('risk_limits', {})
    
    def check_risk_limits(self, order: Dict[str, Any]) -> bool:
        """Check if order complies with risk limits"""
        # Implement risk checking logic
        max_position = self.risk_limits.get('max_position_size', float('inf'))
        max_loss = self.risk_limits.get('max_daily_loss', float('inf'))
        
        # Basic checks
        if abs(order.get('quantity', 0)) > max_position:
            logger.warning(f"Order exceeds max position size: {order}")
            return False
        
        if self.pnl < -max_loss:
            logger.warning(f"Daily loss limit reached: {self.pnl}")
            return False
        
        return True
    
    def update_positions(self, fills: List[Dict[str, Any]]) -> None:
        """Update positions based on filled orders"""
        for fill in fills:
            symbol = fill['symbol']
            quantity = fill['quantity']
            price = fill['price']
            
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
            
            current_pos = self.positions[symbol]
            new_quantity = current_pos['quantity'] + quantity
            
            if new_quantity == 0:
                self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
            else:
                # Update average price
                total_cost = (current_pos['quantity'] * current_pos['avg_price'] + 
                             quantity * price)
                self.positions[symbol] = {
                    'quantity': new_quantity,
                    'avg_price': total_cost / new_quantity
                }
        
        logger.info(f"Updated positions for {self.name}: {self.positions}")


class MultiTimeframeAgent(TradingAgent):
    """Agent that operates on multiple timeframes"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.timeframes = config.get('timeframes', ['5m', '30m', '1h'])
        self.data_buffers = {tf: [] for tf in self.timeframes}
        self.indicators = {tf: {} for tf in self.timeframes}
    
    def aggregate_signals(self) -> Dict[str, float]:
        """Aggregate signals from multiple timeframes"""
        # Implement signal aggregation logic
        signals = {}
        for tf in self.timeframes:
            tf_indicators = self.indicators.get(tf, {})
            # Process indicators for each timeframe
            # This is a placeholder - implement actual logic
            signals[tf] = 0.0
        
        return signals