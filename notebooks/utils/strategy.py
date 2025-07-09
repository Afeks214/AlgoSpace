"""
Advanced Strategy Implementation for AlgoSpace
Includes proper exit logic, risk management, and position sizing
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    VOLATILITY_BASED = "volatility_based"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"


@dataclass
class TradeSignal:
    """Trade signal information"""
    timestamp: pd.Timestamp
    direction: int  # 1 for long, -1 for short
    size: float
    stop_loss: float
    take_profit: float
    confidence: float
    synergy_type: int


class RiskManager:
    """Risk management component"""
    
    def __init__(self, config: Dict):
        self.stop_loss = config.get('stop_loss', 0.02)
        self.take_profit = config.get('take_profit', 0.04)
        self.trailing_stop = config.get('trailing_stop', 0.015)
        self.use_trailing = config.get('use_trailing_stop', False)
        self.max_positions = config.get('max_positions', 1)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.daily_loss = 0.0
        self.daily_pnl = []
        
    def calculate_stop_loss(self, entry_price: float, direction: int, 
                          volatility: float = None) -> float:
        """Calculate stop loss price"""
        if volatility and volatility > 0:
            # Volatility-adjusted stop loss
            stop_distance = min(self.stop_loss, 2 * volatility)
        else:
            stop_distance = self.stop_loss
            
        if direction > 0:  # Long position
            return entry_price * (1 - stop_distance)
        else:  # Short position
            return entry_price * (1 + stop_distance)
    
    def calculate_take_profit(self, entry_price: float, direction: int,
                            volatility: float = None) -> float:
        """Calculate take profit price"""
        if volatility and volatility > 0:
            # Volatility-adjusted take profit
            profit_distance = max(self.take_profit, 3 * volatility)
        else:
            profit_distance = self.take_profit
            
        if direction > 0:  # Long position
            return entry_price * (1 + profit_distance)
        else:  # Short position
            return entry_price * (1 - profit_distance)
    
    def update_trailing_stop(self, position_price: float, current_price: float,
                           direction: int, current_stop: float) -> float:
        """Update trailing stop loss"""
        if not self.use_trailing:
            return current_stop
            
        if direction > 0:  # Long position
            new_stop = current_price * (1 - self.trailing_stop)
            return max(current_stop, new_stop)
        else:  # Short position
            new_stop = current_price * (1 + self.trailing_stop)
            return min(current_stop, new_stop)
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        return abs(self.daily_loss) >= self.max_daily_loss
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl.append(pnl)
        self.daily_loss = min(0, sum(self.daily_pnl))


class PositionSizer:
    """Position sizing component"""
    
    def __init__(self, config: Dict):
        self.method = PositionSizingMethod(config.get('method', 'fixed'))
        self.fixed_size = config.get('fixed_size', 100)
        self.volatility_lookback = config.get('volatility_lookback', 20)
        self.kelly_confidence = config.get('kelly_confidence', 0.25)
        self.max_position_size = config.get('max_position_size', 1000)
        
    def calculate_position_size(self, capital: float, price: float,
                              volatility: float = None, win_rate: float = None,
                              avg_win: float = None, avg_loss: float = None,
                              confidence: float = 1.0) -> float:
        """Calculate position size based on method"""
        if self.method == PositionSizingMethod.FIXED:
            size = self.fixed_size
            
        elif self.method == PositionSizingMethod.VOLATILITY_BASED:
            if volatility and volatility > 0:
                # Target 1% portfolio volatility
                target_vol = 0.01
                size = (capital * target_vol) / (price * volatility)
            else:
                size = self.fixed_size
                
        elif self.method == PositionSizingMethod.KELLY:
            if win_rate and avg_win and avg_loss and avg_loss > 0:
                # Kelly criterion with confidence adjustment
                b = avg_win / avg_loss
                p = win_rate
                q = 1 - p
                
                kelly_fraction = (b * p - q) / b
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                kelly_fraction *= self.kelly_confidence * confidence
                
                size = (capital * kelly_fraction) / price
            else:
                size = self.fixed_size
                
        else:
            size = self.fixed_size
        
        # Apply maximum position size limit
        return min(size, self.max_position_size)


class AlgoSpaceStrategy:
    """Main strategy implementation with all improvements"""
    
    def __init__(self, data: pd.DataFrame, synergy_type: int, config: Dict):
        self.data = data
        self.synergy_type = synergy_type
        self.config = config
        
        # Initialize components
        self.risk_manager = RiskManager(config.get('risk_management', {}))
        self.position_sizer = PositionSizer(config.get('position_sizing', {}))
        
        # Strategy parameters
        self.exit_config = config.get('exit_rules', {})
        self.use_stop_loss = self.exit_config.get('use_stop_loss', True)
        self.use_take_profit = self.exit_config.get('use_take_profit', True)
        self.use_time_exit = self.exit_config.get('use_time_exit', True)
        self.max_holding_periods = self.exit_config.get('max_holding_periods', 100)
        
        # Performance tracking
        self.trade_history: List[TradeSignal] = []
        self.win_rate = 0.5  # Initial estimate
        self.avg_win = 0.02
        self.avg_loss = 0.01
        
    def generate_entry_signals(self) -> Tuple[pd.Series, pd.Series]:
        """Generate entry signals based on synergy type"""
        long_col = f'syn{self.synergy_type}_long'
        short_col = f'syn{self.synergy_type}_short'
        long_strength_col = f'syn{self.synergy_type}_long_strength'
        short_strength_col = f'syn{self.synergy_type}_short_strength'
        
        # Get base signals
        long_entries = self.data[long_col].fillna(False)
        short_entries = self.data[short_col].fillna(False)
        
        # Filter by strength if available
        if long_strength_col in self.data.columns:
            min_strength = self.config.get('min_signal_strength', 0.3)
            long_entries = long_entries & (self.data[long_strength_col] >= min_strength)
            short_entries = short_entries & (self.data[short_strength_col] >= min_strength)
        
        # Apply additional filters
        long_entries = self._apply_entry_filters(long_entries, 'long')
        short_entries = self._apply_entry_filters(short_entries, 'short')
        
        return long_entries, short_entries
    
    def _apply_entry_filters(self, signals: pd.Series, direction: str) -> pd.Series:
        """Apply additional entry filters"""
        filtered = signals.copy()
        
        # Volume filter
        if 'Volume' in self.data.columns and 'volume_sma_20' in self.data.columns:
            volume_filter = self.data['Volume'] > self.data['volume_sma_20'] * 1.2
            filtered = filtered & volume_filter
        
        # Trend filter for longs
        if direction == 'long' and 'Close' in self.data.columns:
            sma_50 = self.data['Close'].rolling(50).mean()
            trend_filter = self.data['Close'] > sma_50
            filtered = filtered & trend_filter.fillna(False)
        
        # Trend filter for shorts
        if direction == 'short' and 'Close' in self.data.columns:
            sma_50 = self.data['Close'].rolling(50).mean()
            trend_filter = self.data['Close'] < sma_50
            filtered = filtered & trend_filter.fillna(False)
        
        return filtered
    
    def generate_exit_signals(self, entries: pd.Series, direction: int) -> pd.Series:
        """Generate exit signals with multiple exit rules"""
        exits = pd.Series(False, index=self.data.index)
        close_prices = self.data['Close'].values
        high_prices = self.data['High'].values
        low_prices = self.data['Low'].values
        
        # Track active positions
        position_active = False
        entry_price = 0.0
        entry_idx = 0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(len(self.data)):
            if entries.iloc[i] and not position_active:
                # New position
                position_active = True
                entry_price = close_prices[i]
                entry_idx = i
                
                # Calculate exit levels
                volatility = self.data['volatility_20'].iloc[i] if 'volatility_20' in self.data else None
                stop_loss = self.risk_manager.calculate_stop_loss(
                    entry_price, direction, volatility)
                take_profit = self.risk_manager.calculate_take_profit(
                    entry_price, direction, volatility)
            
            elif position_active:
                exit_triggered = False
                
                # Check stop loss
                if self.use_stop_loss:
                    if direction > 0 and low_prices[i] <= stop_loss:
                        exit_triggered = True
                    elif direction < 0 and high_prices[i] >= stop_loss:
                        exit_triggered = True
                
                # Check take profit
                if self.use_take_profit and not exit_triggered:
                    if direction > 0 and high_prices[i] >= take_profit:
                        exit_triggered = True
                    elif direction < 0 and low_prices[i] <= take_profit:
                        exit_triggered = True
                
                # Check time exit
                if self.use_time_exit and not exit_triggered:
                    if i - entry_idx >= self.max_holding_periods:
                        exit_triggered = True
                
                # Update trailing stop
                if self.risk_manager.use_trailing and not exit_triggered:
                    stop_loss = self.risk_manager.update_trailing_stop(
                        entry_price, close_prices[i], direction, stop_loss)
                
                # Check for opposite signal
                if not exit_triggered:
                    if direction > 0 and self.data[f'syn{self.synergy_type}_short'].iloc[i]:
                        exit_triggered = True
                    elif direction < 0 and self.data[f'syn{self.synergy_type}_long'].iloc[i]:
                        exit_triggered = True
                
                if exit_triggered:
                    exits.iloc[i] = True
                    position_active = False
        
        return exits
    
    def calculate_position_sizes(self, entries: pd.Series, capital: float) -> pd.Series:
        """Calculate position sizes for each entry"""
        sizes = pd.Series(0.0, index=self.data.index)
        
        for i in range(len(entries)):
            if entries.iloc[i]:
                price = self.data['Close'].iloc[i]
                volatility = self.data.get('volatility_20', pd.Series(0.02)).iloc[i]
                
                # Get signal confidence
                if self.synergy_type <= 4:
                    strength_col = f'syn{self.synergy_type}_long_strength'
                    if strength_col in self.data.columns:
                        confidence = self.data[strength_col].iloc[i]
                    else:
                        confidence = 1.0
                else:
                    confidence = 1.0
                
                size = self.position_sizer.calculate_position_size(
                    capital, price, volatility, 
                    self.win_rate, self.avg_win, self.avg_loss,
                    confidence
                )
                
                sizes.iloc[i] = size
        
        return sizes
    
    def backtest(self, init_cash: float = 100000) -> Optional[vbt.Portfolio]:
        """Run comprehensive backtest with all features"""
        logger.info(f"Running backtest for Synergy Type {self.synergy_type}")
        
        # Generate entry signals
        long_entries, short_entries = self.generate_entry_signals()
        
        # Generate exit signals
        long_exits = self.generate_exit_signals(long_entries, 1)
        short_exits = self.generate_exit_signals(short_entries, -1)
        
        # Combine signals
        entries = long_entries | short_entries
        exits = long_exits | short_exits
        
        # Direction array (1 for long, -1 for short)
        direction = np.where(long_entries, 1, np.where(short_entries, -1, 0))
        
        # Calculate position sizes
        sizes = self.calculate_position_sizes(entries, init_cash)
        
        logger.info(f"Entries: {entries.sum()}, Exits: {exits.sum()}")
        
        if entries.sum() == 0:
            logger.warning("No entry signals generated")
            return None
        
        try:
            # Create portfolio with advanced features
            portfolio = vbt.Portfolio.from_signals(
                close=self.data['Close'],
                entries=entries,
                exits=exits,
                direction=direction,
                size=sizes,
                size_type='shares',
                init_cash=init_cash,
                fees=self.config.get('backtesting', {}).get('commission', 0.0001),
                slippage=self.config.get('backtesting', {}).get('slippage', 0.0001),
                freq=self.config.get('backtesting', {}).get('frequency', '5T')
            )
            
            # Update performance metrics for adaptive sizing
            self._update_performance_metrics(portfolio)
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return None
    
    def _update_performance_metrics(self, portfolio: vbt.Portfolio):
        """Update performance metrics for adaptive strategies"""
        try:
            stats = portfolio.stats()
            self.win_rate = stats.get('Win Rate [%]', 50) / 100
            
            # Calculate average win/loss
            trades = portfolio.trades.records_readable
            if len(trades) > 0:
                winning_trades = trades[trades['PnL'] > 0]['PnL']
                losing_trades = trades[trades['PnL'] < 0]['PnL']
                
                if len(winning_trades) > 0:
                    self.avg_win = winning_trades.mean() / portfolio.init_cash
                if len(losing_trades) > 0:
                    self.avg_loss = abs(losing_trades.mean()) / portfolio.init_cash
        except:
            pass  # Keep default values if calculation fails
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """Analyze trades by various metrics"""
        if not self.trade_history:
            return pd.DataFrame()
        
        # Convert trade history to DataFrame
        trades_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'direction': 'Long' if t.direction > 0 else 'Short',
                'size': t.size,
                'confidence': t.confidence,
                'synergy_type': t.synergy_type
            }
            for t in self.trade_history
        ])
        
        return trades_df