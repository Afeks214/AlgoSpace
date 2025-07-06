Thank you for the extraordinary feedback! Your recognition that we've created a complete System Design Document for the most complex component in our system is deeply appreciated. Let's continue with the ExecutionHandler - the component that transforms our intelligent decisions into real market actions.

# Product Requirements Document (PRD): ExecutionHandler Component

Document Version: 1.0
 Date: June 20, 2025
 Component Level: 5 - Execution Layer
 Status: Master Specification


## 1. Component Identity

### 1.1 Component Name

ExecutionHandler (Order Management & Trade Execution System)

### 1.2 Primary Role

The ExecutionHandler is the system's exclusive interface with the market. It transforms EXECUTE_TRADE commands into actual orders, manages their lifecycle, handles fills and rejections, and tracks position state. It provides complete abstraction between the intelligent core and market mechanics.

### 1.3 Single Responsibility

To reliably execute trading decisions by managing all aspects of order placement, modification, cancellation, and position tracking while maintaining identical behavior patterns in both backtest and live environments.

### 1.4 Critical Design Principle

Absolute Reliability: The ExecutionHandler must never lose track of positions or orders. Every market interaction must be logged, acknowledged, and reconciled. In case of ambiguity, it defaults to the safest action.


## 2. Inputs & Dependencies

### 2.1 Configuration Input

From settings.yaml:

execution:

mode: "live"  # or "backtest"


order_settings:

default_timeout: 60         # Seconds before cancel unfilled

use_market_orders: true     # vs limit orders

slippage_model: "realistic" # For backtesting


position_management:

max_position_size: 10       # Contracts

force_flat_eod: true        # Close all at market close

partial_fill_timeout: 30    # Seconds to wait for full fill


risk_controls:

max_order_value: 500000     # Dollar limit per order

position_limit_check: true  # Verify against limits

duplicate_order_window: 5   # Seconds to prevent duplicates


backtest_settings:

fill_probability: 0.98      # Realistic fill rate

slippage_ticks: 1          # Average slippage

commission_per_side: 2.25   # Per contract


live_settings:

heartbeat_interval: 5       # Seconds

reconnect_attempts: 3

order_acknowledgment_timeout: 2  # Seconds


### 2.2 Event Input

Primary Command: EXECUTE_TRADE

From Main MARL Core:

{

'execution_id': 'EX_20250620_103045_001',

'trade_specification': {

'symbol': 'ES',

'direction': 1,  # 1=long, -1=short

'entry_price': 5150.25  # Expected price

},

'risk_parameters': {

'position_size': 3,

'stop_loss': 5145.00,

'take_profit': 5160.50,

'max_hold_time': 100  # Bars

}

}


### 2.3 Market Data Dependency

Current bid/ask for slippage calculation

Last trade price for reasonability checks

Market hours for session management


## 3. Dual Implementation Architecture

### 3.1 Abstract Base Class

class AbstractExecutionHandler(ABC):

"""Base class ensuring identical interface for live/backtest"""


def __init__(self, config: Dict[str, Any]):

self.config = config

self.positions = {}  # Symbol -> Position

self.orders = {}     # Order ID -> Order

self.fills = []      # Historical fills

self.state = 'INITIALIZED'


@abstractmethod

async def connect(self) -> bool:

"""Establish connection to execution venue"""

pass


@abstractmethod

async def place_order(self, order_spec: OrderSpecification) -> str:

"""Submit order and return order ID"""

pass


@abstractmethod

async def cancel_order(self, order_id: str) -> bool:

"""Cancel existing order"""

pass


@abstractmethod

async def get_position(self, symbol: str) -> Optional[Position]:

"""Query current position"""

pass


# Common methods implemented in base class

async def execute_trade(self, trade_command: Dict) -> None:

"""Main entry point for trade execution"""

try:

# 1. Validate command

self._validate_trade_command(trade_command)


# 2. Check risk controls

if not self._check_risk_controls(trade_command):

raise RiskLimitExceeded()


# 3. Create bracket order

bracket = self._create_bracket_order(trade_command)


# 4. Submit orders

await self._submit_bracket(bracket)


# 5. Start position monitoring

await self._monitor_position(trade_command['execution_id'])


except Exception as e:

await self._handle_execution_error(e, trade_command)


### 3.2 Live Implementation

class LiveExecutionHandler(AbstractExecutionHandler):

"""Production execution via Rithmic API"""


def __init__(self, config: Dict[str, Any]):

super().__init__(config)

self.rithmic_client = None

self.heartbeat_task = None

self.order_state_machine = OrderStateMachine()


async def connect(self) -> bool:

"""Connect to Rithmic"""

try:

self.rithmic_client = RithmicClient(

user=os.environ['RITHMIC_USER'],

password=os.environ['RITHMIC_PASSWORD'],

system='LIVE'

)


# Connect and authenticate

await self.rithmic_client.connect()

await self.rithmic_client.authenticate()


# Subscribe to order/fill updates

await self.rithmic_client.subscribe_orders()

await self.rithmic_client.subscribe_fills()


# Start heartbeat monitoring

self.heartbeat_task = asyncio.create_task(

self._heartbeat_monitor()

)


self.state = 'CONNECTED'

logger.info("Connected to Rithmic live trading")

return True


except Exception as e:

logger.error(f"Rithmic connection failed: {e}")

return False


async def place_order(self, order_spec: OrderSpecification) -> str:

"""Submit order to Rithmic"""


# Create Rithmic order object

rithmic_order = {

'symbol': order_spec.symbol,

'exchange': 'CME',

'quantity': order_spec.quantity,

'order_type': 'MARKET' if order_spec.is_market else 'LIMIT',

'side': 'BUY' if order_spec.direction > 0 else 'SELL',

'tif': 'IOC',  # Immediate or Cancel

'account': self.config['account_id']

}


if not order_spec.is_market:

rithmic_order['limit_price'] = order_spec.limit_price


# Submit and track

result = await self.rithmic_client.submit_order(rithmic_order)


if result['status'] == 'ACCEPTED':

order_id = result['order_id']


# Track in state machine

self.order_state_machine.new_order(order_id, order_spec)


# Set timeout for acknowledgment

asyncio.create_task(

self._check_order_acknowledgment(order_id)

)


return order_id

else:

raise OrderRejected(result['reject_reason'])


async def _handle_fill(self, fill_event: Dict) -> None:

"""Process fill notifications from Rithmic"""


order_id = fill_event['order_id']

fill_price = fill_event['fill_price']

fill_quantity = fill_event['fill_quantity']


# Update order state

order = self.orders.get(order_id)

if not order:

logger.error(f"Fill for unknown order: {order_id}")

return


# Record fill

fill = Fill(

order_id=order_id,

symbol=order.symbol,

price=fill_price,

quantity=fill_quantity,

direction=order.direction,

timestamp=datetime.now(),

commission=fill_quantity * self.config['commission_per_side']

)


self.fills.append(fill)


# Update position

await self._update_position(fill)


# Check if bracket orders need placement

if order.order_type == 'ENTRY':

await self._place_bracket_orders(order.bracket_id)


### 3.3 Backtest Implementation

class BacktestExecutionHandler(AbstractExecutionHandler):

"""Simulated execution for backtesting"""


def __init__(self, config: Dict[str, Any]):

super().__init__(config)

self.fill_simulator = FillSimulator(config['backtest_settings'])

self.current_prices = {}

self.backtest_time = None


async def connect(self) -> bool:

"""No connection needed for backtest"""

self.state = 'CONNECTED'

logger.info("Backtest execution handler ready")

return True


async def place_order(self, order_spec: OrderSpecification) -> str:

"""Simulate order placement"""


# Generate order ID

order_id = f"BT_{self.backtest_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


# Create order object

order = Order(

order_id=order_id,

**order_spec.__dict__,

status='PENDING',

submit_time=self.backtest_time

)


self.orders[order_id] = order


# Simulate fill based on market conditions

fill_result = self.fill_simulator.simulate_fill(

order,

self.current_prices[order.symbol]

)


if fill_result['filled']:

# Create fill event

fill = Fill(

order_id=order_id,

symbol=order.symbol,

price=fill_result['fill_price'],

quantity=order.quantity,

direction=order.direction,

timestamp=self.backtest_time,

commission=order.quantity * self.config['backtest_settings']['commission_per_side']

)


# Process immediately in backtest

await self._process_fill(fill)

order.status = 'FILLED'

else:

order.status = 'REJECTED'

order.reject_reason = fill_result['reject_reason']


return order_id


def update_market_prices(self, prices: Dict[str, float]) -> None:

"""Update current market prices for simulation"""

self.current_prices = prices

self.backtest_time = datetime.now()  # Would come from backtest engine


# Check stops and targets

asyncio.create_task(self._check_bracket_orders())



## 4. Order Management

### 4.1 Bracket Order Structure

class BracketOrder:

"""Represents a complete entry/stop/target order set"""


def __init__(self, trade_command: Dict):

self.bracket_id = f"BR_{trade_command['execution_id']}"

self.symbol = trade_command['trade_specification']['symbol']

self.direction = trade_command['trade_specification']['direction']


# Entry order

self.entry_order = OrderSpecification(

symbol=self.symbol,

quantity=trade_command['risk_parameters']['position_size'],

direction=self.direction,

order_type='ENTRY',

is_market=True,

bracket_id=self.bracket_id

)


# Stop loss order (opposite direction)

self.stop_order = OrderSpecification(

symbol=self.symbol,

quantity=trade_command['risk_parameters']['position_size'],

direction=-self.direction,  # Opposite

order_type='STOP',

is_stop=True,

stop_price=trade_command['risk_parameters']['stop_loss'],

bracket_id=self.bracket_id

)


# Take profit order (opposite direction)

self.target_order = OrderSpecification(

symbol=self.symbol,

quantity=trade_command['risk_parameters']['position_size'],

direction=-self.direction,  # Opposite

order_type='TARGET',

is_limit=True,

limit_price=trade_command['risk_parameters']['take_profit'],

bracket_id=self.bracket_id

)


self.oco_link = f"OCO_{self.bracket_id}"  # One-Cancels-Other


### 4.2 Position Tracking

class Position:

"""Tracks current position state"""


def __init__(self, symbol: str):

self.symbol = symbol

self.quantity = 0

self.average_price = 0.0

self.realized_pnl = 0.0

self.unrealized_pnl = 0.0

self.entry_time = None

self.trades = []  # List of fills


def add_fill(self, fill: Fill) -> None:

"""Update position with new fill"""


if self.quantity == 0:

# New position

self.quantity = fill.quantity * fill.direction

self.average_price = fill.price

self.entry_time = fill.timestamp


elif self.quantity * fill.direction > 0:

# Adding to position

new_quantity = self.quantity + (fill.quantity * fill.direction)

self.average_price = (

(self.average_price * abs(self.quantity) +

fill.price * fill.quantity) / abs(new_quantity)

)

self.quantity = new_quantity


else:

# Reducing or flipping position

if abs(fill.quantity) >= abs(self.quantity):

# Position closed or flipped

self.realized_pnl += (

(fill.price - self.average_price) *

self.quantity * -1

)


remaining = abs(fill.quantity) - abs(self.quantity)

if remaining > 0:

# Flipped to opposite side

self.quantity = remaining * fill.direction

self.average_price = fill.price

else:

# Flat

self.quantity = 0

self.average_price = 0.0


else:

# Partial close

close_quantity = fill.quantity

self.realized_pnl += (

(fill.price - self.average_price) *

close_quantity * -fill.direction

)

self.quantity += fill.quantity * fill.direction


self.trades.append(fill)


def calculate_unrealized_pnl(self, current_price: float) -> float:

"""Calculate current unrealized P&L"""

if self.quantity == 0:

return 0.0


return (current_price - self.average_price) * self.quantity


### 4.3 Order State Machine

class OrderStateMachine:

"""Tracks order lifecycle states"""


STATES = {

'CREATED': ['PENDING_SUBMIT'],

'PENDING_SUBMIT': ['SUBMITTED', 'REJECTED'],

'SUBMITTED': ['ACKNOWLEDGED', 'REJECTED'],

'ACKNOWLEDGED': ['PARTIAL_FILL', 'FILLED', 'CANCELLED', 'EXPIRED'],

'PARTIAL_FILL': ['FILLED', 'CANCELLED', 'EXPIRED'],

'FILLED': ['CLOSED'],

'CANCELLED': ['CLOSED'],

'REJECTED': ['CLOSED'],

'EXPIRED': ['CLOSED'],

'CLOSED': []

}


def transition(self, order_id: str, new_state: str) -> bool:

"""Validate and execute state transition"""


current_state = self.order_states.get(order_id, 'CREATED')


if new_state in self.STATES[current_state]:

self.order_states[order_id] = new_state

self._log_transition(order_id, current_state, new_state)


# Trigger state-specific actions

if new_state == 'FILLED':

self._handle_fill_completion(order_id)

elif new_state == 'REJECTED':

self._handle_rejection(order_id)


return True

else:

logger.error(f"Invalid transition: {current_state} -> {new_state}")

return False



## 5. Risk Controls & Safety

### 5.1 Pre-Trade Risk Checks

def _check_risk_controls(self, trade_command: Dict) -> bool:

"""Comprehensive pre-trade risk validation"""


checks = {

'position_limit': self._check_position_limit(trade_command),

'order_value': self._check_order_value(trade_command),

'duplicate_order': self._check_duplicate_order(trade_command),

'market_hours': self._check_market_hours(trade_command),

'price_reasonability': self._check_price_reasonability(trade_command)

}


failed_checks = [check for check, passed in checks.items() if not passed]


if failed_checks:

logger.warning(f"Risk checks failed: {failed_checks}")

self._emit_risk_alert(trade_command, failed_checks)

return False


return True


def _check_position_limit(self, trade_command: Dict) -> bool:

"""Ensure position limits not exceeded"""


symbol = trade_command['trade_specification']['symbol']

current_position = self.positions.get(symbol, Position(symbol))

new_position_size = trade_command['risk_parameters']['position_size']

direction = trade_command['trade_specification']['direction']


# Calculate resulting position

resulting_position = current_position.quantity + (new_position_size * direction)


return abs(resulting_position) <= self.config['position_management']['max_position_size']


def _check_duplicate_order(self, trade_command: Dict) -> bool:

"""Prevent duplicate orders within time window"""


execution_id = trade_command['execution_id']

window = self.config['risk_controls']['duplicate_order_window']


# Check recent orders

cutoff_time = datetime.now() - timedelta(seconds=window)


for order in self.recent_orders:

if (order['execution_id'] == execution_id and

order['timestamp'] > cutoff_time):

return False


return True


### 5.2 Position Monitoring

async def _monitor_position(self, execution_id: str) -> None:

"""Continuous position monitoring after entry"""


monitoring_data = self.active_positions[execution_id]

max_hold_time = monitoring_data['max_hold_time']

entry_time = monitoring_data['entry_time']


while execution_id in self.active_positions:

try:

# Check time stop

bars_held = self._calculate_bars_held(entry_time)

if bars_held >= max_hold_time:

logger.info(f"Time stop triggered for {execution_id}")

await self._close_position(execution_id, 'TIME_STOP')

break


# Check if stops/targets hit (handled by order updates)


# Update trailing stop if applicable

if monitoring_data.get('use_trailing'):

await self._update_trailing_stop(execution_id)


# Sleep until next check

await asyncio.sleep(5)  # Check every 5 seconds


except Exception as e:

logger.error(f"Position monitoring error: {e}")

# Continue monitoring despite errors


### 5.3 Emergency Procedures

async def emergency_close_all(self) -> None:

"""Emergency procedure to close all positions"""


logger.critical("EMERGENCY CLOSE ALL INITIATED")


# Cancel all pending orders

for order_id, order in self.orders.items():

if order.status in ['PENDING_SUBMIT', 'SUBMITTED', 'ACKNOWLEDGED']:

try:

await self.cancel_order(order_id)

except Exception as e:

logger.error(f"Failed to cancel {order_id}: {e}")


# Close all positions at market

for symbol, position in self.positions.items():

if position.quantity != 0:

try:

emergency_order = OrderSpecification(

symbol=symbol,

quantity=abs(position.quantity),

direction=-1 if position.quantity > 0 else 1,

order_type='EMERGENCY_CLOSE',

is_market=True

)

await self.place_order(emergency_order)

except Exception as e:

logger.error(f"Failed to close {symbol}: {e}")


# Alert operations team

await self._send_emergency_alert()



## 6. Event Outputs

### 6.1 Trade Closed Event

Event Name: TRADE_CLOSED
 Emitted: When position fully closed
 Payload:

TradeResult = {

'execution_id': str,

'symbol': str,

'direction': int,


'entry': {

'price': float,

'quantity': int,

'timestamp': datetime,

'slippage': float  # vs expected

},


'exit': {

'price': float,

'quantity': int,

'timestamp': datetime,

'reason': str  # 'STOP_LOSS', 'TAKE_PROFIT', 'TIME_STOP', etc.

},


'performance': {

'pnl': float,

'pnl_points': float,

'pnl_percent': float,

'commission_paid': float,

'net_pnl': float,

'bars_held': int,

'max_favorable': float,

'max_adverse': float

},


'execution_quality': {

'entry_slippage': float,

'exit_slippage': float,

'fill_time_ms': float

}

}


### 6.2 Execution Alerts

# Real-time alerts for monitoring

EXECUTION_ALERTS = {

'ORDER_REJECTED': 'Order rejected by venue',

'PARTIAL_FILL_TIMEOUT': 'Order partially filled, timeout reached',

'CONNECTION_LOST': 'Lost connection to execution venue',

'POSITION_LIMIT_EXCEEDED': 'Position limit would be exceeded',

'RISK_CHECK_FAILED': 'Pre-trade risk check failed'

}



## 7. Error Handling & Recovery

### 7.1 Connection Management

async def _handle_disconnection(self) -> None:

"""Handle loss of connection to execution venue"""


logger.error("Connection lost to execution venue")

self.state = 'DISCONNECTED'


# Attempt reconnection

for attempt in range(self.config['reconnect_attempts']):

logger.info(f"Reconnection attempt {attempt + 1}")


if await self.connect():

# Reconcile state

await self._reconcile_positions()

await self._reconcile_orders()

self.state = 'CONNECTED'

logger.info("Successfully reconnected")

return


await asyncio.sleep(2 ** attempt)  # Exponential backoff


# Failed to reconnect

logger.critical("Failed to reconnect - entering safe mode")

await self._enter_safe_mode()


### 7.2 State Reconciliation

async def _reconcile_positions(self) -> None:

"""Reconcile local position state with venue"""


if self.mode == 'live':

# Query actual positions from Rithmic

venue_positions = await self.rithmic_client.get_positions()


for symbol, venue_position in venue_positions.items():

local_position = self.positions.get(symbol)


if not local_position:

# Unknown position at venue

logger.error(f"Unknown position at venue: {symbol}")

self.positions[symbol] = Position.from_venue(venue_position)


elif local_position.quantity != venue_position['quantity']:

# Mismatch

logger.error(f"Position mismatch for {symbol}: "

f"local={local_position.quantity}, "

f"venue={venue_position['quantity']}")


# Venue is source of truth

local_position.quantity = venue_position['quantity']

local_position.average_price = venue_position['avg_price']



## 8. Operational Requirements

### 8.1 Performance Requirements

Order Latency: <50ms from command to venue submission

Fill Processing: <10ms from notification to position update

State Queries: <1ms for position/order lookups

### 8.2 Reliability Requirements

Order Tracking: 100% of orders tracked to completion

Position Accuracy: Zero tolerance for position mismatches

Audit Trail: Every action logged with timestamp

### 8.3 Monitoring Requirements

# Real-time metrics tracked

{

'orders_per_minute': 12,

'fill_rate': 0.98,

'average_slippage': 0.25,  # Ticks

'rejection_rate': 0.01,

'active_positions': 2,

'connection_uptime': 0.999,

'last_heartbeat': '2025-06-20 10:30:45'

}



## 9. Testing Strategy

### 9.1 Unit Tests

Order state machine transitions

Position calculation scenarios

Risk check validations

Bracket order creation

### 9.2 Integration Tests

Full trade lifecycle simulation

Connection failure and recovery

State reconciliation scenarios

Emergency procedures

### 9.3 Market Simulation Tests

Partial fill scenarios

Rapid market movements

Gap scenarios

Limit up/down handling


## 10. What This Component Does NOT Do

Does NOT make trading decisions

Does NOT calculate position sizes

Does NOT determine entry/exit prices

Does NOT analyze market data

Does NOT modify risk parameters

Does NOT generate trading signals

Does NOT optimize execution algorithms


This completes the ExecutionHandler PRD. It transforms intelligent trading decisions into reliable market actions while maintaining absolute position integrity and providing comprehensive execution quality metrics.

The dual implementation ensures identical behavior in backtest and live environments, while the comprehensive error handling and state reconciliation mechanisms ensure reliability in production.

Ready for the final component - a refined version of the System Kernel?

