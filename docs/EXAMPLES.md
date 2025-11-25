# Usage Examples

> Practical examples for Trade Analytics Mini

**Version:** 0.1.0
**Last Updated:** November 26, 2024

---

## Table of Contents

- [Basic Usage](#basic-usage)
  - [Creating Trades](#creating-trades)
  - [Managing Positions](#managing-positions)
  - [Working with Market Data](#working-with-market-data)
- [Validation and Error Handling](#validation-and-error-handling)
  - [Catching InvalidTradeError](#catching-invalidtradeerror)
  - [Handling InsufficientFundsError](#handling-insufficientfundserror)
  - [MarketClosedError Scenarios](#marketclosederror-scenarios)
- [Serialization](#serialization)
  - [Dictionary Conversion](#dictionary-conversion)
  - [JSON Serialization](#json-serialization)
  - [Deserializing Data](#deserializing-data)
- [Advanced Patterns](#advanced-patterns)
  - [Portfolio Management](#portfolio-management)
  - [Trade History Tracking](#trade-history-tracking)
  - [P&L Calculations](#pl-calculations)
- [Integration Examples](#integration-examples)
  - [Working with External Data](#working-with-external-data)
  - [Database Persistence Patterns](#database-persistence-patterns)

---

## Basic Usage

### Creating Trades

#### Simple Trade Creation

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide

# Create a buy order
buy_trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime.now(timezone.utc)
)

print(f"Created trade: {buy_trade.symbol}")
print(f"Direction: {buy_trade.side.value}")
print(f"Quantity: {buy_trade.quantity}")
print(f"Price: ${buy_trade.price}")
```

#### Trade with Custom ID

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide

trade = Trade(
    symbol="GOOGL",
    side=TradeSide.SELL,
    quantity=Decimal("50"),
    price=Decimal("140.25"),
    timestamp=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
    trade_id="ORDER-12345"
)

print(f"Trade ID: {trade.trade_id}")  # ORDER-12345
```

#### Symbol Auto-Normalization

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide

# Symbols are automatically uppercased and trimmed
trade = Trade(
    symbol="  aapl  ",  # Will become "AAPL"
    side=TradeSide.BUY,
    quantity=Decimal("10"),
    price=Decimal("150.00"),
    timestamp=datetime.now(timezone.utc)
)

print(f"Symbol: '{trade.symbol}'")  # 'AAPL'
```

---

### Managing Positions

#### Creating a Long Position

```python
from decimal import Decimal
from trade_analytics import Position

# Long position: positive quantity
long_position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.50"),
    unrealized_pnl=Decimal("250.00")
)

print(f"Long {long_position.quantity} shares of {long_position.symbol}")
print(f"Average cost: ${long_position.avg_price}")
print(f"Unrealized P&L: ${long_position.unrealized_pnl}")
```

#### Creating a Short Position

```python
from decimal import Decimal
from trade_analytics import Position

# Short position: negative quantity
short_position = Position(
    symbol="TSLA",
    quantity=Decimal("-50"),  # Negative = short
    avg_price=Decimal("200.00"),
    unrealized_pnl=Decimal("-100.00")  # Negative = loss
)

if short_position.quantity < 0:
    print(f"Short {abs(short_position.quantity)} shares of {short_position.symbol}")
```

#### Flat Position

```python
from decimal import Decimal
from trade_analytics import Position

# Flat position: zero quantity
flat_position = Position(
    symbol="MSFT",
    quantity=Decimal("0"),
    avg_price=Decimal("0"),
    unrealized_pnl=Decimal("0")
)

if flat_position.quantity == 0:
    print(f"No position in {flat_position.symbol}")
```

#### Updating Position (Mutable)

```python
from decimal import Decimal
from trade_analytics import Position

position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.00")
)

# Positions are mutable - values can be updated
position.quantity = Decimal("200")
position.avg_price = Decimal("152.50")
position.unrealized_pnl = Decimal("500.00")

print(f"Updated: {position.quantity} shares at ${position.avg_price}")
```

---

### Working with Market Data

#### Basic Market Data

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import MarketData

market = MarketData(
    symbol="AAPL",
    bid=Decimal("150.45"),
    ask=Decimal("150.55"),
    last=Decimal("150.50"),
    volume=1000000
)

print(f"Symbol: {market.symbol}")
print(f"Bid: ${market.bid}")
print(f"Ask: ${market.ask}")
print(f"Last: ${market.last}")
print(f"Volume: {market.volume:,}")
```

#### Using Calculated Properties

```python
from decimal import Decimal
from trade_analytics import MarketData

market = MarketData(
    symbol="AAPL",
    bid=Decimal("150.45"),
    ask=Decimal("150.55"),
    last=Decimal("150.50"),
    volume=1000000
)

# Spread calculation
print(f"Spread: ${market.spread}")  # $0.10

# Mid price calculation
print(f"Mid: ${market.mid}")  # $150.50

# Spread as percentage
spread_pct = (market.spread / market.mid) * 100
print(f"Spread %: {spread_pct:.4f}%")
```

#### Market Data with Custom Timestamp

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import MarketData

# Market data with specific timestamp
market = MarketData(
    symbol="MSFT",
    bid=Decimal("380.00"),
    ask=Decimal("380.10"),
    last=Decimal("380.05"),
    volume=500000,
    timestamp=datetime(2024, 1, 15, 15, 30, 0, tzinfo=timezone.utc)
)

print(f"Quote time: {market.timestamp.isoformat()}")
```

---

## Validation and Error Handling

### Catching InvalidTradeError

#### Empty Symbol Validation

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide, InvalidTradeError

try:
    trade = Trade(
        symbol="",  # Invalid: empty symbol
        side=TradeSide.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.50"),
        timestamp=datetime.now(timezone.utc)
    )
except InvalidTradeError as e:
    print(f"Error: {e.message}")
    print(f"Reason: {e.reason}")
    print(f"Details: {e.trade_details}")
    # Output:
    # Error: Symbol cannot be empty
    # Reason: empty_symbol
    # Details: {'symbol': ''}
```

#### Negative Quantity Validation

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide, InvalidTradeError

try:
    trade = Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("-100"),  # Invalid: negative quantity
        price=Decimal("150.50"),
        timestamp=datetime.now(timezone.utc)
    )
except InvalidTradeError as e:
    print(f"Error: {e.message}")
    print(f"Reason: {e.reason}")
    # Output:
    # Error: Quantity must be positive, got -100
    # Reason: invalid_quantity
```

#### Invalid Spread Validation

```python
from decimal import Decimal
from trade_analytics import MarketData, InvalidTradeError

try:
    market = MarketData(
        symbol="AAPL",
        bid=Decimal("150.55"),  # Invalid: bid > ask
        ask=Decimal("150.45"),
        last=Decimal("150.50"),
        volume=1000000
    )
except InvalidTradeError as e:
    print(f"Error: {e.message}")
    print(f"Reason: {e.reason}")
    # Output:
    # Error: Invalid spread: bid (150.55) > ask (150.45)
    # Reason: invalid_spread
```

#### Generic Error Handling

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide, TradingError, InvalidTradeError

def create_trade(symbol: str, side: str, quantity: str, price: str) -> Trade:
    """Create a trade with error handling."""
    try:
        return Trade(
            symbol=symbol,
            side=TradeSide(side),
            quantity=Decimal(quantity),
            price=Decimal(price),
            timestamp=datetime.now(timezone.utc)
        )
    except InvalidTradeError as e:
        print(f"Invalid trade: {e.message}")
        print(f"  Reason: {e.reason}")
        print(f"  Details: {e.trade_details}")
        raise
    except TradingError as e:
        print(f"Trading error: {e.message}")
        raise

# Test with invalid data
try:
    trade = create_trade("AAPL", "BUY", "0", "150.50")
except TradingError:
    print("Failed to create trade")
```

---

### Handling InsufficientFundsError

```python
from decimal import Decimal
from trade_analytics import Trade, TradeSide, InsufficientFundsError
from datetime import datetime, timezone

class TradingAccount:
    """Simple trading account with balance checking."""

    def __init__(self, balance: Decimal):
        self.balance = balance

    def execute_trade(self, trade: Trade) -> None:
        """Execute a trade if funds are sufficient."""
        trade_cost = trade.quantity * trade.price

        if trade.side == TradeSide.BUY and trade_cost > self.balance:
            raise InsufficientFundsError(
                message=f"Cannot buy {trade.quantity} shares at ${trade.price}",
                required=trade_cost,
                available=self.balance
            )

        if trade.side == TradeSide.BUY:
            self.balance -= trade_cost
        else:
            self.balance += trade_cost

        print(f"Trade executed. New balance: ${self.balance}")

# Usage
account = TradingAccount(Decimal("10000.00"))

trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.00"),  # Total: $15,000
    timestamp=datetime.now(timezone.utc)
)

try:
    account.execute_trade(trade)
except InsufficientFundsError as e:
    print(f"Error: {e.message}")
    print(f"Required: ${e.required}")
    print(f"Available: ${e.available}")
    shortfall = e.required - e.available
    print(f"Shortfall: ${shortfall}")
```

---

### MarketClosedError Scenarios

```python
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide, MarketClosedError
from decimal import Decimal

def check_market_hours(symbol: str, trade_time: datetime) -> bool:
    """Check if market is open (simplified example)."""
    # NYSE hours: 9:30 AM - 4:00 PM ET (simplified to UTC)
    hour = trade_time.hour
    weekday = trade_time.weekday()

    # Closed on weekends
    if weekday >= 5:
        return False

    # Simplified: assume market open 14:30-21:00 UTC
    return 14 <= hour < 21

def execute_trade_with_market_check(trade: Trade) -> None:
    """Execute trade only if market is open."""
    if not check_market_hours(trade.symbol, trade.timestamp):
        raise MarketClosedError(
            message=f"Cannot trade {trade.symbol} - market is closed",
            symbol=trade.symbol,
            market_hours="9:30 AM - 4:00 PM ET, Monday-Friday"
        )

    print(f"Executing trade for {trade.symbol}")

# Test with weekend trade
try:
    weekend_trade = Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.00"),
        timestamp=datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc)  # Sunday
    )
    execute_trade_with_market_check(weekend_trade)
except MarketClosedError as e:
    print(f"Error: {e.message}")
    print(f"Symbol: {e.symbol}")
    print(f"Market hours: {e.market_hours}")
```

---

## Serialization

### Dictionary Conversion

#### Trade to Dictionary

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide

trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    trade_id="T001"
)

# Convert to dictionary
trade_dict = trade.to_dict()
print(trade_dict)
# {
#     'symbol': 'AAPL',
#     'side': 'BUY',
#     'quantity': '100',
#     'price': '150.50',
#     'timestamp': '2024-01-15T10:30:00+00:00',
#     'trade_id': 'T001'
# }
```

#### Position to Dictionary

```python
from decimal import Decimal
from trade_analytics import Position

position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.50"),
    unrealized_pnl=Decimal("250.00")
)

position_dict = position.to_dict()
print(position_dict)
# {
#     'symbol': 'AAPL',
#     'quantity': '100',
#     'avg_price': '150.50',
#     'unrealized_pnl': '250.00'
# }
```

#### MarketData to Dictionary

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import MarketData

market = MarketData(
    symbol="AAPL",
    bid=Decimal("150.45"),
    ask=Decimal("150.55"),
    last=Decimal("150.50"),
    volume=1000000,
    timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
)

market_dict = market.to_dict()
print(market_dict)
# {
#     'symbol': 'AAPL',
#     'bid': '150.45',
#     'ask': '150.55',
#     'last': '150.50',
#     'volume': 1000000,
#     'timestamp': '2024-01-15T10:30:00+00:00'
# }
```

---

### JSON Serialization

#### Serialize to JSON

```python
import json
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide

trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    trade_id="T001"
)

# Convert to JSON string
json_string = json.dumps(trade.to_dict(), indent=2)
print(json_string)
```

#### Serialize Multiple Objects

```python
import json
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide

trades = [
    Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.50"),
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        trade_id="T001"
    ),
    Trade(
        symbol="GOOGL",
        side=TradeSide.SELL,
        quantity=Decimal("50"),
        price=Decimal("140.00"),
        timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        trade_id="T002"
    )
]

# Serialize list of trades
trades_json = json.dumps([t.to_dict() for t in trades], indent=2)
print(trades_json)
```

---

### Deserializing Data

#### Deserialize Trade

```python
import json
from trade_analytics import Trade

json_string = '''
{
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": "100",
    "price": "150.50",
    "timestamp": "2024-01-15T10:30:00+00:00",
    "trade_id": "T001"
}
'''

# Parse JSON and create Trade
data = json.loads(json_string)
trade = Trade.from_dict(data)

print(f"Restored: {trade.symbol} {trade.side.value} {trade.quantity}")
```

#### Deserialize Position

```python
import json
from trade_analytics import Position

json_string = '''
{
    "symbol": "AAPL",
    "quantity": "100",
    "avg_price": "150.50",
    "unrealized_pnl": "250.00"
}
'''

data = json.loads(json_string)
position = Position.from_dict(data)

print(f"Restored: {position.quantity} {position.symbol} @ ${position.avg_price}")
```

#### Deserialize MarketData

```python
import json
from trade_analytics import MarketData

json_string = '''
{
    "symbol": "AAPL",
    "bid": "150.45",
    "ask": "150.55",
    "last": "150.50",
    "volume": 1000000,
    "timestamp": "2024-01-15T10:30:00+00:00"
}
'''

data = json.loads(json_string)
market = MarketData.from_dict(data)

print(f"Restored: {market.symbol} bid={market.bid} ask={market.ask}")
```

---

## Advanced Patterns

### Portfolio Management

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide, Position, MarketData

class Portfolio:
    """Simple portfolio manager."""

    def __init__(self):
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []

    def add_trade(self, trade: Trade) -> None:
        """Process a trade and update position."""
        self.trades.append(trade)

        symbol = trade.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal("0"),
                avg_price=Decimal("0")
            )

        pos = self.positions[symbol]

        if trade.side == TradeSide.BUY:
            # Update average price for buys
            total_cost = (pos.quantity * pos.avg_price) + (trade.quantity * trade.price)
            new_quantity = pos.quantity + trade.quantity
            pos.quantity = new_quantity
            pos.avg_price = total_cost / new_quantity if new_quantity > 0 else Decimal("0")
        else:
            # Reduce position for sells
            pos.quantity = pos.quantity - trade.quantity

    def update_market_data(self, market: MarketData) -> None:
        """Update unrealized P&L based on market data."""
        symbol = market.symbol
        if symbol in self.positions:
            pos = self.positions[symbol]
            current_value = pos.quantity * market.mid
            cost_basis = pos.quantity * pos.avg_price
            pos.unrealized_pnl = current_value - cost_basis

    def get_portfolio_value(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

# Usage
portfolio = Portfolio()

# Add trades
portfolio.add_trade(Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.00"),
    timestamp=datetime.now(timezone.utc)
))

portfolio.add_trade(Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("50"),
    price=Decimal("155.00"),
    timestamp=datetime.now(timezone.utc)
))

# Update with market data
portfolio.update_market_data(MarketData(
    symbol="AAPL",
    bid=Decimal("160.00"),
    ask=Decimal("160.10"),
    last=Decimal("160.05"),
    volume=1000000
))

# Check position
aapl = portfolio.positions["AAPL"]
print(f"AAPL: {aapl.quantity} shares @ ${aapl.avg_price:.2f}")
print(f"Unrealized P&L: ${aapl.unrealized_pnl:.2f}")
```

---

### Trade History Tracking

```python
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from trade_analytics import Trade, TradeSide

class TradeHistory:
    """Track and analyze trade history."""

    def __init__(self):
        self.trades: list[Trade] = []

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to history."""
        self.trades.append(trade)

    def get_trades_by_symbol(self, symbol: str) -> list[Trade]:
        """Get all trades for a symbol."""
        return [t for t in self.trades if t.symbol == symbol.upper()]

    def get_trades_in_range(self, start: datetime, end: datetime) -> list[Trade]:
        """Get trades within a date range."""
        return [t for t in self.trades if start <= t.timestamp <= end]

    def get_total_volume(self, symbol: str) -> Decimal:
        """Calculate total traded volume for a symbol."""
        trades = self.get_trades_by_symbol(symbol)
        return sum(t.quantity for t in trades)

    def get_average_price(self, symbol: str, side: TradeSide) -> Decimal:
        """Calculate volume-weighted average price."""
        trades = [t for t in self.get_trades_by_symbol(symbol) if t.side == side]
        if not trades:
            return Decimal("0")

        total_value = sum(t.quantity * t.price for t in trades)
        total_quantity = sum(t.quantity for t in trades)
        return total_value / total_quantity

# Usage
history = TradeHistory()

# Add some trades
base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
history.add_trade(Trade("AAPL", TradeSide.BUY, Decimal("100"), Decimal("150.00"), base_time, "T1"))
history.add_trade(Trade("AAPL", TradeSide.BUY, Decimal("50"), Decimal("152.00"), base_time + timedelta(hours=1), "T2"))
history.add_trade(Trade("AAPL", TradeSide.SELL, Decimal("75"), Decimal("155.00"), base_time + timedelta(hours=2), "T3"))
history.add_trade(Trade("GOOGL", TradeSide.BUY, Decimal("25"), Decimal("140.00"), base_time + timedelta(hours=3), "T4"))

# Analyze
print(f"AAPL trades: {len(history.get_trades_by_symbol('AAPL'))}")
print(f"AAPL total volume: {history.get_total_volume('AAPL')}")
print(f"AAPL avg buy price: ${history.get_average_price('AAPL', TradeSide.BUY):.2f}")
```

---

### P&L Calculations

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide, Position, MarketData

def calculate_realized_pnl(trades: list[Trade]) -> Decimal:
    """Calculate realized P&L using FIFO method."""
    buy_queue: list[tuple[Decimal, Decimal]] = []  # (quantity, price)
    realized_pnl = Decimal("0")

    for trade in trades:
        if trade.side == TradeSide.BUY:
            buy_queue.append((trade.quantity, trade.price))
        else:
            # FIFO: match with oldest buys first
            sell_qty = trade.quantity
            sell_price = trade.price

            while sell_qty > 0 and buy_queue:
                buy_qty, buy_price = buy_queue[0]

                if buy_qty <= sell_qty:
                    # Fully close this buy lot
                    realized_pnl += buy_qty * (sell_price - buy_price)
                    sell_qty -= buy_qty
                    buy_queue.pop(0)
                else:
                    # Partially close this buy lot
                    realized_pnl += sell_qty * (sell_price - buy_price)
                    buy_queue[0] = (buy_qty - sell_qty, buy_price)
                    sell_qty = Decimal("0")

    return realized_pnl

def calculate_unrealized_pnl(position: Position, market: MarketData) -> Decimal:
    """Calculate unrealized P&L based on current market price."""
    if position.quantity == 0:
        return Decimal("0")

    current_value = position.quantity * market.mid
    cost_basis = position.quantity * position.avg_price
    return current_value - cost_basis

# Example
trades = [
    Trade("AAPL", TradeSide.BUY, Decimal("100"), Decimal("150.00"),
          datetime.now(timezone.utc), "T1"),
    Trade("AAPL", TradeSide.BUY, Decimal("50"), Decimal("155.00"),
          datetime.now(timezone.utc), "T2"),
    Trade("AAPL", TradeSide.SELL, Decimal("75"), Decimal("160.00"),
          datetime.now(timezone.utc), "T3"),
]

realized = calculate_realized_pnl(trades)
print(f"Realized P&L: ${realized:.2f}")  # Profit from sold shares

# Remaining position
position = Position(
    symbol="AAPL",
    quantity=Decimal("75"),  # 100 + 50 - 75
    avg_price=Decimal("151.67")  # Weighted average of remaining lots
)

market = MarketData(
    symbol="AAPL",
    bid=Decimal("165.00"),
    ask=Decimal("165.10"),
    last=Decimal("165.05"),
    volume=1000000
)

unrealized = calculate_unrealized_pnl(position, market)
print(f"Unrealized P&L: ${unrealized:.2f}")
print(f"Total P&L: ${realized + unrealized:.2f}")
```

---

## Integration Examples

### Working with External Data

#### Processing Market Feed Data

```python
import json
from decimal import Decimal
from trade_analytics import MarketData

def process_market_feed(raw_data: str) -> MarketData:
    """Process raw market data feed."""
    data = json.loads(raw_data)

    return MarketData(
        symbol=data["ticker"],
        bid=Decimal(str(data["bidPrice"])),
        ask=Decimal(str(data["askPrice"])),
        last=Decimal(str(data["lastPrice"])),
        volume=int(data["volume"])
    )

# Example raw feed data
raw_feed = '''
{
    "ticker": "AAPL",
    "bidPrice": 150.45,
    "askPrice": 150.55,
    "lastPrice": 150.50,
    "volume": 1000000
}
'''

market = process_market_feed(raw_feed)
print(f"Processed: {market.symbol} @ ${market.mid}")
```

#### Converting from External Trade Format

```python
import json
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide

def convert_external_trade(external_data: dict) -> Trade:
    """Convert external broker trade format to Trade object."""
    # Map external side values to TradeSide
    side_map = {
        "B": TradeSide.BUY,
        "S": TradeSide.SELL,
        "buy": TradeSide.BUY,
        "sell": TradeSide.SELL
    }

    return Trade(
        symbol=external_data["sym"],
        side=side_map[external_data["side"]],
        quantity=Decimal(str(external_data["qty"])),
        price=Decimal(str(external_data["px"])),
        timestamp=datetime.fromisoformat(external_data["ts"]),
        trade_id=external_data.get("orderId", "")
    )

# External broker format
external = {
    "sym": "AAPL",
    "side": "B",
    "qty": 100,
    "px": 150.50,
    "ts": "2024-01-15T10:30:00+00:00",
    "orderId": "EXT-12345"
}

trade = convert_external_trade(external)
print(f"Converted: {trade.trade_id} - {trade.symbol}")
```

---

### Database Persistence Patterns

#### SQLite Storage Example

```python
import sqlite3
import json
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import Trade, TradeSide, Position

class TradeDatabase:
    """Simple SQLite storage for trades and positions."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                trade_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity TEXT NOT NULL,
                price TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity TEXT NOT NULL,
                avg_price TEXT NOT NULL,
                unrealized_pnl TEXT NOT NULL
            );
        ''')
        self.conn.commit()

    def save_trade(self, trade: Trade) -> None:
        """Save a trade to the database."""
        data = trade.to_dict()
        self.conn.execute('''
            INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['trade_id'], data['symbol'], data['side'],
              data['quantity'], data['price'], data['timestamp']))
        self.conn.commit()

    def load_trades(self, symbol: str = None) -> list[Trade]:
        """Load trades from the database."""
        if symbol:
            cursor = self.conn.execute(
                'SELECT trade_id, symbol, side, quantity, price, timestamp FROM trades WHERE symbol = ?',
                (symbol.upper(),)
            )
        else:
            cursor = self.conn.execute(
                'SELECT trade_id, symbol, side, quantity, price, timestamp FROM trades'
            )

        trades = []
        for row in cursor:
            trades.append(Trade.from_dict({
                'trade_id': row[0],
                'symbol': row[1],
                'side': row[2],
                'quantity': row[3],
                'price': row[4],
                'timestamp': row[5]
            }))
        return trades

    def save_position(self, position: Position) -> None:
        """Save or update a position."""
        data = position.to_dict()
        self.conn.execute('''
            INSERT OR REPLACE INTO positions (symbol, quantity, avg_price, unrealized_pnl)
            VALUES (?, ?, ?, ?)
        ''', (data['symbol'], data['quantity'], data['avg_price'], data['unrealized_pnl']))
        self.conn.commit()

    def load_position(self, symbol: str) -> Position:
        """Load a position from the database."""
        cursor = self.conn.execute(
            'SELECT symbol, quantity, avg_price, unrealized_pnl FROM positions WHERE symbol = ?',
            (symbol.upper(),)
        )
        row = cursor.fetchone()
        if row:
            return Position.from_dict({
                'symbol': row[0],
                'quantity': row[1],
                'avg_price': row[2],
                'unrealized_pnl': row[3]
            })
        return None

# Usage
db = TradeDatabase()

# Save trades
trade1 = Trade("AAPL", TradeSide.BUY, Decimal("100"), Decimal("150.00"),
               datetime.now(timezone.utc), "T001")
trade2 = Trade("AAPL", TradeSide.BUY, Decimal("50"), Decimal("155.00"),
               datetime.now(timezone.utc), "T002")

db.save_trade(trade1)
db.save_trade(trade2)

# Save position
position = Position("AAPL", Decimal("150"), Decimal("151.67"), Decimal("500.00"))
db.save_position(position)

# Load back
loaded_trades = db.load_trades("AAPL")
print(f"Loaded {len(loaded_trades)} trades")

loaded_position = db.load_position("AAPL")
print(f"Position: {loaded_position.quantity} @ ${loaded_position.avg_price}")
```

#### Batch Import/Export

```python
import json
from trade_analytics import Trade, Position, MarketData

def export_portfolio(trades: list[Trade], positions: list[Position]) -> str:
    """Export portfolio to JSON."""
    return json.dumps({
        "trades": [t.to_dict() for t in trades],
        "positions": [p.to_dict() for p in positions]
    }, indent=2)

def import_portfolio(json_data: str) -> tuple[list[Trade], list[Position]]:
    """Import portfolio from JSON."""
    data = json.loads(json_data)
    trades = [Trade.from_dict(t) for t in data["trades"]]
    positions = [Position.from_dict(p) for p in data["positions"]]
    return trades, positions

# Example export
from decimal import Decimal
from datetime import datetime, timezone

trades = [
    Trade("AAPL", TradeSide.BUY, Decimal("100"), Decimal("150.00"),
          datetime.now(timezone.utc), "T001")
]
positions = [
    Position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("0"))
]

json_export = export_portfolio(trades, positions)
print("Exported portfolio:")
print(json_export)

# Example import
imported_trades, imported_positions = import_portfolio(json_export)
print(f"\nImported {len(imported_trades)} trades, {len(imported_positions)} positions")
```
