# Usage Examples

Practical examples demonstrating how to use the Trade Analytics library.

## Table of Contents

- [Creating Trades](#creating-trades)
- [Managing Positions](#managing-positions)
- [Working with Market Data](#working-with-market-data)
- [Portfolio Calculations](#portfolio-calculations)
- [Error Handling](#error-handling)
- [Serialization](#serialization)

---

## Creating Trades

### Basic Trade Creation

```python
from datetime import datetime, timezone
from decimal import Decimal
from trade_analytics import Trade, TradeSide

# Create a buy trade
buy_trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime.now(timezone.utc)
)

print(f"Bought {buy_trade.quantity} shares of {buy_trade.symbol} at ${buy_trade.price}")
# Output: Bought 100 shares of AAPL at $150.50

# Create a sell trade
sell_trade = Trade(
    symbol="AAPL",
    side=TradeSide.SELL,
    quantity=Decimal("50"),
    price=Decimal("155.00"),
    timestamp=datetime.now(timezone.utc)
)

print(f"Sold {sell_trade.quantity} shares of {sell_trade.symbol} at ${sell_trade.price}")
# Output: Sold 50 shares of AAPL at $155.00
```

### Trade with ID

```python
from datetime import datetime, timezone
from decimal import Decimal
from trade_analytics import Trade, TradeSide

trade = Trade(
    symbol="GOOGL",
    side=TradeSide.BUY,
    quantity=Decimal("25"),
    price=Decimal("140.00"),
    timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    trade_id="ORDER-12345"
)

print(f"Trade ID: {trade.trade_id}")
# Output: Trade ID: ORDER-12345
```

### Symbol Normalization

Symbols are automatically uppercased:

```python
from datetime import datetime, timezone
from decimal import Decimal
from trade_analytics import Trade, TradeSide

trade = Trade(
    symbol="aapl",  # lowercase
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.00"),
    timestamp=datetime.now(timezone.utc)
)

print(trade.symbol)  # "AAPL" - automatically uppercased
```

---

## Managing Positions

### Long Position

```python
from decimal import Decimal
from trade_analytics import Position

# Long 100 shares of AAPL at $150.00 average
position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.00")
)

print(f"Long {position.quantity} {position.symbol} @ ${position.avg_price}")
# Output: Long 100 AAPL @ $150.00
```

### Short Position

```python
from decimal import Decimal
from trade_analytics import Position

# Short 50 shares of TSLA at $200.00 average
short_position = Position(
    symbol="TSLA",
    quantity=Decimal("-50"),  # Negative quantity = short
    avg_price=Decimal("200.00")
)

print(f"Short {abs(short_position.quantity)} {short_position.symbol} @ ${short_position.avg_price}")
# Output: Short 50 TSLA @ $200.00
```

### Flat Position

```python
from decimal import Decimal
from trade_analytics import Position

# No position (flat)
flat_position = Position(
    symbol="MSFT",
    quantity=Decimal("0"),
    avg_price=Decimal("0")
)

is_flat = flat_position.quantity == 0
print(f"Position is flat: {is_flat}")  # True
```

### Position with Unrealized P&L

```python
from decimal import Decimal
from trade_analytics import Position

position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.00"),
    unrealized_pnl=Decimal("500.00")  # $5 profit per share
)

print(f"Unrealized P&L: ${position.unrealized_pnl}")
# Output: Unrealized P&L: $500.00
```

---

## Working with Market Data

### Basic Market Data

```python
from decimal import Decimal
from trade_analytics import MarketData

quote = MarketData(
    symbol="AAPL",
    bid=Decimal("150.00"),
    ask=Decimal("150.10"),
    last=Decimal("150.05"),
    volume=1000000
)

print(f"{quote.symbol}: ${quote.last}")
print(f"Bid: ${quote.bid} | Ask: ${quote.ask}")
print(f"Volume: {quote.volume:,}")
# Output:
# AAPL: $150.05
# Bid: $150.00 | Ask: $150.10
# Volume: 1,000,000
```

### Using Spread and Mid Properties

```python
from decimal import Decimal
from trade_analytics import MarketData

quote = MarketData(
    symbol="AAPL",
    bid=Decimal("150.00"),
    ask=Decimal("150.10"),
    last=Decimal("150.05"),
    volume=1000000
)

print(f"Spread: ${quote.spread}")  # $0.10
print(f"Mid price: ${quote.mid}")  # $150.05 (average of bid/ask)
```

### Market Data with Timestamp

```python
from datetime import datetime, timezone
from decimal import Decimal
from trade_analytics import MarketData

# Explicit timestamp
quote = MarketData(
    symbol="AAPL",
    bid=Decimal("150.00"),
    ask=Decimal("150.10"),
    last=Decimal("150.05"),
    volume=1000000,
    timestamp=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
)

print(f"Quote time: {quote.timestamp.isoformat()}")
# Output: Quote time: 2024-01-15T14:30:00+00:00
```

### Market Data Dictionary

The `PortfolioCalculator` expects market data as a dictionary keyed by symbol:

```python
from decimal import Decimal
from trade_analytics import MarketData

market_data = {
    "AAPL": MarketData(
        symbol="AAPL",
        bid=Decimal("155.00"),
        ask=Decimal("155.10"),
        last=Decimal("155.05"),
        volume=1000000
    ),
    "GOOGL": MarketData(
        symbol="GOOGL",
        bid=Decimal("145.00"),
        ask=Decimal("145.15"),
        last=Decimal("145.10"),
        volume=500000
    ),
    "MSFT": MarketData(
        symbol="MSFT",
        bid=Decimal("385.00"),
        ask=Decimal("385.20"),
        last=Decimal("385.10"),
        volume=750000
    ),
}

# Access individual quotes
print(f"AAPL last: ${market_data['AAPL'].last}")
# Output: AAPL last: $155.05
```

---

## Portfolio Calculations

### Calculate Total Portfolio Value

```python
from decimal import Decimal
from trade_analytics import Position, MarketData, PortfolioCalculator

# Create portfolio
positions = [
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")),
    Position(symbol="GOOGL", quantity=Decimal("50"), avg_price=Decimal("140.00")),
]

# Current market prices
market_data = {
    "AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.05"), volume=1000),
    "GOOGL": MarketData(symbol="GOOGL", bid=Decimal("145"), ask=Decimal("146"), last=Decimal("145.10"), volume=1000),
}

# Calculate total value
total_value = PortfolioCalculator.calculate_total_value(positions, market_data)
print(f"Portfolio value: ${total_value:,.2f}")
# Output: Portfolio value: $22,760.00
# (100 × $155.05) + (50 × $145.10) = $15,505 + $7,255 = $22,760
```

### Calculate Unrealized P&L

```python
from decimal import Decimal
from trade_analytics import Position, MarketData, PortfolioCalculator

positions = [
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")),
]

market_data = {
    "AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.05"), volume=1000),
}

pnl = PortfolioCalculator.calculate_pnl(positions, market_data)
print(f"Unrealized P&L: ${pnl:,.2f}")
# Output: Unrealized P&L: $505.00
# 100 × ($155.05 - $150.00) = $505
```

### Calculate Exposure by Symbol

```python
from decimal import Decimal
from trade_analytics import Position, MarketData, PortfolioCalculator

positions = [
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")),
    Position(symbol="TSLA", quantity=Decimal("-50"), avg_price=Decimal("200.00")),  # Short
    Position(symbol="GOOGL", quantity=Decimal("25"), avg_price=Decimal("140.00")),
]

market_data = {
    "AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.00"), volume=1000),
    "TSLA": MarketData(symbol="TSLA", bid=Decimal("210"), ask=Decimal("211"), last=Decimal("210.50"), volume=1000),
    "GOOGL": MarketData(symbol="GOOGL", bid=Decimal("145"), ask=Decimal("146"), last=Decimal("145.00"), volume=1000),
}

exposure = PortfolioCalculator.calculate_exposure_by_symbol(positions, market_data)

for symbol, value in exposure.items():
    print(f"{symbol}: ${value:,.2f}")
# Output:
# AAPL: $15,500.00
# TSLA: $10,525.00
# GOOGL: $3,625.00
```

### Empty Portfolio Handling

```python
from decimal import Decimal
from trade_analytics import PortfolioCalculator

# Empty portfolio returns sensible defaults
empty_positions = []
empty_market_data = {}

total = PortfolioCalculator.calculate_total_value(empty_positions, empty_market_data)
print(f"Total: ${total}")  # $0

pnl = PortfolioCalculator.calculate_pnl(empty_positions, empty_market_data)
print(f"P&L: ${pnl}")  # $0

exposure = PortfolioCalculator.calculate_exposure_by_symbol(empty_positions, empty_market_data)
print(f"Exposure: {exposure}")  # {}
```

---

## Error Handling

### Handling Invalid Trade Parameters

```python
from datetime import datetime, timezone
from decimal import Decimal
from trade_analytics import Trade, TradeSide, InvalidTradeError

# Invalid quantity
try:
    trade = Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("-100"),  # Negative quantity
        price=Decimal("150.00"),
        timestamp=datetime.now(timezone.utc)
    )
except InvalidTradeError as e:
    print(f"Error: {e.message}")
    print(f"Reason: {e.reason}")
    print(f"Details: {e.trade_details}")
# Output:
# Error: Quantity must be positive, got -100
# Reason: invalid_quantity
# Details: {'quantity': '-100'}
```

### Handling Empty Symbol

```python
from decimal import Decimal
from trade_analytics import Position, InvalidTradeError

try:
    position = Position(
        symbol="",  # Empty symbol
        quantity=Decimal("100"),
        avg_price=Decimal("150.00")
    )
except InvalidTradeError as e:
    print(f"Error: {e.message}")
    print(f"Reason: {e.reason}")
# Output:
# Error: Symbol cannot be empty
# Reason: empty_symbol
```

### Handling Missing Market Data

```python
from decimal import Decimal
from trade_analytics import Position, MarketData, PortfolioCalculator, MissingMarketDataError

positions = [
    Position(symbol="TSLA", quantity=Decimal("100"), avg_price=Decimal("200.00"))
]

# Market data doesn't include TSLA
market_data = {
    "AAPL": MarketData(symbol="AAPL", bid=Decimal("150"), ask=Decimal("151"), last=Decimal("150.50"), volume=1000),
}

try:
    total = PortfolioCalculator.calculate_total_value(positions, market_data)
except MissingMarketDataError as e:
    print(f"Missing data for: {e.symbol}")
    print(f"Available symbols: {e.available_symbols}")
# Output:
# Missing data for: TSLA
# Available symbols: ['AAPL']
```

### Catch-All Trading Errors

```python
from decimal import Decimal
from trade_analytics import (
    Position, MarketData, PortfolioCalculator,
    TradingError, InvalidTradeError, MissingMarketDataError
)

def calculate_portfolio_metrics(positions, market_data):
    """Calculate portfolio metrics with error handling."""
    try:
        total = PortfolioCalculator.calculate_total_value(positions, market_data)
        pnl = PortfolioCalculator.calculate_pnl(positions, market_data)
        return {"total": total, "pnl": pnl}
    except MissingMarketDataError as e:
        # Handle specific error
        print(f"Need market data for: {e.symbol}")
        return None
    except TradingError as e:
        # Catch-all for any trading error
        print(f"Trading error: {e.message}")
        return None

# Usage
positions = [Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150"))]
market_data = {"AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.50"), volume=1000)}

result = calculate_portfolio_metrics(positions, market_data)
if result:
    print(f"Total: ${result['total']}, P&L: ${result['pnl']}")
# Output: Total: $15550.00, P&L: $550.00
```

### Invalid Market Data (Crossed Market)

```python
from decimal import Decimal
from trade_analytics import MarketData, InvalidTradeError

try:
    # Bid > Ask is invalid
    quote = MarketData(
        symbol="AAPL",
        bid=Decimal("151.00"),  # Bid higher than ask
        ask=Decimal("150.00"),
        last=Decimal("150.50"),
        volume=1000
    )
except InvalidTradeError as e:
    print(f"Error: {e.message}")
    print(f"Reason: {e.reason}")
# Output:
# Error: Invalid spread: bid (151.00) > ask (150.00)
# Reason: invalid_spread
```

---

## Serialization

### Trade Serialization

```python
from datetime import datetime, timezone
from decimal import Decimal
import json
from trade_analytics import Trade, TradeSide

# Create a trade
trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    trade_id="T001"
)

# Serialize to dictionary
data = trade.to_dict()
print(json.dumps(data, indent=2))
# Output:
# {
#   "symbol": "AAPL",
#   "side": "BUY",
#   "quantity": "100",
#   "price": "150.50",
#   "timestamp": "2024-01-15T10:30:00+00:00",
#   "trade_id": "T001"
# }

# Deserialize from dictionary
restored_trade = Trade.from_dict(data)
print(f"Restored: {restored_trade.symbol} {restored_trade.side.value}")
# Output: Restored: AAPL BUY
```

### Position Serialization

```python
from decimal import Decimal
import json
from trade_analytics import Position

position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.00"),
    unrealized_pnl=Decimal("500.00")
)

# Serialize
data = position.to_dict()
print(json.dumps(data, indent=2))
# Output:
# {
#   "symbol": "AAPL",
#   "quantity": "100",
#   "avg_price": "150.00",
#   "unrealized_pnl": "500.00"
# }

# Deserialize
restored = Position.from_dict(data)
print(f"Restored: {restored.quantity} {restored.symbol}")
# Output: Restored: 100 AAPL
```

### MarketData Serialization

```python
from datetime import datetime, timezone
from decimal import Decimal
import json
from trade_analytics import MarketData

quote = MarketData(
    symbol="AAPL",
    bid=Decimal("150.00"),
    ask=Decimal("150.10"),
    last=Decimal("150.05"),
    volume=1000000,
    timestamp=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
)

# Serialize
data = quote.to_dict()
print(json.dumps(data, indent=2))
# Output:
# {
#   "symbol": "AAPL",
#   "bid": "150.00",
#   "ask": "150.10",
#   "last": "150.05",
#   "volume": 1000000,
#   "timestamp": "2024-01-15T14:30:00+00:00"
# }

# Deserialize
restored = MarketData.from_dict(data)
print(f"Restored: {restored.symbol} @ ${restored.last}")
# Output: Restored: AAPL @ $150.05
```

### Saving Portfolio to JSON

```python
from decimal import Decimal
import json
from trade_analytics import Position

# Portfolio
portfolio = [
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")),
    Position(symbol="GOOGL", quantity=Decimal("50"), avg_price=Decimal("140.00")),
    Position(symbol="MSFT", quantity=Decimal("25"), avg_price=Decimal("380.00")),
]

# Serialize all positions
portfolio_data = [p.to_dict() for p in portfolio]
json_string = json.dumps(portfolio_data, indent=2)
print(json_string)

# Later, restore the portfolio
loaded_data = json.loads(json_string)
restored_portfolio = [Position.from_dict(p) for p in loaded_data]

for pos in restored_portfolio:
    print(f"{pos.symbol}: {pos.quantity} shares @ ${pos.avg_price}")
# Output:
# AAPL: 100 shares @ $150.00
# GOOGL: 50 shares @ $140.00
# MSFT: 25 shares @ $380.00
```

---

[Back to README](../README.md) | [API Reference](API.md)
