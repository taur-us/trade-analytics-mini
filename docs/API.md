# API Reference

Complete API documentation for the Trade Analytics library.

## Table of Contents

- [Overview](#overview)
- [Package Imports](#package-imports)
- [Models](#models)
  - [TradeSide](#tradeside)
  - [Trade](#trade)
  - [Position](#position)
  - [MarketData](#marketdata)
- [Calculator](#calculator)
  - [PortfolioCalculator](#portfoliocalculator)
- [Exceptions](#exceptions)
  - [TradingError](#tradingerror)
  - [InvalidTradeError](#invalidtradeerror)
  - [InsufficientFundsError](#insufficientfundserror)
  - [MarketClosedError](#marketclosederror)
  - [MissingMarketDataError](#missingmarketdataerror)
- [Type Reference](#type-reference)

---

## Overview

The `trade_analytics` package provides data models and utilities for trading analytics. All public classes and exceptions are exported from the main package.

### Package Structure

```
trade_analytics/
├── __init__.py       # Main exports
├── models.py         # TradeSide, Trade, Position, MarketData
├── calculator.py     # PortfolioCalculator
└── exceptions.py     # Exception hierarchy
```

---

## Package Imports

All public classes can be imported from the main package:

```python
from trade_analytics import (
    # Models
    TradeSide,
    Trade,
    Position,
    MarketData,
    # Calculator
    PortfolioCalculator,
    # Exceptions
    TradingError,
    InvalidTradeError,
    InsufficientFundsError,
    MarketClosedError,
    MissingMarketDataError,
)
```

---

## Models

### TradeSide

Enumeration representing the direction of a trade.

```python
from trade_analytics import TradeSide
```

#### Values

| Value | Description |
|-------|-------------|
| `TradeSide.BUY` | Buy order (going long) |
| `TradeSide.SELL` | Sell order (going short or closing long) |

#### Example

```python
from trade_analytics import TradeSide

side = TradeSide.BUY
print(side.value)  # "BUY"
```

---

### Trade

Immutable dataclass representing a trade execution.

```python
from trade_analytics import Trade
```

#### Constructor

```python
Trade(
    symbol: str,
    side: TradeSide,
    quantity: Decimal,
    price: Decimal,
    timestamp: datetime,
    trade_id: str = ""
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbol` | `str` | Ticker symbol (e.g., "AAPL"). Automatically uppercased. |
| `side` | `TradeSide` | Trade direction (BUY or SELL) |
| `quantity` | `Decimal` | Number of shares/units. Must be positive. |
| `price` | `Decimal` | Execution price. Must be positive. |
| `timestamp` | `datetime` | Execution timestamp (UTC recommended) |
| `trade_id` | `str` | Optional unique identifier for the trade |

#### Attributes

All constructor parameters are available as read-only attributes.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `Dict[str, Any]` | Convert trade to dictionary |
| `from_dict(data)` | `Trade` | Class method to create Trade from dictionary |

#### Validation

The Trade class validates all inputs on construction and raises `InvalidTradeError` for:

- Empty or whitespace-only symbol (`reason="empty_symbol"`)
- Zero or negative quantity (`reason="invalid_quantity"`)
- Zero or negative price (`reason="invalid_price"`)

#### Example

```python
from datetime import datetime, timezone
from decimal import Decimal
from trade_analytics import Trade, TradeSide

trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime.now(timezone.utc),
    trade_id="T001"
)

# Access attributes
print(trade.symbol)    # "AAPL"
print(trade.quantity)  # Decimal("100")

# Serialize to dictionary
data = trade.to_dict()

# Deserialize from dictionary
trade2 = Trade.from_dict(data)
```

---

### Position

Dataclass representing a portfolio position.

```python
from trade_analytics import Position
```

#### Constructor

```python
Position(
    symbol: str,
    quantity: Decimal,
    avg_price: Decimal,
    unrealized_pnl: Decimal = Decimal("0")
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbol` | `str` | Ticker symbol. Automatically uppercased. |
| `quantity` | `Decimal` | Net position (positive=long, negative=short, zero=flat) |
| `avg_price` | `Decimal` | Volume-weighted average entry price. Must be non-negative. |
| `unrealized_pnl` | `Decimal` | Unrealized profit/loss. Defaults to 0. |

#### Attributes

All constructor parameters are available as attributes. Unlike `Trade`, `Position` is mutable.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `Dict[str, Any]` | Convert position to dictionary |
| `from_dict(data)` | `Position` | Class method to create Position from dictionary |

#### Validation

The Position class validates inputs and raises `InvalidTradeError` for:

- Empty or whitespace-only symbol (`reason="empty_symbol"`)
- Negative average price (`reason="invalid_avg_price"`)

#### Example

```python
from decimal import Decimal
from trade_analytics import Position

# Long position
long_position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.00")
)

# Short position
short_position = Position(
    symbol="TSLA",
    quantity=Decimal("-50"),  # Negative = short
    avg_price=Decimal("200.00")
)

# Serialize/deserialize
data = long_position.to_dict()
restored = Position.from_dict(data)
```

---

### MarketData

Immutable dataclass representing current market quote data.

```python
from trade_analytics import MarketData
```

#### Constructor

```python
MarketData(
    symbol: str,
    bid: Decimal,
    ask: Decimal,
    last: Decimal,
    volume: int,
    timestamp: datetime = <current UTC time>
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbol` | `str` | Ticker symbol. Automatically uppercased. |
| `bid` | `Decimal` | Best bid price. Must be non-negative. |
| `ask` | `Decimal` | Best ask price. Must be non-negative and >= bid. |
| `last` | `Decimal` | Last traded price. Must be non-negative. |
| `volume` | `int` | Trading volume. Must be non-negative. |
| `timestamp` | `datetime` | Quote timestamp. Defaults to current UTC time. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `spread` | `Decimal` | Bid-ask spread (`ask - bid`) |
| `mid` | `Decimal` | Mid price (`(bid + ask) / 2`) |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `Dict[str, Any]` | Convert market data to dictionary |
| `from_dict(data)` | `MarketData` | Class method to create MarketData from dictionary |

#### Validation

The MarketData class validates inputs and raises `InvalidTradeError` for:

- Empty or whitespace-only symbol (`reason="empty_symbol"`)
- Negative bid price (`reason="invalid_bid"`)
- Negative ask price (`reason="invalid_ask"`)
- Bid greater than ask (`reason="invalid_spread"`)
- Negative last price (`reason="invalid_last"`)
- Negative volume (`reason="invalid_volume"`)

#### Example

```python
from decimal import Decimal
from trade_analytics import MarketData

market = MarketData(
    symbol="AAPL",
    bid=Decimal("150.00"),
    ask=Decimal("150.10"),
    last=Decimal("150.05"),
    volume=1000000
)

# Access computed properties
print(market.spread)  # Decimal("0.10")
print(market.mid)     # Decimal("150.05")
```

---

## Calculator

### PortfolioCalculator

Static class providing portfolio analytics calculations.

```python
from trade_analytics import PortfolioCalculator
```

All methods are static (pure functions) that do not modify input data.

#### calculate_total_value

Calculate the total market value of all positions.

```python
@staticmethod
def calculate_total_value(
    positions: List[Position],
    market_data: Dict[str, MarketData]
) -> Decimal
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `positions` | `List[Position]` | List of portfolio positions |
| `market_data` | `Dict[str, MarketData]` | Dictionary mapping symbols to market data |

**Returns:** `Decimal` - Total portfolio market value (sum of quantity × last price)

**Raises:** `MissingMarketDataError` if market data is missing for any position's symbol

**Example:**

```python
from decimal import Decimal
from trade_analytics import Position, MarketData, PortfolioCalculator

positions = [
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))
]
market_data = {
    "AAPL": MarketData(
        symbol="AAPL",
        bid=Decimal("155.00"),
        ask=Decimal("155.10"),
        last=Decimal("155.05"),
        volume=1000000
    )
}

total = PortfolioCalculator.calculate_total_value(positions, market_data)
print(total)  # Decimal("15505.00")
```

---

#### calculate_pnl

Calculate the total unrealized P&L across all positions.

```python
@staticmethod
def calculate_pnl(
    positions: List[Position],
    market_data: Dict[str, MarketData]
) -> Decimal
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `positions` | `List[Position]` | List of portfolio positions |
| `market_data` | `Dict[str, MarketData]` | Dictionary mapping symbols to market data |

**Returns:** `Decimal` - Total unrealized P&L (positive = profit, negative = loss)

**Calculation:** Sum of `quantity × (current_price - avg_price)` for each position

**Raises:** `MissingMarketDataError` if market data is missing for any position's symbol

**Example:**

```python
from decimal import Decimal
from trade_analytics import Position, MarketData, PortfolioCalculator

positions = [
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))
]
market_data = {
    "AAPL": MarketData(
        symbol="AAPL",
        bid=Decimal("155.00"),
        ask=Decimal("155.10"),
        last=Decimal("155.05"),
        volume=1000000
    )
}

pnl = PortfolioCalculator.calculate_pnl(positions, market_data)
print(pnl)  # Decimal("505.00") = 100 × (155.05 - 150.00)
```

---

#### calculate_exposure_by_symbol

Calculate the absolute value exposure for each symbol.

```python
@staticmethod
def calculate_exposure_by_symbol(
    positions: List[Position],
    market_data: Dict[str, MarketData]
) -> Dict[str, Decimal]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `positions` | `List[Position]` | List of portfolio positions |
| `market_data` | `Dict[str, MarketData]` | Dictionary mapping symbols to market data |

**Returns:** `Dict[str, Decimal]` - Dictionary mapping symbols to their absolute exposure

**Calculation:** For each symbol, calculates `abs(quantity × current_price)`. Multiple positions in the same symbol are aggregated.

**Raises:** `MissingMarketDataError` if market data is missing for any position's symbol

**Example:**

```python
from decimal import Decimal
from trade_analytics import Position, MarketData, PortfolioCalculator

positions = [
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")),
    Position(symbol="TSLA", quantity=Decimal("-50"), avg_price=Decimal("200.00")),  # Short
]
market_data = {
    "AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.05"), volume=1000),
    "TSLA": MarketData(symbol="TSLA", bid=Decimal("210"), ask=Decimal("211"), last=Decimal("210.50"), volume=1000),
}

exposure = PortfolioCalculator.calculate_exposure_by_symbol(positions, market_data)
# {"AAPL": Decimal("15505.00"), "TSLA": Decimal("10525.00")}
```

---

## Exceptions

All exceptions inherit from `TradingError`, which inherits from `Exception`.

### Exception Hierarchy

```
Exception
└── TradingError
    ├── InvalidTradeError
    ├── InsufficientFundsError
    ├── MarketClosedError
    └── MissingMarketDataError
```

---

### TradingError

Base exception for all trading-related errors.

```python
from trade_analytics import TradingError
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |

#### Example

```python
from trade_analytics import TradingError

try:
    # ... trading operation
    pass
except TradingError as e:
    print(f"Trading error: {e.message}")
```

---

### InvalidTradeError

Raised when trade parameters are invalid.

```python
from trade_analytics import InvalidTradeError
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `trade_details` | `dict` | Dictionary containing the invalid trade details |
| `reason` | `str` | Specific reason code (e.g., "empty_symbol", "invalid_quantity") |

#### Reason Codes

| Reason | Description |
|--------|-------------|
| `empty_symbol` | Symbol is empty or whitespace |
| `invalid_quantity` | Quantity is zero or negative |
| `invalid_price` | Price is zero or negative |
| `invalid_avg_price` | Average price is negative |
| `invalid_bid` | Bid price is negative |
| `invalid_ask` | Ask price is negative |
| `invalid_spread` | Bid is greater than ask |
| `invalid_last` | Last price is negative |
| `invalid_volume` | Volume is negative |

#### Example

```python
from decimal import Decimal
from trade_analytics import Trade, TradeSide, InvalidTradeError
from datetime import datetime, timezone

try:
    trade = Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("-100"),  # Invalid: negative
        price=Decimal("150.00"),
        timestamp=datetime.now(timezone.utc)
    )
except InvalidTradeError as e:
    print(f"Invalid trade: {e.message}")
    print(f"Reason: {e.reason}")  # "invalid_quantity"
    print(f"Details: {e.trade_details}")  # {"quantity": "-100"}
```

---

### InsufficientFundsError

Raised when there are insufficient funds for a trade.

```python
from trade_analytics import InsufficientFundsError
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `required` | `Decimal` | Amount of funds required |
| `available` | `Decimal` | Amount of funds available |

#### Example

```python
from decimal import Decimal
from trade_analytics import InsufficientFundsError

try:
    # ... trading operation
    raise InsufficientFundsError(
        "Insufficient funds for trade",
        required=Decimal("10000.00"),
        available=Decimal("5000.00")
    )
except InsufficientFundsError as e:
    print(f"Need ${e.required}, have ${e.available}")
```

---

### MarketClosedError

Raised when attempting to trade in a closed market.

```python
from trade_analytics import MarketClosedError
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `symbol` | `str` | The symbol of the closed market |
| `market_hours` | `str` | Description of market hours |

#### Example

```python
from trade_analytics import MarketClosedError

try:
    # ... trading operation
    raise MarketClosedError(
        "Market is closed",
        symbol="NYSE:AAPL",
        market_hours="9:30 AM - 4:00 PM ET"
    )
except MarketClosedError as e:
    print(f"Cannot trade {e.symbol}. Hours: {e.market_hours}")
```

---

### MissingMarketDataError

Raised when market data is unavailable for a required symbol.

```python
from trade_analytics import MissingMarketDataError
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `symbol` | `str` | The symbol missing market data |
| `available_symbols` | `List[str]` | List of symbols with available data |

#### Example

```python
from decimal import Decimal
from trade_analytics import Position, PortfolioCalculator, MissingMarketDataError

positions = [
    Position(symbol="TSLA", quantity=Decimal("100"), avg_price=Decimal("200.00"))
]
market_data = {}  # Empty - no data

try:
    PortfolioCalculator.calculate_total_value(positions, market_data)
except MissingMarketDataError as e:
    print(f"Missing data for: {e.symbol}")
    print(f"Available symbols: {e.available_symbols}")
```

---

## Type Reference

### Quick Reference Table

| Class | Mutability | Key Fields |
|-------|------------|------------|
| `TradeSide` | N/A (enum) | `BUY`, `SELL` |
| `Trade` | Immutable | `symbol`, `side`, `quantity`, `price`, `timestamp` |
| `Position` | Mutable | `symbol`, `quantity`, `avg_price`, `unrealized_pnl` |
| `MarketData` | Immutable | `symbol`, `bid`, `ask`, `last`, `volume` |

### Common Type Patterns

```python
from decimal import Decimal
from datetime import datetime
from typing import Dict, List
from trade_analytics import Position, MarketData

# Position list (portfolio)
positions: List[Position]

# Market data dictionary
market_data: Dict[str, MarketData]

# Exposure by symbol
exposure: Dict[str, Decimal]
```

---

[Back to README](../README.md) | [View Examples](EXAMPLES.md)
