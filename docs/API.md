# API Reference

> Complete API documentation for Trade Analytics Mini

**Version:** 0.1.0
**Last Updated:** November 26, 2024

---

## Table of Contents

- [Package Overview](#package-overview)
- [Data Models](#data-models)
  - [TradeSide](#tradeside)
  - [Trade](#trade)
  - [Position](#position)
  - [MarketData](#marketdata)
- [Exceptions](#exceptions)
  - [Exception Hierarchy](#exception-hierarchy)
  - [TradingError](#tradingerror)
  - [InvalidTradeError](#invalidtradeerror)
  - [InsufficientFundsError](#insufficientfundserror)
  - [MarketClosedError](#marketclosederror)
- [Serialization](#serialization)
- [Type Annotations](#type-annotations)

---

## Package Overview

The `trade_analytics` package provides foundational data structures for trading domain entities. All components are accessible from the top-level package:

```python
from trade_analytics import (
    # Data Models
    TradeSide,
    Trade,
    Position,
    MarketData,
    # Exceptions
    TradingError,
    InvalidTradeError,
    InsufficientFundsError,
    MarketClosedError,
)

# Version
from trade_analytics import __version__
print(__version__)  # "0.1.0"
```

### Module Structure

```
trade_analytics/
├── __init__.py      # Package exports and version
├── models.py        # TradeSide, Trade, Position, MarketData
└── exceptions.py    # Exception hierarchy
```

---

## Data Models

### TradeSide

An enumeration representing the direction of a trade.

```python
from enum import Enum

class TradeSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
```

#### Values

| Value | Description |
|-------|-------------|
| `BUY` | Represents a buy order (going long) |
| `SELL` | Represents a sell order (going short or closing long) |

#### Usage

```python
from trade_analytics import TradeSide

# Access values
buy = TradeSide.BUY
sell = TradeSide.SELL

# Get string value
print(buy.value)  # "BUY"

# Create from string
side = TradeSide("BUY")  # TradeSide.BUY
```

---

### Trade

A frozen (immutable) dataclass representing a single trade execution.

```python
@dataclass(frozen=True)
class Trade:
    symbol: str
    side: TradeSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    trade_id: str = ""
```

#### Attributes

| Attribute | Type | Description | Validation |
|-----------|------|-------------|------------|
| `symbol` | `str` | Ticker symbol (e.g., "AAPL") | Cannot be empty; auto-uppercased |
| `side` | `TradeSide` | Trade direction | Must be `TradeSide.BUY` or `TradeSide.SELL` |
| `quantity` | `Decimal` | Number of shares/units | Must be > 0 |
| `price` | `Decimal` | Execution price per unit | Must be > 0 |
| `timestamp` | `datetime` | Execution timestamp (UTC) | Required |
| `trade_id` | `str` | Optional unique identifier | Optional, defaults to "" |

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert Trade to dictionary representation suitable for serialization.

```python
trade.to_dict()
# Returns:
# {
#     "symbol": "AAPL",
#     "side": "BUY",
#     "quantity": "100",
#     "price": "150.50",
#     "timestamp": "2024-01-15T10:30:00+00:00",
#     "trade_id": "T001"
# }
```

##### `from_dict(data: Dict[str, Any]) -> Trade` (classmethod)

Create Trade from dictionary representation.

```python
data = {
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": "100",
    "price": "150.50",
    "timestamp": "2024-01-15T10:30:00+00:00",
    "trade_id": "T001"
}
trade = Trade.from_dict(data)
```

**Raises:**
- `InvalidTradeError`: If data validation fails
- `KeyError`: If required fields are missing
- `ValueError`: If data types are incorrect

#### Validation Rules

1. **Symbol**: Cannot be empty or whitespace-only
2. **Quantity**: Must be a positive number (`quantity > 0`)
3. **Price**: Must be a positive number (`price > 0`)

#### Immutability

Trade is a frozen dataclass - attributes cannot be modified after creation:

```python
trade = Trade(...)
trade.price = Decimal("200")  # Raises FrozenInstanceError
```

---

### Position

A dataclass representing a portfolio position in a specific security.

```python
@dataclass
class Position:
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
```

#### Attributes

| Attribute | Type | Description | Validation |
|-----------|------|-------------|------------|
| `symbol` | `str` | Ticker symbol (e.g., "AAPL") | Cannot be empty; auto-uppercased |
| `quantity` | `Decimal` | Net position quantity | Positive = long, negative = short, zero = flat |
| `avg_price` | `Decimal` | Volume-weighted average entry price | Must be >= 0 |
| `unrealized_pnl` | `Decimal` | Unrealized profit/loss | Optional, defaults to 0 |

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert Position to dictionary representation.

```python
position.to_dict()
# Returns:
# {
#     "symbol": "AAPL",
#     "quantity": "100",
#     "avg_price": "150.50",
#     "unrealized_pnl": "250.00"
# }
```

##### `from_dict(data: Dict[str, Any]) -> Position` (classmethod)

Create Position from dictionary representation.

```python
data = {
    "symbol": "AAPL",
    "quantity": "100",
    "avg_price": "150.50",
    "unrealized_pnl": "250.00"
}
position = Position.from_dict(data)
```

**Raises:**
- `InvalidTradeError`: If data validation fails
- `KeyError`: If required fields are missing

#### Validation Rules

1. **Symbol**: Cannot be empty or whitespace-only
2. **Average Price**: Must be non-negative (`avg_price >= 0`)

#### Mutability

Position is a mutable dataclass - attributes can be modified:

```python
position = Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))
position.quantity = Decimal("200")  # OK - Position is mutable
position.unrealized_pnl = Decimal("500.00")  # OK
```

---

### MarketData

A frozen (immutable) dataclass representing a market quote for a security.

```python
@dataclass(frozen=True)
class MarketData:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

#### Attributes

| Attribute | Type | Description | Validation |
|-----------|------|-------------|------------|
| `symbol` | `str` | Ticker symbol (e.g., "AAPL") | Cannot be empty; auto-uppercased |
| `bid` | `Decimal` | Best bid price | Must be >= 0 |
| `ask` | `Decimal` | Best ask price | Must be >= 0 and >= bid |
| `last` | `Decimal` | Last traded price | Must be >= 0 |
| `volume` | `int` | Trading volume | Must be >= 0 |
| `timestamp` | `datetime` | Quote timestamp (UTC) | Defaults to current time |

#### Properties

##### `spread -> Decimal`

Calculate the bid-ask spread.

```python
market = MarketData(
    symbol="AAPL",
    bid=Decimal("150.45"),
    ask=Decimal("150.55"),
    last=Decimal("150.50"),
    volume=1000000
)
print(market.spread)  # Decimal("0.10")
```

##### `mid -> Decimal`

Calculate the mid price (average of bid and ask).

```python
print(market.mid)  # Decimal("150.50")
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert MarketData to dictionary representation.

```python
market.to_dict()
# Returns:
# {
#     "symbol": "AAPL",
#     "bid": "150.45",
#     "ask": "150.55",
#     "last": "150.50",
#     "volume": 1000000,
#     "timestamp": "2024-01-15T10:30:00+00:00"
# }
```

##### `from_dict(data: Dict[str, Any]) -> MarketData` (classmethod)

Create MarketData from dictionary representation.

```python
data = {
    "symbol": "AAPL",
    "bid": "150.45",
    "ask": "150.55",
    "last": "150.50",
    "volume": 1000000,
    "timestamp": "2024-01-15T10:30:00+00:00"
}
market = MarketData.from_dict(data)
```

**Raises:**
- `InvalidTradeError`: If data validation fails
- `KeyError`: If required fields are missing

#### Validation Rules

1. **Symbol**: Cannot be empty or whitespace-only
2. **Bid**: Must be non-negative (`bid >= 0`)
3. **Ask**: Must be non-negative (`ask >= 0`) and >= bid
4. **Spread**: Bid must not exceed ask (`bid <= ask`)
5. **Last**: Must be non-negative (`last >= 0`)
6. **Volume**: Must be non-negative (`volume >= 0`)

---

## Exceptions

### Exception Hierarchy

```
Exception
└── TradingError (Base)
    ├── InvalidTradeError
    ├── InsufficientFundsError
    └── MarketClosedError
```

All trading-specific exceptions inherit from `TradingError`, enabling catch-all handling:

```python
from trade_analytics import TradingError, InvalidTradeError

try:
    # Trading operations
    ...
except TradingError as e:
    # Catches any trading-related error
    print(f"Trading error: {e.message}")
```

---

### TradingError

Base exception for all trading-related errors.

```python
class TradingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |

#### Usage

```python
from trade_analytics import TradingError

try:
    raise TradingError("Something went wrong")
except TradingError as e:
    print(e.message)  # "Something went wrong"
```

---

### InvalidTradeError

Raised when trade parameters are invalid.

```python
class InvalidTradeError(TradingError):
    def __init__(
        self,
        message: str,
        trade_details: Optional[dict] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.trade_details = trade_details or {}
        self.reason = reason
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `trade_details` | `dict` | Dictionary containing the invalid trade details |
| `reason` | `str` or `None` | Specific reason code for the error |

#### Reason Codes

| Reason | Trigger |
|--------|---------|
| `empty_symbol` | Symbol is empty or whitespace |
| `invalid_quantity` | Quantity is zero or negative |
| `invalid_price` | Price is zero or negative |
| `invalid_avg_price` | Average price is negative |
| `invalid_bid` | Bid price is negative |
| `invalid_ask` | Ask price is negative |
| `invalid_spread` | Bid exceeds ask |
| `invalid_last` | Last price is negative |
| `invalid_volume` | Volume is negative |

#### Usage

```python
from trade_analytics import Trade, TradeSide, InvalidTradeError
from decimal import Decimal
from datetime import datetime, timezone

try:
    trade = Trade(
        symbol="",  # Empty symbol
        side=TradeSide.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.50"),
        timestamp=datetime.now(timezone.utc)
    )
except InvalidTradeError as e:
    print(f"Error: {e.message}")        # "Symbol cannot be empty"
    print(f"Details: {e.trade_details}") # {"symbol": ""}
    print(f"Reason: {e.reason}")         # "empty_symbol"
```

---

### InsufficientFundsError

Raised when there are insufficient funds for a trade.

```python
class InsufficientFundsError(TradingError):
    def __init__(
        self,
        message: str,
        required: Optional[Decimal] = None,
        available: Optional[Decimal] = None,
    ) -> None:
        super().__init__(message)
        self.required = required
        self.available = available
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `required` | `Decimal` or `None` | Amount of funds required for the trade |
| `available` | `Decimal` or `None` | Amount of funds currently available |

#### Usage

```python
from trade_analytics import InsufficientFundsError
from decimal import Decimal

# Example: Raising when checking account balance
def execute_trade(cost: Decimal, balance: Decimal):
    if cost > balance:
        raise InsufficientFundsError(
            message=f"Cannot execute trade: need ${cost}, have ${balance}",
            required=cost,
            available=balance
        )

try:
    execute_trade(Decimal("10000"), Decimal("5000"))
except InsufficientFundsError as e:
    print(f"Error: {e.message}")
    print(f"Required: ${e.required}")   # $10000
    print(f"Available: ${e.available}") # $5000
    print(f"Shortfall: ${e.required - e.available}")  # $5000
```

---

### MarketClosedError

Raised when attempting to trade in a closed market.

```python
class MarketClosedError(TradingError):
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        market_hours: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.symbol = symbol
        self.market_hours = market_hours
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `symbol` | `str` or `None` | Ticker symbol of the closed market |
| `market_hours` | `str` or `None` | String describing the market's trading hours |

#### Usage

```python
from trade_analytics import MarketClosedError

# Example: Checking market hours before trading
def check_market_open(symbol: str, is_open: bool):
    if not is_open:
        raise MarketClosedError(
            message=f"Market for {symbol} is currently closed",
            symbol=symbol,
            market_hours="9:30 AM - 4:00 PM ET, Mon-Fri"
        )

try:
    check_market_open("AAPL", False)
except MarketClosedError as e:
    print(f"Error: {e.message}")
    print(f"Symbol: {e.symbol}")              # "AAPL"
    print(f"Trading hours: {e.market_hours}") # "9:30 AM - 4:00 PM ET, Mon-Fri"
```

---

## Serialization

All data models support dictionary serialization via `to_dict()` and `from_dict()` methods.

### Serialization Format

Values are serialized as follows:

| Type | Serialization |
|------|---------------|
| `Decimal` | String (preserves precision) |
| `datetime` | ISO 8601 format string |
| `TradeSide` | Value string ("BUY" or "SELL") |
| `int` | Integer |
| `str` | String |

### JSON Serialization

```python
import json
from trade_analytics import Trade

# Serialize to JSON
trade_dict = trade.to_dict()
json_string = json.dumps(trade_dict, indent=2)

# Deserialize from JSON
data = json.loads(json_string)
trade = Trade.from_dict(data)
```

### Batch Serialization

```python
# Serialize multiple trades
trades = [trade1, trade2, trade3]
trades_data = [t.to_dict() for t in trades]
json_string = json.dumps(trades_data)

# Deserialize multiple trades
data_list = json.loads(json_string)
trades = [Trade.from_dict(d) for d in data_list]
```

---

## Type Annotations

All classes and methods include complete type annotations:

```python
from typing import Any, Dict, Optional
from decimal import Decimal
from datetime import datetime

# Trade methods
def to_dict(self) -> Dict[str, Any]: ...
def from_dict(cls, data: Dict[str, Any]) -> "Trade": ...

# Exception attributes
trade_details: Optional[dict]
reason: Optional[str]
required: Optional[Decimal]
available: Optional[Decimal]
```

For static type checking, use:

```bash
pip install mypy
mypy your_code.py
```
