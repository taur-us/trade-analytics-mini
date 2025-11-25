# Trade Analytics Mini

A comprehensive trading analytics library providing foundational data structures for representing trades, positions, and market data with built-in validation and serialization support.

## Features

- **Trade Execution Tracking** - Immutable trade records with validation
- **Position Management** - Track long/short positions with P&L
- **Market Data Handling** - Real-time quote data with spread/mid calculations
- **Custom Exceptions** - Domain-specific error handling for trading operations
- **Serialization** - Full JSON/dictionary serialization support
- **Type Safety** - Complete type annotations throughout

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/trade-analytics-mini.git
cd trade-analytics-mini

# Install test dependencies
pip install pytest pytest-cov

# Add src to PYTHONPATH for development
# Linux/macOS:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Windows (PowerShell):
$env:PYTHONPATH = "$env:PYTHONPATH;$(pwd)\src"

# Windows (CMD):
set PYTHONPATH=%PYTHONPATH%;%cd%\src
```

## Quick Start

### Import the Library

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import (
    Trade, TradeSide, Position, MarketData,
    InvalidTradeError, InsufficientFundsError, MarketClosedError
)
```

### Create a Trade

```python
# Create a buy order
trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime.now(timezone.utc),
    trade_id="T001"
)

print(f"Trade: {trade.symbol} {trade.side.value} {trade.quantity} @ ${trade.price}")
# Output: Trade: AAPL BUY 100 @ $150.50
```

### Track a Position

```python
# Create a position
position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.50"),
    unrealized_pnl=Decimal("250.00")
)

print(f"Position: {position.quantity} shares at avg ${position.avg_price}")
# Output: Position: 100 shares at avg $150.50
```

### Handle Market Data

```python
# Create market data quote
market = MarketData(
    symbol="AAPL",
    bid=Decimal("150.45"),
    ask=Decimal("150.55"),
    last=Decimal("150.50"),
    volume=1000000
)

print(f"Spread: ${market.spread}, Mid: ${market.mid}")
# Output: Spread: $0.10, Mid: $150.50
```

### Serialize and Deserialize

```python
import json

# Convert to dictionary
trade_dict = trade.to_dict()
print(json.dumps(trade_dict, indent=2))

# Recreate from dictionary
restored_trade = Trade.from_dict(trade_dict)
```

### Handle Errors

```python
try:
    # This will fail - quantity must be positive
    invalid_trade = Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("-100"),  # Invalid!
        price=Decimal("150.50"),
        timestamp=datetime.now(timezone.utc)
    )
except InvalidTradeError as e:
    print(f"Error: {e.message}")
    print(f"Reason: {e.reason}")
    # Output: Error: Quantity must be positive, got -100
    # Output: Reason: invalid_quantity
```

## Documentation

- **[API Reference](docs/API.md)** - Complete API documentation with all classes, methods, and validation rules
- **[Usage Examples](docs/EXAMPLES.md)** - Practical examples from basic to advanced usage

## CLI Usage

```bash
# Show portfolio
trade-analytics portfolio

# Show trade history
trade-analytics history --symbol AAPL --days 30

# Run analytics
trade-analytics analyze
```

## Testing

```bash
# Ensure PYTHONPATH is set (see Installation section)

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=trade_analytics --cov-report=term-missing

# Run specific test file
pytest tests/test_models.py -v
```

## Project Structure

```
trade-analytics-mini/
├── src/
│   └── trade_analytics/
│       ├── __init__.py      # Package exports
│       ├── models.py        # Data models (Trade, Position, MarketData)
│       └── exceptions.py    # Custom exceptions
├── tests/
│   ├── conftest.py          # Test fixtures
│   └── test_models.py       # Model tests
├── docs/
│   ├── API.md               # API reference
│   └── EXAMPLES.md          # Usage examples
└── README.md
```

## Version

Current version: **0.1.0**

## License

MIT License
