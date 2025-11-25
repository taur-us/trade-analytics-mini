# Trade Analytics Mini

A Python library for trading analytics providing data models and portfolio calculations.

## Features

- **Data Models**: Immutable `Trade`, `Position`, and `MarketData` classes with validation
- **Portfolio Calculator**: Calculate total value, P&L, and exposure by symbol
- **Type Safety**: Full type hints for IDE support and static analysis
- **Validation**: Comprehensive input validation with clear error messages
- **Serialization**: Built-in `to_dict()` and `from_dict()` methods for JSON compatibility

## Installation

### Prerequisites

- Python 3.8 or higher

### Install from source

```bash
git clone <repository>
cd trade-analytics-mini
pip install -e .
```

## Quick Start

```python
from datetime import datetime, timezone
from decimal import Decimal
from trade_analytics import (
    Trade, TradeSide, Position, MarketData,
    PortfolioCalculator
)

# Create a trade
trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime.now(timezone.utc)
)

# Create a position
position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.00")
)

# Get current market data
market_data = {
    "AAPL": MarketData(
        symbol="AAPL",
        bid=Decimal("155.00"),
        ask=Decimal("155.10"),
        last=Decimal("155.05"),
        volume=1000000
    )
}

# Calculate portfolio metrics
total_value = PortfolioCalculator.calculate_total_value([position], market_data)
pnl = PortfolioCalculator.calculate_pnl([position], market_data)

print(f"Total Value: ${total_value}")  # Total Value: $15505.00
print(f"P&L: ${pnl}")                   # P&L: $505.00
```

## Documentation

- [API Reference](docs/API.md) - Complete API documentation
- [Examples](docs/EXAMPLES.md) - Usage examples and patterns

## Development

### Install development dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Run tests with coverage

```bash
pytest --cov=src/trade_analytics
```

## License

MIT
