# Technical Design Document: CORE-002

## Implement Portfolio Calculator

**Task ID:** CORE-002
**Priority:** HIGH
**Estimated Hours:** 2.0
**Author:** Technical Lead
**Date:** 2024-11-25
**Status:** DRAFT
**Depends On:** CORE-001 (Completed)

---

## 1. Problem Summary

The trade analytics system requires a portfolio analytics calculator to perform essential portfolio-level computations. Investment professionals and automated trading systems need to:

- **Calculate Total Portfolio Value**: Determine the current market value of all positions across multiple symbols
- **Calculate Profit & Loss (P&L)**: Compute unrealized gains/losses based on current market prices vs. average entry prices
- **Analyze Exposure by Symbol**: Understand concentration risk by calculating the value exposure to each individual symbol
- **Support Risk Management**: Enable downstream risk metrics calculations by providing accurate position valuations

Without these calculations, users cannot assess portfolio performance, manage risk, or make informed trading decisions.

---

## 2. Current State

### Existing Codebase Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| `src/trade_analytics/models.py` | ✅ Exists | Contains `Position`, `MarketData`, `Trade`, `TradeSide` |
| `src/trade_analytics/exceptions.py` | ✅ Exists | Contains `TradingError`, `InvalidTradeError`, etc. |
| `src/trade_analytics/calculator.py` | **Does not exist** | Needs to be created |
| `tests/test_calculator.py` | **Does not exist** | Needs to be created |

### Available Data Models (from CORE-001)

**Position Dataclass:**
```python
@dataclass
class Position:
    symbol: str           # Ticker symbol (uppercased)
    quantity: Decimal     # Net position (positive=long, negative=short, zero=flat)
    avg_price: Decimal    # Volume-weighted average entry price (>= 0)
    unrealized_pnl: Decimal = Decimal("0")  # Unrealized profit/loss
```

**MarketData Dataclass:**
```python
@dataclass(frozen=True)
class MarketData:
    symbol: str       # Ticker symbol (uppercased)
    bid: Decimal      # Best bid price
    ask: Decimal      # Best ask price
    last: Decimal     # Last traded price
    volume: int       # Trading volume
    timestamp: datetime

    @property
    def mid(self) -> Decimal:
        """Calculate the mid price (bid + ask) / 2."""
```

### Dependencies
- Depends on CORE-001 models: `Position`, `MarketData`
- Depends on CORE-001 exceptions: `TradingError`, `InvalidTradeError`

---

## 3. Proposed Solution

### High-Level Approach

Create a `PortfolioCalculator` class that provides static methods for portfolio analytics calculations. The calculator will:

1. **Be Stateless**: All methods are pure functions that take inputs and return results without side effects
2. **Use Decimal Arithmetic**: Maintain precision for financial calculations using Python's `Decimal` type
3. **Handle Edge Cases Gracefully**: Return sensible defaults (zero values) for empty portfolios and handle missing market data explicitly
4. **Follow Single Responsibility**: Each method performs one calculation type
5. **Be Type-Safe**: Full type annotations for IDE support and static analysis

### Design Principles

- **Immutability**: Calculator methods do not modify input data
- **Explicit Error Handling**: Raise specific exceptions for error cases (e.g., missing market data)
- **Testability**: Pure functions with no side effects enable comprehensive unit testing
- **Extensibility**: Class design allows easy addition of new calculation methods

### Valuation Methodology

| Metric | Formula | Notes |
|--------|---------|-------|
| Position Value | `quantity × current_price` | Use `MarketData.last` as current price |
| Total Portfolio Value | `Σ position_values` | Sum across all positions |
| Position P&L | `quantity × (current_price - avg_price)` | Per-position unrealized P&L |
| Total P&L | `Σ position_pnls` | Sum across all positions |
| Symbol Exposure | `abs(position_value)` | Absolute value for exposure (longs and shorts both count) |

---

## 4. Components

### 4.1 Module: `src/trade_analytics/calculator.py`

#### Class: PortfolioCalculator

| Method | Purpose | Inputs | Output |
|--------|---------|--------|--------|
| `calculate_total_value` | Compute total portfolio market value | `positions: List[Position]`, `market_data: Dict[str, MarketData]` | `Decimal` |
| `calculate_pnl` | Compute total unrealized P&L | `positions: List[Position]`, `market_data: Dict[str, MarketData]` | `Decimal` |
| `calculate_exposure_by_symbol` | Calculate exposure breakdown | `positions: List[Position]` | `Dict[str, Decimal]` |

### 4.2 Module: `src/trade_analytics/exceptions.py` (Modifications)

#### New Exception Class

| Exception | Purpose | Key Attributes |
|-----------|---------|----------------|
| `MissingMarketDataError` | Raised when market data is unavailable for a symbol | `symbol`, `available_symbols` |

### 4.3 Module: `tests/test_calculator.py`

#### Test Classes

| Test Class | Coverage Target |
|------------|-----------------|
| `TestCalculateTotalValue` | Total value calculation with various scenarios |
| `TestCalculatePnl` | P&L calculation with long/short positions |
| `TestCalculateExposureBySymbol` | Exposure aggregation and breakdown |
| `TestEdgeCases` | Empty portfolio, missing data, zero quantities |

---

## 5. Data Models

### 5.1 Input Data Structures

The calculator uses existing models from CORE-001:

**Positions Input:**
```python
positions: List[Position]
# Example:
[
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")),
    Position(symbol="GOOGL", quantity=Decimal("-50"), avg_price=Decimal("140.00")),  # Short
    Position(symbol="MSFT", quantity=Decimal("200"), avg_price=Decimal("380.00")),
]
```

**Market Data Input:**
```python
market_data: Dict[str, MarketData]
# Example:
{
    "AAPL": MarketData(symbol="AAPL", bid=Decimal("155.00"), ask=Decimal("155.10"), last=Decimal("155.05"), volume=1000000),
    "GOOGL": MarketData(symbol="GOOGL", bid=Decimal("145.00"), ask=Decimal("145.10"), last=Decimal("145.05"), volume=500000),
    "MSFT": MarketData(symbol="MSFT", bid=Decimal("385.00"), ask=Decimal("385.20"), last=Decimal("385.10"), volume=750000),
}
```

### 5.2 Output Data Structures

**Total Value Output:**
```python
Decimal("92520.00")  # Sum of all position market values
```

**P&L Output:**
```python
Decimal("1257.50")  # Sum of all unrealized P&L (can be negative)
```

**Exposure by Symbol Output:**
```python
{
    "AAPL": Decimal("15505.00"),   # abs(100 × 155.05)
    "GOOGL": Decimal("7252.50"),   # abs(-50 × 145.05)
    "MSFT": Decimal("77020.00"),   # abs(200 × 385.10)
}
```

### 5.3 New Exception: MissingMarketDataError

```python
@dataclass
class MissingMarketDataError(TradingError):
    """Raised when market data is unavailable for a required symbol."""
    symbol: str                    # The symbol missing market data
    available_symbols: List[str]   # Symbols that DO have market data
```

---

## 6. API Contracts

### 6.1 PortfolioCalculator Class Interface

```python
class PortfolioCalculator:
    """Portfolio analytics calculator for computing portfolio metrics.

    This class provides static methods for calculating portfolio-level
    metrics such as total value, P&L, and exposure by symbol.

    All methods are pure functions that do not modify input data.
    """

    @staticmethod
    def calculate_total_value(
        positions: List[Position],
        market_data: Dict[str, MarketData],
    ) -> Decimal:
        """Calculate the total market value of all positions.

        Args:
            positions: List of portfolio positions.
            market_data: Dictionary mapping symbols to current market data.

        Returns:
            Total portfolio market value (sum of quantity × last price).
            Returns Decimal("0") for empty portfolios.

        Raises:
            MissingMarketDataError: If market data is missing for any position's symbol.

        Example:
            >>> positions = [Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))]
            >>> market_data = {"AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.50"), volume=1000)}
            >>> PortfolioCalculator.calculate_total_value(positions, market_data)
            Decimal("15550.00")
        """

    @staticmethod
    def calculate_pnl(
        positions: List[Position],
        market_data: Dict[str, MarketData],
    ) -> Decimal:
        """Calculate the total unrealized P&L across all positions.

        Args:
            positions: List of portfolio positions.
            market_data: Dictionary mapping symbols to current market data.

        Returns:
            Total unrealized P&L (sum of quantity × (current_price - avg_price)).
            Returns Decimal("0") for empty portfolios.
            Positive values indicate profit, negative values indicate loss.

        Raises:
            MissingMarketDataError: If market data is missing for any position's symbol.

        Example:
            >>> positions = [Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))]
            >>> market_data = {"AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.50"), volume=1000)}
            >>> PortfolioCalculator.calculate_pnl(positions, market_data)
            Decimal("550.00")  # 100 × (155.50 - 150.00)
        """

    @staticmethod
    def calculate_exposure_by_symbol(
        positions: List[Position],
        market_data: Dict[str, MarketData],
    ) -> Dict[str, Decimal]:
        """Calculate the absolute value exposure for each symbol.

        Exposure represents the total market value at risk for each symbol,
        calculated as the absolute value of (quantity × current price).
        Both long and short positions contribute positive exposure.

        Args:
            positions: List of portfolio positions.
            market_data: Dictionary mapping symbols to current market data.

        Returns:
            Dictionary mapping symbols to their absolute exposure values.
            Returns empty dict for empty portfolios.

        Raises:
            MissingMarketDataError: If market data is missing for any position's symbol.

        Example:
            >>> positions = [
            ...     Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150")),
            ...     Position(symbol="AAPL", quantity=Decimal("50"), avg_price=Decimal("155")),  # Same symbol
            ... ]
            >>> # Returns aggregated exposure per unique symbol
        """
```

### 6.2 MissingMarketDataError Interface

```python
class MissingMarketDataError(TradingError):
    """Raised when market data is unavailable for a required symbol.

    Attributes:
        message: Human-readable error description.
        symbol: The symbol for which market data is missing.
        available_symbols: List of symbols that have market data available.
    """

    def __init__(
        self,
        message: str,
        symbol: str,
        available_symbols: Optional[List[str]] = None,
    ) -> None:
        """Initialize MissingMarketDataError.

        Args:
            message: Human-readable error description.
            symbol: The symbol for which market data is missing.
            available_symbols: List of symbols that have market data available.
        """
```

---

## 7. Error Handling

### 7.1 Error Scenarios and Responses

| Scenario | Exception | Recovery Strategy |
|----------|-----------|-------------------|
| Empty positions list | None | Return `Decimal("0")` or empty dict |
| Position with zero quantity | None | Include in calculation (contributes 0) |
| Missing market data for symbol | `MissingMarketDataError` | Caller must provide complete market data |
| Empty market_data dict (with positions) | `MissingMarketDataError` | First missing symbol triggers error |
| Negative position quantity (short) | None | Valid case, calculate normally |
| Market data symbol case mismatch | None | Symbols are normalized to uppercase in models |

### 7.2 Error Message Format

All exceptions follow this format:
```
{ErrorType}: {brief description}. Symbol: {symbol}. Available: {available_symbols}
```

Example:
```
MissingMarketDataError: No market data available for symbol. Symbol: TSLA. Available: ['AAPL', 'GOOGL', 'MSFT']
```

### 7.3 Validation Strategy

| Input | Validation | Location |
|-------|------------|----------|
| `positions` | Must be a list (can be empty) | Method entry |
| `market_data` | Must be a dict (can be empty if no positions) | Method entry |
| Position symbols | Already validated by `Position.__post_init__` | CORE-001 |
| MarketData fields | Already validated by `MarketData.__post_init__` | CORE-001 |

### 7.4 Design Decision: Strict vs. Lenient Market Data Handling

**Chosen Approach: Strict (Raise Exception)**

**Rationale:**
- Financial calculations require complete data for accuracy
- Silent handling (e.g., skipping positions) could lead to incorrect investment decisions
- Explicit errors force callers to ensure data completeness
- Matches industry best practices for financial software

**Alternative Considered (Lenient):**
- Skip positions without market data
- Return partial results with warnings
- Rejected because: Could mask data issues and lead to incorrect valuations

---

## 8. Implementation Plan

### Phase 1: Add New Exception (15 min)

| Step | Task | File |
|------|------|------|
| 1.1 | Add `MissingMarketDataError` class | `src/trade_analytics/exceptions.py` |
| 1.2 | Add to `__all__` exports | `src/trade_analytics/exceptions.py` |
| 1.3 | Export from package `__init__.py` | `src/trade_analytics/__init__.py` |

### Phase 2: Implement PortfolioCalculator (45 min)

| Step | Task | File |
|------|------|------|
| 2.1 | Create `calculator.py` with imports and docstring | `src/trade_analytics/calculator.py` |
| 2.2 | Implement `PortfolioCalculator` class structure | `src/trade_analytics/calculator.py` |
| 2.3 | Implement `calculate_total_value` method | `src/trade_analytics/calculator.py` |
| 2.4 | Implement `calculate_pnl` method | `src/trade_analytics/calculator.py` |
| 2.5 | Implement `calculate_exposure_by_symbol` method | `src/trade_analytics/calculator.py` |
| 2.6 | Add `__all__` exports | `src/trade_analytics/calculator.py` |
| 2.7 | Export from package `__init__.py` | `src/trade_analytics/__init__.py` |

### Phase 3: Write Unit Tests (45 min)

| Step | Task | File |
|------|------|------|
| 3.1 | Create `test_calculator.py` with imports | `tests/test_calculator.py` |
| 3.2 | Add fixtures for multi-position portfolios | `tests/conftest.py` |
| 3.3 | Write `TestCalculateTotalValue` tests | `tests/test_calculator.py` |
| 3.4 | Write `TestCalculatePnl` tests | `tests/test_calculator.py` |
| 3.5 | Write `TestCalculateExposureBySymbol` tests | `tests/test_calculator.py` |
| 3.6 | Write `TestEdgeCases` tests | `tests/test_calculator.py` |
| 3.7 | Write `TestMissingMarketDataError` tests | `tests/test_calculator.py` |

### Phase 4: Integration & Verification (15 min)

| Step | Task | Command/File |
|------|------|--------------|
| 4.1 | Run full test suite | `pytest tests/ -v` |
| 4.2 | Check test coverage | `pytest --cov=src/trade_analytics --cov-report=term-missing` |
| 4.3 | Run type checking | `mypy src/trade_analytics/` |
| 4.4 | Run linter | `ruff check src/trade_analytics/` |
| 4.5 | Verify all exports work | `python -c "from trade_analytics import PortfolioCalculator"` |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Decimal precision issues** | Medium | High | Use `Decimal` consistently; test with many decimal places; avoid float conversions |
| **Symbol case sensitivity bugs** | Low | Medium | Models already normalize to uppercase; add tests for mixed-case inputs |
| **Performance with large portfolios** | Low | Medium | O(n) algorithms are acceptable; profile if >10k positions become common |
| **Short position calculation errors** | Medium | High | Extensive testing with negative quantities; verify P&L signs are correct |
| **Concurrent market data updates** | Low | Low | Calculator is stateless; caller responsible for data consistency |
| **Division by zero** | Low | High | No division operations in core calculations; guard against zero avg_price if needed |
| **Aggregation of duplicate symbols** | Medium | Medium | Document behavior; `calculate_exposure_by_symbol` aggregates by design |

### Detailed Mitigation Strategies

**Decimal Precision:**
```python
# GOOD: String constructors
price = Decimal("150.50")

# BAD: Float constructors (precision loss)
price = Decimal(150.50)  # Don't do this!
```

**Short Position P&L:**
```python
# Long position: quantity=100, avg_price=150, current=155
# P&L = 100 × (155 - 150) = +500 (profit)

# Short position: quantity=-50, avg_price=140, current=145
# P&L = -50 × (145 - 140) = -250 (loss on short, price went up)
```

---

## 10. Success Criteria

### Functional Requirements

| Criteria | Verification Method |
|----------|-------------------|
| `calculate_total_value` returns correct sum | Unit tests with known values |
| `calculate_pnl` handles long positions correctly | Unit test: positive P&L when price > avg_price |
| `calculate_pnl` handles short positions correctly | Unit test: positive P&L when price < avg_price (for shorts) |
| `calculate_exposure_by_symbol` aggregates correctly | Unit test with multiple positions in same symbol |
| Empty portfolio returns zero/empty | Unit tests for empty input list |
| Missing market data raises exception | Unit test verifies `MissingMarketDataError` |

### Quality Requirements

| Criteria | Target | Verification |
|----------|--------|--------------|
| Test coverage | ≥ 95% | `pytest --cov=src/trade_analytics --cov-report=term-missing` |
| Type hints | 100% | `mypy src/trade_analytics/` passes |
| Docstrings | All public APIs | Manual review |
| Linting | No errors | `ruff check src/trade_analytics/` or similar |

### Test Cases (Minimum Required)

```
tests/test_calculator.py:
  TestCalculateTotalValue:
    ✓ test_single_position
    ✓ test_multiple_positions
    ✓ test_empty_portfolio_returns_zero
    ✓ test_missing_market_data_raises_error
    ✓ test_zero_quantity_position

  TestCalculatePnl:
    ✓ test_long_position_profit
    ✓ test_long_position_loss
    ✓ test_short_position_profit
    ✓ test_short_position_loss
    ✓ test_multiple_positions_mixed
    ✓ test_empty_portfolio_returns_zero
    ✓ test_missing_market_data_raises_error
    ✓ test_breakeven_returns_zero

  TestCalculateExposureBySymbol:
    ✓ test_single_position
    ✓ test_multiple_symbols
    ✓ test_aggregates_same_symbol
    ✓ test_short_position_positive_exposure
    ✓ test_empty_portfolio_returns_empty_dict
    ✓ test_missing_market_data_raises_error

  TestMissingMarketDataError:
    ✓ test_exception_attributes
    ✓ test_exception_inheritance
    ✓ test_error_message_format
```

---

## Appendix A: File Structure After Implementation

```
trade-analytics-mini/
├── src/
│   └── trade_analytics/
│       ├── __init__.py          # Updated: exports PortfolioCalculator, MissingMarketDataError
│       ├── exceptions.py        # Updated: adds MissingMarketDataError
│       ├── models.py            # Unchanged (from CORE-001)
│       └── calculator.py        # NEW: PortfolioCalculator class
├── tests/
│   ├── __init__.py              # Unchanged
│   ├── conftest.py              # Updated: new fixtures for calculator tests
│   ├── test_models.py           # Unchanged (from CORE-001)
│   └── test_calculator.py       # NEW: Calculator unit tests
└── ...
```

---

## Appendix B: Example Usage

```python
from decimal import Decimal
from datetime import datetime, timezone
from trade_analytics import (
    Position,
    MarketData,
    PortfolioCalculator,
    MissingMarketDataError,
)

# Create portfolio positions
positions = [
    Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")),
    Position(symbol="GOOGL", quantity=Decimal("-50"), avg_price=Decimal("140.00")),  # Short
    Position(symbol="MSFT", quantity=Decimal("200"), avg_price=Decimal("380.00")),
]

# Create market data dictionary
market_data = {
    "AAPL": MarketData(
        symbol="AAPL",
        bid=Decimal("155.00"),
        ask=Decimal("155.10"),
        last=Decimal("155.05"),
        volume=1000000,
    ),
    "GOOGL": MarketData(
        symbol="GOOGL",
        bid=Decimal("145.00"),
        ask=Decimal("145.10"),
        last=Decimal("145.05"),
        volume=500000,
    ),
    "MSFT": MarketData(
        symbol="MSFT",
        bid=Decimal("385.00"),
        ask=Decimal("385.20"),
        last=Decimal("385.10"),
        volume=750000,
    ),
}

# Calculate portfolio metrics
total_value = PortfolioCalculator.calculate_total_value(positions, market_data)
print(f"Total Portfolio Value: ${total_value:,.2f}")
# Output: Total Portfolio Value: $84,267.50

total_pnl = PortfolioCalculator.calculate_pnl(positions, market_data)
print(f"Total Unrealized P&L: ${total_pnl:,.2f}")
# Output: Total Unrealized P&L: $1,767.50

exposure = PortfolioCalculator.calculate_exposure_by_symbol(positions, market_data)
print("Exposure by Symbol:")
for symbol, value in exposure.items():
    print(f"  {symbol}: ${value:,.2f}")
# Output:
#   AAPL: $15,505.00
#   GOOGL: $7,252.50
#   MSFT: $77,020.00

# Handle missing market data
try:
    positions_with_unknown = positions + [
        Position(symbol="TSLA", quantity=Decimal("25"), avg_price=Decimal("250.00"))
    ]
    PortfolioCalculator.calculate_total_value(positions_with_unknown, market_data)
except MissingMarketDataError as e:
    print(f"Error: Missing data for {e.symbol}")
    print(f"Available symbols: {e.available_symbols}")
# Output:
#   Error: Missing data for TSLA
#   Available symbols: ['AAPL', 'GOOGL', 'MSFT']
```

---

## Appendix C: Calculation Examples

### Example 1: Total Value Calculation

| Position | Quantity | Current Price | Value |
|----------|----------|---------------|-------|
| AAPL | 100 | $155.05 | $15,505.00 |
| GOOGL | -50 | $145.05 | -$7,252.50 |
| MSFT | 200 | $385.10 | $77,020.00 |
| **Total** | | | **$85,272.50** |

Note: Total value includes negative values for short positions.

### Example 2: P&L Calculation

| Position | Quantity | Avg Price | Current | P&L |
|----------|----------|-----------|---------|-----|
| AAPL | 100 | $150.00 | $155.05 | +$505.00 |
| GOOGL | -50 | $140.00 | $145.05 | -$252.50 |
| MSFT | 200 | $380.00 | $385.10 | +$1,020.00 |
| **Total** | | | | **+$1,272.50** |

Note: GOOGL short position lost money because price went up (from $140 to $145.05).

### Example 3: Exposure Calculation

| Position | Quantity | Current | Market Value | Exposure |
|----------|----------|---------|--------------|----------|
| AAPL | 100 | $155.05 | $15,505.00 | $15,505.00 |
| GOOGL | -50 | $145.05 | -$7,252.50 | $7,252.50 |
| MSFT | 200 | $385.10 | $77,020.00 | $77,020.00 |
| **Total Exposure** | | | | **$99,777.50** |

Note: Exposure uses absolute values - both long and short positions contribute positive exposure.

---

**Document Version:** 1.0
**Last Updated:** 2024-11-25
**Next Review:** Before implementation begins
