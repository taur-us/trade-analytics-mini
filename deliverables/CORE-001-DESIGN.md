# Technical Design Document: CORE-001

## Create Data Models and Exceptions

**Task ID:** CORE-001
**Priority:** CRITICAL
**Estimated Hours:** 1.0
**Author:** Technical Lead
**Date:** 2024-11-25
**Status:** DRAFT

---

## 1. Problem Summary

The trade-analytics-mini system requires foundational data structures to represent trading domain entities. Currently, there are no data models defined for trades, positions, or market data, which are essential for:

- Recording and tracking trade executions
- Managing portfolio positions
- Processing real-time market data
- Performing analytics calculations

Additionally, the system needs a consistent error handling strategy with domain-specific exceptions to provide meaningful error messages and enable proper error recovery throughout the application.

---

## 2. Current State

### Existing Codebase Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| `src/trade_analytics/__init__.py` | Exists | Empty file (module marker only) |
| `src/trade_analytics/models.py` | **Does not exist** | Needs to be created |
| `src/trade_analytics/exceptions.py` | **Does not exist** | Needs to be created |
| `tests/` | **Does not exist** | Test directory needs to be created |

### Dependencies
- No external dependencies on other modules
- This is a foundational module that other tasks depend on:
  - `CORE-002` (Portfolio Calculator)
  - `STORE-001` (SQLite Storage)

---

## 3. Proposed Solution

### High-Level Approach

1. **Data Models**: Use Python's `dataclasses` module to create immutable, type-hinted data structures that:
   - Leverage `@dataclass(frozen=True)` for immutability where appropriate
   - Include validation in `__post_init__` methods
   - Support serialization/deserialization patterns
   - Use `Enum` types for constrained fields (e.g., trade side)

2. **Exception Hierarchy**: Create a custom exception hierarchy rooted in a base `TradingError` class that:
   - Provides specific error types for different failure scenarios
   - Includes contextual information in exception attributes
   - Follows Python exception best practices

### Design Principles

- **Type Safety**: Full type annotations for IDE support and static analysis
- **Immutability**: Frozen dataclasses to prevent accidental mutation
- **Validation**: Early validation with clear error messages
- **Extensibility**: Easy to add new fields or exception types

---

## 4. Components

### 4.1 Module: `src/trade_analytics/models.py`

#### Classes to Implement

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `TradeSide` | Enum for trade direction | `BUY`, `SELL` |
| `Trade` | Represents a trade execution | symbol, side, quantity, price, timestamp |
| `Position` | Represents a portfolio position | symbol, quantity, avg_price, unrealized_pnl |
| `MarketData` | Represents current market quote | symbol, bid, ask, last, volume |

### 4.2 Module: `src/trade_analytics/exceptions.py`

#### Classes to Implement

| Exception | Purpose | Key Attributes |
|-----------|---------|----------------|
| `TradingError` | Base exception for all trading errors | message |
| `InvalidTradeError` | Invalid trade parameters | trade_details, reason |
| `InsufficientFundsError` | Not enough funds for trade | required, available |
| `MarketClosedError` | Market is not open for trading | symbol, market_hours |

### 4.3 Module: `tests/test_models.py`

#### Test Classes

| Test Class | Coverage Target |
|------------|-----------------|
| `TestTradeSide` | Enum values and string conversion |
| `TestTrade` | Creation, validation, edge cases |
| `TestPosition` | Creation, P&L calculation, validation |
| `TestMarketData` | Creation, spread calculation, validation |
| `TestExceptions` | Exception hierarchy, attributes, messages |

---

## 5. Data Models

### 5.1 TradeSide Enum

```python
class TradeSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
```

### 5.2 Trade Dataclass

```python
@dataclass(frozen=True)
class Trade:
    symbol: str           # Ticker symbol (e.g., "AAPL")
    side: TradeSide       # BUY or SELL
    quantity: Decimal     # Number of shares/units
    price: Decimal        # Execution price
    timestamp: datetime   # Execution timestamp (UTC)
    trade_id: str = ""    # Optional unique identifier
```

**Validation Rules:**
- `symbol`: Non-empty string, uppercase
- `quantity`: Positive decimal, > 0
- `price`: Positive decimal, > 0
- `timestamp`: Valid datetime object

### 5.3 Position Dataclass

```python
@dataclass
class Position:
    symbol: str           # Ticker symbol
    quantity: Decimal     # Net position (positive=long, negative=short)
    avg_price: Decimal    # Volume-weighted average entry price
    unrealized_pnl: Decimal = Decimal("0")  # Unrealized profit/loss
```

**Validation Rules:**
- `symbol`: Non-empty string, uppercase
- `quantity`: Any decimal (can be 0 or negative for short)
- `avg_price`: Non-negative decimal, >= 0

### 5.4 MarketData Dataclass

```python
@dataclass(frozen=True)
class MarketData:
    symbol: str       # Ticker symbol
    bid: Decimal      # Best bid price
    ask: Decimal      # Best ask price
    last: Decimal     # Last traded price
    volume: int       # Trading volume
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

**Validation Rules:**
- `symbol`: Non-empty string, uppercase
- `bid`, `ask`, `last`: Non-negative decimals
- `bid` <= `ask` (spread cannot be negative)
- `volume`: Non-negative integer

**Computed Properties:**
- `spread`: `ask - bid`
- `mid`: `(bid + ask) / 2`

---

## 6. API Contracts

### 6.1 Model Factory Methods (Optional Enhancement)

```python
class Trade:
    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        """Create Trade from dictionary representation."""

    def to_dict(self) -> dict:
        """Convert Trade to dictionary representation."""
```

### 6.2 Exception Interface

```python
class TradingError(Exception):
    """Base exception for all trading-related errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class InvalidTradeError(TradingError):
    """Raised when trade parameters are invalid."""

    def __init__(self, message: str, trade_details: dict = None, reason: str = None):
        super().__init__(message)
        self.trade_details = trade_details or {}
        self.reason = reason

class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds for a trade."""

    def __init__(self, message: str, required: Decimal = None, available: Decimal = None):
        super().__init__(message)
        self.required = required
        self.available = available

class MarketClosedError(TradingError):
    """Raised when attempting to trade in a closed market."""

    def __init__(self, message: str, symbol: str = None, market_hours: str = None):
        super().__init__(message)
        self.symbol = symbol
        self.market_hours = market_hours
```

---

## 7. Error Handling

### 7.1 Validation Error Strategy

| Scenario | Exception | Example |
|----------|-----------|---------|
| Empty symbol | `InvalidTradeError` | `symbol=""` |
| Negative quantity | `InvalidTradeError` | `quantity=-100` |
| Zero/negative price | `InvalidTradeError` | `price=0` |
| Invalid spread (bid > ask) | `InvalidTradeError` | `bid=100, ask=99` |

### 7.2 Error Message Format

All exceptions should follow this format:
```
{ErrorType}: {brief description}. Details: {specific context}
```

Example:
```
InvalidTradeError: Invalid quantity for trade. Details: quantity=-100 must be positive
```

### 7.3 Exception Chaining

When wrapping lower-level exceptions, preserve the original cause:
```python
try:
    # operation
except ValueError as e:
    raise InvalidTradeError(f"Validation failed: {e}") from e
```

---

## 8. Implementation Plan

### Phase 1: Exception Hierarchy (30 min)

| Step | Task | File |
|------|------|------|
| 1.1 | Create `exceptions.py` with base `TradingError` | `src/trade_analytics/exceptions.py` |
| 1.2 | Implement `InvalidTradeError` with attributes | `src/trade_analytics/exceptions.py` |
| 1.3 | Implement `InsufficientFundsError` with attributes | `src/trade_analytics/exceptions.py` |
| 1.4 | Implement `MarketClosedError` with attributes | `src/trade_analytics/exceptions.py` |
| 1.5 | Add module docstring and `__all__` exports | `src/trade_analytics/exceptions.py` |

### Phase 2: Data Models (45 min)

| Step | Task | File |
|------|------|------|
| 2.1 | Create `models.py` with imports | `src/trade_analytics/models.py` |
| 2.2 | Implement `TradeSide` enum | `src/trade_analytics/models.py` |
| 2.3 | Implement `Trade` dataclass with validation | `src/trade_analytics/models.py` |
| 2.4 | Implement `Position` dataclass with validation | `src/trade_analytics/models.py` |
| 2.5 | Implement `MarketData` dataclass with validation | `src/trade_analytics/models.py` |
| 2.6 | Add `__all__` exports | `src/trade_analytics/models.py` |

### Phase 3: Unit Tests (45 min)

| Step | Task | File |
|------|------|------|
| 3.1 | Create `tests/` directory and `conftest.py` | `tests/conftest.py` |
| 3.2 | Write `TestTradeSide` tests | `tests/test_models.py` |
| 3.3 | Write `TestTrade` tests (valid, invalid, edge cases) | `tests/test_models.py` |
| 3.4 | Write `TestPosition` tests | `tests/test_models.py` |
| 3.5 | Write `TestMarketData` tests | `tests/test_models.py` |
| 3.6 | Write `TestExceptions` tests | `tests/test_models.py` |
| 3.7 | Verify 95%+ coverage with `pytest --cov` | - |

### Phase 4: Integration & Documentation (10 min)

| Step | Task | File |
|------|------|------|
| 4.1 | Update `__init__.py` with public exports | `src/trade_analytics/__init__.py` |
| 4.2 | Add docstrings to all public classes/methods | All files |
| 4.3 | Run final test suite and coverage check | - |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Decimal precision issues** | Medium | High | Use `Decimal` type consistently; test with edge cases (very small/large numbers) |
| **Timezone handling errors** | Medium | Medium | Store all timestamps in UTC; document timezone expectations |
| **Validation too strict** | Low | Medium | Start with essential validations; add more based on usage patterns |
| **Breaking changes for downstream tasks** | Medium | High | Finalize API before CORE-002/STORE-001 begin; use semantic versioning principles |
| **Missing edge cases in tests** | Medium | Medium | Use property-based testing with hypothesis if time permits; review boundary conditions |

### Mitigation Details

**Decimal Precision:**
- Import `Decimal` from `decimal` module
- Use string constructors: `Decimal("0.01")` not `Decimal(0.01)`
- Define precision constants if needed

**Timezone Handling:**
- Use `datetime.utcnow()` for current time
- Consider `datetime.timezone.utc` for explicit UTC
- Document that all timestamps are UTC in docstrings

---

## 10. Success Criteria

### Functional Requirements

| Criteria | Verification Method |
|----------|-------------------|
| Trade dataclass with all required fields | Unit test creates Trade with valid data |
| Position dataclass with all required fields | Unit test creates Position with valid data |
| MarketData dataclass with all required fields | Unit test creates MarketData with valid data |
| Custom exceptions with proper hierarchy | Unit test verifies `isinstance(e, TradingError)` |
| Validation rejects invalid data | Unit tests for each validation rule |

### Quality Requirements

| Criteria | Target | Verification |
|----------|--------|--------------|
| Test coverage | ≥ 95% | `pytest --cov=src/trade_analytics --cov-report=term-missing` |
| Type hints | 100% | `mypy src/trade_analytics/` passes |
| Docstrings | All public APIs | Manual review |
| Linting | No errors | `ruff check src/trade_analytics/` or similar |

### Test Cases (Minimum)

```
tests/test_models.py:
  ✓ test_trade_creation_valid
  ✓ test_trade_creation_invalid_symbol
  ✓ test_trade_creation_invalid_quantity
  ✓ test_trade_creation_invalid_price
  ✓ test_trade_immutability
  ✓ test_position_creation_valid
  ✓ test_position_with_negative_quantity (short position)
  ✓ test_market_data_creation_valid
  ✓ test_market_data_spread_calculation
  ✓ test_market_data_invalid_spread
  ✓ test_trading_error_base
  ✓ test_invalid_trade_error_attributes
  ✓ test_insufficient_funds_error_attributes
  ✓ test_market_closed_error_attributes
  ✓ test_exception_inheritance
```

---

## Appendix A: File Structure After Implementation

```
trade-analytics-mini/
├── src/
│   └── trade_analytics/
│       ├── __init__.py          # Public exports
│       ├── exceptions.py        # NEW: Exception hierarchy
│       └── models.py            # NEW: Data models
├── tests/
│   ├── __init__.py              # NEW: Test package marker
│   ├── conftest.py              # NEW: Pytest fixtures
│   └── test_models.py           # NEW: Model and exception tests
└── ...
```

## Appendix B: Example Usage

```python
from datetime import datetime
from decimal import Decimal
from trade_analytics.models import Trade, TradeSide, Position, MarketData
from trade_analytics.exceptions import InvalidTradeError, InsufficientFundsError

# Create a trade
trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime.utcnow()
)

# Create a position
position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.50"),
    unrealized_pnl=Decimal("250.00")
)

# Create market data
market_data = MarketData(
    symbol="AAPL",
    bid=Decimal("150.45"),
    ask=Decimal("150.55"),
    last=Decimal("150.50"),
    volume=1000000
)

# Exception handling
try:
    validate_trade(trade)
except InvalidTradeError as e:
    print(f"Trade invalid: {e.reason}")
except InsufficientFundsError as e:
    print(f"Need ${e.required}, have ${e.available}")
```

---

**Document Version:** 1.0
**Last Updated:** 2024-11-25
**Next Review:** Before implementation begins
