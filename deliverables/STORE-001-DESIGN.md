# Technical Design Document: STORE-001

## Add SQLite Storage Layer

**Task ID:** STORE-001
**Priority:** HIGH
**Estimated Hours:** 2.0
**Author:** Technical Lead
**Date:** 2024-11-25
**Status:** COMPLETED

---

## 1. Problem Summary

The trade-analytics-mini system requires persistent storage for trades and positions to enable:

- **Historical data retention**: Store trade executions for later analysis and reporting
- **Position tracking**: Persist portfolio positions across application restarts
- **Query capabilities**: Retrieve trades filtered by symbol and date range
- **Data integrity**: Ensure atomic operations and consistent state

Currently, all data exists only in memory and is lost when the application terminates. A SQLite-based storage layer will provide lightweight, file-based persistence without requiring external database infrastructure.

---

## 2. Current State

### Existing Codebase Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| `src/trade_analytics/models.py` | **Exists** | `Trade` and `Position` dataclasses with `to_dict()`/`from_dict()` |
| `src/trade_analytics/exceptions.py` | **Exists** | Exception hierarchy available for error handling |
| `src/trade_analytics/storage.py` | **Does not exist** | Needs to be created |
| `tests/test_storage.py` | **Does not exist** | Needs to be created |

### Relevant Model Details

**Trade dataclass** (frozen=True):
- `symbol: str` - Ticker symbol (uppercased)
- `side: TradeSide` - BUY or SELL enum
- `quantity: Decimal` - Number of shares
- `price: Decimal` - Execution price
- `timestamp: datetime` - Execution time (UTC)
- `trade_id: str` - Optional unique identifier (default: "")

**Position dataclass** (mutable):
- `symbol: str` - Ticker symbol (uppercased)
- `quantity: Decimal` - Net position (positive=long, negative=short)
- `avg_price: Decimal` - Volume-weighted average entry price
- `unrealized_pnl: Decimal` - Unrealized P&L (default: 0)

### Dependencies
- **Depends on**: `CORE-001` (models.py, exceptions.py) - COMPLETED
- **Depended by**: Future analytics and CLI modules

---

## 3. Proposed Solution

### High-Level Approach

Implement a `TradeStore` class that provides:

1. **SQLite Database**: File-based or in-memory database using Python's built-in `sqlite3` module
2. **Context Manager**: Safe connection handling with automatic cleanup
3. **CRUD Operations**: Create, read, update, delete for trades and positions
4. **Query Methods**: Filter trades by symbol and date range
5. **Schema Migration**: Auto-create tables on first connection

### Design Principles

- **No External Dependencies**: Use Python's built-in `sqlite3` module only
- **Thread Safety**: Use connection per operation pattern for concurrent access
- **Type Safety**: Full type annotations for IDE support and static analysis
- **Decimal Precision**: Store Decimal values as TEXT to preserve precision
- **UTC Timestamps**: Store all timestamps in ISO-8601 format (UTC)
- **Testability**: Support in-memory database for fast unit tests

---

## 4. Components

### 4.1 Module: `src/trade_analytics/storage.py`

#### Classes to Implement

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `TradeStore` | Main storage interface | `save_trade()`, `get_trades()`, `save_position()`, `get_positions()` |

#### Functions to Implement

| Function | Purpose |
|----------|---------|
| `_create_tables(conn)` | Initialize database schema |
| `_trade_to_row(trade)` | Convert Trade to database row tuple |
| `_row_to_trade(row)` | Convert database row to Trade |
| `_position_to_row(position)` | Convert Position to database row tuple |
| `_row_to_position(row)` | Convert database row to Position |

### 4.2 Module: `tests/test_storage.py`

#### Test Classes

| Test Class | Coverage Target |
|------------|-----------------|
| `TestTradeStore` | Store initialization, context manager |
| `TestSaveTrade` | Trade persistence operations |
| `TestGetTrades` | Trade retrieval and filtering |
| `TestSavePosition` | Position persistence operations |
| `TestGetPositions` | Position retrieval |
| `TestEdgeCases` | Error handling, empty results, large datasets |

---

## 5. Data Models

### 5.1 Database Schema

#### Table: `trades`

```sql
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity TEXT NOT NULL,
    price TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp);
```

#### Table: `positions`

```sql
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    quantity TEXT NOT NULL,
    avg_price TEXT NOT NULL,
    unrealized_pnl TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 5.2 Type Mappings

| Python Type | SQLite Type | Conversion Notes |
|-------------|-------------|------------------|
| `str` | `TEXT` | Direct mapping |
| `Decimal` | `TEXT` | Store as string to preserve precision |
| `datetime` | `TEXT` | ISO-8601 format with timezone |
| `TradeSide` | `TEXT` | Store enum value ("BUY"/"SELL") |
| `int` | `INTEGER` | For auto-increment ID only |

---

## 6. API Contracts

### 6.1 TradeStore Class

```python
class TradeStore:
    """SQLite-based storage for trades and positions.

    Provides persistent storage with CRUD operations and query methods.
    Supports both file-based and in-memory databases.

    Usage:
        # File-based database
        store = TradeStore("trades.db")

        # In-memory database (for testing)
        store = TradeStore(":memory:")

        # Context manager for connections
        with store:
            store.save_trade(trade)
            trades = store.get_trades("AAPL")

    Attributes:
        db_path: Path to SQLite database file or ":memory:" for in-memory.
    """

    def __init__(self, db_path: str = "trades.db") -> None:
        """Initialize TradeStore with database path.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory.
                     Creates the database file if it doesn't exist.
        """

    def __enter__(self) -> "TradeStore":
        """Enter context manager, establish database connection.

        Returns:
            Self for use in with statement.
        """

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        """Exit context manager, close database connection.

        Commits transaction if no exception, rolls back otherwise.
        """

    def save_trade(self, trade: Trade) -> None:
        """Save a trade to the database.

        If trade.trade_id is empty, generates a UUID.
        If trade_id already exists, updates the existing record.

        Args:
            trade: Trade object to persist.

        Raises:
            TradingError: If database operation fails.
        """

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Trade]:
        """Retrieve trades with optional filtering.

        Args:
            symbol: Filter by ticker symbol (case-insensitive).
            start_date: Filter trades on or after this datetime (inclusive).
            end_date: Filter trades on or before this datetime (inclusive).

        Returns:
            List of Trade objects matching criteria, ordered by timestamp desc.
            Empty list if no matches found.

        Raises:
            TradingError: If database query fails.
        """

    def save_position(self, position: Position) -> None:
        """Save or update a position in the database.

        Uses UPSERT semantics - inserts new or updates existing by symbol.

        Args:
            position: Position object to persist.

        Raises:
            TradingError: If database operation fails.
        """

    def get_positions(self) -> List[Position]:
        """Retrieve all positions.

        Returns:
            List of all Position objects, ordered by symbol.
            Empty list if no positions exist.

        Raises:
            TradingError: If database query fails.
        """

    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade by ID.

        Args:
            trade_id: Unique identifier of trade to delete.

        Returns:
            True if trade was deleted, False if not found.

        Raises:
            TradingError: If database operation fails.
        """

    def delete_position(self, symbol: str) -> bool:
        """Delete a position by symbol.

        Args:
            symbol: Ticker symbol of position to delete.

        Returns:
            True if position was deleted, False if not found.

        Raises:
            TradingError: If database operation fails.
        """

    def clear_all(self) -> None:
        """Delete all trades and positions.

        WARNING: This operation is irreversible.

        Raises:
            TradingError: If database operation fails.
        """
```

### 6.2 Exception Extension

Add to `src/trade_analytics/exceptions.py`:

```python
class StorageError(TradingError):
    """Raised when a storage operation fails.

    Attributes:
        message: Human-readable error description.
        operation: The operation that failed (e.g., "save_trade", "get_trades").
        original_error: The underlying database error, if any.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error
```

---

## 7. Error Handling

### 7.1 Error Scenarios and Handling

| Scenario | Exception | Handling Strategy |
|----------|-----------|-------------------|
| Database file permission error | `StorageError` | Wrap `sqlite3.OperationalError`, include path in message |
| Connection failure | `StorageError` | Retry once, then raise with details |
| Integrity constraint violation | `StorageError` | Log duplicate trade_id, wrap error |
| Invalid data in database | `InvalidTradeError` | Log corrupted row, skip and continue |
| Query timeout | `StorageError` | Set reasonable timeout (30s), raise on exceed |
| Disk full | `StorageError` | Wrap OS error, suggest cleanup |

### 7.2 Transaction Management

```python
def save_trade(self, trade: Trade) -> None:
    try:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(INSERT_SQL, _trade_to_row(trade))
            conn.commit()
    except sqlite3.IntegrityError as e:
        raise StorageError(
            f"Trade with ID {trade.trade_id} already exists",
            operation="save_trade",
            original_error=e,
        ) from e
    except sqlite3.Error as e:
        raise StorageError(
            f"Failed to save trade: {e}",
            operation="save_trade",
            original_error=e,
        ) from e
```

### 7.3 Connection Management

```python
@contextmanager
def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with proper cleanup.

    Yields:
        SQLite connection configured for the store.
    """
    conn = sqlite3.connect(
        self.db_path,
        timeout=30.0,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
```

---

## 8. Implementation Plan

### Phase 1: Exception and Module Setup (15 min)

| Step | Task | File |
|------|------|------|
| 1.1 | Add `StorageError` exception class | `src/trade_analytics/exceptions.py` |
| 1.2 | Export `StorageError` in `__all__` | `src/trade_analytics/exceptions.py` |
| 1.3 | Create `storage.py` with imports and constants | `src/trade_analytics/storage.py` |

### Phase 2: Database Schema (20 min)

| Step | Task | File |
|------|------|------|
| 2.1 | Define SQL constants for table creation | `src/trade_analytics/storage.py` |
| 2.2 | Define SQL constants for CRUD operations | `src/trade_analytics/storage.py` |
| 2.3 | Implement `_create_tables()` function | `src/trade_analytics/storage.py` |

### Phase 3: TradeStore Class - Core (30 min)

| Step | Task | File |
|------|------|------|
| 3.1 | Implement `__init__` method | `src/trade_analytics/storage.py` |
| 3.2 | Implement `__enter__` and `__exit__` context manager | `src/trade_analytics/storage.py` |
| 3.3 | Implement `_get_connection()` helper | `src/trade_analytics/storage.py` |
| 3.4 | Implement row conversion helpers | `src/trade_analytics/storage.py` |

### Phase 4: Trade Operations (25 min)

| Step | Task | File |
|------|------|------|
| 4.1 | Implement `save_trade()` method | `src/trade_analytics/storage.py` |
| 4.2 | Implement `get_trades()` with filtering | `src/trade_analytics/storage.py` |
| 4.3 | Implement `delete_trade()` method | `src/trade_analytics/storage.py` |

### Phase 5: Position Operations (20 min)

| Step | Task | File |
|------|------|------|
| 5.1 | Implement `save_position()` with UPSERT | `src/trade_analytics/storage.py` |
| 5.2 | Implement `get_positions()` method | `src/trade_analytics/storage.py` |
| 5.3 | Implement `delete_position()` method | `src/trade_analytics/storage.py` |
| 5.4 | Implement `clear_all()` method | `src/trade_analytics/storage.py` |

### Phase 6: Unit Tests (40 min)

| Step | Task | File |
|------|------|------|
| 6.1 | Create test fixtures for in-memory database | `tests/test_storage.py` |
| 6.2 | Write `TestTradeStore` initialization tests | `tests/test_storage.py` |
| 6.3 | Write `TestSaveTrade` tests | `tests/test_storage.py` |
| 6.4 | Write `TestGetTrades` with filter tests | `tests/test_storage.py` |
| 6.5 | Write `TestSavePosition` tests | `tests/test_storage.py` |
| 6.6 | Write `TestGetPositions` tests | `tests/test_storage.py` |
| 6.7 | Write edge case and error handling tests | `tests/test_storage.py` |

### Phase 7: Integration & Documentation (10 min)

| Step | Task | File |
|------|------|------|
| 7.1 | Update `__init__.py` with TradeStore export | `src/trade_analytics/__init__.py` |
| 7.2 | Add module docstring with usage examples | `src/trade_analytics/storage.py` |
| 7.3 | Run final test suite with coverage check | - |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Decimal precision loss** | Medium | High | Store Decimal as TEXT, test with high-precision values |
| **Timezone confusion** | Medium | High | Enforce UTC storage, document timezone handling |
| **Concurrent access issues** | Low | Medium | Use WAL mode for file DB, connection-per-operation pattern |
| **Large dataset performance** | Low | Medium | Add indexes on frequently queried columns |
| **trade_id collision** | Low | High | Generate UUID when empty, handle IntegrityError |
| **Database corruption** | Very Low | High | Use transactions, implement backup method if needed |
| **Path injection attacks** | Low | Medium | Validate db_path input, reject suspicious patterns |

### Mitigation Details

**Decimal Precision:**
```python
# Store as TEXT, not REAL
cursor.execute("INSERT INTO trades (quantity) VALUES (?)", (str(trade.quantity),))

# Retrieve and convert back
quantity = Decimal(row["quantity"])
```

**Timezone Handling:**
```python
# Always store in UTC ISO-8601 format
timestamp_str = trade.timestamp.astimezone(timezone.utc).isoformat()

# Parse back with timezone awareness
timestamp = datetime.fromisoformat(row["timestamp"])
```

**Concurrent Access (WAL mode):**
```python
conn.execute("PRAGMA journal_mode=WAL")
```

**trade_id Generation:**
```python
import uuid

def save_trade(self, trade: Trade) -> None:
    trade_id = trade.trade_id or str(uuid.uuid4())
    # Use trade_id in INSERT
```

---

## 10. Success Criteria

### Functional Requirements

| Criteria | Verification Method |
|----------|-------------------|
| `TradeStore` class exists in `storage.py` | Import succeeds |
| `save_trade(trade)` persists trade data | Unit test reads back saved trade |
| `get_trades(symbol, start_date, end_date)` filters correctly | Unit tests for each filter combination |
| `save_position(position)` persists position data | Unit test reads back saved position |
| `get_positions()` returns all positions | Unit test verifies count and data |
| SQLite database created with proper schema | Inspect database file, verify tables |
| Context manager handles connections | Unit test verifies cleanup on exception |
| Tests use in-memory SQLite | Test fixtures use `:memory:` |

### Quality Requirements

| Criteria | Target | Verification |
|----------|--------|--------------|
| Test coverage | >= 95% | `pytest --cov=src/trade_analytics --cov-report=term-missing` |
| Type hints | 100% | `mypy src/trade_analytics/storage.py` passes |
| Docstrings | All public APIs | Manual review |
| Linting | No errors | `ruff check src/trade_analytics/storage.py` |

### Test Cases (Minimum Required)

```
tests/test_storage.py:
  # Initialization
  ✓ test_store_creates_database_file
  ✓ test_store_in_memory_database
  ✓ test_store_creates_tables_on_init
  ✓ test_store_context_manager_enter_exit
  ✓ test_store_context_manager_cleanup_on_exception

  # Trade Operations
  ✓ test_save_trade_valid
  ✓ test_save_trade_generates_uuid_when_empty
  ✓ test_save_trade_updates_existing
  ✓ test_get_trades_all
  ✓ test_get_trades_filter_by_symbol
  ✓ test_get_trades_filter_by_symbol_case_insensitive
  ✓ test_get_trades_filter_by_start_date
  ✓ test_get_trades_filter_by_end_date
  ✓ test_get_trades_filter_by_date_range
  ✓ test_get_trades_empty_result
  ✓ test_delete_trade_exists
  ✓ test_delete_trade_not_found

  # Position Operations
  ✓ test_save_position_new
  ✓ test_save_position_update_existing
  ✓ test_get_positions_all
  ✓ test_get_positions_empty
  ✓ test_delete_position_exists
  ✓ test_delete_position_not_found

  # Edge Cases
  ✓ test_decimal_precision_preserved
  ✓ test_timestamp_timezone_handled
  ✓ test_special_characters_in_symbol
  ✓ test_clear_all_removes_everything
  ✓ test_storage_error_raised_on_failure
```

---

## Appendix A: File Structure After Implementation

```
trade-analytics-mini/
├── src/
│   └── trade_analytics/
│       ├── __init__.py          # Add TradeStore, StorageError exports
│       ├── exceptions.py        # Add StorageError class
│       ├── models.py            # Existing (no changes)
│       └── storage.py           # NEW: TradeStore implementation
├── tests/
│   ├── __init__.py              # Existing
│   ├── conftest.py              # Add storage fixtures
│   ├── test_models.py           # Existing
│   └── test_storage.py          # NEW: Storage tests
└── ...
```

## Appendix B: Example Usage

```python
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from trade_analytics import Trade, TradeSide, Position
from trade_analytics.storage import TradeStore

# Create store (file-based)
store = TradeStore("my_trades.db")

# Save trades using context manager
with store:
    trade1 = Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.50"),
        timestamp=datetime.now(timezone.utc),
    )
    store.save_trade(trade1)

    trade2 = Trade(
        symbol="GOOGL",
        side=TradeSide.BUY,
        quantity=Decimal("50"),
        price=Decimal("2800.00"),
        timestamp=datetime.now(timezone.utc),
    )
    store.save_trade(trade2)

# Query trades
with store:
    # Get all AAPL trades
    aapl_trades = store.get_trades(symbol="AAPL")

    # Get trades from last 7 days
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent_trades = store.get_trades(start_date=week_ago)

    # Get AAPL trades in date range
    filtered = store.get_trades(
        symbol="AAPL",
        start_date=week_ago,
        end_date=datetime.now(timezone.utc),
    )

# Save and retrieve positions
with store:
    position = Position(
        symbol="AAPL",
        quantity=Decimal("100"),
        avg_price=Decimal("150.50"),
        unrealized_pnl=Decimal("250.00"),
    )
    store.save_position(position)

    all_positions = store.get_positions()
    for pos in all_positions:
        print(f"{pos.symbol}: {pos.quantity} @ {pos.avg_price}")

# Testing with in-memory database
test_store = TradeStore(":memory:")
with test_store:
    # Fast, isolated tests
    test_store.save_trade(sample_trade)
    assert len(test_store.get_trades()) == 1
```

## Appendix C: SQL Queries Reference

```sql
-- Insert trade
INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(trade_id) DO UPDATE SET
    symbol=excluded.symbol,
    side=excluded.side,
    quantity=excluded.quantity,
    price=excluded.price,
    timestamp=excluded.timestamp;

-- Select trades with filters
SELECT * FROM trades
WHERE (?1 IS NULL OR UPPER(symbol) = UPPER(?1))
  AND (?2 IS NULL OR timestamp >= ?2)
  AND (?3 IS NULL OR timestamp <= ?3)
ORDER BY timestamp DESC;

-- Upsert position
INSERT INTO positions (symbol, quantity, avg_price, unrealized_pnl)
VALUES (?, ?, ?, ?)
ON CONFLICT(symbol) DO UPDATE SET
    quantity=excluded.quantity,
    avg_price=excluded.avg_price,
    unrealized_pnl=excluded.unrealized_pnl,
    updated_at=CURRENT_TIMESTAMP;

-- Select all positions
SELECT * FROM positions ORDER BY symbol;
```

---

**Document Version:** 1.0
**Last Updated:** 2024-11-25
**Next Review:** Before implementation begins
