# Technical Design Document: STORE-001

## Add SQLite Storage Layer

**Task ID:** STORE-001
**Priority:** HIGH
**Estimated Hours:** 2.0
**Author:** Technical Lead
**Date:** 2024-11-26
**Status:** DRAFT

---

## 1. Problem Summary

The trade-analytics-mini system currently operates entirely in-memory with no data persistence. This creates several limitations:

- **No Historical Data**: Trades and positions are lost when the application terminates
- **No Audit Trail**: Cannot track trade history for compliance or analysis
- **Limited Analytics**: Cannot perform historical analytics without persistent storage
- **No Recovery**: System state cannot be restored after restart

This task implements a SQLite-based storage layer that provides:
- Persistent storage for trades and positions
- Full CRUD (Create, Read, Update, Delete) operations for trades
- Query interface for positions with filtering capabilities
- Thread-safe database access with connection pooling

---

## 2. Current State

### Existing Codebase Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| `src/trade_analytics/models.py` | **Exists** | Contains `Trade`, `Position`, `MarketData` dataclasses with `to_dict()`/`from_dict()` methods |
| `src/trade_analytics/exceptions.py` | **Exists** | Contains exception hierarchy (`TradingError`, `InvalidTradeError`, etc.) |
| `src/trade_analytics/calculator.py` | **Exists** | Contains `PortfolioCalculator` with static methods |
| `src/trade_analytics/__init__.py` | **Exists** | Exports models, exceptions, and calculator |
| `src/trade_analytics/storage.py` | **Does not exist** | Needs to be created |
| `tests/test_storage.py` | **Does not exist** | Needs to be created |

### Existing Data Models

The `Trade` and `Position` models already support serialization:

```python
# Trade model (frozen dataclass)
@dataclass(frozen=True)
class Trade:
    symbol: str           # Ticker symbol (e.g., "AAPL")
    side: TradeSide       # BUY or SELL
    quantity: Decimal     # Number of shares/units
    price: Decimal        # Execution price
    timestamp: datetime   # Execution timestamp (UTC)
    trade_id: str = ""    # Optional unique identifier

# Position model (mutable dataclass)
@dataclass
class Position:
    symbol: str           # Ticker symbol
    quantity: Decimal     # Net position (positive=long, negative=short)
    avg_price: Decimal    # Volume-weighted average entry price
    unrealized_pnl: Decimal = Decimal("0")
```

### Dependencies
- Depends on: `CORE-001` (Data Models) - **COMPLETED**
- Required by: Future CLI and reporting tasks

---

## 3. Proposed Solution

### High-Level Approach

1. **Storage Abstraction**: Create an abstract `StorageBackend` protocol/interface that defines the storage contract. This allows for future storage backends (PostgreSQL, Redis, etc.) while implementing SQLite first.

2. **SQLite Implementation**: Implement `SQLiteStorage` class that:
   - Uses Python's built-in `sqlite3` module (no external dependencies)
   - Implements connection pooling via `threading.local()` for thread safety
   - Stores trades and positions in separate tables
   - Uses parameterized queries to prevent SQL injection

3. **Repository Pattern**: Implement `TradeRepository` and `PositionRepository` classes that provide domain-specific operations and convert between domain models and database rows.

### Design Principles

- **Thread Safety**: Use thread-local connections to ensure safe concurrent access
- **ACID Compliance**: Leverage SQLite's transaction support for data integrity
- **Separation of Concerns**: Keep storage logic separate from domain models
- **Testability**: Use in-memory SQLite databases for unit tests
- **Type Safety**: Full type annotations for IDE support and static analysis

---

## 4. Components

### 4.1 Module: `src/trade_analytics/storage.py`

#### Classes to Implement

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `StorageBackend` | Abstract protocol for storage operations | `connect()`, `disconnect()`, `execute()` |
| `SQLiteStorage` | SQLite implementation of storage backend | `get_connection()`, `execute_query()`, `execute_many()` |
| `TradeRepository` | Trade-specific CRUD operations | `create()`, `get_by_id()`, `get_all()`, `update()`, `delete()`, `query()` |
| `PositionRepository` | Position query and update operations | `get_by_symbol()`, `get_all()`, `upsert()`, `delete()`, `query()` |

### 4.2 Module: `src/trade_analytics/exceptions.py` (Additions)

#### New Exceptions

| Exception | Purpose | Key Attributes |
|-----------|---------|----------------|
| `StorageError` | Base exception for storage operations | `message`, `operation` |
| `RecordNotFoundError` | Record not found in database | `record_type`, `record_id` |
| `DuplicateRecordError` | Duplicate primary key | `record_type`, `record_id` |
| `ConnectionError` | Database connection failure | `database_path`, `original_error` |

### 4.3 Module: `tests/test_storage.py`

#### Test Classes

| Test Class | Coverage Target |
|------------|-----------------|
| `TestSQLiteStorage` | Connection management, thread safety, basic operations |
| `TestTradeRepository` | CRUD operations, queries, edge cases |
| `TestPositionRepository` | Upsert, queries, edge cases |
| `TestStorageExceptions` | Exception hierarchy and attributes |
| `TestThreadSafety` | Concurrent access scenarios |

---

## 5. Data Models

### 5.1 Database Schema

#### `trades` Table

```sql
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity TEXT NOT NULL,  -- Stored as TEXT for Decimal precision
    price TEXT NOT NULL,     -- Stored as TEXT for Decimal precision
    timestamp TEXT NOT NULL, -- ISO 8601 format
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id);
```

#### `positions` Table

```sql
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    quantity TEXT NOT NULL,      -- Stored as TEXT for Decimal precision
    avg_price TEXT NOT NULL,     -- Stored as TEXT for Decimal precision
    unrealized_pnl TEXT NOT NULL DEFAULT '0',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Index for symbol lookups
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
```

### 5.2 Data Type Mapping

| Python Type | SQLite Type | Notes |
|-------------|-------------|-------|
| `str` | `TEXT` | Direct mapping |
| `Decimal` | `TEXT` | String representation preserves precision |
| `datetime` | `TEXT` | ISO 8601 format for portability |
| `TradeSide` | `TEXT` | Enum value string (`"BUY"` or `"SELL"`) |
| `int` | `INTEGER` | Direct mapping (for auto-increment IDs) |

### 5.3 Row-to-Model Conversion

```python
# Trade row tuple format
TradeRow = Tuple[int, str, str, str, str, str, str, str, str]
# (id, trade_id, symbol, side, quantity, price, timestamp, created_at, updated_at)

# Position row tuple format
PositionRow = Tuple[int, str, str, str, str, str, str]
# (id, symbol, quantity, avg_price, unrealized_pnl, created_at, updated_at)
```

---

## 6. API Contracts

### 6.1 StorageBackend Protocol

```python
from typing import Protocol, Optional, Any, List, Tuple
from pathlib import Path

class StorageBackend(Protocol):
    """Abstract storage backend protocol."""

    def connect(self, database_path: Optional[Path] = None) -> None:
        """Establish database connection."""
        ...

    def disconnect(self) -> None:
        """Close database connection."""
        ...

    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None
    ) -> List[Tuple[Any, ...]]:
        """Execute a query and return results."""
        ...

    def execute_many(
        self,
        query: str,
        params_list: List[Tuple[Any, ...]]
    ) -> int:
        """Execute a query with multiple parameter sets."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to database."""
        ...
```

### 6.2 SQLiteStorage Class

```python
from pathlib import Path
from typing import Optional, List, Tuple, Any
import threading

class SQLiteStorage:
    """Thread-safe SQLite storage implementation."""

    def __init__(
        self,
        database_path: Optional[Path] = None,
        *,
        check_same_thread: bool = False
    ) -> None:
        """
        Initialize SQLite storage.

        Args:
            database_path: Path to SQLite database file.
                          None or ":memory:" for in-memory database.
            check_same_thread: SQLite thread checking (disabled for pooling).
        """
        ...

    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        ...

    def connect(self) -> None:
        """Initialize database and create tables."""
        ...

    def disconnect(self) -> None:
        """Close all connections."""
        ...

    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None
    ) -> List[Tuple[Any, ...]]:
        """Execute query with optional parameters."""
        ...

    def execute_many(
        self,
        query: str,
        params_list: List[Tuple[Any, ...]]
    ) -> int:
        """Execute query with multiple parameter sets (batch insert)."""
        ...

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for transaction handling."""
        ...
```

### 6.3 TradeRepository Class

```python
from typing import Optional, List
from datetime import datetime
from trade_analytics.models import Trade

class TradeRepository:
    """Repository for trade CRUD operations."""

    def __init__(self, storage: SQLiteStorage) -> None:
        """Initialize with storage backend."""
        ...

    def create(self, trade: Trade) -> Trade:
        """
        Create a new trade record.

        Args:
            trade: Trade to persist. If trade_id is empty, generates UUID.

        Returns:
            Trade with populated trade_id.

        Raises:
            DuplicateRecordError: If trade_id already exists.
            StorageError: On database error.
        """
        ...

    def get_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Retrieve trade by trade_id.

        Args:
            trade_id: Unique trade identifier.

        Returns:
            Trade if found, None otherwise.
        """
        ...

    def get_all(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Trade]:
        """
        Retrieve all trades with pagination.

        Args:
            limit: Maximum number of trades to return.
            offset: Number of trades to skip.

        Returns:
            List of trades ordered by timestamp descending.
        """
        ...

    def query(
        self,
        *,
        symbol: Optional[str] = None,
        side: Optional[TradeSide] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Trade]:
        """
        Query trades with filters.

        Args:
            symbol: Filter by symbol.
            side: Filter by trade side.
            start_date: Filter trades on or after this date.
            end_date: Filter trades on or before this date.
            limit: Maximum results.
            offset: Skip first N results.

        Returns:
            List of matching trades.
        """
        ...

    def update(self, trade: Trade) -> Trade:
        """
        Update existing trade.

        Args:
            trade: Trade with updated values. trade_id must match existing record.

        Returns:
            Updated trade.

        Raises:
            RecordNotFoundError: If trade_id doesn't exist.
        """
        ...

    def delete(self, trade_id: str) -> bool:
        """
        Delete trade by trade_id.

        Args:
            trade_id: Trade to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def count(
        self,
        *,
        symbol: Optional[str] = None,
        side: Optional[TradeSide] = None
    ) -> int:
        """Count trades matching criteria."""
        ...
```

### 6.4 PositionRepository Class

```python
from typing import Optional, List
from trade_analytics.models import Position

class PositionRepository:
    """Repository for position query and update operations."""

    def __init__(self, storage: SQLiteStorage) -> None:
        """Initialize with storage backend."""
        ...

    def get_by_symbol(self, symbol: str) -> Optional[Position]:
        """
        Retrieve position by symbol.

        Args:
            symbol: Ticker symbol (case-insensitive).

        Returns:
            Position if found, None otherwise.
        """
        ...

    def get_all(self) -> List[Position]:
        """
        Retrieve all positions.

        Returns:
            List of all positions ordered by symbol.
        """
        ...

    def upsert(self, position: Position) -> Position:
        """
        Insert or update position.

        Uses SQLite UPSERT (INSERT OR REPLACE) semantics.

        Args:
            position: Position to persist.

        Returns:
            Persisted position.
        """
        ...

    def delete(self, symbol: str) -> bool:
        """
        Delete position by symbol.

        Args:
            symbol: Symbol to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def query(
        self,
        *,
        symbols: Optional[List[str]] = None,
        min_quantity: Optional[Decimal] = None,
        max_quantity: Optional[Decimal] = None,
        has_position: Optional[bool] = None
    ) -> List[Position]:
        """
        Query positions with filters.

        Args:
            symbols: Filter by list of symbols.
            min_quantity: Filter positions >= quantity.
            max_quantity: Filter positions <= quantity.
            has_position: If True, only non-zero positions; if False, only zero.

        Returns:
            List of matching positions.
        """
        ...

    def count(self) -> int:
        """Count all positions."""
        ...
```

---

## 7. Error Handling

### 7.1 Exception Hierarchy

```python
class StorageError(TradingError):
    """Base exception for all storage-related errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


class RecordNotFoundError(StorageError):
    """Raised when a requested record doesn't exist."""

    def __init__(
        self,
        message: str,
        record_type: Optional[str] = None,
        record_id: Optional[str] = None
    ) -> None:
        super().__init__(message, operation="read")
        self.record_type = record_type
        self.record_id = record_id


class DuplicateRecordError(StorageError):
    """Raised when attempting to create a duplicate record."""

    def __init__(
        self,
        message: str,
        record_type: Optional[str] = None,
        record_id: Optional[str] = None
    ) -> None:
        super().__init__(message, operation="create")
        self.record_type = record_type
        self.record_id = record_id


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails."""

    def __init__(
        self,
        message: str,
        database_path: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        super().__init__(message, operation="connect", original_error=original_error)
        self.database_path = database_path
```

### 7.2 Error Handling Strategy

| Scenario | Exception | Recovery Action |
|----------|-----------|-----------------|
| Database file not writable | `DatabaseConnectionError` | Check permissions, use different path |
| Duplicate trade_id | `DuplicateRecordError` | Generate new ID or update existing |
| Trade not found for update | `RecordNotFoundError` | Create new record or handle gracefully |
| SQL syntax error | `StorageError` | Log error, fix query (programming error) |
| Database locked | `StorageError` | Retry with backoff |
| Connection lost | `DatabaseConnectionError` | Reconnect and retry |

### 7.3 Transaction Error Handling

```python
@contextmanager
def transaction(self) -> Iterator[sqlite3.Connection]:
    """Context manager with automatic rollback on error."""
    conn = self.get_connection()
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise StorageError(
            f"Transaction failed: {e}",
            operation="transaction",
            original_error=e
        ) from e
```

---

## 8. Implementation Plan

### Phase 1: Storage Exceptions (15 min)

| Step | Task | File |
|------|------|------|
| 1.1 | Add `StorageError` base exception | `src/trade_analytics/exceptions.py` |
| 1.2 | Add `RecordNotFoundError` | `src/trade_analytics/exceptions.py` |
| 1.3 | Add `DuplicateRecordError` | `src/trade_analytics/exceptions.py` |
| 1.4 | Add `DatabaseConnectionError` | `src/trade_analytics/exceptions.py` |
| 1.5 | Update `__all__` exports | `src/trade_analytics/exceptions.py` |

### Phase 2: SQLiteStorage Core (30 min)

| Step | Task | File |
|------|------|------|
| 2.1 | Create `storage.py` with imports | `src/trade_analytics/storage.py` |
| 2.2 | Implement `SQLiteStorage.__init__()` with thread-local storage | `src/trade_analytics/storage.py` |
| 2.3 | Implement `get_connection()` with connection pooling | `src/trade_analytics/storage.py` |
| 2.4 | Implement `connect()` with schema initialization | `src/trade_analytics/storage.py` |
| 2.5 | Implement `disconnect()` | `src/trade_analytics/storage.py` |
| 2.6 | Implement `execute()` and `execute_many()` | `src/trade_analytics/storage.py` |
| 2.7 | Implement `transaction()` context manager | `src/trade_analytics/storage.py` |

### Phase 3: TradeRepository (30 min)

| Step | Task | File |
|------|------|------|
| 3.1 | Implement `TradeRepository.__init__()` | `src/trade_analytics/storage.py` |
| 3.2 | Implement `create()` with UUID generation | `src/trade_analytics/storage.py` |
| 3.3 | Implement `get_by_id()` | `src/trade_analytics/storage.py` |
| 3.4 | Implement `get_all()` with pagination | `src/trade_analytics/storage.py` |
| 3.5 | Implement `query()` with dynamic filters | `src/trade_analytics/storage.py` |
| 3.6 | Implement `update()` | `src/trade_analytics/storage.py` |
| 3.7 | Implement `delete()` | `src/trade_analytics/storage.py` |
| 3.8 | Implement `count()` | `src/trade_analytics/storage.py` |
| 3.9 | Add helper method `_row_to_trade()` | `src/trade_analytics/storage.py` |

### Phase 4: PositionRepository (20 min)

| Step | Task | File |
|------|------|------|
| 4.1 | Implement `PositionRepository.__init__()` | `src/trade_analytics/storage.py` |
| 4.2 | Implement `get_by_symbol()` | `src/trade_analytics/storage.py` |
| 4.3 | Implement `get_all()` | `src/trade_analytics/storage.py` |
| 4.4 | Implement `upsert()` | `src/trade_analytics/storage.py` |
| 4.5 | Implement `delete()` | `src/trade_analytics/storage.py` |
| 4.6 | Implement `query()` with filters | `src/trade_analytics/storage.py` |
| 4.7 | Add helper method `_row_to_position()` | `src/trade_analytics/storage.py` |

### Phase 5: Unit Tests (30 min)

| Step | Task | File |
|------|------|------|
| 5.1 | Create test fixtures for storage | `tests/conftest.py` |
| 5.2 | Write `TestSQLiteStorage` tests | `tests/test_storage.py` |
| 5.3 | Write `TestTradeRepository` CRUD tests | `tests/test_storage.py` |
| 5.4 | Write `TestTradeRepository` query tests | `tests/test_storage.py` |
| 5.5 | Write `TestPositionRepository` tests | `tests/test_storage.py` |
| 5.6 | Write `TestStorageExceptions` tests | `tests/test_storage.py` |
| 5.7 | Write `TestThreadSafety` concurrent access tests | `tests/test_storage.py` |
| 5.8 | Verify ≥95% coverage | - |

### Phase 6: Integration & Documentation (15 min)

| Step | Task | File |
|------|------|------|
| 6.1 | Update `__init__.py` with storage exports | `src/trade_analytics/__init__.py` |
| 6.2 | Add docstrings to all public methods | All files |
| 6.3 | Run full test suite and coverage | - |
| 6.4 | Create sample usage documentation | `src/trade_analytics/storage.py` |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Decimal precision loss** | Medium | High | Store as TEXT, use `Decimal(str)` for conversion; add precision tests |
| **Thread contention** | Medium | Medium | Use thread-local connections; add concurrent access tests |
| **Database corruption** | Low | High | Use transactions; implement backup strategy; WAL mode for durability |
| **SQLite file locking** | Medium | Medium | Use WAL mode; implement retry logic with exponential backoff |
| **Memory leaks from unclosed connections** | Low | Medium | Implement `__del__` cleanup; use context managers; add cleanup tests |
| **Query injection** | Low | Critical | Always use parameterized queries; never interpolate user input |
| **Timezone handling issues** | Medium | Medium | Store all timestamps in UTC ISO 8601; document timezone policy |
| **Breaking changes to models** | Low | High | Models are stable from CORE-001; add integration tests |

### Mitigation Details

**Decimal Precision:**
```python
# Store Decimal as string
cursor.execute(
    "INSERT INTO trades (..., quantity, price) VALUES (?, ?, ...)",
    (..., str(trade.quantity), str(trade.price), ...)
)

# Retrieve Decimal from string
quantity = Decimal(row[4])  # Preserves full precision
```

**Thread Safety:**
```python
# Thread-local connection storage
_local = threading.local()

def get_connection(self) -> sqlite3.Connection:
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False
        )
    return _local.connection
```

**WAL Mode for Concurrency:**
```python
def connect(self) -> None:
    conn = self.get_connection()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
```

---

## 10. Success Criteria

### Functional Requirements

| Criteria | Verification Method |
|----------|-------------------|
| SQLite database created with correct schema | Test creates DB and verifies table structure |
| Trades table stores all Trade fields | Round-trip test: create → retrieve → compare |
| Positions table stores all Position fields | Round-trip test: upsert → retrieve → compare |
| CRUD operations work for trades | Individual test for each operation |
| Query interface filters correctly | Tests with various filter combinations |
| Connection pooling works | Test with multiple threads |
| Thread safety verified | Concurrent read/write test |
| In-memory database works for tests | All tests use `:memory:` |

### Quality Requirements

| Criteria | Target | Verification |
|----------|--------|--------------|
| Test coverage | ≥ 95% | `pytest --cov=src/trade_analytics --cov-report=term-missing` |
| Type hints | 100% | `mypy src/trade_analytics/` passes |
| Docstrings | All public APIs | Manual review |
| Linting | No errors | `ruff check src/trade_analytics/` |

### Performance Requirements

| Operation | Target | Verification |
|-----------|--------|--------------|
| Single trade insert | < 10ms | Performance test |
| Batch insert (100 trades) | < 100ms | Performance test |
| Query by symbol | < 10ms | Performance test |
| Get all positions | < 50ms (1000 records) | Performance test |

### Test Cases (Minimum)

```
tests/test_storage.py:
  # SQLiteStorage tests
  ✓ test_create_in_memory_database
  ✓ test_create_file_database
  ✓ test_connect_creates_tables
  ✓ test_disconnect_closes_connections
  ✓ test_execute_returns_results
  ✓ test_execute_with_params
  ✓ test_transaction_commit
  ✓ test_transaction_rollback_on_error

  # TradeRepository tests
  ✓ test_create_trade
  ✓ test_create_trade_generates_id
  ✓ test_create_trade_duplicate_id_raises
  ✓ test_get_by_id_found
  ✓ test_get_by_id_not_found
  ✓ test_get_all_empty
  ✓ test_get_all_with_trades
  ✓ test_get_all_pagination
  ✓ test_query_by_symbol
  ✓ test_query_by_side
  ✓ test_query_by_date_range
  ✓ test_query_combined_filters
  ✓ test_update_trade
  ✓ test_update_trade_not_found_raises
  ✓ test_delete_trade
  ✓ test_delete_trade_not_found
  ✓ test_count_all
  ✓ test_count_with_filter

  # PositionRepository tests
  ✓ test_get_by_symbol_found
  ✓ test_get_by_symbol_not_found
  ✓ test_get_by_symbol_case_insensitive
  ✓ test_get_all_empty
  ✓ test_get_all_with_positions
  ✓ test_upsert_insert
  ✓ test_upsert_update
  ✓ test_delete_position
  ✓ test_delete_position_not_found
  ✓ test_query_by_symbols
  ✓ test_query_by_quantity_range
  ✓ test_query_has_position

  # Exception tests
  ✓ test_storage_error_attributes
  ✓ test_record_not_found_error_attributes
  ✓ test_duplicate_record_error_attributes
  ✓ test_database_connection_error_attributes
  ✓ test_exception_inheritance

  # Thread safety tests
  ✓ test_concurrent_reads
  ✓ test_concurrent_writes
  ✓ test_concurrent_mixed_operations
```

---

## Appendix A: File Structure After Implementation

```
trade-analytics-mini/
├── src/
│   └── trade_analytics/
│       ├── __init__.py          # Updated with storage exports
│       ├── calculator.py        # Unchanged
│       ├── exceptions.py        # UPDATED: Added storage exceptions
│       ├── models.py            # Unchanged
│       └── storage.py           # NEW: SQLite storage implementation
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # UPDATED: Added storage fixtures
│   ├── test_calculator.py       # Unchanged
│   ├── test_models.py           # Unchanged
│   └── test_storage.py          # NEW: Storage tests
└── ...
```

## Appendix B: Example Usage

```python
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal

from trade_analytics import Trade, TradeSide, Position
from trade_analytics.storage import (
    SQLiteStorage,
    TradeRepository,
    PositionRepository,
)

# Create storage (file-based)
storage = SQLiteStorage(Path("./trades.db"))
storage.connect()

# Initialize repositories
trade_repo = TradeRepository(storage)
position_repo = PositionRepository(storage)

# Create a trade
trade = Trade(
    symbol="AAPL",
    side=TradeSide.BUY,
    quantity=Decimal("100"),
    price=Decimal("150.50"),
    timestamp=datetime.now(timezone.utc),
)
saved_trade = trade_repo.create(trade)
print(f"Created trade: {saved_trade.trade_id}")

# Query trades
aapl_trades = trade_repo.query(symbol="AAPL", limit=10)
for t in aapl_trades:
    print(f"  {t.timestamp}: {t.side.value} {t.quantity} @ {t.price}")

# Upsert position
position = Position(
    symbol="AAPL",
    quantity=Decimal("100"),
    avg_price=Decimal("150.50"),
)
position_repo.upsert(position)

# Get all positions
positions = position_repo.get_all()
for p in positions:
    print(f"{p.symbol}: {p.quantity} shares @ {p.avg_price}")

# Cleanup
storage.disconnect()
```

## Appendix C: SQL Schema Reference

```sql
-- Full schema for reference

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity TEXT NOT NULL,
    price TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    quantity TEXT NOT NULL,
    avg_price TEXT NOT NULL,
    unrealized_pnl TEXT NOT NULL DEFAULT '0',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- Trigger to update updated_at on trade update
CREATE TRIGGER IF NOT EXISTS trades_updated_at
AFTER UPDATE ON trades
BEGIN
    UPDATE trades SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- Trigger to update updated_at on position update
CREATE TRIGGER IF NOT EXISTS positions_updated_at
AFTER UPDATE ON positions
BEGIN
    UPDATE positions SET updated_at = datetime('now') WHERE id = NEW.id;
END;
```

---

**Document Version:** 1.0
**Last Updated:** 2024-11-26
**Next Review:** Before implementation begins
