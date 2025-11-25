"""SQLite-based storage layer for trades and positions.

This module provides persistent storage for trades and positions using SQLite.
It includes thread-safe connection pooling, CRUD operations for trades,
and query interface for positions.

Example usage:
    from pathlib import Path
    from trade_analytics.storage import SQLiteStorage, TradeRepository, PositionRepository

    storage = SQLiteStorage(Path("./trades.db"))
    storage.connect()

    trade_repo = TradeRepository(storage)
    position_repo = PositionRepository(storage)

    # Create a trade
    trade = trade_repo.create(trade)

    # Query positions
    positions = position_repo.get_all()

    storage.disconnect()
"""

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

from .exceptions import (
    DatabaseConnectionError,
    DuplicateRecordError,
    RecordNotFoundError,
    StorageError,
)
from .models import Position, Trade, TradeSide

__all__ = [
    "SQLiteStorage",
    "TradeRepository",
    "PositionRepository",
]


# SQL Schema definitions
TRADES_TABLE_SQL = """
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
)
"""

TRADES_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id)",
]

POSITIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    quantity TEXT NOT NULL,
    avg_price TEXT NOT NULL,
    unrealized_pnl TEXT NOT NULL DEFAULT '0',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

POSITIONS_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
]

TRADES_UPDATED_AT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trades_updated_at
AFTER UPDATE ON trades
BEGIN
    UPDATE trades SET updated_at = datetime('now') WHERE id = NEW.id;
END
"""

POSITIONS_UPDATED_AT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS positions_updated_at
AFTER UPDATE ON positions
BEGIN
    UPDATE positions SET updated_at = datetime('now') WHERE id = NEW.id;
END
"""


class SQLiteStorage:
    """Thread-safe SQLite storage implementation.

    Uses thread-local storage for connection pooling to ensure each thread
    gets its own database connection, providing thread safety without
    explicit locking.

    Attributes:
        database_path: Path to the SQLite database file, or ":memory:" for in-memory.
    """

    def __init__(
        self,
        database_path: Optional[Union[Path, str]] = None,
        *,
        check_same_thread: bool = False,
    ) -> None:
        """Initialize SQLite storage.

        Args:
            database_path: Path to SQLite database file.
                          None or ":memory:" for in-memory database.
            check_same_thread: SQLite thread checking (disabled for pooling).
        """
        if database_path is None:
            self._database_path = ":memory:"
        elif isinstance(database_path, Path):
            self._database_path = str(database_path)
        else:
            self._database_path = database_path

        self._check_same_thread = check_same_thread
        self._local = threading.local()
        self._connected = False
        self._lock = threading.Lock()

    @property
    def database_path(self) -> str:
        """Get the database path."""
        return self._database_path

    def is_connected(self) -> bool:
        """Check if storage is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._connected

    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        Creates a new connection for the current thread if one doesn't exist.

        Returns:
            SQLite connection for the current thread.

        Raises:
            DatabaseConnectionError: If connection cannot be established.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            try:
                self._local.connection = sqlite3.connect(
                    self._database_path,
                    check_same_thread=self._check_same_thread,
                )
                # Enable foreign keys
                self._local.connection.execute("PRAGMA foreign_keys = ON")
            except sqlite3.Error as e:
                raise DatabaseConnectionError(
                    f"Failed to connect to database: {e}",
                    database_path=self._database_path,
                    original_error=e,
                ) from e

        return self._local.connection

    def connect(self) -> None:
        """Initialize database and create tables.

        Creates the necessary tables and indexes if they don't exist.
        Sets up WAL mode for better concurrent access.

        Raises:
            DatabaseConnectionError: If connection or initialization fails.
        """
        try:
            conn = self.get_connection()

            # Use WAL mode for better concurrency (only for file-based databases)
            if self._database_path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

            # Create tables
            conn.execute(TRADES_TABLE_SQL)
            conn.execute(POSITIONS_TABLE_SQL)

            # Create indexes
            for index_sql in TRADES_INDEXES_SQL:
                conn.execute(index_sql)
            for index_sql in POSITIONS_INDEXES_SQL:
                conn.execute(index_sql)

            # Create triggers
            conn.execute(TRADES_UPDATED_AT_TRIGGER)
            conn.execute(POSITIONS_UPDATED_AT_TRIGGER)

            conn.commit()
            self._connected = True

        except sqlite3.Error as e:
            raise DatabaseConnectionError(
                f"Failed to initialize database: {e}",
                database_path=self._database_path,
                original_error=e,
            ) from e

    def disconnect(self) -> None:
        """Close all connections.

        Closes the connection for the current thread if one exists.
        """
        if hasattr(self._local, "connection") and self._local.connection is not None:
            try:
                self._local.connection.close()
            except sqlite3.Error:
                pass  # Ignore errors during cleanup
            self._local.connection = None

        self._connected = False

    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> List[Tuple[Any, ...]]:
        """Execute query with optional parameters.

        Args:
            query: SQL query to execute.
            params: Optional tuple of query parameters.

        Returns:
            List of result tuples.

        Raises:
            StorageError: If query execution fails.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            conn.commit()
            return results
        except sqlite3.Error as e:
            raise StorageError(
                f"Query execution failed: {e}",
                operation="execute",
                original_error=e,
            ) from e

    def execute_many(
        self,
        query: str,
        params_list: List[Tuple[Any, ...]],
    ) -> int:
        """Execute query with multiple parameter sets (batch insert).

        Args:
            query: SQL query to execute.
            params_list: List of parameter tuples.

        Returns:
            Number of rows affected.

        Raises:
            StorageError: If query execution fails.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            raise StorageError(
                f"Batch execution failed: {e}",
                operation="execute_many",
                original_error=e,
            ) from e

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for transaction handling.

        Automatically commits on success or rolls back on error.

        Yields:
            SQLite connection for the transaction.

        Raises:
            StorageError: If transaction fails.
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise StorageError(
                f"Transaction failed: {e}",
                operation="transaction",
                original_error=e,
            ) from e


class TradeRepository:
    """Repository for trade CRUD operations.

    Provides methods to create, read, update, and delete trades in the database.
    Also supports querying trades with various filters.
    """

    def __init__(self, storage: SQLiteStorage) -> None:
        """Initialize with storage backend.

        Args:
            storage: SQLiteStorage instance to use for database operations.
        """
        self._storage = storage

    def _row_to_trade(self, row: Tuple[Any, ...]) -> Trade:
        """Convert database row to Trade model.

        Args:
            row: Database row tuple (id, trade_id, symbol, side, quantity,
                 price, timestamp, created_at, updated_at).

        Returns:
            Trade instance.
        """
        return Trade(
            symbol=row[2],
            side=TradeSide(row[3]),
            quantity=Decimal(row[4]),
            price=Decimal(row[5]),
            timestamp=datetime.fromisoformat(row[6]),
            trade_id=row[1],
        )

    def create(self, trade: Trade) -> Trade:
        """Create a new trade record.

        If trade_id is empty, generates a UUID.

        Args:
            trade: Trade to persist.

        Returns:
            Trade with populated trade_id.

        Raises:
            DuplicateRecordError: If trade_id already exists.
            StorageError: On database error.
        """
        trade_id = trade.trade_id if trade.trade_id else str(uuid.uuid4())

        query = """
            INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            trade_id,
            trade.symbol,
            trade.side.value,
            str(trade.quantity),
            str(trade.price),
            trade.timestamp.isoformat(),
        )

        try:
            self._storage.execute(query, params)
        except StorageError as e:
            if e.original_error and "UNIQUE constraint failed" in str(e.original_error):
                raise DuplicateRecordError(
                    f"Trade with id '{trade_id}' already exists",
                    record_type="Trade",
                    record_id=trade_id,
                ) from e
            raise

        # Return trade with the generated/used trade_id
        if not trade.trade_id:
            # Need to create a new Trade since it's frozen
            return Trade(
                symbol=trade.symbol,
                side=trade.side,
                quantity=trade.quantity,
                price=trade.price,
                timestamp=trade.timestamp,
                trade_id=trade_id,
            )
        return trade

    def get_by_id(self, trade_id: str) -> Optional[Trade]:
        """Retrieve trade by trade_id.

        Args:
            trade_id: Unique trade identifier.

        Returns:
            Trade if found, None otherwise.
        """
        query = "SELECT * FROM trades WHERE trade_id = ?"
        results = self._storage.execute(query, (trade_id,))

        if not results:
            return None

        return self._row_to_trade(results[0])

    def get_all(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Trade]:
        """Retrieve all trades with pagination.

        Args:
            limit: Maximum number of trades to return.
            offset: Number of trades to skip.

        Returns:
            List of trades ordered by timestamp descending.
        """
        query = "SELECT * FROM trades ORDER BY timestamp DESC"
        params: List[Any] = []

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        if offset > 0:
            if limit is None:
                query += " LIMIT -1"
            query += " OFFSET ?"
            params.append(offset)

        results = self._storage.execute(query, tuple(params) if params else None)
        return [self._row_to_trade(row) for row in results]

    def query(
        self,
        *,
        symbol: Optional[str] = None,
        side: Optional[TradeSide] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Trade]:
        """Query trades with filters.

        Args:
            symbol: Filter by symbol (case-insensitive).
            side: Filter by trade side.
            start_date: Filter trades on or after this date.
            end_date: Filter trades on or before this date.
            limit: Maximum results.
            offset: Skip first N results.

        Returns:
            List of matching trades.
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []

        if symbol is not None:
            query += " AND UPPER(symbol) = UPPER(?)"
            params.append(symbol)

        if side is not None:
            query += " AND side = ?"
            params.append(side.value)

        if start_date is not None:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date is not None:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        if offset > 0:
            if limit is None:
                query += " LIMIT -1"
            query += " OFFSET ?"
            params.append(offset)

        results = self._storage.execute(query, tuple(params) if params else None)
        return [self._row_to_trade(row) for row in results]

    def update(self, trade: Trade) -> Trade:
        """Update existing trade.

        Args:
            trade: Trade with updated values. trade_id must match existing record.

        Returns:
            Updated trade.

        Raises:
            RecordNotFoundError: If trade_id doesn't exist.
            StorageError: On database error.
        """
        if not trade.trade_id:
            raise RecordNotFoundError(
                "Cannot update trade without trade_id",
                record_type="Trade",
                record_id="",
            )

        # Check if trade exists
        existing = self.get_by_id(trade.trade_id)
        if existing is None:
            raise RecordNotFoundError(
                f"Trade with id '{trade.trade_id}' not found",
                record_type="Trade",
                record_id=trade.trade_id,
            )

        query = """
            UPDATE trades
            SET symbol = ?, side = ?, quantity = ?, price = ?, timestamp = ?
            WHERE trade_id = ?
        """
        params = (
            trade.symbol,
            trade.side.value,
            str(trade.quantity),
            str(trade.price),
            trade.timestamp.isoformat(),
            trade.trade_id,
        )

        self._storage.execute(query, params)
        return trade

    def delete(self, trade_id: str) -> bool:
        """Delete trade by trade_id.

        Args:
            trade_id: Trade to delete.

        Returns:
            True if deleted, False if not found.
        """
        query = "DELETE FROM trades WHERE trade_id = ?"
        conn = self._storage.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (trade_id,))
        conn.commit()
        return cursor.rowcount > 0

    def count(
        self,
        *,
        symbol: Optional[str] = None,
        side: Optional[TradeSide] = None,
    ) -> int:
        """Count trades matching criteria.

        Args:
            symbol: Filter by symbol (case-insensitive).
            side: Filter by trade side.

        Returns:
            Number of matching trades.
        """
        query = "SELECT COUNT(*) FROM trades WHERE 1=1"
        params: List[Any] = []

        if symbol is not None:
            query += " AND UPPER(symbol) = UPPER(?)"
            params.append(symbol)

        if side is not None:
            query += " AND side = ?"
            params.append(side.value)

        results = self._storage.execute(query, tuple(params) if params else None)
        return results[0][0] if results else 0


class PositionRepository:
    """Repository for position query and update operations.

    Provides methods to query, upsert, and delete positions in the database.
    """

    def __init__(self, storage: SQLiteStorage) -> None:
        """Initialize with storage backend.

        Args:
            storage: SQLiteStorage instance to use for database operations.
        """
        self._storage = storage

    def _row_to_position(self, row: Tuple[Any, ...]) -> Position:
        """Convert database row to Position model.

        Args:
            row: Database row tuple (id, symbol, quantity, avg_price,
                 unrealized_pnl, created_at, updated_at).

        Returns:
            Position instance.
        """
        return Position(
            symbol=row[1],
            quantity=Decimal(row[2]),
            avg_price=Decimal(row[3]),
            unrealized_pnl=Decimal(row[4]),
        )

    def get_by_symbol(self, symbol: str) -> Optional[Position]:
        """Retrieve position by symbol.

        Args:
            symbol: Ticker symbol (case-insensitive).

        Returns:
            Position if found, None otherwise.
        """
        query = "SELECT * FROM positions WHERE UPPER(symbol) = UPPER(?)"
        results = self._storage.execute(query, (symbol,))

        if not results:
            return None

        return self._row_to_position(results[0])

    def get_all(self) -> List[Position]:
        """Retrieve all positions.

        Returns:
            List of all positions ordered by symbol.
        """
        query = "SELECT * FROM positions ORDER BY symbol"
        results = self._storage.execute(query)
        return [self._row_to_position(row) for row in results]

    def upsert(self, position: Position) -> Position:
        """Insert or update position.

        Uses SQLite UPSERT semantics - inserts if the symbol doesn't exist,
        updates if it does.

        Args:
            position: Position to persist.

        Returns:
            Persisted position.

        Raises:
            StorageError: On database error.
        """
        query = """
            INSERT INTO positions (symbol, quantity, avg_price, unrealized_pnl)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity = excluded.quantity,
                avg_price = excluded.avg_price,
                unrealized_pnl = excluded.unrealized_pnl,
                updated_at = datetime('now')
        """
        params = (
            position.symbol.upper(),
            str(position.quantity),
            str(position.avg_price),
            str(position.unrealized_pnl),
        )

        self._storage.execute(query, params)
        return position

    def delete(self, symbol: str) -> bool:
        """Delete position by symbol.

        Args:
            symbol: Symbol to delete (case-insensitive).

        Returns:
            True if deleted, False if not found.
        """
        query = "DELETE FROM positions WHERE UPPER(symbol) = UPPER(?)"
        conn = self._storage.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (symbol,))
        conn.commit()
        return cursor.rowcount > 0

    def query(
        self,
        *,
        symbols: Optional[List[str]] = None,
        min_quantity: Optional[Decimal] = None,
        max_quantity: Optional[Decimal] = None,
        has_position: Optional[bool] = None,
    ) -> List[Position]:
        """Query positions with filters.

        Args:
            symbols: Filter by list of symbols (case-insensitive).
            min_quantity: Filter positions >= quantity.
            max_quantity: Filter positions <= quantity.
            has_position: If True, only non-zero positions; if False, only zero.

        Returns:
            List of matching positions.
        """
        query = "SELECT * FROM positions WHERE 1=1"
        params: List[Any] = []

        if symbols is not None and symbols:
            placeholders = ", ".join("?" for _ in symbols)
            query += f" AND UPPER(symbol) IN ({placeholders})"
            params.extend(s.upper() for s in symbols)

        if min_quantity is not None:
            query += " AND CAST(quantity AS REAL) >= ?"
            params.append(float(min_quantity))

        if max_quantity is not None:
            query += " AND CAST(quantity AS REAL) <= ?"
            params.append(float(max_quantity))

        if has_position is not None:
            if has_position:
                query += " AND CAST(quantity AS REAL) != 0"
            else:
                query += " AND CAST(quantity AS REAL) = 0"

        query += " ORDER BY symbol"

        results = self._storage.execute(query, tuple(params) if params else None)
        return [self._row_to_position(row) for row in results]

    def count(self) -> int:
        """Count all positions.

        Returns:
            Total number of positions.
        """
        query = "SELECT COUNT(*) FROM positions"
        results = self._storage.execute(query)
        return results[0][0] if results else 0
