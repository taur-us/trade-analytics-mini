"""SQLite-based storage layer for trades and positions.

This module provides persistent storage for trading data using SQLite.
Supports both file-based and in-memory databases for production and testing.

Example usage:
    from trade_analytics import Trade, TradeSide, Position
    from trade_analytics.storage import TradeStore
    from decimal import Decimal
    from datetime import datetime, timezone

    # Create file-based store
    store = TradeStore("trades.db")

    # Save trades using context manager
    with store:
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
        )
        store.save_trade(trade)
        trades = store.get_trades("AAPL")

    # Use in-memory database for testing
    test_store = TradeStore(":memory:")
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Generator, List, Optional, Type
from types import TracebackType

from .exceptions import StorageError
from .models import Position, Trade, TradeSide


__all__ = ["TradeStore"]


# SQL statements for table creation
CREATE_TRADES_TABLE = """
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
"""

CREATE_TRADES_INDICES = """
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp);
"""

CREATE_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    quantity TEXT NOT NULL,
    avg_price TEXT NOT NULL,
    unrealized_pnl TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


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
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None

        # Initialize database schema (only for file-based databases)
        # For in-memory databases, tables will be created when context manager is used
        if db_path != ":memory:":
            with self._get_connection() as conn:
                self._create_tables(conn)

    def __enter__(self) -> "TradeStore":
        """Enter context manager, establish database connection.

        Returns:
            Self for use in with statement.
        """
        if self._connection is not None:
            raise StorageError(
                "Connection already open",
                operation="__enter__",
            )

        self._connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._connection.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access (not supported for in-memory)
        if self.db_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")

        # For in-memory databases, create tables now
        if self.db_path == ":memory:":
            self._create_tables(self._connection)

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit context manager, close database connection.

        Commits transaction if no exception, rolls back otherwise.
        """
        if self._connection is None:
            return

        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            self._connection.close()
            self._connection = None

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup.

        If a connection is already open (via context manager), use it.
        Otherwise, create a temporary connection.

        Yields:
            SQLite connection configured for the store.
        """
        if self._connection is not None:
            # Use existing connection from context manager
            yield self._connection
        else:
            # Create temporary connection
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema.

        Args:
            conn: SQLite connection to use.
        """
        cursor = conn.cursor()
        cursor.execute(CREATE_TRADES_TABLE)
        cursor.executescript(CREATE_TRADES_INDICES)
        cursor.execute(CREATE_POSITIONS_TABLE)
        conn.commit()

    def save_trade(self, trade: Trade) -> None:
        """Save a trade to the database.

        If trade.trade_id is empty, generates a UUID.
        If trade_id already exists, updates the existing record.

        Args:
            trade: Trade object to persist.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            # Generate UUID if trade_id is empty
            trade_id = trade.trade_id if trade.trade_id else str(uuid.uuid4())

            # Convert timestamp to UTC ISO-8601 format
            timestamp_str = trade.timestamp.astimezone(timezone.utc).isoformat()

            # UPSERT query
            sql = """
            INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                symbol=excluded.symbol,
                side=excluded.side,
                quantity=excluded.quantity,
                price=excluded.price,
                timestamp=excluded.timestamp
            """

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    sql,
                    (
                        trade_id,
                        trade.symbol,
                        trade.side.value,
                        str(trade.quantity),
                        str(trade.price),
                        timestamp_str,
                    ),
                )
        except sqlite3.Error as e:
            raise StorageError(
                f"Failed to save trade: {e}",
                operation="save_trade",
                original_error=e,
            ) from e

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
            StorageError: If database query fails.
        """
        try:
            # Build query with optional filters
            sql = "SELECT * FROM trades WHERE 1=1"
            params = []

            if symbol is not None:
                sql += " AND UPPER(symbol) = UPPER(?)"
                params.append(symbol)

            if start_date is not None:
                sql += " AND timestamp >= ?"
                params.append(start_date.astimezone(timezone.utc).isoformat())

            if end_date is not None:
                sql += " AND timestamp <= ?"
                params.append(end_date.astimezone(timezone.utc).isoformat())

            sql += " ORDER BY timestamp DESC"

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()

            # Convert rows to Trade objects
            trades = []
            for row in rows:
                trades.append(self._row_to_trade(row))

            return trades
        except sqlite3.Error as e:
            raise StorageError(
                f"Failed to retrieve trades: {e}",
                operation="get_trades",
                original_error=e,
            ) from e

    def save_position(self, position: Position) -> None:
        """Save or update a position in the database.

        Uses UPSERT semantics - inserts new or updates existing by symbol.

        Args:
            position: Position object to persist.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            sql = """
            INSERT INTO positions (symbol, quantity, avg_price, unrealized_pnl)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity=excluded.quantity,
                avg_price=excluded.avg_price,
                unrealized_pnl=excluded.unrealized_pnl,
                updated_at=CURRENT_TIMESTAMP
            """

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    sql,
                    (
                        position.symbol,
                        str(position.quantity),
                        str(position.avg_price),
                        str(position.unrealized_pnl),
                    ),
                )
        except sqlite3.Error as e:
            raise StorageError(
                f"Failed to save position: {e}",
                operation="save_position",
                original_error=e,
            ) from e

    def get_positions(self) -> List[Position]:
        """Retrieve all positions.

        Returns:
            List of all Position objects, ordered by symbol.
            Empty list if no positions exist.

        Raises:
            StorageError: If database query fails.
        """
        try:
            sql = "SELECT * FROM positions ORDER BY symbol"

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()

            # Convert rows to Position objects
            positions = []
            for row in rows:
                positions.append(self._row_to_position(row))

            return positions
        except sqlite3.Error as e:
            raise StorageError(
                f"Failed to retrieve positions: {e}",
                operation="get_positions",
                original_error=e,
            ) from e

    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade by ID.

        Args:
            trade_id: Unique identifier of trade to delete.

        Returns:
            True if trade was deleted, False if not found.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            sql = "DELETE FROM trades WHERE trade_id = ?"

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (trade_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(
                f"Failed to delete trade: {e}",
                operation="delete_trade",
                original_error=e,
            ) from e

    def delete_position(self, symbol: str) -> bool:
        """Delete a position by symbol.

        Args:
            symbol: Ticker symbol of position to delete.

        Returns:
            True if position was deleted, False if not found.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            sql = "DELETE FROM positions WHERE symbol = ?"

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (symbol,))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(
                f"Failed to delete position: {e}",
                operation="delete_position",
                original_error=e,
            ) from e

    def clear_all(self) -> None:
        """Delete all trades and positions.

        WARNING: This operation is irreversible.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM trades")
                cursor.execute("DELETE FROM positions")
        except sqlite3.Error as e:
            raise StorageError(
                f"Failed to clear all data: {e}",
                operation="clear_all",
                original_error=e,
            ) from e

    def _row_to_trade(self, row: sqlite3.Row) -> Trade:
        """Convert database row to Trade object.

        Args:
            row: SQLite row from trades table.

        Returns:
            Trade object.
        """
        return Trade(
            trade_id=row["trade_id"],
            symbol=row["symbol"],
            side=TradeSide(row["side"]),
            quantity=Decimal(row["quantity"]),
            price=Decimal(row["price"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        """Convert database row to Position object.

        Args:
            row: SQLite row from positions table.

        Returns:
            Position object.
        """
        return Position(
            symbol=row["symbol"],
            quantity=Decimal(row["quantity"]),
            avg_price=Decimal(row["avg_price"]),
            unrealized_pnl=Decimal(row["unrealized_pnl"]),
        )
