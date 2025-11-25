"""Tests for the SQLite storage layer.

This module provides comprehensive tests for:
- SQLiteStorage: Connection management, thread safety, basic operations
- TradeRepository: CRUD operations, queries, edge cases
- PositionRepository: Upsert, queries, edge cases
- Storage exceptions: Exception hierarchy and attributes
- Thread safety: Concurrent access scenarios
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from trade_analytics import Position, Trade, TradeSide
from trade_analytics.exceptions import (
    DatabaseConnectionError,
    DuplicateRecordError,
    RecordNotFoundError,
    StorageError,
    TradingError,
)
from trade_analytics.storage import PositionRepository, SQLiteStorage, TradeRepository


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage() -> SQLiteStorage:
    """Create an in-memory SQLite storage for testing."""
    storage = SQLiteStorage()
    storage.connect()
    yield storage
    storage.disconnect()


@pytest.fixture
def trade_repo(storage: SQLiteStorage) -> TradeRepository:
    """Create a TradeRepository with test storage."""
    return TradeRepository(storage)


@pytest.fixture
def position_repo(storage: SQLiteStorage) -> PositionRepository:
    """Create a PositionRepository with test storage."""
    return PositionRepository(storage)


@pytest.fixture
def sample_trade() -> Trade:
    """Create a sample trade for testing."""
    return Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.50"),
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        trade_id="T001",
    )


@pytest.fixture
def sample_position() -> Position:
    """Create a sample position for testing."""
    return Position(
        symbol="AAPL",
        quantity=Decimal("100"),
        avg_price=Decimal("150.50"),
        unrealized_pnl=Decimal("250.00"),
    )


# ============================================================================
# SQLiteStorage Tests
# ============================================================================


class TestSQLiteStorage:
    """Tests for SQLiteStorage class."""

    def test_create_in_memory_database(self) -> None:
        """Test creating an in-memory database."""
        storage = SQLiteStorage()
        assert storage.database_path == ":memory:"
        assert not storage.is_connected()

        storage.connect()
        assert storage.is_connected()

        storage.disconnect()
        assert not storage.is_connected()

    def test_create_file_database(self) -> None:
        """Test creating a file-based database."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = SQLiteStorage(db_path)

            assert storage.database_path == str(db_path)
            storage.connect()

            # Verify file was created
            assert db_path.exists()

            storage.disconnect()

    def test_connect_creates_tables(self, storage: SQLiteStorage) -> None:
        """Test that connect() creates the required tables."""
        # Check trades table exists
        result = storage.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        assert len(result) == 1
        assert result[0][0] == "trades"

        # Check positions table exists
        result = storage.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='positions'"
        )
        assert len(result) == 1
        assert result[0][0] == "positions"

    def test_disconnect_closes_connections(self) -> None:
        """Test that disconnect() properly closes connections."""
        storage = SQLiteStorage()
        storage.connect()

        # Execute a query to ensure connection is established
        storage.execute("SELECT 1")

        storage.disconnect()
        assert not storage.is_connected()

    def test_execute_returns_results(self, storage: SQLiteStorage) -> None:
        """Test that execute() returns query results."""
        results = storage.execute("SELECT 1, 'hello', 3.14")
        assert len(results) == 1
        assert results[0] == (1, "hello", 3.14)

    def test_execute_with_params(self, storage: SQLiteStorage) -> None:
        """Test execute() with parameterized query."""
        storage.execute(
            "INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("T001", "AAPL", "BUY", "100", "150.50", "2024-01-15T10:30:00+00:00"),
        )

        results = storage.execute("SELECT trade_id, symbol FROM trades WHERE trade_id = ?", ("T001",))
        assert len(results) == 1
        assert results[0][0] == "T001"
        assert results[0][1] == "AAPL"

    def test_execute_many(self, storage: SQLiteStorage) -> None:
        """Test batch execution with execute_many()."""
        params_list = [
            ("T001", "AAPL", "BUY", "100", "150.50", "2024-01-15T10:30:00+00:00"),
            ("T002", "GOOGL", "SELL", "50", "140.00", "2024-01-15T11:00:00+00:00"),
            ("T003", "MSFT", "BUY", "200", "380.00", "2024-01-15T11:30:00+00:00"),
        ]

        count = storage.execute_many(
            "INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            params_list,
        )

        assert count == 3

        results = storage.execute("SELECT COUNT(*) FROM trades")
        assert results[0][0] == 3

    def test_transaction_commit(self, storage: SQLiteStorage) -> None:
        """Test that transaction commits on success."""
        with storage.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("T001", "AAPL", "BUY", "100", "150.50", "2024-01-15T10:30:00+00:00"),
            )

        # Verify data was committed
        results = storage.execute("SELECT COUNT(*) FROM trades")
        assert results[0][0] == 1

    def test_transaction_rollback_on_error(self, storage: SQLiteStorage) -> None:
        """Test that transaction rolls back on error."""
        try:
            with storage.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("T001", "AAPL", "BUY", "100", "150.50", "2024-01-15T10:30:00+00:00"),
                )
                # This should cause a constraint violation and trigger rollback
                cursor.execute(
                    "INSERT INTO trades (trade_id, symbol, side, quantity, price, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("T001", "GOOGL", "SELL", "50", "140.00", "2024-01-15T11:00:00+00:00"),
                )
        except StorageError:
            pass

        # Verify rollback occurred - no data should be present
        results = storage.execute("SELECT COUNT(*) FROM trades")
        assert results[0][0] == 0


# ============================================================================
# TradeRepository Tests
# ============================================================================


class TestTradeRepository:
    """Tests for TradeRepository class."""

    def test_create_trade(self, trade_repo: TradeRepository, sample_trade: Trade) -> None:
        """Test creating a trade."""
        created = trade_repo.create(sample_trade)

        assert created.trade_id == sample_trade.trade_id
        assert created.symbol == sample_trade.symbol
        assert created.side == sample_trade.side
        assert created.quantity == sample_trade.quantity
        assert created.price == sample_trade.price

    def test_create_trade_generates_id(self, trade_repo: TradeRepository) -> None:
        """Test that create() generates trade_id if not provided."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
            trade_id="",  # Empty trade_id
        )

        created = trade_repo.create(trade)
        assert created.trade_id  # Should have generated UUID
        assert len(created.trade_id) == 36  # UUID format

    def test_create_trade_duplicate_id_raises(
        self, trade_repo: TradeRepository, sample_trade: Trade
    ) -> None:
        """Test that duplicate trade_id raises DuplicateRecordError."""
        trade_repo.create(sample_trade)

        with pytest.raises(DuplicateRecordError) as exc_info:
            trade_repo.create(sample_trade)

        assert exc_info.value.record_type == "Trade"
        assert exc_info.value.record_id == sample_trade.trade_id

    def test_get_by_id_found(self, trade_repo: TradeRepository, sample_trade: Trade) -> None:
        """Test retrieving a trade by ID."""
        trade_repo.create(sample_trade)

        found = trade_repo.get_by_id(sample_trade.trade_id)
        assert found is not None
        assert found.trade_id == sample_trade.trade_id
        assert found.symbol == sample_trade.symbol

    def test_get_by_id_not_found(self, trade_repo: TradeRepository) -> None:
        """Test that get_by_id returns None for non-existent trade."""
        found = trade_repo.get_by_id("nonexistent")
        assert found is None

    def test_get_all_empty(self, trade_repo: TradeRepository) -> None:
        """Test get_all with no trades."""
        trades = trade_repo.get_all()
        assert trades == []

    def test_get_all_with_trades(self, trade_repo: TradeRepository) -> None:
        """Test get_all returns all trades."""
        trades_data = [
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
                trade_id="T001",
            ),
            Trade(
                symbol="GOOGL",
                side=TradeSide.SELL,
                quantity=Decimal("50"),
                price=Decimal("140.00"),
                timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
                trade_id="T002",
            ),
        ]

        for trade in trades_data:
            trade_repo.create(trade)

        result = trade_repo.get_all()
        assert len(result) == 2

    def test_get_all_pagination(self, trade_repo: TradeRepository) -> None:
        """Test get_all with pagination."""
        # Create 5 trades
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            trade = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=base_time + timedelta(hours=i),
                trade_id=f"T{i:03d}",
            )
            trade_repo.create(trade)

        # Test limit
        result = trade_repo.get_all(limit=2)
        assert len(result) == 2

        # Test offset
        result = trade_repo.get_all(limit=2, offset=2)
        assert len(result) == 2

        # Test offset beyond data
        result = trade_repo.get_all(offset=10)
        assert len(result) == 0

    def test_query_by_symbol(self, trade_repo: TradeRepository) -> None:
        """Test querying trades by symbol."""
        trade_repo.create(
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
                trade_id="T001",
            )
        )
        trade_repo.create(
            Trade(
                symbol="GOOGL",
                side=TradeSide.SELL,
                quantity=Decimal("50"),
                price=Decimal("140.00"),
                timestamp=datetime.now(timezone.utc),
                trade_id="T002",
            )
        )

        result = trade_repo.query(symbol="AAPL")
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

        # Test case insensitivity
        result = trade_repo.query(symbol="aapl")
        assert len(result) == 1

    def test_query_by_side(self, trade_repo: TradeRepository) -> None:
        """Test querying trades by side."""
        trade_repo.create(
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
                trade_id="T001",
            )
        )
        trade_repo.create(
            Trade(
                symbol="GOOGL",
                side=TradeSide.SELL,
                quantity=Decimal("50"),
                price=Decimal("140.00"),
                timestamp=datetime.now(timezone.utc),
                trade_id="T002",
            )
        )

        result = trade_repo.query(side=TradeSide.BUY)
        assert len(result) == 1
        assert result[0].side == TradeSide.BUY

    def test_query_by_date_range(self, trade_repo: TradeRepository) -> None:
        """Test querying trades by date range."""
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        trade_repo.create(
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=base_time,
                trade_id="T001",
            )
        )
        trade_repo.create(
            Trade(
                symbol="GOOGL",
                side=TradeSide.SELL,
                quantity=Decimal("50"),
                price=Decimal("140.00"),
                timestamp=base_time + timedelta(days=1),
                trade_id="T002",
            )
        )
        trade_repo.create(
            Trade(
                symbol="MSFT",
                side=TradeSide.BUY,
                quantity=Decimal("200"),
                price=Decimal("380.00"),
                timestamp=base_time + timedelta(days=2),
                trade_id="T003",
            )
        )

        # Query by start date
        result = trade_repo.query(start_date=base_time + timedelta(days=1))
        assert len(result) == 2

        # Query by end date
        result = trade_repo.query(end_date=base_time + timedelta(days=1))
        assert len(result) == 2

        # Query by date range
        result = trade_repo.query(
            start_date=base_time + timedelta(hours=12),
            end_date=base_time + timedelta(days=1, hours=12),
        )
        assert len(result) == 1

    def test_query_combined_filters(self, trade_repo: TradeRepository) -> None:
        """Test querying with multiple filters."""
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        trade_repo.create(
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=base_time,
                trade_id="T001",
            )
        )
        trade_repo.create(
            Trade(
                symbol="AAPL",
                side=TradeSide.SELL,
                quantity=Decimal("50"),
                price=Decimal("155.00"),
                timestamp=base_time + timedelta(hours=1),
                trade_id="T002",
            )
        )
        trade_repo.create(
            Trade(
                symbol="GOOGL",
                side=TradeSide.BUY,
                quantity=Decimal("200"),
                price=Decimal("140.00"),
                timestamp=base_time + timedelta(hours=2),
                trade_id="T003",
            )
        )

        result = trade_repo.query(symbol="AAPL", side=TradeSide.BUY)
        assert len(result) == 1
        assert result[0].trade_id == "T001"

    def test_update_trade(self, trade_repo: TradeRepository, sample_trade: Trade) -> None:
        """Test updating a trade."""
        trade_repo.create(sample_trade)

        updated_trade = Trade(
            symbol="AAPL",
            side=TradeSide.SELL,  # Changed
            quantity=Decimal("200"),  # Changed
            price=Decimal("160.00"),  # Changed
            timestamp=sample_trade.timestamp,
            trade_id=sample_trade.trade_id,
        )

        result = trade_repo.update(updated_trade)
        assert result.side == TradeSide.SELL
        assert result.quantity == Decimal("200")
        assert result.price == Decimal("160.00")

        # Verify in database
        found = trade_repo.get_by_id(sample_trade.trade_id)
        assert found is not None
        assert found.side == TradeSide.SELL

    def test_update_trade_not_found_raises(self, trade_repo: TradeRepository) -> None:
        """Test that updating non-existent trade raises RecordNotFoundError."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
            trade_id="nonexistent",
        )

        with pytest.raises(RecordNotFoundError) as exc_info:
            trade_repo.update(trade)

        assert exc_info.value.record_type == "Trade"
        assert exc_info.value.record_id == "nonexistent"

    def test_update_trade_without_id_raises(self, trade_repo: TradeRepository) -> None:
        """Test that updating trade without trade_id raises RecordNotFoundError."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
            trade_id="",
        )

        with pytest.raises(RecordNotFoundError):
            trade_repo.update(trade)

    def test_delete_trade(self, trade_repo: TradeRepository, sample_trade: Trade) -> None:
        """Test deleting a trade."""
        trade_repo.create(sample_trade)

        result = trade_repo.delete(sample_trade.trade_id)
        assert result is True

        # Verify deleted
        found = trade_repo.get_by_id(sample_trade.trade_id)
        assert found is None

    def test_delete_trade_not_found(self, trade_repo: TradeRepository) -> None:
        """Test deleting non-existent trade returns False."""
        result = trade_repo.delete("nonexistent")
        assert result is False

    def test_count_all(self, trade_repo: TradeRepository) -> None:
        """Test counting all trades."""
        assert trade_repo.count() == 0

        for i in range(3):
            trade = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
                trade_id=f"T{i:03d}",
            )
            trade_repo.create(trade)

        assert trade_repo.count() == 3

    def test_count_with_filter(self, trade_repo: TradeRepository) -> None:
        """Test counting trades with filters."""
        trade_repo.create(
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
                trade_id="T001",
            )
        )
        trade_repo.create(
            Trade(
                symbol="AAPL",
                side=TradeSide.SELL,
                quantity=Decimal("50"),
                price=Decimal("155.00"),
                timestamp=datetime.now(timezone.utc),
                trade_id="T002",
            )
        )
        trade_repo.create(
            Trade(
                symbol="GOOGL",
                side=TradeSide.BUY,
                quantity=Decimal("200"),
                price=Decimal("140.00"),
                timestamp=datetime.now(timezone.utc),
                trade_id="T003",
            )
        )

        assert trade_repo.count(symbol="AAPL") == 2
        assert trade_repo.count(side=TradeSide.BUY) == 2
        assert trade_repo.count(symbol="AAPL", side=TradeSide.BUY) == 1


# ============================================================================
# PositionRepository Tests
# ============================================================================


class TestPositionRepository:
    """Tests for PositionRepository class."""

    def test_get_by_symbol_found(
        self, position_repo: PositionRepository, sample_position: Position
    ) -> None:
        """Test retrieving a position by symbol."""
        position_repo.upsert(sample_position)

        found = position_repo.get_by_symbol("AAPL")
        assert found is not None
        assert found.symbol == "AAPL"
        assert found.quantity == sample_position.quantity

    def test_get_by_symbol_not_found(self, position_repo: PositionRepository) -> None:
        """Test that get_by_symbol returns None for non-existent position."""
        found = position_repo.get_by_symbol("NONEXISTENT")
        assert found is None

    def test_get_by_symbol_case_insensitive(
        self, position_repo: PositionRepository, sample_position: Position
    ) -> None:
        """Test that get_by_symbol is case-insensitive."""
        position_repo.upsert(sample_position)

        found = position_repo.get_by_symbol("aapl")
        assert found is not None
        assert found.symbol == "AAPL"

        found = position_repo.get_by_symbol("AaPl")
        assert found is not None
        assert found.symbol == "AAPL"

    def test_get_all_empty(self, position_repo: PositionRepository) -> None:
        """Test get_all with no positions."""
        positions = position_repo.get_all()
        assert positions == []

    def test_get_all_with_positions(self, position_repo: PositionRepository) -> None:
        """Test get_all returns all positions ordered by symbol."""
        positions_data = [
            Position(
                symbol="MSFT",
                quantity=Decimal("200"),
                avg_price=Decimal("380.00"),
            ),
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                avg_price=Decimal("150.50"),
            ),
            Position(
                symbol="GOOGL",
                quantity=Decimal("-50"),
                avg_price=Decimal("140.00"),
            ),
        ]

        for position in positions_data:
            position_repo.upsert(position)

        result = position_repo.get_all()
        assert len(result) == 3
        # Should be ordered by symbol
        assert result[0].symbol == "AAPL"
        assert result[1].symbol == "GOOGL"
        assert result[2].symbol == "MSFT"

    def test_upsert_insert(self, position_repo: PositionRepository) -> None:
        """Test upserting a new position (insert)."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("150.50"),
        )

        result = position_repo.upsert(position)
        assert result.symbol == "AAPL"

        found = position_repo.get_by_symbol("AAPL")
        assert found is not None
        assert found.quantity == Decimal("100")

    def test_upsert_update(self, position_repo: PositionRepository) -> None:
        """Test upserting an existing position (update)."""
        # Insert initial position
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("150.50"),
        )
        position_repo.upsert(position)

        # Update position
        updated_position = Position(
            symbol="AAPL",
            quantity=Decimal("200"),  # Changed
            avg_price=Decimal("155.00"),  # Changed
            unrealized_pnl=Decimal("500.00"),  # Changed
        )
        position_repo.upsert(updated_position)

        # Verify update
        found = position_repo.get_by_symbol("AAPL")
        assert found is not None
        assert found.quantity == Decimal("200")
        assert found.avg_price == Decimal("155.00")
        assert found.unrealized_pnl == Decimal("500.00")

        # Verify only one record exists
        assert position_repo.count() == 1

    def test_delete_position(
        self, position_repo: PositionRepository, sample_position: Position
    ) -> None:
        """Test deleting a position."""
        position_repo.upsert(sample_position)

        result = position_repo.delete("AAPL")
        assert result is True

        # Verify deleted
        found = position_repo.get_by_symbol("AAPL")
        assert found is None

    def test_delete_position_not_found(self, position_repo: PositionRepository) -> None:
        """Test deleting non-existent position returns False."""
        result = position_repo.delete("NONEXISTENT")
        assert result is False

    def test_delete_position_case_insensitive(
        self, position_repo: PositionRepository, sample_position: Position
    ) -> None:
        """Test that delete is case-insensitive."""
        position_repo.upsert(sample_position)

        result = position_repo.delete("aapl")
        assert result is True

        found = position_repo.get_by_symbol("AAPL")
        assert found is None

    def test_query_by_symbols(self, position_repo: PositionRepository) -> None:
        """Test querying positions by symbol list."""
        positions_data = [
            Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.50")),
            Position(symbol="GOOGL", quantity=Decimal("50"), avg_price=Decimal("140.00")),
            Position(symbol="MSFT", quantity=Decimal("200"), avg_price=Decimal("380.00")),
        ]

        for position in positions_data:
            position_repo.upsert(position)

        result = position_repo.query(symbols=["AAPL", "MSFT"])
        assert len(result) == 2
        symbols = {p.symbol for p in result}
        assert symbols == {"AAPL", "MSFT"}

    def test_query_by_quantity_range(self, position_repo: PositionRepository) -> None:
        """Test querying positions by quantity range."""
        positions_data = [
            Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.50")),
            Position(symbol="GOOGL", quantity=Decimal("50"), avg_price=Decimal("140.00")),
            Position(symbol="MSFT", quantity=Decimal("200"), avg_price=Decimal("380.00")),
            Position(symbol="TSLA", quantity=Decimal("-50"), avg_price=Decimal("250.00")),
        ]

        for position in positions_data:
            position_repo.upsert(position)

        # Min quantity
        result = position_repo.query(min_quantity=Decimal("100"))
        assert len(result) == 2
        symbols = {p.symbol for p in result}
        assert symbols == {"AAPL", "MSFT"}

        # Max quantity
        result = position_repo.query(max_quantity=Decimal("50"))
        assert len(result) == 2
        symbols = {p.symbol for p in result}
        assert symbols == {"GOOGL", "TSLA"}

        # Range
        result = position_repo.query(min_quantity=Decimal("50"), max_quantity=Decimal("100"))
        assert len(result) == 2

    def test_query_has_position(self, position_repo: PositionRepository) -> None:
        """Test querying by has_position filter."""
        positions_data = [
            Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.50")),
            Position(symbol="GOOGL", quantity=Decimal("0"), avg_price=Decimal("140.00")),
            Position(symbol="MSFT", quantity=Decimal("-50"), avg_price=Decimal("380.00")),
        ]

        for position in positions_data:
            position_repo.upsert(position)

        # Non-zero positions
        result = position_repo.query(has_position=True)
        assert len(result) == 2
        symbols = {p.symbol for p in result}
        assert symbols == {"AAPL", "MSFT"}

        # Zero positions
        result = position_repo.query(has_position=False)
        assert len(result) == 1
        assert result[0].symbol == "GOOGL"

    def test_count(self, position_repo: PositionRepository) -> None:
        """Test counting positions."""
        assert position_repo.count() == 0

        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            position_repo.upsert(
                Position(symbol=symbol, quantity=Decimal("100"), avg_price=Decimal("150.00"))
            )

        assert position_repo.count() == 3


# ============================================================================
# Exception Tests
# ============================================================================


class TestStorageExceptions:
    """Tests for storage exception hierarchy."""

    def test_storage_error_attributes(self) -> None:
        """Test StorageError attributes."""
        error = StorageError("Test error", operation="test", original_error=ValueError("original"))

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.operation == "test"
        assert isinstance(error.original_error, ValueError)

    def test_record_not_found_error_attributes(self) -> None:
        """Test RecordNotFoundError attributes."""
        error = RecordNotFoundError("Trade not found", record_type="Trade", record_id="T001")

        assert str(error) == "Trade not found"
        assert error.record_type == "Trade"
        assert error.record_id == "T001"
        assert error.operation == "read"

    def test_duplicate_record_error_attributes(self) -> None:
        """Test DuplicateRecordError attributes."""
        error = DuplicateRecordError("Duplicate trade", record_type="Trade", record_id="T001")

        assert str(error) == "Duplicate trade"
        assert error.record_type == "Trade"
        assert error.record_id == "T001"
        assert error.operation == "create"

    def test_database_connection_error_attributes(self) -> None:
        """Test DatabaseConnectionError attributes."""
        original = OSError("Permission denied")
        error = DatabaseConnectionError(
            "Connection failed", database_path="/path/to/db", original_error=original
        )

        assert str(error) == "Connection failed"
        assert error.database_path == "/path/to/db"
        assert error.original_error is original
        assert error.operation == "connect"

    def test_exception_inheritance(self) -> None:
        """Test exception inheritance hierarchy."""
        # All storage exceptions inherit from StorageError
        assert issubclass(RecordNotFoundError, StorageError)
        assert issubclass(DuplicateRecordError, StorageError)
        assert issubclass(DatabaseConnectionError, StorageError)

        # StorageError inherits from TradingError
        assert issubclass(StorageError, TradingError)

        # TradingError inherits from Exception
        assert issubclass(TradingError, Exception)


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for concurrent access scenarios."""

    def test_concurrent_reads(self) -> None:
        """Test concurrent read operations."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Initialize database with some data
            storage = SQLiteStorage(db_path)
            storage.connect()
            trade_repo = TradeRepository(storage)

            for i in range(10):
                trade = Trade(
                    symbol="AAPL",
                    side=TradeSide.BUY,
                    quantity=Decimal("100"),
                    price=Decimal("150.50"),
                    timestamp=datetime.now(timezone.utc),
                    trade_id=f"T{i:03d}",
                )
                trade_repo.create(trade)

            storage.disconnect()

            results = []
            errors = []
            lock = threading.Lock()

            def read_trades() -> None:
                try:
                    # Each thread gets its own storage connection to the file
                    thread_storage = SQLiteStorage(db_path)
                    thread_storage.connect()
                    thread_repo = TradeRepository(thread_storage)

                    for _ in range(10):
                        trades = thread_repo.get_all()
                        with lock:
                            results.append(len(trades))

                    thread_storage.disconnect()
                except Exception as e:
                    with lock:
                        errors.append(e)

            threads = [threading.Thread(target=read_trades) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert all(r == 10 for r in results)

    def test_concurrent_writes(self) -> None:
        """Test concurrent write operations."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Initialize database
            storage = SQLiteStorage(db_path)
            storage.connect()
            storage.disconnect()

            errors = []
            written_ids = []
            lock = threading.Lock()

            def write_trades(thread_id: int) -> None:
                try:
                    thread_storage = SQLiteStorage(db_path)
                    thread_storage.connect()
                    thread_repo = TradeRepository(thread_storage)

                    for i in range(5):
                        trade = Trade(
                            symbol="AAPL",
                            side=TradeSide.BUY,
                            quantity=Decimal("100"),
                            price=Decimal("150.50"),
                            timestamp=datetime.now(timezone.utc),
                            trade_id=f"T{thread_id:02d}{i:02d}",
                        )
                        created = thread_repo.create(trade)
                        with lock:
                            written_ids.append(created.trade_id)

                    thread_storage.disconnect()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=write_trades, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(written_ids) == 25
            assert len(set(written_ids)) == 25  # All unique

    def test_concurrent_mixed_operations(self) -> None:
        """Test concurrent mixed read/write operations."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Initialize database with some data
            storage = SQLiteStorage(db_path)
            storage.connect()
            trade_repo = TradeRepository(storage)

            for i in range(10):
                trade = Trade(
                    symbol="AAPL",
                    side=TradeSide.BUY,
                    quantity=Decimal("100"),
                    price=Decimal("150.50"),
                    timestamp=datetime.now(timezone.utc),
                    trade_id=f"INIT{i:03d}",
                )
                trade_repo.create(trade)

            storage.disconnect()

            errors = []

            def mixed_operations(thread_id: int) -> None:
                try:
                    thread_storage = SQLiteStorage(db_path)
                    thread_storage.connect()
                    thread_repo = TradeRepository(thread_storage)

                    for i in range(5):
                        # Read
                        thread_repo.get_all()

                        # Write
                        trade = Trade(
                            symbol="GOOGL",
                            side=TradeSide.SELL,
                            quantity=Decimal("50"),
                            price=Decimal("140.00"),
                            timestamp=datetime.now(timezone.utc),
                            trade_id=f"T{thread_id:02d}{i:02d}",
                        )
                        thread_repo.create(trade)

                        # Query
                        thread_repo.query(symbol="AAPL")

                    thread_storage.disconnect()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=mixed_operations, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

            # Verify final state
            storage = SQLiteStorage(db_path)
            storage.connect()
            trade_repo = TradeRepository(storage)
            assert trade_repo.count() == 35  # 10 initial + 25 new
            storage.disconnect()


# ============================================================================
# Integration Tests
# ============================================================================


class TestStorageIntegration:
    """Integration tests for storage layer."""

    def test_trade_roundtrip_decimal_precision(self, trade_repo: TradeRepository) -> None:
        """Test that Decimal precision is preserved through storage."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("123.456789012345"),  # High precision
            price=Decimal("150.123456789012"),
            timestamp=datetime.now(timezone.utc),
            trade_id="T001",
        )

        trade_repo.create(trade)
        retrieved = trade_repo.get_by_id("T001")

        assert retrieved is not None
        assert retrieved.quantity == trade.quantity
        assert retrieved.price == trade.price

    def test_position_roundtrip_negative_quantity(
        self, position_repo: PositionRepository
    ) -> None:
        """Test that negative quantities (short positions) are preserved."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-100"),
            avg_price=Decimal("250.00"),
            unrealized_pnl=Decimal("-500.00"),
        )

        position_repo.upsert(position)
        retrieved = position_repo.get_by_symbol("TSLA")

        assert retrieved is not None
        assert retrieved.quantity == Decimal("-100")
        assert retrieved.unrealized_pnl == Decimal("-500.00")

    def test_trade_timestamp_timezone_preservation(self, trade_repo: TradeRepository) -> None:
        """Test that timestamps with timezone info are preserved."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=timestamp,
            trade_id="T001",
        )

        trade_repo.create(trade)
        retrieved = trade_repo.get_by_id("T001")

        assert retrieved is not None
        assert retrieved.timestamp == timestamp
