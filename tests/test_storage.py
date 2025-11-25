"""Tests for trade_analytics storage layer."""

import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trade_analytics.exceptions import StorageError
from trade_analytics.models import Position, Trade, TradeSide
from trade_analytics.storage import TradeStore


@pytest.fixture
def memory_store() -> TradeStore:
    """Create an in-memory TradeStore for testing.

    Returns:
        TradeStore instance using :memory: database.
    """
    return TradeStore(":memory:")


@pytest.fixture
def sample_trade() -> Trade:
    """Create a sample valid trade for testing.

    Returns:
        Trade instance with predefined values.
    """
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
    """Create a sample valid position for testing.

    Returns:
        Position instance with predefined values.
    """
    return Position(
        symbol="AAPL",
        quantity=Decimal("100"),
        avg_price=Decimal("150.50"),
        unrealized_pnl=Decimal("250.00"),
    )


class TestTradeStoreInitialization:
    """Tests for TradeStore initialization and setup."""

    def test_store_creates_database_file(self) -> None:
        """Test that TradeStore creates a database file on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_trades.db"
            store = TradeStore(str(db_path))

            # Database file should be created
            assert db_path.exists()

            # Should be a valid SQLite database
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Should have both tables
            assert "trades" in tables
            assert "positions" in tables

    def test_store_in_memory_database(self) -> None:
        """Test that TradeStore can use in-memory database."""
        store = TradeStore(":memory:")
        assert store.db_path == ":memory:"

    def test_store_creates_tables_on_init(self, memory_store: TradeStore) -> None:
        """Test that tables are created when using context manager."""
        with memory_store:
            # Should be able to query tables
            trades = memory_store.get_trades()
            positions = memory_store.get_positions()

            assert trades == []
            assert positions == []

    def test_store_context_manager_enter_exit(
        self, memory_store: TradeStore, sample_trade: Trade
    ) -> None:
        """Test that context manager properly enters and exits."""
        # Should not have connection before entering
        assert memory_store._connection is None

        with memory_store:
            # Should have connection inside context
            assert memory_store._connection is not None

            # Should be able to perform operations
            memory_store.save_trade(sample_trade)
            trades = memory_store.get_trades()
            assert len(trades) == 1

        # Connection should be closed after exiting
        assert memory_store._connection is None

    def test_store_context_manager_cleanup_on_exception(
        self, memory_store: TradeStore, sample_trade: Trade
    ) -> None:
        """Test that context manager properly cleans up on exception."""
        try:
            with memory_store:
                memory_store.save_trade(sample_trade)
                # Force an exception
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        # Connection should be closed even after exception
        assert memory_store._connection is None

        # Transaction should have been rolled back
        with memory_store:
            trades = memory_store.get_trades()
            # Trade should not be in database (rolled back)
            assert len(trades) == 0


class TestSaveTrade:
    """Tests for saving trades to storage."""

    def test_save_trade_valid(
        self, memory_store: TradeStore, sample_trade: Trade
    ) -> None:
        """Test saving a valid trade."""
        with memory_store:
            memory_store.save_trade(sample_trade)

            # Retrieve and verify
            trades = memory_store.get_trades()
            assert len(trades) == 1
            assert trades[0].symbol == "AAPL"
            assert trades[0].side == TradeSide.BUY
            assert trades[0].quantity == Decimal("100")
            assert trades[0].price == Decimal("150.50")
            assert trades[0].trade_id == "T001"

    def test_save_trade_generates_uuid_when_empty(
        self, memory_store: TradeStore
    ) -> None:
        """Test that TradeStore generates UUID when trade_id is empty."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
            trade_id="",  # Empty trade_id
        )

        with memory_store:
            memory_store.save_trade(trade)

            # Retrieve and verify UUID was generated
            trades = memory_store.get_trades()
            assert len(trades) == 1
            assert trades[0].trade_id != ""
            # Should be a valid UUID format (36 characters with hyphens)
            assert len(trades[0].trade_id) == 36
            assert trades[0].trade_id.count("-") == 4

    def test_save_trade_updates_existing(
        self, memory_store: TradeStore, sample_trade: Trade
    ) -> None:
        """Test that saving trade with existing trade_id updates the record."""
        with memory_store:
            # Save initial trade
            memory_store.save_trade(sample_trade)

            # Create updated trade with same trade_id
            updated_trade = Trade(
                symbol="AAPL",
                side=TradeSide.SELL,  # Changed side
                quantity=Decimal("200"),  # Changed quantity
                price=Decimal("155.00"),  # Changed price
                timestamp=datetime.now(timezone.utc),
                trade_id="T001",  # Same trade_id
            )
            memory_store.save_trade(updated_trade)

            # Should only have one trade with updated values
            trades = memory_store.get_trades()
            assert len(trades) == 1
            assert trades[0].trade_id == "T001"
            assert trades[0].side == TradeSide.SELL
            assert trades[0].quantity == Decimal("200")
            assert trades[0].price == Decimal("155.00")


class TestGetTrades:
    """Tests for retrieving trades from storage."""

    def test_get_trades_all(self, memory_store: TradeStore) -> None:
        """Test retrieving all trades without filters."""
        with memory_store:
            # Save multiple trades
            trade1 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            )
            trade2 = Trade(
                symbol="GOOGL",
                side=TradeSide.BUY,
                quantity=Decimal("50"),
                price=Decimal("2800.00"),
                timestamp=datetime(2024, 1, 16, 11, 0, 0, tzinfo=timezone.utc),
            )
            trade3 = Trade(
                symbol="MSFT",
                side=TradeSide.SELL,
                quantity=Decimal("75"),
                price=Decimal("380.00"),
                timestamp=datetime(2024, 1, 17, 14, 15, 0, tzinfo=timezone.utc),
            )

            memory_store.save_trade(trade1)
            memory_store.save_trade(trade2)
            memory_store.save_trade(trade3)

            # Retrieve all
            trades = memory_store.get_trades()
            assert len(trades) == 3
            # Should be ordered by timestamp DESC
            assert trades[0].symbol == "MSFT"
            assert trades[1].symbol == "GOOGL"
            assert trades[2].symbol == "AAPL"

    def test_get_trades_filter_by_symbol(self, memory_store: TradeStore) -> None:
        """Test filtering trades by symbol."""
        with memory_store:
            # Save multiple trades
            trade1 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
            )
            trade2 = Trade(
                symbol="GOOGL",
                side=TradeSide.BUY,
                quantity=Decimal("50"),
                price=Decimal("2800.00"),
                timestamp=datetime.now(timezone.utc),
            )
            trade3 = Trade(
                symbol="AAPL",
                side=TradeSide.SELL,
                quantity=Decimal("50"),
                price=Decimal("155.00"),
                timestamp=datetime.now(timezone.utc),
            )

            memory_store.save_trade(trade1)
            memory_store.save_trade(trade2)
            memory_store.save_trade(trade3)

            # Filter by AAPL
            aapl_trades = memory_store.get_trades(symbol="AAPL")
            assert len(aapl_trades) == 2
            assert all(t.symbol == "AAPL" for t in aapl_trades)

    def test_get_trades_filter_by_symbol_case_insensitive(
        self, memory_store: TradeStore
    ) -> None:
        """Test that symbol filter is case-insensitive."""
        with memory_store:
            trade = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
            )
            memory_store.save_trade(trade)

            # Search with lowercase should find uppercase symbol
            trades_lower = memory_store.get_trades(symbol="aapl")
            assert len(trades_lower) == 1
            assert trades_lower[0].symbol == "AAPL"

            # Search with mixed case
            trades_mixed = memory_store.get_trades(symbol="AaPl")
            assert len(trades_mixed) == 1
            assert trades_mixed[0].symbol == "AAPL"

    def test_get_trades_filter_by_start_date(self, memory_store: TradeStore) -> None:
        """Test filtering trades by start date."""
        with memory_store:
            # Save trades with different timestamps
            trade1 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
            )
            trade2 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("151.00"),
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            )
            trade3 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("152.00"),
                timestamp=datetime(2024, 1, 20, 10, 0, 0, tzinfo=timezone.utc),
            )

            memory_store.save_trade(trade1)
            memory_store.save_trade(trade2)
            memory_store.save_trade(trade3)

            # Filter by start date (inclusive)
            start_date = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
            trades = memory_store.get_trades(start_date=start_date)
            assert len(trades) == 2
            assert all(t.timestamp >= start_date for t in trades)

    def test_get_trades_filter_by_end_date(self, memory_store: TradeStore) -> None:
        """Test filtering trades by end date."""
        with memory_store:
            # Save trades with different timestamps
            trade1 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
            )
            trade2 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("151.00"),
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            )
            trade3 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("152.00"),
                timestamp=datetime(2024, 1, 20, 10, 0, 0, tzinfo=timezone.utc),
            )

            memory_store.save_trade(trade1)
            memory_store.save_trade(trade2)
            memory_store.save_trade(trade3)

            # Filter by end date (inclusive)
            end_date = datetime(2024, 1, 15, 23, 59, 59, tzinfo=timezone.utc)
            trades = memory_store.get_trades(end_date=end_date)
            assert len(trades) == 2
            assert all(t.timestamp <= end_date for t in trades)

    def test_get_trades_filter_by_date_range(self, memory_store: TradeStore) -> None:
        """Test filtering trades by date range."""
        with memory_store:
            # Save trades with different timestamps
            trade1 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
            )
            trade2 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("151.00"),
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            )
            trade3 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("152.00"),
                timestamp=datetime(2024, 1, 20, 10, 0, 0, tzinfo=timezone.utc),
            )
            trade4 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("153.00"),
                timestamp=datetime(2024, 1, 25, 10, 0, 0, tzinfo=timezone.utc),
            )

            memory_store.save_trade(trade1)
            memory_store.save_trade(trade2)
            memory_store.save_trade(trade3)
            memory_store.save_trade(trade4)

            # Filter by date range
            start_date = datetime(2024, 1, 12, 0, 0, 0, tzinfo=timezone.utc)
            end_date = datetime(2024, 1, 22, 0, 0, 0, tzinfo=timezone.utc)
            trades = memory_store.get_trades(start_date=start_date, end_date=end_date)
            assert len(trades) == 2
            assert all(start_date <= t.timestamp <= end_date for t in trades)

    def test_get_trades_empty_result(self, memory_store: TradeStore) -> None:
        """Test that empty result returns empty list."""
        with memory_store:
            # No trades saved, should return empty list
            trades = memory_store.get_trades()
            assert trades == []

            # Filter with no matches should return empty list
            trades = memory_store.get_trades(symbol="NONEXISTENT")
            assert trades == []


class TestDeleteTrade:
    """Tests for deleting trades from storage."""

    def test_delete_trade_exists(
        self, memory_store: TradeStore, sample_trade: Trade
    ) -> None:
        """Test deleting an existing trade."""
        with memory_store:
            memory_store.save_trade(sample_trade)

            # Verify trade exists
            trades = memory_store.get_trades()
            assert len(trades) == 1

            # Delete the trade
            result = memory_store.delete_trade("T001")
            assert result is True

            # Verify trade is deleted
            trades = memory_store.get_trades()
            assert len(trades) == 0

    def test_delete_trade_not_found(self, memory_store: TradeStore) -> None:
        """Test deleting a non-existent trade returns False."""
        with memory_store:
            result = memory_store.delete_trade("NONEXISTENT")
            assert result is False


class TestSavePosition:
    """Tests for saving positions to storage."""

    def test_save_position_new(
        self, memory_store: TradeStore, sample_position: Position
    ) -> None:
        """Test saving a new position."""
        with memory_store:
            memory_store.save_position(sample_position)

            # Retrieve and verify
            positions = memory_store.get_positions()
            assert len(positions) == 1
            assert positions[0].symbol == "AAPL"
            assert positions[0].quantity == Decimal("100")
            assert positions[0].avg_price == Decimal("150.50")
            assert positions[0].unrealized_pnl == Decimal("250.00")

    def test_save_position_update_existing(
        self, memory_store: TradeStore, sample_position: Position
    ) -> None:
        """Test that saving position with existing symbol updates the record."""
        with memory_store:
            # Save initial position
            memory_store.save_position(sample_position)

            # Create updated position with same symbol
            updated_position = Position(
                symbol="AAPL",  # Same symbol
                quantity=Decimal("200"),  # Changed quantity
                avg_price=Decimal("152.00"),  # Changed avg_price
                unrealized_pnl=Decimal("500.00"),  # Changed pnl
            )
            memory_store.save_position(updated_position)

            # Should only have one position with updated values
            positions = memory_store.get_positions()
            assert len(positions) == 1
            assert positions[0].symbol == "AAPL"
            assert positions[0].quantity == Decimal("200")
            assert positions[0].avg_price == Decimal("152.00")
            assert positions[0].unrealized_pnl == Decimal("500.00")


class TestGetPositions:
    """Tests for retrieving positions from storage."""

    def test_get_positions_all(self, memory_store: TradeStore) -> None:
        """Test retrieving all positions."""
        with memory_store:
            # Save multiple positions
            pos1 = Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                avg_price=Decimal("150.50"),
            )
            pos2 = Position(
                symbol="GOOGL",
                quantity=Decimal("50"),
                avg_price=Decimal("2800.00"),
            )
            pos3 = Position(
                symbol="MSFT",
                quantity=Decimal("75"),
                avg_price=Decimal("380.00"),
            )

            memory_store.save_position(pos1)
            memory_store.save_position(pos2)
            memory_store.save_position(pos3)

            # Retrieve all
            positions = memory_store.get_positions()
            assert len(positions) == 3
            # Should be ordered by symbol
            assert positions[0].symbol == "AAPL"
            assert positions[1].symbol == "GOOGL"
            assert positions[2].symbol == "MSFT"

    def test_get_positions_empty(self, memory_store: TradeStore) -> None:
        """Test that empty result returns empty list."""
        with memory_store:
            positions = memory_store.get_positions()
            assert positions == []


class TestDeletePosition:
    """Tests for deleting positions from storage."""

    def test_delete_position_exists(
        self, memory_store: TradeStore, sample_position: Position
    ) -> None:
        """Test deleting an existing position."""
        with memory_store:
            memory_store.save_position(sample_position)

            # Verify position exists
            positions = memory_store.get_positions()
            assert len(positions) == 1

            # Delete the position
            result = memory_store.delete_position("AAPL")
            assert result is True

            # Verify position is deleted
            positions = memory_store.get_positions()
            assert len(positions) == 0

    def test_delete_position_not_found(self, memory_store: TradeStore) -> None:
        """Test deleting a non-existent position returns False."""
        with memory_store:
            result = memory_store.delete_position("NONEXISTENT")
            assert result is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_decimal_precision_preserved(self, memory_store: TradeStore) -> None:
        """Test that Decimal precision is preserved through save/load."""
        with memory_store:
            # Create trade with high-precision decimals
            trade = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("123.456789"),
                price=Decimal("150.123456"),
                timestamp=datetime.now(timezone.utc),
            )
            memory_store.save_trade(trade)

            # Retrieve and verify precision is preserved
            trades = memory_store.get_trades()
            assert len(trades) == 1
            assert trades[0].quantity == Decimal("123.456789")
            assert trades[0].price == Decimal("150.123456")

            # Test position precision
            position = Position(
                symbol="AAPL",
                quantity=Decimal("123.456789"),
                avg_price=Decimal("150.123456"),
                unrealized_pnl=Decimal("999.999999"),
            )
            memory_store.save_position(position)

            positions = memory_store.get_positions()
            assert len(positions) == 1
            assert positions[0].quantity == Decimal("123.456789")
            assert positions[0].avg_price == Decimal("150.123456")
            assert positions[0].unrealized_pnl == Decimal("999.999999")

    def test_timestamp_timezone_handled(self, memory_store: TradeStore) -> None:
        """Test that timestamps with timezones are handled correctly."""
        with memory_store:
            # Create trade with UTC timezone
            utc_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            trade = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=utc_time,
            )
            memory_store.save_trade(trade)

            # Retrieve and verify timezone is preserved
            trades = memory_store.get_trades()
            assert len(trades) == 1
            # Timestamp should have timezone information
            assert trades[0].timestamp.tzinfo is not None
            # Should be equivalent to original UTC time
            assert trades[0].timestamp == utc_time

    def test_special_characters_in_symbol(self, memory_store: TradeStore) -> None:
        """Test that symbols with special characters are handled correctly."""
        with memory_store:
            # Note: Trade model will uppercase and validate the symbol
            # Test with symbols that might have special chars (before validation)
            trade = Trade(
                symbol="BRK.B",  # Symbol with dot
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("350.00"),
                timestamp=datetime.now(timezone.utc),
            )
            memory_store.save_trade(trade)

            # Retrieve and verify
            trades = memory_store.get_trades(symbol="BRK.B")
            assert len(trades) == 1
            assert trades[0].symbol == "BRK.B"

    def test_clear_all_removes_everything(self, memory_store: TradeStore) -> None:
        """Test that clear_all removes all trades and positions."""
        with memory_store:
            # Add multiple trades
            trade1 = Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
            )
            trade2 = Trade(
                symbol="GOOGL",
                side=TradeSide.BUY,
                quantity=Decimal("50"),
                price=Decimal("2800.00"),
                timestamp=datetime.now(timezone.utc),
            )
            memory_store.save_trade(trade1)
            memory_store.save_trade(trade2)

            # Add multiple positions
            pos1 = Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                avg_price=Decimal("150.50"),
            )
            pos2 = Position(
                symbol="GOOGL",
                quantity=Decimal("50"),
                avg_price=Decimal("2800.00"),
            )
            memory_store.save_position(pos1)
            memory_store.save_position(pos2)

            # Verify data exists
            assert len(memory_store.get_trades()) == 2
            assert len(memory_store.get_positions()) == 2

            # Clear all
            memory_store.clear_all()

            # Verify everything is deleted
            assert len(memory_store.get_trades()) == 0
            assert len(memory_store.get_positions()) == 0

    def test_storage_error_raised_on_failure(self, memory_store: TradeStore) -> None:
        """Test that StorageError is raised when database operations fail."""
        # This test verifies that StorageError is properly raised
        # We can test this by attempting to use a closed connection

        with memory_store:
            pass  # Close the connection

        # Attempting to use methods outside context should work
        # because _get_connection creates a temporary connection
        # So we need to test a different failure scenario

        # Test with a corrupted store by using an invalid db path
        # that will cause issues (this is a bit contrived, but tests the exception)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file (not a directory) at the path we want to use as a directory
            bad_path = Path(tmpdir) / "bad.db" / "nested.db"
            bad_path.parent.parent.mkdir(exist_ok=True)

            # Create the parent as a file, not a directory
            with open(bad_path.parent, "w") as f:
                f.write("not a directory")

            # This should fail when trying to create the database
            with pytest.raises((StorageError, sqlite3.OperationalError)):
                bad_store = TradeStore(str(bad_path))
