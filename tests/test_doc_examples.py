"""Tests for documentation code examples.

This module verifies that all code examples in the documentation work correctly.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trade_analytics import (
    InsufficientFundsError,
    InvalidTradeError,
    MarketClosedError,
    MarketData,
    MissingMarketDataError,
    PortfolioCalculator,
    Position,
    Trade,
    TradeSide,
    TradingError,
)


class TestReadmeExamples:
    """Test examples from README.md."""

    def test_quick_start_example(self):
        """Test the quick start example from README."""
        # Create a trade
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
        )
        assert trade.symbol == "AAPL"
        assert trade.side == TradeSide.BUY

        # Create a position
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")
        )
        assert position.symbol == "AAPL"

        # Get current market data
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155.00"),
                ask=Decimal("155.10"),
                last=Decimal("155.05"),
                volume=1000000,
            )
        }

        # Calculate portfolio metrics
        total_value = PortfolioCalculator.calculate_total_value([position], market_data)
        pnl = PortfolioCalculator.calculate_pnl([position], market_data)

        assert total_value == Decimal("15505.00")
        assert pnl == Decimal("505.00")


class TestApiExamples:
    """Test examples from docs/API.md."""

    def test_tradesside_example(self):
        """Test TradeSide enum example."""
        side = TradeSide.BUY
        assert side.value == "BUY"

    def test_trade_example(self):
        """Test Trade creation and serialization example."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
            trade_id="T001",
        )

        assert trade.symbol == "AAPL"
        assert trade.quantity == Decimal("100")

        # Serialize to dictionary
        data = trade.to_dict()

        # Deserialize from dictionary
        trade2 = Trade.from_dict(data)
        assert trade2.symbol == trade.symbol

    def test_position_long_short_example(self):
        """Test Position long and short examples."""
        # Long position
        long_position = Position(
            symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")
        )
        assert long_position.quantity > 0

        # Short position
        short_position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),  # Negative = short
            avg_price=Decimal("200.00"),
        )
        assert short_position.quantity < 0

        # Serialize/deserialize
        data = long_position.to_dict()
        restored = Position.from_dict(data)
        assert restored.symbol == long_position.symbol

    def test_marketdata_example(self):
        """Test MarketData creation and properties."""
        market = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.10"),
            last=Decimal("150.05"),
            volume=1000000,
        )

        assert market.spread == Decimal("0.10")
        assert market.mid == Decimal("150.05")

    def test_calculator_total_value_example(self):
        """Test PortfolioCalculator.calculate_total_value example."""
        positions = [
            Position(
                symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")
            )
        ]
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155.00"),
                ask=Decimal("155.10"),
                last=Decimal("155.05"),
                volume=1000000,
            )
        }

        total = PortfolioCalculator.calculate_total_value(positions, market_data)
        assert total == Decimal("15505.00")

    def test_calculator_pnl_example(self):
        """Test PortfolioCalculator.calculate_pnl example."""
        positions = [
            Position(
                symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")
            )
        ]
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155.00"),
                ask=Decimal("155.10"),
                last=Decimal("155.05"),
                volume=1000000,
            )
        }

        pnl = PortfolioCalculator.calculate_pnl(positions, market_data)
        assert pnl == Decimal("505.00")

    def test_invalid_trade_error_example(self):
        """Test InvalidTradeError example."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("-100"),
                price=Decimal("150.00"),
                timestamp=datetime.now(timezone.utc),
            )

        assert exc_info.value.reason == "invalid_quantity"

    def test_missing_market_data_error_example(self):
        """Test MissingMarketDataError example."""
        positions = [
            Position(
                symbol="TSLA", quantity=Decimal("100"), avg_price=Decimal("200.00")
            )
        ]
        market_data = {}  # Empty - no data

        with pytest.raises(MissingMarketDataError) as exc_info:
            PortfolioCalculator.calculate_total_value(positions, market_data)

        assert exc_info.value.symbol == "TSLA"
        assert exc_info.value.available_symbols == []


class TestExamplesDocExamples:
    """Test examples from docs/EXAMPLES.md."""

    def test_creating_trades_basic(self):
        """Test basic trade creation example."""
        buy_trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
        )
        assert buy_trade.quantity == Decimal("100")
        assert buy_trade.symbol == "AAPL"

        sell_trade = Trade(
            symbol="AAPL",
            side=TradeSide.SELL,
            quantity=Decimal("50"),
            price=Decimal("155.00"),
            timestamp=datetime.now(timezone.utc),
        )
        assert sell_trade.quantity == Decimal("50")

    def test_symbol_normalization(self):
        """Test symbol normalization example."""
        trade = Trade(
            symbol="aapl",  # lowercase
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now(timezone.utc),
        )
        assert trade.symbol == "AAPL"

    def test_flat_position(self):
        """Test flat position example."""
        flat_position = Position(
            symbol="MSFT", quantity=Decimal("0"), avg_price=Decimal("0")
        )
        is_flat = flat_position.quantity == 0
        assert is_flat is True

    def test_market_data_spread_mid(self):
        """Test market data spread and mid properties example."""
        quote = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.10"),
            last=Decimal("150.05"),
            volume=1000000,
        )
        assert quote.spread == Decimal("0.10")
        assert quote.mid == Decimal("150.05")

    def test_portfolio_total_value(self):
        """Test portfolio total value example."""
        positions = [
            Position(
                symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")
            ),
            Position(
                symbol="GOOGL", quantity=Decimal("50"), avg_price=Decimal("140.00")
            ),
        ]

        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155"),
                ask=Decimal("156"),
                last=Decimal("155.05"),
                volume=1000,
            ),
            "GOOGL": MarketData(
                symbol="GOOGL",
                bid=Decimal("145"),
                ask=Decimal("146"),
                last=Decimal("145.10"),
                volume=1000,
            ),
        }

        total_value = PortfolioCalculator.calculate_total_value(positions, market_data)
        # 100 * 155.05 + 50 * 145.10 = 15505 + 7255 = 22760
        assert total_value == Decimal("22760.00")

    def test_exposure_by_symbol(self):
        """Test exposure by symbol example."""
        positions = [
            Position(
                symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00")
            ),
            Position(
                symbol="TSLA", quantity=Decimal("-50"), avg_price=Decimal("200.00")
            ),
            Position(
                symbol="GOOGL", quantity=Decimal("25"), avg_price=Decimal("140.00")
            ),
        ]

        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155"),
                ask=Decimal("156"),
                last=Decimal("155.00"),
                volume=1000,
            ),
            "TSLA": MarketData(
                symbol="TSLA",
                bid=Decimal("210"),
                ask=Decimal("211"),
                last=Decimal("210.50"),
                volume=1000,
            ),
            "GOOGL": MarketData(
                symbol="GOOGL",
                bid=Decimal("145"),
                ask=Decimal("146"),
                last=Decimal("145.00"),
                volume=1000,
            ),
        }

        exposure = PortfolioCalculator.calculate_exposure_by_symbol(
            positions, market_data
        )
        assert exposure["AAPL"] == Decimal("15500.00")
        assert exposure["TSLA"] == Decimal("10525.00")
        assert exposure["GOOGL"] == Decimal("3625.00")

    def test_empty_portfolio(self):
        """Test empty portfolio handling example."""
        empty_positions = []
        empty_market_data = {}

        total = PortfolioCalculator.calculate_total_value(
            empty_positions, empty_market_data
        )
        assert total == Decimal("0")

        pnl = PortfolioCalculator.calculate_pnl(empty_positions, empty_market_data)
        assert pnl == Decimal("0")

        exposure = PortfolioCalculator.calculate_exposure_by_symbol(
            empty_positions, empty_market_data
        )
        assert exposure == {}

    def test_invalid_quantity_error(self):
        """Test invalid quantity error example."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("-100"),
                price=Decimal("150.00"),
                timestamp=datetime.now(timezone.utc),
            )
        assert exc_info.value.reason == "invalid_quantity"

    def test_empty_symbol_error(self):
        """Test empty symbol error example."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Position(symbol="", quantity=Decimal("100"), avg_price=Decimal("150.00"))
        assert exc_info.value.reason == "empty_symbol"

    def test_crossed_market_error(self):
        """Test crossed market (bid > ask) error example."""
        with pytest.raises(InvalidTradeError) as exc_info:
            MarketData(
                symbol="AAPL",
                bid=Decimal("151.00"),
                ask=Decimal("150.00"),
                last=Decimal("150.50"),
                volume=1000,
            )
        assert exc_info.value.reason == "invalid_spread"

    def test_trade_serialization(self):
        """Test trade serialization example."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            trade_id="T001",
        )

        data = trade.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["side"] == "BUY"
        assert data["quantity"] == "100"

        restored_trade = Trade.from_dict(data)
        assert restored_trade.symbol == trade.symbol
        assert restored_trade.side == trade.side

    def test_position_serialization(self):
        """Test position serialization example."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("150.00"),
            unrealized_pnl=Decimal("500.00"),
        )

        data = position.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["quantity"] == "100"

        restored = Position.from_dict(data)
        assert restored.quantity == position.quantity

    def test_marketdata_serialization(self):
        """Test marketdata serialization example."""
        quote = MarketData(
            symbol="AAPL",
            bid=Decimal("150.00"),
            ask=Decimal("150.10"),
            last=Decimal("150.05"),
            volume=1000000,
            timestamp=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        )

        data = quote.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["volume"] == 1000000

        restored = MarketData.from_dict(data)
        assert restored.symbol == quote.symbol
        assert restored.last == quote.last


class TestExceptionAttributes:
    """Test exception attributes mentioned in documentation."""

    def test_insufficient_funds_error_attributes(self):
        """Test InsufficientFundsError attributes."""
        error = InsufficientFundsError(
            "Insufficient funds for trade",
            required=Decimal("10000.00"),
            available=Decimal("5000.00"),
        )
        assert error.required == Decimal("10000.00")
        assert error.available == Decimal("5000.00")

    def test_market_closed_error_attributes(self):
        """Test MarketClosedError attributes."""
        error = MarketClosedError(
            "Market is closed",
            symbol="NYSE:AAPL",
            market_hours="9:30 AM - 4:00 PM ET",
        )
        assert error.symbol == "NYSE:AAPL"
        assert error.market_hours == "9:30 AM - 4:00 PM ET"

    def test_trading_error_catch_all(self):
        """Test TradingError as catch-all."""
        try:
            raise InvalidTradeError("test error", reason="test")
        except TradingError as e:
            assert e.message == "test error"
