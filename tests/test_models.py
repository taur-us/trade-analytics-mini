"""Tests for trade_analytics models and exceptions."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trade_analytics.exceptions import (
    InsufficientFundsError,
    InvalidTradeError,
    MarketClosedError,
    TradingError,
)
from trade_analytics.models import MarketData, Position, Trade, TradeSide


class TestTradeSide:
    """Tests for TradeSide enum."""

    def test_buy_value(self) -> None:
        """Test BUY enum value."""
        assert TradeSide.BUY.value == "BUY"

    def test_sell_value(self) -> None:
        """Test SELL enum value."""
        assert TradeSide.SELL.value == "SELL"

    def test_from_string(self) -> None:
        """Test creating TradeSide from string value."""
        assert TradeSide("BUY") == TradeSide.BUY
        assert TradeSide("SELL") == TradeSide.SELL

    def test_invalid_value(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            TradeSide("INVALID")


class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation_valid(self, sample_trade: Trade) -> None:
        """Test creating a valid trade."""
        assert sample_trade.symbol == "AAPL"
        assert sample_trade.side == TradeSide.BUY
        assert sample_trade.quantity == Decimal("100")
        assert sample_trade.price == Decimal("150.50")
        assert sample_trade.trade_id == "T001"

    def test_trade_symbol_uppercase(self) -> None:
        """Test that symbols are uppercased."""
        trade = Trade(
            symbol="aapl",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
        )
        assert trade.symbol == "AAPL"

    def test_trade_symbol_stripped(self) -> None:
        """Test that symbols are stripped of whitespace."""
        trade = Trade(
            symbol="  AAPL  ",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
        )
        assert trade.symbol == "AAPL"

    def test_trade_creation_invalid_symbol_empty(self) -> None:
        """Test that empty symbol raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Trade(
                symbol="",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
            )
        assert exc_info.value.reason == "empty_symbol"

    def test_trade_creation_invalid_symbol_whitespace(self) -> None:
        """Test that whitespace-only symbol raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Trade(
                symbol="   ",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
            )
        assert exc_info.value.reason == "empty_symbol"

    def test_trade_creation_invalid_quantity_zero(self) -> None:
        """Test that zero quantity raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("0"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
            )
        assert exc_info.value.reason == "invalid_quantity"

    def test_trade_creation_invalid_quantity_negative(self) -> None:
        """Test that negative quantity raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("-100"),
                price=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc),
            )
        assert exc_info.value.reason == "invalid_quantity"

    def test_trade_creation_invalid_price_zero(self) -> None:
        """Test that zero price raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("0"),
                timestamp=datetime.now(timezone.utc),
            )
        assert exc_info.value.reason == "invalid_price"

    def test_trade_creation_invalid_price_negative(self) -> None:
        """Test that negative price raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("-150.50"),
                timestamp=datetime.now(timezone.utc),
            )
        assert exc_info.value.reason == "invalid_price"

    def test_trade_immutability(self, sample_trade: Trade) -> None:
        """Test that trades are immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_trade.symbol = "MSFT"  # type: ignore

    def test_trade_to_dict(self, sample_trade: Trade) -> None:
        """Test trade serialization to dictionary."""
        result = sample_trade.to_dict()
        assert result["symbol"] == "AAPL"
        assert result["side"] == "BUY"
        assert result["quantity"] == "100"
        assert result["price"] == "150.50"
        assert result["trade_id"] == "T001"
        assert "timestamp" in result

    def test_trade_from_dict(self, sample_trade: Trade) -> None:
        """Test trade deserialization from dictionary."""
        data = sample_trade.to_dict()
        trade = Trade.from_dict(data)
        assert trade.symbol == sample_trade.symbol
        assert trade.side == sample_trade.side
        assert trade.quantity == sample_trade.quantity
        assert trade.price == sample_trade.price
        assert trade.trade_id == sample_trade.trade_id

    def test_trade_default_trade_id(self) -> None:
        """Test that trade_id defaults to empty string."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc),
        )
        assert trade.trade_id == ""

    def test_trade_sell_side(self) -> None:
        """Test creating a SELL trade."""
        trade = Trade(
            symbol="AAPL",
            side=TradeSide.SELL,
            quantity=Decimal("50"),
            price=Decimal("155.00"),
            timestamp=datetime.now(timezone.utc),
        )
        assert trade.side == TradeSide.SELL


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation_valid(self, sample_position: Position) -> None:
        """Test creating a valid position."""
        assert sample_position.symbol == "AAPL"
        assert sample_position.quantity == Decimal("100")
        assert sample_position.avg_price == Decimal("150.50")
        assert sample_position.unrealized_pnl == Decimal("250.00")

    def test_position_symbol_uppercase(self) -> None:
        """Test that symbols are uppercased."""
        position = Position(
            symbol="aapl",
            quantity=Decimal("100"),
            avg_price=Decimal("150.50"),
        )
        assert position.symbol == "AAPL"

    def test_position_symbol_stripped(self) -> None:
        """Test that symbols are stripped of whitespace."""
        position = Position(
            symbol="  AAPL  ",
            quantity=Decimal("100"),
            avg_price=Decimal("150.50"),
        )
        assert position.symbol == "AAPL"

    def test_position_creation_invalid_symbol_empty(self) -> None:
        """Test that empty symbol raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Position(
                symbol="",
                quantity=Decimal("100"),
                avg_price=Decimal("150.50"),
            )
        assert exc_info.value.reason == "empty_symbol"

    def test_position_creation_invalid_avg_price_negative(self) -> None:
        """Test that negative avg_price raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                avg_price=Decimal("-150.50"),
            )
        assert exc_info.value.reason == "invalid_avg_price"

    def test_position_with_negative_quantity(self) -> None:
        """Test short position with negative quantity."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("-100"),
            avg_price=Decimal("150.50"),
        )
        assert position.quantity == Decimal("-100")

    def test_position_with_zero_quantity(self) -> None:
        """Test flat position with zero quantity."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            avg_price=Decimal("150.50"),
        )
        assert position.quantity == Decimal("0")

    def test_position_default_unrealized_pnl(self) -> None:
        """Test that unrealized_pnl defaults to zero."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("150.50"),
        )
        assert position.unrealized_pnl == Decimal("0")

    def test_position_mutable(self, sample_position: Position) -> None:
        """Test that positions are mutable."""
        sample_position.unrealized_pnl = Decimal("500.00")
        assert sample_position.unrealized_pnl == Decimal("500.00")

    def test_position_to_dict(self, sample_position: Position) -> None:
        """Test position serialization to dictionary."""
        result = sample_position.to_dict()
        assert result["symbol"] == "AAPL"
        assert result["quantity"] == "100"
        assert result["avg_price"] == "150.50"
        assert result["unrealized_pnl"] == "250.00"

    def test_position_from_dict(self, sample_position: Position) -> None:
        """Test position deserialization from dictionary."""
        data = sample_position.to_dict()
        position = Position.from_dict(data)
        assert position.symbol == sample_position.symbol
        assert position.quantity == sample_position.quantity
        assert position.avg_price == sample_position.avg_price
        assert position.unrealized_pnl == sample_position.unrealized_pnl

    def test_position_from_dict_default_pnl(self) -> None:
        """Test position deserialization with missing unrealized_pnl."""
        data = {
            "symbol": "AAPL",
            "quantity": "100",
            "avg_price": "150.50",
        }
        position = Position.from_dict(data)
        assert position.unrealized_pnl == Decimal("0")


class TestMarketData:
    """Tests for MarketData dataclass."""

    def test_market_data_creation_valid(self, sample_market_data: MarketData) -> None:
        """Test creating valid market data."""
        assert sample_market_data.symbol == "AAPL"
        assert sample_market_data.bid == Decimal("150.45")
        assert sample_market_data.ask == Decimal("150.55")
        assert sample_market_data.last == Decimal("150.50")
        assert sample_market_data.volume == 1000000

    def test_market_data_symbol_uppercase(self) -> None:
        """Test that symbols are uppercased."""
        market_data = MarketData(
            symbol="aapl",
            bid=Decimal("150.45"),
            ask=Decimal("150.55"),
            last=Decimal("150.50"),
            volume=1000000,
        )
        assert market_data.symbol == "AAPL"

    def test_market_data_symbol_stripped(self) -> None:
        """Test that symbols are stripped of whitespace."""
        market_data = MarketData(
            symbol="  AAPL  ",
            bid=Decimal("150.45"),
            ask=Decimal("150.55"),
            last=Decimal("150.50"),
            volume=1000000,
        )
        assert market_data.symbol == "AAPL"

    def test_market_data_creation_invalid_symbol_empty(self) -> None:
        """Test that empty symbol raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            MarketData(
                symbol="",
                bid=Decimal("150.45"),
                ask=Decimal("150.55"),
                last=Decimal("150.50"),
                volume=1000000,
            )
        assert exc_info.value.reason == "empty_symbol"

    def test_market_data_creation_invalid_bid_negative(self) -> None:
        """Test that negative bid raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            MarketData(
                symbol="AAPL",
                bid=Decimal("-150.45"),
                ask=Decimal("150.55"),
                last=Decimal("150.50"),
                volume=1000000,
            )
        assert exc_info.value.reason == "invalid_bid"

    def test_market_data_creation_invalid_ask_negative(self) -> None:
        """Test that negative ask raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            MarketData(
                symbol="AAPL",
                bid=Decimal("150.45"),
                ask=Decimal("-150.55"),
                last=Decimal("150.50"),
                volume=1000000,
            )
        assert exc_info.value.reason == "invalid_ask"

    def test_market_data_creation_invalid_spread(self) -> None:
        """Test that bid > ask raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            MarketData(
                symbol="AAPL",
                bid=Decimal("150.55"),
                ask=Decimal("150.45"),
                last=Decimal("150.50"),
                volume=1000000,
            )
        assert exc_info.value.reason == "invalid_spread"

    def test_market_data_creation_invalid_last_negative(self) -> None:
        """Test that negative last raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            MarketData(
                symbol="AAPL",
                bid=Decimal("150.45"),
                ask=Decimal("150.55"),
                last=Decimal("-150.50"),
                volume=1000000,
            )
        assert exc_info.value.reason == "invalid_last"

    def test_market_data_creation_invalid_volume_negative(self) -> None:
        """Test that negative volume raises InvalidTradeError."""
        with pytest.raises(InvalidTradeError) as exc_info:
            MarketData(
                symbol="AAPL",
                bid=Decimal("150.45"),
                ask=Decimal("150.55"),
                last=Decimal("150.50"),
                volume=-1000000,
            )
        assert exc_info.value.reason == "invalid_volume"

    def test_market_data_spread_calculation(
        self, sample_market_data: MarketData
    ) -> None:
        """Test spread property calculation."""
        assert sample_market_data.spread == Decimal("0.10")

    def test_market_data_mid_calculation(self, sample_market_data: MarketData) -> None:
        """Test mid price property calculation."""
        assert sample_market_data.mid == Decimal("150.50")

    def test_market_data_immutability(self, sample_market_data: MarketData) -> None:
        """Test that market data is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_market_data.symbol = "MSFT"  # type: ignore

    def test_market_data_to_dict(self, sample_market_data: MarketData) -> None:
        """Test market data serialization to dictionary."""
        result = sample_market_data.to_dict()
        assert result["symbol"] == "AAPL"
        assert result["bid"] == "150.45"
        assert result["ask"] == "150.55"
        assert result["last"] == "150.50"
        assert result["volume"] == 1000000
        assert "timestamp" in result

    def test_market_data_from_dict(self, sample_market_data: MarketData) -> None:
        """Test market data deserialization from dictionary."""
        data = sample_market_data.to_dict()
        market_data = MarketData.from_dict(data)
        assert market_data.symbol == sample_market_data.symbol
        assert market_data.bid == sample_market_data.bid
        assert market_data.ask == sample_market_data.ask
        assert market_data.last == sample_market_data.last
        assert market_data.volume == sample_market_data.volume

    def test_market_data_from_dict_no_timestamp(self) -> None:
        """Test market data deserialization without timestamp."""
        data = {
            "symbol": "AAPL",
            "bid": "150.45",
            "ask": "150.55",
            "last": "150.50",
            "volume": 1000000,
        }
        market_data = MarketData.from_dict(data)
        assert market_data.symbol == "AAPL"
        assert market_data.timestamp is not None

    def test_market_data_zero_values(self) -> None:
        """Test market data with zero values (valid edge case)."""
        market_data = MarketData(
            symbol="AAPL",
            bid=Decimal("0"),
            ask=Decimal("0"),
            last=Decimal("0"),
            volume=0,
        )
        assert market_data.bid == Decimal("0")
        assert market_data.spread == Decimal("0")
        assert market_data.mid == Decimal("0")

    def test_market_data_default_timestamp(self) -> None:
        """Test that timestamp defaults to current time."""
        market_data = MarketData(
            symbol="AAPL",
            bid=Decimal("150.45"),
            ask=Decimal("150.55"),
            last=Decimal("150.50"),
            volume=1000000,
        )
        assert market_data.timestamp is not None
        # Should be approximately now (within 1 second)
        now = datetime.now(timezone.utc)
        delta = abs((now - market_data.timestamp).total_seconds())
        assert delta < 1


class TestExceptions:
    """Tests for custom exception hierarchy."""

    def test_trading_error_base(self) -> None:
        """Test TradingError base exception."""
        error = TradingError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"

    def test_trading_error_inheritance(self) -> None:
        """Test that TradingError inherits from Exception."""
        error = TradingError("Test error")
        assert isinstance(error, Exception)

    def test_invalid_trade_error_basic(self) -> None:
        """Test InvalidTradeError with basic message."""
        error = InvalidTradeError("Invalid trade")
        assert str(error) == "Invalid trade"
        assert error.message == "Invalid trade"
        assert error.trade_details == {}
        assert error.reason is None

    def test_invalid_trade_error_attributes(self) -> None:
        """Test InvalidTradeError with all attributes."""
        error = InvalidTradeError(
            "Invalid trade",
            trade_details={"symbol": "AAPL", "quantity": "-100"},
            reason="negative_quantity",
        )
        assert error.message == "Invalid trade"
        assert error.trade_details == {"symbol": "AAPL", "quantity": "-100"}
        assert error.reason == "negative_quantity"

    def test_invalid_trade_error_inheritance(self) -> None:
        """Test InvalidTradeError inheritance."""
        error = InvalidTradeError("Invalid trade")
        assert isinstance(error, TradingError)
        assert isinstance(error, Exception)

    def test_insufficient_funds_error_basic(self) -> None:
        """Test InsufficientFundsError with basic message."""
        error = InsufficientFundsError("Not enough funds")
        assert str(error) == "Not enough funds"
        assert error.message == "Not enough funds"
        assert error.required is None
        assert error.available is None

    def test_insufficient_funds_error_attributes(self) -> None:
        """Test InsufficientFundsError with all attributes."""
        error = InsufficientFundsError(
            "Not enough funds",
            required=Decimal("10000.00"),
            available=Decimal("5000.00"),
        )
        assert error.message == "Not enough funds"
        assert error.required == Decimal("10000.00")
        assert error.available == Decimal("5000.00")

    def test_insufficient_funds_error_inheritance(self) -> None:
        """Test InsufficientFundsError inheritance."""
        error = InsufficientFundsError("Not enough funds")
        assert isinstance(error, TradingError)
        assert isinstance(error, Exception)

    def test_market_closed_error_basic(self) -> None:
        """Test MarketClosedError with basic message."""
        error = MarketClosedError("Market is closed")
        assert str(error) == "Market is closed"
        assert error.message == "Market is closed"
        assert error.symbol is None
        assert error.market_hours is None

    def test_market_closed_error_attributes(self) -> None:
        """Test MarketClosedError with all attributes."""
        error = MarketClosedError(
            "Market is closed",
            symbol="AAPL",
            market_hours="9:30 AM - 4:00 PM ET",
        )
        assert error.message == "Market is closed"
        assert error.symbol == "AAPL"
        assert error.market_hours == "9:30 AM - 4:00 PM ET"

    def test_market_closed_error_inheritance(self) -> None:
        """Test MarketClosedError inheritance."""
        error = MarketClosedError("Market is closed")
        assert isinstance(error, TradingError)
        assert isinstance(error, Exception)

    def test_exception_can_be_raised_and_caught(self) -> None:
        """Test that custom exceptions can be raised and caught."""
        with pytest.raises(TradingError):
            raise InvalidTradeError("Test")

        with pytest.raises(TradingError):
            raise InsufficientFundsError("Test")

        with pytest.raises(TradingError):
            raise MarketClosedError("Test")

    def test_exception_chaining(self) -> None:
        """Test exception chaining works correctly."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise InvalidTradeError("Wrapped error") from e
        except InvalidTradeError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"
