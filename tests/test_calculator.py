"""Tests for PortfolioCalculator and MissingMarketDataError."""

from decimal import Decimal
from typing import Dict, List

import pytest

from trade_analytics import (
    MarketData,
    MissingMarketDataError,
    PortfolioCalculator,
    Position,
    TradingError,
)


class TestCalculateTotalValue:
    """Tests for PortfolioCalculator.calculate_total_value."""

    def test_single_position(
        self,
        single_position: Position,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test total value calculation with a single position."""
        result = PortfolioCalculator.calculate_total_value(
            [single_position], single_market_data
        )
        # 100 shares × $155.05 = $15,505.00
        assert result == Decimal("15505.00")

    def test_multiple_positions(
        self,
        multi_position_portfolio: List[Position],
        multi_symbol_market_data: Dict[str, MarketData],
    ) -> None:
        """Test total value calculation with multiple positions."""
        result = PortfolioCalculator.calculate_total_value(
            multi_position_portfolio, multi_symbol_market_data
        )
        # AAPL: 100 × 155.05 = 15505.00
        # GOOGL: -50 × 145.05 = -7252.50 (short position)
        # MSFT: 200 × 385.10 = 77020.00
        # Total: 15505.00 - 7252.50 + 77020.00 = 85272.50
        expected = Decimal("15505.00") - Decimal("7252.50") + Decimal("77020.00")
        assert result == expected

    def test_empty_portfolio_returns_zero(self) -> None:
        """Test that empty portfolio returns zero."""
        result = PortfolioCalculator.calculate_total_value([], {})
        assert result == Decimal("0")

    def test_missing_market_data_raises_error(
        self,
        single_position: Position,
    ) -> None:
        """Test that missing market data raises MissingMarketDataError."""
        with pytest.raises(MissingMarketDataError) as exc_info:
            PortfolioCalculator.calculate_total_value([single_position], {})
        assert exc_info.value.symbol == "AAPL"
        assert exc_info.value.available_symbols == []

    def test_zero_quantity_position(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test that zero quantity position contributes zero value."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            avg_price=Decimal("150.00"),
        )
        result = PortfolioCalculator.calculate_total_value(
            [position], single_market_data
        )
        assert result == Decimal("0")


class TestCalculatePnl:
    """Tests for PortfolioCalculator.calculate_pnl."""

    def test_long_position_profit(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test P&L calculation for long position with profit."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("150.00"),  # Bought at $150
        )
        # Current price is $155.05
        result = PortfolioCalculator.calculate_pnl([position], single_market_data)
        # 100 × (155.05 - 150.00) = 505.00
        assert result == Decimal("505.00")

    def test_long_position_loss(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test P&L calculation for long position with loss."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("160.00"),  # Bought at $160
        )
        # Current price is $155.05
        result = PortfolioCalculator.calculate_pnl([position], single_market_data)
        # 100 × (155.05 - 160.00) = -495.00
        assert result == Decimal("-495.00")

    def test_short_position_profit(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test P&L calculation for short position with profit (price went down)."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("-100"),  # Short position
            avg_price=Decimal("160.00"),  # Sold short at $160
        )
        # Current price is $155.05
        result = PortfolioCalculator.calculate_pnl([position], single_market_data)
        # -100 × (155.05 - 160.00) = -100 × -4.95 = 495.00 (profit)
        assert result == Decimal("495.00")

    def test_short_position_loss(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test P&L calculation for short position with loss (price went up)."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("-100"),  # Short position
            avg_price=Decimal("150.00"),  # Sold short at $150
        )
        # Current price is $155.05
        result = PortfolioCalculator.calculate_pnl([position], single_market_data)
        # -100 × (155.05 - 150.00) = -100 × 5.05 = -505.00 (loss)
        assert result == Decimal("-505.00")

    def test_multiple_positions_mixed(
        self,
        multi_position_portfolio: List[Position],
        multi_symbol_market_data: Dict[str, MarketData],
    ) -> None:
        """Test P&L calculation with mixed long and short positions."""
        result = PortfolioCalculator.calculate_pnl(
            multi_position_portfolio, multi_symbol_market_data
        )
        # AAPL: 100 × (155.05 - 150.00) = 505.00 (profit)
        # GOOGL: -50 × (145.05 - 140.00) = -50 × 5.05 = -252.50 (loss on short)
        # MSFT: 200 × (385.10 - 380.00) = 200 × 5.10 = 1020.00 (profit)
        # Total: 505.00 - 252.50 + 1020.00 = 1272.50
        expected = Decimal("505.00") - Decimal("252.50") + Decimal("1020.00")
        assert result == expected

    def test_empty_portfolio_returns_zero(self) -> None:
        """Test that empty portfolio returns zero P&L."""
        result = PortfolioCalculator.calculate_pnl([], {})
        assert result == Decimal("0")

    def test_missing_market_data_raises_error(
        self,
        single_position: Position,
    ) -> None:
        """Test that missing market data raises MissingMarketDataError."""
        with pytest.raises(MissingMarketDataError) as exc_info:
            PortfolioCalculator.calculate_pnl([single_position], {})
        assert exc_info.value.symbol == "AAPL"

    def test_breakeven_returns_zero(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test P&L returns zero when price equals avg_price."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("155.05"),  # Same as current price
        )
        result = PortfolioCalculator.calculate_pnl([position], single_market_data)
        assert result == Decimal("0")


class TestCalculateExposureBySymbol:
    """Tests for PortfolioCalculator.calculate_exposure_by_symbol."""

    def test_single_position(
        self,
        single_position: Position,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test exposure calculation with a single position."""
        result = PortfolioCalculator.calculate_exposure_by_symbol(
            [single_position], single_market_data
        )
        # abs(100 × 155.05) = 15505.00
        assert result == {"AAPL": Decimal("15505.00")}

    def test_multiple_symbols(
        self,
        multi_position_portfolio: List[Position],
        multi_symbol_market_data: Dict[str, MarketData],
    ) -> None:
        """Test exposure calculation with multiple symbols."""
        result = PortfolioCalculator.calculate_exposure_by_symbol(
            multi_position_portfolio, multi_symbol_market_data
        )
        # AAPL: abs(100 × 155.05) = 15505.00
        # GOOGL: abs(-50 × 145.05) = 7252.50
        # MSFT: abs(200 × 385.10) = 77020.00
        assert result == {
            "AAPL": Decimal("15505.00"),
            "GOOGL": Decimal("7252.50"),
            "MSFT": Decimal("77020.00"),
        }

    def test_aggregates_same_symbol(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test that positions in the same symbol are aggregated."""
        positions = [
            Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150")),
            Position(symbol="AAPL", quantity=Decimal("50"), avg_price=Decimal("155")),
        ]
        result = PortfolioCalculator.calculate_exposure_by_symbol(
            positions, single_market_data
        )
        # (100 × 155.05) + (50 × 155.05) = 15505.00 + 7752.50 = 23257.50
        assert result == {"AAPL": Decimal("23257.50")}

    def test_short_position_positive_exposure(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test that short positions contribute positive exposure."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("-100"),  # Short
            avg_price=Decimal("150.00"),
        )
        result = PortfolioCalculator.calculate_exposure_by_symbol(
            [position], single_market_data
        )
        # abs(-100 × 155.05) = 15505.00
        assert result == {"AAPL": Decimal("15505.00")}

    def test_empty_portfolio_returns_empty_dict(self) -> None:
        """Test that empty portfolio returns empty dict."""
        result = PortfolioCalculator.calculate_exposure_by_symbol([], {})
        assert result == {}

    def test_missing_market_data_raises_error(
        self,
        single_position: Position,
    ) -> None:
        """Test that missing market data raises MissingMarketDataError."""
        with pytest.raises(MissingMarketDataError) as exc_info:
            PortfolioCalculator.calculate_exposure_by_symbol([single_position], {})
        assert exc_info.value.symbol == "AAPL"

    def test_mixed_long_short_same_symbol(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test aggregation with mixed long and short positions in same symbol."""
        positions = [
            Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150")),
            Position(symbol="AAPL", quantity=Decimal("-30"), avg_price=Decimal("155")),
        ]
        result = PortfolioCalculator.calculate_exposure_by_symbol(
            positions, single_market_data
        )
        # (100 × 155.05) + (-30 × 155.05) = 15505.00 - 4651.50 = 10853.50
        # abs(10853.50) = 10853.50
        assert result == {"AAPL": Decimal("10853.50")}


class TestMissingMarketDataError:
    """Tests for MissingMarketDataError exception."""

    def test_exception_attributes(self) -> None:
        """Test that exception has correct attributes."""
        error = MissingMarketDataError(
            "Missing data for TSLA",
            symbol="TSLA",
            available_symbols=["AAPL", "GOOGL"],
        )
        assert error.symbol == "TSLA"
        assert error.available_symbols == ["AAPL", "GOOGL"]
        assert error.message == "Missing data for TSLA"

    def test_exception_inheritance(self) -> None:
        """Test that exception inherits from TradingError."""
        error = MissingMarketDataError("Test error")
        assert isinstance(error, TradingError)
        assert isinstance(error, Exception)

    def test_error_message_format(self) -> None:
        """Test error message string representation."""
        error = MissingMarketDataError(
            "No market data available for symbol. Symbol: TSLA. Available: ['AAPL', 'GOOGL']"
        )
        assert str(error) == "No market data available for symbol. Symbol: TSLA. Available: ['AAPL', 'GOOGL']"

    def test_default_available_symbols(self) -> None:
        """Test that available_symbols defaults to empty list."""
        error = MissingMarketDataError("Test", symbol="TSLA")
        assert error.available_symbols == []

    def test_can_be_raised_and_caught(self) -> None:
        """Test that exception can be raised and caught."""
        with pytest.raises(MissingMarketDataError):
            raise MissingMarketDataError("Test error", symbol="TSLA")

        # Can also catch as TradingError
        with pytest.raises(TradingError):
            raise MissingMarketDataError("Test error", symbol="TSLA")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_position(
        self,
        single_market_data: Dict[str, MarketData],
    ) -> None:
        """Test calculations with very large position size."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("1000000"),
            avg_price=Decimal("150.00"),
        )
        result = PortfolioCalculator.calculate_total_value(
            [position], single_market_data
        )
        # 1,000,000 × 155.05 = 155,050,000.00
        assert result == Decimal("155050000.00")

    def test_high_precision_decimals(
        self,
    ) -> None:
        """Test calculations with high precision decimal values."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100.123456789"),
            avg_price=Decimal("150.987654321"),
        )
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155.111111111"),
                ask=Decimal("155.222222222"),
                last=Decimal("155.123456789"),
                volume=1000000,
            ),
        }
        # Should not raise any errors
        result = PortfolioCalculator.calculate_total_value([position], market_data)
        assert result is not None

    def test_partial_missing_market_data(
        self,
        multi_position_portfolio: List[Position],
    ) -> None:
        """Test that partial market data raises error for missing symbol."""
        partial_market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155.00"),
                ask=Decimal("155.10"),
                last=Decimal("155.05"),
                volume=1000000,
            ),
            # GOOGL and MSFT missing
        }
        with pytest.raises(MissingMarketDataError) as exc_info:
            PortfolioCalculator.calculate_total_value(
                multi_position_portfolio, partial_market_data
            )
        # Should fail on GOOGL (second position)
        assert exc_info.value.symbol == "GOOGL"
        assert "AAPL" in exc_info.value.available_symbols

    def test_zero_price_market_data(self) -> None:
        """Test calculations with zero price market data."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("0"),  # Zero avg_price is valid
        )
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("0"),
                ask=Decimal("0"),
                last=Decimal("0"),
                volume=0,
            ),
        }
        result = PortfolioCalculator.calculate_total_value([position], market_data)
        assert result == Decimal("0")

        pnl = PortfolioCalculator.calculate_pnl([position], market_data)
        assert pnl == Decimal("0")

    def test_multiple_calls_same_data(
        self,
        multi_position_portfolio: List[Position],
        multi_symbol_market_data: Dict[str, MarketData],
    ) -> None:
        """Test that multiple calls with same data return consistent results."""
        result1 = PortfolioCalculator.calculate_total_value(
            multi_position_portfolio, multi_symbol_market_data
        )
        result2 = PortfolioCalculator.calculate_total_value(
            multi_position_portfolio, multi_symbol_market_data
        )
        assert result1 == result2

    def test_calculator_does_not_modify_inputs(
        self,
        multi_position_portfolio: List[Position],
        multi_symbol_market_data: Dict[str, MarketData],
    ) -> None:
        """Test that calculator methods don't modify input data."""
        original_positions = [p.quantity for p in multi_position_portfolio]
        original_symbols = list(multi_symbol_market_data.keys())

        PortfolioCalculator.calculate_total_value(
            multi_position_portfolio, multi_symbol_market_data
        )
        PortfolioCalculator.calculate_pnl(
            multi_position_portfolio, multi_symbol_market_data
        )
        PortfolioCalculator.calculate_exposure_by_symbol(
            multi_position_portfolio, multi_symbol_market_data
        )

        # Verify inputs unchanged
        assert [p.quantity for p in multi_position_portfolio] == original_positions
        assert list(multi_symbol_market_data.keys()) == original_symbols
