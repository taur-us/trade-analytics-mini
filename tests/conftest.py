"""Pytest fixtures for trade_analytics tests."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List

import pytest

from trade_analytics.models import MarketData, Position, Trade, TradeSide


@pytest.fixture
def sample_trade() -> Trade:
    """Create a sample valid trade for testing."""
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
    """Create a sample valid position for testing."""
    return Position(
        symbol="AAPL",
        quantity=Decimal("100"),
        avg_price=Decimal("150.50"),
        unrealized_pnl=Decimal("250.00"),
    )


@pytest.fixture
def sample_market_data() -> MarketData:
    """Create a sample valid market data for testing."""
    return MarketData(
        symbol="AAPL",
        bid=Decimal("150.45"),
        ask=Decimal("150.55"),
        last=Decimal("150.50"),
        volume=1000000,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    )


# Calculator test fixtures


@pytest.fixture
def multi_position_portfolio() -> List[Position]:
    """Create a multi-position portfolio for calculator tests."""
    return [
        Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("150.00"),
        ),
        Position(
            symbol="GOOGL",
            quantity=Decimal("-50"),  # Short position
            avg_price=Decimal("140.00"),
        ),
        Position(
            symbol="MSFT",
            quantity=Decimal("200"),
            avg_price=Decimal("380.00"),
        ),
    ]


@pytest.fixture
def multi_symbol_market_data() -> Dict[str, MarketData]:
    """Create market data for multiple symbols."""
    return {
        "AAPL": MarketData(
            symbol="AAPL",
            bid=Decimal("155.00"),
            ask=Decimal("155.10"),
            last=Decimal("155.05"),
            volume=1000000,
        ),
        "GOOGL": MarketData(
            symbol="GOOGL",
            bid=Decimal("145.00"),
            ask=Decimal("145.10"),
            last=Decimal("145.05"),
            volume=500000,
        ),
        "MSFT": MarketData(
            symbol="MSFT",
            bid=Decimal("385.00"),
            ask=Decimal("385.20"),
            last=Decimal("385.10"),
            volume=750000,
        ),
    }


@pytest.fixture
def single_position() -> Position:
    """Create a single position for simple tests."""
    return Position(
        symbol="AAPL",
        quantity=Decimal("100"),
        avg_price=Decimal("150.00"),
    )


@pytest.fixture
def single_market_data() -> Dict[str, MarketData]:
    """Create market data for a single symbol."""
    return {
        "AAPL": MarketData(
            symbol="AAPL",
            bid=Decimal("155.00"),
            ask=Decimal("155.10"),
            last=Decimal("155.05"),
            volume=1000000,
        ),
    }
