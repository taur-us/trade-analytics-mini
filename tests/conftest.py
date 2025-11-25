"""Pytest fixtures for trade_analytics tests."""

from datetime import datetime, timezone
from decimal import Decimal

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
