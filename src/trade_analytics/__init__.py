"""Trade Analytics - Core data models and exceptions for trading analytics.

This package provides foundational data structures for representing trading domain
entities including trades, positions, and market data, along with a custom exception
hierarchy for trading-related errors.

Example usage:
    from trade_analytics import Trade, TradeSide, Position, MarketData
    from trade_analytics import InvalidTradeError, InsufficientFundsError

    trade = Trade(
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.50"),
        timestamp=datetime.now(timezone.utc)
    )
"""

from .exceptions import (
    InsufficientFundsError,
    InvalidTradeError,
    MarketClosedError,
    StorageError,
    TradingError,
)
from .models import MarketData, Position, Trade, TradeSide
from .storage import TradeStore

__all__ = [
    # Models
    "TradeSide",
    "Trade",
    "Position",
    "MarketData",
    # Exceptions
    "TradingError",
    "InvalidTradeError",
    "InsufficientFundsError",
    "MarketClosedError",
    "StorageError",
    # Storage
    "TradeStore",
]

__version__ = "0.1.0"
