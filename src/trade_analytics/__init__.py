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

from .cli import (
    cmd_analyze,
    cmd_history,
    cmd_portfolio,
    create_parser,
    format_error,
    format_positions_table,
    format_trades_table,
    main,
    parse_date,
)
from .exceptions import (
    InsufficientFundsError,
    InvalidTradeError,
    MarketClosedError,
    TradingError,
)
from .models import MarketData, Position, Trade, TradeSide

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
    # CLI
    "main",
    "create_parser",
    "format_positions_table",
    "format_trades_table",
    "format_error",
    "parse_date",
    "cmd_portfolio",
    "cmd_history",
    "cmd_analyze",
]

__version__ = "0.1.0"
