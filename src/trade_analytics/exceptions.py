"""Custom exception hierarchy for trading-related errors.

This module provides a consistent error handling strategy with domain-specific
exceptions that provide meaningful error messages and enable proper error recovery.
"""

from decimal import Decimal
from typing import Optional


__all__ = [
    "TradingError",
    "InvalidTradeError",
    "InsufficientFundsError",
    "MarketClosedError",
]


class TradingError(Exception):
    """Base exception for all trading-related errors.

    All trading-specific exceptions inherit from this class, enabling
    catch-all handling of trading errors when appropriate.

    Attributes:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        """Initialize TradingError.

        Args:
            message: Human-readable error description.
        """
        self.message = message
        super().__init__(message)


class InvalidTradeError(TradingError):
    """Raised when trade parameters are invalid.

    This exception is raised when a trade cannot be executed due to
    invalid parameters such as empty symbol, negative quantity, or
    zero/negative price.

    Attributes:
        message: Human-readable error description.
        trade_details: Dictionary containing the invalid trade details.
        reason: Specific reason why the trade is invalid.
    """

    def __init__(
        self,
        message: str,
        trade_details: Optional[dict] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Initialize InvalidTradeError.

        Args:
            message: Human-readable error description.
            trade_details: Dictionary containing the invalid trade details.
            reason: Specific reason why the trade is invalid.
        """
        super().__init__(message)
        self.trade_details = trade_details or {}
        self.reason = reason


class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds for a trade.

    This exception is raised when a trade cannot be executed because
    the account does not have enough funds to cover the transaction.

    Attributes:
        message: Human-readable error description.
        required: The amount of funds required for the trade.
        available: The amount of funds currently available.
    """

    def __init__(
        self,
        message: str,
        required: Optional[Decimal] = None,
        available: Optional[Decimal] = None,
    ) -> None:
        """Initialize InsufficientFundsError.

        Args:
            message: Human-readable error description.
            required: The amount of funds required for the trade.
            available: The amount of funds currently available.
        """
        super().__init__(message)
        self.required = required
        self.available = available


class MarketClosedError(TradingError):
    """Raised when attempting to trade in a closed market.

    This exception is raised when a trade cannot be executed because
    the market for the specified symbol is currently closed.

    Attributes:
        message: Human-readable error description.
        symbol: The ticker symbol of the market that is closed.
        market_hours: String describing the market's trading hours.
    """

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        market_hours: Optional[str] = None,
    ) -> None:
        """Initialize MarketClosedError.

        Args:
            message: Human-readable error description.
            symbol: The ticker symbol of the market that is closed.
            market_hours: String describing the market's trading hours.
        """
        super().__init__(message)
        self.symbol = symbol
        self.market_hours = market_hours
