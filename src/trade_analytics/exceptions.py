"""Custom exception hierarchy for trading-related errors.

This module provides a consistent error handling strategy with domain-specific
exceptions that provide meaningful error messages and enable proper error recovery.
"""

from decimal import Decimal
from typing import List, Optional


__all__ = [
    "TradingError",
    "InvalidTradeError",
    "InsufficientFundsError",
    "MarketClosedError",
    "MissingMarketDataError",
    # Storage exceptions
    "StorageError",
    "RecordNotFoundError",
    "DuplicateRecordError",
    "DatabaseConnectionError",
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


class MissingMarketDataError(TradingError):
    """Raised when market data is unavailable for a required symbol.

    This exception is raised when a portfolio calculation requires market
    data for a symbol that is not present in the provided market data dictionary.

    Attributes:
        message: Human-readable error description.
        symbol: The symbol for which market data is missing.
        available_symbols: List of symbols that have market data available.
    """

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        available_symbols: Optional[List[str]] = None,
    ) -> None:
        """Initialize MissingMarketDataError.

        Args:
            message: Human-readable error description.
            symbol: The symbol for which market data is missing.
            available_symbols: List of symbols that have market data available.
        """
        super().__init__(message)
        self.symbol = symbol
        self.available_symbols = available_symbols or []


class StorageError(TradingError):
    """Base exception for all storage-related errors.

    This exception is raised when a database operation fails due to
    connection issues, query errors, or other storage-related problems.

    Attributes:
        message: Human-readable error description.
        operation: The storage operation that failed (e.g., "read", "write").
        original_error: The underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize StorageError.

        Args:
            message: Human-readable error description.
            operation: The storage operation that failed.
            original_error: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


class RecordNotFoundError(StorageError):
    """Raised when a requested record doesn't exist.

    This exception is raised when attempting to retrieve, update, or delete
    a record that doesn't exist in the database.

    Attributes:
        message: Human-readable error description.
        record_type: The type of record that was not found (e.g., "Trade", "Position").
        record_id: The identifier of the record that was not found.
    """

    def __init__(
        self,
        message: str,
        record_type: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> None:
        """Initialize RecordNotFoundError.

        Args:
            message: Human-readable error description.
            record_type: The type of record that was not found.
            record_id: The identifier of the record that was not found.
        """
        super().__init__(message, operation="read")
        self.record_type = record_type
        self.record_id = record_id


class DuplicateRecordError(StorageError):
    """Raised when attempting to create a duplicate record.

    This exception is raised when attempting to insert a record with a
    primary key or unique constraint that already exists.

    Attributes:
        message: Human-readable error description.
        record_type: The type of record that caused the duplicate.
        record_id: The identifier that is duplicated.
    """

    def __init__(
        self,
        message: str,
        record_type: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> None:
        """Initialize DuplicateRecordError.

        Args:
            message: Human-readable error description.
            record_type: The type of record that caused the duplicate.
            record_id: The identifier that is duplicated.
        """
        super().__init__(message, operation="create")
        self.record_type = record_type
        self.record_id = record_id


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails.

    This exception is raised when the database connection cannot be
    established or is lost during an operation.

    Attributes:
        message: Human-readable error description.
        database_path: The path to the database that failed to connect.
        original_error: The underlying exception that caused the connection failure.
    """

    def __init__(
        self,
        message: str,
        database_path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize DatabaseConnectionError.

        Args:
            message: Human-readable error description.
            database_path: The path to the database that failed to connect.
            original_error: The underlying exception that caused the connection failure.
        """
        super().__init__(message, operation="connect", original_error=original_error)
        self.database_path = database_path
