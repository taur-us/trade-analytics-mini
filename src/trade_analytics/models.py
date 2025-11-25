"""Data models for the trade analytics system.

This module provides foundational data structures for representing trading domain
entities including trades, positions, and market data. All models use type hints
and validation to ensure data integrity.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict

from .exceptions import InvalidTradeError


__all__ = [
    "TradeSide",
    "Trade",
    "Position",
    "MarketData",
]


class TradeSide(Enum):
    """Enumeration representing the direction of a trade.

    Attributes:
        BUY: Represents a buy order (going long).
        SELL: Represents a sell order (going short or closing long).
    """

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Trade:
    """Represents a trade execution.

    A frozen (immutable) dataclass representing a single trade execution.
    Includes validation to ensure all trade parameters are valid.

    Attributes:
        symbol: Ticker symbol (e.g., "AAPL"). Will be uppercased.
        side: Trade direction (BUY or SELL).
        quantity: Number of shares/units (must be positive).
        price: Execution price (must be positive).
        timestamp: Execution timestamp (UTC).
        trade_id: Optional unique identifier for the trade.

    Raises:
        InvalidTradeError: If any validation fails.
    """

    symbol: str
    side: TradeSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    trade_id: str = ""

    def __post_init__(self) -> None:
        """Validate trade parameters after initialization."""
        # Validate symbol
        if not self.symbol or not self.symbol.strip():
            raise InvalidTradeError(
                "Symbol cannot be empty",
                trade_details={"symbol": self.symbol},
                reason="empty_symbol",
            )

        # Uppercase the symbol (need to use object.__setattr__ due to frozen)
        object.__setattr__(self, "symbol", self.symbol.upper().strip())

        # Validate quantity
        if self.quantity <= 0:
            raise InvalidTradeError(
                f"Quantity must be positive, got {self.quantity}",
                trade_details={"quantity": str(self.quantity)},
                reason="invalid_quantity",
            )

        # Validate price
        if self.price <= 0:
            raise InvalidTradeError(
                f"Price must be positive, got {self.price}",
                trade_details={"price": str(self.price)},
                reason="invalid_price",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Trade to dictionary representation.

        Returns:
            Dictionary with trade data, suitable for serialization.
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "timestamp": self.timestamp.isoformat(),
            "trade_id": self.trade_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Create Trade from dictionary representation.

        Args:
            data: Dictionary containing trade data.

        Returns:
            Trade instance.

        Raises:
            InvalidTradeError: If data is invalid.
            KeyError: If required fields are missing.
            ValueError: If data types are incorrect.
        """
        return cls(
            symbol=data["symbol"],
            side=TradeSide(data["side"]),
            quantity=Decimal(data["quantity"]),
            price=Decimal(data["price"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            trade_id=data.get("trade_id", ""),
        )


@dataclass
class Position:
    """Represents a portfolio position.

    A dataclass representing a position in a specific security.
    Positions can be long (positive quantity) or short (negative quantity).

    Attributes:
        symbol: Ticker symbol (e.g., "AAPL"). Will be uppercased.
        quantity: Net position (positive=long, negative=short, zero=flat).
        avg_price: Volume-weighted average entry price (must be non-negative).
        unrealized_pnl: Unrealized profit/loss.

    Raises:
        InvalidTradeError: If any validation fails.
    """

    symbol: str
    quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))

    def __post_init__(self) -> None:
        """Validate position parameters after initialization."""
        # Validate symbol
        if not self.symbol or not self.symbol.strip():
            raise InvalidTradeError(
                "Symbol cannot be empty",
                trade_details={"symbol": self.symbol},
                reason="empty_symbol",
            )

        # Uppercase the symbol
        self.symbol = self.symbol.upper().strip()

        # Validate avg_price
        if self.avg_price < 0:
            raise InvalidTradeError(
                f"Average price cannot be negative, got {self.avg_price}",
                trade_details={"avg_price": str(self.avg_price)},
                reason="invalid_avg_price",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Position to dictionary representation.

        Returns:
            Dictionary with position data, suitable for serialization.
        """
        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "avg_price": str(self.avg_price),
            "unrealized_pnl": str(self.unrealized_pnl),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create Position from dictionary representation.

        Args:
            data: Dictionary containing position data.

        Returns:
            Position instance.

        Raises:
            InvalidTradeError: If data is invalid.
            KeyError: If required fields are missing.
        """
        return cls(
            symbol=data["symbol"],
            quantity=Decimal(data["quantity"]),
            avg_price=Decimal(data["avg_price"]),
            unrealized_pnl=Decimal(data.get("unrealized_pnl", "0")),
        )


@dataclass(frozen=True)
class MarketData:
    """Represents current market quote data.

    A frozen (immutable) dataclass representing a market quote for a security.
    Includes bid/ask prices, last traded price, and volume.

    Attributes:
        symbol: Ticker symbol (e.g., "AAPL"). Will be uppercased.
        bid: Best bid price (must be non-negative).
        ask: Best ask price (must be non-negative and >= bid).
        last: Last traded price (must be non-negative).
        volume: Trading volume (must be non-negative).
        timestamp: Quote timestamp (UTC). Defaults to current time.

    Raises:
        InvalidTradeError: If any validation fails.
    """

    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate market data parameters after initialization."""
        # Validate symbol
        if not self.symbol or not self.symbol.strip():
            raise InvalidTradeError(
                "Symbol cannot be empty",
                trade_details={"symbol": self.symbol},
                reason="empty_symbol",
            )

        # Uppercase the symbol (need to use object.__setattr__ due to frozen)
        object.__setattr__(self, "symbol", self.symbol.upper().strip())

        # Validate bid
        if self.bid < 0:
            raise InvalidTradeError(
                f"Bid price cannot be negative, got {self.bid}",
                trade_details={"bid": str(self.bid)},
                reason="invalid_bid",
            )

        # Validate ask
        if self.ask < 0:
            raise InvalidTradeError(
                f"Ask price cannot be negative, got {self.ask}",
                trade_details={"ask": str(self.ask)},
                reason="invalid_ask",
            )

        # Validate spread (bid <= ask)
        if self.bid > self.ask:
            raise InvalidTradeError(
                f"Invalid spread: bid ({self.bid}) > ask ({self.ask})",
                trade_details={"bid": str(self.bid), "ask": str(self.ask)},
                reason="invalid_spread",
            )

        # Validate last
        if self.last < 0:
            raise InvalidTradeError(
                f"Last price cannot be negative, got {self.last}",
                trade_details={"last": str(self.last)},
                reason="invalid_last",
            )

        # Validate volume
        if self.volume < 0:
            raise InvalidTradeError(
                f"Volume cannot be negative, got {self.volume}",
                trade_details={"volume": self.volume},
                reason="invalid_volume",
            )

    @property
    def spread(self) -> Decimal:
        """Calculate the bid-ask spread.

        Returns:
            The difference between ask and bid prices.
        """
        return self.ask - self.bid

    @property
    def mid(self) -> Decimal:
        """Calculate the mid price.

        Returns:
            The average of bid and ask prices.
        """
        return (self.bid + self.ask) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert MarketData to dictionary representation.

        Returns:
            Dictionary with market data, suitable for serialization.
        """
        return {
            "symbol": self.symbol,
            "bid": str(self.bid),
            "ask": str(self.ask),
            "last": str(self.last),
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketData":
        """Create MarketData from dictionary representation.

        Args:
            data: Dictionary containing market data.

        Returns:
            MarketData instance.

        Raises:
            InvalidTradeError: If data is invalid.
            KeyError: If required fields are missing.
        """
        return cls(
            symbol=data["symbol"],
            bid=Decimal(data["bid"]),
            ask=Decimal(data["ask"]),
            last=Decimal(data["last"]),
            volume=int(data["volume"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(timezone.utc),
        )
