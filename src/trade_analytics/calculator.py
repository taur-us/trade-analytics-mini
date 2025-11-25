"""Portfolio analytics calculator for computing portfolio metrics.

This module provides the PortfolioCalculator class with methods for calculating
portfolio-level metrics such as total value, P&L, and exposure by symbol.
"""

from decimal import Decimal
from typing import Dict, List

from .exceptions import MissingMarketDataError
from .models import MarketData, Position


__all__ = [
    "PortfolioCalculator",
]


class PortfolioCalculator:
    """Portfolio analytics calculator for computing portfolio metrics.

    This class provides static methods for calculating portfolio-level
    metrics such as total value, P&L, and exposure by symbol.

    All methods are pure functions that do not modify input data.
    """

    @staticmethod
    def calculate_total_value(
        positions: List[Position],
        market_data: Dict[str, MarketData],
    ) -> Decimal:
        """Calculate the total market value of all positions.

        Args:
            positions: List of portfolio positions.
            market_data: Dictionary mapping symbols to current market data.

        Returns:
            Total portfolio market value (sum of quantity × last price).
            Returns Decimal("0") for empty portfolios.

        Raises:
            MissingMarketDataError: If market data is missing for any position's symbol.

        Example:
            >>> positions = [Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))]
            >>> market_data = {"AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.50"), volume=1000)}
            >>> PortfolioCalculator.calculate_total_value(positions, market_data)
            Decimal("15550.00")
        """
        if not positions:
            return Decimal("0")

        total_value = Decimal("0")
        available_symbols = list(market_data.keys())

        for position in positions:
            if position.symbol not in market_data:
                raise MissingMarketDataError(
                    f"No market data available for symbol. Symbol: {position.symbol}. "
                    f"Available: {available_symbols}",
                    symbol=position.symbol,
                    available_symbols=available_symbols,
                )
            current_price = market_data[position.symbol].last
            position_value = position.quantity * current_price
            total_value += position_value

        return total_value

    @staticmethod
    def calculate_pnl(
        positions: List[Position],
        market_data: Dict[str, MarketData],
    ) -> Decimal:
        """Calculate the total unrealized P&L across all positions.

        Args:
            positions: List of portfolio positions.
            market_data: Dictionary mapping symbols to current market data.

        Returns:
            Total unrealized P&L (sum of quantity × (current_price - avg_price)).
            Returns Decimal("0") for empty portfolios.
            Positive values indicate profit, negative values indicate loss.

        Raises:
            MissingMarketDataError: If market data is missing for any position's symbol.

        Example:
            >>> positions = [Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))]
            >>> market_data = {"AAPL": MarketData(symbol="AAPL", bid=Decimal("155"), ask=Decimal("156"), last=Decimal("155.50"), volume=1000)}
            >>> PortfolioCalculator.calculate_pnl(positions, market_data)
            Decimal("550.00")  # 100 × (155.50 - 150.00)
        """
        if not positions:
            return Decimal("0")

        total_pnl = Decimal("0")
        available_symbols = list(market_data.keys())

        for position in positions:
            if position.symbol not in market_data:
                raise MissingMarketDataError(
                    f"No market data available for symbol. Symbol: {position.symbol}. "
                    f"Available: {available_symbols}",
                    symbol=position.symbol,
                    available_symbols=available_symbols,
                )
            current_price = market_data[position.symbol].last
            position_pnl = position.quantity * (current_price - position.avg_price)
            total_pnl += position_pnl

        return total_pnl

    @staticmethod
    def calculate_exposure_by_symbol(
        positions: List[Position],
        market_data: Dict[str, MarketData],
    ) -> Dict[str, Decimal]:
        """Calculate the absolute value exposure for each symbol.

        Exposure represents the total market value at risk for each symbol,
        calculated as the absolute value of (quantity × current price).
        Both long and short positions contribute positive exposure.

        Args:
            positions: List of portfolio positions.
            market_data: Dictionary mapping symbols to current market data.

        Returns:
            Dictionary mapping symbols to their absolute exposure values.
            Returns empty dict for empty portfolios.

        Raises:
            MissingMarketDataError: If market data is missing for any position's symbol.

        Example:
            >>> positions = [
            ...     Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150")),
            ...     Position(symbol="AAPL", quantity=Decimal("50"), avg_price=Decimal("155")),
            ... ]
            >>> # Returns aggregated exposure per unique symbol
        """
        if not positions:
            return {}

        exposure: Dict[str, Decimal] = {}
        available_symbols = list(market_data.keys())

        for position in positions:
            if position.symbol not in market_data:
                raise MissingMarketDataError(
                    f"No market data available for symbol. Symbol: {position.symbol}. "
                    f"Available: {available_symbols}",
                    symbol=position.symbol,
                    available_symbols=available_symbols,
                )
            current_price = market_data[position.symbol].last
            position_value = position.quantity * current_price

            if position.symbol in exposure:
                # Aggregate positions for the same symbol
                exposure[position.symbol] += position_value
            else:
                exposure[position.symbol] = position_value

        # Convert to absolute values (exposure is always positive)
        return {symbol: abs(value) for symbol, value in exposure.items()}
