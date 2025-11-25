"""Command-line interface for trade analytics.

This module provides a CLI for viewing portfolio positions,
trade history, and running analytics.

Usage:
    trade-analytics portfolio
    trade-analytics history --symbol AAPL --days 30
    trade-analytics analyze
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from .exceptions import TradingError
from .models import Position, Trade, TradeSide


__all__ = [
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


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="trade-analytics",
        description="Trade Analytics CLI - Manage and analyze your trading portfolio",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Portfolio command
    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="Show current portfolio positions",
    )
    portfolio_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    portfolio_parser.set_defaults(func=cmd_portfolio)

    # History command
    history_parser = subparsers.add_parser(
        "history",
        help="Show trade history",
    )
    history_parser.add_argument(
        "--symbol",
        type=str,
        help="Filter by ticker symbol",
    )
    history_parser.add_argument(
        "--start-date",
        type=parse_date,
        help="Start date (YYYY-MM-DD)",
    )
    history_parser.add_argument(
        "--end-date",
        type=parse_date,
        help="End date (YYYY-MM-DD)",
    )
    history_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to look back (default: 30)",
    )
    history_parser.set_defaults(func=cmd_history)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run portfolio analytics",
    )
    analyze_parser.add_argument(
        "--metric",
        choices=["pnl", "sharpe", "volatility", "all"],
        default="all",
        help="Metric to calculate (default: all)",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    return parser


def parse_date(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string to parse.

    Returns:
        Datetime object at midnight UTC.

    Raises:
        argparse.ArgumentTypeError: If date format is invalid.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD (e.g., 2024-01-15)"
        ) from e


def format_currency(amount: Decimal) -> str:
    """Format a decimal amount as currency.

    Args:
        amount: Decimal amount to format.

    Returns:
        Formatted currency string with $ sign and commas.
    """
    # Format with 2 decimal places and thousands separators
    if amount > 0:
        return f"+${amount:,.2f}"
    elif amount < 0:
        return f"-${abs(amount):,.2f}"
    else:
        return f"${amount:,.2f}"


def format_number(value: Decimal, decimals: int = 2) -> str:
    """Format a decimal number with specified decimal places.

    Args:
        value: Decimal value to format.
        decimals: Number of decimal places.

    Returns:
        Formatted number string.
    """
    format_str = f"{{:,.{decimals}f}}"
    return format_str.format(value)


def get_box_chars() -> Dict[str, str]:
    """Get box drawing characters with ASCII fallback for Windows.

    Returns:
        Dictionary of box drawing characters.
    """
    # Try to use Unicode box drawing characters
    # Fall back to ASCII if encoding not supported
    try:
        # Test if we can encode Unicode box chars
        test = "│"
        test.encode(sys.stdout.encoding or 'utf-8')
        return {
            "tl": "┌",
            "tr": "┐",
            "bl": "└",
            "br": "┘",
            "h": "─",
            "v": "│",
            "t": "┬",
            "b": "┴",
            "l": "├",
            "r": "┤",
            "c": "┼",
        }
    except (UnicodeEncodeError, AttributeError, LookupError):
        # Fall back to ASCII
        return {
            "tl": "+",
            "tr": "+",
            "bl": "+",
            "br": "+",
            "h": "-",
            "v": "|",
            "t": "+",
            "b": "+",
            "l": "+",
            "r": "+",
            "c": "+",
        }


def format_positions_table(
    positions: List[Position], *, include_totals: bool = True
) -> str:
    """Format a list of positions as an ASCII table.

    Args:
        positions: List of Position objects to format.
        include_totals: Whether to include a totals row.

    Returns:
        Formatted ASCII table string.

    Example:
        >>> positions = [Position("AAPL", Decimal("100"), Decimal("150.50"))]
        >>> print(format_positions_table(positions))
        +--------+----------+-----------+--------------+
        | Symbol | Quantity | Avg Price | Unrealized   |
        ...
    """
    if not positions:
        return "No positions found."

    # Box drawing characters with fallback for encoding issues
    chars = get_box_chars()

    # Calculate column widths
    symbol_width = max(len("Symbol"), max(len(p.symbol) for p in positions))
    quantity_width = max(
        len("Quantity"), max(len(format_number(p.quantity, 0)) for p in positions)
    )
    price_width = max(
        len("Avg Price"), max(len(format_currency(p.avg_price)) for p in positions)
    )
    pnl_width = max(
        len("Unrealized"), max(len(format_currency(p.unrealized_pnl)) for p in positions)
    )

    # Build table
    lines = []

    # Top border
    lines.append(
        chars["tl"]
        + chars["h"] * (symbol_width + 2)
        + chars["t"]
        + chars["h"] * (quantity_width + 2)
        + chars["t"]
        + chars["h"] * (price_width + 2)
        + chars["t"]
        + chars["h"] * (pnl_width + 2)
        + chars["tr"]
    )

    # Header
    lines.append(
        chars["v"]
        + " "
        + "Symbol".ljust(symbol_width)
        + " "
        + chars["v"]
        + " "
        + "Quantity".rjust(quantity_width)
        + " "
        + chars["v"]
        + " "
        + "Avg Price".rjust(price_width)
        + " "
        + chars["v"]
        + " "
        + "Unrealized".rjust(pnl_width)
        + " "
        + chars["v"]
    )

    # Header separator
    lines.append(
        chars["l"]
        + chars["h"] * (symbol_width + 2)
        + chars["c"]
        + chars["h"] * (quantity_width + 2)
        + chars["c"]
        + chars["h"] * (price_width + 2)
        + chars["c"]
        + chars["h"] * (pnl_width + 2)
        + chars["r"]
    )

    # Data rows
    for pos in positions:
        lines.append(
            chars["v"]
            + " "
            + pos.symbol.ljust(symbol_width)
            + " "
            + chars["v"]
            + " "
            + format_number(pos.quantity, 0).rjust(quantity_width)
            + " "
            + chars["v"]
            + " "
            + format_currency(pos.avg_price).rjust(price_width)
            + " "
            + chars["v"]
            + " "
            + format_currency(pos.unrealized_pnl).rjust(pnl_width)
            + " "
            + chars["v"]
        )

    # Bottom border
    lines.append(
        chars["bl"]
        + chars["h"] * (symbol_width + 2)
        + chars["b"]
        + chars["h"] * (quantity_width + 2)
        + chars["b"]
        + chars["h"] * (price_width + 2)
        + chars["b"]
        + chars["h"] * (pnl_width + 2)
        + chars["br"]
    )

    # Add totals if requested
    if include_totals:
        total_pnl = sum(p.unrealized_pnl for p in positions)
        lines.append(f"\nTotal Unrealized P&L: {format_currency(total_pnl)}")

    return "\n".join(lines)


def format_trades_table(trades: List[Trade], *, max_rows: int = 50) -> str:
    """Format a list of trades as an ASCII table.

    Args:
        trades: List of Trade objects to format.
        max_rows: Maximum number of rows to display.

    Returns:
        Formatted ASCII table string.
    """
    if not trades:
        return "No trades found."

    # Limit trades to max_rows
    display_trades = trades[:max_rows]
    truncated = len(trades) > max_rows

    # Box drawing characters with fallback for encoding issues
    chars = get_box_chars()

    # Calculate column widths
    timestamp_width = len("2024-01-15 10:30:00")
    symbol_width = max(len("Symbol"), max(len(t.symbol) for t in display_trades))
    side_width = len("SELL")
    quantity_width = max(
        len("Quantity"), max(len(format_number(t.quantity, 0)) for t in display_trades)
    )
    price_width = max(
        len("Price"), max(len(format_currency(t.price)) for t in display_trades)
    )

    # Build table
    lines = []

    # Top border
    lines.append(
        chars["tl"]
        + chars["h"] * (timestamp_width + 2)
        + chars["t"]
        + chars["h"] * (symbol_width + 2)
        + chars["t"]
        + chars["h"] * (side_width + 2)
        + chars["t"]
        + chars["h"] * (quantity_width + 2)
        + chars["t"]
        + chars["h"] * (price_width + 2)
        + chars["tr"]
    )

    # Header
    lines.append(
        chars["v"]
        + " "
        + "Timestamp".ljust(timestamp_width)
        + " "
        + chars["v"]
        + " "
        + "Symbol".ljust(symbol_width)
        + " "
        + chars["v"]
        + " "
        + "Side".ljust(side_width)
        + " "
        + chars["v"]
        + " "
        + "Quantity".rjust(quantity_width)
        + " "
        + chars["v"]
        + " "
        + "Price".rjust(price_width)
        + " "
        + chars["v"]
    )

    # Header separator
    lines.append(
        chars["l"]
        + chars["h"] * (timestamp_width + 2)
        + chars["c"]
        + chars["h"] * (symbol_width + 2)
        + chars["c"]
        + chars["h"] * (side_width + 2)
        + chars["c"]
        + chars["h"] * (quantity_width + 2)
        + chars["c"]
        + chars["h"] * (price_width + 2)
        + chars["r"]
    )

    # Data rows
    for trade in display_trades:
        # Format timestamp without timezone info for cleaner display
        ts_str = trade.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(
            chars["v"]
            + " "
            + ts_str.ljust(timestamp_width)
            + " "
            + chars["v"]
            + " "
            + trade.symbol.ljust(symbol_width)
            + " "
            + chars["v"]
            + " "
            + trade.side.value.ljust(side_width)
            + " "
            + chars["v"]
            + " "
            + format_number(trade.quantity, 0).rjust(quantity_width)
            + " "
            + chars["v"]
            + " "
            + format_currency(trade.price).rjust(price_width)
            + " "
            + chars["v"]
        )

    # Bottom border
    lines.append(
        chars["bl"]
        + chars["h"] * (timestamp_width + 2)
        + chars["b"]
        + chars["h"] * (symbol_width + 2)
        + chars["b"]
        + chars["h"] * (side_width + 2)
        + chars["b"]
        + chars["h"] * (quantity_width + 2)
        + chars["b"]
        + chars["h"] * (price_width + 2)
        + chars["br"]
    )

    # Add count and truncation notice
    if truncated:
        lines.append(
            f"\nShowing {len(display_trades)} of {len(trades)} trade(s) (use --limit to show more)"
        )
    else:
        lines.append(f"\n{len(trades)} trade(s) found")

    return "\n".join(lines)


def format_error(error: Exception) -> str:
    """Format an exception as a user-friendly error message.

    Args:
        error: Exception to format.

    Returns:
        Human-readable error message without stack trace.
    """
    if isinstance(error, argparse.ArgumentTypeError):
        return f"Error: {error}"
    elif isinstance(error, TradingError):
        return f"Error: {error.message}"
    else:
        return f"Unexpected error: {error}"


def get_sample_positions() -> List[Position]:
    """Get sample positions for demonstration.

    Returns:
        List of sample Position objects.
    """
    return [
        Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_price=Decimal("150.50"),
            unrealized_pnl=Decimal("250.00"),
        ),
        Position(
            symbol="GOOGL",
            quantity=Decimal("50"),
            avg_price=Decimal("2750.00"),
            unrealized_pnl=Decimal("-125.00"),
        ),
        Position(
            symbol="MSFT",
            quantity=Decimal("200"),
            avg_price=Decimal("380.25"),
            unrealized_pnl=Decimal("890.50"),
        ),
    ]


def get_sample_trades() -> List[Trade]:
    """Get sample trades for demonstration.

    Returns:
        List of sample Trade objects.
    """
    now = datetime.now(timezone.utc)
    return [
        Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=now - timedelta(days=15, hours=5, minutes=30),
            trade_id="T001",
        ),
        Trade(
            symbol="GOOGL",
            side=TradeSide.SELL,
            quantity=Decimal("25"),
            price=Decimal("2780.00"),
            timestamp=now - timedelta(days=14, hours=2, minutes=22, seconds=15),
            trade_id="T002",
        ),
        Trade(
            symbol="MSFT",
            side=TradeSide.BUY,
            quantity=Decimal("200"),
            price=Decimal("380.25"),
            timestamp=now - timedelta(days=10, hours=10),
            trade_id="T003",
        ),
        Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("148.75"),
            timestamp=now - timedelta(days=5, hours=14, minutes=45),
            trade_id="T004",
        ),
        Trade(
            symbol="GOOGL",
            side=TradeSide.BUY,
            quantity=Decimal("75"),
            price=Decimal("2750.00"),
            timestamp=now - timedelta(days=3, hours=9, minutes=15),
            trade_id="T005",
        ),
    ]


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Execute the portfolio command.

    Args:
        args: Parsed command-line arguments containing:
            - format: Output format ('table' or 'json')

    Returns:
        Exit code (0 for success, non-zero for error)

    Raises:
        TradingError: If portfolio data cannot be retrieved
    """
    try:
        # Get positions (using sample data for now)
        positions = get_sample_positions()

        if args.format == "json":
            # Output as JSON
            output = {
                "positions": [p.to_dict() for p in positions],
                "total_unrealized_pnl": str(sum(p.unrealized_pnl for p in positions)),
            }
            print(json.dumps(output, indent=2))
        else:
            # Output as table
            print(format_positions_table(positions))

        return 0
    except Exception as e:
        print(format_error(e), file=sys.stderr)
        return 1


def cmd_history(args: argparse.Namespace) -> int:
    """Execute the history command.

    Args:
        args: Parsed command-line arguments containing:
            - symbol: Optional ticker symbol filter
            - start_date: Optional start date
            - end_date: Optional end date
            - days: Number of days to look back (default 30)

    Returns:
        Exit code (0 for success, non-zero for error)

    Raises:
        TradingError: If history data cannot be retrieved
    """
    try:
        # Get trades (using sample data for now)
        trades = get_sample_trades()

        # Apply date filtering
        if args.start_date or args.end_date:
            # Use explicit date range if provided
            start = args.start_date or datetime.min.replace(tzinfo=timezone.utc)
            end = args.end_date or datetime.now(timezone.utc)
            trades = [t for t in trades if start <= t.timestamp <= end]
        else:
            # Use days lookback
            cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
            trades = [t for t in trades if t.timestamp >= cutoff]

        # Apply symbol filter
        if args.symbol:
            symbol_upper = args.symbol.upper()
            trades = [t for t in trades if t.symbol == symbol_upper]

        # Display results
        if args.symbol:
            print(f"Trade History for {args.symbol.upper()}:")
        else:
            print("Trade History:")

        if args.start_date and args.end_date:
            print(
                f"Date range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}"
            )
        elif args.start_date:
            print(f"From: {args.start_date.strftime('%Y-%m-%d')}")
        elif args.end_date:
            print(f"Until: {args.end_date.strftime('%Y-%m-%d')}")
        else:
            print(f"Last {args.days} days")

        print()
        print(format_trades_table(trades))

        return 0
    except Exception as e:
        print(format_error(e), file=sys.stderr)
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute the analyze command.

    Args:
        args: Parsed command-line arguments containing:
            - metric: Which metric(s) to calculate

    Returns:
        Exit code (0 for success, non-zero for error)

    Raises:
        TradingError: If analysis cannot be performed
    """
    try:
        # Get sample data
        positions = get_sample_positions()
        total_pnl = sum(p.unrealized_pnl for p in positions)

        print("Portfolio Analytics")
        print("=" * 50)

        if args.metric in ["pnl", "all"]:
            print(f"Total P&L:        {format_currency(total_pnl)}")

        if args.metric in ["sharpe", "all"]:
            # Placeholder - actual calculation requires historical data
            print(f"Sharpe Ratio:     1.45")

        if args.metric in ["volatility", "all"]:
            # Placeholder - actual calculation requires historical data
            print(f"Volatility:       12.3%")

        if args.metric == "all":
            print()
            print("Note: Analytics calculator (CORE-002) not yet implemented.")
            print("These are placeholder values for demonstration.")

        return 0
    except Exception as e:
        print(format_error(e), file=sys.stderr)
        return 1


def main() -> None:
    """Main entry point for the trade-analytics CLI.

    Parses command-line arguments and dispatches to the appropriate
    command handler. Exits with appropriate status code.

    Exit Codes:
        0: Success
        1: User error (invalid arguments, no data found)
        2: System error (file not found, database error)
    """
    parser = create_parser()

    try:
        args = parser.parse_args()
        exit_code = args.func(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        sys.exit(130)
    except TradingError as e:
        print(format_error(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
