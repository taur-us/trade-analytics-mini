"""Command-line interface for the trade analytics system.

This module provides a CLI for interacting with portfolio analytics,
viewing trade history, and running analytics calculations.

Usage:
    trade-analytics portfolio [--symbol SYMBOL] [--format {table,json,csv}]
    trade-analytics history [--symbol SYMBOL] [--start-date DATE] [--end-date DATE]
    trade-analytics analyze [--symbol SYMBOL] [--metric {pnl,exposure,value,all}]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .calculator import PortfolioCalculator

# Version is defined here to avoid circular imports
__version__ = "0.1.0"
from .exceptions import TradingError
from .models import MarketData, Position, Trade, TradeSide


__all__ = [
    "main",
    "create_parser",
    "cmd_portfolio",
    "cmd_history",
    "cmd_analyze",
]


# =============================================================================
# Demo Data Functions (to be replaced with storage layer later)
# =============================================================================


def get_demo_positions() -> List[Position]:
    """Get demo portfolio positions for development.

    Returns:
        List of sample Position objects.

    Note:
        This will be replaced with storage.get_positions() when STORE-001 is ready.
    """
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


def get_demo_trades() -> List[Trade]:
    """Get demo trade history for development.

    Returns:
        List of sample Trade objects.

    Note:
        This will be replaced with storage.get_trades() when STORE-001 is ready.
    """
    now = datetime.now(timezone.utc)
    return [
        Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            timestamp=now - timedelta(days=1, hours=2),
            trade_id="T001",
        ),
        Trade(
            symbol="GOOGL",
            side=TradeSide.SELL,
            quantity=Decimal("50"),
            price=Decimal("142.00"),
            timestamp=now - timedelta(days=2, hours=5),
            trade_id="T002",
        ),
        Trade(
            symbol="MSFT",
            side=TradeSide.BUY,
            quantity=Decimal("200"),
            price=Decimal("380.00"),
            timestamp=now - timedelta(days=3, hours=1),
            trade_id="T003",
        ),
        Trade(
            symbol="AAPL",
            side=TradeSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("148.00"),
            timestamp=now - timedelta(days=5),
            trade_id="T004",
        ),
        Trade(
            symbol="GOOGL",
            side=TradeSide.BUY,
            quantity=Decimal("25"),
            price=Decimal("138.00"),
            timestamp=now - timedelta(days=7),
            trade_id="T005",
        ),
    ]


def get_demo_market_data() -> Dict[str, MarketData]:
    """Get demo market data for development.

    Returns:
        Dictionary mapping symbols to MarketData objects.

    Note:
        This will be replaced with market data provider when available.
    """
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


# =============================================================================
# Helper Functions
# =============================================================================


def print_error(message: str) -> None:
    """Print an error message to stderr.

    Args:
        message: Error message to display.
    """
    print(f"Error: {message}", file=sys.stderr)


def parse_date(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        datetime object with UTC timezone.

    Raises:
        ValueError: If the date format is invalid.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(
            f"Invalid date format '{date_str}'. Please use YYYY-MM-DD format (e.g., 2024-01-15)"
        )


def format_decimal(value: Decimal, decimals: int = 2, prefix: str = "") -> str:
    """Format a Decimal value for display.

    Args:
        value: Decimal value to format.
        decimals: Number of decimal places.
        prefix: Optional prefix (e.g., "$" or "+").

    Returns:
        Formatted string representation.
    """
    formatted = f"{value:,.{decimals}f}"
    return f"{prefix}{formatted}"


def format_currency(value: Decimal) -> str:
    """Format a Decimal value as currency.

    Args:
        value: Decimal value to format.

    Returns:
        Formatted string with dollar sign and commas.
    """
    return f"${value:,.2f}"


def format_pnl(value: Decimal) -> str:
    """Format a P&L value with sign indicator.

    Args:
        value: P&L value to format.

    Returns:
        Formatted string with +/- prefix and dollar sign.
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}${value:,.2f}"


def format_percent(value: Decimal) -> str:
    """Format a percentage value with sign indicator.

    Args:
        value: Percentage value to format.

    Returns:
        Formatted string with +/- prefix and % suffix.
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


# =============================================================================
# Table Formatting Functions
# =============================================================================


def format_positions_table(
    positions: List[Position],
    market_data: Dict[str, MarketData],
) -> str:
    """Format positions as a pretty table.

    Args:
        positions: List of portfolio positions.
        market_data: Dictionary of current market data.

    Returns:
        Formatted table string ready for printing.
    """
    if not positions:
        return "No positions found."

    lines: List[str] = []
    lines.append("")
    lines.append("Portfolio Positions")
    lines.append("=" * 19)
    lines.append("")

    # Define column headers and widths
    headers = ["Symbol", "Qty", "Avg Price", "Market", "Value", "P&L", "P&L %"]
    widths = [8, 10, 12, 12, 14, 14, 10]

    # Header line
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Track totals
    total_value = Decimal("0")
    total_cost = Decimal("0")

    # Data rows
    for pos in positions:
        md = market_data.get(pos.symbol)
        if not md:
            continue

        value = pos.quantity * md.last
        cost = pos.quantity * pos.avg_price
        pnl = value - cost
        pnl_pct = (pnl / abs(cost)) * 100 if cost != 0 else Decimal(0)

        total_value += value
        total_cost += cost

        row = [
            pos.symbol,
            f"{pos.quantity:,}",
            format_currency(pos.avg_price),
            format_currency(md.last),
            format_currency(value),
            format_pnl(pnl),
            format_percent(pnl_pct),
        ]

        row_line = "  ".join(
            val.rjust(w) if i > 0 else val.ljust(w)
            for i, (val, w) in enumerate(zip(row, widths))
        )
        lines.append(row_line)

    # Summary line
    lines.append("-" * len(header_line))
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / abs(total_cost)) * 100 if total_cost != 0 else Decimal(0)

    summary = [
        "Total",
        "",
        "",
        "",
        format_currency(total_value),
        format_pnl(total_pnl),
        format_percent(total_pnl_pct),
    ]
    summary_line = "  ".join(
        val.rjust(w) if i > 0 else val.ljust(w)
        for i, (val, w) in enumerate(zip(summary, widths))
    )
    lines.append(summary_line)
    lines.append("")

    return "\n".join(lines)


def format_positions_json(
    positions: List[Position],
    market_data: Dict[str, MarketData],
) -> str:
    """Format positions as JSON.

    Args:
        positions: List of portfolio positions.
        market_data: Dictionary of current market data.

    Returns:
        JSON string representation.
    """
    data = []
    for pos in positions:
        md = market_data.get(pos.symbol)
        if not md:
            continue

        value = pos.quantity * md.last
        cost = pos.quantity * pos.avg_price
        pnl = value - cost
        pnl_pct = (pnl / abs(cost)) * 100 if cost != 0 else Decimal(0)

        data.append({
            "symbol": pos.symbol,
            "quantity": str(pos.quantity),
            "avg_price": str(pos.avg_price),
            "market_price": str(md.last),
            "value": str(value),
            "pnl": str(pnl),
            "pnl_percent": str(pnl_pct),
        })

    return json.dumps(data, indent=2)


def format_positions_csv(
    positions: List[Position],
    market_data: Dict[str, MarketData],
) -> str:
    """Format positions as CSV.

    Args:
        positions: List of portfolio positions.
        market_data: Dictionary of current market data.

    Returns:
        CSV string representation.
    """
    lines = ["Symbol,Quantity,Avg Price,Market Price,Value,P&L,P&L %"]

    for pos in positions:
        md = market_data.get(pos.symbol)
        if not md:
            continue

        value = pos.quantity * md.last
        cost = pos.quantity * pos.avg_price
        pnl = value - cost
        pnl_pct = (pnl / abs(cost)) * 100 if cost != 0 else Decimal(0)

        lines.append(
            f"{pos.symbol},{pos.quantity},{pos.avg_price},{md.last},{value},{pnl},{pnl_pct}"
        )

    return "\n".join(lines)


def format_trades_table(trades: List[Trade]) -> str:
    """Format trades as a pretty table.

    Args:
        trades: List of trades to format.

    Returns:
        Formatted table string ready for printing.
    """
    if not trades:
        return "No trades found."

    lines: List[str] = []
    lines.append("")
    lines.append("Trade History")
    lines.append("=" * 13)
    lines.append("")

    # Define column headers and widths
    headers = ["Date", "Symbol", "Side", "Qty", "Price", "Total", "ID"]
    widths = [18, 8, 6, 10, 12, 14, 10]

    # Header line
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Data rows
    for trade in trades:
        total = trade.quantity * trade.price

        row = [
            trade.timestamp.strftime("%Y-%m-%d %H:%M"),
            trade.symbol,
            trade.side.value,
            f"{trade.quantity:,}",
            format_currency(trade.price),
            format_currency(total),
            trade.trade_id[:10] if trade.trade_id else "",
        ]

        row_line = "  ".join(
            val.rjust(w) if i > 2 else val.ljust(w)
            for i, (val, w) in enumerate(zip(row, widths))
        )
        lines.append(row_line)

    lines.append("")
    lines.append(f"{len(trades)} trade(s) shown")
    lines.append("")

    return "\n".join(lines)


def format_analytics_report(
    positions: List[Position],
    market_data: Dict[str, MarketData],
    metrics: List[str],
) -> str:
    """Format analytics results as a report.

    Args:
        positions: List of portfolio positions.
        market_data: Dictionary of current market data.
        metrics: List of metrics to include.

    Returns:
        Formatted report string ready for printing.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("Portfolio Analytics")
    lines.append("=" * 19)
    lines.append("")

    # Calculate metrics
    total_value = PortfolioCalculator.calculate_total_value(positions, market_data)
    total_pnl = PortfolioCalculator.calculate_pnl(positions, market_data)
    exposure = PortfolioCalculator.calculate_exposure_by_symbol(positions, market_data)

    # Calculate total cost for P&L percentage
    total_cost = sum(pos.quantity * pos.avg_price for pos in positions)
    pnl_pct = (total_pnl / abs(total_cost)) * 100 if total_cost != 0 else Decimal(0)

    # Define column widths for metrics table
    label_width = 24
    value_width = 16

    if "all" in metrics or "value" in metrics:
        lines.append(f"{'Total Portfolio Value':<{label_width}}{format_currency(total_value):>{value_width}}")

    if "all" in metrics or "pnl" in metrics:
        lines.append(f"{'Total Unrealized P&L':<{label_width}}{format_pnl(total_pnl):>{value_width}}")
        lines.append(f"{'P&L Percentage':<{label_width}}{format_percent(pnl_pct):>{value_width}}")

    if "all" in metrics or "exposure" in metrics:
        lines.append("")
        lines.append("Exposure by Symbol:")

        total_exposure = sum(exposure.values())
        for symbol, exp_value in sorted(exposure.items()):
            exp_pct = (exp_value / total_exposure) * 100 if total_exposure > 0 else Decimal(0)
            lines.append(f"  {symbol}:  {format_currency(exp_value):>14} ({exp_pct:.1f}%)")

    lines.append("")

    return "\n".join(lines)


def format_analytics_json(
    positions: List[Position],
    market_data: Dict[str, MarketData],
    metrics: List[str],
) -> str:
    """Format analytics results as JSON.

    Args:
        positions: List of portfolio positions.
        market_data: Dictionary of current market data.
        metrics: List of metrics to include.

    Returns:
        JSON string representation.
    """
    result: Dict[str, Any] = {}

    total_value = PortfolioCalculator.calculate_total_value(positions, market_data)
    total_pnl = PortfolioCalculator.calculate_pnl(positions, market_data)
    exposure = PortfolioCalculator.calculate_exposure_by_symbol(positions, market_data)

    total_cost = sum(pos.quantity * pos.avg_price for pos in positions)
    pnl_pct = (total_pnl / abs(total_cost)) * 100 if total_cost != 0 else Decimal(0)

    if "all" in metrics or "value" in metrics:
        result["total_value"] = str(total_value)

    if "all" in metrics or "pnl" in metrics:
        result["total_pnl"] = str(total_pnl)
        result["pnl_percent"] = str(pnl_pct)

    if "all" in metrics or "exposure" in metrics:
        result["exposure"] = {symbol: str(val) for symbol, val in exposure.items()}

    return json.dumps(result, indent=2)


# =============================================================================
# Command Handlers
# =============================================================================


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Handle the 'portfolio' subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    try:
        positions = get_demo_positions()
        market_data = get_demo_market_data()

        # Filter by symbol if specified
        if args.symbol:
            symbol = args.symbol.upper()
            positions = [p for p in positions if p.symbol == symbol]
            if not positions:
                available = [p.symbol for p in get_demo_positions()]
                print_error(
                    f"Symbol '{symbol}' not found in portfolio.\n"
                    f"Available symbols: {', '.join(available)}"
                )
                return 1

        # Format output
        if args.format == "json":
            output = format_positions_json(positions, market_data)
        elif args.format == "csv":
            output = format_positions_csv(positions, market_data)
        else:
            output = format_positions_table(positions, market_data)

        print(output)
        return 0

    except TradingError as e:
        print_error(e.message)
        return 1


def cmd_history(args: argparse.Namespace) -> int:
    """Handle the 'history' subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    try:
        trades = get_demo_trades()

        # Filter by symbol if specified
        if args.symbol:
            symbol = args.symbol.upper()
            trades = [t for t in trades if t.symbol == symbol]

        # Filter by side if specified
        if args.side:
            side = TradeSide(args.side.upper())
            trades = [t for t in trades if t.side == side]

        # Filter by date range
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None

        if args.days:
            start_date = datetime.now(timezone.utc) - timedelta(days=args.days)
        elif args.start_date:
            start_date = parse_date(args.start_date)

        if args.end_date:
            end_date = parse_date(args.end_date)
            # Set to end of day
            end_date = end_date.replace(hour=23, minute=59, second=59)

        # Validate date range
        if start_date and end_date and start_date > end_date:
            print_error("Start date must be before end date.")
            return 1

        # Apply date filters
        if start_date:
            trades = [t for t in trades if t.timestamp >= start_date]
        if end_date:
            trades = [t for t in trades if t.timestamp <= end_date]

        # Sort by timestamp (newest first)
        trades.sort(key=lambda t: t.timestamp, reverse=True)

        # Apply limit
        if args.limit and args.limit > 0:
            trades = trades[: args.limit]

        # Format output
        output = format_trades_table(trades)
        print(output)
        return 0

    except ValueError as e:
        print_error(str(e))
        return 1
    except TradingError as e:
        print_error(e.message)
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle the 'analyze' subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    try:
        positions = get_demo_positions()
        market_data = get_demo_market_data()

        # Filter by symbol if specified
        if args.symbol:
            symbol = args.symbol.upper()
            positions = [p for p in positions if p.symbol == symbol]
            if not positions:
                available = [p.symbol for p in get_demo_positions()]
                print_error(
                    f"Symbol '{symbol}' not found in portfolio.\n"
                    f"Available symbols: {', '.join(available)}"
                )
                return 1

        # Determine which metrics to show
        metrics = [args.metric] if args.metric != "all" else ["all"]

        # Format output
        if args.format == "json":
            output = format_analytics_json(positions, market_data, metrics)
        else:
            output = format_analytics_report(positions, market_data, metrics)

        print(output)
        return 0

    except TradingError as e:
        print_error(e.message)
        return 1


# =============================================================================
# Argument Parser
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="trade-analytics",
        description="Trade Analytics CLI - Portfolio analysis and tracking",
        epilog="For more information, see the documentation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
    )

    # Portfolio subcommand
    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="Show current portfolio positions",
        description="Display current portfolio positions with market values and P&L",
    )
    portfolio_parser.add_argument(
        "-s", "--symbol",
        metavar="SYMBOL",
        help="Filter by symbol (e.g., AAPL)",
    )
    portfolio_parser.add_argument(
        "-f", "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    portfolio_parser.set_defaults(func=cmd_portfolio)

    # History subcommand
    history_parser = subparsers.add_parser(
        "history",
        help="Show trade history",
        description="Display trade history with optional filtering by date, symbol, or side",
    )
    history_parser.add_argument(
        "-s", "--symbol",
        metavar="SYMBOL",
        help="Filter by symbol (e.g., AAPL)",
    )
    history_parser.add_argument(
        "--start-date",
        metavar="DATE",
        help="Start date (YYYY-MM-DD)",
    )
    history_parser.add_argument(
        "--end-date",
        metavar="DATE",
        help="End date (YYYY-MM-DD)",
    )
    history_parser.add_argument(
        "-d", "--days",
        type=int,
        metavar="N",
        help="Show last N days of trades",
    )
    history_parser.add_argument(
        "--side",
        choices=["buy", "sell", "BUY", "SELL"],
        help="Filter by trade side (buy or sell)",
    )
    history_parser.add_argument(
        "-n", "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of trades to show (default: 50)",
    )
    history_parser.set_defaults(func=cmd_history)

    # Analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run portfolio analytics",
        description="Run portfolio analytics and display results",
    )
    analyze_parser.add_argument(
        "-s", "--symbol",
        metavar="SYMBOL",
        help="Analyze specific symbol",
    )
    analyze_parser.add_argument(
        "-m", "--metric",
        choices=["pnl", "exposure", "value", "all"],
        default="all",
        help="Specific metric to calculate (default: all)",
    )
    analyze_parser.add_argument(
        "-f", "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the trade-analytics CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for user error, 2 for system error.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except TradingError as e:
        print_error(f"Trading error: {e.message}")
        return 1
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        return 1
    except KeyboardInterrupt:
        print_error("\nOperation cancelled.")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
