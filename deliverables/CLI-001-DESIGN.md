# Technical Design Document: CLI-001

## Create CLI Interface

**Task ID:** CLI-001
**Priority:** HIGH
**Estimated Hours:** 2.0
**Author:** Technical Lead
**Date:** 2024-11-26
**Status:** DRAFT

---

## 1. Problem Summary

The trade-analytics-mini system currently lacks a user-facing interface. While the core data models (`Trade`, `Position`, `MarketData`) and exception hierarchy exist, there is no way for users to interact with the system from the command line.

This task addresses the need for:
- **User accessibility**: Provide a CLI entry point for portfolio management and analytics
- **Data visibility**: Allow users to view current positions in a readable table format
- **Historical analysis**: Enable users to query trade history with date filtering
- **Analytics execution**: Give users the ability to run portfolio analytics on demand

The CLI serves as the primary user interface for the trading analytics system, making the underlying functionality accessible to traders and analysts.

---

## 2. Current State

### Existing Codebase Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| `src/trade_analytics/__init__.py` | ✅ Exists | Exports models and exceptions |
| `src/trade_analytics/models.py` | ✅ Exists | Trade, Position, MarketData dataclasses |
| `src/trade_analytics/exceptions.py` | ✅ Exists | TradingError hierarchy |
| `src/trade_analytics/cli.py` | ❌ **Does not exist** | Needs to be created |
| `tests/test_cli.py` | ❌ **Does not exist** | Needs to be created |

### Available Data Models

The CLI will work with the following existing models:

```python
# From src/trade_analytics/models.py
class TradeSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass(frozen=True)
class Trade:
    symbol: str
    side: TradeSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    trade_id: str = ""

@dataclass
class Position:
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
```

### Available Exceptions

```python
# From src/trade_analytics/exceptions.py
class TradingError(Exception): ...
class InvalidTradeError(TradingError): ...
class InsufficientFundsError(TradingError): ...
class MarketClosedError(TradingError): ...
```

### Dependencies

- **Upstream**: This module depends on `models.py` and `exceptions.py` (complete)
- **Downstream**: None (this is a user-facing interface)
- **Future Integration**: Will integrate with storage layer (STORE-001) and analytics calculator (CORE-002) when available

---

## 3. Proposed Solution

### High-Level Approach

Build a command-line interface using Python's `argparse` module with subcommands for each operation. The CLI will:

1. **Use argparse subparsers** for clean command separation (`portfolio`, `history`, `analyze`)
2. **Implement pretty table output** using a lightweight table formatter for positions
3. **Support date range filtering** via `--start-date` and `--end-date` arguments
4. **Provide user-friendly error handling** with clear messages and appropriate exit codes

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  cli.py                              │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │  portfolio  │ │   history   │ │   analyze   │   │   │
│  │  │   command   │ │   command   │ │   command   │   │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │   │
│  │         │               │               │           │   │
│  │         ▼               ▼               ▼           │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │           Output Formatters                  │   │   │
│  │  │  (table_formatter, error_handler)           │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Core Layer (existing)     │
              │  models.py  │  exceptions.py  │
              └───────────────────────────────┘
```

### Design Principles

- **Simplicity**: Use standard library `argparse` (no external dependencies)
- **Testability**: Separate parsing logic from execution logic for easy unit testing
- **Extensibility**: Modular command structure allows easy addition of new commands
- **User Experience**: Clear help messages, intuitive defaults, meaningful error messages

---

## 4. Components

### 4.1 Module: `src/trade_analytics/cli.py`

#### Functions to Implement

| Function | Purpose | Signature |
|----------|---------|-----------|
| `create_parser()` | Build the argparse parser with subcommands | `() -> argparse.ArgumentParser` |
| `cmd_portfolio(args)` | Execute portfolio command | `(argparse.Namespace) -> int` |
| `cmd_history(args)` | Execute history command with date filtering | `(argparse.Namespace) -> int` |
| `cmd_analyze(args)` | Execute analyze command | `(argparse.Namespace) -> int` |
| `format_positions_table(positions)` | Format positions as ASCII table | `(List[Position]) -> str` |
| `format_trades_table(trades)` | Format trades as ASCII table | `(List[Trade]) -> str` |
| `format_error(error)` | Format exception as user-friendly message | `(Exception) -> str` |
| `parse_date(date_str)` | Parse date string to datetime | `(str) -> datetime` |
| `main()` | Entry point for CLI | `() -> None` |

### 4.2 Module: `tests/test_cli.py`

#### Test Classes

| Test Class | Coverage Target |
|------------|-----------------|
| `TestCreateParser` | Parser creation and configuration |
| `TestCmdPortfolio` | Portfolio command execution |
| `TestCmdHistory` | History command with date filtering |
| `TestCmdAnalyze` | Analyze command execution |
| `TestTableFormatters` | Table output formatting |
| `TestErrorHandling` | Error formatting and exit codes |
| `TestDateParsing` | Date argument parsing |

---

## 5. Data Models

### 5.1 CLI Command Arguments

The CLI uses the existing `Trade` and `Position` models. No new data models are required.

#### Portfolio Command

```
trade-analytics portfolio [--format {table,json}]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--format` | choice | `table` | Output format (table or json) |

#### History Command

```
trade-analytics history [--symbol SYMBOL] [--start-date DATE] [--end-date DATE] [--days N]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbol` | string | None | Filter by ticker symbol |
| `--start-date` | date | None | Start date (YYYY-MM-DD) |
| `--end-date` | date | None | End date (YYYY-MM-DD) |
| `--days` | int | 30 | Number of days to look back |

#### Analyze Command

```
trade-analytics analyze [--metric {pnl,sharpe,volatility,all}]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--metric` | choice | `all` | Specific metric to calculate |

### 5.2 Table Output Schemas

#### Positions Table

```
┌────────┬──────────┬───────────┬──────────────┐
│ Symbol │ Quantity │ Avg Price │ Unrealized   │
├────────┼──────────┼───────────┼──────────────┤
│ AAPL   │      100 │   $150.50 │      +$250.00│
│ GOOGL  │       50 │  $2750.00 │     -$125.00 │
│ MSFT   │      200 │   $380.25 │      +$890.50│
└────────┴──────────┴───────────┴──────────────┘
```

#### Trades Table

```
┌──────────────────────┬────────┬──────┬──────────┬───────────┐
│ Timestamp            │ Symbol │ Side │ Quantity │ Price     │
├──────────────────────┼────────┼──────┼──────────┼───────────┤
│ 2024-01-15 10:30:00  │ AAPL   │ BUY  │      100 │   $150.50 │
│ 2024-01-16 14:22:15  │ GOOGL  │ SELL │       25 │ $2,780.00 │
└──────────────────────┴────────┴──────┴──────────┴───────────┘
```

---

## 6. API Contracts

### 6.1 CLI Entry Point

```python
def main() -> None:
    """Main entry point for the trade-analytics CLI.

    Parses command-line arguments and dispatches to the appropriate
    command handler. Exits with appropriate status code.

    Exit Codes:
        0: Success
        1: User error (invalid arguments, no data found)
        2: System error (file not found, database error)
    """
```

### 6.2 Command Handlers

```python
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
```

### 6.3 Table Formatters

```python
def format_positions_table(
    positions: List[Position],
    *,
    include_totals: bool = True
) -> str:
    """Format a list of positions as an ASCII table.

    Args:
        positions: List of Position objects to format
        include_totals: Whether to include a totals row

    Returns:
        Formatted ASCII table string

    Example:
        >>> positions = [Position("AAPL", Decimal("100"), Decimal("150.50"))]
        >>> print(format_positions_table(positions))
        ┌────────┬──────────┬───────────┬──────────────┐
        │ Symbol │ Quantity │ Avg Price │ Unrealized   │
        ...
    """

def format_trades_table(
    trades: List[Trade],
    *,
    max_rows: int = 50
) -> str:
    """Format a list of trades as an ASCII table.

    Args:
        trades: List of Trade objects to format
        max_rows: Maximum number of rows to display

    Returns:
        Formatted ASCII table string
    """
```

### 6.4 Utility Functions

```python
def parse_date(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string to parse

    Returns:
        Datetime object at midnight UTC

    Raises:
        argparse.ArgumentTypeError: If date format is invalid
    """

def format_error(error: Exception) -> str:
    """Format an exception as a user-friendly error message.

    Args:
        error: Exception to format

    Returns:
        Human-readable error message without stack trace
    """
```

---

## 7. Error Handling

### 7.1 Error Categories and Exit Codes

| Exit Code | Category | Examples |
|-----------|----------|----------|
| 0 | Success | Command completed successfully |
| 1 | User Error | Invalid arguments, no data found, invalid date format |
| 2 | System Error | File not found, database connection failed |

### 7.2 Error Message Format

```
Error: {brief description}

{details if available}

Hint: {suggestion for resolution}
```

Example:
```
Error: Invalid date format for --start-date

Expected format: YYYY-MM-DD (e.g., 2024-01-15)
Received: "Jan 15, 2024"

Hint: Use ISO date format like --start-date 2024-01-15
```

### 7.3 Exception Mapping

| Exception Type | User Message | Exit Code |
|----------------|--------------|-----------|
| `argparse.ArgumentError` | "Invalid argument: {details}" | 1 |
| `InvalidTradeError` | "Trade validation error: {reason}" | 1 |
| `FileNotFoundError` | "Data file not found: {path}" | 2 |
| `ConnectionError` | "Database connection failed" | 2 |
| `TradingError` | "{exception.message}" | 1 |
| `Exception` (unexpected) | "Unexpected error occurred" | 2 |

### 7.4 Error Handling Strategy

```python
def main() -> None:
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
```

---

## 8. Implementation Plan

### Phase 1: Core CLI Structure (30 min)

| Step | Task | File |
|------|------|------|
| 1.1 | Create `cli.py` with imports and module docstring | `src/trade_analytics/cli.py` |
| 1.2 | Implement `create_parser()` with subparsers | `src/trade_analytics/cli.py` |
| 1.3 | Add argument definitions for `portfolio` command | `src/trade_analytics/cli.py` |
| 1.4 | Add argument definitions for `history` command | `src/trade_analytics/cli.py` |
| 1.5 | Add argument definitions for `analyze` command | `src/trade_analytics/cli.py` |
| 1.6 | Implement `main()` entry point with error handling | `src/trade_analytics/cli.py` |

### Phase 2: Table Formatters (30 min)

| Step | Task | File |
|------|------|------|
| 2.1 | Implement `format_positions_table()` with box drawing | `src/trade_analytics/cli.py` |
| 2.2 | Implement `format_trades_table()` with box drawing | `src/trade_analytics/cli.py` |
| 2.3 | Implement currency/number formatting helpers | `src/trade_analytics/cli.py` |
| 2.4 | Add column alignment and width calculation | `src/trade_analytics/cli.py` |

### Phase 3: Command Handlers (30 min)

| Step | Task | File |
|------|------|------|
| 3.1 | Implement `cmd_portfolio()` with table output | `src/trade_analytics/cli.py` |
| 3.2 | Implement `parse_date()` utility function | `src/trade_analytics/cli.py` |
| 3.3 | Implement `cmd_history()` with date filtering logic | `src/trade_analytics/cli.py` |
| 3.4 | Implement `cmd_analyze()` placeholder | `src/trade_analytics/cli.py` |

### Phase 4: Error Handling & Polish (15 min)

| Step | Task | File |
|------|------|------|
| 4.1 | Implement `format_error()` for user-friendly messages | `src/trade_analytics/cli.py` |
| 4.2 | Add comprehensive help text to all commands | `src/trade_analytics/cli.py` |
| 4.3 | Add `__all__` exports and complete docstrings | `src/trade_analytics/cli.py` |

### Phase 5: Unit Tests (30 min)

| Step | Task | File |
|------|------|------|
| 5.1 | Create `test_cli.py` with test fixtures | `tests/test_cli.py` |
| 5.2 | Write parser creation tests | `tests/test_cli.py` |
| 5.3 | Write table formatter tests | `tests/test_cli.py` |
| 5.4 | Write command handler tests with mocks | `tests/test_cli.py` |
| 5.5 | Write error handling tests | `tests/test_cli.py` |
| 5.6 | Write date parsing tests | `tests/test_cli.py` |

### Phase 6: Integration (15 min)

| Step | Task | File |
|------|------|------|
| 6.1 | Update `__init__.py` with CLI exports | `src/trade_analytics/__init__.py` |
| 6.2 | Add console_scripts entry point to setup | `pyproject.toml` or `setup.py` |
| 6.3 | Run full test suite and verify coverage | - |
| 6.4 | Manual testing of all commands | - |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **No data layer available yet** | High | Medium | Use mock/stub data sources; design for dependency injection |
| **Terminal width issues** | Medium | Low | Detect terminal width; truncate or wrap long values |
| **Date timezone confusion** | Medium | Medium | Use UTC consistently; document timezone handling |
| **Unicode box drawing not supported** | Low | Low | Fallback to ASCII characters (`|`, `-`, `+`) |
| **Large output overwhelms terminal** | Medium | Low | Implement pagination or `--limit` option |
| **External dependencies add complexity** | Low | Medium | Use only stdlib; no rich/tabulate dependencies |

### Mitigation Details

**No Data Layer Available:**
```python
# Design with dependency injection for testability
def cmd_portfolio(args: argparse.Namespace, *, data_source=None) -> int:
    """Allow injection of data source for testing."""
    if data_source is None:
        # Use default/real data source when available
        data_source = get_default_data_source()
    positions = data_source.get_positions()
    ...
```

**Terminal Width Detection:**
```python
import shutil

def get_terminal_width() -> int:
    """Get terminal width with fallback for non-TTY environments."""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80  # Default fallback
```

**Unicode Fallback:**
```python
def get_box_chars() -> dict:
    """Get box drawing characters with ASCII fallback."""
    try:
        # Test if terminal supports unicode
        print("│", end="", file=io.StringIO())
        return {"h": "─", "v": "│", "tl": "┌", "tr": "┐", ...}
    except UnicodeEncodeError:
        return {"h": "-", "v": "|", "tl": "+", "tr": "+", ...}
```

---

## 10. Success Criteria

### Functional Requirements

| Criteria | Verification Method |
|----------|-------------------|
| CLI entry point works with `python -m trade_analytics.cli` | Manual test |
| `portfolio` command displays positions in table format | Unit test + manual test |
| `history` command accepts and validates date arguments | Unit test |
| `history --days N` filters to last N days | Unit test |
| `history --symbol AAPL` filters by symbol | Unit test |
| `analyze` command runs without error | Unit test |
| Invalid arguments produce helpful error messages | Unit test |
| Keyboard interrupt (Ctrl+C) exits gracefully | Manual test |

### Quality Requirements

| Criteria | Target | Verification |
|----------|--------|--------------|
| Test coverage | ≥ 90% | `pytest --cov=src/trade_analytics/cli --cov-report=term-missing` |
| Type hints | 100% | `mypy src/trade_analytics/cli.py` passes |
| Docstrings | All public APIs | Manual review |
| Help text | All commands | `trade-analytics --help`, `trade-analytics portfolio --help` |

### Test Cases (Minimum)

```
tests/test_cli.py:
  ✓ test_parser_creation
  ✓ test_parser_portfolio_subcommand
  ✓ test_parser_history_subcommand
  ✓ test_parser_history_with_dates
  ✓ test_parser_analyze_subcommand
  ✓ test_format_positions_table_empty
  ✓ test_format_positions_table_single
  ✓ test_format_positions_table_multiple
  ✓ test_format_trades_table_empty
  ✓ test_format_trades_table_with_data
  ✓ test_parse_date_valid
  ✓ test_parse_date_invalid_format
  ✓ test_format_error_trading_error
  ✓ test_format_error_generic
  ✓ test_cmd_portfolio_success
  ✓ test_cmd_history_with_date_filter
  ✓ test_cmd_history_with_symbol_filter
  ✓ test_cmd_analyze_success
  ✓ test_main_keyboard_interrupt
  ✓ test_main_trading_error
```

### Acceptance Criteria Mapping

| Acceptance Criteria | Implementation | Test |
|---------------------|----------------|------|
| CLI entry point in `src/trade_analytics/cli.py` | `main()` function | `test_parser_creation` |
| Commands: portfolio, history, analyze | `create_parser()` with subparsers | `test_parser_*_subcommand` |
| Pretty table output for positions | `format_positions_table()` | `test_format_positions_table_*` |
| Date range filtering for history | `--start-date`, `--end-date`, `--days` args | `test_cmd_history_with_date_filter` |
| Error handling with user-friendly messages | `format_error()` + try/except in `main()` | `test_format_error_*`, `test_main_*` |

---

## Appendix A: File Structure After Implementation

```
trade-analytics-mini/
├── src/
│   └── trade_analytics/
│       ├── __init__.py          # Updated with CLI exports
│       ├── cli.py               # NEW: CLI implementation
│       ├── exceptions.py        # Existing
│       └── models.py            # Existing
├── tests/
│   ├── __init__.py              # Existing
│   ├── conftest.py              # Update with CLI fixtures
│   ├── test_cli.py              # NEW: CLI tests
│   └── test_models.py           # Existing
└── ...
```

## Appendix B: Example Usage

```bash
# Display help
$ trade-analytics --help
usage: trade-analytics [-h] {portfolio,history,analyze} ...

Trade Analytics CLI - Manage and analyze your trading portfolio

positional arguments:
  {portfolio,history,analyze}
    portfolio           Show current portfolio positions
    history             Show trade history
    analyze             Run portfolio analytics

optional arguments:
  -h, --help            show this help message and exit

# Show portfolio
$ trade-analytics portfolio
┌────────┬──────────┬───────────┬──────────────┐
│ Symbol │ Quantity │ Avg Price │ Unrealized   │
├────────┼──────────┼───────────┼──────────────┤
│ AAPL   │      100 │   $150.50 │      +$250.00│
│ GOOGL  │       50 │ $2,750.00 │     -$125.00 │
└────────┴──────────┴───────────┴──────────────┘
Total Unrealized P&L: +$125.00

# Show history with filters
$ trade-analytics history --symbol AAPL --days 30
Trade History for AAPL (last 30 days):
┌──────────────────────┬────────┬──────┬──────────┬───────────┐
│ Timestamp            │ Symbol │ Side │ Quantity │ Price     │
├──────────────────────┼────────┼──────┼──────────┼───────────┤
│ 2024-01-15 10:30:00  │ AAPL   │ BUY  │      100 │   $150.50 │
└──────────────────────┴────────┴──────┴──────────┴───────────┘
1 trade(s) found

# Run analytics
$ trade-analytics analyze
Portfolio Analytics
===================
Total P&L:     +$1,250.00
Sharpe Ratio:  1.45
Volatility:    12.3%

# Error handling example
$ trade-analytics history --start-date "bad-date"
Error: Invalid date format for --start-date

Expected format: YYYY-MM-DD (e.g., 2024-01-15)
Received: "bad-date"

Hint: Use ISO date format like --start-date 2024-01-15
```

## Appendix C: Sample Implementation Snippet

```python
"""Command-line interface for trade analytics.

This module provides a CLI for viewing portfolio positions,
trade history, and running analytics.

Usage:
    trade-analytics portfolio
    trade-analytics history --symbol AAPL --days 30
    trade-analytics analyze
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .models import Position, Trade
from .exceptions import TradingError

__all__ = [
    "main",
    "create_parser",
    "format_positions_table",
    "format_trades_table",
]


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
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
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD"
        ) from e


def main() -> None:
    """Main entry point for the CLI."""
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
```

---

**Document Version:** 1.0
**Last Updated:** 2024-11-26
**Next Review:** Before implementation begins
