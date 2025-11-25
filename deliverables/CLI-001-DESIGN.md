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

The trade-analytics-mini system requires a command-line interface (CLI) to allow users to interact with the portfolio analytics functionality. Currently, while the core data models (`Trade`, `Position`, `MarketData`) and the `PortfolioCalculator` exist, there is no user-facing interface to:

- View current portfolio positions in a human-readable format
- Query and filter trade history by date range or symbol
- Run analytics calculations and view results
- Interact with the system without writing Python code

A well-designed CLI will provide:
- **Accessibility**: Users can interact with the system via command line
- **Productivity**: Quick access to portfolio information without launching a GUI
- **Scriptability**: Commands can be integrated into shell scripts and automation
- **Discoverability**: Built-in help and clear command structure

---

## 2. Current State

### Existing Codebase Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| `src/trade_analytics/__init__.py` | Exists | Exports models, calculator, exceptions |
| `src/trade_analytics/models.py` | Exists | `Trade`, `Position`, `MarketData`, `TradeSide` |
| `src/trade_analytics/calculator.py` | Exists | `PortfolioCalculator` with analytics methods |
| `src/trade_analytics/exceptions.py` | Exists | Custom exception hierarchy |
| `src/trade_analytics/cli.py` | **Does not exist** | Needs to be created |
| `tests/test_cli.py` | **Does not exist** | Needs to be created |

### Available Data Models

```python
# From models.py
Trade(symbol, side, quantity, price, timestamp, trade_id)
Position(symbol, quantity, avg_price, unrealized_pnl)
MarketData(symbol, bid, ask, last, volume, timestamp)
TradeSide(BUY, SELL)
```

### Available Calculator Methods

```python
# From calculator.py
PortfolioCalculator.calculate_total_value(positions, market_data) -> Decimal
PortfolioCalculator.calculate_pnl(positions, market_data) -> Decimal
PortfolioCalculator.calculate_exposure_by_symbol(positions, market_data) -> Dict[str, Decimal]
```

### Dependencies

- Depends on: `CORE-001` (Data Models), `CORE-002` (Portfolio Calculator) - **Completed**
- Future integration: `STORE-001` (SQLite Storage) - will provide data persistence

---

## 3. Proposed Solution

### High-Level Approach

1. **CLI Framework**: Use Python's built-in `argparse` module for command parsing:
   - Subcommand pattern for `portfolio`, `history`, `analyze` commands
   - Consistent argument naming and help text
   - Support for common flags (`--help`, `--version`)

2. **Output Formatting**: Use `tabulate` or custom formatting for pretty table output:
   - Aligned columns for numeric data
   - Color coding for positive/negative P&L (optional)
   - Human-readable number formatting (commas, currency symbols)

3. **Data Layer Abstraction**: Create a service layer that:
   - Currently uses mock/demo data for development
   - Will integrate with SQLite storage when `STORE-001` is complete
   - Provides clean separation between CLI and data source

4. **Error Handling**: User-friendly error messages:
   - Catch all `TradingError` exceptions and display friendly messages
   - Provide suggestions for common mistakes
   - Exit with appropriate status codes

### Design Principles

- **UNIX Philosophy**: Do one thing well per command, composable output
- **User Experience**: Clear help text, sensible defaults, forgiving input
- **Testability**: CLI logic separated from I/O for easy unit testing
- **Extensibility**: Easy to add new commands and options

---

## 4. Components

### 4.1 Module: `src/trade_analytics/cli.py`

#### Functions to Implement

| Function | Purpose | Arguments |
|----------|---------|-----------|
| `main()` | Entry point, creates parser and dispatches | None (uses sys.argv) |
| `create_parser()` | Builds argparse argument parser | None |
| `cmd_portfolio(args)` | Handles 'portfolio' subcommand | Namespace with options |
| `cmd_history(args)` | Handles 'history' subcommand | Namespace with options |
| `cmd_analyze(args)` | Handles 'analyze' subcommand | Namespace with options |
| `format_positions_table(positions)` | Formats positions as table | List[Position] |
| `format_trades_table(trades)` | Formats trades as table | List[Trade] |
| `format_analytics_report(analytics)` | Formats analytics output | Dict[str, Any] |

#### Helper Functions

| Function | Purpose |
|----------|---------|
| `get_demo_positions()` | Returns mock position data for development |
| `get_demo_trades()` | Returns mock trade data for development |
| `get_demo_market_data()` | Returns mock market data for development |
| `parse_date(date_str)` | Parses date string with user-friendly errors |
| `format_decimal(value, decimals=2)` | Formats Decimal for display |
| `print_error(message)` | Prints error message to stderr |

### 4.2 Module: `tests/test_cli.py`

#### Test Classes

| Test Class | Coverage Target |
|------------|-----------------|
| `TestCreateParser` | Parser creation, subcommands, help |
| `TestCmdPortfolio` | Portfolio command with various inputs |
| `TestCmdHistory` | History command, date filtering, symbol filtering |
| `TestCmdAnalyze` | Analyze command output formatting |
| `TestFormatters` | Table formatting functions |
| `TestErrorHandling` | Error messages, exit codes |

---

## 5. Data Models

### 5.1 CLI Command Structure

```
trade-analytics <command> [options]

Commands:
  portfolio    Show current portfolio positions
  history      Show trade history
  analyze      Run portfolio analytics
```

### 5.2 Command Arguments

#### `portfolio` Command

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbol`, `-s` | str | None | Filter by symbol (optional) |
| `--format`, `-f` | str | "table" | Output format: table, json, csv |

#### `history` Command

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbol`, `-s` | str | None | Filter by symbol |
| `--start-date` | date | None | Start date (YYYY-MM-DD) |
| `--end-date` | date | None | End date (YYYY-MM-DD) |
| `--days`, `-d` | int | None | Show last N days |
| `--side` | str | None | Filter by side: buy, sell |
| `--limit`, `-n` | int | 50 | Maximum records to show |

#### `analyze` Command

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbol`, `-s` | str | None | Analyze specific symbol |
| `--metric`, `-m` | str | "all" | Specific metric: pnl, exposure, value, all |
| `--format`, `-f` | str | "table" | Output format: table, json |

### 5.3 Output Data Structures

#### Position Table Columns

| Column | Source | Format |
|--------|--------|--------|
| Symbol | `position.symbol` | String |
| Quantity | `position.quantity` | Integer with commas |
| Avg Price | `position.avg_price` | Currency ($X.XX) |
| Market Price | `market_data.last` | Currency ($X.XX) |
| Market Value | `quantity × last` | Currency ($X,XXX.XX) |
| Unrealized P&L | Calculated | Currency with +/- |
| P&L % | Calculated | Percentage |

#### Trade History Table Columns

| Column | Source | Format |
|--------|--------|--------|
| Date | `trade.timestamp` | YYYY-MM-DD HH:MM |
| Symbol | `trade.symbol` | String |
| Side | `trade.side` | BUY/SELL |
| Quantity | `trade.quantity` | Integer with commas |
| Price | `trade.price` | Currency ($X.XX) |
| Total | `quantity × price` | Currency ($X,XXX.XX) |
| Trade ID | `trade.trade_id` | String (truncated) |

---

## 6. API Contracts

### 6.1 CLI Entry Point

```python
def main() -> int:
    """Main entry point for the trade-analytics CLI.

    Returns:
        Exit code: 0 for success, 1 for user error, 2 for system error.
    """
```

### 6.2 Command Handlers

```python
def cmd_portfolio(args: argparse.Namespace) -> int:
    """Display current portfolio positions.

    Args:
        args: Parsed command-line arguments containing:
            - symbol: Optional symbol filter
            - format: Output format (table, json, csv)

    Returns:
        Exit code: 0 for success, 1 for error.
    """

def cmd_history(args: argparse.Namespace) -> int:
    """Display trade history with optional filtering.

    Args:
        args: Parsed command-line arguments containing:
            - symbol: Optional symbol filter
            - start_date: Optional start date filter
            - end_date: Optional end date filter
            - days: Optional last N days filter
            - side: Optional buy/sell filter
            - limit: Maximum records to display

    Returns:
        Exit code: 0 for success, 1 for error.
    """

def cmd_analyze(args: argparse.Namespace) -> int:
    """Run portfolio analytics and display results.

    Args:
        args: Parsed command-line arguments containing:
            - symbol: Optional symbol to analyze
            - metric: Metric to calculate (pnl, exposure, value, all)
            - format: Output format (table, json)

    Returns:
        Exit code: 0 for success, 1 for error.
    """
```

### 6.3 Formatting Functions

```python
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

def format_trades_table(trades: List[Trade]) -> str:
    """Format trades as a pretty table.

    Args:
        trades: List of trades to format.

    Returns:
        Formatted table string ready for printing.
    """

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
```

---

## 7. Error Handling

### 7.1 Error Categories

| Category | Exit Code | Example |
|----------|-----------|---------|
| Success | 0 | Command completed successfully |
| User Error | 1 | Invalid arguments, unknown symbol |
| System Error | 2 | File not found, database error |

### 7.2 User-Friendly Error Messages

| Scenario | Error Message |
|----------|---------------|
| Unknown symbol | `Error: Symbol 'XYZ' not found in portfolio. Available symbols: AAPL, GOOGL, MSFT` |
| Invalid date format | `Error: Invalid date format 'abc'. Please use YYYY-MM-DD format (e.g., 2024-01-15)` |
| Date range invalid | `Error: Start date must be before end date` |
| No data found | `No positions found matching your criteria.` |
| Missing market data | `Error: Market data unavailable for symbol 'AAPL'. Try again later.` |

### 7.3 Error Handling Strategy

```python
def main() -> int:
    try:
        args = parser.parse_args()
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
```

### 7.4 Error Output Format

Errors should be printed to stderr with clear formatting:

```
Error: <brief description>

<additional context or suggestion if applicable>

For help, run: trade-analytics <command> --help
```

---

## 8. Implementation Plan

### Phase 1: CLI Skeleton (30 min)

| Step | Task | File |
|------|------|------|
| 1.1 | Create `cli.py` with imports and docstring | `src/trade_analytics/cli.py` |
| 1.2 | Implement `create_parser()` with subcommands | `src/trade_analytics/cli.py` |
| 1.3 | Implement `main()` entry point | `src/trade_analytics/cli.py` |
| 1.4 | Add stub command handlers that print "Not implemented" | `src/trade_analytics/cli.py` |
| 1.5 | Update `__init__.py` with CLI exports | `src/trade_analytics/__init__.py` |

### Phase 2: Demo Data Layer (20 min)

| Step | Task | File |
|------|------|------|
| 2.1 | Implement `get_demo_positions()` | `src/trade_analytics/cli.py` |
| 2.2 | Implement `get_demo_trades()` | `src/trade_analytics/cli.py` |
| 2.3 | Implement `get_demo_market_data()` | `src/trade_analytics/cli.py` |
| 2.4 | Add helper functions (`parse_date`, `format_decimal`) | `src/trade_analytics/cli.py` |

### Phase 3: Portfolio Command (25 min)

| Step | Task | File |
|------|------|------|
| 3.1 | Implement `format_positions_table()` | `src/trade_analytics/cli.py` |
| 3.2 | Implement `cmd_portfolio()` with filtering | `src/trade_analytics/cli.py` |
| 3.3 | Add JSON and CSV output format support | `src/trade_analytics/cli.py` |
| 3.4 | Write tests for portfolio command | `tests/test_cli.py` |

### Phase 4: History Command (25 min)

| Step | Task | File |
|------|------|------|
| 4.1 | Implement `format_trades_table()` | `src/trade_analytics/cli.py` |
| 4.2 | Implement `cmd_history()` with date filtering | `src/trade_analytics/cli.py` |
| 4.3 | Add symbol and side filtering | `src/trade_analytics/cli.py` |
| 4.4 | Write tests for history command | `tests/test_cli.py` |

### Phase 5: Analyze Command (20 min)

| Step | Task | File |
|------|------|------|
| 5.1 | Implement `format_analytics_report()` | `src/trade_analytics/cli.py` |
| 5.2 | Implement `cmd_analyze()` using PortfolioCalculator | `src/trade_analytics/cli.py` |
| 5.3 | Add metric filtering and JSON output | `src/trade_analytics/cli.py` |
| 5.4 | Write tests for analyze command | `tests/test_cli.py` |

### Phase 6: Error Handling & Polish (20 min)

| Step | Task | File |
|------|------|------|
| 6.1 | Implement `print_error()` helper | `src/trade_analytics/cli.py` |
| 6.2 | Add comprehensive error handling to all commands | `src/trade_analytics/cli.py` |
| 6.3 | Write error handling tests | `tests/test_cli.py` |
| 6.4 | Add --version flag support | `src/trade_analytics/cli.py` |
| 6.5 | Final test run and coverage check | - |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **No persistent storage yet** | Certain | Medium | Use demo data initially; design for easy storage integration when STORE-001 completes |
| **Table formatting library dependency** | Low | Low | Use built-in string formatting; avoid external dependencies |
| **Terminal width issues** | Medium | Low | Detect terminal width; truncate long values; support `--format json` for scripts |
| **Date parsing edge cases** | Medium | Medium | Support multiple date formats; provide clear error messages with examples |
| **Unicode/emoji display issues** | Low | Low | Use ASCII characters for table borders; make symbols configurable |
| **Performance with large datasets** | Low | Medium | Implement pagination with `--limit`; lazy loading when storage is added |

### Mitigation Details

**Demo Data Design:**
```python
# Design demo data to be easily replaceable
def get_positions() -> List[Position]:
    """Get positions from storage (currently demo data)."""
    # TODO: Replace with storage.get_positions() when STORE-001 is ready
    return get_demo_positions()
```

**Terminal Width Handling:**
```python
def get_terminal_width() -> int:
    """Get terminal width, defaulting to 80 if detection fails."""
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except Exception:
        return 80
```

---

## 10. Success Criteria

### Functional Requirements

| Criteria | Verification Method |
|----------|-------------------|
| CLI entry point works | `python -m trade_analytics.cli --help` shows usage |
| `portfolio` command shows positions | Command outputs formatted position table |
| `history` command shows trades | Command outputs formatted trade table |
| `history` supports date filtering | `--start-date` and `--end-date` filter correctly |
| `history` supports `--days` shortcut | `--days 7` shows last 7 days |
| `analyze` command shows metrics | Command outputs P&L, exposure, total value |
| Pretty table output | Tables have aligned columns, headers |
| Error handling is user-friendly | Invalid input shows helpful message |

### Quality Requirements

| Criteria | Target | Verification |
|----------|--------|--------------|
| Test coverage | ≥ 90% | `pytest --cov=src/trade_analytics/cli` |
| Type hints | 100% | `mypy src/trade_analytics/cli.py` passes |
| Docstrings | All public functions | Manual review |
| Exit codes | Consistent | Tests verify exit codes |

### Test Cases (Minimum)

```
tests/test_cli.py:
  ✓ test_parser_creation
  ✓ test_parser_portfolio_command
  ✓ test_parser_history_command
  ✓ test_parser_analyze_command
  ✓ test_cmd_portfolio_basic
  ✓ test_cmd_portfolio_with_symbol_filter
  ✓ test_cmd_portfolio_json_format
  ✓ test_cmd_history_basic
  ✓ test_cmd_history_date_range
  ✓ test_cmd_history_days_filter
  ✓ test_cmd_history_symbol_filter
  ✓ test_cmd_analyze_all_metrics
  ✓ test_cmd_analyze_single_metric
  ✓ test_format_positions_table
  ✓ test_format_trades_table
  ✓ test_format_analytics_report
  ✓ test_error_invalid_symbol
  ✓ test_error_invalid_date_format
  ✓ test_error_date_range_invalid
  ✓ test_help_output
```

### Example CLI Output

#### Portfolio Command
```
$ trade-analytics portfolio

Portfolio Positions
==================

Symbol    Qty      Avg Price    Market     Value         P&L         P&L %
--------  -------  -----------  ---------  ------------  ----------  ------
AAPL      100      $150.00      $155.05    $15,505.00    +$505.00    +3.37%
GOOGL     -50      $140.00      $145.05    -$7,252.50    -$252.50    -3.61%
MSFT      200      $380.00      $385.10    $77,020.00    +$1,020.00  +1.34%
--------  -------  -----------  ---------  ------------  ----------  ------
Total                                      $85,272.50    +$1,272.50  +1.51%
```

#### History Command
```
$ trade-analytics history --days 7

Trade History (Last 7 days)
===========================

Date              Symbol  Side  Qty     Price     Total       ID
----------------  ------  ----  ------  --------  ----------  --------
2024-11-25 10:30  AAPL    BUY   100     $150.50   $15,050.00  T001
2024-11-24 14:15  GOOGL   SELL  50      $142.00   $7,100.00   T002
2024-11-23 09:45  MSFT    BUY   200     $380.00   $76,000.00  T003

3 trades shown
```

#### Analyze Command
```
$ trade-analytics analyze

Portfolio Analytics
===================

Metric                  Value
----------------------  -------------
Total Portfolio Value   $85,272.50
Total Unrealized P&L    +$1,272.50
P&L Percentage          +1.51%

Exposure by Symbol:
  AAPL:   $15,505.00 (18.2%)
  GOOGL:  $7,252.50  (8.5%)
  MSFT:   $77,020.00 (90.3%)
```

---

## Appendix A: File Structure After Implementation

```
trade-analytics-mini/
├── src/
│   └── trade_analytics/
│       ├── __init__.py          # Updated with CLI exports
│       ├── cli.py               # NEW: CLI implementation
│       ├── calculator.py        # Existing
│       ├── exceptions.py        # Existing
│       └── models.py            # Existing
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Updated with CLI fixtures
│   ├── test_calculator.py       # Existing
│   ├── test_cli.py              # NEW: CLI tests
│   └── test_models.py           # Existing
└── ...
```

## Appendix B: Example Implementation Snippets

### Argument Parser Setup

```python
def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="trade-analytics",
        description="Trade Analytics CLI - Portfolio analysis and tracking",
        epilog="For more information, visit: https://github.com/example/trade-analytics",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        required=True,
    )

    # Portfolio subcommand
    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="Show current portfolio positions",
        description="Display current portfolio positions with market values and P&L",
    )
    portfolio_parser.add_argument(
        "-s", "--symbol",
        help="Filter by symbol (e.g., AAPL)",
    )
    portfolio_parser.add_argument(
        "-f", "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    portfolio_parser.set_defaults(func=cmd_portfolio)

    # ... similar for history and analyze

    return parser
```

### Table Formatting Without External Dependencies

```python
def format_positions_table(
    positions: List[Position],
    market_data: Dict[str, MarketData],
) -> str:
    """Format positions as a pretty table using built-in string formatting."""
    if not positions:
        return "No positions found."

    # Define column widths
    headers = ["Symbol", "Qty", "Avg Price", "Market", "Value", "P&L", "P&L %"]
    widths = [8, 10, 12, 10, 14, 12, 8]

    lines = []

    # Header
    header_line = "  ".join(
        h.ljust(w) for h, w in zip(headers, widths)
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Data rows
    for pos in positions:
        md = market_data.get(pos.symbol)
        if not md:
            continue

        value = pos.quantity * md.last
        pnl = pos.quantity * (md.last - pos.avg_price)
        pnl_pct = (pnl / (pos.quantity * pos.avg_price)) * 100 if pos.avg_price else Decimal(0)

        row = [
            pos.symbol,
            f"{pos.quantity:,}",
            f"${pos.avg_price:.2f}",
            f"${md.last:.2f}",
            f"${value:,.2f}",
            f"{'+' if pnl >= 0 else ''}{pnl:,.2f}",
            f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%",
        ]
        lines.append("  ".join(
            val.rjust(w) if i > 0 else val.ljust(w)
            for i, (val, w) in enumerate(zip(row, widths))
        ))

    return "\n".join(lines)
```

---

**Document Version:** 1.0
**Last Updated:** 2024-11-26
**Next Review:** Before implementation begins
