"""Unit tests for the CLI module.

This module tests the command-line interface including:
- Argument parsing
- Table formatting
- Date parsing
- Error handling
- Command execution
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from unittest.mock import patch

import pytest

from trade_analytics.cli import (
    cmd_analyze,
    cmd_history,
    cmd_portfolio,
    create_parser,
    format_currency,
    format_error,
    format_number,
    format_positions_table,
    format_trades_table,
    get_box_chars,
    parse_date,
)
from trade_analytics.exceptions import InvalidTradeError, TradingError
from trade_analytics.models import Position, Trade, TradeSide


class TestCreateParser:
    """Tests for parser creation and configuration."""

    def test_parser_creation(self):
        """Test that parser is created with correct configuration."""
        parser = create_parser()
        assert parser.prog == "trade-analytics"
        assert "Trade Analytics CLI" in parser.description

    def test_parser_requires_subcommand(self):
        """Test that parser requires a subcommand."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_portfolio_subcommand(self):
        """Test portfolio subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["portfolio"])
        assert args.command == "portfolio"
        assert args.format == "table"

    def test_parser_portfolio_with_format(self):
        """Test portfolio subcommand with format argument."""
        parser = create_parser()
        args = parser.parse_args(["portfolio", "--format", "json"])
        assert args.format == "json"

    def test_parser_history_subcommand(self):
        """Test history subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["history"])
        assert args.command == "history"
        assert args.days == 30
        assert args.symbol is None

    def test_parser_history_with_symbol(self):
        """Test history subcommand with symbol filter."""
        parser = create_parser()
        args = parser.parse_args(["history", "--symbol", "AAPL"])
        assert args.symbol == "AAPL"

    def test_parser_history_with_dates(self):
        """Test history subcommand with date range."""
        parser = create_parser()
        args = parser.parse_args(
            ["history", "--start-date", "2024-01-01", "--end-date", "2024-01-31"]
        )
        assert args.start_date.year == 2024
        assert args.start_date.month == 1
        assert args.start_date.day == 1
        assert args.end_date.month == 1
        assert args.end_date.day == 31

    def test_parser_history_with_days(self):
        """Test history subcommand with days argument."""
        parser = create_parser()
        args = parser.parse_args(["history", "--days", "60"])
        assert args.days == 60

    def test_parser_analyze_subcommand(self):
        """Test analyze subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["analyze"])
        assert args.command == "analyze"
        assert args.metric == "all"

    def test_parser_analyze_with_metric(self):
        """Test analyze subcommand with specific metric."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "--metric", "pnl"])
        assert args.metric == "pnl"


class TestDateParsing:
    """Tests for date parsing functionality."""

    def test_parse_date_valid(self):
        """Test parsing a valid date string."""
        result = parse_date("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == timezone.utc

    def test_parse_date_invalid_format(self):
        """Test parsing an invalid date format."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            parse_date("01/15/2024")
        assert "Invalid date format" in str(exc_info.value)
        assert "YYYY-MM-DD" in str(exc_info.value)

    def test_parse_date_invalid_date(self):
        """Test parsing an invalid date value."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_date("2024-13-45")

    def test_parse_date_non_date_string(self):
        """Test parsing a non-date string."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_date("not-a-date")


class TestNumberFormatting:
    """Tests for number and currency formatting."""

    def test_format_currency_positive(self):
        """Test formatting positive currency values."""
        result = format_currency(Decimal("1250.50"))
        assert result == "+$1,250.50"

    def test_format_currency_negative(self):
        """Test formatting negative currency values."""
        result = format_currency(Decimal("-125.00"))
        assert result == "-$125.00"

    def test_format_currency_zero(self):
        """Test formatting zero currency value."""
        result = format_currency(Decimal("0.00"))
        assert result == "$0.00"

    def test_format_currency_large_value(self):
        """Test formatting large currency values."""
        result = format_currency(Decimal("1000000.00"))
        assert result == "+$1,000,000.00"

    def test_format_number_default_decimals(self):
        """Test formatting numbers with default decimal places."""
        result = format_number(Decimal("1234.567"))
        assert result == "1,234.57"

    def test_format_number_custom_decimals(self):
        """Test formatting numbers with custom decimal places."""
        result = format_number(Decimal("1234.567"), decimals=0)
        assert result == "1,235"

    def test_format_number_zero_decimals(self):
        """Test formatting numbers with zero decimal places."""
        result = format_number(Decimal("100"), decimals=0)
        assert result == "100"


class TestBoxCharacters:
    """Tests for box character selection."""

    def test_get_box_chars_returns_dict(self):
        """Test that get_box_chars returns a dictionary."""
        chars = get_box_chars()
        assert isinstance(chars, dict)

    def test_get_box_chars_has_required_keys(self):
        """Test that box chars dict has all required keys."""
        chars = get_box_chars()
        required_keys = ["tl", "tr", "bl", "br", "h", "v", "t", "b", "l", "r", "c"]
        for key in required_keys:
            assert key in chars


class TestTableFormatters:
    """Tests for table formatting functions."""

    def test_format_positions_table_empty(self):
        """Test formatting an empty positions list."""
        result = format_positions_table([])
        assert result == "No positions found."

    def test_format_positions_table_single(self):
        """Test formatting a single position."""
        positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                avg_price=Decimal("150.50"),
                unrealized_pnl=Decimal("250.00"),
            )
        ]
        result = format_positions_table(positions)
        assert "AAPL" in result
        assert "100" in result
        assert "$150.50" in result
        assert "$250.00" in result

    def test_format_positions_table_multiple(self):
        """Test formatting multiple positions."""
        positions = [
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
        ]
        result = format_positions_table(positions)
        assert "AAPL" in result
        assert "GOOGL" in result
        assert "Total Unrealized P&L" in result

    def test_format_positions_table_no_totals(self):
        """Test formatting positions without totals."""
        positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                avg_price=Decimal("150.50"),
            )
        ]
        result = format_positions_table(positions, include_totals=False)
        assert "AAPL" in result
        assert "Total" not in result

    def test_format_trades_table_empty(self):
        """Test formatting an empty trades list."""
        result = format_trades_table([])
        assert result == "No trades found."

    def test_format_trades_table_with_data(self):
        """Test formatting trades with data."""
        trades = [
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
                trade_id="T001",
            ),
            Trade(
                symbol="GOOGL",
                side=TradeSide.SELL,
                quantity=Decimal("25"),
                price=Decimal("2780.00"),
                timestamp=datetime(2024, 1, 16, 14, 22, 15, tzinfo=timezone.utc),
                trade_id="T002",
            ),
        ]
        result = format_trades_table(trades)
        assert "AAPL" in result
        assert "GOOGL" in result
        assert "BUY" in result
        assert "SELL" in result
        assert "2 trade(s) found" in result

    def test_format_trades_table_max_rows(self):
        """Test formatting trades with max rows limit."""
        trades = [
            Trade(
                symbol=f"SYM{i}",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("100.00"),
                timestamp=datetime(2024, 1, i, 10, 0, 0, tzinfo=timezone.utc),
                trade_id=f"T{i:03d}",
            )
            for i in range(1, 11)
        ]
        result = format_trades_table(trades, max_rows=5)
        assert "Showing 5 of 10" in result


class TestErrorFormatting:
    """Tests for error message formatting."""

    def test_format_error_trading_error(self):
        """Test formatting TradingError."""
        error = TradingError("Test trading error")
        result = format_error(error)
        assert "Error: Test trading error" in result

    def test_format_error_invalid_trade_error(self):
        """Test formatting InvalidTradeError."""
        error = InvalidTradeError("Invalid trade", reason="test_reason")
        result = format_error(error)
        assert "Error: Invalid trade" in result

    def test_format_error_argument_type_error(self):
        """Test formatting ArgumentTypeError."""
        error = argparse.ArgumentTypeError("Invalid argument")
        result = format_error(error)
        assert "Error: Invalid argument" in result

    def test_format_error_generic_exception(self):
        """Test formatting generic exception."""
        error = ValueError("Some value error")
        result = format_error(error)
        assert "Unexpected error" in result
        assert "Some value error" in result


class TestCommandHandlers:
    """Tests for command handler functions."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_portfolio_table_format(self, mock_stdout):
        """Test portfolio command with table format."""
        args = argparse.Namespace(format="table")
        exit_code = cmd_portfolio(args)
        assert exit_code == 0
        output = mock_stdout.getvalue()
        assert "Symbol" in output
        assert "Quantity" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_portfolio_json_format(self, mock_stdout):
        """Test portfolio command with JSON format."""
        args = argparse.Namespace(format="json")
        exit_code = cmd_portfolio(args)
        assert exit_code == 0
        output = mock_stdout.getvalue()
        assert "positions" in output
        assert "total_unrealized_pnl" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_history_default(self, mock_stdout):
        """Test history command with defaults."""
        args = argparse.Namespace(
            symbol=None, start_date=None, end_date=None, days=30
        )
        exit_code = cmd_history(args)
        assert exit_code == 0
        output = mock_stdout.getvalue()
        assert "Trade History" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_history_with_symbol_filter(self, mock_stdout):
        """Test history command with symbol filter."""
        args = argparse.Namespace(
            symbol="AAPL", start_date=None, end_date=None, days=30
        )
        exit_code = cmd_history(args)
        assert exit_code == 0
        output = mock_stdout.getvalue()
        assert "AAPL" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_history_with_date_filter(self, mock_stdout):
        """Test history command with date range."""
        args = argparse.Namespace(
            symbol=None,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
            days=30,
        )
        exit_code = cmd_history(args)
        assert exit_code == 0
        output = mock_stdout.getvalue()
        assert "Date range" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_analyze_all_metrics(self, mock_stdout):
        """Test analyze command with all metrics."""
        args = argparse.Namespace(metric="all")
        exit_code = cmd_analyze(args)
        assert exit_code == 0
        output = mock_stdout.getvalue()
        assert "Portfolio Analytics" in output
        assert "Total P&L" in output
        assert "Sharpe Ratio" in output
        assert "Volatility" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_analyze_specific_metric(self, mock_stdout):
        """Test analyze command with specific metric."""
        args = argparse.Namespace(metric="pnl")
        exit_code = cmd_analyze(args)
        assert exit_code == 0
        output = mock_stdout.getvalue()
        assert "Total P&L" in output
        assert "Sharpe Ratio" not in output
        assert "Volatility" not in output


class TestMainFunction:
    """Tests for the main entry point."""

    @patch("sys.argv", ["trade-analytics", "portfolio"])
    @patch("sys.exit")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_portfolio_success(self, mock_stdout, mock_exit):
        """Test main function with portfolio command."""
        from trade_analytics.cli import main

        main()
        mock_exit.assert_called_once_with(0)

    @patch("sys.argv", ["trade-analytics", "history", "--symbol", "AAPL"])
    @patch("sys.exit")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_history_success(self, mock_stdout, mock_exit):
        """Test main function with history command."""
        from trade_analytics.cli import main

        main()
        mock_exit.assert_called_once_with(0)

    @patch("sys.argv", ["trade-analytics", "analyze"])
    @patch("sys.exit")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_analyze_success(self, mock_stdout, mock_exit):
        """Test main function with analyze command."""
        from trade_analytics.cli import main

        main()
        mock_exit.assert_called_once_with(0)
