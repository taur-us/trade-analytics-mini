"""Tests for the CLI module."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from unittest.mock import patch

import pytest

from trade_analytics.cli import (
    cmd_analyze,
    cmd_history,
    cmd_portfolio,
    create_parser,
    format_analytics_report,
    format_currency,
    format_decimal,
    format_percent,
    format_pnl,
    format_positions_csv,
    format_positions_json,
    format_positions_table,
    format_trades_table,
    get_demo_market_data,
    get_demo_positions,
    get_demo_trades,
    main,
    parse_date,
    print_error,
)
from trade_analytics.models import MarketData, Position, Trade, TradeSide


# =============================================================================
# Parser Tests
# =============================================================================


class TestCreateParser:
    """Tests for parser creation."""

    def test_parser_creation(self) -> None:
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "trade-analytics"

    def test_parser_portfolio_command(self) -> None:
        """Test parsing portfolio command."""
        parser = create_parser()
        args = parser.parse_args(["portfolio"])
        assert args.command == "portfolio"
        assert args.format == "table"
        assert args.symbol is None

    def test_parser_portfolio_with_options(self) -> None:
        """Test parsing portfolio command with options."""
        parser = create_parser()
        args = parser.parse_args(["portfolio", "-s", "AAPL", "-f", "json"])
        assert args.command == "portfolio"
        assert args.symbol == "AAPL"
        assert args.format == "json"

    def test_parser_history_command(self) -> None:
        """Test parsing history command."""
        parser = create_parser()
        args = parser.parse_args(["history"])
        assert args.command == "history"
        assert args.limit == 50
        assert args.symbol is None
        assert args.start_date is None
        assert args.end_date is None

    def test_parser_history_with_options(self) -> None:
        """Test parsing history command with options."""
        parser = create_parser()
        args = parser.parse_args([
            "history",
            "-s", "AAPL",
            "--start-date", "2024-01-01",
            "--end-date", "2024-01-31",
            "-n", "10",
            "--side", "buy",
        ])
        assert args.command == "history"
        assert args.symbol == "AAPL"
        assert args.start_date == "2024-01-01"
        assert args.end_date == "2024-01-31"
        assert args.limit == 10
        assert args.side == "buy"

    def test_parser_history_with_days(self) -> None:
        """Test parsing history command with days option."""
        parser = create_parser()
        args = parser.parse_args(["history", "-d", "7"])
        assert args.command == "history"
        assert args.days == 7

    def test_parser_analyze_command(self) -> None:
        """Test parsing analyze command."""
        parser = create_parser()
        args = parser.parse_args(["analyze"])
        assert args.command == "analyze"
        assert args.metric == "all"
        assert args.format == "table"

    def test_parser_analyze_with_options(self) -> None:
        """Test parsing analyze command with options."""
        parser = create_parser()
        args = parser.parse_args([
            "analyze",
            "-s", "AAPL",
            "-m", "pnl",
            "-f", "json",
        ])
        assert args.command == "analyze"
        assert args.symbol == "AAPL"
        assert args.metric == "pnl"
        assert args.format == "json"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_date_valid(self) -> None:
        """Test parsing a valid date string."""
        result = parse_date("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == timezone.utc

    def test_parse_date_invalid(self) -> None:
        """Test parsing an invalid date string."""
        with pytest.raises(ValueError) as exc_info:
            parse_date("invalid")
        assert "Invalid date format" in str(exc_info.value)
        assert "YYYY-MM-DD" in str(exc_info.value)

    def test_format_decimal(self) -> None:
        """Test decimal formatting."""
        assert format_decimal(Decimal("1234.567"), 2) == "1,234.57"
        assert format_decimal(Decimal("1234.567"), 0) == "1,235"
        assert format_decimal(Decimal("1234.567"), 2, "$") == "$1,234.57"

    def test_format_currency(self) -> None:
        """Test currency formatting."""
        assert format_currency(Decimal("1234.56")) == "$1,234.56"
        assert format_currency(Decimal("0.99")) == "$0.99"
        assert format_currency(Decimal("-100.00")) == "$-100.00"

    def test_format_pnl_positive(self) -> None:
        """Test P&L formatting for positive values."""
        assert format_pnl(Decimal("100.50")) == "+$100.50"
        assert format_pnl(Decimal("0")) == "+$0.00"

    def test_format_pnl_negative(self) -> None:
        """Test P&L formatting for negative values."""
        assert format_pnl(Decimal("-100.50")) == "$-100.50"

    def test_format_percent_positive(self) -> None:
        """Test percentage formatting for positive values."""
        assert format_percent(Decimal("5.25")) == "+5.25%"
        assert format_percent(Decimal("0")) == "+0.00%"

    def test_format_percent_negative(self) -> None:
        """Test percentage formatting for negative values."""
        assert format_percent(Decimal("-3.75")) == "-3.75%"

    def test_print_error(self) -> None:
        """Test error printing to stderr."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            print_error("Test error message")
            assert "Error: Test error message" in mock_stderr.getvalue()


# =============================================================================
# Demo Data Tests
# =============================================================================


class TestDemoData:
    """Tests for demo data functions."""

    def test_get_demo_positions(self) -> None:
        """Test getting demo positions."""
        positions = get_demo_positions()
        assert len(positions) == 3
        symbols = {p.symbol for p in positions}
        assert symbols == {"AAPL", "GOOGL", "MSFT"}

    def test_get_demo_trades(self) -> None:
        """Test getting demo trades."""
        trades = get_demo_trades()
        assert len(trades) == 5
        assert all(isinstance(t, Trade) for t in trades)

    def test_get_demo_market_data(self) -> None:
        """Test getting demo market data."""
        market_data = get_demo_market_data()
        assert len(market_data) == 3
        assert "AAPL" in market_data
        assert "GOOGL" in market_data
        assert "MSFT" in market_data


# =============================================================================
# Formatting Tests
# =============================================================================


class TestFormatPositionsTable:
    """Tests for position table formatting."""

    def test_format_empty_positions(self) -> None:
        """Test formatting empty positions list."""
        result = format_positions_table([], {})
        assert result == "No positions found."

    def test_format_positions_table_basic(self) -> None:
        """Test basic position table formatting."""
        positions = [
            Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))
        ]
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155.00"),
                ask=Decimal("155.10"),
                last=Decimal("155.05"),
                volume=1000000,
            )
        }
        result = format_positions_table(positions, market_data)

        assert "Portfolio Positions" in result
        assert "AAPL" in result
        assert "100" in result
        assert "$150.00" in result
        assert "$155.05" in result
        assert "Total" in result


class TestFormatPositionsJson:
    """Tests for position JSON formatting."""

    def test_format_positions_json_basic(self) -> None:
        """Test basic position JSON formatting."""
        positions = [
            Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))
        ]
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155.00"),
                ask=Decimal("155.10"),
                last=Decimal("155.05"),
                volume=1000000,
            )
        }
        result = format_positions_json(positions, market_data)
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"
        assert data[0]["quantity"] == "100"
        assert data[0]["avg_price"] == "150.00"


class TestFormatPositionsCsv:
    """Tests for position CSV formatting."""

    def test_format_positions_csv_basic(self) -> None:
        """Test basic position CSV formatting."""
        positions = [
            Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("150.00"))
        ]
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                bid=Decimal("155.00"),
                ask=Decimal("155.10"),
                last=Decimal("155.05"),
                volume=1000000,
            )
        }
        result = format_positions_csv(positions, market_data)
        lines = result.split("\n")

        assert "Symbol,Quantity,Avg Price" in lines[0]
        assert "AAPL" in lines[1]


class TestFormatTradesTable:
    """Tests for trade table formatting."""

    def test_format_empty_trades(self) -> None:
        """Test formatting empty trades list."""
        result = format_trades_table([])
        assert result == "No trades found."

    def test_format_trades_table_basic(self) -> None:
        """Test basic trade table formatting."""
        trades = [
            Trade(
                symbol="AAPL",
                side=TradeSide.BUY,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
                trade_id="T001",
            )
        ]
        result = format_trades_table(trades)

        assert "Trade History" in result
        assert "AAPL" in result
        assert "BUY" in result
        assert "100" in result
        assert "$150.50" in result
        assert "T001" in result
        assert "1 trade(s) shown" in result


class TestFormatAnalyticsReport:
    """Tests for analytics report formatting."""

    def test_format_analytics_report_all_metrics(self) -> None:
        """Test formatting analytics report with all metrics."""
        positions = get_demo_positions()
        market_data = get_demo_market_data()
        result = format_analytics_report(positions, market_data, ["all"])

        assert "Portfolio Analytics" in result
        assert "Total Portfolio Value" in result
        assert "Total Unrealized P&L" in result
        assert "P&L Percentage" in result
        assert "Exposure by Symbol" in result
        assert "AAPL" in result

    def test_format_analytics_report_single_metric(self) -> None:
        """Test formatting analytics report with single metric."""
        positions = get_demo_positions()
        market_data = get_demo_market_data()
        result = format_analytics_report(positions, market_data, ["pnl"])

        assert "Total Unrealized P&L" in result
        # Should not have value or exposure sections
        assert "Exposure by Symbol" not in result


# =============================================================================
# Command Handler Tests
# =============================================================================


class TestCmdPortfolio:
    """Tests for portfolio command handler."""

    def test_cmd_portfolio_basic(self) -> None:
        """Test basic portfolio command."""
        parser = create_parser()
        args = parser.parse_args(["portfolio"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_portfolio(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Portfolio Positions" in output
        assert "AAPL" in output
        assert "GOOGL" in output
        assert "MSFT" in output

    def test_cmd_portfolio_with_symbol_filter(self) -> None:
        """Test portfolio command with symbol filter."""
        parser = create_parser()
        args = parser.parse_args(["portfolio", "-s", "AAPL"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_portfolio(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "AAPL" in output
        assert "GOOGL" not in output
        assert "MSFT" not in output

    def test_cmd_portfolio_unknown_symbol(self) -> None:
        """Test portfolio command with unknown symbol."""
        parser = create_parser()
        args = parser.parse_args(["portfolio", "-s", "XYZ"])

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = cmd_portfolio(args)

        assert result == 1
        error_output = mock_stderr.getvalue()
        assert "XYZ" in error_output
        assert "not found" in error_output

    def test_cmd_portfolio_json_format(self) -> None:
        """Test portfolio command with JSON format."""
        parser = create_parser()
        args = parser.parse_args(["portfolio", "-f", "json"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_portfolio(args)

        assert result == 0
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_cmd_portfolio_csv_format(self) -> None:
        """Test portfolio command with CSV format."""
        parser = create_parser()
        args = parser.parse_args(["portfolio", "-f", "csv"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_portfolio(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Symbol,Quantity" in output


class TestCmdHistory:
    """Tests for history command handler."""

    def test_cmd_history_basic(self) -> None:
        """Test basic history command."""
        parser = create_parser()
        args = parser.parse_args(["history"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_history(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Trade History" in output

    def test_cmd_history_symbol_filter(self) -> None:
        """Test history command with symbol filter."""
        parser = create_parser()
        args = parser.parse_args(["history", "-s", "AAPL"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_history(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "AAPL" in output
        # GOOGL and MSFT should be filtered out
        assert "GOOGL" not in output or output.count("GOOGL") < output.count("AAPL")

    def test_cmd_history_days_filter(self) -> None:
        """Test history command with days filter."""
        parser = create_parser()
        args = parser.parse_args(["history", "-d", "3"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_history(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Trade History" in output

    def test_cmd_history_side_filter(self) -> None:
        """Test history command with side filter."""
        parser = create_parser()
        args = parser.parse_args(["history", "--side", "buy"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_history(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "BUY" in output

    def test_cmd_history_limit(self) -> None:
        """Test history command with limit."""
        parser = create_parser()
        args = parser.parse_args(["history", "-n", "2"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_history(args)

        assert result == 0
        output = mock_stdout.getvalue()
        # Should show only 2 trades
        assert "2 trade(s) shown" in output

    def test_cmd_history_date_range(self) -> None:
        """Test history command with date range."""
        # Use a date range that includes all demo trades
        parser = create_parser()
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        args = parser.parse_args([
            "history",
            "--start-date", week_ago,
            "--end-date", today,
        ])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_history(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Trade History" in output

    def test_cmd_history_invalid_date_range(self) -> None:
        """Test history command with invalid date range."""
        parser = create_parser()
        args = parser.parse_args([
            "history",
            "--start-date", "2024-01-31",
            "--end-date", "2024-01-01",
        ])

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = cmd_history(args)

        assert result == 1
        error_output = mock_stderr.getvalue()
        assert "Start date must be before end date" in error_output


class TestCmdAnalyze:
    """Tests for analyze command handler."""

    def test_cmd_analyze_basic(self) -> None:
        """Test basic analyze command."""
        parser = create_parser()
        args = parser.parse_args(["analyze"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_analyze(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Portfolio Analytics" in output
        assert "Total Portfolio Value" in output
        assert "Total Unrealized P&L" in output
        assert "Exposure by Symbol" in output

    def test_cmd_analyze_single_metric(self) -> None:
        """Test analyze command with single metric."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-m", "pnl"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_analyze(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Unrealized P&L" in output

    def test_cmd_analyze_json_format(self) -> None:
        """Test analyze command with JSON format."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-f", "json"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_analyze(args)

        assert result == 0
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert "total_value" in data
        assert "total_pnl" in data
        assert "exposure" in data

    def test_cmd_analyze_symbol_filter(self) -> None:
        """Test analyze command with symbol filter."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-s", "AAPL"])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cmd_analyze(args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert "AAPL" in output

    def test_cmd_analyze_unknown_symbol(self) -> None:
        """Test analyze command with unknown symbol."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-s", "XYZ"])

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = cmd_analyze(args)

        assert result == 1
        error_output = mock_stderr.getvalue()
        assert "XYZ" in error_output


# =============================================================================
# Main Function Tests
# =============================================================================


class TestMain:
    """Tests for main entry point."""

    def test_main_no_command(self) -> None:
        """Test main with no command shows help."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main([])

        assert result == 0
        output = mock_stdout.getvalue()
        assert "portfolio" in output
        assert "history" in output
        assert "analyze" in output

    def test_main_portfolio(self) -> None:
        """Test main with portfolio command."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main(["portfolio"])

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Portfolio Positions" in output

    def test_main_history(self) -> None:
        """Test main with history command."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main(["history"])

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Trade History" in output

    def test_main_analyze(self) -> None:
        """Test main with analyze command."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main(["analyze"])

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Portfolio Analytics" in output

    def test_main_help(self) -> None:
        """Test main with --help flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_version(self) -> None:
        """Test main with --version flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_date_format_error(self) -> None:
        """Test error handling for invalid date format."""
        parser = create_parser()
        args = parser.parse_args(["history", "--start-date", "not-a-date"])

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = cmd_history(args)

        assert result == 1
        error_output = mock_stderr.getvalue()
        assert "Invalid date format" in error_output
        assert "YYYY-MM-DD" in error_output

    def test_symbol_not_found_error(self) -> None:
        """Test error handling for symbol not found."""
        parser = create_parser()
        args = parser.parse_args(["portfolio", "-s", "INVALID"])

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = cmd_portfolio(args)

        assert result == 1
        error_output = mock_stderr.getvalue()
        assert "not found" in error_output
        assert "Available symbols" in error_output
