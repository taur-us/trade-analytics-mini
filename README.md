# Trade Analytics Mini

A mini trading analytics system for testing the autonomous development workflow.

## Features

- Trade and position data models
- Portfolio analytics calculator
- SQLite storage layer
- Command-line interface

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Show portfolio
trade-analytics portfolio

# Show trade history
trade-analytics history --symbol AAPL --days 30

# Run analytics
trade-analytics analyze
```
