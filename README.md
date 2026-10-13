# vibe-trading-local

A local, measurement-first research and **paper-trading** stack for equities and crypto, written in Python.

It backtests strategies against a buy-and-hold benchmark, tests whether signals hold up out of sample, runs a scheduled paper-trading collector, and shows everything on a live dashboard. It is built to find out whether an edge is real, so most of the code exists to *disprove* results rather than to flatter them.

> **Simulation only.** There is no real-money order path anywhere in this project. The only broker adapter refuses to start unless it is pointed at a paper endpoint, and rejects live keys.

## Highlights

- **Event-driven backtester** (`backtest.py`): walks bars in order so intrabar stops and targets are modelled correctly. Every result reports buy-and-hold return, excess return and information ratio, because a strategy that makes money while losing to simply holding the asset has not made money.
- **No look-ahead by construction**: all indicators are causal, and signals are shifted one bar so a backtest cannot trade on the bar it is evaluating.
- **Overfitting guards** (`sweep.py`, `robustness.py`): parameter sweeps run on a search set and are only accepted if they pass a held-out validation set. Survivorship bias is corrected by adding back delisted names.
- **Factor research**: cross-sectional information-coefficient tests (`cross_sectional.py`, `crypto_ic.py`), long/short portfolio construction with turnover and borrow costs (`xs_portfolio.py`, `crypto_portfolio.py`), and "kill tests" that try to break a promising signal before anything is built on it (`crypto_kill.py`).
- **Deterministic risk controls** (`risk.py`): per-trade risk, daily and total drawdown halts, position caps. No model or network call can influence a stop.
- **Breakeven-then-trail stop manager** (`position_manager.py`) with tests.
- **Automated paper-trading collector** (`collector.py`): runs on a schedule, writes every closed trade to an idempotent SQLite ledger, and journals to Obsidian.
- **Graduation gate** (`status.py`): pre-committed PASS/FAIL criteria (minimum trades, days, drawdown, capture vs buy-and-hold) written *before* the first trade, so the bar cannot be moved after seeing results.
- **Live dashboard** (`dashboard.py`): FastAPI + server-sent events. Failing criteria are shown as prominently as passing ones, and missing data is never shown as zero.
- **TradingView integration**: generated Pine Script scanner (`gen_scanner.py`, `pine/`), a webhook receiver for paper fills (`webhook_paper.py`), and a Chrome DevTools Protocol reader that syncs chart state into the ledger (`tv_cdp.py`, `tv_sync.py`).
- **Alpaca paper trading** (`paper_broker.py`, `live_track.py`): real order lifecycle against a paper account, with keys stored in environment variables and never in files.

## Tech stack

Python · pandas · NumPy · FastAPI · Uvicorn · SQLite · httpx · yfinance · Alpaca API · TradingView Pine Script · Chrome DevTools Protocol · Obsidian

## Getting started

```bash
git clone https://github.com/BrianGigz1/vibe-trading-local.git
cd vibe-trading-local
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# Backtest a strategy on historical data
python main.py --mode backtest --symbol AAPL --start 2020-01-01

# Simulated paper replay (the default)
python main.py --symbol MSFT --live-pace

# Compare two stop-management approaches on identical data
python ab_compare.py AAPL MSFT NVDA

# Live dashboard at http://127.0.0.1:8765
python dashboard.py
```

Optional: `python setup_alpaca.py` stores Alpaca **paper** keys as user environment variables (the secret is entered hidden and never written to disk). Crypto market data needs no key.

## Tests

```bash
python test_position_manager.py
python test_stop_sync.py
python smoke_webhook.py
```

## Project layout

| Area | Files |
|---|---|
| Data | `data.py`, `crypto_data.py`, `stooq_probe.py`, `pit_universe.py` |
| Strategy & indicators | `strategy.py`, `indicators.py`, `position_manager.py` |
| Backtesting & research | `backtest.py`, `sweep.py`, `robustness.py`, `ab_compare.py`, `cross_sectional.py`, `xs_portfolio.py`, `core_tilt.py`, `overnight.py`, `turn_of_month.py`, `crypto_*.py` |
| Risk & execution (paper) | `risk.py`, `paper_trader.py`, `paper_broker.py`, `webhook_paper.py`, `live_track.py` |
| Automation & reporting | `collector.py`, `status.py`, `dashboard.py`, `journal.py`, `run_*.cmd` |
| TradingView | `gen_scanner.py`, `tv_cdp.py`, `tv_sync.py`, `import_tradingview.py`, `pine/` |

## Disclaimer

For research and education. Nothing here is financial advice.
