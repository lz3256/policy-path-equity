# Data contract and assumptions

Inputs are `schedule.csv`, `quotes.csv`, `market_windows.csv`, `stock_windows.csv`, `snapshots.csv`, `market_returns.csv`, `asset_returns.csv`, and `metadata.json`. Meeting-aligned configurations additionally require `meetings.csv`.

Schedules specify decision, entry, exit, and outcome-availability times. Feature windows and subsequent holding returns use separate files. Historical sensitivity estimates exclude the current signal window, and recursive forecasts use only matured labels. A metadata flag cannot prove point-in-time provenance; final-vintage proxies must be identified as such.

Portfolio targets impose gross exposure and estimated industry/market-beta constraints. Turnover reflects drift in existing positions; an unchanged share count differs from an unchanged weight. Closing-price execution, uniform borrow fees, and cash collateral are research assumptions. Capacity, financing statements, securities-lending recall, and live fills remain unverified.

The demonstration implants a delayed relationship and is synthetic. No real-market empirical conclusion follows from its returns.
