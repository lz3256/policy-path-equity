"""Frozen v0.1 reference for semantic parity tests; not used in production."""
import numpy as np
import pandas as pd
from policy_path.data import unique
from policy_path.adapters import compound, asset_interval

def aggregate_frames(schedule, daily_market, daily_assets, calendar):
    """Calendar timestamps must be official session closes, including early closes."""
    unique(daily_market, ["close_at"], "daily_market")
    unique(daily_assets, ["close_at", "asset_id"], "daily_assets")
    unique(calendar, ["close_at"], "calendar")
    expected = pd.DatetimeIndex(calendar.close_at.sort_values())
    if len(expected) == 0 or expected[0] > schedule.window_start.min() or expected[-1] < schedule.exit_at.max():
        raise ValueError("Calendar must include a baseline close before the first signal window and cover the final exit")
    if set(daily_market.close_at) != set(expected):
        raise ValueError("Daily market rows must match every supplied calendar session exactly")
    if not set(daily_assets.close_at).issubset(set(expected)):
        raise ValueError("Stock date absent from calendar")
    for name, f in (("market", daily_market), ("assets", daily_assets)):
        if (f.available_at < f.close_at).any():
            raise ValueError(f"{name}: availability before close")
        if not np.isfinite(f.total_return).all() or (f.total_return < -1).any():
            raise ValueError(f"{name}: invalid daily total return")
    if not np.isfinite(daily_market.rf_return).all() or (daily_market.rf_return <= -1).any():
        raise ValueError("Invalid daily cash returns")
    if not daily_assets.terminal.isin([0, 1]).all():
        raise ValueError("Daily terminal flags must be 0 or 1")
    groups = {asset: g.set_index("close_at").sort_index() for asset, g in daily_assets.groupby("asset_id", observed=False)}
    for asset, g in groups.items():
        terminal = g[g.terminal == 1]
        if len(terminal) > 1 or (len(terminal) and (g.index > terminal.index[0]).any()):
            raise ValueError(f"{asset}: rows after terminal event; audit PERMNO/lifecycle")
    dm = daily_market.set_index("close_at").sort_index()
    output = {k: [] for k in ("market_windows", "stock_windows", "market_returns", "asset_returns")}
    incomplete = []
    for row in schedule.itertuples(index=False):
        if row.entry_at not in expected or row.exit_at not in expected:
            raise ValueError("entry_at and exit_at must be exact calendar session closes")
        for kind, start, end in (("window", row.window_start, row.signal_at), ("outcome", row.entry_at, row.exit_at)):
            times = expected[(expected > start) & (expected <= end)]
            if len(times) == 0:
                raise ValueError("Schedule interval has no market sessions")
            market = dm.loc[times]
            base = {"signal_id": row.signal_id, "total_return": compound(market.total_return),
                    "rf_return": compound(market.rf_return)}
            if kind == "window":
                base["available_at"] = market.available_at.max()
                output["market_windows"].append(base)
            else:
                if market.available_at.max() > row.outcome_available_at:
                    raise ValueError("Outcome availability precedes source market data")
                output["market_returns"].append(base)
            for asset, f in groups.items():
                # Outside a security's life: omit. Core backtest will fail if actually held.
                if f.index[-1] < times[0] or f.index[0] > times[-1]:
                    continue
                result = asset_interval(f, times)
                if result is None:
                    incomplete.append({"signal_id": row.signal_id, "asset_id": asset, "interval": kind,
                                       "reason": "incomplete_sessions_not_imputed"})
                    continue
                total, available = result
                base = {"signal_id": row.signal_id, "asset_id": asset, "total_return": total}
                if kind == "window":
                    base["available_at"] = available
                    output["stock_windows"].append(base)
                else:
                    if available > row.outcome_available_at:
                        raise ValueError("Outcome availability precedes source stock data")
                    output["asset_returns"].append(base)
    return {k: pd.DataFrame(v) for k, v in output.items()}, pd.DataFrame(incomplete)

