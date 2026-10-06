"""Aggregate audited daily total returns into the explicit study intervals.

Provider-independent CSV adapter, NOT an authenticated Bloomberg/WRDS client.
Daily CRSP fields can be renamed to this contract after verifying your schema.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import utc, unique


def compound(values):
    r = np.asarray(values, float)
    if len(r) == 0 or not np.isfinite(r).all() or np.any(r < -1):
        raise ValueError("Cannot compound empty/missing/invalid returns")
    return float(np.prod(1 + r) - 1)  # Includes a genuine -100% return correctly.


def asset_interval(frame, expected_times):
    """Require all sessions, except cash after an explicit terminal observation.

    After terminal settlement the proceeds earn zero within this short interval.
    Never infer delisting from a missing price. Return None for incomplete history.
    """
    if len(expected_times) == 0:
        raise ValueError("Empty interval")
    observed = frame.reindex(expected_times)
    terminal_positions = np.flatnonzero(observed.terminal.fillna(0).to_numpy() == 1)
    if len(terminal_positions):
        last = terminal_positions[0]
        if observed.total_return.iloc[:last + 1].isna().any():
            return None
        if observed.total_return.iloc[last + 1:].notna().any():
            raise ValueError("Return observations occur after terminal settlement")
        values = observed.total_return.iloc[:last + 1].to_numpy()
        available = observed.available_at.iloc[:last + 1].max()
    else:
        if observed.total_return.isna().any():
            return None
        values, available = observed.total_return.to_numpy(), observed.available_at.max()
    return compound(values), available


def _reduce_returns(keys, returns, available_ns, size):
    """Zero-aware log compounding; -100% is an absorbing wealth loss."""
    count = np.bincount(keys, minlength=size)
    loss = returns == -1
    logs = np.zeros(len(returns))
    np.log1p(returns, out=logs, where=~loss)
    sums = np.bincount(keys, weights=logs, minlength=size)
    total = np.expm1(sums)
    total[np.bincount(keys, weights=loss, minlength=size) > 0] = -1.
    latest = np.full(size, np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(latest, keys, available_ns)
    if not np.isfinite(total).all():
        raise ValueError("Non-finite compounded return")
    return count, total, latest


def aggregate_frames(schedule, daily_market, daily_assets, calendar, asset_chunk_size=256):
    """Vectorized interval mapping, with bounded asset-by-interval working arrays.

    Calendar completeness and terminal exceptions match the v0.1 reference loop.
    No per-asset/per-period reindex; each daily row maps into each window family once.
    Chunking bounds intermediate matrices, not the memory of the supplied DataFrames.
    """
    if not isinstance(asset_chunk_size, int) or asset_chunk_size < 1:
        raise ValueError("asset_chunk_size must be a positive integer")
    unique(daily_market, ["close_at"], "daily_market")
    unique(daily_assets, ["close_at", "asset_id"], "daily_assets")
    unique(calendar, ["close_at"], "calendar")
    expected = pd.DatetimeIndex(calendar.close_at.sort_values()).as_unit("ns")
    if len(expected) == 0 or expected[0] > schedule.window_start.min() or expected[-1] < schedule.exit_at.max():
        raise ValueError("Calendar must include a baseline close before the first signal window and cover the final exit")
    dm = daily_market.sort_values("close_at")
    if not pd.DatetimeIndex(dm.close_at).equals(expected):
        raise ValueError("Daily market rows must match every supplied calendar session exactly")
    sessions = expected.get_indexer(daily_assets.close_at)
    if (sessions < 0).any():
        raise ValueError("Stock date absent from calendar")
    for name, f in (("market", daily_market), ("assets", daily_assets)):
        if (f.available_at < f.close_at).any():
            raise ValueError(f"{name}: availability before close")
        if not np.isfinite(f.total_return).all() or (f.total_return < -1).any():
            raise ValueError(f"{name}: invalid daily total return")
    if not np.isfinite(dm.rf_return).all() or (dm.rf_return <= -1).any():
        raise ValueError("Invalid daily cash returns")
    if not daily_assets.terminal.isin([0, 1]).all():
        raise ValueError("Daily terminal flags must be 0 or 1")
    if not schedule.entry_at.isin(expected).all() or not schedule.exit_at.isin(expected).all():
        raise ValueError("entry_at and exit_at must be exact calendar session closes")
    codes, names = pd.factorize(daily_assets.asset_id, sort=True)
    n_assets, n_periods = len(names), len(schedule)
    if (codes < 0).any():
        raise ValueError("Missing asset_id")
    first = np.full(n_assets, len(expected), dtype=np.int64)
    last = np.full(n_assets, -1, dtype=np.int64)
    np.minimum.at(first, codes, sessions)
    np.maximum.at(last, codes, sessions)
    terminal = daily_assets.terminal.to_numpy() == 1
    terminal_count = np.bincount(codes[terminal], minlength=n_assets)
    terminal_session = np.full(n_assets, -1, dtype=np.int64)
    terminal_session[codes[terminal]] = sessions[terminal]
    if (terminal_count > 1).any() or ((terminal_count == 1) & (last > terminal_session)).any():
        raise ValueError("Return observations occur after terminal settlement or multiple terminal events")
    order = np.argsort(codes, kind="stable")
    codes, sessions = codes[order], sessions[order]
    returns = daily_assets.total_return.to_numpy(float)[order]
    available = pd.DatetimeIndex(daily_assets.available_at).as_unit("ns").asi8[order]
    market_available = pd.DatetimeIndex(dm.available_at).as_unit("ns").asi8
    outcomes_available = pd.DatetimeIndex(schedule.outcome_available_at).as_unit("ns").asi8
    ids = schedule.signal_id.to_numpy()
    output, audit_parts = {}, []
    for kind, start_col, end_col in (("window", "window_start", "signal_at"), ("outcome", "entry_at", "exit_at")):
        starts = pd.DatetimeIndex(schedule[start_col]).as_unit("ns")
        ends = pd.DatetimeIndex(schedule[end_col]).as_unit("ns")
        if not ends.is_monotonic_increasing or ends.duplicated().any() or (starts >= ends).any() or (starts[1:] < ends[:-1]).any():
            raise ValueError("Interval family must be ordered, nonempty and nonoverlapping")
        left = expected.searchsorted(starts, side="right")
        right = expected.searchsorted(ends, side="right")
        required = right - left
        if (required <= 0).any():
            raise ValueError("Schedule interval has no market sessions")
        interval = ends.searchsorted(expected, side="left")
        candidate = np.minimum(interval, n_periods - 1)
        valid = (interval < n_periods) & (expected.asi8 > starts.asi8[candidate])
        interval = np.where(valid, interval, -1)
        mk = interval[valid]
        _, market_total, market_latest = _reduce_returns(mk, dm.total_return.to_numpy()[valid], market_available[valid], n_periods)
        _, cash_total, _ = _reduce_returns(mk, dm.rf_return.to_numpy()[valid], market_available[valid], n_periods)
        market = pd.DataFrame({"signal_id": ids, "total_return": market_total, "rf_return": cash_total})
        if kind == "window":
            market["available_at"] = pd.to_datetime(market_latest, utc=True)
        elif (market_latest > outcomes_available).any():
            raise ValueError("Outcome availability precedes source market data")
        output["market_windows" if kind == "window" else "market_returns"] = market
        stock_parts = []
        for lo in range(0, n_assets, asset_chunk_size):
            hi = min(lo + asset_chunk_size, n_assets)
            a, b = np.searchsorted(codes, [lo, hi])
            group_interval = interval[sessions[a:b]]
            keep = group_interval >= 0
            keys = (codes[a:b][keep] - lo) * n_periods + group_interval[keep]
            size = (hi - lo) * n_periods
            counts, total, latest = _reduce_returns(keys, returns[a:b][keep], available[a:b][keep], size)
            alive = (last[lo:hi, None] >= left) & (first[lo:hi, None] < right)
            term = terminal_session[lo:hi, None]
            terminal_inside = (term >= left) & (term < right)
            needed = np.where(terminal_inside, term - left + 1, required)
            complete = (counts.reshape(hi-lo, n_periods) == needed) & alive
            ai, wi = np.where(complete)
            flat = ai * n_periods + wi
            if kind == "outcome" and (latest[flat] > outcomes_available[wi]).any():
                raise ValueError("Outcome availability precedes source stock data")
            stock = pd.DataFrame({"signal_id": ids[wi], "asset_id": names.to_numpy()[lo + ai], "total_return": total[flat]})
            if kind == "window":
                stock["available_at"] = pd.to_datetime(latest[flat], utc=True)
            stock_parts.append(stock)
            ai, wi = np.where(alive & ~complete)
            audit_parts.append(pd.DataFrame({"signal_id": ids[wi], "asset_id": names.to_numpy()[lo + ai],
                                            "interval": kind, "reason": "incomplete_sessions_not_imputed"}))
        columns = ["signal_id", "asset_id", "total_return"] + (["available_at"] if kind == "window" else [])
        output["stock_windows" if kind == "window" else "asset_returns"] = pd.concat(stock_parts, ignore_index=True) if stock_parts else pd.DataFrame(columns=columns)
    audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame(columns=["signal_id", "asset_id", "interval", "reason"])
    return output, audit


def aggregate_csv(source, schedule_path, out):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(schedule_path, dtype={"signal_id": str})
    for c in ("window_start", "signal_at", "entry_at", "exit_at", "outcome_available_at"):
        s[c] = utc(s[c], c)
    frames = {}
    for name in ("daily_market", "daily_assets", "calendar"):
        f = pd.read_csv(source / f"{name}.csv", dtype={"asset_id": str})
        f["close_at"] = utc(f.close_at, name)
        if "available_at" in f:
            f["available_at"] = utc(f.available_at, name)
        frames[name] = f
    tables, audit = aggregate_frames(s, frames["daily_market"], frames["daily_assets"], frames["calendar"])
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    audit.to_csv(out / "incomplete_intervals.csv", index=False)
    return {"written": list(tables), "incomplete_intervals": len(audit)}


def make_schedule(calendar_path, first_wednesday, last_wednesday, output):
    """Wednesday 20:00 New York decision, next supplied session close execution.

    Calendar is supplied by the researcher; no weekday-only approximation for real data.
    The final exit requires at least one session after the following Wednesday cutoff.
    """
    calendar = pd.read_csv(calendar_path)
    closes = pd.DatetimeIndex(utc(calendar.close_at, "calendar.close_at")).sort_values()
    if closes.duplicated().any():
        raise ValueError("Duplicate calendar close")
    first, last = pd.Timestamp(first_wednesday), pd.Timestamp(last_wednesday)
    if first.weekday() != 2 or last.weekday() != 2 or last < first:
        raise ValueError("First/last dates must be Wednesdays, last >= first")
    days = pd.date_range(first, last + pd.Timedelta(days=7), freq="W-WED")
    signals = [d.tz_localize("America/New_York") + pd.Timedelta(hours=20) for d in days]
    signals = [s.tz_convert("UTC") for s in signals]
    entries = []
    for signal in signals:
        idx = closes.searchsorted(signal, side="right")
        if idx == len(closes):
            raise ValueError("Calendar does not cover final exit")
        entries.append(closes[idx])
    rows = []
    previous_local = (first - pd.Timedelta(days=7)).tz_localize("America/New_York") + pd.Timedelta(hours=20)
    for i, signal in enumerate(signals[:-1]):
        rows.append({"signal_id": days[i].strftime("%Y-%m-%d"),
                     "window_start": previous_local.tz_convert("UTC") if i == 0 else signals[i-1],
                     "signal_at": signal, "entry_at": entries[i], "exit_at": entries[i+1],
                     "outcome_available_at": entries[i+1] + pd.Timedelta(days=1)})
    frame = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return {"rows": len(frame), "output": str(output), "assumption": "20:00 America/New_York decision; outcomes available 24h after exit. Verify against source."}


def normalize_quotes(prices_path, contracts_path, output):
    """Wide Bloomberg export + manually verified fixed-contract manifest -> long CSV."""
    prices = pd.read_csv(prices_path)
    contracts = pd.read_csv(contracts_path, dtype=str)
    for col in ("observed_at", "available_at"):
        prices[col] = utc(prices[col], col)
    for col in ("source_column", "contract_id", "delivery_month"):
        if col not in contracts or contracts[col].isna().any():
            raise ValueError(f"Invalid contract manifest: {col}")
        unique(contracts, [col], "contract_manifest")
    actual = set(prices.columns) - {"observed_at", "available_at"}
    if actual != set(contracts.source_column):
        raise ValueError("Price columns must match the verified manifest exactly")
    long = prices.melt(id_vars=["observed_at", "available_at"], var_name="source_column", value_name="settle")
    missing = int(long.settle.isna().sum())
    long = long.dropna(subset=["settle"]).merge(contracts, on="source_column", validate="many_to_one")
    long["settle"] = pd.to_numeric(long.settle, errors="raise")
    if not np.isfinite(long.settle).all():
        raise ValueError("Non-finite futures quote")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    long[["contract_id", "delivery_month", "observed_at", "available_at", "settle"]].to_csv(output, index=False)
    return {"rows": len(long), "missing_quotes_omitted": missing, "output": str(output)}
