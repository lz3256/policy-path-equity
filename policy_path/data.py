"""Strict CSV contracts. Characteristics and forward returns are separate files."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd


SCHEMAS = {
    "schedule": ["signal_id", "window_start", "signal_at", "entry_at", "exit_at", "outcome_available_at"],
    "quotes": ["contract_id", "delivery_month", "observed_at", "available_at", "settle"],
    "market_windows": ["signal_id", "total_return", "rf_return", "available_at"],
    "stock_windows": ["signal_id", "asset_id", "total_return", "available_at"],
    "snapshots": ["signal_id", "asset_id", "sector", "eligible", "shortable", "price", "market_cap", "snapshot_at", "available_at"],
    "market_returns": ["signal_id", "total_return", "rf_return"],
    "asset_returns": ["signal_id", "asset_id", "total_return"],
}


def utc(series, name):
    # Reject naive timestamps instead of silently interpreting local time as UTC.
    raw = series.astype(str)
    if not raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
        raise ValueError(f"{name}: timestamps require an explicit UTC offset or Z")
    result = pd.to_datetime(raw, utc=True, errors="raise", format="mixed")
    if result.isna().any():
        raise ValueError(f"{name}: missing timestamp")
    return result


def unique(frame, keys, name):
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate key {keys}")


@dataclass
class Dataset:
    tables: dict
    metadata: dict
    path: Path

    def __getitem__(self, name):
        return self.tables[name]


def load_dataset(path, config):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("dataset_mode") not in ("synthetic", "real"):
        raise ValueError("metadata.dataset_mode must be synthetic or real")
    if metadata["dataset_mode"] != config["dataset_mode"]:
        raise ValueError("Config/data mode mismatch: synthetic data cannot be presented as real")
    if not metadata.get("provenance"):
        raise ValueError("metadata.provenance is required")
    tables = {}
    for name, cols in SCHEMAS.items():
        frame = pd.read_csv(path / f"{name}.csv", dtype={"signal_id": str, "asset_id": str, "contract_id": str, "sector": str})
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame[cols].isna().any().any():
            raise ValueError(f"{name}: required fields contain missing values")
        for c in cols:
            if c.endswith("_at") or c == "window_start":
                frame[c] = utc(frame[c], f"{name}.{c}")
        for c in ("settle", "price", "market_cap", "total_return", "rf_return"):
            if c in frame:
                frame[c] = pd.to_numeric(frame[c], errors="raise")
                if not np.isfinite(frame[c]).all():
                    raise ValueError(f"{name}.{c}: non-finite numeric value")
        if "total_return" in frame and (frame.total_return < -1).any():
            raise ValueError(f"{name}: total return below -100%")
        if "rf_return" in frame and (frame.rf_return <= -1).any():
            raise ValueError(f"{name}: invalid cash return")
        tables[name] = frame
    s = tables["schedule"].sort_values("signal_at").reset_index(drop=True)
    unique(s, ["signal_id"], "schedule")
    unique(s, ["signal_at"], "schedule")
    if len(s) < 3:
        raise ValueError("Need at least three schedule rows")
    if not ((s.window_start < s.signal_at) & (s.signal_at < s.entry_at) &
            (s.entry_at < s.exit_at) & (s.exit_at <= s.outcome_available_at)).all():
        raise ValueError("Schedule order must be window_start < signal < entry < exit <= outcome availability")
    if not (s.exit_at.iloc[:-1].reset_index(drop=True) == s.entry_at.iloc[1:].reset_index(drop=True)).all():
        raise ValueError("Holding periods must be contiguous: each exit equals the next entry")
    if not (s.window_start.iloc[1:].reset_index(drop=True) == s.signal_at.iloc[:-1].reset_index(drop=True)).all():
        raise ValueError("Signal windows must be contiguous")
    days = (s.exit_at - s.entry_at).dt.total_seconds() / 86400
    if not days.between(3, 11).all():
        raise ValueError("This implementation expects roughly weekly holding periods (3-11 days)")
    tables["schedule"] = s
    ids = set(s.signal_id)
    clocks = s.set_index("signal_id")
    for name, frame in tables.items():
        if name not in ("schedule", "quotes"):
            if not set(frame.signal_id).issubset(ids):
                raise ValueError(f"{name}: unknown signal_id")
            unique(frame, ["signal_id", "asset_id"] if "asset_id" in frame else ["signal_id"], name)
    for name in ("market_returns", "market_windows"):
        if set(tables[name].signal_id) != ids:
            raise ValueError(f"{name}: one row required per scheduled period")
    for name in ("market_windows", "stock_windows"):
        f = tables[name]
        if (f.available_at < f.signal_id.map(clocks.window_start)).any():
            raise ValueError(f"{name}: availability predates the return window")
    # Window data are usable only when available; late source data are NOT backfilled.
    snap = tables["snapshots"]
    if not snap.eligible.isin([0, 1]).all() or not snap.shortable.isin([0, 1]).all():
        raise ValueError("snapshots: eligible/shortable must be 0 or 1")
    if not ((snap.snapshot_at <= snap.available_at) &
            (snap.available_at <= snap.signal_id.map(clocks.signal_at))).all():
        raise ValueError("snapshots: future information at decision time")
    if (snap.price <= 0).any() or (snap.market_cap <= 0).any():
        raise ValueError("snapshots: price and market_cap must be positive")
    q = tables["quotes"]
    unique(q, ["contract_id", "observed_at"], "quotes")
    if not (q.observed_at <= q.available_at).all():
        raise ValueError("quotes: publication predates observation")
    if not q.delivery_month.astype(str).str.fullmatch(r"\d{4}-\d{2}").all():
        raise ValueError("delivery_month must be YYYY-MM")
    q["delivery_month"] = pd.PeriodIndex(q.delivery_month, freq="M")
    if q.groupby("contract_id").delivery_month.nunique().max() != 1:
        raise ValueError("Each contract_id must have one fixed delivery month; generics forbidden")
    if q.groupby("delivery_month").contract_id.nunique().max() != 1:
        raise ValueError("Use one canonical contract_id per delivery month")
    if config.get("signal_mode", "calendar_months") == "meeting_aligned":
        meetings = pd.read_csv(path / "meetings.csv", dtype={"meeting_id": str})
        for col in ("calendar_asof", "decision_at"):
            meetings[col] = utc(meetings[col], f"meetings.{col}")
        unique(meetings, ["calendar_asof", "meeting_id"], "meetings")
        unique(meetings, ["calendar_asof", "decision_at"], "meetings")
        if meetings.meeting_id.isna().any():
            raise ValueError("Missing meeting_id")
        tables["meetings"] = meetings
    if metadata["dataset_mode"] == "real":
        policy = metadata.get("availability_policy", {})
        required = ("observable_returns", "classifications_and_shares", "delisting_and_revisions")
        if any(not policy.get(key) for key in required):
            raise ValueError("Real data require explicit availability_policy for returns, classifications/shares and delisting/revisions")
        if policy.get("feature_return_basis") not in ("reconstructed_observable", "final_vintage_proxy", "release_vintage"):
            raise ValueError("Declare feature_return_basis: reconstructed_observable, final_vintage_proxy or release_vintage")
        if policy["feature_return_basis"] != "release_vintage":
            lag = policy.get("observable_lag_hours")
            if not isinstance(lag, (int, float)) or not np.isfinite(lag) or lag < 0:
                raise ValueError("Declare nonnegative observable_lag_hours for observable/proxy returns")
    return Dataset(tables, metadata, path)


def validate_config(c):
    if c.get("signal_mode", "calendar_months") not in ("calendar_months", "meeting_aligned"):
        raise ValueError("Unknown signal_mode")
    if c.get("position_mode", "binary") not in ("binary", "scaled_smooth"):
        raise ValueError("Unknown position_mode")
    for key, default in (("smoothing_weeks", 1), ("beta_max_staleness_days", 21), ("meeting_near_index", 1), ("meeting_far_index", 3)):
        if not isinstance(c.get(key, default), int) or c.get(key, default) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c.get("meeting_near_index", 1) >= c.get("meeting_far_index", 3):
        raise ValueError("Meeting index ordering invalid")
    if c.get("z_full_exposure", 2.) <= 0:
        raise ValueError("z_full_exposure must be positive")
    for key in ("rebalance_band_l1", "hold_beta_tolerance", "hold_sector_tolerance"):
        if c.get(key, 0.) < 0:
            raise ValueError(f"{key} must be nonnegative")
    horizons = c.get("forecast_horizons", [1, 2, 4, 8])
    if not horizons or 1 not in horizons or len(set(horizons)) != len(horizons) or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("forecast_horizons must be unique positive integers including 1")
    if c.get("hac_lag_rule", "fixed") not in ("fixed", "sample_size"):
        raise ValueError("Unknown HAC lag rule")
    positive_ints = ["near_month", "far_month", "signal_min_history", "signal_lookback", "beta_min_history",
                    "beta_lookback", "forecast_min_history", "max_universe", "min_names", "min_sector_names",
                    "bootstrap_block", "bootstrap_reps"]
    for key in positive_ints:
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["near_month"] >= c["far_month"] or c["signal_lookback"] < c["signal_min_history"] or c["beta_lookback"] < c["beta_min_history"]:
        raise ValueError("Invalid maturity ordering or history lengths")
    for key in ("signal_std_floor_bps", "quote_max_age_hours", "max_weight"):
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("min_price", "min_market_cap", "signal_threshold_z", "one_way_cost_bps", "borrow_annual", "hac_lags"):
        if c[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if not 0 < c["tail_fraction"] <= .5:
        raise ValueError("tail_fraction must be in (0, .5]")
    if not isinstance(c["neutralize_beta"], bool):
        raise ValueError("neutralize_beta must be a JSON boolean")
    pd.Timestamp(c["evaluation_start"], tz="UTC")
    return c
