"""Deterministic synthetic data with an intentionally planted predictive effect.

This is a software demonstration, not empirical evidence. The artificial calendar
uses weekdays, ignores holidays, and uses UTC closes. NEVER use it for real data.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import Dataset
from .signals import build_signals
from .adapters import aggregate_frames


def make_demo(out, weeks=260, n_assets=60, seed=1729):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).resolve().parent / "default_config.json"
    config = json.loads(base.read_text())
    config.update(dataset_mode="synthetic", near_month=1, far_month=3, signal_mode="calendar_months",
                  signal_calendar_controls=False, position_mode="binary", smoothing_weeks=1,
                  rebalance_band_l1=0., hac_lag_rule="fixed", signal_min_history=26, signal_lookback=104,
                  beta_min_history=26, beta_lookback=78, forecast_min_history=52,
                  max_universe=n_assets, min_names=24, max_weight=.10,
                  bootstrap_reps=200, seed=seed)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01 23:30", periods=weeks + 1, freq="W-WED", tz="UTC")
    schedule = []
    for i in range(weeks):
        signal = dates[i + 1]
        entry = signal.normalize() + pd.Timedelta(days=1, hours=20)
        schedule.append({"signal_id": f"W{i:04d}", "window_start": dates[i], "signal_at": signal,
                         "entry_at": entry, "exit_at": entry + pd.Timedelta(days=7),
                         "outcome_available_at": entry + pd.Timedelta(days=7, hours=3)})
    schedule = pd.DataFrame(schedule)
    config["evaluation_start"] = schedule.signal_at.iloc[110].strftime("%Y-%m-%d")
    quotes, level, curve = [], 3.0, 0.0
    for date in dates:
        level += rng.normal(0, .045)
        curve = .65 * curve + rng.normal(0, .025)
        anchor = pd.Period(date.strftime("%Y-%m"), freq="M")
        for horizon in range(1, 6):
            month = anchor + horizon
            quotes.append({"contract_id": f"DEMO_FF_{month}", "delivery_month": month,
                           "observed_at": date.normalize() + pd.Timedelta(hours=20),
                           "available_at": date.normalize() + pd.Timedelta(hours=22),
                           "settle": 100 - (level + horizon * curve)})
    quotes = pd.DataFrame(quotes)
    temporary = Dataset({"schedule": schedule, "quotes": quotes}, {}, out)
    signals = build_signals(temporary, config)
    synthetic_shock = signals.signal_bps.fillna(signals.delta_far_bps - signals.delta_near_bps).fillna(0).to_numpy()
    calendar_dates = pd.date_range(dates[0].normalize(), schedule.exit_at.iloc[-1].normalize(), freq="B", tz="UTC")
    closes = calendar_dates + pd.Timedelta(hours=20)
    calendar = pd.DataFrame({"close_at": closes})
    shock = np.zeros(len(closes))
    pos = {d: i for i, d in enumerate(calendar_dates)}
    for row, value in zip(schedule.itertuples(index=False), synthetic_shock):
        shock[pos[row.signal_at.normalize()]] += value
        delayed_day = row.signal_at.normalize() + pd.Timedelta(days=5)
        if delayed_day in pos:
            shock[pos[delayed_day]] += .35 * value
    market_r = rng.normal(.00015, .007, len(closes)) - .00015 * shock
    rf_daily = (1.02 ** (1 / 252) - 1) * np.ones(len(closes))
    market = pd.DataFrame({"close_at": closes, "available_at": closes + pd.Timedelta(hours=3),
                           "total_return": market_r, "rf_return": rf_daily})
    market_betas = rng.uniform(.6, 1.4, n_assets)
    policy_betas = np.tile(np.linspace(-.002, .002, 10), int(np.ceil(n_assets / 10)))[:n_assets]
    daily, snapshots = [], []
    delist_day = schedule.signal_at.iloc[weeks - 35].normalize() + pd.Timedelta(days=2)
    all_returns = market_r[:, None] * market_betas[None, :] + shock[:, None] * policy_betas[None, :] + rng.normal(0, .009, (len(closes), n_assets))
    for j in range(n_assets):
        asset = f"DEMO_{j:03d}"
        r = all_returns[:, j].copy()
        last = pos[delist_day] if j == 0 else len(closes) - 1
        if j == 0:
            r[last] = -1.0
        daily.append(pd.DataFrame({"asset_id": asset, "close_at": closes[:last + 1],
                                   "available_at": closes[:last + 1] + pd.Timedelta(hours=3),
                                   "total_return": r[:last + 1],
                                   "terminal": [0] * last + [int(j == 0)]}))
        for row in schedule.itertuples(index=False):
            if j == 0 and row.signal_at.normalize() >= delist_day:
                continue
            snapshots.append({"signal_id": row.signal_id, "asset_id": asset, "sector": f"SECTOR_{j // 10}",
                              "eligible": 1, "shortable": 1, "price": 50., "market_cap": 2e9 + j * 1e7,
                              "snapshot_at": row.signal_at.normalize() + pd.Timedelta(hours=20),
                              "available_at": row.signal_at.normalize() + pd.Timedelta(hours=23)})
    daily = pd.concat(daily, ignore_index=True)
    tables, audit = aggregate_frames(schedule, market, daily, calendar)
    tables.update(schedule=schedule, quotes=quotes, snapshots=pd.DataFrame(snapshots))
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    metadata = {"dataset_mode": "synthetic", "seed": seed, "weeks": weeks, "n_assets": n_assets,
                "provenance": "Artificial fixed-month futures curve and coherent daily equity paths; intentionally planted delayed effect. NOT market observations.",
                "calendar": "Artificial weekday calendar, UTC20 close, no holidays; NOT valid for real research.",
                "terminal_example": "DEMO_000 has an explicit -100% terminal return; cash thereafter within that interval.",
                "incomplete_intervals": len(audit)}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return config
