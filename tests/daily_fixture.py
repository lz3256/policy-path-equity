"""Artificial weekday data used only in tests and performance benchmarks."""
import numpy as np
import pandas as pd


def daily_fixture(weeks=12, assets=8):
    w = pd.date_range("2000-01-05", periods=weeks + 2, freq="W-WED", tz="UTC")
    schedule = pd.DataFrame({"signal_id": [f"S{i:04d}" for i in range(weeks)],
                             "window_start": w[:weeks] + pd.Timedelta(hours=23),
                             "signal_at": w[1:weeks+1] + pd.Timedelta(hours=23),
                             "entry_at": w[1:weeks+1] + pd.Timedelta(days=1, hours=20),
                             "exit_at": w[2:weeks+2] + pd.Timedelta(days=1, hours=20),
                             "outcome_available_at": w[2:weeks+2] + pd.Timedelta(days=2, hours=20)})
    closes = pd.date_range(w[0], schedule.exit_at.iloc[-1].normalize(), freq="B") + pd.Timedelta(hours=20)
    market = pd.DataFrame({"close_at": closes, "available_at": closes + pd.Timedelta(hours=3),
                           "total_return": np.sin(np.arange(len(closes))) * .01, "rf_return": .0001})
    names = [f"A{i:05d}" for i in range(assets)]
    daily = pd.DataFrame({"asset_id": pd.Categorical.from_codes(np.repeat(np.arange(assets), len(closes)), categories=names),
                          "close_at": pd.to_datetime(np.tile(closes.asi8, assets), utc=True),
                          "available_at": pd.to_datetime(np.tile((closes + pd.Timedelta(hours=3)).asi8, assets), utc=True),
                          "total_return": np.tile(np.sin(np.arange(len(closes))) * .008, assets),
                          "terminal": np.zeros(assets * len(closes), dtype=np.int8)})
    return schedule, market, daily, pd.DataFrame({"close_at": closes})
