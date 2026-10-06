import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from policy_path.adapters import compound, asset_interval, make_schedule, normalize_quotes
from policy_path.data import Dataset, load_dataset, utc
from policy_path.demo import make_demo
from policy_path.signals import build_signals, get_quote
from policy_path.research import aggregate_forecasts, build_targets, exposure_history, estimate_exposures, cross_sectional_diagnostics
from policy_path.portfolio import ledger_step, backtest
from policy_path.stats import ols, hac_regression
from policy_path.report import run_research


class AccountingTests(unittest.TestCase):
    def test_compounding_includes_total_loss(self):
        self.assertEqual(compound([.1, -1., .2]), -1.)
        self.assertAlmostEqual(compound([.1, -.1]), -.01)
        with self.assertRaises(ValueError):
            compound([.1, np.nan])

    def test_drift_changes_turnover(self):
        w, first = ledger_step({}, {"A": .5, "B": -.5}, {"A": .1, "B": -.1}, 0, 7, 0, 0)
        self.assertAlmostEqual(first["total_return"], .1)
        _, second = ledger_step(w, {"A": .5, "B": -.5}, {"A": 0, "B": 0}, 0, 7, 0, 0)
        self.assertAlmostEqual(second["turnover"], 1 / 11)

    def test_cash_cost_and_short_borrow(self):
        _, result = ledger_step({}, {"A": .5, "B": -.5}, {"A": .1, "B": -.1}, .001, 7, 10, .02)
        borrow = .5 * .02 * 7 / 365.25
        expected = (1 - .001) * (1 + .001) + .5 * 1.1 - .5 * .9 - borrow - 1
        self.assertAlmostEqual(result["total_return"], expected)
        self.assertAlmostEqual(result["borrow_cost"], borrow)

    def test_missing_held_return_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing held-asset"):
            ledger_step({}, {"DELISTED": .5}, {}, 0, 7, 10, .02)
        _, r = ledger_step({}, {"DELISTED": .5}, {"DELISTED": -1}, 0, 7, 0, 0)
        self.assertAlmostEqual(r["total_return"], -.5)

    def test_explicit_terminal_cash_only(self):
        dates = pd.date_range("2020-01-01", periods=3, tz="UTC")
        f = pd.DataFrame({"total_return": [.1, -.5], "terminal": [0, 1], "available_at": dates[:2]}, index=dates[:2])
        r, _ = asset_interval(f, dates)
        self.assertAlmostEqual(r, -.45)
        f["terminal"] = 0
        self.assertIsNone(asset_interval(f, dates))

    def test_naive_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            utc(pd.Series(["2020-01-01 16:00:00"]), "test")

    def test_ols_and_hac_reference_case(self):
        x = np.arange(50, dtype=float)
        y = 2 + 3 * x
        np.testing.assert_allclose(ols(x, y), [2, 3], atol=1e-10)
        fit = hac_regression(x, y)
        np.testing.assert_allclose(fit["coef"], [2, 3], atol=1e-10)
        self.assertLess(max(fit["se"]), 1e-10)

    def test_schedule_obeys_holiday_and_dst_calendar(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            closes = pd.date_range("2024-03-01", "2024-03-22", freq="B", tz="America/New_York") + pd.Timedelta(hours=16)
            # Simulate a supplied holiday, testing that code follows the calendar.
            closes = closes[closes.strftime("%Y-%m-%d") != "2024-03-14"]
            pd.DataFrame({"close_at": closes}).to_csv(p / "calendar.csv", index=False)
            make_schedule(p / "calendar.csv", "2024-03-06", "2024-03-13", p / "schedule.csv")
            f = pd.read_csv(p / "schedule.csv")
            self.assertEqual(pd.Timestamp(f.entry_at.iloc[1]), pd.Timestamp("2024-03-15T20:00:00Z"))
            self.assertEqual(pd.Timestamp(f.entry_at.iloc[0]), pd.Timestamp("2024-03-07T21:00:00Z"))
            self.assertEqual(f.exit_at.iloc[0], f.entry_at.iloc[1])

    def test_wide_quote_manifest_adapter(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            pd.DataFrame({"observed_at": ["2024-01-01T20:00:00Z"], "available_at": ["2024-01-01T21:00:00Z"],
                          "verified_contract": [95.]}).to_csv(p / "wide.csv", index=False)
            pd.DataFrame({"source_column": ["verified_contract"], "contract_id": ["FEB24"],
                          "delivery_month": ["2024-02"]}).to_csv(p / "manifest.csv", index=False)
            normalize_quotes(p / "wide.csv", p / "manifest.csv", p / "quotes.csv")
            f = pd.read_csv(p / "quotes.csv")
            self.assertEqual(f.contract_id.iloc[0], "FEB24")
            self.assertEqual(f.settle.iloc[0], 95.)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "data"
        cls.config = make_demo(cls.path, weeks=180, n_assets=30, seed=41)
        cls.config["bootstrap_reps"] = 30
        cls.data = load_dataset(cls.path, cls.config)
        cls.signals = build_signals(cls.data, cls.config)
        cls.forecasts = aggregate_forecasts(cls.data, cls.signals, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fixed_contract_changes_ignore_roll_level_gap(self):
        data = copy.deepcopy(self.data)
        q = data["quotes"]
        origin = q.delivery_month.min()
        q["settle"] = q.delivery_month.map(lambda p: 97 - .1 * (p.ordinal - origin.ordinal))
        s = build_signals(data, self.config)
        self.assertTrue(np.allclose(s.delta_near_bps, 0))
        self.assertTrue(np.allclose(s.delta_far_bps, 0))

    def test_quotes_not_yet_available_are_excluded(self):
        row = self.data["schedule"].iloc[50]
        month = pd.Period(row.signal_at.strftime("%Y-%m"), freq="M") + 1
        q = self.data["quotes"].copy()
        mask = (q.delivery_month == month) & (q.observed_at.dt.date == row.signal_at.date())
        q.loc[mask, "available_at"] = row.signal_at + pd.Timedelta(days=1)
        self.assertIsNone(get_quote(q, month, row.signal_at, 96))

    def test_future_quotes_do_not_change_prefix(self):
        cutoff = self.data["schedule"].signal_at.iloc[120]
        data = copy.deepcopy(self.data)
        data["quotes"].loc[data["quotes"].observed_at > cutoff, "settle"] += 5
        altered = build_signals(data, self.config)
        assert_frame_equal(self.signals.iloc[:121], altered.iloc[:121])

    def test_current_quote_is_not_in_residual_fit(self):
        data = copy.deepcopy(self.data)
        row = data["schedule"].iloc[120]
        month = pd.Period(row.signal_at.strftime("%Y-%m"), freq="M") + 3
        q = data["quotes"]
        mask = (q.delivery_month == month) & (q.observed_at.dt.date == row.signal_at.date())
        q.loc[mask, "settle"] += .1
        changed = build_signals(data, self.config).iloc[120]
        old = self.signals.iloc[120]
        self.assertAlmostEqual(changed.residual_beta, old.residual_beta)
        self.assertAlmostEqual(changed.residual_intercept_bps, old.residual_intercept_bps)
        self.assertAlmostEqual(changed.signal_bps - old.signal_bps, -10.)

    def test_only_matured_outcomes_train_forecasts(self):
        f = self.forecasts[self.forecasts.status == "ok"]
        self.assertGreater(len(f), 20)
        self.assertTrue((f.train_last_outcome_available_at <= f.signal_at).all())
        cutoff = self.data["schedule"].signal_at.iloc[120]
        data = copy.deepcopy(self.data)
        unknown = set(data["schedule"].loc[data["schedule"].outcome_available_at > cutoff, "signal_id"])
        data["market_returns"].loc[data["market_returns"].signal_id.isin(unknown), "total_return"] = .5
        altered = aggregate_forecasts(data, self.signals, self.config)
        cols = ["historical_mean", "slope_only", "controls_only", "slope_controls", "n_train"]
        assert_frame_equal(self.forecasts.loc[:120, cols], altered.loc[:120, cols])

    def test_current_window_not_in_exposure_fit(self):
        history = exposure_history(self.data, self.signals)
        row = next(r for r in self.data["schedule"].itertuples(index=False) if r.signal_id == "W0120")
        assets = set(self.data["snapshots"].asset_id)
        old = estimate_exposures(history, row, assets, self.config)
        history.loc[history.signal_id == row.signal_id, "stock_excess"] += .5
        new = estimate_exposures(history, row, assets, self.config)
        assert_frame_equal(old, new)

    def test_forward_returns_do_not_select_portfolio_and_neutrality(self):
        targets, decisions, exposure = build_targets(self.data, self.signals, self.config)
        self.assertGreater((decisions.decision == "active").sum(), 10)
        data = copy.deepcopy(self.data)
        data.tables["asset_returns"] = data["asset_returns"].iloc[:0]
        altered, _, _ = build_targets(data, self.signals, self.config)
        self.assertEqual(targets, altered)
        active = exposure[exposure.target_weight.abs() > 1e-12]
        self.assertLess(active.groupby(["signal_id", "sector"]).target_weight.sum().abs().max(), 1e-8)
        beta_exposure = (active.market_beta * active.target_weight).groupby(active.signal_id).sum()
        self.assertLess(beta_exposure.abs().max(), 1e-8)
        with self.assertRaisesRegex(ValueError, "Missing held-asset"):
            backtest(data, targets, self.config)

    def test_end_to_end_artifacts_and_terminal_cost(self):
        out = Path(self.temp.name) / "results"
        result = run_research(self.path, self.config, out)
        self.assertEqual(result["dataset_mode"], "synthetic")
        self.assertEqual(result["strategy"]["n_periods"], 70)
        self.assertIn("SYNTHETIC DEMO", (out / "report.html").read_text())
        ledger = pd.read_csv(out / "ledger.csv")
        self.assertTrue(np.isfinite(ledger.nav).all())
        self.assertTrue((ledger.entry_at < ledger.exit_at).all())
        self.assertEqual(len(json.loads((out / "manifest.json").read_text())["input_sha256"]), 7)

    def test_mode_mismatch_rejected(self):
        config = dict(self.config, dataset_mode="real")
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            load_dataset(self.path, config)

    def test_future_snapshot_and_duplicate_quotes_fail(self):
        for filename, change, message in (
            ("snapshots.csv", lambda f: f.assign(available_at="2099-01-01T00:00:00Z"), "future information"),
            ("quotes.csv", lambda f: pd.concat([f, f.iloc[:1]], ignore_index=True), "duplicate key"),
        ):
            path = self.path / filename
            original = path.read_bytes()
            try:
                change(pd.read_csv(path)).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    load_dataset(self.path, self.config)
            finally:
                path.write_bytes(original)

    def test_overlapping_execution_schedule_fails(self):
        path = self.path / "schedule.csv"
        original = path.read_bytes()
        try:
            f = pd.read_csv(path)
            f.loc[1, "entry_at"] = f.loc[0, "entry_at"]
            f.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_dataset(self.path, self.config)
        finally:
            path.write_bytes(original)

    def test_final_liquidation_cost_is_charged(self):
        row = self.data["schedule"].iloc[-1]
        targets = {row.signal_id: {"DEMO_001": .5, "DEMO_002": -.5}}
        result = backtest(self.data, targets, self.config)
        ar = self.data["asset_returns"]
        returns = ar[ar.signal_id == row.signal_id].set_index("asset_id").total_return.to_dict()
        rf = self.data["market_returns"].set_index("signal_id").loc[row.signal_id, "rf_return"]
        drift, before = ledger_step({}, targets[row.signal_id], returns, rf, 7, self.config["one_way_cost_bps"], self.config["borrow_annual"])
        liquidation = sum(abs(v) for v in drift.values()) * self.config["one_way_cost_bps"] / 10000
        expected = (1 + before["total_return"]) * (1 - liquidation) - 1
        self.assertAlmostEqual(result.total_return.iloc[0], expected)
        self.assertGreater(result.terminal_liquidation_cost.iloc[0], 0)

    def test_cross_section_recovers_controlled_interaction(self):
        _, _, exposures = build_targets(self.data, self.signals, self.config)
        data = copy.deepcopy(self.data)
        z = self.signals.set_index("signal_id").signal_z
        pieces = []
        for sid, group in exposures.groupby("signal_id"):
            g = group.copy()
            standardized = (g.policy_beta_per_bp - g.policy_beta_per_bp.mean()) / g.policy_beta_per_bp.std()
            g["total_return"] = .02 * z.loc[sid] * standardized + .01 * g.market_beta + .003 * np.log(g.market_cap)
            pieces.append(g[["signal_id", "asset_id", "total_return"]])
        data.tables["asset_returns"] = pd.concat(pieces, ignore_index=True)
        result = cross_sectional_diagnostics(data, self.signals, exposures)
        self.assertTrue((result.status == "ok").all())
        np.testing.assert_allclose(result.policy_premium, .02 * result.signal_z, atol=1e-9)
        lost_sid = data["asset_returns"].signal_id.iloc[0]
        data.tables["asset_returns"] = data["asset_returns"].iloc[1:]
        result = cross_sectional_diagnostics(data, self.signals, exposures)
        self.assertEqual(result.set_index("signal_id").loc[lost_sid, "status"], "missing_outcome_skip_entire_period")


if __name__ == "__main__":
    unittest.main()
