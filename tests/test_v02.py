import copy
import tempfile
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from policy_path.adapters import aggregate_frames
from policy_path.data import Dataset, load_dataset
from policy_path.demo import make_demo
from policy_path.research import aggregate_forecasts, build_targets, form_weights, smooth_targets
from policy_path.portfolio import backtest
from policy_path.signals import delivery_months, build_signals
from policy_path.stats import hac_lags
from daily_fixture import daily_fixture
from reference_aggregation import aggregate_frames as reference


class AggregationParity(unittest.TestCase):
    def compare(self, source):
        slow, slow_audit = reference(*source)
        for chunk in (1, 3, 256):
            fast, audit = aggregate_frames(*source, asset_chunk_size=chunk)
            for name in fast:
                keys = ["signal_id"] + (["asset_id"] if "asset_id" in fast[name] else [])
                a, b = fast[name].sort_values(keys).reset_index(drop=True), slow[name].sort_values(keys).reset_index(drop=True)
                if "asset_id" in a:
                    a["asset_id"], b["asset_id"] = a.asset_id.astype(str), b.asset_id.astype(str)
                assert_frame_equal(a, b, check_dtype=False, atol=1e-12, rtol=1e-12)
            if len(slow_audit):
                for frame in (audit, slow_audit):
                    frame["asset_id"] = frame.asset_id.astype(str)
                keys = ["signal_id", "asset_id", "interval"]
                assert_frame_equal(audit.sort_values(keys).reset_index(drop=True), slow_audit.sort_values(keys).reset_index(drop=True), check_dtype=False)
            else:
                self.assertEqual(len(audit), 0)

    def test_full_sample_chunk_parity(self):
        self.compare(daily_fixture())

    def test_missing_sessions_entire_week_and_ipo(self):
        s, m, a, c = daily_fixture()
        remove = ((a.asset_id == "A00000") & (a.close_at == c.close_at.iloc[14]))
        remove |= ((a.asset_id == "A00001") & a.close_at.between(s.window_start.iloc[4], s.signal_at.iloc[4]))
        remove |= ((a.asset_id == "A00002") & (a.close_at < c.close_at.iloc[23]))
        self.compare((s, m, a[~remove], c))

    def test_terminal_minus_one_and_cash_tail(self):
        s, m, a, c = daily_fixture()
        terminal = c.close_at.iloc[17]
        a = a[~((a.asset_id == "A00000") & (a.close_at > terminal))].copy()
        at = (a.asset_id == "A00000") & (a.close_at == terminal)
        a.loc[at, "terminal"], a.loc[at, "total_return"] = 1, -1.
        self.compare((s, m, a, c))

    def test_missing_before_terminal_is_not_forgiven(self):
        s, m, a, c = daily_fixture()
        terminal = c.close_at.iloc[17]
        a = a[~((a.asset_id == "A00000") & ((a.close_at > terminal) | (a.close_at == c.close_at.iloc[16])))].copy()
        a.loc[(a.asset_id == "A00000") & (a.close_at == terminal), "terminal"] = 1
        self.compare((s, m, a, c))

    def test_late_outcome_publication_fails_both(self):
        s, m, a, c = daily_fixture()
        a.loc[a.close_at == s.exit_at.iloc[2], "available_at"] = s.outcome_available_at.iloc[2] + pd.Timedelta(days=1)
        for fn in (reference, aggregate_frames):
            with self.assertRaisesRegex(ValueError, "Outcome availability"):
                fn(s, m, a, c)


class RevisionResearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "data"
        cls.config = make_demo(cls.path, weeks=180, n_assets=30, seed=41)
        cls.data = load_dataset(cls.path, cls.config)
        cls.signals = build_signals(cls.data, cls.config)
        cls.smooth = dict(cls.config, position_mode="scaled_smooth", z_full_exposure=2., smoothing_weeks=3,
                          rebalance_band_l1=.10, hold_beta_tolerance=.03, hold_sector_tolerance=.02)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_multiweek_compounding_and_tail(self):
        f = aggregate_forecasts(self.data, self.signals, self.config, horizon=4)
        r = self.data["market_returns"].set_index("signal_id")
        s = self.data["schedule"]
        sid = s.signal_id.iloc[120]
        block = r.loc[s.signal_id.iloc[120:124]]
        actual = np.prod(1 + block.total_return) - np.prod(1 + block.rf_return)
        self.assertAlmostEqual(f.set_index("signal_id").loc[sid, "actual_excess"], actual)
        self.assertTrue(f.actual_excess.iloc[-3:].isna().all())
        active = f[f.status == "ok"]
        self.assertTrue((active.train_last_outcome_available_at <= active.signal_at).all())

    def test_multihorizon_future_labels_do_not_train_early(self):
        cutoff = self.data["schedule"].signal_at.iloc[125]
        altered = copy.deepcopy(self.data)
        s = altered["schedule"]
        future = set(s.loc[s.outcome_available_at > cutoff, "signal_id"])
        altered["market_returns"].loc[altered["market_returns"].signal_id.isin(future), "total_return"] = .4
        cols = ["slope_only", "controls_only", "slope_controls", "historical_mean", "n_train"]
        for horizon in (2, 4, 8):
            a = aggregate_forecasts(self.data, self.signals, self.config, horizon)
            b = aggregate_forecasts(altered, self.signals, self.config, horizon)
            assert_frame_equal(a.loc[:125, cols], b.loc[:125, cols])

    def test_hac_lag_floor_for_overlap(self):
        self.assertGreaterEqual(hac_lags(dict(self.config, hac_lag_rule="sample_size"), 200, 8), 7)
        self.assertEqual(hac_lags(dict(self.config, hac_lag_rule="fixed", hac_lags=4), 200, 1), 4)

    def test_meeting_calendar_is_asof_snapshot(self):
        row = self.data["schedule"].iloc[120]
        old = row.signal_at - pd.Timedelta(days=30)
        meetings = pd.DataFrame({"meeting_id": ["one", "two", "three"], "calendar_asof": [old]*3,
                                 "decision_at": [row.signal_at + pd.Timedelta(days=x) for x in (10, 50, 100)]})
        data = copy.deepcopy(self.data)
        data.tables["meetings"] = meetings
        config = dict(self.config, signal_mode="meeting_aligned", meeting_near_index=1, meeting_far_index=3)
        first, vintage = delivery_months(data, row, config)
        future = meetings.assign(calendar_asof=row.signal_at + pd.Timedelta(days=1), decision_at=meetings.decision_at + pd.Timedelta(days=40))
        data.tables["meetings"] = pd.concat([meetings, future])
        second, _ = delivery_months(data, row, config)
        self.assertEqual(first, second)
        self.assertEqual(vintage, old)
        expected = pd.Period(meetings.decision_at.iloc[0].tz_convert("America/New_York").strftime("%Y-%m"), freq="M") + 1
        self.assertEqual(first["near"], expected)

    def test_calendar_control_fit_is_past_only(self):
        config = dict(self.config, signal_calendar_controls=True, near_month=2, far_month=4)
        a = build_signals(self.data, config)
        changed = copy.deepcopy(self.data)
        cutoff = changed["schedule"].signal_at.iloc[120]
        changed["quotes"].loc[changed["quotes"].observed_at > cutoff, "settle"] += 1.
        b = build_signals(changed, config)
        assert_frame_equal(a.iloc[:121], b.iloc[:121])
        self.assertTrue(a.calendar_controls.all())

    def test_scaled_weights_are_linear_before_clip(self):
        _, _, exposure = build_targets(self.data, self.signals, self.config)
        group = next(g for _, g in exposure.groupby("signal_id") if len(g) >= 24)
        ecols = ["asset_id", "market_beta", "policy_beta_per_bp", "n_beta", "beta_last_signal_id"]
        snapshot = group.drop(columns=ecols[1:])
        a, _, _ = form_weights(snapshot, group[ecols], .4, self.smooth)
        b, _, _ = form_weights(snapshot, group[ecols], .6, self.smooth)
        self.assertTrue(a)
        self.assertEqual(set(a), set(b))
        np.testing.assert_allclose([b[k] for k in a], [1.5*a[k] for k in a], atol=1e-10)

    def test_smoothing_and_band_ignore_all_forward_returns(self):
        a, da, _ = build_targets(self.data, self.signals, self.smooth)
        changed = copy.deepcopy(self.data)
        changed.tables["asset_returns"] = changed["asset_returns"].iloc[:0]
        b, db, _ = build_targets(changed, self.signals, self.smooth)
        self.assertEqual(a, b)
        assert_frame_equal(da, db)

    def test_hold_means_zero_turnover_not_target_rebalance(self):
        targets, decisions, exposure = build_targets(self.data, self.signals, self.smooth)
        # A deterministic two-period fixture isolates the execution contract.
        sids = list(targets)[:2]
        book = {s: {"DEMO_001": .2, "DEMO_002": -.2} for s in sids}
        instructions = pd.DataFrame({"signal_id": sids, "rebalance": [True, False]})
        ledger = backtest(self.data, book, self.smooth, decisions=instructions)
        self.assertEqual(ledger.turnover.iloc[1], 0.)
        self.assertNotAlmostEqual(ledger.gross_target.iloc[1], .4, places=10)
        np.testing.assert_allclose(ledger.gross_before_cost_return-ledger.total_return,
                                   ledger.trade_cost*(1+ledger.rf_return)+ledger.borrow_cost+ledger.terminal_liquidation_cost, atol=1e-12)

    def test_ineligible_names_cannot_survive_smoothing(self):
        _, _, exposure = build_targets(self.data, self.signals, self.config)
        group = next(g for _, g in exposure.groupby("signal_id") if len(g) >= 24)
        result = smooth_targets([{"NONEXISTENT": .1, group.asset_id.iloc[0]: -.1}], group, self.smooth)
        self.assertNotIn("NONEXISTENT", result)

    def test_staleness_is_configurable(self):
        short = dict(self.config, beta_max_staleness_days=1)
        _, decisions, _ = build_targets(self.data, self.signals, short)
        self.assertTrue((decisions.estimated_names == 0).all())

    def test_real_return_clock_convention_is_mandatory(self):
        path = self.path / "metadata.json"
        original = path.read_bytes()
        try:
            metadata = json.loads(original)
            metadata["dataset_mode"] = "real"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "availability_policy"):
                load_dataset(self.path, dict(self.config, dataset_mode="real"))
            metadata["availability_policy"] = {"observable_returns": "TEST ONLY", "classifications_and_shares": "TEST ONLY", "delisting_and_revisions": "TEST ONLY", "feature_return_basis": "final_vintage_proxy", "observable_lag_hours": 3}
            path.write_text(json.dumps(metadata))
            data = load_dataset(self.path, dict(self.config, dataset_mode="real"))
            self.assertEqual(data.metadata["availability_policy"]["feature_return_basis"], "final_vintage_proxy")
        finally:
            path.write_bytes(original)

    def test_calendar_interaction_removes_planted_mechanical_component(self):
        data = copy.deepcopy(self.data)
        config = dict(self.config, near_month=2, far_month=4, signal_calendar_controls=True)
        rng = np.random.default_rng(173)
        schedule = data["schedule"]
        months = pd.period_range("2013-12", "2018-12", freq="M")
        prices = {month: 97. for month in months}
        quotes = []
        def append_quotes(at):
            for month, price in prices.items():
                quotes.append({"contract_id": f"T_{month}", "delivery_month": month, "observed_at": at-pd.Timedelta(hours=2), "available_at": at-pd.Timedelta(hours=1), "settle": price})
        append_quotes(schedule.window_start.iloc[0])
        for row in schedule.itertuples(index=False):
            local = row.signal_at.tz_convert("America/New_York")
            phase = (local.day-1)/(local.days_in_month-1)-.5
            anchor = pd.Period(local.strftime("%Y-%m"), freq="M")
            near = rng.normal(0, 5)
            far = .7*near + 3*phase + 2*phase**2 + 8*near*phase + rng.normal(0, .1)
            prices[anchor+2] -= near/100
            prices[anchor+4] -= far/100
            append_quotes(row.signal_at)
        data.tables["quotes"] = pd.DataFrame(quotes)
        result = build_signals(data, config)
        last = result.iloc[-1]
        self.assertEqual(last.status, "ok")
        self.assertAlmostEqual(last.near_phase_coef, 8., delta=.1)
        self.assertAlmostEqual(last.residual_beta, .7, delta=.05)

    def test_smoothing_does_not_depend_on_future_snapshots(self):
        a, da, _ = build_targets(self.data, self.signals, self.smooth)
        cutoff = self.data["schedule"].signal_at.iloc[130]
        changed = copy.deepcopy(self.data)
        future = set(changed["schedule"].loc[changed["schedule"].signal_at > cutoff, "signal_id"])
        changed["snapshots"].loc[changed["snapshots"].signal_id.isin(future), "eligible"] = 0
        b, db, _ = build_targets(changed, self.signals, self.smooth)
        known = set(changed["schedule"].loc[changed["schedule"].signal_at <= cutoff, "signal_id"])
        self.assertEqual({k:v for k,v in a.items() if k in known}, {k:v for k,v in b.items() if k in known})
