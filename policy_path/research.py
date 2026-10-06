"""Expanding forecasts and ex-ante stock exposure estimation."""
import numpy as np
import pandas as pd
from .stats import ols, predict


def aggregate_forecasts(data, signals, config, horizon=1):
    s = data["schedule"].set_index("signal_id")
    mw = data["market_windows"].set_index("signal_id")
    mr = data["market_returns"].set_index("signal_id")
    sg = signals.set_index("signal_id")
    # Forward cumulative labels start at the delayed entry. Maturity is the latest
    # publication among ALL constituent intervals, not merely the final exit.
    ordered = data["schedule"].set_index("signal_id")
    labels, maturity, exits = {}, {}, {}
    for i, sid in enumerate(ordered.index):
        block = ordered.iloc[i:i + horizon]
        if len(block) != horizon:
            labels[sid], maturity[sid], exits[sid] = np.nan, pd.NaT, pd.NaT
            continue
        r = mr.loc[block.index]
        labels[sid] = np.prod(1 + r.total_return) - np.prod(1 + r.rf_return)
        maturity[sid] = block.outcome_available_at.max()
        exits[sid] = block.exit_at.iloc[-1]
    maturity = pd.Series(maturity)
    feature_rows = []
    for row in data["schedule"].itertuples(index=False):
        past = mw.loc[(mw.available_at <= row.signal_at) &
                      (mw.index.map(s.signal_at) <= row.signal_at)]
        past = past.assign(time=past.index.map(s.signal_at)).sort_values("time")
        momentum = np.prod(1 + past.total_return.tail(4)) - 1 if len(past) >= 4 else np.nan
        vol = past.total_return.tail(13).std(ddof=1) if len(past) >= 13 else np.nan
        feature_rows.append({"signal_id": row.signal_id, "signal_z": sg.loc[row.signal_id, "signal_z"],
                             "momentum_4w": momentum, "vol_13w": vol,
                             "actual_excess": labels[row.signal_id]})
    features = pd.DataFrame(feature_rows).set_index("signal_id")
    rows = []
    columns = ["signal_z", "momentum_4w", "vol_13w"]
    for row in data["schedule"].itertuples(index=False):
        current = features.loc[row.signal_id]
        matured = s[(maturity.reindex(s.index) <= row.signal_at) & (s.signal_at < row.signal_at)].index
        training = features.loc[matured].dropna(subset=columns + ["actual_excess"])
        result = {"signal_id": row.signal_id, "signal_at": row.signal_at,
                  "actual_excess": current.actual_excess, "n_train": len(training), "horizon_weeks": horizon,
                  "label_available_at": maturity.loc[row.signal_id],
                  "label_entry_at": row.entry_at, "label_exit_at": exits[row.signal_id],
                  "train_last_outcome_available_at": maturity.loc[training.index].max() if len(training) else pd.NaT,
                  "historical_mean": np.nan, "slope_only": np.nan, "controls_only": np.nan,
                  "slope_controls": np.nan, "status": "warmup"}
        if len(training) >= config["forecast_min_history"] and np.isfinite(current[columns]).all():
            result["historical_mean"] = training.actual_excess.mean()
            result["status"] = "ok"
            for name, cols in (("slope_only", ["signal_z"]), ("controls_only", columns[1:]), ("slope_controls", columns)):
                try:
                    coef = ols(training[cols], training.actual_excess)
                    result[name] = float(predict(coef, current[cols].to_numpy(float)))
                except ValueError:
                    result["status"] = "singular_forecast_fit"
        rows.append(result)
    return pd.DataFrame(rows)


def exposure_history(data, signals):
    s = data["schedule"].set_index("signal_id")
    sw = data["stock_windows"].rename(columns={"available_at": "stock_available_at", "total_return": "stock_return"})
    mw = data["market_windows"].rename(columns={"available_at": "market_available_at", "total_return": "market_return"})
    history = sw.merge(mw, on="signal_id", validate="many_to_one").merge(
        signals[["signal_id", "signal_bps"]], on="signal_id", validate="many_to_one")
    history["signal_at"] = history.signal_id.map(s.signal_at)
    history["stock_excess"] = history.stock_return - history.rf_return
    history["market_excess"] = history.market_return - history.rf_return
    return history.sort_values(["signal_at", "asset_id"])


def estimate_exposures(history, row, assets, config):
    # Exclude the CURRENT window even if already observable, so the current surprise
    # cannot mechanically determine its own cross-sectional exposure ranking.
    candidates = history[(history.signal_at < row.signal_at) &
                         (history.stock_available_at <= row.signal_at) &
                         (history.market_available_at <= row.signal_at) &
                         history.asset_id.isin(assets)].dropna(subset=["signal_bps"])
    estimates = []
    for asset, h in candidates.groupby("asset_id", sort=True):
        h = h.tail(config["beta_lookback"])
        if len(h) < config["beta_min_history"]:
            continue
        if (row.signal_at - h.signal_at.iloc[-1]).total_seconds() > config.get("beta_max_staleness_days", 21) * 86400:
            continue
        if h.signal_bps.std() < config["signal_std_floor_bps"]:
            continue
        try:
            fit = ols(h[["market_excess", "signal_bps"]], h.stock_excess)
        except ValueError:
            continue
        estimates.append({"asset_id": asset, "market_beta": fit[1], "policy_beta_per_bp": fit[2],
                          "n_beta": len(h), "beta_last_signal_id": h.signal_id.iloc[-1]})
    return pd.DataFrame(estimates, columns=["asset_id", "market_beta", "policy_beta_per_bp", "n_beta", "beta_last_signal_id"])


def form_weights(snapshot, exposure, z, config):
    """High-minus-low policy-beta seeds, projected to sector/beta neutrality."""
    universe = snapshot.merge(exposure, on="asset_id", how="inner", validate="one_to_one")
    scaled = config.get("position_mode", "binary") == "scaled_smooth"
    if not np.isfinite(z):
        return {}, universe, "missing_signal"
    if not scaled and abs(z) < config["signal_threshold_z"]:
        return {}, universe, "inactive_signal"
    if scaled and z == 0:
        return {}, universe, "zero_signal"
    counts = universe.groupby("sector").asset_id.transform("size")
    universe = universe[counts >= config["min_sector_names"]].sort_values("asset_id").reset_index(drop=True)
    if len(universe) < config["min_names"]:
        return {}, universe, "insufficient_exposures"
    if universe.policy_beta_per_bp.std() < 1e-12:
        return {}, universe, "insufficient_policy_variation"
    seed = np.zeros(len(universe))
    groups = list(universe.groupby("sector", sort=True))
    for _, group in groups:
        order = group.sort_values(["policy_beta_per_bp", "asset_id"]).index.to_numpy()
        k = max(1, int(len(order) * config["tail_fraction"]))
        seed[order[:k]] = -1 / (2 * k * len(groups))
        seed[order[-k:]] = 1 / (2 * k * len(groups))
    a = pd.get_dummies(universe.sector, dtype=float).to_numpy().T
    if config["neutralize_beta"]:
        beta = universe.market_beta.to_numpy(float)
        # Scaling a constraint preserves the null space and improves conditioning.
        a = np.vstack([a, beta / max(np.std(beta), 1e-8)])
    weights = seed - np.linalg.pinv(a) @ (a @ seed)
    gross = np.abs(weights).sum()
    if gross < 1e-10:
        return {}, universe, "degenerate_neutralization"
    amplitude = min(abs(z) / config.get("z_full_exposure", 2.), 1.) if scaled else 1.
    weights = weights / gross * np.sign(z) * amplitude
    if weights @ universe.policy_beta_per_bp.to_numpy() * np.sign(z) <= 0:
        return {}, universe, "policy_exposure_reversed"
    if np.max(np.abs(weights)) > config["max_weight"] + 1e-10:
        if not scaled:
            return {}, universe, "position_cap_exceeded"
        weights *= config["max_weight"] / np.max(np.abs(weights))
    if np.max(np.abs(a @ weights)) > 1e-8:
        raise ValueError("Neutralization failed numerical tolerance")
    universe["seed_weight"] = seed
    universe["target_weight"] = weights
    return dict(zip(universe.asset_id, weights)), universe, "active"


def smooth_targets(raw_history, universe, config):
    """Equal-weight trailing target cohorts, reprojected on TODAY's PIT risk set.

    This is target averaging, not literal buy-and-hold tranches. Missing startup
    cohorts are cash. Historical names no longer eligible cannot survive smoothing.
    """
    n = config.get("smoothing_weeks", 3)
    if len(universe) < config["min_names"]:
        return {}
    count = universe.groupby("sector").asset_id.transform("size")
    u = universe[count >= config["min_sector_names"]].copy()
    if len(u) < config["min_names"]:
        return {}
    seed = np.array([sum(book.get(a, 0.) for book in raw_history[-n:]) / n for a in u.asset_id])
    constraints = pd.get_dummies(u.sector, dtype=float).to_numpy().T
    if config["neutralize_beta"]:
        constraints = np.vstack([constraints, u.market_beta.to_numpy()])
    w = seed - np.linalg.pinv(constraints) @ (constraints @ seed)
    if np.abs(w).sum() < 1e-12:
        return {}
    scale = min(1., 1. / np.abs(w).sum(), config["max_weight"] / np.abs(w).max())
    w *= scale
    return {a: float(v) for a, v in zip(u.asset_id, w) if abs(v) > 1e-12}


def reference_within_risk(book, universe, config):
    """Decision-time check on the last committed target, NOT future realized marks."""
    if not book or not set(book).issubset(set(universe.asset_id)):
        return False
    u = universe.set_index("asset_id").loc[list(book)]
    w = pd.Series(book).reindex(u.index)
    sector = w.groupby(u.sector).sum().abs().max()
    beta = float(w @ u.market_beta)
    return (sector <= config.get("hold_sector_tolerance", .02) and
            abs(beta) <= config.get("hold_beta_tolerance", .03) and
            w.abs().max() <= config["max_weight"] + 1e-10 and w.abs().sum() <= 1 + 1e-10)


def build_targets(data, signals, config):
    history = exposure_history(data, signals)
    sg = signals.set_index("signal_id")
    targets, decisions, exposures = {}, [], []
    raw_history = []
    previous_reference = {}
    snapshots_by_week = {sid: g for sid, g in data["snapshots"].groupby("signal_id", sort=False)}
    start = pd.Timestamp(config["evaluation_start"], tz="UTC")
    for row in data["schedule"].itertuples(index=False):
        if row.signal_at < start:
            continue
        snapshot = snapshots_by_week.get(row.signal_id, data["snapshots"].iloc[:0])
        snapshot = snapshot[(snapshot.eligible == 1) &
                            (snapshot.shortable == 1) & (snapshot.price >= config["min_price"]) &
                            (snapshot.market_cap >= config["min_market_cap"])]
        # Selection uses only the PIT snapshot; no filter based on forward returns.
        snapshot = snapshot.sort_values(["market_cap", "asset_id"], ascending=[False, True]).head(config["max_universe"])
        exposure = estimate_exposures(history, row, set(snapshot.asset_id), config)
        z = sg.loc[row.signal_id, "signal_z"]
        weights, universe, reason = form_weights(snapshot, exposure, z, config)
        raw_history.append(weights)
        if config.get("position_mode", "binary") == "scaled_smooth":
            # No smoothing through missing signals or insufficient/risk-invalid universes.
            if reason in ("active", "zero_signal"):
                weights = smooth_targets(raw_history, universe, config)
                reason = "active_smoothed" if weights else "smoothed_cash"
            else:
                raw_history = []
            universe = universe.copy()
            universe["target_weight"] = universe.asset_id.map(weights).fillna(0.)
        desired = dict(weights)
        difference = sum(abs(desired.get(a, 0.) - previous_reference.get(a, 0.)) for a in set(desired) | set(previous_reference))
        hold = bool(desired and difference < config.get("rebalance_band_l1", 0.) and reference_within_risk(previous_reference, universe, config))
        if hold:
            weights = dict(previous_reference)
            reason += "_held_band"
        else:
            previous_reference = dict(weights)
        universe = universe.copy()
        universe["desired_weight"] = universe.asset_id.map(desired).fillna(0.)
        universe["target_weight"] = universe.asset_id.map(weights).fillna(0.)
        targets[row.signal_id] = weights
        if not universe.empty:
            universe = universe.copy()
            universe["signal_id"] = row.signal_id
            if "target_weight" not in universe:
                universe["target_weight"] = 0.0
            exposures.append(universe)
        w = pd.Series(weights, dtype=float)
        beta = exposure.set_index("asset_id").market_beta if not exposure.empty else pd.Series(dtype=float)
        gamma = exposure.set_index("asset_id").policy_beta_per_bp if not exposure.empty else pd.Series(dtype=float)
        decisions.append({"signal_id": row.signal_id, "signal_z": z, "decision": reason,
                          "rebalance": not hold, "reference_target_change_l1": difference,
                          "eligible_names": len(snapshot), "estimated_names": len(exposure),
                          "gross_weight": float(w.abs().sum()), "net_weight": float(w.sum()),
                          "ex_ante_market_beta": float((w * beta.reindex(w.index)).sum()),
                          "ex_ante_policy_beta_per_bp": float((w * gamma.reindex(w.index)).sum())})
    exposure_frame = pd.concat(exposures, ignore_index=True) if exposures else pd.DataFrame()
    return targets, pd.DataFrame(decisions), exposure_frame


def cross_sectional_diagnostics(data, signals, exposures):
    """Weekly cross-sectional slope, conditional on PIT beta, size and sector.

    Each period uses the entire estimated universe. If a forward outcome is missing,
    skip the WHOLE diagnostic period and disclose it, never drop that stock alone.
    These are research diagnostics; future outcomes never feed target weights.
    """
    if exposures.empty:
        return pd.DataFrame(columns=["signal_id", "signal_z", "status", "policy_premium"])
    ar = data["asset_returns"]
    returns_by_week = {sid: g[["asset_id", "total_return"]] for sid, g in ar.groupby("signal_id", sort=False)}
    rf = data["market_returns"].set_index("signal_id").rf_return
    z = signals.set_index("signal_id").signal_z
    rows = []
    clock = data["schedule"].set_index("signal_id").signal_at
    ordered = exposures.assign(_time=exposures.signal_id.map(clock)).sort_values(["_time", "asset_id"])
    for sid, e in ordered.groupby("signal_id", sort=False):
        h = e.merge(returns_by_week.get(sid, ar[["asset_id", "total_return"]].iloc[:0]), on="asset_id", how="left", validate="one_to_one")
        row = {"signal_id": sid, "signal_z": z.loc[sid], "n_names": len(h), "status": "ok",
               "policy_premium": np.nan, "rank_ic": np.nan, "signal_oriented_ic": np.nan}
        if h.total_return.isna().any():
            row["status"] = "missing_outcome_skip_entire_period"
        elif len(h) < 10 or h.policy_beta_per_bp.std() < 1e-12 or not np.isfinite(z.loc[sid]):
            row["status"] = "insufficient_variation"
        else:
            gamma = (h.policy_beta_per_bp - h.policy_beta_per_bp.mean()) / h.policy_beta_per_bp.std()
            y = h.total_return - rf.loc[sid]
            columns = [gamma.to_numpy(), h.market_beta.to_numpy(), np.log(h.market_cap.to_numpy())]
            sector = pd.get_dummies(h.sector, drop_first=True, dtype=float).to_numpy()
            x = np.column_stack(columns + [sector])
            try:
                coef = ols(x, y)
                row["policy_premium"] = coef[1]
                row["rank_ic"] = float(gamma.rank().corr(y.rank()))
                row["signal_oriented_ic"] = row["rank_ic"] * np.sign(z.loc[sid])
            except ValueError:
                row["status"] = "singular_cross_section"
        rows.append(row)
    return pd.DataFrame(rows)
