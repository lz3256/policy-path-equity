"""Fixed-delivery-month policy repricing with past-only residualization."""
import numpy as np
import pandas as pd
from .stats import ols


def delivery_months(data, row, config):
    local = row.signal_at.tz_convert("America/New_York")
    anchor = pd.Period(local.strftime("%Y-%m"), freq="M")
    if config.get("signal_mode", "calendar_months") == "calendar_months":
        return {"near": anchor + config["near_month"], "far": anchor + config["far_month"]}, None
    calendars = data["meetings"]
    known = calendars[calendars.calendar_asof <= row.signal_at]
    if known.empty:
        return None, None
    vintage = known.calendar_asof.max()
    upcoming = known[(known.calendar_asof == vintage) & (known.decision_at > row.signal_at)].sort_values("decision_at")
    near, far = config.get("meeting_near_index", 1), config.get("meeting_far_index", 3)
    if len(upcoming) < far:
        return None, vintage
    months = {}
    for label, index in (("near", near), ("far", far)):
        date = upcoming.iloc[index - 1].decision_at.tz_convert("America/New_York")
        months[label] = pd.Period(date.strftime("%Y-%m"), freq="M") + 1
    if months["near"] >= months["far"]:
        return None, vintage
    return months, vintage


def residual_features(near, phase, calendar_controls):
    near, phase = np.asarray(near), np.asarray(phase)
    if not calendar_controls:
        return near[..., None]
    return np.stack([near, phase, phase ** 2, near * phase], axis=-1)


def get_quote(quotes, delivery_month, cutoff, max_age_hours):
    candidates = quotes[(quotes.delivery_month == delivery_month) &
                        (quotes.observed_at <= cutoff) & (quotes.available_at <= cutoff)]
    if candidates.empty:
        return None
    row = candidates.sort_values("observed_at").iloc[-1]
    if (cutoff - row.observed_at).total_seconds() > max_age_hours * 3600:
        return None
    return row


def build_signals(data, config):
    records, q = [], data["quotes"]
    for row in data["schedule"].itertuples(index=False):
        local = row.signal_at.tz_convert("America/New_York")
        phase = (local.day - 1) / (local.days_in_month - 1) - .5
        months, calendar_asof = delivery_months(data, row, config)
        record = {"signal_id": row.signal_id, "signal_at": row.signal_at,
                  "delta_near_bps": np.nan, "delta_far_bps": np.nan,
                  "near_rate_pct": np.nan, "signal_bps": np.nan, "signal_z": np.nan,
                  "residual_intercept_bps": np.nan, "residual_beta": np.nan,
                  "train_last_signal_id": "", "status": "missing_quote", "month_phase": phase,
                  "calendar_asof": calendar_asof, "calendar_controls": config.get("signal_calendar_controls", False)}
        if months is None:
            record["status"] = "missing_meeting_calendar"
            records.append(record)
            continue
        ok = True
        for label, month in months.items():
            record[f"{label}_delivery_month"] = str(month)
            midpoint = pd.Timestamp(f"{month}-15", tz="America/New_York")
            record[f"{label}_days_to_midmonth"] = (midpoint - local).total_seconds() / 86400
            start = get_quote(q, month, row.window_start, config["quote_max_age_hours"])
            end = get_quote(q, month, row.signal_at, config["quote_max_age_hours"])
            if start is None or end is None or start.observed_at == end.observed_at:
                ok = False
                continue
            # Both observations MUST refer to the contract selected at the END of the window.
            record[f"{label}_contract"] = end.contract_id
            record[f"{label}_start_observed_at"] = start.observed_at
            record[f"{label}_end_observed_at"] = end.observed_at
            record[f"delta_{label}_bps"] = -(end.settle - start.settle) * 100
            if label == "near":
                record["near_rate_pct"] = 100 - end.settle
        history = [r for r in records if np.isfinite(r["delta_near_bps"]) and np.isfinite(r["delta_far_bps"])]
        history = history[-config["signal_lookback"]:]
        if ok:
            record["status"] = "warmup"
            if len(history) >= config["signal_min_history"]:
                x = residual_features([r["delta_near_bps"] for r in history], [r["month_phase"] for r in history], config.get("signal_calendar_controls", False))
                y = np.array([r["delta_far_bps"] for r in history])
                try:
                    coef = ols(x, y)
                    sigma = np.std(y - coef[0] - x @ coef[1:], ddof=x.shape[1] + 1)
                    if sigma < config["signal_std_floor_bps"]:
                        record["status"] = "low_signal_variance"
                    else:
                        current = residual_features(record["delta_near_bps"], phase, config.get("signal_calendar_controls", False))
                        value = record["delta_far_bps"] - coef[0] - current @ coef[1:]
                        record.update(signal_bps=value, signal_z=value / sigma,
                                      residual_intercept_bps=coef[0], residual_beta=coef[1],
                                      train_last_signal_id=history[-1]["signal_id"], status="ok")
                        if config.get("signal_calendar_controls", False):
                            record.update(phase_coef=coef[2], phase_squared_coef=coef[3], near_phase_coef=coef[4],
                                          conditional_near_beta=coef[1] + coef[4] * phase)
                except ValueError:
                    record["status"] = "singular_residual_fit"
        else:
            # Do not let a partially observed signal enter a subsequent fit.
            record["delta_near_bps"] = record["delta_far_bps"] = np.nan
        records.append(record)
    return pd.DataFrame(records)
