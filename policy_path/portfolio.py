"""Cash-collateralized long/short ledger, weights measured against pre-cost NAV."""
import numpy as np
import pandas as pd


def ledger_step(previous_weights, target_weights, returns, rf, days, cost_bps, borrow_annual):
    keys = sorted(set(previous_weights) | set(target_weights))
    old = np.array([previous_weights.get(k, 0.0) for k in keys])
    target = np.array([target_weights.get(k, 0.0) for k in keys])
    # A sold asset need not have a forward return; a held one must, even if it delists.
    missing = [k for k, w in zip(keys, target) if abs(w) > 1e-12 and k not in returns]
    if missing:
        raise ValueError(f"Missing held-asset returns, never silently drop/reweight: {missing[:8]}")
    r = np.array([returns.get(k, 0.0) for k in keys], float)
    if not np.isfinite(r).all() or (r < -1).any():
        raise ValueError("Invalid held return")
    turnover = float(np.abs(target - old).sum())
    cost = turnover * cost_bps / 10000
    borrow = float(np.maximum(-target, 0).sum()) * borrow_annual * days / 365.25
    cash = 1 - target.sum() - cost
    end_equity = target * (1 + r)
    growth = float(cash * (1 + rf) + end_equity.sum() - borrow)
    if growth <= 0:
        raise ValueError("Portfolio insolvent under this scenario")
    drifted = {k: float(v / growth) for k, v in zip(keys, end_equity) if abs(v) > 1e-12}
    return drifted, {"total_return": growth - 1, "excess_return": growth - 1 - rf,
                     "turnover": turnover, "trade_cost": cost, "borrow_cost": borrow,
                     "gross_target": float(np.abs(target).sum()), "net_target": float(target.sum()),
                     "gross_before_cost_return": float(rf + target @ (r - rf))}


def backtest(data, targets, config, cost_bps=None, borrow_annual=None, decisions=None, exposures=None):
    cost_bps = config["one_way_cost_bps"] if cost_bps is None else cost_bps
    borrow_annual = config["borrow_annual"] if borrow_annual is None else borrow_annual
    mr = data["market_returns"].set_index("signal_id")
    ar = {key: g.set_index("asset_id").total_return.to_dict() for key, g in data["asset_returns"].groupby("signal_id")}
    if config.get("rebalance_band_l1", 0.) > 0 and decisions is None:
        raise ValueError("A positive rebalance band requires PIT decision instructions")
    actions = decisions.set_index("signal_id").rebalance.to_dict() if decisions is not None else {}
    risks = {sid: g.set_index("asset_id") for sid, g in exposures.groupby("signal_id")} if exposures is not None and not exposures.empty else {}
    previous, rows = {}, []
    for row in data["schedule"].itertuples(index=False):
        if row.signal_id not in targets:
            continue
        days = (row.exit_at - row.entry_at).total_seconds() / 86400
        # The hold instruction was fixed at signal_at using only target history.
        # Realized marks affect accounting, never the hold/rebalance decision.
        held = not actions.get(row.signal_id, True)
        executed = dict(previous) if held else targets[row.signal_id]
        risk = risks.get(row.signal_id)
        entry_beta = float(sum(w * risk.loc[a, "market_beta"] for a, w in executed.items())) if risk is not None and set(executed).issubset(risk.index) else np.nan
        try:
            previous, result = ledger_step(previous, executed, ar.get(row.signal_id, {}),
                                            mr.loc[row.signal_id, "rf_return"], days, cost_bps, borrow_annual)
        except ValueError as e:
            raise ValueError(f"Period {row.signal_id}: {e}") from e
        result.update(signal_id=row.signal_id, entry_at=row.entry_at, exit_at=row.exit_at, days=days,
                      held_without_trading=held, entry_estimated_beta=entry_beta,
                      market_return=mr.loc[row.signal_id, "total_return"], rf_return=mr.loc[row.signal_id, "rf_return"],
                      terminal_liquidation_cost=0.0)
        rows.append(result)
    if not rows:
        raise ValueError("No evaluation periods; check evaluation_start")
    # Liquidate all remaining positions at the final marked close, paying one-way cost.
    final_turnover = sum(abs(w) for w in previous.values())
    final_cost = final_turnover * cost_bps / 10000
    before = 1 + rows[-1]["total_return"]
    rows[-1]["total_return"] = before * (1 - final_cost) - 1
    rows[-1]["excess_return"] = rows[-1]["total_return"] - rows[-1]["rf_return"]
    rows[-1]["terminal_liquidation_cost"] = before * final_cost
    rows[-1]["terminal_turnover"] = final_turnover
    result = pd.DataFrame(rows)
    result["nav"] = (1 + result.total_return).cumprod()
    result["cash_nav"] = (1 + result.rf_return).cumprod()
    result["market_nav"] = (1 + result.market_return).cumprod()
    result["gross_before_cost_excess"] = result.gross_before_cost_return - result.rf_return
    result["gross_before_cost_nav"] = (1 + result.gross_before_cost_return).cumprod()
    return result
