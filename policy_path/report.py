"""Self-contained HTML research diagnostics and machine-readable provenance."""
import hashlib
import html
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from . import __version__
from .stats import performance, block_sharpe_interval, hac_regression, hac_lags
from .signals import build_signals
from .research import aggregate_forecasts, build_targets, cross_sectional_diagnostics
from .portfolio import backtest
from .data import load_dataset, validate_config


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def nav_svg(ledger):
    width, height, margin = 900, 300, 40
    arrays = [np.r_[1., ledger[c].to_numpy(float)] for c in ("nav", "cash_nav", "market_nav")]
    low, high = min(a.min() for a in arrays), max(a.max() for a in arrays)
    if high - low < 1e-9:
        high = low + .01
    pieces = [f'<svg role="img" aria-label="Cumulative total NAV; strategy, cash, and equity benchmark have different risk exposures" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    for y in np.linspace(low, high, 5):
        py = height - margin - (y - low) / (high - low) * (height - 2 * margin)
        pieces.append(f'<line x1="{margin}" x2="{width-margin}" y1="{py:.1f}" y2="{py:.1f}" stroke="#dce2eb"/><text x="0" y="{py:.1f}" font-size="12">{y:.2f}</text>')
    for a, color in zip(arrays, ("#135ac7", "#687386", "#cb7a18")):
        points = " ".join(f"{margin+i/(len(a)-1)*(width-2*margin):.1f},{height-margin-(v-low)/(high-low)*(height-2*margin):.1f}" for i, v in enumerate(a))
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    pieces.append(f'<text x="{margin}" y="295" font-size="12">{ledger.entry_at.iloc[0].date()}</text><text x="{width-110}" y="295" font-size="12">{ledger.exit_at.iloc[-1].date()}</text></svg>')
    return "".join(pieces)


def forecast_summary(forecasts, start):
    f = forecasts[forecasts.signal_at >= start].dropna(subset=["actual_excess", "historical_mean", "slope_only", "controls_only", "slope_controls"])
    if f.empty:
        return {"n_common_oos": 0}
    err0 = ((f.actual_excess - f.historical_mean) ** 2).sum()
    errc = ((f.actual_excess - f.controls_only) ** 2).sum()
    result = {"n_common_oos": len(f), "models": {}}
    for name in ("historical_mean", "slope_only", "controls_only", "slope_controls"):
        errors = f.actual_excess - f[name]
        result["models"][name] = {"mse": float((errors ** 2).mean()),
                                  "oos_r2_vs_mean": float(1 - (errors ** 2).sum() / err0) if err0 > 0 else None,
                                  "direction_accuracy": float((np.sign(f[name]) == np.sign(f.actual_excess)).mean())}
    result["incremental_oos_r2_vs_controls"] = float(1 - ((f.actual_excess - f.slope_controls) ** 2).sum() / errc) if errc > 0 else None
    return result


def run_research(data_path, config, output):
    config = validate_config(config)
    data = load_dataset(data_path, config)
    signals = build_signals(data, config)
    forecasts = aggregate_forecasts(data, signals, config)
    targets, decisions, exposures = build_targets(data, signals, config)
    cross_section = cross_sectional_diagnostics(data, signals, exposures)
    ledger = backtest(data, targets, config, decisions=decisions, exposures=exposures)
    horizon_forecasts = [forecasts]
    for horizon in config.get("forecast_horizons", [1, 2, 4, 8]):
        if horizon != 1:
            horizon_forecasts.append(aggregate_forecasts(data, signals, config, horizon=horizon))
    multihorizon = pd.concat(horizon_forecasts, ignore_index=True)
    start = pd.Timestamp(config["evaluation_start"], tz="UTC")
    summary = {"dataset_mode": data.metadata["dataset_mode"], "provenance": data.metadata["provenance"],
               "availability_policy": data.metadata.get("availability_policy", {"feature_return_basis": "synthetic"}),
               "strategy": performance(ledger.total_return, ledger.excess_return, ledger.days),
               "forecast": forecast_summary(forecasts, start),
               "signal_status_counts": signals.status.value_counts().to_dict(),
               "portfolio_decision_counts": decisions.decision.value_counts().to_dict(),
               "active_fraction": float((ledger.gross_target > 0).mean()),
               "mean_weekly_turnover": float(ledger.turnover.mean()),
               "terminal_cost_included": True,
               "sharpe_block_interval": block_sharpe_interval(ledger.excess_return, ledger.days, config["bootstrap_block"], config["bootstrap_reps"], config["seed"])}
    summary["cross_section_status_counts"] = cross_section.status.value_counts().to_dict()
    cs = cross_section[cross_section.status == "ok"].dropna(subset=["policy_premium", "signal_z"])
    try:
        summary["descriptive_hac_policy_premium_on_signal_z"] = hac_regression(cs[["signal_z"]], cs.policy_premium, hac_lags(config, len(cs)))
    except ValueError as e:
        summary["descriptive_hac_policy_premium_on_signal_z"] = {"unavailable": str(e)}
    summary["cross_section_inference_caveat"] = "Generated historical betas; HAC intervals do not propagate first-stage beta uncertainty. Size/sector/market beta controlled, value/momentum not yet included."
    # Descriptive evaluation-period regression. This is not the recursive forecast fit.
    h = forecasts.merge(signals[["signal_id", "signal_z"]], on="signal_id")
    h = h[(h.signal_at >= start)].dropna(subset=["signal_z", "actual_excess"])
    try:
        summary["descriptive_hac_market_on_signal_z"] = hac_regression(h[["signal_z"]], h.actual_excess, hac_lags(config, len(h)))
    except ValueError as e:
        summary["descriptive_hac_market_on_signal_z"] = {"unavailable": str(e)}
    summary["gross_before_cost_same_holdings"] = performance(ledger.gross_before_cost_return, ledger.gross_before_cost_excess, ledger.days)
    summary["cost_attribution"] = {
        "sum_trade_cost_fraction_of_each_period_nav": float((ledger.trade_cost * (1 + ledger.rf_return)).sum()),
        "sum_borrow_cost_fraction_of_each_period_nav": float(ledger.borrow_cost.sum()),
        "sum_terminal_cost_fraction_of_each_period_nav": float(ledger.terminal_liquidation_cost.sum()),
        "gross_minus_net_final_wealth": float(ledger.gross_before_cost_nav.iloc[-1] - ledger.nav.iloc[-1]),
        "warning": "Period-NAV cost sums are not a compounded cumulative return attribution."}
    summary["entry_estimated_beta"] = {"mean": float(ledger.entry_estimated_beta.mean()), "max_abs": float(ledger.entry_estimated_beta.abs().max())}
    try:
        summary["realized_market_beta_hac"] = hac_regression((ledger.market_return - ledger.rf_return).to_numpy(), ledger.excess_return, hac_lags(config, len(ledger)))
    except ValueError as e:
        summary["realized_market_beta_hac"] = {"unavailable": str(e)}
    summary["held_without_trading_periods"] = int(ledger.held_without_trading.sum())
    horizon_rows = []
    for horizon, f in multihorizon.groupby("horizon_weeks"):
        stats = forecast_summary(f, start)
        h = f.merge(signals[["signal_id", "signal_z"]], on="signal_id")
        h = h[h.signal_at >= start].dropna(subset=["signal_z", "actual_excess"])
        lag = hac_lags(config, len(h), int(horizon))
        # Greedy interval scheduling counts actual nonoverlapping label windows,
        # including gaps/holidays. It does not claim statistical independence.
        nonoverlapping, last_exit = 0, None
        for row in h.sort_values("label_exit_at").itertuples(index=False):
            if last_exit is None or row.label_entry_at >= last_exit:
                nonoverlapping += 1
                last_exit = row.label_exit_at
        try:
            fit = hac_regression(h[["signal_z"]], h.actual_excess, lag)
            slope, se = fit["coef"][1], fit["se"][1]
        except ValueError:
            slope, se = np.nan, np.nan
        for model, values in stats.get("models", {}).items():
            horizon_rows.append({"horizon_weeks": horizon, "model": model, "n_oos": stats["n_common_oos"],
                                 "n_descriptive": len(h), "max_nonoverlapping_label_windows": nonoverlapping,
                                 "hac_lags": lag, "signal_coef": slope, "signal_hac_se": se, **values})
    horizon_table = pd.DataFrame(horizon_rows)
    summary["multihorizon_warning"] = "Overlapping cumulative returns; HAC uses max(prespecified lag rule, horizon-1). Maximum nonoverlapping interval count is not an effective-sample-size estimate; no multiple-testing correction."
    sensitivity = []
    for cost in sorted(set([0., 5., 10., 25., float(config["one_way_cost_bps"])])):
        for borrow in sorted(set([0., .02, .05, float(config["borrow_annual"])])):
            l = backtest(data, targets, config, cost_bps=cost, borrow_annual=borrow, decisions=decisions, exposures=exposures)
            sensitivity.append({"one_way_cost_bps": cost, "borrow_annual": borrow,
                                **performance(l.total_return, l.excess_return, l.days)})
    sensitivity = pd.DataFrame(sensitivity)
    execution_rows = []
    def execution_row(name, book):
        net = performance(book.total_return, book.excess_return, book.days)
        gross = performance(book.gross_before_cost_return, book.gross_before_cost_excess, book.days)
        return {"specification": name, "mean_turnover": book.turnover.mean(),
                "mean_gross_exposure": book.gross_target.mean(), "gross_sharpe": gross.get("sharpe_excess"),
                "net_sharpe": net.get("sharpe_excess"), "gross_final_nav": book.gross_before_cost_nav.iloc[-1],
                "net_final_nav": book.nav.iloc[-1], "cash_final_nav": book.cash_nav.iloc[-1],
                "sum_trade_cost": (book.trade_cost * (1 + book.rf_return)).sum(), "sum_borrow_cost": book.borrow_cost.sum(),
                "held_without_trading": int(book.held_without_trading.sum())}
    execution_rows.append(execution_row(config.get("position_mode", "binary"), ledger))
    if config.get("position_mode", "binary") == "scaled_smooth":
        baseline = dict(config, position_mode="binary", smoothing_weeks=1, rebalance_band_l1=0.)
        bt, bd, be = build_targets(data, signals, baseline)
        baseline_ledger = backtest(data, bt, baseline, decisions=bd, exposures=be)
        execution_rows.append(execution_row("binary_same_signals", baseline_ledger))
        for label, options in (("scaled_only", {"smoothing_weeks": 1, "rebalance_band_l1": 0.}),
                               ("scaled_smooth_no_band", {"rebalance_band_l1": 0.})):
            variant = dict(config, **options)
            vt, vd, ve = build_targets(data, signals, variant)
            vl = backtest(data, vt, variant, decisions=vd, exposures=ve)
            execution_rows.append(execution_row(label, vl))
        positive_signals = signals.copy()
        positive_signals["signal_z"] = positive_signals.signal_z.abs()
        ft, fd, fe = build_targets(data, positive_signals, config)
        fixed_ledger = backtest(data, ft, config, decisions=fd, exposures=fe)
        summary["ablation_caveat"] = "Absolute signal enters before smoothing. Rebalancing dates may differ; amplitude and universe still depend on the signal."
    else:
        z = signals.set_index("signal_id").signal_z
        fixed = {sid: {asset: float(w * np.sign(z.loc[sid])) for asset, w in weights.items()} for sid, weights in targets.items()}
        fixed_ledger = backtest(data, fixed, config, decisions=decisions, exposures=exposures)
        summary["ablation_caveat"] = "Still uses |signal| for activity; not a fully signal-free strategy or a formal significance test."
    summary["fixed_direction_ablation"] = performance(fixed_ledger.total_return, fixed_ledger.excess_return, fixed_ledger.days)
    execution_comparison = pd.DataFrame(execution_rows)
    execution_comparison["turnover_per_mean_gross"] = execution_comparison.mean_turnover / execution_comparison.mean_gross_exposure.replace(0, np.nan)
    coverage = signals[["signal_id", "signal_at", "status"]].merge(ledger[["signal_id", "gross_target"]], on="signal_id", how="left")
    coverage["year"] = coverage.signal_at.dt.tz_convert("America/New_York").dt.year
    coverage = coverage.assign(valid=coverage.status.eq("ok"), low_variance=coverage.status.eq("low_signal_variance"),
                               portfolio_period=coverage.gross_target.notna(), active=coverage.gross_target.fillna(0).gt(1e-12))
    coverage = coverage.groupby("year").agg(signal_rows=("signal_id", "size"), valid_signals=("valid", "sum"),
                                             low_variance=("low_variance", "sum"), portfolio_periods=("portfolio_period", "sum"),
                                             active_periods=("active", "sum")).reset_index()
    group = ledger.merge(signals[["signal_id", "near_rate_pct"]], on="signal_id")
    group["rate_bucket"] = np.where(group.near_rate_pct.isna(), "missing_rate", np.where(group.near_rate_pct <= .25, "near_rate_le_0.25pct", "near_rate_gt_0.25pct"))
    regimes = []
    for label, g in group.groupby("rate_bucket"):
        # Disconnected weeks: report conditional moments, not a misleading compounded drawdown.
        ppy = len(g) / (g.days.sum() / 365.25)
        regimes.append({"rate_bucket": label, "n_periods": len(g), "mean_excess": g.excess_return.mean(),
                        "annualized_vol_excess": g.excess_return.std(ddof=1) * np.sqrt(ppy)})
    output = Path(output)
    if output.resolve() == Path(data_path).resolve():
        raise ValueError("Results directory must differ from input directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("signals", signals), ("forecasts", forecasts), ("decisions", decisions),
                        ("exposures_and_weights", exposures), ("ledger", ledger),
                        ("cost_sensitivity", sensitivity), ("rate_buckets", pd.DataFrame(regimes)),
                        ("fixed_direction_ledger", fixed_ledger)):
        frame.to_csv(output / f"{name}.csv", index=False)
    cross_section.to_csv(output / "cross_section.csv", index=False)
    multihorizon.to_csv(output / "forecasts_multihorizon.csv", index=False)
    horizon_table.to_csv(output / "horizon_summary.csv", index=False)
    execution_comparison.to_csv(output / "execution_comparison.csv", index=False)
    coverage.to_csv(output / "coverage_by_year.csv", index=False)
    summary = clean_json(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    (output / "config_used.json").write_text(json.dumps(config, indent=2))
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "package_version": __version__,
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "dataset_mode": config["dataset_mode"], "metadata": data.metadata,
                "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(data_path).glob("*.csv"))},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    mode = "SYNTHETIC DEMO — 人工模拟数据，不构成策略有效性的证据" if config["dataset_mode"] == "synthetic" else "REAL INPUT DATA — 来源声明尚需人工审核"
    if data.metadata.get("availability_policy", {}).get("feature_return_basis") == "final_vintage_proxy":
        mode += " — FINAL-VINTAGE PROXY：最终版本回填可观测时点，不能声称严格 PIT"
    warnings = ["策略持有股票，期货只用于构造信号。未模拟期货交易。", "权重由过去数据决定；未来收益只用于事后记账。",
                "市场基准、现金、股票多空组合风险不同，净值线不能直接证明 alpha。",
                "收盘成交、全额现金抵押、统一借券费为研究假设；未验证逐笔成交、容量、保证金或真实借券可得性。",
                "估计 beta 中性不等于实现 beta 中性。低利率分组只是近月隐含利率阈值，不是正式货币政策状态识别。",
                "Bootstrap 区间假设局部可重复的收益结构，未修正多重检验；样本很短时解释有限。",
                "本工具不是原论文的逐项精确复现；合约定义、执行延迟和样本外程序有明确改动。"]
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Policy Path Equity — {html.escape(config['dataset_mode'])}</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#162439;background:#f5f7fa;margin:0}}main{{max-width:1040px;margin:auto;padding:32px}}h1{{line-height:1.2}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}.badge{{background:#fff0ce;padding:16px;font-weight:700;border-left:5px solid #bd7b10}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}td,th{{padding:8px;border-bottom:1px solid #dce2eb;text-align:right}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}svg{{width:100%;height:auto}}.blue{{color:#135ac7}}.grey{{color:#687386}}.orange{{color:#cb7a18}}</style>
<main><h1>Policy-Path Repricing &amp; Equity Returns</h1><p class="badge">{html.escape(mode)}</p>
<p>{html.escape(data.metadata['provenance'])}</p><section><h2>净值与风险说明</h2>{nav_svg(ledger)}<p><span class="blue">蓝：成本后策略</span> · <span class="grey">灰：现金</span> · <span class="orange">橙：股票市场基准</span></p>
<p>{'模拟数据预先植入了延迟反应关系；示例表现只能用来检查软件流程。' if config['dataset_mode'] == 'synthetic' else '结果依赖输入数据的来源与完整性；real 标签不代表供应商连接或实盘成交已验证。'}</p></section>
<section><h2>执行规则对照：相同数据与信号</h2><p>同时报告毛/净收益和平均总敞口；平滑可能降低敞口，换手下降不等于 alpha 改善。成本列是逐期 NAV 比例之和。</p>{execution_comparison.to_html(index=False, border=0, float_format=lambda x: f'{x:.5f}')}</section>
<section><h2>1 / 2 / 4 / 8 周预测路径</h2><p>各期限独立等待标签成熟；尾部未实现的多周结果不进入评估。重叠收益不可视为独立周。</p>{horizon_table.to_html(index=False, border=0, float_format=lambda x: f'{x:.5f}')}</section>
<section><h2>逐年覆盖率</h2><p>区分日历样本、有效信号与活跃持仓周。不能将不显著解释为已排除所有经济上有意义的效应。</p>{coverage.to_html(index=False, border=0)}</section>
<section><h2>统计与预测检验</h2><pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre></section>
<section><h2>交易成本 / 借券费敏感性</h2><p>整张表同时展示；不按最优结果选择参数。成本 bps 是每一元成交金额的单边费用。年化借券费为小数。</p>{sensitivity.to_html(index=False, float_format=lambda x: f'{x:.5f}', border=0)}</section>
<section><h2>解释边界</h2><ul>{''.join('<li>'+html.escape(w)+'</li>' for w in warnings)}</ul></section>
<section><h2>审计文件</h2><p>同目录包含 signals.csv、forecasts.csv、exposures_and_weights.csv、ledger.csv、manifest.json，以及配置和成本情景结果。</p></section></main></html>'''
    (output / "report.html").write_text(report, encoding="utf-8")
    return summary
