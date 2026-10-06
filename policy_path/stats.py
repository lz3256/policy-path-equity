"""Small transparent statistical routines; no optimizer or auto model selection."""
import numpy as np


def ols(x, y):
    """OLS including intercept, with scaled regressors and a rank check."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x = x[:, None]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OLS input must be finite")
    if len(y) <= x.shape[1] + 1:
        raise ValueError("OLS has insufficient observations")
    scale = x.std(axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("OLS regressor has no variation")
    design = np.column_stack([np.ones(len(y)), x / scale])
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("OLS design is rank deficient")
    coef[1:] /= scale
    return coef


def predict(coef, x):
    return coef[0] + np.asarray(x) @ coef[1:]


def hac_lags(config, n, horizon=1):
    """Predeclared rule, bounded below by overlapping-label horizon minus one."""
    base = config["hac_lags"] if config.get("hac_lag_rule", "fixed") == "fixed" else int(np.floor(4 * (max(n, 1) / 100) ** (2 / 9)))
    return min(max(base, horizon - 1), max(n - 1, 0))


def hac_regression(x, y, lags=4):
    """Newey-West Bartlett covariance; asymptotic 95% intervals, NOT p-values."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x = x[:, None]
    coef = ols(x, y)
    design = np.column_stack([np.ones(len(y)), x])
    resid = y - design @ coef
    score = design * resid[:, None]
    meat = score.T @ score
    for lag in range(1, min(lags, len(y) - 1) + 1):
        g = score[lag:].T @ score[:-lag]
        meat += (1 - lag / (lags + 1)) * (g + g.T)
    bread = np.linalg.pinv(design.T @ design)
    cov = bread @ meat @ bread * len(y) / (len(y) - design.shape[1])
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return {"n": len(y), "coef": coef.tolist(), "se": se.tolist(),
            "ci95_low": (coef - 1.96 * se).tolist(),
            "ci95_high": (coef + 1.96 * se).tolist(), "lags": lags}


def performance(total, excess, days):
    total, excess, days = map(lambda z: np.asarray(z, float), (total, excess, days))
    if len(total) == 0:
        return {"n_periods": 0}
    nav = np.r_[1.0, np.cumprod(1 + total)]
    years = days.sum() / 365.25
    periods_per_year = len(total) / years
    std = np.std(excess, ddof=1) if len(total) > 1 else 0
    return {"n_periods": len(total), "years": float(years),
            "cagr_total": float(nav[-1] ** (1 / years) - 1),
            "annualized_mean_excess": float(excess.mean() * periods_per_year),
            "annualized_vol_excess": float(std * np.sqrt(periods_per_year)),
            "sharpe_excess": float(excess.mean() / std * np.sqrt(periods_per_year)) if std > 1e-12 else None,
            "max_drawdown_total": float((nav / np.maximum.accumulate(nav) - 1).min())}


def block_sharpe_interval(excess, days, block=13, reps=500, seed=1729):
    """Circular moving-block bootstrap; descriptive, assumes stationarity."""
    x, days = np.asarray(excess, float), np.asarray(days, float)
    n = len(x)
    if n < max(30, 2 * block) or np.std(x) < 1e-12:
        return None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        starts = rng.integers(0, n, int(np.ceil(n / block)))
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        z = x[idx]
        if np.std(z, ddof=1) > 1e-12:
            samples.append(z.mean() / z.std(ddof=1) * np.sqrt(n / (days[idx].sum() / 365.25)))
    if not samples:
        return None
    return {"low": float(np.quantile(samples, .025)), "high": float(np.quantile(samples, .975)),
            "block": block, "reps": reps, "method": "circular moving-block; no multiple-testing adjustment"}
