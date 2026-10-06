# Policy Path and Equity Returns

A research prototype that translates policy-path repricing into equity sensitivity estimates, recursive forecasts, and portfolio targets. The implementation makes information timing and subsequent holding-period returns explicit.

## What is implemented

- Fixed-delivery-month futures signals and controls for maturity drift.
- Historical equity sensitivity estimates and industry/beta constraints.
- Recursive forecasts at 1, 2, 4, and 8-week horizons using matured labels.
- Binary and smoothed target comparisons, no-trade bands, drift-aware turnover, and trading-cost scenarios.
- CSV adapters, source fingerprints, coverage diagnostics, and reproducible synthetic examples.

## Run the demonstration

Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m policy_path demo --out demo
python -m policy_path run --data demo/data --config config/demo_execution.json --out demo/results_v02
python -m unittest discover -s tests -v
```

Open `demo/results_v02/report.html`. The two demonstration runs compare execution specifications on the same synthetic input. They do not require Bloomberg, WRDS, or a broker account.

## Recorded checks and findings

The original and revised execution demonstrations completed end to end, with source/input hashes recorded. Forty original checks covered calendar alignment, future-information rejection, label maturity, accounting, missing data, and execution variants.

The synthetic example deliberately contains a delayed relationship, so its forecast gains, Sharpe ratios, and returns are demonstration outputs rather than market findings. Real-market empirical validation remains unfinished. The current result is an implemented, auditable research workflow, not a confirmed alpha signal.

## Bringing your own data

The pipeline accepts local CSV inputs for the decision calendar, fixed-month futures quotes, historical return windows, stock eligibility snapshots, and subsequent asset/market returns. It validates structure and timing conventions, but metadata labels cannot independently prove historical availability. See [data contract and assumptions](docs/METHOD.md).

Real data, processed panels, and empirical output files are omitted. The public prototype does not contain an automatic vendor downloader or order execution.

## Public package verification

The prepared public copy passed 40 existing automated checks and its credential-free synthetic demonstration. No real-data download or empirical rerun was performed for this verification. Runtime details are in [VALIDATION.json](VALIDATION.json).

## Credits

Project owner: Linfeng Zhao (@lz3256). Developed with substantial OpenAI Codex assistance; see [credits](CREDITS.md).
