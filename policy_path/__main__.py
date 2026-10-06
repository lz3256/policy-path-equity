import argparse
import json
from pathlib import Path
from .data import load_dataset, validate_config
from .demo import make_demo
from .report import run_research
from .adapters import aggregate_csv, make_schedule, normalize_quotes


def main():
    parser = argparse.ArgumentParser(description="Policy-path equity research; synthetic demo is not empirical evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate synthetic data and run the full pipeline")
    demo.add_argument("--out", default="demo")
    demo.add_argument("--seed", type=int, default=1729)
    for name in ("validate", "run"):
        p = commands.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--config", required=True)
        if name == "run":
            p.add_argument("--out", required=True)
    daily = commands.add_parser("aggregate-daily", help="Aggregate normalized daily CSVs to study windows")
    daily.add_argument("--source", required=True)
    daily.add_argument("--schedule", required=True)
    daily.add_argument("--out", required=True)
    schedule = commands.add_parser("make-schedule", help="Build schedule from a supplied official session calendar")
    schedule.add_argument("--calendar", required=True)
    schedule.add_argument("--first-wednesday", required=True)
    schedule.add_argument("--last-wednesday", required=True)
    schedule.add_argument("--out", required=True)
    quotes = commands.add_parser("normalize-quotes", help="Normalize a fixed-contract wide quote export")
    quotes.add_argument("--prices", required=True)
    quotes.add_argument("--contracts", required=True)
    quotes.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        root = Path(args.out)
        config = make_demo(root / "data", seed=args.seed)
        result = run_research(root / "data", config, root / "results")
        print("SYNTHETIC DATA ONLY — intentionally planted relation; no real-market evidence.")
        print(json.dumps({"report": str((root / 'results' / 'report.html').resolve()),
                          "periods": result['strategy']['n_periods'], "active_fraction": result['active_fraction']}, indent=2))
    elif args.command == "aggregate-daily":
        print(json.dumps(aggregate_csv(args.source, args.schedule, args.out), indent=2))
    elif args.command == "make-schedule":
        print(json.dumps(make_schedule(args.calendar, args.first_wednesday, args.last_wednesday, args.out), indent=2))
    elif args.command == "normalize-quotes":
        print(json.dumps(normalize_quotes(args.prices, args.contracts, args.out), indent=2))
    else:
        config = validate_config(json.loads(Path(args.config).read_text()))
        if args.command == "validate":
            data = load_dataset(args.data, config)
            print(json.dumps({"mode": data.metadata['dataset_mode'], "rows": {k: len(v) for k, v in data.tables.items()}}, indent=2))
        else:
            summary = run_research(args.data, config, args.out)
            print(json.dumps({"dataset_mode": summary['dataset_mode'], "report": str((Path(args.out) / 'report.html').resolve())}, indent=2))


if __name__ == "__main__":
    main()
