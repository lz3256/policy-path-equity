"""Reproducible synthetic benchmark; no licensed data or timing extrapolation."""
import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'tests'))
from daily_fixture import daily_fixture
from reference_aggregation import aggregate_frames as reference
from policy_path.adapters import aggregate_frames

p = argparse.ArgumentParser()
p.add_argument('--assets', type=int, default=1000)
p.add_argument('--weeks', type=int, default=520)
p.add_argument('--out', default='benchmarks/results.json')
a = p.parse_args()
small = daily_fixture(weeks=52, assets=50)
results = {}
for label, fn in [('v01_reference', reference), ('v02_vectorized', aggregate_frames)]:
    start = time.perf_counter()
    tables, audit = fn(*small)
    results[label] = {'seconds': time.perf_counter()-start, 'daily_rows': len(small[2]), 'audit_rows': len(audit)}
del small, tables, audit
start = time.perf_counter()
large = daily_fixture(weeks=a.weeks, assets=a.assets)
creation = time.perf_counter()-start
start = time.perf_counter()
tables, audit = aggregate_frames(*large)
results['large_vectorized'] = {'assets': a.assets, 'weeks': a.weeks, 'daily_rows': len(large[2]),
                               'generation_seconds': creation, 'aggregation_seconds': time.perf_counter()-start,
                               'stock_window_rows': len(tables['stock_windows']), 'holding_rows': len(tables['asset_returns']),
                               'audit_rows': len(audit)}
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
results['peak_process_rss_mib'] = rss/(1024**2) if platform.system() == 'Darwin' else rss/1024
results['environment'] = {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine()}
results['caveat'] = 'Artificial weekday data, no CSV I/O. Entire input and output still reside in RAM. Not a WRDS download benchmark or an entire-strategy runtime.'
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
