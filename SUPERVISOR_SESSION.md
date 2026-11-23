# Regenerating results together (review item 3.3)

Plan for one sitting with Dr. Owusu Agyemang, about an hour on any laptop with Python 3, numpy, pandas and scipy.
Nothing is downloaded: every input the aligned results need is in this package.

## 1. Check the package is what it claims to be (2 minutes)
    python -c "import json,hashlib; m=json.load(open('MANIFEST.json')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(p,'rb').read()).hexdigest()!=h]; print('all files match the manifest' if not bad else bad)"

## 2. Run the verification suite (about 2 minutes)
    python -c "import sys; sys.path.insert(0,'code_v0.3.0'); import dcsim_verify as v; v.print_report(v.run_all(quick=True))"
Quick mode runs 15 of the 17 checks; `quick=False` adds the M/M/c (Erlang-C) and tick-halving checks (several minutes).

## 3. Regenerate the current results table (about 30 seconds)
    python regenerate_meas6_rows.py
Re-simulates every row of the results table in the current proposal (both placement bounds, dcsim 0.3.0) and checks every
re-simulated run against the stored output. Expected: `ALL ROWS MATCH`. Section 3a does the same for the older aligned run.

## 3a. Regenerate the rows of the earlier aligned run (about 10 seconds)
    python regenerate_table_rows.py
Re-simulates the base-cell rows of the results table (aligned column) and checks every re-simulated run against the
stored raw output. Expected: `ALL ROWS MATCH`.

| Row | Table value | What it shows |
|---|---|---|
| P2 vs P0 dynamic carbon, high-variability grid | -14.4% | carbon-aware shifting works where the grid varies |
| P2 vs P0, low-variability grid / Ghana-derived scenario | -0.7% / -0.9% | the same code saves almost nothing on a flat grid |
| P4 vs P2 dynamic carbon / mean completion time | -0.1% / -23.9% | with aligned placement the joint policy adds no carbon saving, only completion time |

## 4. The supervisor picks a run (5 minutes)
Choose any row number of any results file (960 main runs, 2,100 sensitivity runs per file); it is re-simulated
from scratch and compared field by field:

    python regenerate_meas6_rows.py any sens_full_meas6_interleaved.csv <row 0-2099>
    python regenerate_meas6_rows.py any main_full_meas6_aligned.csv <row 0-959>
    python regenerate_table_rows.py any main_full_meas3_aligned.csv <row 0-959>      (earlier aligned run)

Twelve rows of the earlier aligned run (21 Oct 2026) and twelve of the current runs, three from each file (22 Oct 2026), drawn at random, matched to about 1e-16.

## 5. Change an input and watch the result move (10 minutes)
In the `base_p2` entry of `regenerate_table_rows.py`, change both `util=0.7` to `util=0.4` and the expected `-14.4`
to `-25.6`, then run `python regenerate_table_rows.py base_p2`. The script now finds the stored 40%-load runs of the
sensitivity sweep, re-simulates them, and reports `MATCH` and a saving of -25.6% (checked 21 Oct 2026): the saving
falls as the cluster gets busier, and the stored numbers follow the inputs rather than being typed in.

## What exists and what does not (the review's second question)
| Exists and is shown above | Not yet done |
|---|---|
| Simulator, 17 verification checks, full runs for both placement bounds (2 x 3,060 runs) with raw outputs | Measured three- and four-pod ns-3 profiles (the runs use a flat-ring extension of the two-pod profiles) |
| ns-3 profiles for 8-64 GPUs, both placements (table in `data/`) | Raw ns-3 measurements (`raw_ns3_runs.csv`) and the ASTRA-sim / ns-3 commit hashes in this package |
| Alibaba-trace runs and the pod-size experiment on the real trace (Colab, `meas5`) | `trace.csv` in this package (rebuild cell included); total network inflation for the pod-size experiment |
| Ghana-derived scenario, two synthetic grids | Real hourly carbon series for further grids; imperfect forecasts as the headline case |
| | Measured compute times, ramp-rate metric, fitted decision rule, thesis chapters |
