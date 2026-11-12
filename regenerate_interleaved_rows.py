"""Regenerate rows of the proposal's results table (worst-case, interleaved column) from scratch and compare them with
the stored raw output of the Colab run tagged meas5_interleaved (dcsim 0.2.0, 22 Oct 2026).

Uses code_v0.2.0/ (the code that produced that run), data/astra_profiles_interleaved.csv (the exact profile table,
exported from results/params_full_meas5_interleaved.json) and the stored power and cluster parameters, including the
rule for jobs over three or more pods. Each run is compared field by field with the stored run; the paired difference
is recomputed as in the digest (mean of per-replication relative differences, censored replications dropped).

    python regenerate_interleaved_rows.py            # all rows (about 10 seconds)
    python regenerate_interleaved_rows.py any sens_full_meas5_interleaved.csv 1234     # one stored run, by row number

Rows (base cell: Poisson arrivals, 30% load, 0.5x communication, high-variability grid):
  base_p2    P2 vs P0 dynamic carbon                           table: -21.3%
  base_p4p2  P4 vs P2 dynamic carbon                           table: -11.1%
  ghana_p2   P2 vs P0 dynamic carbon, Ghana-derived scenario   table: -0.7%
  onset_1x   mean network inflation at 1x communication, P0 / P2 / P3 / P4   table: 1.26 / 1.14 / 1.00 / 1.00
"""
import os, sys, json
from dataclasses import asdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "code_v0.2.0"))
import numpy as np, pandas as pd, dcsim

params = json.load(open(os.path.join(HERE, "results", "params_full_meas5_interleaved.json")))
SEED = params["meta"]["seed"]
PROF = pd.read_csv(os.path.join(HERE, "data", "astra_profiles_interleaved.csv"))
PP = dcsim.PowerParams(); PP.P_gpu_idle, PP.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
CB = dcsim.ClusterParams(k_rule=params["cluster"]["k_rule"])
GHANA = "csv:" + os.path.join(HERE, "data", "ci_ghana_derived.csv")
CHECKED = ("carbon_dyn_kg", "carbon_kg", "energy_kWh", "jct_mean_h", "net_inflation", "n_incomplete")
SENS = "sens_full_meas5_interleaved.csv"
BASE = dict(arrival="poisson", util=0.3, comm_scale=0.5, ci="synthetic_high_var")

ROWS = {
    "base_p2":   dict(filt=dict(tag="sens:util", util=0.3), sc=BASE, pairs=[("P2", "P0", "carbon_dyn_kg", -21.3)]),
    "base_p4p2": dict(filt=dict(tag="sens:util", util=0.3), sc=BASE, pairs=[("P4", "P2", "carbon_dyn_kg", -11.1)]),
    "ghana_p2":  dict(filt=dict(tag="sens:ci", ci="csv:/content/drive/MyDrive/msc_sim/data/ci_ghana_derived.csv"),
                      sc={**BASE, "ci": GHANA}, pairs=[("P2", "P0", "carbon_dyn_kg", -0.7)]),
    "onset_1x":  dict(filt=dict(tag="sens:comm_scale", comm_scale=1.0), sc={**BASE, "comm_scale": 1.0},
                      means=[("P0", 1.26), ("P2", 1.14), ("P3", 1.00), ("P4", 1.00)]),
}


def _sim(sc, pol, rep):
    return dcsim.run_one(dcsim.Scenario(**sc), pol, int(rep), master_seed=SEED, profile_df=PROF, power_base=PP, cluster_base=CB)


def _compare(m, old):
    assert m["workload_hash"] == old["workload_hash"], "different job stream (CRN broken)"
    return max(abs(float(m[k]) - float(old[k])) / max(abs(float(old[k])), 1e-9) for k in CHECKED)


def regenerate(name):
    row = ROWS[name]
    raw = pd.read_csv(os.path.join(HERE, "results", SENS))
    for k, v in row["filt"].items():
        raw = raw[raw[k] == v]
    pols = sorted({p for a, b, *_ in row.get("pairs", []) for p in (a, b)} | {p for p, _ in row.get("means", [])})
    reps = sorted(raw.rep.unique())
    new, worst = {}, 0.0
    for pol in pols:
        new[pol] = {}
        for r in reps:
            old = raw[(raw.policy == pol) & (raw.rep == r)]
            assert len(old) == 1, f"expected one stored run for {pol} rep {r}, found {len(old)}"
            m = _sim(row["sc"], pol, r); worst = max(worst, _compare(m, old.iloc[0])); new[pol][r] = m
    ok = worst < 1e-9
    print(f"[{name}] {len(pols)} policies x {len(reps)} replications re-simulated; largest relative difference to the "
          f"stored Colab output: {worst:.2e} -> {'MATCH' if ok else 'CHECK'}")
    for a, b, met, table in row.get("pairs", []):
        keep = [r for r in reps if new[a][r]["n_incomplete"] == 0 and new[b][r]["n_incomplete"] == 0]
        v = float(np.mean([100 * (new[a][r][met] / new[b][r][met] - 1) for r in keep]))
        same = round(v, 1) == table; ok &= same
        print(f"    {a} vs {b} {met}: {v:+.2f}% over {len(keep)} uncensored replications (table: {table:+.1f}%) -> {'SAME' if same else 'DIFFERS'}")
    for pol, table in row.get("means", []):
        v = float(np.mean([new[pol][r]["net_inflation"] for r in reps]))
        same = round(v, 2) == table; ok &= same
        print(f"    {pol} mean network inflation: {v:.3f} (table: {table:.2f}) -> {'SAME' if same else 'DIFFERS'}")
    return ok


def regenerate_stored_run(csv_name, row_number):
    old = pd.read_csv(os.path.join(HERE, "results", csv_name)).iloc[int(row_number)]
    kw = {}
    for k in asdict(dcsim.Scenario()):
        v = old[k]
        if isinstance(v, float) and np.isnan(v):
            v = None if k == "cap_frac" else ("" if k == "trace" else v)
        kw[k] = v.item() if isinstance(v, (np.floating, np.integer)) else v
    if str(kw["ci"]).startswith("csv:"):
        kw["ci"] = "csv:" + os.path.join(HERE, "data", os.path.basename(kw["ci"][4:]))
    if kw.get("trace"):
        raise SystemExit("trace-driven runs need trace.csv, which is not in the package yet")
    m = _sim(kw, old["policy"], old["rep"]); worst = _compare(m, old)
    print(f"[{csv_name} row {row_number}] {old['tag']} | {old['policy']} rep {old['rep']} | "
          + ", ".join(f"{k}={kw[k]}" for k in ("arrival", "util", "comm_scale", "ci")) + f" -> {'MATCH' if worst < 1e-9 else 'CHECK'} ({worst:.2e})")
    return worst < 1e-9


if __name__ == "__main__":
    if sys.argv[1:2] == ["any"]:
        sys.exit(0 if regenerate_stored_run(sys.argv[2], sys.argv[3]) else 1)
    results = [regenerate(n) for n in (sys.argv[1:] or list(ROWS))]
    print("ALL ROWS MATCH" if all(results) else "AT LEAST ONE ROW DIFFERS: see above")
