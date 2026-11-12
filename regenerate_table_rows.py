"""Regenerate rows of the proposal's results table (aligned column) from scratch and compare with the stored raw output.

Each row is re-simulated with the packaged code, the packaged aligned profile and the stored power parameters, then
compared run by run with the raw per-run file it came from, and the paired difference is recomputed the same way the
digest computes it (mean of per-replication relative differences; replications with unfinished jobs are dropped).

    python regenerate_table_rows.py            # all rows (about 10 seconds on a laptop)
    python regenerate_table_rows.py base_p2    # one row

Rows:
  base_p2    P2 vs P0 dynamic carbon, base cell (Poisson, 70% load, 1x communication, high-variability grid)   table: -14.4%
  base_p4p2  P4 vs P2 dynamic carbon and mean completion time, same cell                                      table: -0.1% / -23.9%
  lowvar_p2  P2 vs P0 dynamic carbon, same cell on the synthetic low-variability grid                        table: -0.7%
  ghana_p2   P2 vs P0 dynamic carbon, same cell on the Ghana-derived scenario                                table: -0.9%
  main_u05   P2 vs P0 dynamic carbon, main factorial cell Poisson / 50% load / 1x / high-variability grid    (-20.0%)
Any single stored run the reviewer picks (aligned main or sensitivity file, by row number):
    python regenerate_table_rows.py any sens_full_meas3_aligned.csv 1234
Needs only numpy, pandas and scipy.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "code"))
import numpy as np, pandas as pd, dcsim

params = json.load(open(os.path.join(HERE, "results", "params_full.json")))
SEED = params["meta"]["seed"]
PROF = pd.read_csv(os.path.join(HERE, "data", "astra_profiles_aligned.csv"))
PP = dcsim.PowerParams(); PP.P_gpu_idle, PP.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
CHECKED = ("carbon_dyn_kg", "carbon_kg", "energy_kWh", "jct_mean_h", "n_incomplete")

ROWS = {
    "base_p2":   dict(file="sens_full_meas3_aligned.csv", filt=dict(tag="sens:util", util=0.7),
                      sc=dict(arrival="poisson", util=0.7, comm_scale=1.0, ci="synthetic_high_var"),
                      pairs=[("P2", "P0", "carbon_dyn_kg", -14.4)]),
    "base_p4p2": dict(file="sens_full_meas3_aligned.csv", filt=dict(tag="sens:util", util=0.7),
                      sc=dict(arrival="poisson", util=0.7, comm_scale=1.0, ci="synthetic_high_var"),
                      pairs=[("P4", "P2", "carbon_dyn_kg", -0.1), ("P4", "P2", "jct_mean_h", -23.9)]),
    "lowvar_p2": dict(file="sens_full_meas3_aligned.csv", filt=dict(tag="sens:ci", ci="synthetic_low_var"),
                      sc=dict(arrival="poisson", util=0.7, comm_scale=1.0, ci="synthetic_low_var"),
                      pairs=[("P2", "P0", "carbon_dyn_kg", -0.7)]),
    "ghana_p2":  dict(file="sens_full_meas3_aligned.csv",
                      filt=dict(tag="sens:ci", ci="csv:/content/drive/MyDrive/msc_sim/data/ci_ghana_derived.csv"),
                      sc=dict(arrival="poisson", util=0.7, comm_scale=1.0,
                              ci="csv:" + os.path.join(HERE, "data", "ci_ghana_derived.csv")),
                      pairs=[("P2", "P0", "carbon_dyn_kg", -0.9)]),
    "main_u05":  dict(file="main_full_meas3_aligned.csv",
                      filt=dict(arrival="poisson", util=0.5, comm_scale=1.0, ci="synthetic_high_var"),
                      sc=dict(arrival="poisson", util=0.5, comm_scale=1.0, ci="synthetic_high_var"),
                      pairs=[("P2", "P0", "carbon_dyn_kg", -20.0)]),
}


def regenerate(name):
    row = ROWS[name]
    raw = pd.read_csv(os.path.join(HERE, "results", row["file"]))
    for k, v in row["filt"].items():
        raw = raw[raw[k] == v]
    sc = dcsim.Scenario(**row["sc"])
    pols = sorted({p for a, b, *_ in row["pairs"] for p in (a, b)})
    reps = sorted(raw.rep.unique())
    new, worst, n_cmp = {}, 0.0, 0
    for pol in pols:
        new[pol] = {}
        for r in reps:
            m = dcsim.run_one(sc, pol, int(r), master_seed=SEED, profile_df=PROF, power_base=PP)
            old = raw[(raw.policy == pol) & (raw.rep == r)]
            assert len(old) == 1, f"expected one stored run for {pol} rep {r}, found {len(old)}"
            old = old.iloc[0]
            assert m["workload_hash"] == old["workload_hash"], "different job stream (CRN broken)"
            for k in CHECKED:
                a, b = float(m[k]), float(old[k]); worst = max(worst, abs(a - b) / max(abs(b), 1e-9)); n_cmp += 1
            new[pol][r] = m
    print(f"[{name}] {len(pols)} policies x {len(reps)} replications re-simulated; "
          f"largest relative difference to stored raw output over {n_cmp} numbers: {worst:.2e} "
          f"-> {'MATCH' if worst < 1e-9 else 'CHECK'}")
    ok = worst < 1e-9
    for a, b, met, table in row["pairs"]:
        keep = [r for r in reps if new[a][r]["n_incomplete"] == 0 and new[b][r]["n_incomplete"] == 0]
        d = [100 * (new[a][r][met] / new[b][r][met] - 1) for r in keep]
        val = round(float(np.mean(d)), 1)
        ok &= val == table
        print(f"    {a} vs {b} {met}: {np.mean(d):+.2f}% over {len(keep)} uncensored replications "
              f"(proposal table: {table:+.1f}%) -> {'SAME' if val == table else 'DIFFERS'}")
    return ok


def regenerate_stored_run(csv_name, row_number):
    """Re-simulate ONE stored run chosen by the reviewer (any row of main_/sens_ aligned files) and compare it."""
    from dataclasses import asdict
    raw = pd.read_csv(os.path.join(HERE, "results", csv_name))
    old = raw.iloc[int(row_number)]
    kw = {}
    for k in asdict(dcsim.Scenario()):
        v = old[k]
        if isinstance(v, float) and np.isnan(v):
            v = None if k == "cap_frac" else ("" if k == "trace" else v)
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        kw[k] = v
    if str(kw["ci"]).startswith("csv:"):           # stored as a Colab Drive path; the same file ships in data/
        kw["ci"] = "csv:" + os.path.join(HERE, "data", os.path.basename(kw["ci"][4:]))
    if kw.get("trace"):
        raise SystemExit("trace-driven runs need trace.csv, which is not in the package yet")
    m = dcsim.run_one(dcsim.Scenario(**kw), old["policy"], int(old["rep"]), master_seed=SEED, profile_df=PROF, power_base=PP)
    worst = max(abs(float(m[k]) - float(old[k])) / max(abs(float(old[k])), 1e-9) for k in CHECKED)
    print(f"[{csv_name} row {row_number}] {old['tag']} | policy {old['policy']} | rep {old['rep']} | "
          + ", ".join(f"{k}={kw[k]}" for k in ("arrival", "util", "comm_scale", "ci", "cap_frac", "fabric")))
    for k in CHECKED:
        print(f"    {k:15s} stored {float(old[k]):14.6f}   regenerated {float(m[k]):14.6f}")
    print(f"    same job stream: {m['workload_hash'] == old['workload_hash']} | largest relative difference {worst:.2e} -> "
          f"{'MATCH' if worst < 1e-9 and m['workload_hash'] == old['workload_hash'] else 'CHECK'}")
    return worst < 1e-9


if __name__ == "__main__":
    if sys.argv[1:2] == ["any"]:         # python regenerate_table_rows.py any sens_full_meas3_aligned.csv 1234
        sys.exit(0 if regenerate_stored_run(sys.argv[2], sys.argv[3]) else 1)
    names = sys.argv[1:] or list(ROWS)
    results = [regenerate(n) for n in names]
    print("ALL ROWS MATCH" if all(results) else "AT LEAST ONE ROW DIFFERS: see above")
