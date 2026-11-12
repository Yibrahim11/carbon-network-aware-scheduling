"""Regenerate the rows of the proposal's results table (both columns) from scratch and compare them with the stored
raw output of the full runs tagged meas6_aligned and meas6_interleaved (dcsim 0.3.0, 22 Oct 2026).

Uses code_v0.3.0/, the exact profile tables in data/, and the stored power and cluster parameters (including the rule
for jobs over three or more pods). Each re-simulated run is compared field by field with the stored run, and each
table value is recomputed as in the digest (mean of per-replication relative differences, censored replications dropped).

    python regenerate_meas6_rows.py                     # all rows, both columns (about 30 seconds)
    python regenerate_meas6_rows.py interleaved         # one column
    python regenerate_meas6_rows.py any sens_full_meas6_interleaved.csv 1234     # any one stored run, by row number
"""
import os, sys, json
from dataclasses import asdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "code_v0.3.0"))
import numpy as np, pandas as pd, dcsim

GHANA_STORED = "csv:/home/claude/localrun/msc_sim_local/data/ci_ghana_derived.csv"     # path recorded in the stored runs
GHANA = "csv:" + os.path.join(HERE, "data", "ci_ghana_derived.csv")
CHECKED = ("carbon_dyn_kg", "carbon_kg", "energy_kWh", "jct_mean_h", "net_inflation", "net_inflation_total", "n_incomplete")
VARIANT = {
    "aligned": dict(profile="astra_profiles_aligned.csv", base=dict(arrival="poisson", util=0.7, comm_scale=1.0, ci="synthetic_high_var"),
                    rows=[("P2 vs P0 dynamic carbon, high-variability grid", "sens:util", {}, [("P2", "P0", "carbon_dyn_kg", -14.4)], None),
                          ("P2 vs P0 dynamic carbon, Ghana-derived scenario", "sens:ci", {"ci": GHANA}, [("P2", "P0", "carbon_dyn_kg", -0.9)], None),
                          ("P4 vs P2 dynamic carbon", "sens:util", {}, [("P4", "P2", "carbon_dyn_kg", -0.1)], None),
                          ("inflation at 1x communication", "sens:comm_scale", {"comm_scale": 1.0}, [],
                           {"net_inflation": (1.00, 1.00, 1.00, 1.00), "net_inflation_total": (1.03, 1.03, 1.03, 1.03)})]),
    "interleaved": dict(profile="astra_profiles_interleaved.csv", base=dict(arrival="poisson", util=0.3, comm_scale=0.5, ci="synthetic_high_var"),
                        rows=[("P2 vs P0 dynamic carbon, high-variability grid", "sens:util", {}, [("P2", "P0", "carbon_dyn_kg", -23.5)], None),
                              ("P2 vs P0 dynamic carbon, Ghana-derived scenario", "sens:ci", {"ci": GHANA}, [("P2", "P0", "carbon_dyn_kg", -0.8)], None),
                              ("P4 vs P2 dynamic carbon", "sens:util", {}, [("P4", "P2", "carbon_dyn_kg", -9.1)], None),
                              ("inflation at 1x communication", "sens:comm_scale", {"comm_scale": 1.0}, [],
                               {"net_inflation": (1.26, 1.14, 1.00, 1.00), "net_inflation_total": (2.88, 2.32, 1.31, 1.39)})]),
}


def setup(variant):
    params = json.load(open(os.path.join(HERE, "results", f"params_full_meas6_{variant}.json")))
    prof = pd.read_csv(os.path.join(HERE, "data", VARIANT[variant]["profile"]))
    pp = dcsim.PowerParams(); pp.P_gpu_idle, pp.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
    cb = dcsim.ClusterParams(k_rule=params["cluster"]["k_rule"], comm_scale_iter=params["cluster"]["comm_scale_iter"])
    return params["meta"]["seed"], prof, pp, cb


def compare(m, old):
    assert m["workload_hash"] == old["workload_hash"], "different job stream (CRN broken)"
    return max(abs(float(m[k]) - float(old[k])) / max(abs(float(old[k])), 1e-9) for k in CHECKED)


def regenerate(variant):
    seed, prof, pp, cb = setup(variant)
    sens = pd.read_csv(os.path.join(HERE, "results", f"sens_full_meas6_{variant}.csv"))
    ok_all = True
    print(f"== {variant} column")
    for label, tag, change, pairs, means in VARIANT[variant]["rows"]:
        sc = {**VARIANT[variant]["base"], **change}
        raw = sens[sens.tag == tag]
        for k, v in sc.items():
            raw = raw[raw[k] == (GHANA_STORED if v == GHANA else v)]
        pols = sorted({p for a, b, *_ in pairs for p in (a, b)} | ({"P0", "P2", "P3", "P4"} if means else set()))
        reps = sorted(raw.rep.unique()); new, worst = {}, 0.0
        for pol in pols:
            new[pol] = {}
            for r in reps:
                old = raw[(raw.policy == pol) & (raw.rep == r)]
                assert len(old) == 1, f"{label}: expected one stored run for {pol} rep {r}, found {len(old)}"
                m = dcsim.run_one(dcsim.Scenario(**sc), pol, int(r), master_seed=seed, profile_df=prof, power_base=pp, cluster_base=cb)
                worst = max(worst, compare(m, old.iloc[0])); new[pol][r] = m
        ok = worst < 1e-9
        res = []
        for a, b, met, table in pairs:
            keep = [r for r in reps if new[a][r]["n_incomplete"] == 0 and new[b][r]["n_incomplete"] == 0]
            v = float(np.mean([100 * (new[a][r][met] / new[b][r][met] - 1) for r in keep]))
            ok &= round(v, 1) == table; res.append(f"{a} vs {b} {v:+.2f}% (table {table:+.1f}%)")
        for met, tab in (means or {}).items():
            vals = [float(np.mean([new[p][r][met] for r in reps])) for p in ("P0", "P2", "P3", "P4")]
            ok &= all(round(v, 2) == t for v, t in zip(vals, tab))
            res.append(f"{met} P0/P2/P3/P4 " + " / ".join(f"{v:.3f}" for v in vals) + " (table " + " / ".join(f"{t:.2f}" for t in tab) + ")")
        print(f"  {label}: {len(pols)} policies x {len(reps)} reps, largest difference to stored {worst:.2e} | "
              + "; ".join(res) + f" -> {'MATCH' if ok else 'CHECK'}")
        ok_all &= ok
    return ok_all


def regenerate_stored_run(csv_name, row_number):
    variant = "interleaved" if "interleaved" in csv_name else "aligned"
    seed, prof, pp, cb = setup(variant)
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
    m = dcsim.run_one(dcsim.Scenario(**kw), old["policy"], int(old["rep"]), master_seed=seed, profile_df=prof, power_base=pp, cluster_base=cb)
    worst = compare(m, old)
    print(f"[{csv_name} row {row_number}] {old['tag']} | {old['policy']} rep {old['rep']} | "
          + ", ".join(f"{k}={kw[k]}" for k in ("arrival", "util", "comm_scale")) + f", ci={os.path.basename(str(kw['ci']))}"
          + f" -> {'MATCH' if worst < 1e-9 else 'CHECK'} ({worst:.2e})")
    return worst < 1e-9


if __name__ == "__main__":
    if sys.argv[1:2] == ["any"]:
        sys.exit(0 if regenerate_stored_run(sys.argv[2], sys.argv[3]) else 1)
    results = [regenerate(v) for v in (sys.argv[1:] or ["aligned", "interleaved"])]
    print("ALL ROWS MATCH" if all(results) else "AT LEAST ONE ROW DIFFERS: see above")
