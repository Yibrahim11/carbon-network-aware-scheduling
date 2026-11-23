"""Oracle bound and simulated P2 saving for the EC-2025-anchored Ghana series (data_build/ghana_ci_ec.py),
at the common operating point. Writes ghana_ec_check.csv."""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "code_v0.3.0"))
import numpy as np, pandas as pd, dcsim

def setup(variant):
    params = json.load(open(os.path.join(ROOT, "results", f"params_full_meas6_{variant}.json")))
    prof = pd.read_csv(os.path.join(ROOT, "data", f"astra_profiles_{variant}.csv"))
    pp = dcsim.PowerParams(); pp.P_gpu_idle, pp.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
    cb = dcsim.ClusterParams(k_rule=params["cluster"]["k_rule"], comm_scale_iter=params["cluster"]["comm_scale_iter"])
    return params["meta"]["seed"], prof, pp, cb

def oracle_saving(jobs, ci, threshold=0.0, step_s=900.0):
    C = dcsim.CI(ci); now = best = 0.0
    for j in jobs:
        d = j.dur_ref_s; c0 = C.integral(j.arrival, j.arrival + d); cb = c0
        if j.deferrable:
            for s in np.arange(j.arrival, max(j.arrival, j.deadline - d) + 1e-9, step_s):
                cb = min(cb, C.integral(s, s + d))
            if (c0 - cb) / c0 < threshold:
                cb = c0
        now += j.n_gpus * c0; best += j.n_gpus * cb
    return 1 - best / now

rows = []
files = sorted(glob.glob(os.path.join(ROOT, "data_build", "ci_ghana_ec20*.csv"))) + [os.path.join(ROOT, "data", "ci_ghana_derived.csv")]
for f in files:
    key = "csv:" + f; name = os.path.basename(f)[:-4]
    for v in ("aligned", "interleaved"):
        seed, prof, pp, cb = setup(v)
        sc = dcsim.Scenario(arrival="poisson", util=0.3, comm_scale=1.0, ci=key)
        for rep in range(10):
            jobs, _, cp, _, ci, _, _ = dcsim.build_world(sc, rep, seed, prof, cb, pp)
            m = {p: dcsim.run_one(sc, p, rep, master_seed=seed, profile_df=prof, power_base=pp, cluster_base=cb) for p in ("P0", "P2")}
            ok = m["P0"]["n_incomplete"] == 0 and m["P2"]["n_incomplete"] == 0
            rows.append(dict(series=name, bound=v, rep=rep, V=dcsim.ci_variability(ci[:int(cp.window_h)])["daily_range_over_mean"],
                             oracle=100 * oracle_saving(jobs, ci), oracle_thr=100 * oracle_saving(jobs, ci, 0.05),
                             p2=-100 * (m["P2"]["carbon_dyn_kg"] / m["P0"]["carbon_dyn_kg"] - 1) if ok else np.nan))
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "ghana_ec_check.csv"), index=False)
print(R.groupby(["series", "bound"])[["V", "oracle", "oracle_thr", "p2"]].mean().round(2).to_string())
