"""Analyses requested in the supervisor's review of 3 Nov 2026, from the stored meas6 runs plus new runs at matched
operating points. Uses dcsim 0.3.0 (code_v0.3.0) and the packaged profile tables."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.dirname(_os.path.dirname(HERE))
import os, sys, json
from concurrent.futures import ProcessPoolExecutor
PK = f"{ROOT}"
sys.path.insert(0, f"{PK}/code_v0.3.0")
import numpy as np, pandas as pd, dcsim, dcsim_exp as X
OUT = HERE
PROF = {"aligned": pd.read_csv(f"{PK}/data/astra_profiles_aligned.csv"), "interleaved": pd.read_csv(f"{PK}/data/astra_profiles_interleaved.csv")}
KRULE = {"aligned": "ring_aligned", "interleaved": "ring_interleaved"}
PP = dcsim.PowerParams(); PP.P_gpu_idle, PP.P_gpu_max = 132.0, 874.0
SEED = 20261004
BASE = {"aligned": dict(util=0.7, comm_scale=1.0), "interleaved": dict(util=0.3, comm_scale=0.5)}

def load(v):
    return pd.read_csv(f"{PK}/results/main_full_meas6_{v}.csv"), pd.read_csv(f"{PK}/results/sens_full_meas6_{v}.csv")

def one(args):
    v, sc, pol, rep, trace = args
    cb = dcsim.ClusterParams(k_rule=KRULE[v])
    r = dcsim.run_one(dcsim.Scenario(**sc), pol, rep, master_seed=SEED, profile_df=PROF[v], power_base=PP, cluster_base=cb, record_trace=trace)
    m, sim = (r if trace else (r, None))
    m = {k: val for k, val in m.items() if not isinstance(val, (dict, list))}
    m.update(variant=v, policy=pol, rep=rep, **{f"sc_{k}": val for k, val in sc.items()})
    if trace:
        tr = np.array(sim.trace)                      # (t0, t1, P_total_W, ...); P_total includes PUE? check below
        m["_trace"] = tr[:, :3].tolist()
    return m

if __name__ == "__main__":
    res = {}
    # ---------------- A. single cells: base cells and matched operating points (new runs) ----------------
    jobs = []
    cells = []
    for v in ("aligned", "interleaved"):
        for util, comm in ((0.7, 1.0), (0.5, 1.0), (0.3, 0.5), (0.3, 1.0)):
            cells.append((v, util, comm))
    for v, util, comm in cells:
        sc = dict(arrival="poisson", util=util, comm_scale=comm, ci="synthetic_high_var")
        for pol in ("P0", "P2", "P3", "P4"):
            for r in range(10):
                jobs.append((v, sc, pol, r, True))
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        runs = list(ex.map(one, jobs, chunksize=4))
    traces = {(m["variant"], m["sc_util"], m["sc_comm_scale"], m["policy"], m["rep"]): m.pop("_trace") for m in runs}
    df = pd.DataFrame(runs)
    df["tag"] = "review"; df["util"] = df.sc_util; df["comm_scale"] = df.sc_comm_scale
    for c in ("arrival", "ci", "fabric", "forecast_sigma", "slack_lo", "slack_hi", "deferrable_frac", "w_delay", "tick_s", "cap_frac", "trace"):
        if c not in df: df[c] = None
    df.to_csv(f"{OUT}/matched_runs.csv", index=False)
    json.dump({"|".join(map(str, k)): v for k, v in traces.items()}, open(f"{OUT}/traces.json", "w"))
    print("new runs:", len(df), "| censored:", int((df.n_incomplete > 0).sum()))
