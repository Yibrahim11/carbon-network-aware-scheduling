import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.dirname(_os.path.dirname(HERE))
import os, sys
from concurrent.futures import ProcessPoolExecutor
PK = f"{ROOT}"
sys.path.insert(0, f"{PK}/code_v0.3.0")
import pandas as pd, dcsim
PROF = {"aligned": pd.read_csv(f"{PK}/data/astra_profiles_aligned.csv"), "interleaved": pd.read_csv(f"{PK}/data/astra_profiles_interleaved.csv")}
KRULE = {"aligned": "ring_aligned", "interleaved": "ring_interleaved"}
PP = dcsim.PowerParams(); PP.P_gpu_idle, PP.P_gpu_max = 132.0, 874.0
GH = "csv:" + f"{PK}/data/ci_ghana_derived.csv"
def one(a):
    v, tag, sc, pol, rep = a
    m = dcsim.run_one(dcsim.Scenario(**sc), pol, rep, master_seed=20261004, profile_df=PROF[v], power_base=PP, cluster_base=dcsim.ClusterParams(k_rule=KRULE[v]))
    m = {k: x for k, x in m.items() if not isinstance(x, (dict, list))}
    m.update(variant=v, tag=tag, policy=pol, rep=rep, util=sc["util"], comm_scale=sc["comm_scale"], ci=os.path.basename(sc["ci"])); return m
if __name__ == "__main__":
    jobs = []
    for v in ("aligned", "interleaved"):
        for ci in ("synthetic_low_var", GH):
            for pol in ("P0", "P2", "P3", "P4"):
                for r in range(10): jobs.append((v, "grid", dict(arrival="poisson", util=0.3, comm_scale=1.0, ci=ci), pol, r))
    for c in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        for pol in ("P0", "P2", "P3", "P4"):
            for r in range(10): jobs.append(("aligned", "sweep30", dict(arrival="poisson", util=0.3, comm_scale=c, ci="synthetic_high_var"), pol, r))
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        out = pd.DataFrame(list(ex.map(one, jobs, chunksize=4)))
    out.to_csv(f"{HERE}/extra_runs.csv", index=False); print(len(out), "runs, censored", int((out.n_incomplete > 0).sum()))
