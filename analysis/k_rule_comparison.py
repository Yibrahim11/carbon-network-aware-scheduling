"""How much does the rule for jobs over k>2 pods matter? Same job streams (CRN), rule switched: 'hier' (v0.1.0) vs flat ring.
INDICATIVE: interleaved cases use the 3-decimal reconstructed profile; the trace cases use a STAND-IN trace, not Alibaba.
Source of the corrected values quoted in the proposal's table note (-21.2%, -11.1%, P0 inflation 1.27).
    python analysis/k_rule_comparison.py      (about 3 minutes on 2 CPUs; writes k_rule_comparison_runs.csv here)
"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code_v0.2.0")); os.chdir(HERE)
from concurrent.futures import ProcessPoolExecutor
import dcsim, dcsim_exp
from dcsim import Scenario, ClusterParams, PowerParams
PP = PowerParams(); PP.P_gpu_idle, PP.P_gpu_max = 132.0, 874.0
PROF = {"aligned": pd.read_csv(os.path.join(ROOT, "data", "astra_profiles_aligned.csv")),
        "interleaved_3dp": pd.read_csv(os.path.join(ROOT, "data", "astra_profiles_interleaved_reconstructed_3dp.csv"))}
RULE = {"aligned": "ring_aligned", "interleaved_3dp": "ring_interleaved"}

def stand_in_trace(path):
    rng = np.random.default_rng(5); n = 20000      # STAND-IN trace: Alibaba-like sizes, NOT the real trace
    pd.DataFrame(dict(arrival_s=np.cumsum(rng.exponential(60, n)),
                      n_gpus=rng.choice([4, 8, 16], n, p=[.35, .45, .20]),
                      duration_s=np.exp(rng.normal(np.log(3600), 1.0, n)))).to_csv(path, index=False)

def one(a):
    case, var, rule, pol, rep, sc, pod = a
    cb = dcsim_exp.podsize_cluster(pod) if pod else ClusterParams()
    cb.k_rule = rule
    m = dcsim.run_one(Scenario(**sc), pol, rep, profile_df=PROF[var], cluster_base=cb, power_base=PP)
    return dict(case=case, var=var, rule=rule, policy=pol, rep=rep, pod=pod or 32, **{k: m[k] for k in
                ("carbon_dyn_kg", "jct_mean_h", "net_inflation", "spread_frac", "gpuh_frac_k3plus", "k_max",
                 "n_incomplete", "peak_kW", "workload_hash")})

if __name__ == "__main__":
    stand_in_trace("standin_trace.csv")
    jobs = []
    R = 10
    cases = [("synthetic mix, base cell", "aligned", dict(arrival="poisson", util=0.7, comm_scale=1.0, ci="synthetic_high_var"), None),
             ("synthetic mix, base cell", "interleaved_3dp", dict(arrival="poisson", util=0.3, comm_scale=0.5, ci="synthetic_high_var"), None),
             ("synthetic mix, 1x comm", "interleaved_3dp", dict(arrival="poisson", util=0.3, comm_scale=1.0, ci="synthetic_high_var"), None)]
    for pod in (32, 16, 8):
        for var, u, c in (("aligned", 0.7, 1.0), ("interleaved_3dp", 0.3, 1.0)):
            cases.append((f"STAND-IN trace, pod {pod}", var, dict(trace="standin_trace.csv", util=u, comm_scale=c, ci="synthetic_high_var"), pod))
    for case, var, sc, pod in cases:
        for rule in ("hier", RULE[var]):
            for pol in ("P0", "P2", "P3", "P4"):
                for r in range(R):
                    jobs.append((case, var, rule, pol, r, sc, pod))
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        df = pd.DataFrame(list(ex.map(one, jobs, chunksize=4)))
    df.to_csv("k_rule_comparison_runs.csv", index=False)
    print(len(df), "runs")
