"""Pod-size experiment on the FINAL pipeline (dcsim 0.4.0, day-ahead persistence forecast), added after the
supervisor's review (M3) so that the decision-rule boundary A* is estimated from final runs only.

The preliminary pod-size experiment (results/podtrace_full_meas6_*) used the Alibaba trace with dcsim 0.3.0 and
perfect forecasts; the trace could not be downloaded again, so this version uses the synthetic large-job mix of the
final experiments. Same 128 GPUs and the same uplink capacity per GPU, in pods of 32, 16 and 8 GPUs.

    python final/podsize_design.py aligned        # or interleaved; checkpointed
"""
import os, sys, json, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code_v0.4.0"))
import pandas as pd
import dcsim, dcsim_exp as X

BOUND = sys.argv[1]
R = 10
params = json.load(open(os.path.join(ROOT, "results", f"params_full_meas6_{BOUND}.json")))
SEED = params["meta"]["seed"]
prof = pd.read_csv(os.path.join(ROOT, "data", f"astra_profiles_{BOUND}.csv"))
pp = dcsim.PowerParams()
pp.P_gpu_idle, pp.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
UTIL = (0.5, 0.8) if BOUND == "aligned" else (0.2, 0.3)
COMM = (1.0, 3.0) if BOUND == "aligned" else (0.5, 1.0)
POL = ["P0", "P2", "P3", "P4"]

if __name__ == "__main__":
    os.chdir(ROOT)
    for pod in (32, 16, 8):
        cb = X.podsize_cluster(pod)
        cb.k_rule, cb.comm_scale_iter = params["cluster"]["k_rule"], params["cluster"]["comm_scale_iter"]
        D = [dict(tag=f"pod{pod}", scenario=dict(arrival="poisson", util=u, comm_scale=cs, ci="synthetic_high_var",
                                               forecast="persistence"), policy=p, rep=r)
             for u, cs in itertools.product(UTIL, COMM) for p in POL for r in range(R)]
        X.run_design(D, os.path.join(ROOT, "final", f"results_{BOUND}_podsize.csv"), workers=2, profile_df=prof,
                     master_seed=SEED, cluster_base=cb, power_base=pp, batch=400)
