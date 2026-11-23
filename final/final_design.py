"""Final experiments (thesis Chapter IV), dcsim 0.4.0. Run from the package root:

    python final/final_design.py aligned       # or interleaved;  checkpointed, re-run to resume
    python final/final_design.py interleaved

Writes final/results_<bound>.csv (one row per run). Headline forecast: day-ahead persistence.
Synthetic inputs (labelled as such in the thesis): carbon series 'synthetic_low_var', 'synthetic_high_var', the 'vfam:V'
family, and the Ghana series anchored to national statistics (data_build/ghana_ci_ec.py). No measured hourly series is used.
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
cb = dcsim.ClusterParams(k_rule=params["cluster"]["k_rule"], comm_scale_iter=params["cluster"]["comm_scale_iter"])
def power(gamma=1.0):
    pp = dcsim.PowerParams(gamma=gamma)
    pp.P_gpu_idle, pp.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
    return pp

GH = "csv:data_build/ci_ghana_ec2025_thermal_flat_a0.081.csv"          # headline Ghana series
GH_ALL = sorted(f"csv:data_build/{f}" for f in os.listdir(os.path.join(ROOT, "data_build")) if f.startswith("ci_ghana_ec20"))
GH_PRELIM = "csv:data/ci_ghana_derived.csv"
POL = ["P0", "P2", "P3", "P4"]; POL_S = ["P0", "P2", "P2raw", "P3", "P4", "P4nc"]
BASE = dict(arrival="poisson", util=0.3, comm_scale=1.0, ci="synthetic_high_var", forecast="persistence")
UTIL_MAIN = (0.5, 0.8) if BOUND == "aligned" else (0.2, 0.3)
COMM_MAIN = (1.0, 3.0) if BOUND == "aligned" else (0.5, 1.0)
UTIL_SWEEP = [0.2, 0.3, 0.5, 0.7, 0.8] if BOUND == "aligned" else [0.1, 0.2, 0.3, 0.4, 0.5]
VFAM = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80, 1.10]

def rows(tag, sc, pols):
    return [dict(tag=tag, scenario=dict(sc), policy=p, rep=r) for p in pols for r in range(R)]

D = []
D += rows("primary", BASE, POL)
for ar, u, cs, ci in itertools.product(("poisson", "bursty"), UTIL_MAIN, COMM_MAIN, ("synthetic_low_var", "synthetic_high_var", GH)):
    D += rows("main", dict(arrival=ar, util=u, comm_scale=cs, ci=ci, forecast="persistence"), POL)
for V, fc in itertools.product(VFAM, ("persistence", "perfect")):
    D += rows("vsweep", {**BASE, "ci": f"vfam:{V}", "forecast": fc}, POL)
for g in GH_ALL + [GH_PRELIM]:
    D += rows("ghana", {**BASE, "ci": g}, POL)
sweeps = {
    "comm_scale": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0], "util": UTIL_SWEEP, "cap_frac": [None, 0.8, 0.7, 0.62],
    "forecast": [("perfect", 0.0), ("persistence", 0.0), ("perfect", 0.1), ("perfect", 0.3)],
    "fabric": ["ocs", "electrical"], "slack": [(1.0, 4.0), (2.0, 12.0), (4.0, 24.0)], "w_delay": [0.0, 0.02, 0.1],
    "deferrable_frac": [0.3, 0.7, 1.0], "arrival": ["poisson", "bursty", "diurnal"]}
for f, levels in sweeps.items():
    for lv in levels:
        sc = dict(BASE)
        if f == "slack": sc["slack_lo"], sc["slack_hi"] = lv
        elif f == "forecast": sc["forecast"], sc["forecast_sigma"] = lv
        else: sc[f] = lv
        D += rows(f"sens:{f}", sc, POL_S)
# de-duplicate identical (scenario, policy, rep) inside the same tag set is fine; across tags they are re-run (cheap)
if __name__ == "__main__":
    os.chdir(ROOT)
    out = os.path.join(ROOT, "final", f"results_{BOUND}.csv")
    X.run_design(D, out, workers=1, profile_df=prof, master_seed=SEED, cluster_base=cb, power_base=power(1.0), batch=400)
    # curved power model (gamma fitted to the T4 curve), common point and communication sweep
    G = []
    for cs in (0.5, 1.0, 2.0):
        G += rows("gamma", {**BASE, "comm_scale": cs}, POL)
    X.run_design(G, os.path.join(ROOT, "final", f"results_{BOUND}_gamma058.csv"), workers=1, profile_df=prof, master_seed=SEED,
                 cluster_base=cb, power_base=power(0.58), batch=400)
