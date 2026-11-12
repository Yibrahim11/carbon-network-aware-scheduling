"""Regenerate one row of the preliminary-results table from scratch and compare it with the stored raw output.

Row: P2 vs P0, dynamic carbon, Poisson arrivals / 50% load / 1x communication / synthetic high-variability grid
(aligned placement profile, 10 replications, seed 20261004). Needs only numpy, pandas, scipy. Runtime: a few seconds.

    python regenerate_one_row.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "code"))
import numpy as np, pandas as pd, dcsim

params = json.load(open(os.path.join(HERE, "results", "params_full.json")))
seed = params["meta"]["seed"]
prof = pd.read_csv(os.path.join(HERE, "data", "astra_profiles_aligned.csv"))
pp = dcsim.PowerParams(); pp.P_gpu_idle, pp.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
sc = dcsim.Scenario(arrival="poisson", util=0.5, comm_scale=1.0, ci="synthetic_high_var")
new = {pol: [dcsim.run_one(sc, pol, r, master_seed=seed, profile_df=prof, power_base=pp) for r in range(10)] for pol in ("P0", "P2")}

raw = pd.read_csv(os.path.join(HERE, "results", "main_full_meas3_aligned.csv"))
raw = raw[(raw.arrival == "poisson") & (raw.util == 0.5) & (raw.comm_scale == 1.0) & (raw.ci == "synthetic_high_var")]
worst = 0.0
for pol in ("P0", "P2"):
    for r in range(10):
        old = raw[(raw.policy == pol) & (raw.rep == r)].iloc[0]
        for m in ("carbon_dyn_kg", "carbon_kg", "energy_kWh", "jct_mean_h", "n_incomplete"):
            a, b = float(new[pol][r][m]), float(old[m]); worst = max(worst, abs(a - b) / max(abs(b), 1e-9))
        assert new[pol][r]["workload_hash"] == old["workload_hash"], "different job stream"
d = [100 * (new["P2"][r]["carbon_dyn_kg"] / new["P0"][r]["carbon_dyn_kg"] - 1) for r in range(10)]
print(f"regenerated: P2 vs P0 dynamic carbon = {np.mean(d):.2f}% (mean of 10 paired replications)")
print(f"largest relative difference to the stored raw output over 100 numbers: {worst:.2e}")
print("MATCH" if worst < 1e-9 else "CHECK: differences above 1e-9")
