"""Analytical results of Chapter III (Section 3.10) and their numerical check against dcsim 0.3.0.

  R1  ring all-reduce traffic per pod uplink: aligned 2(n-1)/n G, interleaved (n-1) G       -> profile tables
  R2  congestion onset: identical split jobs sharing an uplink before slowdown, n* = C t / g -> profile tables
  R3  energy invariance: GPU energy = G P_idle W + dP * sum(n iters t_comp) for the work done -> simulated runs
  R4  deferral bound: S <= (V/2) sinc(pi d/24) for a sinusoidal grid; oracle bound for the real scenarios -> P2 runs

    python analytic.py            # about 1 minute; writes analytic_results.json and analytic_*.csv
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "code_v0.3.0"))
import numpy as np, pandas as pd, dcsim

HOUR = 3600.0
GHANA = "csv:" + os.path.join(ROOT, "data", "ci_ghana_derived.csv")
OUT = {}


def setup(variant):
    params = json.load(open(os.path.join(ROOT, "results", f"params_full_meas6_{variant}.json")))
    prof = pd.read_csv(os.path.join(ROOT, "data", f"astra_profiles_{variant}.csv"))
    pp = dcsim.PowerParams(); pp.P_gpu_idle, pp.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
    cb = dcsim.ClusterParams(k_rule=params["cluster"]["k_rule"], comm_scale_iter=params["cluster"]["comm_scale_iter"])
    return params["meta"]["seed"], prof, pp, cb


# ---------------- R1 traffic and R2 congestion onset ----------------
rows = []
for v in ("aligned", "interleaved"):
    prof = pd.read_csv(os.path.join(ROOT, "data", f"astra_profiles_{v}.csv"))
    for _, r in prof[prof.pclass == "spread"].iterrows():
        n, G = int(r.n_gpus), dcsim.JOB_TYPES[r.jtype]["G_GB"]
        pred = 2 * (n - 1) / n * G if v == "aligned" else (n - 1) * G
        C = dcsim.ClusterParams().uplink_GBps
        rows.append(dict(bound=v, jtype=r.jtype, n=n, G_GB=G, g_measured=r.interpod_GB, g_predicted=pred,
                         ratio=r.interpod_GB / pred, iter_s=r.iter_time_s, demand_GBps=r.interpod_GB / r.iter_time_s,
                         n_star=C * r.iter_time_s / r.interpod_GB))
R1 = pd.DataFrame(rows); R1.to_csv(os.path.join(HERE, "analytic_traffic.csv"), index=False)
OUT["R1_max_abs_ratio_error"] = float((R1.ratio - 1).abs().max())
OUT["R2_n_star"] = {f"{r.bound}|{r.jtype}|{r.n}": round(r.n_star, 2) for r in R1.itertuples()}

# ---------------- R3 energy invariance ----------------
SC = dict(arrival="poisson", util=0.3, comm_scale=1.0, ci="synthetic_high_var")
e_rows = []
for v in ("aligned", "interleaved"):
    seed, prof, pp, cb = setup(v)
    for pol in ("P0", "P2", "P3", "P4"):
        for rep in range(10):
            m, sim = dcsim.run_one(dcsim.Scenario(**SC), pol, rep, master_seed=seed, profile_df=prof, power_base=pp,
                                   cluster_base=cb, record_trace=True)
            cp = sim.cp; dP = pp.P_gpu_max - pp.P_gpu_idle
            # compute work actually executed in the window (completed jobs fully, unfinished jobs partly)
            work = 0.0
            for j in sim.jobs:
                if j.start is None:
                    continue
                done_iters = j.n_iters - (j.remaining if j.finish is None else 0.0)
                work += j.n_gpus * done_iters * j.t_comp
            pred_gpu_J = pp.PUE * (cp.n_gpus * pp.P_gpu_idle * sim.W + dP * work)
            sim_gpu_J = sim.parts_J["gpu_alloc"] + sim.parts_J["gpu_idle"]
            e_rows.append(dict(bound=v, policy=pol, rep=rep, gpu_sim_kWh=sim_gpu_J / 3.6e6, gpu_pred_kWh=pred_gpu_J / 3.6e6,
                               rel_err=sim_gpu_J / pred_gpu_J - 1, net_kWh=sim.parts_J["net"] / 3.6e6,
                               total_kWh=m["energy_kWh"], n_incomplete=m["n_incomplete"]))
E = pd.DataFrame(e_rows); E.to_csv(os.path.join(HERE, "analytic_energy.csv"), index=False)
OUT["R3_max_abs_rel_err"] = float(E.rel_err.abs().max())
d = []
for (v, rep), g in E.groupby(["bound", "rep"]):
    g = g.set_index("policy")
    for pol in ("P2", "P3", "P4"):
        if g.loc[pol, "n_incomplete"] == 0 and g.loc["P0", "n_incomplete"] == 0:
            d.append(dict(bound=v, policy=pol, gpu_diff_pct=100 * (g.loc[pol, "gpu_sim_kWh"] / g.loc["P0", "gpu_sim_kWh"] - 1),
                          tot_diff_pct=100 * (g.loc[pol, "total_kWh"] / g.loc["P0", "total_kWh"] - 1),
                          net_share_of_diff=(g.loc[pol, "net_kWh"] - g.loc["P0", "net_kWh"]) * 1.0))
D = pd.DataFrame(d)
OUT["R3_gpu_energy_diff_vs_P0_pct_max_abs"] = float(D.gpu_diff_pct.abs().max())
OUT["R3_total_energy_diff_vs_P0_pct_max_abs"] = float(D.tot_diff_pct.abs().max())
OUT["R3_net_energy_kWh_mean"] = E.groupby(["bound", "policy"]).net_kWh.mean().round(2).to_dict().__repr__()
OUT["R3_total_energy_kWh_mean"] = E.groupby(["bound", "policy"]).total_kWh.mean().round(1).to_dict().__repr__()


# ---------------- R4 deferral bound ----------------
def sinc(x):
    return np.sinc(x / np.pi)


def oracle_saving(jobs, ci, threshold=0.0, step_s=900.0):
    """Capacity-free oracle: each deferrable job starts at the lowest-carbon time in [arrival, deadline - duration];
    others start on arrival. Returns the fractional saving of compute-weighted carbon."""
    C = dcsim.CI(ci); now = best = 0.0
    for j in jobs:
        d = j.dur_ref_s; w = j.n_gpus
        c0 = C.integral(j.arrival, j.arrival + d)
        cb = c0
        if j.deferrable:
            last = max(j.arrival, j.deadline - d)
            for s in np.arange(j.arrival, last + 1e-9, step_s):
                cb = min(cb, C.integral(s, s + d))
            if (c0 - cb) / c0 < threshold:
                cb = c0
        now += w * c0; best += w * cb
    return 1 - best / now


b_rows = []
for ci_name, ci_key in (("low variability", "synthetic_low_var"), ("Ghana-derived", GHANA), ("high variability", "synthetic_high_var")):
    for v in ("aligned", "interleaved"):
        seed, prof, pp, cb = setup(v)
        sc = dcsim.Scenario(**{**SC, "ci": ci_key})
        for rep in range(10):
            jobs, prof_d, cp, ppx, ci, cif, _ = dcsim.build_world(sc, rep, seed, prof, cb, pp)
            Vv = dcsim.ci_variability(ci[: int(cp.window_h)])["daily_range_over_mean"]
            b_rows.append(dict(ci=ci_name, bound=v, rep=rep, V=Vv, oracle=100 * oracle_saving(jobs, ci),
                               oracle_thr=100 * oracle_saving(jobs, ci, threshold=0.05)))
B = pd.DataFrame(b_rows)
# simulated P2 vs P0 dynamic carbon at the same cells (stored + regenerated runs)
sim = []
for ci_name, ci_key in (("low variability", "synthetic_low_var"), ("Ghana-derived", GHANA), ("high variability", "synthetic_high_var")):
    for v in ("aligned", "interleaved"):
        seed, prof, pp, cb = setup(v)
        for rep in range(10):
            m = {p: dcsim.run_one(dcsim.Scenario(**{**SC, "ci": ci_key}), p, rep, master_seed=seed, profile_df=prof,
                                  power_base=pp, cluster_base=cb) for p in ("P0", "P2")}
            if m["P0"]["n_incomplete"] == 0 and m["P2"]["n_incomplete"] == 0:
                sim.append(dict(ci=ci_name, bound=v, rep=rep, p2=-100 * (m["P2"]["carbon_dyn_kg"] / m["P0"]["carbon_dyn_kg"] - 1)))
S = pd.DataFrame(sim)
B = B.merge(S, on=["ci", "bound", "rep"], how="left"); B.to_csv(os.path.join(HERE, "analytic_bound.csv"), index=False)
summ = B.groupby(["ci", "bound"]).agg(V=("V", "mean"), oracle=("oracle", "mean"), oracle_thr=("oracle_thr", "mean"), p2=("p2", "mean")).round(2)
OUT["R4_summary"] = summ.reset_index().to_dict(orient="records")
OUT["R4_bound_holds_every_rep"] = bool((B.p2 <= B.oracle + 1e-9).all())
# closed-form check: sinusoidal grid, unlimited slack, all jobs deferrable, duration d
t = np.arange(0, 24 * 60) * 60.0
chk = []
for V in (0.1, 0.5, 1.0):
    for dh in (1.0, 4.0, 12.0):
        ci = 400 * (1 + V / 2 * np.sin(2 * np.pi * np.arange(24 * 40) / 24))
        J = [dcsim.Job(jid=i, arrival=float(a), jtype="x", n_gpus=1, n_iters=1, deferrable=True, deadline=float(a) + 48 * HOUR + dh * HOUR,
                       dur_ref_s=dh * HOUR) for i, a in enumerate(np.arange(0, 24 * HOUR, 300.0) + 5 * 24 * HOUR)]
        num = 100 * oracle_saving(J, ci, step_s=300.0)
        chk.append(dict(V=V, d_h=dh, numeric=round(num, 3), closed_form=round(100 * V / 2 * sinc(np.pi * dh / 24), 3)))
OUT["R4_closed_form_check"] = chk
json.dump(OUT, open(os.path.join(HERE, "analytic_results.json"), "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
