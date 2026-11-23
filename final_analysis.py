"""Analysis of the final experiments (thesis Chapter IV). Run from anywhere after final_design.py has finished both bounds:

    python final/final_analysis.py

Writes final/out/*.csv and final/out/summary.json. Every number quoted in Chapter IV comes from these files.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code_v0.4.0"))
import numpy as np, pandas as pd
from scipy import stats
import dcsim

OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
D_C, D_T = 2.0, 5.0
B = ("aligned", "interleaved")
S = {}


def load(b):
    d = pd.read_csv(os.path.join(HERE, f"results_{b}.csv"), low_memory=False)
    assert (d.error.fillna("") == "").all(), "failed runs present"
    d["bound"] = b
    return d


R = pd.concat([load(b) for b in B], ignore_index=True)
for c in ("cap_frac",):
    R[c] = R[c].astype(object).where(R[c].notna(), "none")
S["runs"] = {b: int((R.bound == b).sum()) for b in B}


def cell(d, tag, **kw):
    q = d[d.tag == tag]
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


def paired(q, a, b, met, need_complete=None):
    """Mean of per-replication % differences a vs b, t-interval; JCT-type metrics only on replications where both
    policies completed every job."""
    x = q.pivot_table(index="rep", columns="policy", values=met)
    inc = q.pivot_table(index="rep", columns="policy", values="n_incomplete")
    if need_complete is None:
        need_complete = met in ("jct_mean_h", "jct_p95_h", "wait_mean_h", "net_inflation_total", "net_inflation")
    keep = ((inc[a] == 0) & (inc[b] == 0)) if need_complete else pd.Series(True, index=x.index)
    v = 100 * (x.loc[keep, a] / x.loc[keep, b] - 1)
    n = len(v); m = v.mean(); sd = v.std(ddof=1)
    se = sd / np.sqrt(n) if n > 1 else np.nan
    t = m / se if se and se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if n > 1 and np.isfinite(t) else np.nan
    return dict(mean=m, lo=m - stats.t.ppf(.975, n - 1) * se, hi=m + stats.t.ppf(.975, n - 1) * se, p=p, n=n, se=se, df=n - 1,
                dropped=int((~keep).sum()))


# ---------------- primary comparisons ----------------
prim = []
for b in B:
    q = cell(R[R.bound == b], "primary")
    prim.append(dict(hyp="H1,1", bound=b, cmp="P2 vs P0", metric="carbon_dyn_kg", delta=D_C, **paired(q, "P2", "P0", "carbon_dyn_kg")))
    for ref in ("P2", "P3"):
        prim.append(dict(hyp="H1,2", bound=b, cmp=f"P4 vs {ref}", metric="carbon_per_gpuh_g", delta=D_C, **paired(q, "P4", ref, "carbon_per_gpuh_g")))
        prim.append(dict(hyp="H1,2", bound=b, cmp=f"P4 vs {ref}", metric="jct_mean_h", delta=D_T, **paired(q, "P4", ref, "jct_mean_h")))
qa = cell(R[R.bound == "aligned"], "primary"); qi = cell(R[R.bound == "interleaved"], "primary")
a0 = qa[(qa.policy == "P0") & (qa.n_incomplete == 0)].net_inflation_total
i0 = qi[(qi.policy == "P0") & (qi.n_incomplete == 0)].net_inflation_total
w = stats.ttest_ind(i0, a0, equal_var=False)
m = 100 * (i0.mean() / a0.mean() - 1)
# CI for the ratio of means by the delta method on log scale
se_log = np.sqrt(i0.var(ddof=1) / len(i0) / i0.mean() ** 2 + a0.var(ddof=1) / len(a0) / a0.mean() ** 2)
dfw = w.df if hasattr(w, "df") else len(i0) + len(a0) - 2
tq = stats.t.ppf(.975, dfw)
prim.append(dict(hyp="H1,3", bound="interleaved vs aligned", cmp="P0", metric="net_inflation_total", delta=D_T, mean=m,
                 lo=100 * (np.exp(np.log(i0.mean() / a0.mean()) - tq * se_log) - 1),
                 hi=100 * (np.exp(np.log(i0.mean() / a0.mean()) + tq * se_log) - 1), p=w.pvalue, n=len(i0) + len(a0),
                 se=se_log, df=dfw, dropped=int(20 - len(i0) - len(a0))))
P = pd.DataFrame(prim)
# Holm
order = np.argsort(P.p.values); k = len(P); adj = np.empty(k); run = 0
for r, i in enumerate(order):
    run = max(run, min(1, (k - r) * P.p.values[i])); adj[i] = run
P["p_holm"] = adj
alpha_b = 0.05 / k
for i, r in P.iterrows():
    if r.hyp == "H1,3":
        tb = stats.t.ppf(1 - alpha_b / 2, r.df)
        rat = np.log(1 + r["mean"] / 100)
        P.loc[i, "lo_bonf"] = 100 * (np.exp(rat - tb * r.se) - 1); P.loc[i, "hi_bonf"] = 100 * (np.exp(rat + tb * r.se) - 1)
    else:
        tb = stats.t.ppf(1 - alpha_b / 2, r.df)
        P.loc[i, "lo_bonf"] = r["mean"] - tb * r.se; P.loc[i, "hi_bonf"] = r["mean"] + tb * r.se
# real improvement: Holm-significant, point estimate reaches delta in the favourable direction
def favourable(r):
    if r.hyp == "H1,3":
        return r["mean"] >= r.delta          # interleaved inflation larger than aligned
    return r["mean"] <= -r.delta            # reduction
P["meets_delta"] = [favourable(r) for _, r in P.iterrows()]
P["significant"] = P.p_holm < 0.05
P["real"] = P.meets_delta & P.significant
P.to_csv(os.path.join(OUT, "primary.csv"), index=False)
S["primary"] = P.round(4).to_dict(orient="records")

# ---------------- primary point: all metrics, all policies ----------------
mets = ["carbon_dyn_kg", "carbon_kg", "carbon_per_gpuh_g", "energy_kWh", "jct_mean_h", "jct_p95_h", "wait_mean_h",
        "net_inflation_total", "net_inflation", "deadline_viol_frac", "peak_kW", "spread_frac", "completion_rate"]
rows = []
for b in B:
    q = cell(R[R.bound == b], "primary")
    for pol in ("P2", "P3", "P4"):
        for met in mets:
            if met in ("deadline_viol_frac", "spread_frac", "completion_rate"):
                continue
            rows.append(dict(bound=b, cmp=f"{pol} vs P0", metric=met, **paired(q, pol, "P0", met)))
    for pol in ("P0", "P2", "P3", "P4"):
        g = q[q.policy == pol]
        rows.append(dict(bound=b, cmp=f"{pol} level", metric="levels", **{f"mean_{m_}": g[m_].mean() for m_ in mets}))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "primary_all.csv"), index=False)

# ramp rate at the primary point (re-simulated with traces)
W15 = 900.0
def ramp(trace):
    a = np.array(trace); t0, t1, p = a[:, 0], a[:, 1], a[:, 2]
    T = t1.max(); edges = np.arange(0, T + W15, W15); e = np.zeros(len(edges) - 1)
    for k_ in range(len(edges) - 1):
        lo, hi = edges[k_], edges[k_ + 1]
        ov = np.clip(np.minimum(t1, hi) - np.maximum(t0, lo), 0, None)
        e[k_] = (p * ov).sum() / max(ov.sum(), 1e-9)
    full = e[:int(T // W15)]
    dP = np.abs(np.diff(full)) / 1000.0
    return dP.max(), np.percentile(dP, 99)
rr = []
os.chdir(ROOT)
for b in B:
    params = json.load(open(os.path.join(ROOT, "results", f"params_full_meas6_{b}.json")))
    prof = pd.read_csv(os.path.join(ROOT, "data", f"astra_profiles_{b}.csv"))
    pp = dcsim.PowerParams(); pp.P_gpu_idle, pp.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
    cb = dcsim.ClusterParams(k_rule=params["cluster"]["k_rule"], comm_scale_iter=params["cluster"]["comm_scale_iter"])
    sc = dcsim.Scenario(arrival="poisson", util=0.3, comm_scale=1.0, ci="synthetic_high_var", forecast="persistence")
    for pol in ("P0", "P2", "P3", "P4"):
        for rep in range(10):
            m_, sim = dcsim.run_one(sc, pol, rep, master_seed=params["meta"]["seed"], profile_df=prof, power_base=pp,
                                    cluster_base=cb, record_trace=True)
            mx, p99 = ramp(sim.trace)
            rr.append(dict(bound=b, policy=pol, rep=rep, ramp_max_kW=mx, ramp_p99_kW=p99, peak_kW=m_["peak_kW"],
                           carbon_dyn_kg=m_["carbon_dyn_kg"]))
RR = pd.DataFrame(rr); RR.to_csv(os.path.join(OUT, "ramp_primary.csv"), index=False)
chk = R[(R.tag == "primary")].set_index(["bound", "policy", "rep"]).carbon_dyn_kg
S["ramp_rerun_matches_stored"] = bool(np.allclose(RR.set_index(["bound", "policy", "rep"]).carbon_dyn_kg.sort_index(), chk.sort_index(), rtol=1e-12))
ramp_rows = []
for b in B:
    q = RR[RR.bound == b].assign(n_incomplete=0)
    for pol in ("P2", "P3", "P4"):
        for met in ("ramp_max_kW", "ramp_p99_kW", "peak_kW"):
            ramp_rows.append(dict(bound=b, cmp=f"{pol} vs P0", metric=met, **paired(q, pol, "P0", met, need_complete=False)))
pd.DataFrame(ramp_rows).to_csv(os.path.join(OUT, "ramp_paired.csv"), index=False)

# ---------------- V sweep: carbon boundary ----------------
vs = []
for b in B:
    for fc in ("persistence", "perfect"):
        for V in sorted(R[(R.tag == "vsweep")].ci.unique(), key=lambda s: float(s.split(":")[1])):
            q = cell(R[R.bound == b], "vsweep", ci=V, forecast=fc)
            Vr = np.mean([dcsim.ci_variability(dcsim.make_ci(V, r, 265)[:264])["daily_range_over_mean"] for r in range(10)])
            p2 = paired(q, "P2", "P0", "carbon_dyn_kg"); p4 = paired(q, "P4", "P3", "carbon_per_gpuh_g")
            p4j = paired(q, "P4", "P3", "jct_mean_h"); p2j = paired(q, "P2", "P0", "jct_mean_h")
            vs.append(dict(bound=b, forecast=fc, V_target=float(V.split(":")[1]), V=Vr, P2_dyn=-p2["mean"], P2_lo=-p2["hi"],
                           P2_hi=-p2["lo"], P2_p=p2["p"], P4vP3_cgpuh=-p4["mean"], P4vP3_jct=p4j["mean"], P2_jct=p2j["mean"],
                           P4vP3_lo=-p4["hi"], P4vP3_hi=-p4["lo"]))
VS = pd.DataFrame(vs); VS.to_csv(os.path.join(OUT, "vsweep.csv"), index=False)


def crossing(x, y, thr):
    """First crossing of y over thr (linear interpolation), with the bracketing points."""
    x, y = np.asarray(x), np.asarray(y)
    for i in range(1, len(x)):
        if y[i - 1] < thr <= y[i]:
            return float(x[i - 1] + (thr - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])), float(x[i - 1]), float(x[i])
    return None, None, None
vstar = {}
for (b, fc), g in VS.groupby(["bound", "forecast"]):
    g = g.sort_values("V")
    vstar[f"{b}|{fc}"] = dict(zip(("V_star", "V_below", "V_above"), crossing(g.V, g.P2_dyn, D_C)))
    # where the lower 95% bound of the saving first exceeds delta_C (conservative)
    vstar[f"{b}|{fc}"]["V_star_lower_bound"] = crossing(g.V, g.P2_lo, D_C)[0]
S["V_star"] = vstar

# ---------------- main factorial: decision points for A* ----------------
dp = []
for b in B:
    q0 = R[(R.bound == b) & (R.tag == "main")]
    for key, q in q0.groupby(["arrival", "util", "comm_scale", "ci"]):
        I0 = q[(q.policy == "P0") & (q.n_incomplete == 0)].net_inflation_total.mean()
        I3 = q[(q.policy == "P3") & (q.n_incomplete == 0)].net_inflation_total.mean()
        g = paired(q, "P3", "P0", "jct_mean_h"); c2 = paired(q, "P2", "P0", "carbon_dyn_kg")
        inc = q.groupby("policy").n_incomplete.apply(lambda s: int((s > 0).sum())).to_dict()
        dp.append(dict(src=f"synthetic {b}", bound=b, arrival=key[0], util=key[1], comm_scale=key[2], ci=key[3], A=I0 - I3,
                       I_P0=I0, I_P3=I3, P3_jct=g["mean"], P3_n=g["n"], P2_dyn=c2["mean"], **{f"unfinished_{k_}": v_ for k_, v_ in inc.items()}))
pt = pd.read_csv(os.path.join(ROOT, "results", "podtrace_full_meas6_interleaved_pod32.csv"))
for pod in (32, 16, 8):
    t = pd.read_csv(os.path.join(ROOT, "results", f"podtrace_full_meas6_interleaved_pod{pod}.csv"))
    for key, q in t.groupby(["util", "comm_scale"]):
        I0 = q[(q.policy == "P0") & (q.n_incomplete == 0)].net_inflation_total.mean()
        I3 = q[(q.policy == "P3") & (q.n_incomplete == 0)].net_inflation_total.mean()
        g = paired(q, "P3", "P0", "jct_mean_h")
        dp.append(dict(src=f"trace pod{pod}", bound="interleaved", util=key[0], comm_scale=key[1], A=I0 - I3, I_P0=I0, I_P3=I3,
                       P3_jct=g["mean"], P3_n=g["n"]))
DP = pd.DataFrame(dp)
# P0 and P3 ignore carbon intensity, so the three CI scenarios of a main-factorial cell give identical P3-vs-P0 results:
# keep one per (bound, arrival, util, comm_scale) to avoid counting the same cell three times.
syn = DP.src.str.startswith("synthetic")
DP = pd.concat([DP[syn & (DP.ci == "synthetic_high_var")], DP[~syn]], ignore_index=True)
DP.to_csv(os.path.join(OUT, "decision_points.csv"), index=False)
ok = DP.dropna(subset=["P3_jct"])
S["A_spearman"] = float(stats.spearmanr(ok.A, -ok.P3_jct).correlation)
S["I0_spearman"] = float(stats.spearmanr(ok.I_P0, -ok.P3_jct).correlation)
s_ = ok.sort_values("A")
below = s_[-s_.P3_jct < D_T].A.max(); above = s_[-s_.P3_jct >= D_T].A.min()
S["A_star_bracket"] = [float(below), float(above)]
S["A_star_misordered"] = int(((s_.A > above) & (-s_.P3_jct < D_T)).sum() + ((s_.A < below) & (-s_.P3_jct >= D_T)).sum())

# ---------------- main factorial summaries ----------------
mf = []
for b in B:
    q0 = R[(R.bound == b) & (R.tag == "main")]
    for ci, qc in q0.groupby("ci"):
        for pol in ("P2", "P3", "P4"):
            vals = {m_: [] for m_ in ("carbon_dyn_kg", "carbon_per_gpuh_g", "jct_mean_h", "energy_kWh")}
            for key, q in qc.groupby(["arrival", "util", "comm_scale"]):
                for m_ in vals:
                    vals[m_].append(paired(q, pol, "P0", m_)["mean"])
            mf.append(dict(bound=b, ci=ci, cmp=f"{pol} vs P0", **{f"avg_{k_}": np.nanmean(v_) for k_, v_ in vals.items()},
                           **{f"min_{k_}": np.nanmin(v_) for k_, v_ in vals.items()}, **{f"max_{k_}": np.nanmax(v_) for k_, v_ in vals.items()}))
    unf = q0.groupby(["util", "comm_scale", "policy"]).n_incomplete.apply(lambda s: int((s > 0).sum())).reset_index()
    unf.assign(bound=b).to_csv(os.path.join(OUT, f"unfinished_main_{b}.csv"), index=False)
MF = pd.DataFrame(mf); MF.to_csv(os.path.join(OUT, "main_summary.csv"), index=False)
E = R[R.tag == "main"]
energy_max = 0.0
for keys, q in E.groupby(["bound", "arrival", "util", "comm_scale", "ci"]):
    for pol in ("P2", "P3", "P4"):
        energy_max = max(energy_max, abs(paired(q, pol, "P0", "energy_kWh", need_complete=True)["mean"]))
S["energy_max_abs_pct_main_cell"] = energy_max

# ---------------- Ghana ----------------
gh = []
for b in B:
    for ci, q in cell(R[R.bound == b], "ghana").groupby("ci"):
        path = ci[4:]
        v = pd.read_csv(os.path.join(ROOT, path)).iloc[:, 1].to_numpy()
        Vr = dcsim.ci_variability(np.tile(v, 2)[:264])["daily_range_over_mean"]
        p2 = paired(q, "P2", "P0", "carbon_dyn_kg"); p4 = paired(q, "P4", "P3", "carbon_per_gpuh_g")
        p4c = paired(q, "P4", "P0", "carbon_dyn_kg")
        gh.append(dict(bound=b, series=os.path.basename(path)[:-4], mean=float(v.mean()), V=Vr, P2_dyn=-p2["mean"],
                       P2_lo=-p2["hi"], P2_hi=-p2["lo"], P4_dyn=-p4c["mean"], P4vP3_cgpuh=-p4["mean"]))
GHT = pd.DataFrame(gh); GHT.to_csv(os.path.join(OUT, "ghana.csv"), index=False)

# ---------------- sensitivity ----------------
sens = []
for b in B:
    for tag, qt in R[(R.bound == b) & R.tag.str.startswith("sens:")].groupby("tag"):
        f = tag.split(":")[1]
        col = {"slack": "slack_hi", "forecast": None}.get(f, f)
        if f == "forecast":
            qt = qt.assign(level=qt.forecast + "|" + qt.forecast_sigma.astype(str))
        else:
            qt = qt.assign(level=qt[col].astype(str))
        for lv, q in qt.groupby("level"):
            r_ = dict(bound=b, factor=f, level=lv)
            for pol in ("P2", "P2raw", "P3", "P4", "P4nc"):
                r_[f"{pol}_dyn"] = paired(q, pol, "P0", "carbon_dyn_kg")["mean"]
                r_[f"{pol}_cgpuh"] = paired(q, pol, "P0", "carbon_per_gpuh_g")["mean"]
                r_[f"{pol}_jct"] = paired(q, pol, "P0", "jct_mean_h")["mean"]
            r_["unfinished"] = {p_: int((q[q.policy == p_].n_incomplete > 0).sum()) for p_ in ("P0", "P2", "P3", "P4")}
            sens.append(r_)
pd.DataFrame(sens).to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)

# ---------------- curved power model ----------------
gm = []
for b in B:
    G = pd.read_csv(os.path.join(HERE, f"results_{b}_gamma058.csv"))
    L = R[(R.bound == b) & (R.tag == "sens:comm_scale") & R.policy.isin(["P0", "P2", "P3", "P4"])]
    for cs in (0.5, 1.0, 2.0):
        for lab, q in (("gamma=0.58", G[G.comm_scale == cs]), ("linear", L[L.comm_scale == cs])):
            for pol in ("P2", "P3", "P4"):
                gm.append(dict(bound=b, comm_scale=cs, model=lab, cmp=f"{pol} vs P0",
                               energy=paired(q, pol, "P0", "energy_kWh", need_complete=True)["mean"],
                               dyn=paired(q, pol, "P0", "carbon_dyn_kg")["mean"], jct=paired(q, pol, "P0", "jct_mean_h")["mean"]))
pd.DataFrame(gm).to_csv(os.path.join(OUT, "power_shape.csv"), index=False)
json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in S.items() if k != "primary"}, indent=1, default=float))
print(P[["hyp", "bound", "cmp", "metric", "mean", "lo", "hi", "p_holm", "real"]].round(3).to_string())
