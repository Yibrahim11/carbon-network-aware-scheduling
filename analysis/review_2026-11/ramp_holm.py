import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.dirname(_os.path.dirname(HERE))
import sys, json
sys.path.insert(0, f"{ROOT}/code_v0.3.0")
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(f"{HERE}/matched_runs.csv")
tr = json.load(open(f"{HERE}/traces.json"))
W = 900.0
def ramp(seg):
    a = np.array(seg); t0, t1, p = a[:, 0], a[:, 1], a[:, 2]
    T = t1.max(); edges = np.arange(0, T + W, W); e = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):              # time-weighted mean power in each 15-min window
        lo, hi = edges[k], edges[k + 1]
        ov = np.clip(np.minimum(t1, hi) - np.maximum(t0, lo), 0, None)
        e[k] = (p * ov).sum() / max(ov.sum(), 1e-9)
    full = e[:int(T // W)]                       # drop the trailing partial window
    dP = np.abs(np.diff(full)) / 1000.0
    return dP.max(), np.percentile(dP, 99)
rows = []
for k, seg in tr.items():
    v, u, c, pol, rep = k.split("|")
    mx, p99 = ramp(seg); rows.append(dict(variant=v, util=float(u), comm_scale=float(c), policy=pol, rep=int(rep), ramp_max_kW=mx, ramp_p99_kW=p99))
R = pd.DataFrame(rows)
d = d.merge(R, on=["variant", "util", "comm_scale", "policy", "rep"])
d.to_csv(f"{HERE}/matched_runs_ramp.csv", index=False)
print("ramp rate (kW per 15 min), mean over 10 reps:")
print(d.groupby(["variant", "util", "comm_scale", "policy"])[["ramp_max_kW", "ramp_p99_kW", "peak_kW"]].mean().round(1).unstack("policy").to_string())

def paired(sub, a, b, met):
    A = sub[sub.policy == a].set_index("rep"); B = sub[sub.policy == b].set_index("rep")
    ok = (A.n_incomplete == 0) & (B.n_incomplete == 0)
    x = (A[met][ok] / B[met][ok] - 1) * 100
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    t = m / (s / np.sqrt(n)); p = 2 * stats.t.sf(abs(t), n - 1)
    return n, m, s, p

# ---- primary comparisons: matched operating point (Poisson, 30% load, 1x communication, high-variability grid) ----
cell = lambda v: d[(d.variant == v) & np.isclose(d.util, 0.3) & np.isclose(d.comm_scale, 1.0)]
P = []
for v in ("aligned", "interleaved"):
    P.append(("H1", v, "P2 vs P0", "carbon_dyn_kg", *paired(cell(v), "P2", "P0", "carbon_dyn_kg")))
for v in ("aligned", "interleaved"):
    for b in ("P2", "P3"):
        for met in ("carbon_per_gpuh_g", "jct_mean_h"):
            P.append(("H2", v, f"P4 vs {b}", met, *paired(cell(v), "P4", b, met)))
# H3: same seeds and job stream in both bounds -> pair by replication on P0 total network inflation
A = cell("aligned"); I = cell("interleaved")
a0 = A[A.policy == "P0"].set_index("rep"); i0 = I[I.policy == "P0"].set_index("rep")
same = (a0.workload_hash == i0.workload_hash).all()
x = (i0.net_inflation_total / a0.net_inflation_total - 1) * 100
n, m, s = len(x), x.mean(), x.std(ddof=1); p = 2 * stats.t.sf(abs(m / (s / np.sqrt(n))), n - 1)
P.append(("H3", "both", "P0 interleaved vs aligned", "net_inflation_total", n, m, s, p))
T = pd.DataFrame(P, columns=["hyp", "bound", "comparison", "metric", "n", "mean", "sd", "p"])
K = len(T)
order = T.p.sort_values().index; holm = pd.Series(index=T.index, dtype=float); run = 0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (K - rank) * T.p[i])); holm[i] = run
T["p_holm"] = holm
for lab, alpha in (("ci", 0.05), ("ci_bonf", 0.05 / K)):
    h = stats.t.ppf(1 - alpha / 2, T.n - 1) * T.sd / np.sqrt(T.n)
    T[lab] = [f"[{lo:+.1f}, {hi:+.1f}]" for lo, hi in zip(T["mean"] - h, T["mean"] + h)]
T["reject_holm"] = T.p_holm < 0.05
print(f"\nPRIMARY COMPARISONS (K = {K}; job streams identical across bounds: {same})")
print(T[["hyp", "bound", "comparison", "metric", "n", "mean", "ci", "ci_bonf", "p", "p_holm", "reject_holm"]].round({"mean": 2, "p": 6, "p_holm": 6}).to_string(index=False))
T.to_csv(f"{HERE}/primary_comparisons.csv", index=False)
