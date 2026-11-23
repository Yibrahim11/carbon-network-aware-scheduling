import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.dirname(_os.path.dirname(HERE))
import sys, json
sys.path.insert(0, f"{ROOT}/code_v0.3.0")
import numpy as np, pandas as pd
from scipy import stats
PK = f"{ROOT}/results"
d = pd.read_csv(f"{HERE}/matched_runs.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)

def paired(sub, a, b, met, rel=True):
    A = sub[sub.policy == a].set_index("rep"); B = sub[sub.policy == b].set_index("rep")
    ok = (A.n_incomplete == 0) & (B.n_incomplete == 0)
    x = (A[met][ok] / B[met][ok] - 1) * 100 if rel else (A[met][ok] - B[met][ok])
    n = len(x); m = x.mean(); s = x.std(ddof=1)
    tstat = m / (s / np.sqrt(n)) if s > 0 else np.inf * np.sign(m)
    p = 2 * stats.t.sf(abs(tstat), n - 1) if np.isfinite(tstat) else 0.0
    return dict(mean=m, sd=s, n=n, cens=int((~ok).sum()), p=p)

def ci(r, alpha=0.05):
    h = stats.t.ppf(1 - alpha / 2, r["n"] - 1) * r["sd"] / np.sqrt(r["n"]); return r["mean"] - h, r["mean"] + h

# ---- A. P4 vs P2 and P4 vs P3, both metrics, all new cells ----
rows = []
for (v, u, c), sub in d.groupby(["variant", "util", "comm_scale"]):
    for a, b in (("P2", "P0"), ("P3", "P0"), ("P4", "P0"), ("P4", "P2"), ("P4", "P3")):
        for met in ("carbon_dyn_kg", "carbon_per_gpuh_g", "jct_mean_h", "energy_kWh", "peak_kW", "net_inflation_total"):
            r = paired(sub, a, b, met); lo, hi = ci(r)
            rows.append(dict(variant=v, util=u, comm=c, cmp=f"{a}vs{b}", metric=met, mean=r["mean"], lo=lo, hi=hi, n=r["n"], cens=r["cens"], p=r["p"]))
A = pd.DataFrame(rows); A.to_csv(f"{HERE}/paired_cells.csv", index=False)
f = lambda r: f"{r['mean']:+.1f} [{r['lo']:+.1f},{r['hi']:+.1f}]" + (f" c{r['cens']}" if r['cens'] else "")
A["v"] = A.apply(f, axis=1)
for met in ("carbon_per_gpuh_g", "carbon_dyn_kg", "jct_mean_h", "net_inflation_total"):
    print(f"\n== {met} (% paired, 95% CI)")
    print(A[(A.metric == met)].pivot_table(index=["variant", "util", "comm"], columns="cmp", values="v", aggfunc="first").to_string())
means = d.groupby(["variant", "util", "comm_scale", "policy"]).net_inflation_total.mean().unstack().round(3)
print("\nmean total network inflation:\n", means.to_string())

# ---- B. P4 vs P3 across the interleaved communication sweep (stored sens runs) ----
s = pd.read_csv(f"{PK}/sens_full_meas6_interleaved.csv"); s = s[s.tag == "sens:comm_scale"]
print("\n== interleaved comm sweep (util 0.3): P4 vs P3 and P4 vs P2")
for c, sub in s.groupby("comm_scale"):
    out = []
    for a, b in (("P4", "P3"), ("P4", "P2")):
        for met in ("carbon_per_gpuh_g", "jct_mean_h"):
            r = paired(sub, a, b, met); lo, hi = ci(r); out.append(f"{a}v{b} {met[:6]} {r['mean']:+.1f} [{lo:+.1f},{hi:+.1f}]" + (f" c{r['cens']}" if r['cens'] else ""))
    print(f"  {c}: " + " | ".join(out))

# ---- C. energy bound across every stored cell ----
for v in ("aligned", "interleaved"):
    mm = pd.concat([pd.read_csv(f"{PK}/main_full_meas6_{v}.csv"), pd.read_csv(f"{PK}/sens_full_meas6_{v}.csv")])
    key = ["tag", "arrival", "util", "comm_scale", "ci", "cap_frac", "fabric", "forecast_sigma", "slack_lo", "slack_hi", "deferrable_frac", "w_delay"]
    mm = mm.fillna({"cap_frac": -1}); worst = 0; worst_cell = None; worst_main = 0
    for k, sub in mm.groupby(key, dropna=False):
        for p in ("P2", "P3", "P4"):
            if p not in set(sub.policy): continue
            r = paired(sub, p, "P0", "energy_kWh")
            if r["n"] and abs(r["mean"]) > abs(worst): worst, worst_cell = r["mean"], (k, p)
            if k[0] == "main" and r["n"] and abs(r["mean"]) > abs(worst_main): worst_main = r["mean"]
    print(f"\nenergy {v}: largest |paired diff vs P0| over ALL cells {worst:+.3f}% at {worst_cell} | over main factorial {worst_main:+.3f}%")
    ok = mm.groupby(["tag"] + key[1:] + ["rep"], dropna=False).n_incomplete.transform("max") == 0
    mf = mm[(mm.tag == "main") & ok].groupby("policy").energy_kWh.mean()
    print("   main-factorial ratio of means vs P0:", {p: round(100 * (mf[p] / mf["P0"] - 1), 3) for p in ("P2", "P3", "P4")})
    print(f"   run wall time: main {mm[mm.tag=='main'].runtime_s.mean():.3f} s, sens {mm[mm.tag!='main'].runtime_s.mean():.3f} s, total for {len(mm)} runs {mm.runtime_s.sum()/60:.1f} min (one core)")
