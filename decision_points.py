"""Points for the network boundary of the decision rule (proposal v7, Figure 2): for each cell, the baseline's total
network inflation I_P0, the network-aware policy's I_P3, the avoidable inflation A = I_P0 - I_P3, and the paired change
in mean completion time of P3 against P0. Sources: production-trace pod-size runs and the communication sweeps."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); RES = f"{ROOT}/results"
import numpy as np, pandas as pd
def p3_jct(g):
    A = g[g.policy == "P3"].set_index("rep"); B = g[g.policy == "P0"].set_index("rep"); ok = (A.n_incomplete == 0) & (B.n_incomplete == 0)
    return ((A.jct_mean_h[ok] / B.jct_mean_h[ok] - 1) * 100).mean()
pts = []
pm = pd.read_csv(f"{RES}/podtrace_means_full_meas6_interleaved.csv"); pp = pd.read_csv(f"{RES}/podtrace_paired_full_meas6_interleaved.csv")
for (pod, u, cs), g in pm.groupby(["pod_gpus", "util", "comm_scale"]):
    j = pp[(pp.pod_gpus == pod) & (pp.util == u) & (pp.comm_scale == cs) & (pp.metric == "jct_mean_h") & (pp.a == "P3") & (pp.b == "P0")]["diff"].iloc[0]
    pts.append(dict(src=f"trace pod{pod}", util=u, comm=cs, I0=g[g.policy == "P0"].net_inflation_total.iloc[0], I3=g[g.policy == "P3"].net_inflation_total.iloc[0], P3_jct=j, forced=(pod == 8)))
for v in ("aligned", "interleaved"):
    s = pd.read_csv(f"{RES}/sens_full_meas6_{v}.csv"); s = s[s.tag == "sens:comm_scale"]
    for cs, g in s.groupby("comm_scale"):
        pts.append(dict(src=f"synthetic {v}", util=g.util.iloc[0], comm=cs, I0=g[g.policy == "P0"].net_inflation_total.mean(), I3=g[g.policy == "P3"].net_inflation_total.mean(), P3_jct=p3_jct(g), forced=False))
D = pd.DataFrame(pts); D["A"] = D.I0 - D.I3; D = D.sort_values("A")
D.to_csv(f"{HERE}/decision_points.csv", index=False)
print("Spearman, A vs gain:", round(D.A.corr(-D.P3_jct, method="spearman"), 3), "| I_P0 vs gain:", round(D.I0.corr(-D.P3_jct, method="spearman"), 3))
