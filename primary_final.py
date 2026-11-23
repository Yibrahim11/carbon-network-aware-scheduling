"""Table 3 of proposal v7: the eleven primary comparisons with Holm-adjusted p-values and raw and Bonferroni intervals.
H1 and H2 are paired (common random numbers); H3 is a two-sample Welch test, because the two placement bounds share
seeds but not identical job streams. Run after ramp_holm.py."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd
from scipy import stats
T = pd.read_csv(f"{HERE}/primary_comparisons.csv")
d = pd.read_csv(f"{HERE}/matched_runs_ramp.csv"); c = d[np.isclose(d.util, 0.3) & np.isclose(d.comm_scale, 1.0)]
a = c[(c.variant == "aligned") & (c.policy == "P0")].net_inflation_total.to_numpy(); i = c[(c.variant == "interleaved") & (c.policy == "P0")].net_inflation_total.to_numpy()
w = stats.ttest_ind(i, a, equal_var=False); va, vi = a.var(ddof=1) / len(a), i.var(ddof=1) / len(i); se = np.sqrt(va + vi)
df = (va + vi) ** 2 / (va ** 2 / (len(a) - 1) + vi ** 2 / (len(i) - 1)); diff = i.mean() - a.mean(); K = len(T)
h = lambda al: stats.t.ppf(1 - al / 2, df) * se
k = T.index[T.hyp == "H3"][0]
T.loc[k, ["comparison", "metric", "mean", "p"]] = ["P0, interleaved minus aligned", "net_inflation_total (absolute)", diff, w.pvalue]
T.loc[k, "ci"] = f"[{diff - h(.05):+.2f}, {diff + h(.05):+.2f}]"; T.loc[k, "ci_bonf"] = f"[{diff - h(.05 / K):+.2f}, {diff + h(.05 / K):+.2f}]"
run = 0
for rank, j in enumerate(T.p.sort_values().index):
    run = max(run, min(1.0, (K - rank) * T.p[j])); T.loc[j, "p_holm"] = run
T["reject_holm"] = T.p_holm < 0.05
T.to_csv(f"{HERE}/primary_comparisons_final.csv", index=False)
print(T[["hyp", "bound", "comparison", "metric", "mean", "ci", "ci_bonf", "p_holm", "reject_holm"]].to_string(index=False))
