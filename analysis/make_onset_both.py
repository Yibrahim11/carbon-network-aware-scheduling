"""Figure 1 of the proposal (onset_both.png): total network runtime inflation against communication intensity.

Total network inflation = GPU-weighted actual runtime / runtime the same jobs would have had inside one pod with no
congestion (dcsim >= 0.3.0 metric net_inflation_total). It contains both the placement penalty (a job split across
pods communicates more slowly even on an idle fabric) and congestion on the pod uplinks.
Reads the stored sensitivity runs (comm_scale sweep, 10 replications) and draws policy means with 95% t-intervals.
(a) aligned placement, base load 70%; (b) interleaved placement, base load 30%. Runs tagged meas6 (dcsim 0.3.0).
    python analysis/make_onset_both.py          -> analysis/onset_both.png
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
COLOR = {"P0": "#6b6a66", "P2": "#2a78d6", "P3": "#eb6834", "P4": "#1baf7a"}     # same as dcsim_exp.POLICY_COLOR
MARK = {"P0": "o", "P2": "s", "P3": "^", "P4": "D"}
PANELS = [("sens_full_meas6_aligned.csv", "(a) Aligned placement (best case)"),
          ("sens_full_meas6_interleaved.csv", "(b) Interleaved placement (worst case)")]
METRIC = "net_inflation_total"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"], "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)
for ax, (fn, title) in zip(axes, PANELS):
    d = pd.read_csv(os.path.join(RES, fn))
    d = d[(d.tag == "sens:comm_scale") & d.policy.isin(list(COLOR))]
    for pol in COLOR:
        g = d[d.policy == pol].groupby("comm_scale")[METRIC]
        m, s, n = g.mean(), g.std(ddof=1), g.count()
        h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
        ax.errorbar(m.index, m.values, yerr=h.values, color=COLOR[pol], marker=MARK[pol], ms=4, lw=1.3,
                    capsize=2, elinewidth=0.8, label=pol)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("communication intensity (× measured cost)")
    ax.grid(color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("total network runtime inflation (×)")
axes[1].legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
out = os.path.join(HERE, "onset_both.png")
fig.savefig(out, dpi=200, facecolor="white")
print("saved", out)
