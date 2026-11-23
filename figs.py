import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.dirname(_os.path.dirname(HERE))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
PK = f"{ROOT}/results"
COLOR = {"P0": "#6b6a66", "P2": "#2a78d6", "P3": "#eb6834", "P4": "#1baf7a"}; MARK = {"P0": "o", "P2": "s", "P3": "^", "P4": "D"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"], "font.size": 9})
# ---- Figure 1: total network inflation vs communication intensity, both bounds at 30% load ----
e = pd.read_csv(f"{HERE}/extra_runs.csv"); al = e[e.tag == "sweep30"]
il = pd.read_csv(f"{PK}/sens_full_meas6_interleaved.csv"); il = il[il.tag == "sens:comm_scale"]
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)
for ax, d, title in ((axes[0], al, "(a) Aligned placement (best case)"), (axes[1], il, "(b) Interleaved placement (worst case)")):
    for pol in COLOR:
        g = d[d.policy == pol].groupby("comm_scale").net_inflation_total; m, s, n = g.mean(), g.std(ddof=1), g.count()
        ax.errorbar(m.index, m.values, yerr=(stats.t.ppf(.975, n - 1) * s / np.sqrt(n)).values, color=COLOR[pol], marker=MARK[pol], ms=4, lw=1.3, capsize=2, elinewidth=.8, label=pol)
    ax.set_title(title, fontsize=9); ax.set_xlabel("communication intensity (× measured cost)"); ax.grid(color="#e6e5e1", lw=.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("total network runtime inflation (×)"); axes[1].legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(f"{HERE}/onset_both.png", dpi=200, facecolor="white"); plt.close(fig)
# ---- Figure 2: decision-rule boundary for the network component ----
D = pd.read_csv(f"{HERE}/decision_points.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.3))
sty = {"synthetic aligned": ("#2a78d6", "s", "Synthetic mix, aligned"), "synthetic interleaved": ("#eb6834", "^", "Synthetic mix, interleaved"),
       "trace pod32": ("#1baf7a", "o", "Production trace, 32-GPU pods"), "trace pod16": ("#7a5cc4", "D", "Production trace, 16-GPU pods"),
       "trace pod8": ("#6b6a66", "x", "Production trace, 8-GPU pods (forced)")}
for src, (c, mk, lab) in sty.items():
    g = D[D.src == src]; x = np.where(g.A > 0, g.A, 3e-4)
    ax.scatter(x, -g.P3_jct, color=c, marker=mk, s=26, label=lab, zorder=3)
ax.set_xscale("log"); ax.set_xlim(2e-4, 5)
ax.axhline(5, color="#444", lw=.8, ls="--"); ax.axvline(0.09, color="#444", lw=.8, ls=":")
ax.text(0.1, 14.0, "$A^{*}\\approx0.09$", fontsize=8); ax.text(2.4e-4, 6.0, "$\\delta_T = 5\\%$", fontsize=8)
ax.set_xlabel("avoidable network inflation $A = I_{P0} - I_{P3}$ (log scale; forced-only cells at left edge)")
ax.set_ylabel("completion-time reduction, P3 vs P0 (%)")
ax.grid(color="#e6e5e1", lw=.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(f"{HERE}/decision_boundary.png", dpi=200, facecolor="white")
print("ok")
