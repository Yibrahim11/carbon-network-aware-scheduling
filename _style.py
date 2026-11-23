"""All thesis figures from the stored runs and profile tables (dcsim 0.3.0). Rerun after the final experiments.

    python analytic.py        # first: writes analytic_*.csv used by Figure 3.5
    python figs_thesis.py [OUTDIR]     # writes fig*.pdf (vector) and fig*.png; default OUTDIR = this folder

Colours: P0 neutral grey (reference), P2 blue, P3 orange, P4 aqua; every policy also has its own marker or hatch,
so no figure relies on colour alone and all figures stay readable in greyscale print.
"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code_v0.4.0"))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
import dcsim

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42})
INK2, GRID = "#52514e", "#e6e5e1"
COL = {"P0": "#6b6a66", "P2": "#2a78d6", "P3": "#eb6834", "P4": "#1baf7a"}
MARK = {"P0": "o", "P2": "s", "P3": "^", "P4": "D"}
HATCH = {"P2": "", "P3": "////", "P4": "...."}
NAME = {"P0": "P0 baseline", "P2": "P2 carbon-aware", "P3": "P3 network-aware", "P4": "P4 joint"}
GHANA = "csv:" + os.path.join(ROOT, "data", "ci_ghana_derived.csv")
MATCHED = os.path.join(ROOT, "analysis", "review_2026-11", "matched_runs_ramp.csv")


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, name + ".png"), dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def setup(variant):
    params = json.load(open(os.path.join(ROOT, "results", f"params_full_meas6_{variant}.json")))
    prof = pd.read_csv(os.path.join(ROOT, "data", f"astra_profiles_{variant}.csv"))
    pp = dcsim.PowerParams(); pp.P_gpu_idle, pp.P_gpu_max = params["power"]["P_gpu_idle"], params["power"]["P_gpu_max"]
    cb = dcsim.ClusterParams(k_rule=params["cluster"]["k_rule"], comm_scale_iter=params["cluster"]["comm_scale_iter"])
    return params["meta"]["seed"], prof, pp, cb


def tci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean(), (stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


