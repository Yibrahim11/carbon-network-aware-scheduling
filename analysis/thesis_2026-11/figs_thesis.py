"""All thesis figures from the stored runs and profile tables (dcsim 0.3.0). Rerun after the final experiments.

    python analytic.py        # first: writes analytic_*.csv used by Figure 3.5
    python figs_thesis.py [OUTDIR]     # writes fig*.pdf (vector) and fig*.png; default OUTDIR = this folder

Colours: P0 neutral grey (reference), P2 blue, P3 orange, P4 aqua; every policy also has its own marker or hatch,
so no figure relies on colour alone and all figures stay readable in greyscale print.
"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "code_v0.3.0"))
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


# ---------- Figure 3.2: carbon-intensity scenarios, one week ----------
seed = setup("aligned")[0]
fig, ax = plt.subplots(figsize=(6.3, 2.6))
h = np.arange(24 * 7)
series = [("High variability (synthetic)", dcsim.make_ci("synthetic_high_var", 0, 264, seed), "#2a78d6", "-"),
          ("Low variability (synthetic)", dcsim.make_ci("synthetic_low_var", 0, 264, seed), "#eb6834", "--"),
          ("Ghana-derived", dcsim.make_ci(GHANA, 0, 264, seed), "#1baf7a", "-.")]
for lab, v, c, ls in series:
    V = dcsim.ci_variability(v[:264])["daily_range_over_mean"]
    ax.plot(h / 24, v[:len(h)], color=c, lw=1.6, ls=ls, label=f"{lab}, $V$ = {V:.2f}")
ax.set_xlim(0, 7); ax.set_xticks(range(8)); ax.set_xlabel("day of the simulated week")
ax.set_ylabel("carbon intensity (gCO$_2$e/kWh)"); tidy(ax)
ax.set_ylim(0, 480); ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
save(fig, "fig3_2_ci_scenarios")

# ---------- Figure 3.3: measured iteration times by placement ----------
pa = pd.read_csv(os.path.join(ROOT, "data", "astra_profiles_aligned.csv"))
pi = pd.read_csv(os.path.join(ROOT, "data", "astra_profiles_interleaved.csv"))
cfg = [("vision", 8), ("vision", 16), ("llm_mid", 8), ("llm_mid", 16), ("llm_mid", 32), ("llm_large", 32), ("llm_large", 64)]
lbl = {"vision": "Vision", "llm_mid": "Mid LLM", "llm_large": "Large LLM"}
get = lambda d, jt, n, pc: float(d[(d.jtype == jt) & (d.n_gpus == n) & (d.pclass == pc)].iter_time_s.iloc[0])
vals = {k: [] for k in ("compact", "aligned", "interleaved")}
for jt, n in cfg:
    vals["compact"].append(get(pa, jt, n, "compact")); vals["aligned"].append(get(pa, jt, n, "spread")); vals["interleaved"].append(get(pi, jt, n, "spread"))
fig, ax = plt.subplots(figsize=(6.3, 2.9)); x = np.arange(len(cfg)); w = 0.26
for i, (k, c, ht) in enumerate((("compact", "#6b6a66", ""), ("aligned", "#2a78d6", ""), ("interleaved", "#eb6834", "////"))):
    ax.bar(x + (i - 1) * w, vals[k], w * 0.9, color=c, hatch=ht, edgecolor="white", lw=0.5,
           label={"compact": "Compact (one pod)", "aligned": "Aligned split (best case)", "interleaved": "Interleaved split (worst case)"}[k])
for xi, (c, ii) in enumerate(zip(vals["compact"], vals["interleaved"])):
    ax.text(xi + w, ii * 1.08, f"{ii / c:.1f}×", ha="center", fontsize=7, color="#0b0b0b")
ax.set_yscale("log"); ax.set_ylim(0.3, 50)
ax.set_xticks(x); ax.set_xticklabels([f"{lbl[jt]}\n{n} GPUs" for jt, n in cfg])
ax.set_ylabel("iteration time (s, log scale)"); tidy(ax); ax.legend(frameon=False, loc="upper left")
save(fig, "fig3_3_iteration_times")

# ---------- Figure 3.4: GPU power model, T4 shape check ----------
gp = pd.read_csv(os.path.join(ROOT, "data", "gpu_power.csv"))
p_idle, p_max = gp[gp.frac_compute == 0].power_w.mean(), gp[gp.frac_compute == 1].power_w.mean()
fig, ax = plt.subplots(figsize=(4.2, 2.8))
for wl, mk, c in (("fp16_matmul", "o", "#2a78d6"), ("fp32_matmul", "s", "#eb6834"), ("membound", "^", "#1baf7a")):
    g = gp[gp.workload == wl]; ax.plot(g.frac_compute, g.power_w, mk, color=c, ms=6, mec="white", mew=0.6,
                                         label={"fp16_matmul": "FP16 matrix multiply", "fp32_matmul": "FP32 matrix multiply", "membound": "memory-bound kernel"}[wl])
xx = np.linspace(0, 1, 50); ax.plot(xx, p_idle + (p_max - p_idle) * xx, color=INK2, lw=1.2, ls="--", label="endpoint line (as in the simulator)")
a_ls, b_ls = np.polyfit(gp.frac_compute, gp.power_w, 1)
ax.plot(xx, a_ls * xx + b_ls, color="#0b0b0b", lw=1.2, label="least-squares line")
mid = gp[(gp.frac_compute > 0) & (gp.frac_compute < 1)]
endpoint_err = float(np.mean((p_idle + (p_max - p_idle) * mid.frac_compute) / mid.power_w - 1) * 100)
loo = []
for wl in gp.workload.unique():
    tr, te = gp[gp.workload != wl], gp[gp.workload == wl]
    a, b = np.polyfit(tr.frac_compute, tr.power_w, 1)
    loo.append(float(np.mean(np.abs((a * te.frac_compute + b) / te.power_w - 1))) * 100)
ax.set_xlabel("compute share of each iteration, $f_c$"); ax.set_ylabel("T4 board power (W)"); tidy(ax, "both")
ax.legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "fig3_4_power_model")
json.dump({"T4_leave_one_workload_out_mean_abs_err_pct": [round(v, 1) for v in loo], "endpoint_line_mean_err_interior_pct": round(endpoint_err, 1),
           "endpoint_line_max_under_pct": round(float(((p_idle + (p_max - p_idle) * mid.frac_compute) / mid.power_w - 1).min() * 100), 1)},
          open(os.path.join(HERE, "t4_loo.json"), "w"))

# ---------- Figure 3.5: deferral bound against grid variability ----------
B = pd.read_csv(os.path.join(HERE, "analytic_bound.csv"))
fig, ax = plt.subplots(figsize=(5.6, 3.0))
VV = np.linspace(0, 1.2, 200)
for dh, ls in ((1.5, "-"), (6.0, ":")):
    ax.plot(VV, 100 * 0.7 * VV / 2 * np.sinc(dh / 24), color=INK2, lw=1.1, ls=ls,
            label=f"closed-form bound, sinusoidal grid, {dh:g} h jobs, 70% deferrable")
g = B.groupby("ci").agg(V=("V", "mean"), oracle=("oracle", "mean"), oracle_thr=("oracle_thr", "mean")).reset_index()
ax.plot(g.V, g.oracle, "*", color="#0b0b0b", ms=10, label="capacity-free oracle, actual job streams", zorder=4)
for v, pol, mk, c in (("aligned", "P2", "s", "#2a78d6"), ("interleaved", "P2", "^", "#eb6834")):
    s = B[B.bound == v].groupby("ci").agg(V=("V", "mean"), p2=("p2", "mean")).reset_index()
    ax.plot(s.V, s.p2, mk, color=c, ms=7, mec="white", mew=0.6, label=f"P2 simulated, {v} bound", zorder=5)
GE = os.path.join(HERE, "ghana_ec_check.csv")
if os.path.exists(GE):
    ge = pd.read_csv(GE); ge = ge[ge.series.str.contains("ec2025")]
    for v, mk, c in (("aligned", "s", "#2a78d6"), ("interleaved", "^", "#eb6834")):
        q = ge[ge.bound == v].groupby("series").agg(V=("V", "mean"), p2=("p2", "mean"))
        ax.plot(q.V, q.p2, mk, ms=6, mfc="white", mec=c, mew=1.3, label=f"P2 simulated, {v}, Ghana 2025 (national statistics)", zorder=5)
ax.axhline(2, color="#d03b3b", lw=0.9, ls="--"); ax.text(1.19, 3.0, r"$\delta_C = 2\%$", ha="right", fontsize=8, color="#0b0b0b")
hv = g[g.ci == "high variability"].iloc[0]
ax.annotate("high variability", (hv.V, hv.oracle), xytext=(-70, 8), textcoords="offset points", fontsize=7, color=INK2)
lo = g[g.ci != "high variability"]
ax.annotate("low variability and\npreliminary Ghana series", (lo.V.mean(), lo.oracle.mean()), xytext=(30, 40), textcoords="offset points",
            fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlim(0, 1.22); ax.set_ylim(0, 45)
ax.set_xlabel("grid variability $V$ (mean daily range / mean)"); ax.set_ylabel("dynamic-carbon saving (%)")
tidy(ax, "both"); ax.legend(frameon=False, fontsize=7, loc="upper left")
save(fig, "fig3_5_deferral_bound")

# ---------- Chapter IV figures (matched operating point) ----------
M = pd.read_csv(MATCHED)
M = M[(M.util == 0.3) & (M.comm_scale == 1.0) & (M.ci == "synthetic_high_var")]


def paired(d, a, b, met):
    x = d.pivot_table(index="rep", columns="policy", values=met)
    inc = d.pivot_table(index="rep", columns="policy", values="n_incomplete")
    keep = (inc[a] == 0) & (inc[b] == 0)
    return tci(100 * (x.loc[keep, a] / x.loc[keep, b] - 1))


# Figure 4.1: change against P0 on four metrics, both bounds
mets = [("carbon_dyn_kg", "(a) Dynamic carbon"), ("carbon_per_gpuh_g", "(b) Carbon per completed GPU-hour"),
        ("jct_mean_h", "(c) Mean completion time"), ("net_inflation_total", "(d) Total network inflation")]
fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0)); axes = axes.ravel()
for ax, (met, title) in zip(axes, mets):
    for bi, v in enumerate(("aligned", "interleaved")):
        d = M[M.variant == v]
        for pi_, pol in enumerate(("P2", "P3", "P4")):
            m, h_ = paired(d, pol, "P0", met); xpos = bi + (pi_ - 1) * 0.27
            ax.bar(xpos, m, 0.24, color=COL[pol], hatch=HATCH[pol], edgecolor="white", lw=0.5, label=NAME[pol] if (bi == 0 and ax is axes[0]) else None)
            ax.errorbar(xpos, m, yerr=h_, color="#0b0b0b", lw=0.8, capsize=2)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["aligned", "interleaved"])
    ax.set_title(title, loc="left", fontsize=8.5); tidy(ax)
for a_ in (axes[0], axes[2]): a_.set_ylabel("change against P0 (%)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig4_1_matched_point")

# Figure 4.2: carbon against completion time, every replication
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=False)
for ax, v, title in ((axes[0], "aligned", "(a) Aligned placement (best case)"), (axes[1], "interleaved", "(b) Interleaved placement (worst case)")):
    d = M[M.variant == v]
    for pol in ("P0", "P2", "P3", "P4"):
        g = d[d.policy == pol]
        ax.plot(g.jct_mean_h, g.carbon_per_gpuh_g, MARK[pol], color=COL[pol], ms=4, alpha=0.45, mew=0)
        ax.plot(g.jct_mean_h.mean(), g.carbon_per_gpuh_g.mean(), MARK[pol], color=COL[pol], ms=9, mec="white", mew=0.8, label=NAME[pol])
        lab = "P0, P3" if (v == "aligned" and pol == "P0") else (None if (v == "aligned" and pol == "P3") else pol)
        if lab:
            ax.annotate(lab, (g.jct_mean_h.mean(), g.carbon_per_gpuh_g.mean()), xytext=(8, 6), textcoords="offset points", fontsize=8)
    ax.set_title(title, loc="left"); ax.set_xlabel("mean job completion time (h)"); tidy(ax, "both")
axes[0].set_ylabel("carbon per completed GPU-hour (g)")
axes[1].legend(frameon=False, loc="upper right", fontsize=7)
fig.tight_layout(); save(fig, "fig4_2_tradeoff")

# Figure 4.3: facility power and carbon intensity over three days, interleaved bound, replication 0
seed_i, prof_i, pp_i, cb_i = setup("interleaved")
sc = dcsim.Scenario(arrival="poisson", util=0.3, comm_scale=1.0, ci="synthetic_high_var")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 3.6), sharex=True, gridspec_kw=dict(height_ratios=(1, 1.6)))
ci = None
for pol, ls in (("P0", "-"), ("P4", "-")):
    m, sim = dcsim.run_one(sc, pol, 0, master_seed=seed_i, profile_df=prof_i, power_base=pp_i, cluster_base=cb_i, record_trace=True)
    tr = np.array(sim.trace); ci = sim.ci.a
    grid = np.arange(0, 72 * 3600, 900.0)
    idx = np.searchsorted(tr[:, 1], grid + 1e-9); idx = np.clip(idx, 0, len(tr) - 1)
    a2.step(grid / 3600, tr[idx, 2] / 1000, where="post", color=COL[pol], lw=1.3, label=NAME[pol])
a1.step(np.arange(73), ci[:73], where="post", color="#0b0b0b", lw=1.1); a1.set_ylabel("gCO$_2$e/kWh"); tidy(a1)
a1.set_title("(a) Carbon intensity, high-variability grid", loc="left")
a2.set_title("(b) Facility power, interleaved bound, replication 0", loc="left")
a2.set_ylabel("facility power (kW)"); a2.set_xlabel("hour of the simulated week"); a2.set_xlim(0, 72); a2.set_xticks(range(0, 73, 12))
tidy(a2); a2.legend(frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.08))
fig.tight_layout(); save(fig, "fig4_3_power_trace")

# Figure 4.4: network inflation against communication intensity (same data as the proposal's Figure 1)
e = pd.read_csv(os.path.join(ROOT, "analysis", "review_2026-11", "extra_runs.csv")); al = e[e.tag == "sweep30"]
il = pd.read_csv(os.path.join(ROOT, "results", "sens_full_meas6_interleaved.csv")); il = il[il.tag == "sens:comm_scale"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
for ax, d, title in ((axes[0], al, "(a) Aligned placement (best case)"), (axes[1], il, "(b) Interleaved placement (worst case)")):
    for pol in ("P0", "P2", "P3", "P4"):
        g = d[d.policy == pol].groupby("comm_scale").net_inflation_total
        mm, ss, nn = g.mean(), g.std(ddof=1), g.count()
        ax.errorbar(mm.index, mm.values, yerr=(stats.t.ppf(.975, nn - 1) * ss / np.sqrt(nn)).values, color=COL[pol], marker=MARK[pol],
                    ms=4.5, lw=1.4, capsize=2, elinewidth=0.8, label=NAME[pol])
    ax.set_title(title, loc="left"); ax.set_xlabel("communication intensity (× measured cost)"); tidy(ax, "both")
axes[0].set_ylabel("total network inflation (×)"); axes[1].legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout(); save(fig, "fig4_4_inflation_onset")

# Figure 4.5: pod-size experiment, P3 against P0 completion time
P = pd.read_csv(os.path.join(ROOT, "results", "podtrace_paired_full_meas6_interleaved.csv"))
P = P[(P.a == "P3") & (P.b == "P0") & (P.metric == "jct_mean_h")]
fig, ax = plt.subplots(figsize=(5.0, 2.7)); pods = [32, 16, 8]
for i, (u, cs) in enumerate(((0.2, 0.5), (0.2, 1.0), (0.3, 0.5), (0.3, 1.0))):
    q = P[(P.util == u) & (P.comm_scale == cs)].set_index("pod_gpus").loc[pods]
    xs = np.arange(3) + (i - 1.5) * 0.19
    c = ["#86b6ef", "#3987e5", "#256abf", "#104281"][i]
    ax.bar(xs, -q["diff"], 0.17, color=c, edgecolor="white", lw=0.5, label=f"{int(u * 100)}% load, {cs:g}× communication")
    ax.errorbar(xs, -q["diff"], yerr=q.ci95, color="#0b0b0b", lw=0.8, capsize=2, ls="none")
ax.axhline(5, color="#d03b3b", lw=0.9, ls="--"); ax.text(2.45, 5.4, r"$\delta_T$", fontsize=8, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels(["32-GPU pods\n(all splits avoidable)", "16-GPU pods", "8-GPU pods\n(all splits forced)"])
ax.text(2, 0.4, "0 in every cell:\nno split is avoidable", ha="center", fontsize=7, color=INK2)
ax.set_ylim(0, 14)
ax.set_ylabel("completion-time reduction,\nP3 against P0 (%)"); tidy(ax); ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
save(fig, "fig4_5_pod_size")

# Figure 4.6: decision boundary for the network component
D = pd.read_csv(os.path.join(ROOT, "analysis", "review_2026-11", "decision_points.csv"))
fig, ax = plt.subplots(figsize=(5.8, 3.0))
sty = {"synthetic aligned": ("#2a78d6", "s", "Synthetic mix, aligned"), "synthetic interleaved": ("#eb6834", "^", "Synthetic mix, interleaved"),
       "trace pod32": ("#1baf7a", "o", "Trace, 32-GPU pods"), "trace pod16": ("#eda100", "D", "Trace, 16-GPU pods"),
       "trace pod8": ("#6b6a66", "x", "Trace, 8-GPU pods (forced only)")}
for src, (c, mk, lab) in sty.items():
    g = D[D.src == src]; xg = np.where(g.A > 0, g.A, 3e-4)
    ax.scatter(xg, -g.P3_jct, color=c, marker=mk, s=28, label=lab, zorder=3, linewidths=1.2 if mk == "x" else 0.5, edgecolors="white" if mk != "x" else None)
ax.set_xscale("log"); ax.set_xlim(2e-4, 5)
ax.axhline(5, color="#d03b3b", lw=0.9, ls="--"); ax.axvspan(0.06, 0.09, color="#e6e5e1", zorder=0)
ax.text(0.075, 60, r"$A^{*}\in[0.06,\,0.09]$", fontsize=8, ha="center"); ax.text(2.4e-4, 6.0, r"$\delta_T = 5\%$", fontsize=8)
ax.set_xlabel("avoidable network inflation $A = I_{P0} - I_{P3}$ (log scale; $A = 0$ plotted at left edge)")
ax.set_ylabel("completion-time reduction,\nP3 against P0 (%)"); tidy(ax, "both"); ax.legend(frameon=False, fontsize=7, loc="upper left")
save(fig, "fig4_6_decision_boundary")
print("T4 leave-one-out errors (%):", [round(v, 1) for v in loo])
