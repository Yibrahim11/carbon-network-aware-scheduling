"""Final thesis figures from the final experiments (dcsim 0.4.0, persistence forecast headline).

    python final/final_analysis.py      # first
    python final/figs_final.py [OUTDIR]  # writes fig3_2, fig4_1-4_4, fig4_6, fig4_7, fig4_8 (PDF and PNG)

Figures 3.3, 3.4, 3.5 and 4.5 come from analysis/thesis_2026-11/figs_thesis.py (unchanged inputs).
"""
import os, sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_style.py")).read())
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figs")
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(ROOT)
O = os.path.join(HERE, "out")
R = pd.concat([pd.read_csv(os.path.join(HERE, f"results_{b}.csv"), low_memory=False).assign(bound=b) for b in ("aligned", "interleaved")])
GH = "csv:data_build/ci_ghana_ec2025_thermal_flat_a0.081.csv"


def paired(q, a, b, met, complete=True):
    x = q.pivot_table(index="rep", columns="policy", values=met)
    inc = q.pivot_table(index="rep", columns="policy", values="n_incomplete")
    keep = ((inc[a] == 0) & (inc[b] == 0)) if complete else pd.Series(True, index=x.index)
    return tci(100 * (x.loc[keep, a] / x.loc[keep, b] - 1))


# ---------- Figure 3.2: carbon-intensity scenarios ----------
seed = setup("aligned")[0]
fig, ax = plt.subplots(figsize=(6.3, 2.6)); h = np.arange(24 * 7)
for lab, key, c, ls in (("High variability (synthetic)", "synthetic_high_var", "#2a78d6", "-"),
                        ("Low variability (synthetic)", "synthetic_low_var", "#eb6834", "--"),
                        ("Ghana 2025 (modelled from national statistics)", GH, "#1baf7a", "-.")):
    v = dcsim.make_ci(key, 0, 265, seed)
    V = dcsim.ci_variability(v[:264])["daily_range_over_mean"]
    ax.plot(h / 24, v[:len(h)], color=c, lw=1.6, ls=ls, label=f"{lab}, $V$ = {V:.2f}")
ax.set_xlim(0, 7); ax.set_xticks(range(8)); ax.set_xlabel("day of the simulated week")
ax.set_ylabel("carbon intensity (gCO$_2$/kWh)"); tidy(ax); ax.set_ylim(0, 480)
ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
save(fig, "fig3_2_ci_scenarios")

# ---------- Figure 4.1: primary point, change against P0 ----------
M = R[R.tag == "primary"]
mets = [("carbon_dyn_kg", "(a) Dynamic carbon"), ("carbon_per_gpuh_g", "(b) Carbon per completed GPU-hour"),
        ("jct_mean_h", "(c) Mean completion time"), ("net_inflation_total", "(d) Total network inflation")]
fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0)); axes = axes.ravel()
for ax, (met, title) in zip(axes, mets):
    for bi, v in enumerate(("aligned", "interleaved")):
        d = M[M.bound == v]
        for pi_, pol in enumerate(("P2", "P3", "P4")):
            m, h_ = paired(d, pol, "P0", met); xpos = bi + (pi_ - 1) * 0.27
            ax.bar(xpos, m, 0.24, color=COL[pol], hatch=HATCH[pol], edgecolor="white", lw=0.5, label=NAME[pol] if (bi == 0 and ax is axes[0]) else None)
            ax.errorbar(xpos, m, yerr=h_, color="#0b0b0b", lw=0.8, capsize=2)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["aligned", "interleaved"])
    ax.set_title(title, loc="left", fontsize=8.5); tidy(ax)
for a_ in (axes[0], axes[2]): a_.set_ylabel("change against P0 (%)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig4_1_primary_point")

# ---------- Figure 4.2: carbon against completion time ----------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, v, title in ((axes[0], "aligned", "(a) Aligned placement (best case)"), (axes[1], "interleaved", "(b) Interleaved placement (worst case)")):
    d = M[M.bound == v]
    for pol in ("P0", "P2", "P3", "P4"):
        g = d[d.policy == pol]
        ax.plot(g.jct_mean_h, g.carbon_per_gpuh_g, MARK[pol], color=COL[pol], ms=4, alpha=0.45, mew=0)
        ax.plot(g.jct_mean_h.mean(), g.carbon_per_gpuh_g.mean(), MARK[pol], color=COL[pol], ms=9, mec="white", mew=0.8, label=NAME[pol])
        lab = "P0, P3" if (v == "aligned" and pol == "P0") else (None if (v == "aligned" and pol == "P3") else pol)
        if lab:
            ax.annotate(lab, (g.jct_mean_h.mean(), g.carbon_per_gpuh_g.mean()), xytext=(8, 6), textcoords="offset points", fontsize=8)
    ax.set_title(title, loc="left"); ax.set_xlabel("mean job completion time (h)"); tidy(ax, "both")
axes[0].set_ylabel("carbon per completed GPU-hour (g)"); axes[1].legend(frameon=False, loc="upper right", fontsize=7)
fig.tight_layout(); save(fig, "fig4_2_tradeoff")

# ---------- Figure 4.3: facility power over three days ----------
seed_i, prof_i, pp_i, cb_i = setup("interleaved")
sc = dcsim.Scenario(arrival="poisson", util=0.3, comm_scale=1.0, ci="synthetic_high_var", forecast="persistence")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 3.6), sharex=True, gridspec_kw=dict(height_ratios=(1, 1.6)))
for pol in ("P0", "P4"):
    m, sim = dcsim.run_one(sc, pol, 0, master_seed=seed_i, profile_df=prof_i, power_base=pp_i, cluster_base=cb_i, record_trace=True)
    tr = np.array(sim.trace); ci = sim.ci.a; cif = sim.ci_f.a
    grid = np.arange(0, 72 * 3600, 900.0)
    idx = np.clip(np.searchsorted(tr[:, 1], grid + 1e-9), 0, len(tr) - 1)
    a2.step(grid / 3600, tr[idx, 2] / 1000, where="post", color=COL[pol], lw=1.3, label=NAME[pol])
a1.step(np.arange(73), ci[:73], where="post", color="#0b0b0b", lw=1.1, label="actual")
a1.step(np.arange(73), cif[:73], where="post", color=INK2, lw=1.0, ls=":", label="persistence forecast")
a1.set_ylabel("gCO$_2$/kWh"); tidy(a1); a1.legend(frameon=False, fontsize=7, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.0))
a1.set_title("(a) Carbon intensity, high-variability grid", loc="left")
a2.set_title("(b) Facility power, interleaved bound, replication 0", loc="left")
a2.set_ylabel("facility power (kW)"); a2.set_xlabel("hour of the simulated week"); a2.set_xlim(0, 72); a2.set_xticks(range(0, 73, 12))
tidy(a2); a2.legend(frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.08))
fig.tight_layout(); save(fig, "fig4_3_power_trace")

# ---------- Figure 4.4: network inflation against communication intensity ----------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
for ax, v, title in ((axes[0], "aligned", "(a) Aligned placement (best case)"), (axes[1], "interleaved", "(b) Interleaved placement (worst case)")):
    d = R[(R.bound == v) & (R.tag == "sens:comm_scale") & (R.n_incomplete == 0)]
    for pol in ("P0", "P2", "P3", "P4"):
        g = d[d.policy == pol].groupby("comm_scale").net_inflation_total
        mm, ss, nn = g.mean(), g.std(ddof=1), g.count()
        ax.errorbar(mm.index, mm.values, yerr=(stats.t.ppf(.975, nn - 1) * ss / np.sqrt(nn)).values, color=COL[pol], marker=MARK[pol],
                    ms=4.5, lw=1.4, capsize=2, elinewidth=0.8, label=NAME[pol])
    ax.set_title(title, loc="left"); ax.set_xlabel("communication intensity (× measured cost)"); tidy(ax, "both")
axes[0].set_ylabel("total network inflation (×)"); axes[1].legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout(); save(fig, "fig4_4_inflation_onset")

# ---------- Figure 4.6: decision boundary for the network component ----------
D = pd.read_csv(os.path.join(O, "decision_points.csv"))
fig, ax = plt.subplots(figsize=(5.8, 3.0))
sty = {"synthetic aligned": ("#2a78d6", "s", "Synthetic mix, aligned"), "synthetic interleaved": ("#eb6834", "^", "Synthetic mix, interleaved"),
       "trace pod32": ("#1baf7a", "o", "Trace, 32-GPU pods"), "trace pod16": ("#eda100", "D", "Trace, 16-GPU pods"),
       "trace pod8": ("#6b6a66", "x", "Trace, 8-GPU pods (forced splits only)")}
for src, (c, mk, lab) in sty.items():
    g = D[D.src == src]; xg = np.where(g.A > 3e-4, g.A, 3e-4)
    ax.scatter(xg, -g.P3_jct, color=c, marker=mk, s=28, label=lab, zorder=3, linewidths=1.2 if mk == "x" else 0.5,
               edgecolors="white" if mk != "x" else None)
ax.set_xscale("log"); ax.set_xlim(2e-4, 5)
ax.axhline(5, color="#d03b3b", lw=0.9, ls="--"); ax.axvspan(0.058, 0.091, color="#e6e5e1", zorder=0)
ax.text(0.073, 48, r"$A^{*}$", fontsize=9, ha="center"); ax.text(2.4e-4, 6.5, r"$\delta_T = 5\%$", fontsize=8)
ax.set_xlabel("avoidable network inflation $A = I_{P0} - I_{P3}$ (log scale; $A \\leq 3\\times10^{-4}$ at left edge)")
ax.set_ylabel("completion-time reduction,\nP3 against P0 (%)"); tidy(ax, "both"); ax.legend(frameon=False, fontsize=7, loc="upper left")
save(fig, "fig4_6_decision_boundary")

# ---------- Figure 4.7: carbon boundary from the variability sweep ----------
VS = pd.read_csv(os.path.join(O, "vsweep.csv")); GHT = pd.read_csv(os.path.join(O, "ghana.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
for ax, b, title in ((axes[0], "aligned", "(a) Aligned placement"), (axes[1], "interleaved", "(b) Interleaved placement")):
    for fc, c, mk, ls in (("perfect", "#86b6ef", "o", "--"), ("persistence", "#2a78d6", "s", "-")):
        g = VS[(VS.bound == b) & (VS.forecast == fc)].sort_values("V")
        ax.fill_between(g.V, g.P2_lo, g.P2_hi, color=c, alpha=0.2, lw=0)
        ax.plot(g.V, g.P2_dyn, ls, marker=mk, color=c, ms=4, lw=1.3, label=f"P2, {fc} forecast")
    gg = GHT[(GHT.bound == b) & GHT.series.str.contains("ec2025")]
    ax.plot(gg.V, gg.P2_dyn, "D", ms=5, mfc="white", mec="#1baf7a", mew=1.2, label="P2, Ghana 2025 series (persistence)")
    hd = gg[gg.series.str.contains("thermal_flat_a0.081")]
    ax.plot(hd.V, hd.P2_dyn, "D", ms=6, color="#1baf7a", label="Ghana 2025 headline")
    ax.axhline(2, color="#d03b3b", lw=0.9, ls="--"); ax.text(1.12, 3.2, r"$\delta_C$", fontsize=8, ha="right")
    ax.set_xscale("log"); ax.set_xlim(0.06, 1.25); ax.set_xticks([0.1, 0.2, 0.5, 1.0]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1.0"])
    ax.set_title(title, loc="left"); ax.set_xlabel("grid variability $V$ (log scale)"); tidy(ax, "both")
axes[0].set_ylabel("dynamic-carbon saving, P2 against P0 (%)"); axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); save(fig, "fig4_7_carbon_boundary")

# ---------- Figure 4.8: power-curve shape ----------
PS = pd.read_csv(os.path.join(O, "power_shape.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
for ax, met, title in ((axes[0], "energy", "(a) Total energy"), (axes[1], "dyn", "(b) Dynamic carbon")):
    for pol in ("P2", "P3", "P4"):
        for model, ls, mfc in (("linear", "--", "white"), ("gamma=0.58", "-", None)):
            g = PS[(PS.bound == "interleaved") & (PS.cmp == f"{pol} vs P0") & (PS.model == model)].sort_values("comm_scale")
            ax.plot(g.comm_scale, g[met], ls, marker=MARK[pol], color=COL[pol], ms=5, mfc=mfc or COL[pol], lw=1.2,
                    label=f"{pol}, {'linear' if model == 'linear' else 'concave'} power")
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks([0.5, 1, 2]); ax.set_xlabel("communication intensity (× measured cost)")
    ax.set_title(title, loc="left"); tidy(ax, "both")
axes[0].set_ylabel("change against P0 (%), interleaved bound")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, "fig4_8_power_shape")
