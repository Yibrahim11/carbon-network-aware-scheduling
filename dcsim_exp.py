"""
Experiment layer for dcsim: designed experiments (not wandering), checkpointed parallel runner, paired CRN analysis
with t-intervals, an exhaustive-search benchmark for the P4 heuristic, and plots.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import stats

import dcsim
from dcsim import HOUR, Scenario, ClusterParams, PowerParams, Job, Sim

SC_COLS = list(asdict(Scenario()).keys())
KEY_COLS = ["tag"] + SC_COLS
POLICIES_MAIN = ["P0", "P2", "P3", "P4"]
POLICIES_SENS = ["P0", "P2", "P2raw", "P3", "P4", "P4nc"]
METRICS = ["carbon_dyn_kg", "carbon_kg", "energy_dyn_kWh", "jct_mean_h", "jct_p95_h", "wait_mean_h",
           "deadline_viol_frac", "peak_kW", "net_inflation", "spread_frac", "ci_at_start_gpuw",
           "util_time_avg", "max_spread_running", "cap_exceed_h", "gpuh_frac_k3plus", "net_inflation_total"]


# ----------------------------------------------------------------------------------------------
# Designs
# ----------------------------------------------------------------------------------------------

COMM_MAIN, COMM_SWEEP, BASE_COMM = (1.0, 8.0), [1.0, 2.0, 4.0, 6.0, 8.0, 12.0], 6.0   # synthetic-profile defaults


UTIL_MAIN, UTIL_SWEEP, BASE_UTIL = (0.5, 0.8), [0.4, 0.6, 0.7, 0.8, 0.9], 0.7


def set_stress_levels(on: bool):
    """Worst-case (interleaved) profiles make spread jobs 1.4-8.5x slower, so the nominal utilisation (defined on COMPACT
    run times) is far above real capacity and runs cannot drain. Stress mode lowers the load and communication levels so the
    cluster is loaded but not collapsed; the interleaved profile is then reported as a bound, not as the baseline."""
    global UTIL_MAIN, UTIL_SWEEP, BASE_UTIL, COMM_MAIN, COMM_SWEEP, BASE_COMM
    if on:
        UTIL_MAIN, UTIL_SWEEP, BASE_UTIL = (0.2, 0.3), [0.1, 0.2, 0.3, 0.4, 0.5], 0.3
        COMM_MAIN, COMM_SWEEP, BASE_COMM = (0.5, 1.0), [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 0.5
    else:
        UTIL_MAIN, UTIL_SWEEP, BASE_UTIL = (0.5, 0.8), [0.4, 0.6, 0.7, 0.8, 0.9], 0.7


def set_comm_levels(measured_profile: bool):
    """comm_scale multiplies the communication volume AND time of the profile table. With the synthetic placeholder,
    1.0 is a gentle level and the network effect appears around 6-8. With a MEASURED (ASTRA-sim) table, 1.0 is already the
    measured collective cost, so useful levels are 0.5-4 (a larger model / gradient) and 8x would be unrealistic."""
    global COMM_MAIN, COMM_SWEEP, BASE_COMM
    if measured_profile:
        COMM_MAIN, COMM_SWEEP, BASE_COMM = (1.0, 3.0), [0.5, 1.0, 1.5, 2.0, 3.0, 4.0], 1.0
    else:
        COMM_MAIN, COMM_SWEEP, BASE_COMM = (1.0, 8.0), [1.0, 2.0, 4.0, 6.0, 8.0, 12.0], 6.0


def main_design(R=10, arrivals=("poisson", "bursty"), utils=None, comms=None,
                cis=("synthetic_low_var", "synthetic_high_var"), policies=POLICIES_MAIN, **fixed):
    """Full factorial over a few levels per factor, replicated R times (common random numbers across policies)."""
    comms = comms or COMM_MAIN
    utils = utils or UTIL_MAIN
    rows = []
    for ar, u, cs, ci in itertools.product(arrivals, utils, comms, cis):
        sc = dict(arrival=ar, util=u, comm_scale=cs, ci=ci, **fixed)
        for pol in policies:
            for r in range(R):
                rows.append(dict(tag="main", scenario=sc, policy=pol, rep=r))
    return rows


def base_scenario(**kw):
    d = dict(arrival="poisson", util=BASE_UTIL, comm_scale=BASE_COMM, ci="synthetic_high_var")
    d.update(kw)
    return d


def trace_design(trace_csv, R=10, utils=None, cis=("synthetic_high_var",), comms=None,
                 policies=POLICIES_MAIN):
    """Trace-driven arrivals (replication = different window of the trace). Communication intensity is ASSIGNED."""
    comms = comms or COMM_MAIN
    utils = utils or UTIL_MAIN
    rows = []
    for u, ci, cs in itertools.product(utils, cis, comms):
        sc = dict(trace=trace_csv, util=u, ci=ci, comm_scale=cs)
        for pol in policies:
            for r in range(R):
                rows.append(dict(tag="trace", scenario=sc, policy=pol, rep=r))
    return rows


def sensitivity_design(R=10, policies=POLICIES_SENS, ci_extra=None):
    """One-factor-at-a-time sweeps around a base cell. OAT cannot see interactions: the main factorial covers those."""
    sweeps = {
        "comm_scale": list(COMM_SWEEP),
        "util": list(UTIL_SWEEP),
        "cap_frac": [None, 0.8, 0.7, 0.62],
        "forecast_sigma": [0.0, 0.1, 0.3],
        "fabric": ["ocs", "electrical"],
        "slack": [(1.0, 4.0), (2.0, 12.0), (4.0, 24.0)],
        "w_delay": [0.0, 0.02, 0.1],
        "deferrable_frac": [0.3, 0.7, 1.0],
        "arrival": ["poisson", "bursty", "diurnal"],
    }
    if ci_extra:
        sweeps["ci"] = ["synthetic_low_var", "synthetic_high_var"] + list(ci_extra)
    rows = []
    for factor, levels in sweeps.items():
        for lv in levels:
            sc = base_scenario()
            if factor == "slack":
                sc["slack_lo"], sc["slack_hi"] = lv
            else:
                sc[factor] = lv
            for pol in policies:
                for r in range(R):
                    rows.append(dict(tag=f"sens:{factor}", scenario=sc, policy=pol, rep=r))
    return rows


def _key(item):
    full = asdict(Scenario(**item["scenario"]))
    return json.dumps([item["tag"], full, item["policy"], item["rep"]], sort_keys=True, default=str)


def _work(args):
    item, profile_df, master_seed, cluster_base, power_base = args
    sc = Scenario(**item["scenario"])
    try:
        m = dcsim.run_one(sc, item["policy"], item["rep"], master_seed=master_seed, profile_df=profile_df,
                          cluster_base=cluster_base, power_base=power_base)
        m["tag"] = item["tag"]
        m["key"] = _key(item)
        m["error"] = ""
    except Exception as e:                      # keep the sweep alive, record the failure
        m = dict(tag=item["tag"], key=_key(item), policy=item["policy"], rep=item["rep"],
                 error=f"{type(e).__name__}: {e}", **asdict(Scenario(**item["scenario"])))
    return m


def run_design(design, out_csv, workers=None, profile_df=None, master_seed=dcsim.MASTER_SEED_DEFAULT,
               cluster_base=None, power_base=None, batch=200, verbose=True):
    """Checkpointed: results append to out_csv; re-running skips finished keys (survives Colab disconnects)."""
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv, usecols=["key"])["key"])
    todo = [it for it in design if _key(it) not in done]
    if verbose:
        print(f"{len(design)} runs in design, {len(design) - len(todo)} already done, {len(todo)} to run")
    workers = workers or max(1, (os.cpu_count() or 1))
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        args = [(it, profile_df, master_seed, cluster_base, power_base) for it in chunk]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(_work, args, chunksize=4))
        else:
            res = [_work(a) for a in args]
        df = pd.DataFrame(res)
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        if verbose:
            el = time.time() - t0
            print(f"  {min(i + batch, len(todo))}/{len(todo)} done, {el:.0f}s elapsed")
    return out_csv


# ----------------------------------------------------------------------------------------------
# Analysis (replications + t-intervals; paired differences under common random numbers)
# ----------------------------------------------------------------------------------------------

def load_runs(csv):
    df = pd.read_csv(csv)
    if "error" in df and (df["error"].fillna("") != "").any():
        print("WARNING: failed runs:", int((df["error"].fillna("") != "").sum()))
    df = df[df["error"].fillna("") == ""].copy()
    df["cap_frac"] = df["cap_frac"].astype(float).fillna(-1.0)     # -1 = no power cap
    if "trace" in df:
        df["trace"] = df["trace"].fillna("").astype(str)             # '' = synthetic arrivals (NaN would break joins)
    return df


def _cell_cols(df):
    return [c for c in KEY_COLS if c in df.columns]


def tci(x, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n))


def summarize(df, metrics=("carbon_dyn_kg", "jct_mean_h", "deadline_viol_frac", "net_inflation", "spread_frac"),
              by=None):
    by = by or (_cell_cols(df) + ["policy"])
    rows = []
    for key, g in df.groupby(by, dropna=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n"] = len(g)
        row["censored_runs"] = int((g["n_incomplete"] > 0).sum())
        for m in metrics:
            row[m] = g[m].mean()
            row[m + "_ci"] = tci(g[m])
        rows.append(row)
    return pd.DataFrame(rows)


def paired(df, a, b, metric="carbon_dyn_kg", relative=True, by=None, drop_censored=True):
    """Paired difference a - b over replications with identical job streams (CRN). Returns per-cell mean,
    95% t-interval half-width, whether the interval excludes zero, and how many replications were censored
    (runs where some job did not finish inside the accounting window are dropped from paired comparisons)."""
    by = by or _cell_cols(df)
    keep = by + ["rep", "policy", metric, "n_incomplete", "workload_hash"]
    d = df[df.policy.isin([a, b])][keep]
    pa = d[d.policy == a].set_index(by + ["rep"])
    pb = d[d.policy == b].set_index(by + ["rep"])
    j = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner")
    assert (j["workload_hash_a"] == j["workload_hash_b"]).all(), "CRN violated: job streams differ between policies"
    cens = (j["n_incomplete_a"] > 0) | (j["n_incomplete_b"] > 0)
    j["censored"] = cens
    if drop_censored:
        jj = j[~cens]
    else:
        jj = j
    diff = jj[metric + "_a"] - jj[metric + "_b"]
    if relative:
        diff = 100.0 * diff / jj[metric + "_b"].abs().replace(0, np.nan)
    out = diff.groupby(level=list(range(len(by)))).agg(["mean", "count"])
    out["ci95"] = diff.groupby(level=list(range(len(by)))).apply(lambda x: tci(x.values))
    out["censored_reps"] = j["censored"].groupby(level=list(range(len(by)))).sum()
    out = out.rename(columns={"mean": "diff", "count": "n"}).reset_index()
    out["excludes_zero"] = (out["diff"].abs() > out["ci95"]) & out["ci95"].notna()
    out["a"], out["b"], out["metric"], out["unit"] = a, b, metric, "%" if relative else "abs"
    return out


# ----------------------------------------------------------------------------------------------
# Exhaustive-search benchmark (small instance): how far is the P4 heuristic from the best achievable?
# ----------------------------------------------------------------------------------------------

def tiny_instance(comm_scale=12.0, ci_scenario="synthetic_high_var", rep=0):
    cp, pp = ClusterParams(n_pods=2, gpus_per_pod=32), PowerParams()
    cp.horizon_h, cp.window_h = 12.0, 60.0
    prof_df = dcsim.synthetic_profile(cp, comm_scale=comm_scale)
    prof = dcsim.profile_dict(prof_df)
    spec = [("llm_mid", 16, 0.2, 1.5), ("llm_mid", 16, 0.5, 1.2), ("llm_large", 32, 1.0, 1.0),
            ("llm_mid", 16, 1.5, 1.4), ("vision", 8, 2.0, 1.0)]
    jobs = []
    for i, (jt, n, arr_h, dur_h) in enumerate(spec):
        it = prof[(jt, n, "compact" if n <= cp.gpus_per_pod else "spread")][0]
        jobs.append(Job(i, arr_h * HOUR, jt, n, dur_h * HOUR / it, True, arr_h * HOUR + dur_h * HOUR * 8.0, dur_h * HOUR))
    ci = dcsim.make_ci(ci_scenario, rep, int(cp.window_h) + 1)
    return jobs, prof, cp, pp, ci


def objective(sim, jobs, w_delay, ci_ref):
    """Same scale as the P4 cost: dynamic carbon + w_delay * delay price."""
    pp = sim.pp
    carbon_dyn_g = sim.carbon_g - sim.idle_floor_W() * float(sim.ci.integral(0.0, sim.W)) / dcsim.J_PER_KWH
    price_kW = pp.PUE * pp.P_gpu_max / 1000.0
    delay = sum(j.n_gpus * price_kW * ci_ref * (j.finish - j.arrival) / HOUR for j in jobs if j.finish is not None)
    unfinished = sum(1 for j in jobs if j.finish is None)
    viol = sum(1 for j in jobs if j.finish is not None and j.finish > j.deadline) + unfinished
    return carbon_dyn_g + w_delay * delay, carbon_dyn_g, viol, unfinished


def _fresh_jobs(jobs):
    return [Job(j.jid, j.arrival, j.jtype, j.n_gpus, j.n_iters, j.deferrable, j.deadline, j.dur_ref_s) for j in jobs]


def oracle_benchmark(comm_scale=12.0, ci_scenario="synthetic_high_var", w_delay=0.02, max_delay_h=14,
                     restarts=300, seed=1, verbose=True):
    """Best-known-solution benchmark on a 5-job instance. Decision per job = (start hour, compact|spread).
    Search = coordinate descent from random restarts and from each heuristic's own decisions. This is NOT a proof of
    optimality (the space is too large to enumerate); it is a strong reference ("best known") against which the
    heuristics' gap is reported. The heuristics are included in the best-known set so the gap is never negative."""
    jobs0, prof, cp, pp, ci = tiny_instance(comm_scale, ci_scenario)
    ci_ref = float(np.mean(ci[:24]))

    def run(dec_or_policy, jobs):
        pol = dec_or_policy if not isinstance(dec_or_policy, dict) else dcsim.ForcedPolicy(dec_or_policy)
        sim = Sim(jobs, prof, ClusterParams(**asdict(cp)), pp, ci, pol)
        sim.run()
        return objective(sim, jobs, w_delay, ci_ref)

    res, seeds = {}, []
    for name in ["P0", "P2", "P3", "P4"]:
        jobs = _fresh_jobs(jobs0)
        pol = dcsim.P4(w_delay=w_delay) if name == "P4" else dcsim.POLICIES[name]()
        J, c, v, u = run(pol, jobs)
        res[name] = dict(J=J, carbon_dyn_kg=c / 1000, violations=v, unfinished=u)
        seeds.append({j.jid: (j.start if j.start is not None else j.arrival, "compact" if j.k == 1 else "spread")
                      for j in jobs})
    options = []
    for j in jobs0:
        base = math.ceil(j.arrival / HOUR) * HOUR
        o = [(base + d * HOUR, m) for d in range(max_delay_h + 1) for m in ("compact", "spread")
             if base + d * HOUR <= j.deadline - j.dur_ref_s * 1.2]
        options.append(o)

    def evalJ(dec):
        J, c, v, u = run(dec, _fresh_jobs(jobs0))
        return (J if (v == 0 and u == 0) else J + 1e9), c, v, u

    rng = dcsim.rng_for(seed, 0, "misc")
    starts = [dict(s) for s in seeds]
    for _ in range(restarts):
        starts.append({j.jid: options[i][int(rng.integers(len(options[i])))] for i, j in enumerate(jobs0)})
    best_J, best_dec, best_c, n_eval = math.inf, None, None, 0
    for dec in starts:
        cur = dict(dec)
        curJ = evalJ(cur)[0]
        n_eval += 1
        improved = True
        while improved:
            improved = False
            for i, j in enumerate(jobs0):
                for opt in options[i]:
                    if opt == cur[j.jid]:
                        continue
                    trial = dict(cur)
                    trial[j.jid] = opt
                    Jt = evalJ(trial)[0]
                    n_eval += 1
                    if Jt < curJ - 1e-9:
                        cur, curJ, improved = trial, Jt, True
        if curJ < best_J:
            best_J, best_dec = curJ, cur
    Jb, cb, vb, ub = evalJ(best_dec)
    ok_heur = {k: v["J"] for k, v in res.items() if v["violations"] == 0 and v["unfinished"] == 0}
    ref = min([best_J] + list(ok_heur.values()))
    res["BEST_KNOWN"] = dict(J=ref, carbon_dyn_kg=cb / 1000 if ref == best_J else float("nan"),
                             violations=vb if ref == best_J else 0, unfinished=ub if ref == best_J else 0)
    df = pd.DataFrame(res).T
    df["gap_vs_best_known_%"] = 100 * (df["J"] - ref) / ref
    if verbose:
        print(f"coordinate-descent search: {len(starts)} starts, {n_eval} simulations; "
              f"local-search best J={best_J:.0f}")
    return df, best_dec


# ----------------------------------------------------------------------------------------------
# Plots (palette: reference categorical slots 1-4, validated; policies keep their colour everywhere)
# ----------------------------------------------------------------------------------------------

POLICY_COLOR = {"P0": "#6b6a66", "P2": "#2a78d6", "P3": "#eb6834", "P4": "#1baf7a", "P4nc": "#eda100", "P2raw": "#8a63c9"}
POLICY_MARK = {"P0": "o", "P2": "s", "P3": "^", "P4": "D", "P4nc": "v", "P2raw": "P"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b8b7b2")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_onset(df_sens, factor="comm_scale", metric="carbon_dyn_kg", ylabel=None, outfile=None):
    """Policy lines against one swept factor, with 95% intervals; direct labels at line ends + legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df_sens[df_sens.tag == f"sens:{factor}"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for pol, g in d.groupby("policy"):
        s = g.groupby(factor)[metric].agg(["mean", lambda x: tci(x)]).reset_index()
        s.columns = [factor, "mean", "ci"]
        ax.errorbar(s[factor], s["mean"], yerr=s["ci"], color=POLICY_COLOR[pol], marker=POLICY_MARK[pol], ms=6,
                    lw=1.8, capsize=0, elinewidth=0.8, label=pol, mec="#fcfcfb", mew=1.5)
    ax.set_xlabel(factor)
    ax.set_ylabel(ylabel or metric)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=160, facecolor="#fcfcfb")
    return fig


def plot_paired_bars(tab, title, outfile=None, xcol="label"):
    """Horizontal grouped bars of paired % differences vs P0 with 95% CI whiskers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(dict.fromkeys(tab[xcol]))
    pols = [p for p in ["P2", "P3", "P4"] if p in set(tab["a"])]
    fig, ax = plt.subplots(figsize=(8.5, min(14, max(4, 0.17 * len(labels) * len(pols) + 1.5))))
    w = 0.8 / len(pols)
    for i, p in enumerate(pols):
        g = tab[tab.a == p].set_index(xcol).reindex(labels)
        ys = np.arange(len(labels)) + (i - (len(pols) - 1) / 2) * w
        ax.barh(ys, g["diff"], height=w - 0.04, color=POLICY_COLOR[p], label=p, xerr=g["ci95"],
                error_kw=dict(ecolor=INK2, lw=0.8, capsize=0))
    ax.axvline(0, color="#52514e", lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right")
    _style(ax)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=160, facecolor="#fcfcfb")
    return fig


def plot_tradeoff(tab_carbon, tab_jct, outfile=None):
    """Each point = one scenario cell: x = % change in mean JCT vs P0, y = % change in dynamic carbon vs P0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for p in ["P2", "P3", "P4"]:
        a = tab_carbon[tab_carbon.a == p].set_index("label")
        b = tab_jct[tab_jct.a == p].set_index("label").reindex(a.index)
        ax.scatter(b["diff"], a["diff"], s=34, color=POLICY_COLOR[p], marker=POLICY_MARK[p], label=p,
                   edgecolor="#fcfcfb", linewidth=1.2, alpha=0.95)
    ax.axhline(0, color="#52514e", lw=1)
    ax.axvline(0, color="#52514e", lw=1)
    ax.set_xlabel("change in mean JCT vs P0 (%)  → slower")
    ax.set_ylabel("change in dynamic carbon vs P0 (%)  ↓ cleaner")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=160, facecolor="#fcfcfb")
    return fig


def cell_label(df, cols=("arrival", "util", "comm_scale", "ci")):
    short = {"synthetic_low_var": "lowCI", "synthetic_high_var": "highCI"}
    def _ci(v):
        return short.get(v, os.path.basename(str(v)[4:]) if str(v).startswith("csv:") else v)
    return df.apply(lambda r: " / ".join(
        f"{_ci(r[c])}" if c == "ci" else (f"u{r[c]}" if c == "util" else (f"c{r[c]:g}" if c == "comm_scale" else str(r[c])))
        for c in cols), axis=1)


# ----------------------------------------------------------------------------------------------
# Reporting: tables, figures, and a compact text digest to paste back for review
# ----------------------------------------------------------------------------------------------

def sens_label(r):
    f = str(r["tag"]).split(":")[1] if ":" in str(r["tag"]) else str(r["tag"])
    if f == "slack":
        return f"slack={r['slack_lo']:g}-{r['slack_hi']:g}"
    v = r[f]
    if f == "cap_frac":
        v = "none" if v < 0 else v
    return f"{f}={v}"


def _pair_cols(df, a, b, metric, relative, label_fn):
    t = paired(df, a, b, metric, relative)
    t["label"] = t.apply(label_fn, axis=1) if len(t) else []
    t[f"{a}-{b}:{metric}"] = t["diff"].round(1).astype(str) + np.where(t["excludes_zero"], "*", "")
    return t


def make_tables(main, sens, trace=None):
    """Returns dict of DataFrames: paired tables for the main design and the sensitivity design."""
    out = {}
    comps = [("P2", "P0"), ("P3", "P0"), ("P4", "P0"), ("P4", "P2")]
    mets = [("carbon_dyn_kg", True), ("jct_mean_h", True), ("deadline_viol_frac", False), ("net_inflation", False),
            ("peak_kW", True)]
    def trace_label(r):
        return f"trace u{r['util']} c{r['comm_scale']:g} {os.path.basename(str(r['ci'])[4:]) if str(r['ci']).startswith('csv:') else str(r['ci']).replace('synthetic_', '')}"
    for name, df, lab in (("main", main, lambda r: cell_label(r.to_frame().T).iloc[0]), ("sens", sens, sens_label),
                          ("trace", trace, trace_label)):
        if df is None or df.empty:
            continue
        parts = []
        for a, b in comps + ([("P4", "P4nc"), ("P2", "P2raw")] if name == "sens" else []):
            if not (set([a, b]) <= set(df.policy)):
                continue
            for met, rel in mets:
                t = paired(df, a, b, met, rel)
                if len(t):
                    t["label"] = t.apply(lab, axis=1)
                    parts.append(t)
        out[name] = pd.concat(parts, ignore_index=True)
    return out


def save_figures(main, sens, tabs, outdir):
    os.makedirs(outdir, exist_ok=True)
    files = []
    if sens is not None and (sens.tag == "sens:comm_scale").any():
        f = os.path.join(outdir, "fig_onset_net_inflation.png")
        plot_onset(sens, "comm_scale", "net_inflation", "network runtime inflation (×, GPU-weighted)", f); files.append(f)
        f = os.path.join(outdir, "fig_onset_carbon.png")
        plot_onset(sens, "comm_scale", "carbon_dyn_kg", "dynamic carbon (kg CO2e per week)", f); files.append(f)
    if "main" in tabs:
        t = tabs["main"]
        tc = t[(t.metric == "carbon_dyn_kg") & (t.b == "P0")]
        tj = t[(t.metric == "jct_mean_h") & (t.b == "P0")]
        f = os.path.join(outdir, "fig_main_carbon_vs_p0.png")
        plot_paired_bars(tc, "dynamic carbon change vs P0 (%, paired, 95% CI)", f); files.append(f)
        f = os.path.join(outdir, "fig_main_tradeoff.png")
        plot_tradeoff(tc, tj, f); files.append(f)
    return files


def digest(main, sens, oracle_df, verify_rep, meta, tabs=None, trace=None):
    tabs = tabs or make_tables(main, sens, trace)
    L = []
    L.append("=== RESULT DIGEST (copy everything between the lines) ===")
    L.append(f"code {meta.get('code_version')} | seed {meta.get('seed')} | mode {meta.get('mode')} | "
             f"reps {meta.get('R')} | profile source: {meta.get('profile_source')} | ci scenarios: {meta.get('ci_list')}")
    L.append(f"power model: {meta.get('power_source')}")
    if meta.get("k_extension"):
        L.append(f"jobs over k>2 pods: {meta.get('k_extension')}")
    if "synthetic" in str(meta.get("profile_source")):
        L.append("!! PROFILE = SYNTHETIC PLACEHOLDER and power/fabric numbers are ILLUSTRATIVE: results below test the "
                 "pipeline and show structure; they are NOT findings until ASTRA-sim profiles + calibrated power are used.")
    nf = sum(r["status"] == "FAIL" for r in verify_rep); nw = sum(r["status"] == "WARN" for r in verify_rep)
    L.append(f"verification: {len(verify_rep)} checks, {len(verify_rep) - nf - nw} PASS, {nw} WARN, {nf} FAIL")
    for r in verify_rep:
        if r["status"] != "PASS":
            L.append(f"  [{r['status']}] {r['name']}: {r['detail']}")
    for nm, df in (("main", main), ("sens", sens)):
        if df is not None:
            L.append(f"{nm}: {len(df)} runs, censored (unfinished jobs) {(df.n_incomplete > 0).sum()}, "
                     f"mean run time {df.runtime_s.mean():.2f}s")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)

    def wide(t, metric, a, b, col):
        x = t[(t.metric == metric) & (t.a == a) & (t.b == b)].set_index("label")
        return x["diff"].round(1).astype(str) + np.where(x["excludes_zero"], "*", "")

    if "main" in tabs:
        t = tabs["main"]
        cols = {"carbon P2vsP0 %": wide(t, "carbon_dyn_kg", "P2", "P0", ""), "carbon P3vsP0 %": wide(t, "carbon_dyn_kg", "P3", "P0", ""),
                "carbon P4vsP0 %": wide(t, "carbon_dyn_kg", "P4", "P0", ""), "carbon P4vsP2 %": wide(t, "carbon_dyn_kg", "P4", "P2", ""),
                "JCT P2vsP0 %": wide(t, "jct_mean_h", "P2", "P0", ""), "JCT P4vsP2 %": wide(t, "jct_mean_h", "P4", "P2", ""),
                "viol P4vsP2 abs": wide(t, "deadline_viol_frac", "P4", "P2", "")}
        T = pd.DataFrame(cols)
        L.append("\nMAIN FACTORIAL: paired % differences (CRN, 95% t-interval; * = interval excludes 0). "
                 "Cell = arrival / util / comm_scale / CI scenario")
        L.append(T.to_string())
        ni = main.groupby(["comm_scale", "policy"]).net_inflation.mean().unstack().round(3)
        L.append("\nmean network runtime inflation by comm_scale (main design):")
        L.append(ni.to_string())
        if "completion_rate" in main:      # audit: unfinished jobs must stay visible next to carbon/JCT comparisons
            cr = main.groupby(["comm_scale", "policy"]).completion_rate.mean().unstack().round(3)
            L.append("\ncompletion rate (share of jobs finished inside the window) by comm_scale:"); L.append(cr.to_string())
            cg = main.groupby(["comm_scale", "policy"]).carbon_per_gpuh_g.mean().unstack().round(1)
            L.append("\ntotal carbon per COMPLETED GPU-hour (g) by comm_scale (fair when policies finish different amounts of work):"); L.append(cg.to_string())
        if "net_inflation_total" in main:
            ti = main.groupby(["comm_scale", "policy"]).net_inflation_total.mean().unstack().round(3)
            L.append("\nmean TOTAL network inflation by comm_scale (placement + congestion; baseline = the job in one pod):"); L.append(ti.to_string())
        if "gpuh_frac_k3plus" in main:     # exposure to the k>2 extension: how much of the work the measured/extended rows govern
            kx = main.groupby(["comm_scale", "policy"]).gpuh_frac_k3plus.mean().unstack().round(3)
            L.append("\nshare of completed GPU-hours in jobs spread over 3 or more pods, by comm_scale:"); L.append(kx.to_string())
    if "sens" in tabs:
        t = tabs["sens"]
        keep = t[t.metric.isin(["carbon_dyn_kg", "jct_mean_h", "net_inflation"])]
        rows = {}
        for (a, b, met), g in keep.groupby(["a", "b", "metric"]):
            rows[f"{a}-{b} {met}"] = g.set_index("label").apply(
                lambda r: f"{r['diff']:.1f}" + ("*" if r["excludes_zero"] else "") + (f" (c{int(r['censored_reps'])})" if r["censored_reps"] else ""), axis=1)
        T = pd.DataFrame(rows)
        sel = [c for c in ["P2-P0 carbon_dyn_kg", "P3-P0 jct_mean_h", "P4-P2 carbon_dyn_kg", "P4-P2 jct_mean_h",
                           "P4-P0 carbon_dyn_kg", "P4-P4nc carbon_dyn_kg", "P2-P2raw jct_mean_h"] if c in T.columns]
        L.append("\nSENSITIVITY (one factor at a time around the base cell; (cN) = N replications censored and dropped):")
        L.append(T[sel].to_string())
        if (sens.tag == "sens:comm_scale").any():
            d = sens[sens.tag == "sens:comm_scale"].groupby(["comm_scale", "policy"]).net_inflation.mean().unstack().round(3)
            L.append("\nONSET: mean network inflation vs comm_scale (sens design; congestion only, baseline = the job's own placement):")
            L.append(d.to_string())
            if "net_inflation_total" in sens:
                d2 = sens[sens.tag == "sens:comm_scale"].groupby(["comm_scale", "policy"]).net_inflation_total.mean().unstack().round(3)
                L.append("\nONSET: mean TOTAL network inflation vs comm_scale (placement + congestion, baseline = the job in one pod):")
                L.append(d2.to_string())
    if "trace" in tabs:
        t = tabs["trace"]
        T = pd.DataFrame({"carbon P2vsP0 %": wide(t, "carbon_dyn_kg", "P2", "P0", ""), "carbon P4vsP0 %": wide(t, "carbon_dyn_kg", "P4", "P0", ""),
                          "carbon P4vsP2 %": wide(t, "carbon_dyn_kg", "P4", "P2", ""), "JCT P4vsP2 %": wide(t, "jct_mean_h", "P4", "P2", "")})
        L.append("\nTRACE-DRIVEN (replication = trace window; communication intensity ASSIGNED, not measured):")
        L.append(T.to_string())
    if oracle_df is not None:
        L.append("\nBEST-KNOWN BENCHMARK on a 5-job instance (local search + heuristics; not a proof of optimality):")
        L.append(oracle_df.round(2).to_string())
    L.append("=== END DIGEST ===")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------
# Pod-size experiment: same 128 GPUs and same uplink per GPU, smaller pods, so that real trace jobs span pods
# ----------------------------------------------------------------------------------------------

def podsize_cluster(pod: int, total_gpus: int = 128) -> ClusterParams:
    """Cluster of total_gpus GPUs in pods of `pod` GPUs. Uplink and transceivers per GPU are held constant (pod=32 -> 50 GB/s, 8 ports)."""
    assert total_gpus % pod == 0
    return ClusterParams(n_pods=total_gpus // pod, gpus_per_pod=pod, ports_per_pod=max(1, 8 * pod // 32),
                         uplink_scale=pod / 32.0)


def pod_trace_summary(df):
    """Per (utilisation, communication level, CI): policy means and paired differences vs P0 and P4 vs P2 (CRN, censored reps dropped)."""
    by = ["util", "comm_scale", "ci"]
    rows = []
    for metric in [m for m in ("net_inflation", "net_inflation_total", "carbon_dyn_kg", "carbon_per_gpuh_g", "jct_mean_h",
                               "peak_kW", "completion_rate") if m in df.columns]:
        for a, b in (("P2", "P0"), ("P3", "P0"), ("P4", "P0"), ("P4", "P2")):
            p = paired(df, a, b, metric=metric, by=by)
            rows.append(p[by + ["diff", "ci95", "n", "censored_reps", "excludes_zero", "a", "b", "metric"]])
    out = pd.concat(rows, ignore_index=True)
    cols = [c for c in ("net_inflation", "net_inflation_total", "spread_frac", "gpuh_frac_k3plus", "completion_rate") if c in df.columns]
    means = df.groupby(by + ["policy"])[cols].mean().reset_index()
    return out, means