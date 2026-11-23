"""
dcsim: cross-layer simulator for carbon-, power- and network-aware scheduling of AI training jobs.

Study type: TERMINATING discrete-event simulation (cluster starts empty, fixed accounting window).
Layers:
  1. cluster layer  : job arrivals, queue (FIFO + EASY backfill), GPU pods, policies P0/P2/P3/P4
  2. job layer      : per-job profile table (iter_time, compute fraction, inter-pod GB per iteration).
                      Source = ASTRA-sim output (preferred) or a synthetic analytic profile (placeholder).
  3. network layer  : fluid pod-uplink congestion with a calibratable slowdown curve (ns-3 hook)
  4. energy layer   : phase-dependent GPU power, idle GPUs, servers, network ports, PUE, carbon = energy x CI(t)

ALL numeric defaults marked ILLUSTRATIVE are placeholders, not measurements. They must be calibrated
and reported before any result is treated as a finding.
"""
from __future__ import annotations

import functools
import hashlib
import heapq
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

__version__ = "0.4.0"
HOUR = 3600.0
J_PER_KWH = 3.6e6
MASTER_SEED_DEFAULT = 20261004

# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------


@dataclass
class PowerParams:
    """ILLUSTRATIVE placeholders. Calibrate P_gpu_* against measurements (reference [2])."""
    P_gpu_idle: float = 100.0         # W per GPU when idle or waiting on communication
    P_gpu_max: float = 650.0          # W per GPU in a pure compute phase
    P_server: float = 800.0           # W per server host overhead (always on)
    P_port: float = 10.0              # W per active uplink port / transceiver
    P_net_static_pod: float = 400.0   # W per pod switch (always on)
    P_core_static: float = 400.0      # W core fabric (OCS or spine, always on)
    PUE: float = 1.3                  # constant facility overhead (assumption, state and justify)
    gamma: float = 1.0                # GPU power shape: P = P_idle + (P_max - P_idle) * f_c**gamma (1 = linear; <1 concave)


@dataclass
class ClusterParams:
    """ILLUSTRATIVE placeholders for a small training cluster (default 128 GPUs)."""
    n_pods: int = 4
    gpus_per_pod: int = 32
    gpus_per_server: int = 8
    fabric: str = "ocs"                 # 'ocs' | 'electrical' (see FABRICS)
    uplink_GBps: float = 50.0           # aggregate per-pod uplink capacity (GB/s)
    intra_GBps: float = 300.0           # effective intra-pod all-reduce bandwidth (GB/s)
    reconf_delay_s: float = 0.05        # OCS circuit set-up added when a job spans pods
    ports_per_pod: int = 8              # transceivers per pod uplink
    uplink_scale: float = 1.0           # multiplies the fabric preset's per-pod uplink (keeps uplink per GPU constant when pod size changes)
    sleep_idle_ports: bool = True       # uplink ports of pods with no inter-pod traffic power down
    # How a profile measured on 2 pods is extended to a job spread over k > 2 pods, when no measured row for that k exists.
    # 'hier'             hierarchical ring all-reduce, per-pod volume x 2(k-1)/k relative to k=2 (v0.1.0 behaviour; kept so
    #                    stored results reproduce). NOT consistent with the flat-ring ns-3 profiles.
    # 'ring_interleaved' flat ring, ranks round-robin over pods (every hop crosses): per-pod volume x 2/k
    # 'ring_aligned'     flat ring, ranks in contiguous blocks (one crossing hop per pod): per-pod volume x 1
    # Measured rows (pclass 'spread_k3', 'spread_k4', ...) always take precedence; the rule extends from the largest measured k.
    k_rule: str = "hier"
    # How Scenario.comm_scale (communication intensity, x measured cost) is applied to a measured profile table.
    # False (v0.1.0-0.2.1 behaviour, kept so stored results reproduce): scales inter-pod volume and the congestible communication time only; the uncongested
    #       iteration time keeps the measured (1x) communication, so the placement penalty does not follow comm_scale.
    # True: also rescales the communication inside the uncongested iteration time (it = t_comp + comm, no overlap, as
    #       in the ns-3 profile construction), so the whole communication cost follows comm_scale. Default from 0.3.0.
    comm_scale_iter: bool = True
    # Congestion curve: (offered load / capacity) -> slowdown of the exposed communication time.
    # Default = fluid processor sharing. REPLACE with a curve fitted from ns-3 runs.
    congestion_pts: tuple = ((0.0, 1.0), (1.0, 1.0), (8.0, 8.0))
    power_cap_W: Optional[float] = None  # facility power cap (admission control), None = no cap
    tick_s: float = HOUR                 # scheduler re-evaluation period
    horizon_h: float = 168.0             # arrivals are generated in [0, horizon]
    window_h: float = 264.0              # accounting window (horizon + drain)

    @property
    def n_gpus(self) -> int:
        return self.n_pods * self.gpus_per_pod

    @property
    def n_servers(self) -> int:
        return self.n_gpus // self.gpus_per_server


# Fabric presets: the "optical" layer is an ABSTRACTION (capacity, port power, set-up delay),
# not a physical-layer or wavelength-level model.  ILLUSTRATIVE numbers.
FABRICS = {
    "ocs":        dict(uplink_GBps=50.0, reconf_delay_s=0.05, P_port=8.0,  P_core_static=300.0),
    "electrical": dict(uplink_GBps=50.0, reconf_delay_s=0.0,  P_port=15.0, P_core_static=900.0),
}


def apply_fabric(cluster: ClusterParams, power: PowerParams, name: str):
    f = FABRICS[name]
    cluster.fabric = name
    cluster.uplink_GBps = f["uplink_GBps"]
    cluster.reconf_delay_s = f["reconf_delay_s"]
    power.P_port = f["P_port"]
    power.P_core_static = f["P_core_static"]
    return cluster, power


# ----------------------------------------------------------------------------------------------
# Job types and profile table
# ----------------------------------------------------------------------------------------------

# ILLUSTRATIVE job mix. G_GB = gradient volume per iteration; t_comp = compute time per iteration.
JOB_TYPES = {
    "llm_large": dict(weight=0.15, gpus=(32, 64), gprob=(0.6, 0.4), G_GB=14.0, t_comp=2.0, overlap=0.5,
                      med_h=2.5, sigma=0.8),
    "llm_mid":   dict(weight=0.35, gpus=(8, 16, 32), gprob=(0.3, 0.4, 0.3), G_GB=4.0, t_comp=1.0, overlap=0.5,
                      med_h=1.5, sigma=0.9),
    "vision":    dict(weight=0.50, gpus=(8, 16), gprob=(0.6, 0.4), G_GB=0.4, t_comp=0.5, overlap=0.7,
                      med_h=1.0, sigma=0.8),
}


def kscale(k: int) -> float:
    """Per-pod uplink volume of a hierarchical ring all-reduce over k pods, relative to gradient size G."""
    return 2.0 * (k - 1) / k if k > 1 else 0.0


K_RULES = {
    "hier": lambda k: kscale(k) / kscale(2),
    "ring_interleaved": lambda k: 2.0 / k,
    "ring_aligned": lambda k: 1.0,
}


def spread_pclass(k: int) -> str:
    """Profile class name for a job spanning k pods: 'spread' for k=2 (as in every existing table), 'spread_k<k>' above."""
    return "spread" if k == 2 else f"spread_k{k}"


def synthetic_profile(cluster: ClusterParams, job_types=JOB_TYPES, comm_scale: float = 1.0) -> pd.DataFrame:
    """PLACEHOLDER profile from an analytic alpha-beta style model (hierarchical ring all-reduce with
    partial overlap). Replace with ASTRA-sim measurements: same schema, source='astra-sim'.
    interpod_GB = GB crossing EACH involved pod uplink per iteration, evaluated at k=2 pods."""
    rows = []
    for jt, p in job_types.items():
        G = p["G_GB"] * comm_scale
        for n in p["gpus"]:
            for pclass in ("compact", "spread"):
                if pclass == "compact" and n > cluster.gpus_per_pod:
                    continue
                k = 1 if pclass == "compact" else 2
                n_loc = n if k == 1 else n / 2.0
                comm_intra = 2.0 * (n_loc - 1) / n_loc * G / cluster.intra_GBps
                comm_inter = kscale(k) * G / cluster.uplink_GBps if k > 1 else 0.0
                comm = comm_intra + comm_inter
                hidden = min(comm, p["overlap"] * p["t_comp"])
                exposed = comm - hidden
                it = p["t_comp"] + exposed
                rows.append(dict(jtype=jt, n_gpus=n, pclass=pclass, iter_time_s=it,
                                 frac_compute=p["t_comp"] / it,
                                 interpod_GB=(kscale(2) * G if k > 1 else 0.0),
                                 comm_time_s=comm_inter, source="synthetic-parametric"))
    return pd.DataFrame(rows)


def scale_profile(profile_df: pd.DataFrame, comm_scale: float, comm_scale_iter: bool = False) -> pd.DataFrame:
    """Apply communication intensity (x measured cost) to a measured profile table; see ClusterParams.comm_scale_iter."""
    profile_df = profile_df.copy()
    profile_df["interpod_GB"] = profile_df["interpod_GB"] * comm_scale
    if "comm_time_s" in profile_df:                       # communication time scales with volume
        comm0 = profile_df["comm_time_s"].copy()
        profile_df["comm_time_s"] = comm0 * comm_scale
        if comm_scale_iter:                               # whole communication cost follows comm_scale
            t_comp = profile_df["iter_time_s"] * profile_df["frac_compute"]
            profile_df["iter_time_s"] = profile_df["iter_time_s"] + comm0 * (comm_scale - 1.0)
            profile_df["frac_compute"] = t_comp / profile_df["iter_time_s"]
    return profile_df


def profile_dict(df: pd.DataFrame) -> dict:
    """Profile table -> {(jtype, n_gpus, pclass): (iter_time_s, frac_compute, interpod_GB, comm_time_s)}.
    comm_time_s = time per iteration of the communication that crosses the (congestible) pod uplinks at no
    congestion, hidden or exposed. Optional column; if absent it defaults to the EXPOSED communication time
    (iter_time*(1-frac_compute)), which is conservative-low when communication is mostly hidden."""
    has = "comm_time_s" in df.columns
    out = {}
    for r in df.itertuples(index=False):
        it, fc = float(r.iter_time_s), float(r.frac_compute)
        comm = float(r.comm_time_s) if has and not pd.isna(r.comm_time_s) else it * (1.0 - fc)
        out[(r.jtype, int(r.n_gpus), r.pclass)] = (it, fc, float(r.interpod_GB), comm)
    return out


_ASTRA_LINE = None


def parse_astra_output(text: str):
    """Parse ASTRA-sim stdout lines 'sys[i] finished, N cycles, exposed communication M cycles'.
    Returns (cycles array, exposed array) ordered by NPU id."""
    import re
    rows = sorted((int(a), int(b), int(c)) for a, b, c in
                  re.findall(r"sys\[(\d+)\] finished, (\d+) cycles, exposed communication (\d+) cycles", text))
    return np.array([r[1] for r in rows], float), np.array([r[2] for r in rows], float)


def profile_row_from_astra(jtype, n_gpus, pclass, cycles_per_npu, exposed_per_npu, n_iterations,
                           interpod_GB, ns_per_cycle=1.0, comm_time_s=None):
    """Convert ASTRA-sim per-NPU output ('finished, N cycles, exposed communication M cycles') to a profile row.
    ns_per_cycle: ASSUMED 1.0 (analytical backend time unit = ns). VERIFY against the ASTRA-sim docs.
    n_iterations: number of training iterations contained in the simulated Chakra trace.
    comm_time_s: optional total (hidden+exposed) inter-pod communication time per iteration, e.g. from an
    ASTRA-sim run of the communication alone; if None the model uses the exposed time (conservative-low)."""
    total = float(np.max(cycles_per_npu))
    exposed = float(np.mean(exposed_per_npu))
    it_s = total * ns_per_cycle * 1e-9 / n_iterations
    fc = max(0.0, min(1.0, 1.0 - exposed / total))
    return dict(jtype=jtype, n_gpus=int(n_gpus), pclass=pclass, iter_time_s=it_s, frac_compute=fc,
                interpod_GB=float(interpod_GB), comm_time_s=(float(comm_time_s) if comm_time_s is not None else float("nan")),
                source="astra-sim")


# ----------------------------------------------------------------------------------------------
# Randomness, carbon intensity, workloads
# ----------------------------------------------------------------------------------------------

STREAMS = {"arrival": 1, "attrs": 2, "ci": 3, "forecast": 4, "calib": 5, "misc": 6}


def rng_for(master_seed: int, rep: int, stream: str) -> np.random.Generator:
    """Independent, well-separated streams via SeedSequence (never hand-picked seeds 1,2,3)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([master_seed, rep, STREAMS[stream]])))


class CI:
    """Hourly step-function carbon intensity (gCO2/kWh) with exact interval integrals."""

    def __init__(self, arr):
        self.a = np.asarray(arr, float)
        self.n = len(self.a)
        self.cum = np.concatenate([[0.0], np.cumsum(self.a * HOUR)])

    def at(self, t: float) -> float:
        return float(self.a[min(int(t // HOUR), self.n - 1)])

    def _F(self, t):
        t = np.asarray(t, float)
        h = np.minimum((t // HOUR).astype(int), self.n - 1)
        return self.cum[h] + self.a[h] * (t - h * HOUR)

    def integral(self, t0, t1):
        """Integral of CI dt over [t0, t1] in (gCO2/kWh)*s. Extends the last value beyond the series."""
        return self._F(t1) - self._F(t0)

    def mean(self, t0, t1):
        return float(self.integral(t0, t1)) / max(t1 - t0, 1e-9)


def make_ci(scenario: str, rep: int, hours: int, master_seed: int = MASTER_SEED_DEFAULT) -> np.ndarray:
    """Carbon-intensity scenarios. The synthetic scenarios are STRUCTURAL stand-ins (low vs high variability),
    NOT any real grid. Use 'csv:<path>' (columns timestamp,gco2_per_kwh; hourly) for real data such as Ghana."""
    if scenario.startswith("csv:"):
        df = pd.read_csv(scenario[4:])
        col = [c for c in df.columns if c != df.columns[0]][0]
        v = df[col].astype(float).to_numpy()
        reps = int(np.ceil(hours / len(v)))
        return np.tile(v, reps)[:hours]
    rng = rng_for(master_seed, rep, "ci")
    h = np.arange(hours)
    hod = h % 24
    day = h // 24
    n_days = int(day.max()) + 1
    ar = np.zeros(n_days)
    for d in range(1, n_days):
        ar[d] = 0.6 * ar[d - 1] + rng.normal(0, 1)
    noise = np.zeros(hours)
    for i in range(1, hours):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 1)
    if scenario == "synthetic_low_var":
        base = 420.0
        v = base * (1 + 0.03 * np.cos(2 * np.pi * (hod - 19) / 24) + 0.01 * noise + 0.02 * ar[day])
    elif scenario == "synthetic_high_var":
        base = 330.0
        solar = np.exp(-0.5 * ((hod - 13) / 3.5) ** 2)
        evening = np.exp(-0.5 * ((hod - 19.5) / 2.0) ** 2)
        v = base * (1 - 0.65 * solar + 0.25 * evening + 0.04 * noise + 0.20 * ar[day] * (0.4 + solar))
    elif scenario.startswith("vfam:"):
        # Synthetic family with controlled daily variability V (0.4.0): the high-variability daily shape (midday solar
        # trough, evening peak) normalised to unit range and scaled so that its daily range is V times the mean,
        # plus small day-to-day and hourly noise. STRUCTURAL scenario, not any real grid. Realised V is reported per run.
        Vt = float(scenario.split(":")[1]); base = 380.0
        solar = np.exp(-0.5 * ((hod - 13) / 3.5) ** 2)
        evening = np.exp(-0.5 * ((hod - 19.5) / 2.0) ** 2)
        shp = -0.65 * solar + 0.25 * evening
        hz = np.arange(24); sh24 = -0.65 * np.exp(-0.5 * ((hz - 13) / 3.5) ** 2) + 0.25 * np.exp(-0.5 * ((hz - 19.5) / 2.0) ** 2)
        z = (shp - sh24.mean()) / (sh24.max() - sh24.min())
        v = base * (1 + Vt * z + 0.01 * noise + 0.02 * ar[day])
    else:
        raise ValueError(f"unknown CI scenario {scenario}")
    return np.maximum(v, 20.0)


def ci_variability(ci) -> dict:
    ci = pd.Series(np.asarray(ci, float))
    hod = ci.groupby(np.arange(len(ci)) % 24).mean()
    day = np.arange(len(ci)) // 24
    rng_day = ci.groupby(day).max() - ci.groupby(day).min()
    return dict(mean=float(ci.mean()), cv=float(ci.std() / ci.mean()), min=float(ci.min()), max=float(ci.max()),
                max_over_min=float(ci.max() / ci.min()), mean_daily_range=float(rng_day.mean()),
                hour_of_day_swing=float(hod.max() - hod.min()),
                daily_range_over_mean=float(rng_day.mean() / ci.mean()))


@dataclass
class Job:
    jid: int
    arrival: float
    jtype: str
    n_gpus: int
    n_iters: float
    deferrable: bool
    deadline: float
    dur_ref_s: float
    # runtime state
    start: Optional[float] = None
    finish: Optional[float] = None
    remaining: float = 0.0
    pods: Optional[dict] = None
    pods_list: Optional[list] = None
    k: int = 1
    iter_unc: float = 0.0
    fc_unc: float = 1.0
    t_comp: float = 0.0
    gb_k: float = 0.0
    demand: float = 0.0
    s: float = 1.0
    iter_eff: float = 0.0
    finish_est: float = 0.0
    version: int = 0
    target: Optional[float] = None
    unc_runtime: float = 0.0
    cmp_runtime: float = 0.0       # runtime had the job been placed in one pod, uncongested (baseline of net_inflation_total)
    comm_unc: float = 0.0
    ci_at_start: float = 0.0


def _ref_class(n, cluster):
    return "compact" if n <= cluster.gpus_per_pod else "spread"


def arrivals(model: str, lam: float, horizon_s: float, rng) -> np.ndarray:
    """Arrival times. Exponential gaps by inverse transform (-ln(1-U)/lam)."""
    out = []
    if model == "poisson":
        t = 0.0
        while True:
            t += -math.log1p(-rng.random()) / lam
            if t >= horizon_s:
                break
            out.append(t)
    elif model == "bursty":                 # 2-state MMPP, same long-run mean rate lam
        t_on, t_off, ratio = 2 * HOUR, 6 * HOUR, 0.1
        pi_on = t_on / (t_on + t_off)
        lam_on = lam / (pi_on + ratio * (1 - pi_on))
        lam_off = ratio * lam_on
        t, on = 0.0, rng.random() < pi_on
        while t < horizon_s:
            dur = -math.log1p(-rng.random()) * (t_on if on else t_off)
            end = min(t + dur, horizon_s)
            r = lam_on if on else lam_off
            u = t
            while True:
                u += -math.log1p(-rng.random()) / r
                if u >= end:
                    break
                out.append(u)
            t, on = end, not on
    elif model == "diurnal":                # NHPP by thinning, mean rate lam, peak at 14:00
        a = 0.8
        lmax = lam * (1 + a)
        t = 0.0
        while True:
            t += -math.log1p(-rng.random()) / lmax
            if t >= horizon_s:
                break
            lt = lam * (1 + a * math.cos(2 * math.pi * ((t / HOUR) % 24 - 14) / 24))
            if rng.random() < lt / lmax:
                out.append(t)
    else:
        raise ValueError(f"unknown arrival model {model}")
    return np.array(out)


def _draw_attrs(rng, job_types, cluster, prof, dur_lo=0.25, dur_hi=24.0):
    names = list(job_types)
    w = np.array([job_types[n]["weight"] for n in names])
    jt = names[int(rng.choice(len(names), p=w / w.sum()))]
    p = job_types[jt]
    n = int(p["gpus"][int(rng.choice(len(p["gpus"]), p=np.array(p["gprob"]) / np.sum(p["gprob"])))])
    dur_h = float(np.clip(math.exp(math.log(p["med_h"]) + p["sigma"] * rng.normal()), dur_lo, dur_hi))
    return jt, n, dur_h * HOUR



@functools.lru_cache(maxsize=16)
def expected_gpu_seconds(job_types_json: str, dur_lo=0.25, dur_hi=24.0, m=20001) -> float:
    """E[n_gpus x duration_s] per job, by deterministic quantile integration (no sampling noise)."""
    from scipy.stats import norm
    jt = json.loads(job_types_json)
    z = norm.ppf((np.arange(m) + 0.5) / m)
    wsum = sum(p["weight"] for p in jt.values())
    tot = 0.0
    for p in jt.values():
        dur = np.clip(np.exp(math.log(p["med_h"]) + p["sigma"] * z), dur_lo, dur_hi).mean() * HOUR
        en = sum(g * q for g, q in zip(p["gpus"], np.array(p["gprob"]) / np.sum(p["gprob"])))
        tot += p["weight"] / wsum * en * dur
    return float(tot)


def make_workload(rep: int, arrival_model: str, util: float, cluster: ClusterParams, profile: dict,
                  job_types=JOB_TYPES, master_seed: int = MASTER_SEED_DEFAULT, deferrable_frac: float = 0.7,
                  slack_range=(2.0, 12.0)):
    """Common-random-numbers workload: depends only on (master_seed, rep, arrival_model, util, params),
    never on the policy. Nominal utilisation = offered GPU-seconds / capacity at reference profile."""
    e_gpu_s = expected_gpu_seconds(json.dumps(job_types, sort_keys=True))
    lam = util * cluster.n_gpus / e_gpu_s
    horizon_s = cluster.horizon_h * HOUR
    rng_a = rng_for(master_seed, rep, "arrival")
    rng_j = rng_for(master_seed, rep, "attrs")
    times = arrivals(arrival_model, lam, horizon_s, rng_a)
    jobs = []
    for i, t in enumerate(times):
        jt, n, dur = _draw_attrs(rng_j, job_types, cluster, profile)
        deferrable = bool(rng_j.random() < deferrable_frac)
        slack = slack_range[0] + (slack_range[1] - slack_range[0]) * rng_j.random()
        it, fc, gb = profile[(jt, n, _ref_class(n, cluster))][:3]
        n_iters = dur / it
        deadline = t + dur * (1.0 + slack) if deferrable else t + dur * 1.5
        jobs.append(Job(jid=i, arrival=float(t), jtype=jt, n_gpus=n, n_iters=n_iters, deferrable=deferrable,
                        deadline=float(deadline), dur_ref_s=dur))
    return jobs, dict(lam_per_h=lam * HOUR, e_gpu_hours=e_gpu_s / HOUR)



def workload_from_trace(df, rep, util, cluster, profile, job_types=JOB_TYPES, master_seed=MASTER_SEED_DEFAULT,
                        cols=None, deferrable_frac=0.7, slack_range=(2.0, 12.0)):
    """Trace-driven workload (e.g. a production GPU-cluster trace). Replication `rep` = a different window of the trace.
    Consecutive jobs are taken from the window start until the offered GPU-seconds equal util x capacity x horizon, then
    arrival times are rescaled linearly to span the horizon (keeps the trace's burstiness shape).
    Traces carry no communication information: job type (hence communication intensity) is ASSIGNED at random per
    GPU count using the job-type weights. State this limitation wherever results are reported."""
    cols = cols or dict(arrival="arrival_s", n_gpus="n_gpus", duration="duration_s")
    d = df[[cols["arrival"], cols["n_gpus"], cols["duration"]]].dropna().sort_values(cols["arrival"]).reset_index(drop=True)
    d.columns = ["arrival", "n_gpus", "duration"]
    sizes = sorted({g for p in job_types.values() for g in p["gpus"]})
    d["n"] = d["n_gpus"].apply(lambda x: min(sizes, key=lambda s: abs(s - x)))
    d["dur"] = d["duration"].clip(0.25 * HOUR, 24 * HOUR)
    target = util * cluster.n_gpus * cluster.horizon_h * HOUR
    start = (rep * 7919) % max(len(d) - 1, 1)
    idx = np.concatenate([np.arange(start, len(d)), np.arange(0, start)])      # wrap around
    d["arr_u"] = d["arrival"]
    if start > 0:                       # unwrap: the wrapped head of the trace continues AFTER its tail
        gap = float(np.median(np.diff(d["arrival"].to_numpy()))) if len(d) > 2 else 1.0
        d.loc[:start - 1, "arr_u"] = d.loc[:start - 1, "arrival"] + (d["arrival"].iloc[-1] - d["arrival"].iloc[0]) + gap
    cum = np.cumsum((d.loc[idx, "n"] * d.loc[idx, "dur"]).to_numpy())
    m = int(np.searchsorted(cum, target)) + 1
    sel = d.loc[idx[:min(m, len(idx))]].copy()
    t0, t1 = sel["arr_u"].iloc[0], sel["arr_u"].iloc[-1]
    span = max(t1 - t0, 1.0)
    sel["t"] = (sel["arr_u"] - t0) / span * (cluster.horizon_h * HOUR * 0.999)
    rng = rng_for(master_seed, rep, "attrs")
    jobs = []
    for i, r in enumerate(sel.itertuples(index=False)):
        cands = [jt for jt, p in job_types.items() if r.n in p["gpus"]]
        w = np.array([job_types[c]["weight"] for c in cands])
        jt = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        deferrable = bool(rng.random() < deferrable_frac)
        slack = slack_range[0] + (slack_range[1] - slack_range[0]) * rng.random()
        it, fc, gb = profile[(jt, int(r.n), _ref_class(int(r.n), cluster))][:3]
        dur = float(r.dur)
        jobs.append(Job(jid=i, arrival=float(r.t), jtype=jt, n_gpus=int(r.n), n_iters=dur / it, deferrable=deferrable,
                        deadline=float(r.t + dur * (1.0 + slack)) if deferrable else float(r.t + dur * 1.5),
                        dur_ref_s=dur))
    return jobs, dict(lam_per_h=len(jobs) / cluster.horizon_h, e_gpu_hours=float((sel["n"] * sel["dur"]).mean() / HOUR))


def workload_hash(jobs) -> str:
    h = hashlib.sha256()
    for j in jobs:
        h.update(f"{j.jid}|{j.arrival:.6f}|{j.jtype}|{j.n_gpus}|{j.n_iters:.6f}|{j.deferrable}|{j.deadline:.3f}".encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------------------------


class Sim:
    def __init__(self, jobs, profile: dict, cluster: ClusterParams, power: PowerParams, ci_true, policy,
                 ci_forecast=None, record_trace=False, check=False):
        self.cp, self.pp = cluster, power
        self.profile = profile
        self.jobs = jobs
        self.jobs_by_id = {j.jid: j for j in jobs}
        self.ci = CI(ci_true)
        self.ci_f = CI(ci_forecast if ci_forecast is not None else ci_true)
        self.policy = policy
        self.record_trace = record_trace
        self.check = check
        self.W = cluster.window_h * HOUR
        P = cluster.n_pods
        self.free = [cluster.gpus_per_pod] * P
        self.queue: list[Job] = []
        self.running: dict[int, Job] = {}
        self.done: list[Job] = []
        self.heap: list = []
        self.seq = 0
        self.t = 0.0
        self.last_t = 0.0
        self.load = np.zeros(P)
        self.E_J = 0.0            # total energy (J, after PUE)
        self.carbon_g = 0.0
        self.area_n = 0.0
        self.area_busy = 0.0
        self.peak_W = 0.0
        self.cap_exceed_s = 0.0
        self.cap_block_events = 0
        self.trace = []
        self.parts_J = dict(gpu_alloc=0.0, gpu_idle=0.0, server=0.0, net=0.0)   # pre-PUE
        self.max_spread_running = 0
        self.full_reeval = False
        self._memo = {}
        xs, ys = zip(*cluster.congestion_pts)
        self._cx, self._cy = np.array(xs, float), np.array(ys, float)
        self.P = None
        self._power_cache()
        for j in jobs:
            self._push(j.arrival, "arr", j.jid)
        self._push(0.0, "tick", None)

    # ---- events ----
    def _push(self, t, typ, payload):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, typ, payload))

    # ---- network model ----
    def curve(self, ratio):
        r = np.asarray(ratio, float)
        y = np.interp(r, self._cx, self._cy)
        last_x, last_y = self._cx[-1], self._cy[-1]
        return np.where(r > last_x, last_y * r / last_x, y)

    def slowdowns(self, specs):
        """specs: list of (key, pods_list, demand_GBps). Returns (dict key->s, load array)."""
        key = tuple((k, tuple(p), round(d, 9)) for k, p, d in specs)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        load = np.zeros(self.cp.n_pods)
        for k, pods, d in specs:
            if len(pods) > 1:
                for p in pods:
                    load[p] += d
        sp = self.curve(load / self.cp.uplink_GBps)
        out = {}
        for k, pods, d in specs:
            out[k] = float(max(sp[p] for p in pods)) if len(pods) > 1 else 1.0
        if len(self._memo) > 200000:
            self._memo.clear()
        self._memo[key] = (out, load)
        return out, load

    # ---- profile helpers ----
    def prof(self, job, k):
        """(iter_time, frac_compute, GB per pod uplink per iteration, congestible comm time) for k pods.
        Uses the measured row for this k if the table has one; otherwise the row for the largest measured k0 < k,
        with volume and communication time scaled by K_RULES[k_rule](k) / K_RULES[k_rule](k0)."""
        if k == 1:
            v = self.profile[(job.jtype, job.n_gpus, "compact")]
            return v[0], v[1], 0.0, 0.0
        k0 = k
        while k0 > 2 and (job.jtype, job.n_gpus, spread_pclass(k0)) not in self.profile:
            k0 -= 1
        v = self.profile[(job.jtype, job.n_gpus, spread_pclass(k0))]
        it, fc, gb = v[0], v[1], v[2]
        comm = v[3] if len(v) > 3 else it * (1.0 - fc)
        if k0 == k:
            return it, fc, gb, comm
        rule = K_RULES[self.cp.k_rule]
        f = rule(k) / rule(k0)
        if self.cp.k_rule == "hier":                # v0.1.0 behaviour, unchanged: iteration time of the 2-pod row
            return it, fc, gb * f, comm * f
        # flat-ring rules: the ns-3 profiles have no overlap (it = t_comp + comm), so the uncongested iteration time
        # moves with the communication time; communication never falls below the compact (single-pod) value
        vc = self.profile[(job.jtype, job.n_gpus, "compact")]
        comm_c = vc[3] if len(vc) > 3 else vc[0] * (1.0 - vc[1])
        comm_k = max(comm * f, comm_c)
        t_comp = it * fc
        it_k = t_comp + (it - t_comp) * (comm_k / comm if comm > 0 else 1.0)
        return it_k, t_comp / it_k, gb * f, comm_k

    @staticmethod
    def it_eff(it, comm, s):
        """Congested iteration time: only the congestible communication stretches (conservative: no overlap slack)."""
        return it + comm * (s - 1.0)

    def gpu_power(self, n, fc):
        return n * (self.pp.P_gpu_idle + (self.pp.P_gpu_max - self.pp.P_gpu_idle) * fc ** self.pp.gamma)

    # ---- power ----
    def _power_cache(self):
        pp, cp = self.pp, self.cp
        alloc = 0
        gpu_alloc = 0.0
        for j in self.running.values():
            alloc += j.n_gpus
            gpu_alloc += self.gpu_power(j.n_gpus, j.t_comp / j.iter_eff)
        gpu_idle = (cp.n_gpus - alloc) * pp.P_gpu_idle
        server = cp.n_servers * pp.P_server
        n_active = int(np.sum(self.load > 1e-12)) if cp.sleep_idle_ports else cp.n_pods
        net = n_active * cp.ports_per_pod * pp.P_port + cp.n_pods * pp.P_net_static_pod + pp.P_core_static
        self.P = dict(gpu_alloc=gpu_alloc, gpu_idle=gpu_idle, server=server, net=net, alloc=alloc,
                      total=pp.PUE * (gpu_alloc + gpu_idle + server + net), n_active=n_active)

    def idle_floor_W(self):
        pp, cp = self.pp, self.cp
        return pp.PUE * (cp.n_gpus * pp.P_gpu_idle + cp.n_servers * pp.P_server
                         + cp.n_pods * pp.P_net_static_pod + pp.P_core_static
                         + (0 if cp.sleep_idle_ports else cp.n_pods * cp.ports_per_pod * pp.P_port))

    def max_power_W(self):
        pp, cp = self.pp, self.cp
        return pp.PUE * (cp.n_gpus * pp.P_gpu_max + cp.n_servers * pp.P_server
                         + cp.n_pods * (pp.P_net_static_pod + cp.ports_per_pod * pp.P_port) + pp.P_core_static)

    def est_total_power(self, active_jobs, extra_dyn_W=0.0, extra_pods=()):
        """Cluster power if only `active_jobs` run (+ extra dynamic W). For cap checks and forecasts."""
        pp, cp = self.pp, self.cp
        specs = [(j.jid, j.pods_list, j.demand) for j in active_jobs]
        s, load = self.slowdowns(specs) if specs else ({}, np.zeros(cp.n_pods))
        gpu_dyn = 0.0
        for j in active_jobs:
            ie = self.it_eff(j.iter_unc, j.comm_unc, s[j.jid])
            gpu_dyn += j.n_gpus * (pp.P_gpu_max - pp.P_gpu_idle) * (j.t_comp / ie) ** pp.gamma
        act = set(np.nonzero(load > 1e-12)[0].tolist()) | set(extra_pods)
        n_active = len(act) if cp.sleep_idle_ports else cp.n_pods
        net = n_active * cp.ports_per_pod * pp.P_port + cp.n_pods * pp.P_net_static_pod + pp.P_core_static
        base = cp.n_gpus * pp.P_gpu_idle + cp.n_servers * pp.P_server
        return pp.PUE * (base + gpu_dyn + net) + extra_dyn_W

    def cap_now(self):
        return self.cp.power_cap_W

    # ---- time advance ----
    def _advance(self, t):
        dt = t - self.last_t
        if dt > 0:
            for j in self.running.values():
                j.remaining -= dt / j.iter_eff
            P = self.P
            self.E_J += P["total"] * dt
            self.carbon_g += P["total"] * float(self.ci.integral(self.last_t, t)) / J_PER_KWH   # exact over CI steps inside (last_t, t]
            self.area_n += (len(self.queue) + len(self.running)) * dt
            self.area_busy += P["alloc"] * dt
            pue = self.pp.PUE
            self.parts_J["gpu_alloc"] += pue * P["gpu_alloc"] * dt
            self.parts_J["gpu_idle"] += pue * P["gpu_idle"] * dt
            self.parts_J["server"] += pue * P["server"] * dt
            self.parts_J["net"] += pue * P["net"] * dt
            self.peak_W = max(self.peak_W, P["total"])
            cap = self.cap_now()
            if cap is not None and P["total"] > cap + 1e-6:
                self.cap_exceed_s += dt
            if self.record_trace:
                self.trace.append((self.last_t, t, P["total"], P["gpu_alloc"], P["gpu_idle"], P["server"], P["net"]))
        self.last_t = t
        self.t = t

    def _refresh(self):
        """Recompute congestion slowdowns, completion events and power after any state change."""
        specs = [(j.jid, j.pods_list, j.demand) for j in self.running.values()]
        s, load = self.slowdowns(specs) if specs else ({}, np.zeros(self.cp.n_pods))
        self.load = load
        n_spread = 0
        for j in self.running.values():
            j.s = s[j.jid]
            j.iter_eff = self.it_eff(j.iter_unc, j.comm_unc, j.s)
            j.finish_est = self.t + max(j.remaining, 0.0) * j.iter_eff
            j.version += 1
            self._push(j.finish_est, "fin", (j.jid, j.version))
            n_spread += 1 if j.k > 1 else 0
        self.max_spread_running = max(self.max_spread_running, n_spread)
        self._power_cache()

    # ---- placement ----
    def first_fit(self, n):
        if sum(self.free) < n:
            return None
        rem, pl = n, {}
        for p in range(self.cp.n_pods):
            take = min(self.free[p], rem)
            if take > 0:
                pl[p] = take
                rem -= take
            if rem == 0:
                break
        return pl

    def candidates(self, job, free=None, load=None):
        free = self.free if free is None else free
        n = job.n_gpus
        if sum(free) < n:
            return []
        P = self.cp.n_pods
        comp = [p for p in range(P) if free[p] >= n]
        if comp:
            p = min(comp, key=lambda q: (free[q], q))
            return [{p: n}]
        load = self.load if load is None else load
        orders = [sorted(range(P), key=lambda q: (-free[q], load[q], q)),
                  sorted([q for q in range(P) if free[q] > 0], key=lambda q: (load[q], -free[q], q))]
        out, seen = [], set()
        for order in orders:
            rem, pl = n, {}
            for q in order:
                if free[q] <= 0:
                    continue
                take = min(free[q], rem)
                pl[q] = take
                rem -= take
                if rem == 0:
                    break
            key = tuple(sorted(pl.items()))
            if rem == 0 and key not in seen:
                seen.add(key)
                out.append(pl)
        return out

    def forecast_free(self, t):
        free = list(self.free)
        for j in self.running.values():
            if j.finish_est <= t:
                for p, c in j.pods.items():
                    free[p] += c
        return free

    def active_at(self, t):
        return [j for j in self.running.values() if j.finish_est > t + 1e-9]

    # ---- evaluation of a candidate (used by P3/P4 and the cap check) ----
    def eval_candidate(self, job, placement, t=None, active=None):
        """Estimated runtime, job dynamic power and externality (extra seconds imposed on running jobs)."""
        t = self.t if t is None else t
        active = list(self.running.values()) if active is None else active
        pods = sorted(placement)
        k = len(pods)
        it, fc, gb, comm = self.prof(job, k)
        specs = [(j.jid, j.pods_list, j.demand) for j in active]
        s_wo, _ = self.slowdowns(specs) if specs else ({}, None)
        cand = ("cand", pods, gb / it if k > 1 else 0.0)
        s_w, _ = self.slowdowns(specs + [cand])
        sc = s_w["cand"]
        ie = self.it_eff(it, comm, sc)
        R = job.n_iters * ie + (self.cp.reconf_delay_s if k > 1 else 0.0)
        t_comp = it * fc
        p_gpu = self.gpu_power(job.n_gpus, t_comp / ie)
        extra = []
        for j in active:
            if abs(s_w[j.jid] - s_wo[j.jid]) < 1e-12:
                continue
            ie_w = self.it_eff(j.iter_unc, j.comm_unc, s_w[j.jid])
            ie_wo = self.it_eff(j.iter_unc, j.comm_unc, s_wo[j.jid])
            rem_it = j.remaining if t <= self.t + 1e-9 else max(0.0, (j.finish_est - t) / j.iter_eff)
            extra.append((j, rem_it * (ie_w - ie_wo)))
        return dict(R=R, p_gpu=p_gpu, ie=ie, s=sc, k=k, extra=extra, pods=pods, gb=gb, it=it)

    def power_ok(self, job, placement):
        cap = self.cap_now()
        if cap is None:
            return True
        ev = self.eval_candidate(job, placement)
        net_extra = 0.0
        if self.cp.sleep_idle_ports and ev["k"] > 1:
            newp = [p for p in ev["pods"] if self.load[p] <= 1e-12]
            net_extra = len(newp) * self.cp.ports_per_pod * self.pp.P_port
        dW = self.pp.PUE * (ev["p_gpu"] - job.n_gpus * self.pp.P_gpu_idle + net_extra)
        ok = self.P["total"] + dW <= cap + 1e-6
        if not ok:
            self.cap_block_events += 1
        return ok

    # ---- lifecycle ----
    def _start(self, job, placement):
        pods = sorted(placement)
        k = len(pods)
        it, fc, gb, comm = self.prof(job, k)
        job.pods, job.pods_list, job.k = dict(placement), pods, k
        job.iter_unc, job.fc_unc, job.t_comp, job.comm_unc = it, fc, it * fc, comm
        job.gb_k = gb
        job.demand = gb / it if k > 1 else 0.0
        job.remaining = job.n_iters + (self.cp.reconf_delay_s / it if k > 1 else 0.0)
        job.unc_runtime = job.remaining * it
        # one-pod baseline; a table without a compact row for this size (job larger than a pod, synthetic profile)
        # falls back to the job's own uncongested runtime, i.e. no placement penalty is attributed
        job.cmp_runtime = (job.n_iters * self.prof(job, 1)[0] if (job.jtype, job.n_gpus, 'compact') in self.profile
                           else job.unc_runtime)
        job.start = self.t
        job.ci_at_start = self.ci.at(self.t)
        for p, c in placement.items():
            self.free[p] -= c
        self.queue.remove(job)
        self.running[job.jid] = job

    def _finish(self, job):
        job.finish = self.t
        job.remaining = 0.0
        for p, c in job.pods.items():
            self.free[p] += c
        del self.running[job.jid]
        self.done.append(job)

    def _reservation(self, job, power=False):
        fe = sorted((j.finish_est, j.n_gpus) for j in self.running.values())
        free = sum(self.free)
        if power and free >= job.n_gpus:
            return (fe[0][0] if fe else math.inf), 0
        for t_f, n in fe:
            free += n
            if free >= job.n_gpus:
                return t_f, free - job.n_gpus
        return math.inf, 0

    def _schedule(self):
        shadow, extra = None, 0
        for job in list(self.queue):
            status, placement = self.policy.decide(self, job)
            if status == "defer":
                continue
            if status == "blocked" or placement is None:
                if shadow is None:
                    shadow, extra = self._reservation(job)
                continue
            if shadow is not None:
                it = self.prof(job, len(placement))[0]
                est_fin = self.t + job.n_iters * it
                if est_fin <= shadow + 1e-9:
                    pass
                elif job.n_gpus <= extra:
                    extra -= job.n_gpus
                else:
                    continue
            if not self.power_ok(job, placement):
                if shadow is None:
                    shadow, extra = self._reservation(job, power=True)
                continue
            self._start(job, placement)
            self._refresh()

    def _check_invariants(self):
        alloc = [0] * self.cp.n_pods
        for j in self.running.values():
            for p, c in j.pods.items():
                alloc[p] += c
        for p in range(self.cp.n_pods):
            assert self.free[p] >= 0, "negative free GPUs"
            assert self.free[p] + alloc[p] == self.cp.gpus_per_pod, "GPU conservation violated"
        ids = [j.jid for j in self.queue] + list(self.running) + [j.jid for j in self.done]
        assert len(ids) == len(set(ids)), "job in two states"

    def run(self):
        arrived = 0
        while self.heap:
            t, _, typ, pl = heapq.heappop(self.heap)
            if t > self.W:
                break
            if typ == "fin":
                jid, ver = pl
                j = self.running.get(jid)
                if j is None or j.version != ver:
                    continue
            self._advance(t)
            self.full_reeval = False
            if typ == "arr":
                self.queue.append(self.jobs_by_id[pl])
                arrived += 1
            elif typ == "fin":
                self._finish(self.running[pl[0]])
                self._refresh()
            elif typ == "tick":
                self.full_reeval = True
                if t + self.cp.tick_s <= self.W:
                    self._push(t + self.cp.tick_s, "tick", None)
            self._schedule()
            if self.check:
                self._check_invariants()
        self._advance(self.W)
        return self.metrics()

    # ---- metrics ----
    def metrics(self):
        cp, pp = self.cp, self.pp
        done = self.done
        n_all = len(self.jobs)
        floor_W = self.idle_floor_W()
        floor_carbon = floor_W * float(self.ci.integral(0.0, self.W)) / J_PER_KWH
        floor_energy_kWh = floor_W * self.W / J_PER_KWH
        out = dict(n_jobs=n_all, n_done=len(done), n_incomplete=n_all - len(done))
        out["completion_rate"] = len(done) / max(n_all, 1)
        out["done_gpu_h"] = float(sum(j.n_gpus * j.dur_ref_s for j in done) / HOUR)
        out["energy_kWh"] = self.E_J / J_PER_KWH
        out["carbon_kg"] = self.carbon_g / 1000.0
        out["energy_dyn_kWh"] = out["energy_kWh"] - floor_energy_kWh
        out["carbon_dyn_kg"] = (self.carbon_g - floor_carbon) / 1000.0
        out["carbon_per_gpuh_g"] = self.carbon_g / max(out["done_gpu_h"], 1e-9)      # total carbon per COMPLETED GPU-hour
        out["peak_kW"] = self.peak_W / 1000.0
        out["mean_kW"] = self.E_J / self.W / 1000.0
        out["cap_exceed_h"] = self.cap_exceed_s / HOUR
        out["cap_block_events"] = self.cap_block_events
        out["util_time_avg"] = self.area_busy / (self.W * cp.n_gpus)
        if done:
            jct = np.array([j.finish - j.arrival for j in done])
            wait = np.array([j.start - j.arrival for j in done])
            run = np.array([j.finish - j.start for j in done])
            unc = np.array([j.unc_runtime for j in done])
            gp = np.array([j.n_gpus for j in done], float)
            out["jct_mean_h"] = float(jct.mean() / HOUR)
            out["jct_p95_h"] = float(np.percentile(jct, 95) / HOUR)
            out["wait_mean_h"] = float(wait.mean() / HOUR)
            out["net_inflation"] = float((run * gp).sum() / (unc * gp).sum())    # congestion only (baseline: chosen placement)
            cmp_ = np.array([j.cmp_runtime for j in done])
            out["net_inflation_total"] = float((run * gp).sum() / (cmp_ * gp).sum())   # placement + congestion (baseline: one pod)
            out["spread_frac"] = float(np.mean([j.k > 1 for j in done]))
            out["gpuh_frac_k3plus"] = float(np.sum([j.n_gpus * j.dur_ref_s for j in done if j.k >= 3]) /
                                            np.sum([j.n_gpus * j.dur_ref_s for j in done]))      # exposure to the k>2 extension
            out["k_max"] = int(max(j.k for j in done))
            out["ci_at_start_gpuw"] = float(np.sum([j.ci_at_start * j.n_gpus * j.dur_ref_s for j in done]) /
                                            np.sum([j.n_gpus * j.dur_ref_s for j in done]))
        viol = sum(1 for j in done if j.finish > j.deadline) + (n_all - len(done))
        out["deadline_viol_frac"] = viol / max(n_all, 1)
        out["max_spread_running"] = self.max_spread_running
        out["area_n"] = self.area_n
        return out


# ----------------------------------------------------------------------------------------------
# Policies. decide() -> ('start', placement) | ('defer', None) | ('blocked', None)
# ----------------------------------------------------------------------------------------------


class Policy:
    name = "base"

    def decide(self, sim, job):
        raise NotImplementedError


class P0(Policy):
    """Baseline: FIFO + EASY backfill, first-fit GPU fill, no carbon/network/power awareness (cap is a hard limit)."""
    name = "P0"

    def decide(self, sim, job):
        pl = sim.first_fit(job.n_gpus)
        return ("start", pl) if pl else ("blocked", None)


def _oblivious_est(sim, job):
    """Network-oblivious estimate: reference class profile, no congestion."""
    cls = _ref_class(job.n_gpus, sim.cp)
    it, fc, gb = sim.profile[(job.jtype, job.n_gpus, cls)][:3]
    R = job.n_iters * it
    dynW = sim.pp.PUE * job.n_gpus * (sim.pp.P_gpu_max - sim.pp.P_gpu_idle) * fc ** sim.pp.gamma
    return R, dynW


def _start_grid(sim, job, R, margin=0.15):
    now = sim.t
    latest = job.deadline - R * (1.0 + margin)
    last_ok = sim.W - R * 1.2
    latest = min(latest, last_ok)
    if (not job.deferrable) or latest <= now + 1e-9:
        return np.array([now])
    t0 = math.floor(now / HOUR + 1e-9) * HOUR + HOUR
    grid = np.arange(t0, latest + 1e-9, HOUR)
    return np.concatenate([[now], grid[:96]])


class P2(Policy):
    """Carbon-aware: shift deferrable jobs to the lowest-forecast-carbon start within deadline slack, but only if the
    forecast saving is at least `min_gain` (default 5%); this avoids a straw-man baseline that delays jobs for
    negligible gains. 'P2raw' uses min_gain=0 (pure carbon minimiser).
    Network- and power-oblivious placement (first-fit)."""
    name = "P2"

    def __init__(self, min_gain: float = 0.05):
        self.min_gain = min_gain

    def decide(self, sim, job):
        if job.deferrable and job.target is not None and not sim.full_reeval \
                and sim.t < job.target - sim.cp.tick_s - 1e-9:
            return ("defer", None)
        R, dynW = _oblivious_est(sim, job)
        ts = _start_grid(sim, job, R)
        if len(ts) > 1:
            carb = dynW / 1000.0 * (sim.ci_f.integral(ts, ts + R)) / HOUR
            i = int(np.argmin(carb))               # ties -> earliest
            best = ts[i] if carb[i] < carb[0] * (1.0 - self.min_gain) - 1e-12 else ts[0]
        else:
            best = ts[0]
        job.target = float(best)
        if best > sim.t + 1e-9:
            return ("defer", None)
        pl = sim.first_fit(job.n_gpus)
        return ("start", pl) if pl else ("blocked", None)


class P3(Policy):
    """Network-aware: start ASAP, choose placement minimising own runtime plus interference imposed on others."""
    name = "P3"

    def decide(self, sim, job):
        cands = sim.candidates(job)
        if not cands:
            return ("blocked", None)
        best, bc = None, math.inf
        for pl in cands:
            ev = sim.eval_candidate(job, pl)
            c = job.n_gpus * ev["R"] + sum(j.n_gpus * x for j, x in ev["extra"])
            if c < bc - 1e-9:
                best, bc = pl, c
        return ("start", best)


class P4(Policy):
    """Joint carbon + network + power. Cost in gCO2-equivalent:
         carbon(start, runtime incl. congestion) + externality on running jobs + w_delay x delay price.
    Considers start-now (best placement) versus hourly future starts (placement class from forecast free GPUs,
    congestion from the jobs still active then). Power cap enforced at start and in future-start feasibility."""
    name = "P4"

    def __init__(self, w_delay: float = 0.02, defer_margin: float = 0.005, claim_aware: bool = True):
        self.w_delay = w_delay
        self.defer_margin = defer_margin
        self.claim_aware = claim_aware      # False = ablation 'P4nc': ignore GPUs already claimed by earlier queued jobs

    def _cost(self, sim, job, t, pl, active, ci_ref):
        pp = sim.pp
        ev = sim.eval_candidate(job, pl, t=t, active=active)
        net_dW = 0.0
        if sim.cp.sleep_idle_ports and ev["k"] > 1:
            act_load = sim.slowdowns([(j.jid, j.pods_list, j.demand) for j in active])[1] if active else np.zeros(sim.cp.n_pods)
            net_dW = sum(1 for p in ev["pods"] if act_load[p] <= 1e-12) * sim.cp.ports_per_pod * pp.P_port
        dynW = pp.PUE * (ev["p_gpu"] - job.n_gpus * pp.P_gpu_idle + net_dW)
        carbon = dynW / 1000.0 * float(sim.ci_f.integral(t, t + ev["R"])) / HOUR
        price_kW = pp.PUE * pp.P_gpu_max / 1000.0
        delay_h = (t + ev["R"] - job.arrival) / HOUR
        delay_cost = self.w_delay * job.n_gpus * price_kW * ci_ref * delay_h
        ext = 0.0
        for j, x in ev["extra"]:
            kW_j = pp.PUE * j.n_gpus * (pp.P_gpu_max - pp.P_gpu_idle) * (j.t_comp / j.iter_eff) ** pp.gamma / 1000.0
            ext += x / HOUR * (kW_j * ci_ref + self.w_delay * j.n_gpus * price_kW * ci_ref)
        return carbon + delay_cost + ext, ev, dynW

    def decide(self, sim, job):
        if job.deferrable and job.target is not None and not sim.full_reeval \
                and sim.t < job.target - sim.cp.tick_s - 1e-9:
            return ("defer", None)
        now = sim.t
        ci_ref = sim.ci_f.mean(now, now + 24 * HOUR)
        cands = sim.candidates(job)
        if not cands:
            # resource-blocked now; deferrable jobs still record a target for caching, but queue position holds
            job.target = now
            return ("blocked", None)
        running = list(sim.running.values())
        best_now, c_now, ev_now = None, math.inf, None
        for pl in cands:
            c, ev, dW = self._cost(sim, job, now, pl, running, ci_ref)
            if c < c_now - 1e-9:
                best_now, c_now, ev_now = pl, c, ev
        R_est = ev_now["R"]
        ts = _start_grid(sim, job, R_est)
        if len(ts) > 1:
            best_t, best_c = now, c_now
            for t in ts[1:]:
                free_t = sim.forecast_free(t)
                claimed = 0
                if self.claim_aware:
                    claimed = sum(j.n_gpus for j in sim.queue
                                  if j.jid < job.jid and j.target is not None and j.target <= t + 1e-9)
                if sum(free_t) - claimed < job.n_gpus:
                    continue
                active = sim.active_at(t)
                load_t = sim.slowdowns([(j.jid, j.pods_list, j.demand) for j in active])[1] if active else np.zeros(sim.cp.n_pods)
                cl = sim.candidates(job, free=free_t, load=load_t)
                if not cl:
                    continue
                for pl in cl:
                    c, ev, dW = self._cost(sim, job, float(t), pl, active, ci_ref)
                    if sim.cap_now() is not None:
                        extra_pods = [p for p in ev["pods"] if load_t[p] <= 1e-12] if ev["k"] > 1 else []
                        if sim.est_total_power(active, dW, ()) > sim.cap_now() + 1e-6:
                            continue
                    if c < best_c * (1.0 - self.defer_margin) - 1e-9:
                        best_t, best_c = float(t), c
            job.target = best_t
            if best_t > now + 1e-9:
                return ("defer", None)
        else:
            job.target = now
        return ("start", best_now)


class ForcedPolicy(Policy):
    """Executes a given decision table {jid: (start_time_s, 'compact'|'spread')}. Used by the exhaustive-search
    benchmark and by verification tests."""
    name = "FORCED"

    def __init__(self, decisions):
        self.dec = decisions

    def decide(self, sim, job):
        t0, mode = self.dec[job.jid]
        if sim.t < t0 - 1e-9:
            return ("defer", None)
        n = job.n_gpus
        P = sim.cp.n_pods
        if mode == "compact" and n <= sim.cp.gpus_per_pod:
            comp = [p for p in range(P) if sim.free[p] >= n]
            if comp:
                return ("start", {min(comp, key=lambda q: (sim.free[q], q)): n})
            return ("blocked", None)
        # spread over exactly two pods, as evenly as free GPUs allow
        order = sorted(range(P), key=lambda q: (-sim.free[q], q))
        a = n // 2
        if len(order) >= 2 and sim.free[order[0]] >= n - a and sim.free[order[1]] >= a:
            return ("start", {order[0]: n - a, order[1]: a})
        pl = sim.first_fit(n)
        if pl and len(pl) >= 2:
            return ("start", pl)
        return ("blocked", None)


POLICIES = {"P0": P0, "P2": P2, "P3": P3, "P4": P4}


# ----------------------------------------------------------------------------------------------
# One run + experiment design
# ----------------------------------------------------------------------------------------------


@dataclass
class Scenario:
    arrival: str = "poisson"
    util: float = 0.6
    ci: str = "synthetic_high_var"
    comm_scale: float = 1.0
    cap_frac: Optional[float] = None       # power cap as fraction of the cluster's maximum power
    fabric: str = "ocs"
    forecast_sigma: float = 0.0
    slack_lo: float = 2.0
    slack_hi: float = 12.0
    deferrable_frac: float = 0.7
    w_delay: float = 0.02
    tick_s: float = HOUR
    trace: str = ""                       # path to a trace CSV (arrival_s,n_gpus,duration_s); '' = synthetic arrivals
    forecast: str = "perfect"             # 'perfect' | 'persistence' (day-ahead: forecast = same hour yesterday) (0.4.0)


def build_world(sc: Scenario, rep: int, master_seed=MASTER_SEED_DEFAULT, profile_df: Optional[pd.DataFrame] = None,
                cluster_base: Optional[ClusterParams] = None, power_base: Optional[PowerParams] = None):
    cp = ClusterParams(**asdict(cluster_base)) if cluster_base else ClusterParams()
    pp = PowerParams(**asdict(power_base)) if power_base else PowerParams()
    apply_fabric(cp, pp, sc.fabric)
    cp.uplink_GBps *= cp.uplink_scale
    cp.tick_s = sc.tick_s
    if profile_df is None:
        profile_df = synthetic_profile(cp, comm_scale=sc.comm_scale)
    elif sc.comm_scale != 1.0:
        profile_df = scale_profile(profile_df, sc.comm_scale, cp.comm_scale_iter)
    prof = profile_dict(profile_df)
    if sc.cap_frac is not None:
        tmp = Sim([], prof, cp, pp, np.ones(10), P0())
        cp.power_cap_W = sc.cap_frac * tmp.max_power_W()
    if sc.trace:
        jobs, wmeta = workload_from_trace(pd.read_csv(sc.trace), rep, sc.util, cp, prof, master_seed=master_seed,
                                          deferrable_frac=sc.deferrable_frac, slack_range=(sc.slack_lo, sc.slack_hi))
    else:
        jobs, wmeta = make_workload(rep, sc.arrival, sc.util, cp, prof, master_seed=master_seed,
                                    deferrable_frac=sc.deferrable_frac, slack_range=(sc.slack_lo, sc.slack_hi))
    ci = make_ci(sc.ci, rep, int(cp.window_h) + 1, master_seed)
    cif = None
    if sc.forecast == "persistence":
        # day-ahead persistence: the forecast for hour h is the observed value at h - 24; for the first day, with no
        # history inside the window, the forecast is flat at that day's mean (no shifting information).
        cif = np.empty_like(ci)
        cif[:24] = ci[:24].mean()
        cif[24:] = ci[:-24]
    elif sc.forecast != "perfect":
        raise ValueError(f"unknown forecast mode {sc.forecast}")
    if sc.forecast_sigma > 0:
        rf = rng_for(master_seed, rep, "forecast")
        cif = (ci if cif is None else cif) * np.exp(rf.normal(0.0, sc.forecast_sigma, size=len(ci)))
    return jobs, prof, cp, pp, ci, cif, wmeta


def run_one(sc: Scenario, policy_name: str, rep: int, master_seed=MASTER_SEED_DEFAULT,
            profile_df=None, cluster_base=None, power_base=None, check=False, record_trace=False):
    import time as _t
    t0 = _t.time()
    jobs, prof, cp, pp, ci, cif, wmeta = build_world(sc, rep, master_seed, profile_df, cluster_base, power_base)
    if policy_name == "P2raw":
        pol = P2(min_gain=0.0)
    elif policy_name.startswith("P4"):
        pol = P4(w_delay=sc.w_delay, claim_aware=(policy_name != "P4nc"))
    else:
        pol = POLICIES[policy_name]()
    sim = Sim(jobs, prof, cp, pp, ci, pol, ci_forecast=cif, check=check, record_trace=record_trace)
    m = sim.run()
    m.update(asdict(sc))
    m.update(policy=policy_name, rep=rep, workload_hash=workload_hash(jobs), runtime_s=_t.time() - t0,
             lam_per_h=wmeta["lam_per_h"], code_version=__version__,
             profile_source=(profile_df["source"].iloc[0] if profile_df is not None else "synthetic-parametric"))
    return (m, sim) if record_trace or check else m