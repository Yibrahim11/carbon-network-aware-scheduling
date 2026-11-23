"""
Verification battery for dcsim (textbook Ch. 13.2 programme): known-curve checks, degenerate cases, internal
consistency, distributional checks, discretisation insensitivity.  Each check returns a dict
{name, status in PASS|FAIL|WARN, detail}.  Run run_all() and keep the report with your results.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy import stats

import dcsim
from dcsim import (HOUR, J_PER_KWH, ClusterParams, PowerParams, Job, Sim, P0, ForcedPolicy, Scenario,
                   rng_for, arrivals)


def _res(name, ok, detail, warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    return dict(name=name, status=status, detail=detail)


def _tiny_profile(entries):
    return {k: v for k, v in entries.items()}


# ---------------------------------------------------------------------------------------------
def check_hand_case():
    """One 8-GPU compact job, constant CI: energy and carbon computed by hand."""
    cp, pp = ClusterParams(), PowerParams()
    cp.horizon_h, cp.window_h = 10, 10
    prof = {("hand", 8, "compact"): (2.0, 0.8, 0.0)}
    job = Job(0, 0.0, "hand", 8, 1800.0, False, 1e9, 3600.0)
    ci = np.full(11, 400.0)
    sim = Sim([job], prof, cp, pp, ci, P0())
    m = sim.run()
    pg = 8 * (100 + 550 * 0.8)
    floor = 1.3 * (128 * 100 + 16 * 800 + 4 * 400 + 400)
    during = 1.3 * (pg + 120 * 100 + 16 * 800 + 4 * 400 + 400)
    E_kwh = (floor * 10 * HOUR + (during - floor) * 3600.0) / J_PER_KWH
    ok = (abs(m["energy_kWh"] - E_kwh) < 1e-6 and abs(m["carbon_kg"] - E_kwh * 0.4) < 1e-6
          and abs(job.finish - 3600.0) < 1e-6)
    return _res("hand-computed single job (energy, carbon, runtime)", ok,
                f"sim E={m['energy_kWh']:.6f} kWh vs hand {E_kwh:.6f}; finish={job.finish:.3f}s (expect 3600)")



def check_hand_case_varying_ci():
    """Job spanning three different CI hours: carbon must be split exactly per hour (catches CI-timing bugs)."""
    cp, pp = ClusterParams(), PowerParams()
    cp.horizon_h, cp.window_h = 6, 6
    prof = {("hand", 8, "compact"): (2.0, 0.8, 0.0)}
    job = Job(0, 0.0, "hand", 8, 4500.0, False, 1e9, 9000.0)          # runs 0..9000 s = 2.5 h
    ci = np.array([100.0, 300.0, 500.0, 50.0, 50.0, 50.0, 50.0])
    m = Sim([job], prof, cp, pp, ci, P0()).run() if False else None
    sim = Sim([job], prof, cp, pp, ci, P0())
    m = sim.run()
    floor = 1.3 * (128 * 100 + 16 * 800 + 4 * 400 + 400)
    during = 1.3 * (8 * (100 + 550 * 0.8) + 120 * 100 + 16 * 800 + 4 * 400 + 400)
    kwh = lambda W, s: W * s / J_PER_KWH
    carbon = (kwh(floor, 6 * 3600) * 0 +
              sum(kwh(floor, 3600) * c for c in ci[:6]) +
              kwh(during - floor, 3600) * 100 + kwh(during - floor, 3600) * 300 + kwh(during - floor, 1800) * 500)
    ok = abs(m["carbon_kg"] * 1000 - carbon) < 1e-6 and abs(job.finish - 9000.0) < 1e-6
    return _res("hand-computed carbon with hourly-varying CI (job spans three CI hours)", ok,
                f"sim {m['carbon_kg'] * 1000:.4f} g vs hand {carbon:.4f} g")


def check_carbon_misaligned_tick():
    """Regression (audit finding 1): scheduler tick = 2 h while CI changes hourly; carbon must still integrate exactly."""
    cp, pp = ClusterParams(), PowerParams()
    cp.horizon_h, cp.window_h, cp.tick_s = 6, 6, 7200.0
    prof = {("hand", 8, "compact"): (2.0, 0.8, 0.0)}
    job = Job(0, 0.0, "hand", 8, 4500.0, False, 1e9, 9000.0)
    ci = np.array([100.0, 300.0, 500.0, 50.0, 50.0, 50.0, 50.0])
    sim = Sim([job], prof, cp, pp, ci, P0()); m = sim.run()
    floor = 1.3 * (128 * 100 + 16 * 800 + 4 * 400 + 400)
    during = 1.3 * (8 * (100 + 550 * 0.8) + 120 * 100 + 16 * 800 + 4 * 400 + 400)
    kwh = lambda W, s: W * s / J_PER_KWH
    carbon = (sum(kwh(floor, 3600) * c for c in ci[:6]) +
              kwh(during - floor, 3600) * 100 + kwh(during - floor, 3600) * 300 + kwh(during - floor, 1800) * 500)
    ok = abs(m["carbon_kg"] * 1000 - carbon) < 1e-6
    return _res("carbon integration exact with a 2-hour scheduler tick (CI changes hourly)", ok,
                f"sim {m['carbon_kg'] * 1000:.4f} g vs hand {carbon:.4f} g")


def check_k_extension():
    """Jobs over k > 2 pods: measured rows take precedence; otherwise the 2-pod row is extended by ClusterParams.k_rule.
    Checks the three rules against their formulas, the compact floor, and that 'hier' keeps the v0.1.0 numbers."""
    cp = ClusterParams(n_pods=4, gpus_per_pod=32)
    prof = {("x", 64, "compact"): (2.68, 2.0 / 2.68, 0.0, 0.68), ("x", 64, "spread"): (22.853, 2.0 / 22.853, 882.0, 20.853)}
    job = Job(0, 0.0, "x", 64, 10.0, False, 1e9, 3600.0)
    got, ok = [], True
    for rule, k, want_gb, want_comm in (("hier", 3, 882.0 * 4 / 3, 20.853 * 4 / 3), ("hier", 4, 882.0 * 1.5, 20.853 * 1.5),
                                        ("ring_interleaved", 3, 882.0 * 2 / 3, 20.853 * 2 / 3),
                                        ("ring_interleaved", 4, 441.0, 20.853 / 2), ("ring_aligned", 4, 882.0, 20.853)):
        c = ClusterParams(**{**dcsim.asdict(cp), "k_rule": rule})
        it, fc, gb, comm = Sim([], prof, c, PowerParams(), np.ones(10), P0()).prof(job, k)
        ok &= abs(gb - want_gb) < 1e-9 and abs(comm - want_comm) < 1e-9
        if rule == "hier":
            ok &= abs(it - 22.853) < 1e-12                       # v0.1.0: uncongested iteration time of the 2-pod row
        else:
            ok &= abs(it - (2.0 + want_comm)) < 1e-9 and abs(fc * it - 2.0) < 1e-9   # it = t_comp + comm (no overlap)
        got.append(f"{rule} k={k}: GB {gb:.1f} (want {want_gb:.1f})")
    # compact floor: a very spread interleaved job never communicates faster than the same job inside one pod
    c = ClusterParams(**{**dcsim.asdict(cp), "k_rule": "ring_interleaved", "n_pods": 64, "gpus_per_pod": 1})
    comm_floor = Sim([], prof, c, PowerParams(), np.ones(10), P0()).prof(job, 64)[3]
    ok &= abs(comm_floor - 0.68) < 1e-12
    # measured row for k=3 wins; k=4 then extends from the measured k=3 row
    prof3 = dict(prof); prof3[("x", 64, "spread_k3")] = (12.0, 2.0 / 12.0, 500.0, 10.0)
    c = ClusterParams(**{**dcsim.asdict(cp), "k_rule": "ring_interleaved"})
    sim = Sim([], prof3, c, PowerParams(), np.ones(10), P0())
    m3, m4 = sim.prof(job, 3), sim.prof(job, 4)
    ok &= m3 == (12.0, 2.0 / 12.0, 500.0, 10.0) and abs(m4[2] - 500.0 * (2 / 4) / (2 / 3)) < 1e-9
    return _res("jobs over k>2 pods: measured rows first, then the k_rule extension (hier / flat-ring), compact floor", ok,
                "; ".join(got) + f"; floor {comm_floor:.2f}s; measured k=3 used {m3[2]:.0f} GB, k=4 from it {m4[2]:.0f} GB")


def check_congestion_degenerate():
    """Spread job: runtime must equal n*it*(fc+(1-fc)*s) with s from the load ratio."""
    out = []
    for uplink, expect_s in [(40.0, 1.0), (10.0, 2.0)]:
        cp, pp = ClusterParams(n_pods=2, gpus_per_pod=32), PowerParams()
        cp.horizon_h, cp.window_h = 10, 10
        cp.uplink_GBps, cp.reconf_delay_s = uplink, 0.0
        prof = {("d", 16, "spread"): (1.0, 0.5, 20.0), ("d", 16, "compact"): (1.0, 0.5, 0.0)}
        job = Job(0, 0.0, "d", 16, 1000.0, False, 1e9, 1000.0)
        sim = Sim([job], prof, cp, pp, np.full(11, 300.0), ForcedPolicy({0: (0.0, "spread")}))
        sim.run()
        want = 1000 * 1.0 * (0.5 + 0.5 * expect_s)
        out.append((uplink, expect_s, job.finish - job.start, want))
    # two identical concurrent spread jobs share the same two uplinks: load doubles
    cp, pp = ClusterParams(n_pods=2, gpus_per_pod=32), PowerParams()
    cp.horizon_h, cp.window_h = 10, 10
    cp.uplink_GBps, cp.reconf_delay_s = 40.0, 0.0
    prof = {("d", 16, "spread"): (1.0, 0.5, 20.0), ("d", 16, "compact"): (1.0, 0.5, 0.0)}
    jobs = [Job(i, 0.0, "d", 16, 1000.0, False, 1e9, 1000.0) for i in range(2)]
    sim = Sim(jobs, prof, cp, pp, np.full(11, 300.0), ForcedPolicy({0: (0.0, "spread"), 1: (0.0, "spread")}))
    sim.run()
    want2 = 1000 * (0.5 + 0.5 * 1.0)  # load 40 on uplink 40 -> ratio 1 -> s=1
    out.append(("2 jobs, ratio 1", 1.0, jobs[1].finish - jobs[1].start, want2))
    cp.uplink_GBps = 20.0
    jobs = [Job(i, 0.0, "d", 16, 1000.0, False, 1e9, 1000.0) for i in range(2)]
    sim = Sim(jobs, prof, cp, pp, np.full(11, 300.0), ForcedPolicy({0: (0.0, "spread"), 1: (0.0, "spread")}))
    sim.run()
    out.append(("2 jobs, ratio 2", 2.0, jobs[1].finish - jobs[1].start, 1000 * (0.5 + 0.5 * 2.0)))
    # fully hidden communication (fc=1, exposed=0) must still slow down under congestion via comm_time_s
    cp, pp = ClusterParams(n_pods=2, gpus_per_pod=32), PowerParams()
    cp.horizon_h, cp.window_h = 10, 10
    cp.uplink_GBps, cp.reconf_delay_s = 10.0, 0.0
    prof = {("h", 16, "spread"): (1.0, 1.0, 20.0, 0.4), ("h", 16, "compact"): (1.0, 1.0, 0.0, 0.0)}
    job = Job(0, 0.0, "h", 16, 1000.0, False, 1e9, 1000.0)
    Sim([job], prof, cp, pp, np.full(11, 300.0), ForcedPolicy({0: (0.0, "spread")})).run()
    out.append(("hidden comm, s=2", 2.0, job.finish - job.start, 1000 * (1.0 + 0.4 * (2.0 - 1.0))))
    ok = all(abs(a - b) < 1e-6 for *_, a, b in out)
    return _res("congestion degenerate cases (runtime = n*(it + comm*(s-1)), incl. fully hidden communication)", ok,
                "; ".join(f"{x[0]}: got {x[2]:.3f}s want {x[3]:.3f}s" for x in out))


def erlang_c_W(lam, mu, c):
    a = lam / mu
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) * c / (c - a)
    Cw = top / (s + top)
    return Cw / (c * mu - lam) + 1.0 / mu


def _mmc_run(rho, c, mu, rep, horizon_h=200.0, master=dcsim.MASTER_SEED_DEFAULT):
    cp, pp = ClusterParams(n_pods=1, gpus_per_pod=c, gpus_per_server=1), PowerParams()
    cp.horizon_h, cp.window_h = horizon_h, horizon_h + 60.0
    lam = rho * c * mu
    ra, rj = rng_for(master, rep, "arrival"), rng_for(master, rep, "attrs")
    times = arrivals("poisson", lam, horizon_h * HOUR, ra)
    prof = {("m", 1, "compact"): (1.0, 1.0, 0.0)}
    jobs = [Job(i, float(t), "m", 1, float(-math.log1p(-rj.random()) / mu), False, 1e18, 1.0)
            for i, t in enumerate(times)]
    sim = Sim(jobs, prof, cp, pp, np.full(int(cp.window_h) + 2, 100.0), P0(), check=False)
    sim.run()
    return jobs, sim, lam, cp


def check_mmc(rhos=(0.5, 0.8), c=8, mu=1.0 / 600.0, reps=8):
    """DES verification against Erlang-C (textbook Ch. 9.3): terminating runs, warm-up deleted, replications."""
    details, ok_all = [], True
    for rho in rhos:
        means = []
        for r in range(reps):
            jobs, sim, lam, cp = _mmc_run(rho, c, mu, r)
            warm = 0.1 * cp.horizon_h * HOUR
            soj = [j.finish - j.arrival for j in jobs if j.finish is not None and j.arrival > warm]
            means.append(np.mean(soj))
        m, s = np.mean(means), np.std(means, ddof=1)
        half = stats.t.ppf(0.995, reps - 1) * s / math.sqrt(reps)     # 99% interval (fewer false alarms)
        theory = erlang_c_W(rho * c * mu, mu, c)
        ok = abs(m - theory) <= half or abs(m - theory) / theory < 0.03
        ok_all &= ok
        details.append(f"rho={rho}: sim W={m:.1f}s ±{half:.1f} (99% CI) vs Erlang-C {theory:.1f}s")
    return _res("M/M/c mean sojourn vs Erlang-C (replications, warm-up deleted)", ok_all, " | ".join(details))


def check_little():
    """Internal consistency: time-average number in system L must equal lambda_hat * W_hat (identity)."""
    jobs, sim, lam, cp = _mmc_run(0.8, 8, 1 / 600.0, 0, horizon_h=100.0)
    W = cp.window_h * HOUR
    L = sim.area_n / W
    lam_hat = len(jobs) / W
    What = np.mean([j.finish - j.arrival for j in jobs])
    ok = all(j.finish is not None for j in jobs) and abs(L - lam_hat * What) / L < 1e-9
    return _res("Little's law L = lambda*W (time average vs customer average)", ok,
                f"L={L:.6f}, lambda*W={lam_hat * What:.6f}")


def check_energy_conservation():
    sc = Scenario(util=0.6, comm_scale=4.0, arrival="bursty")
    m, sim = dcsim.run_one(sc, "P0", 0, record_trace=True)
    tr = sim.trace
    E = sum((t1 - t0) * P for t0, t1, P, *_ in tr)
    ok1 = abs(E / J_PER_KWH - m["energy_kWh"]) / m["energy_kWh"] < 1e-9
    pue = sim.pp.PUE
    ok2 = all(abs(pue * (a + b + c + d) - P) < 1e-6 * max(P, 1) for _, _, P, a, b, c, d in tr)
    ok3 = abs(tr[0][0]) < 1e-9 and abs(tr[-1][1] - sim.W) < 1e-6 and all(abs(tr[i][1] - tr[i + 1][0]) < 1e-9 for i in range(len(tr) - 1))
    carb = sum((t1 - t0) * P / J_PER_KWH * sim.ci.at(t0) for t0, t1, P, *_ in tr)
    ok4 = abs(carb / 1000 - m["carbon_kg"]) / m["carbon_kg"] < 1e-9
    return _res("energy/carbon conservation (trace integral, component sum, contiguous cover)",
                ok1 and ok2 and ok3 and ok4, f"energy {ok1}, components {ok2}, coverage {ok3}, carbon {ok4}")


def check_invariants():
    bad = []
    for pol in ["P0", "P2", "P3", "P4", "P4nc"]:
        for sc in [Scenario(util=0.7, comm_scale=8.0, arrival="bursty"), Scenario(util=0.5, cap_frac=0.7)]:
            try:
                dcsim.run_one(sc, pol, 1, check=True)
            except AssertionError as e:
                bad.append(f"{pol}: {e}")
    return _res("invariants every event (GPU conservation, no job in two states) for all policies", not bad,
                "ok" if not bad else "; ".join(bad))


def check_power_cap():
    sc = Scenario(util=0.7, cap_frac=0.62, ci="synthetic_high_var")
    res = []
    ok = True
    for pol in ["P0", "P2", "P3", "P4"]:
        m, sim = dcsim.run_one(sc, pol, 0, check=True)
        cap = sim.cp.power_cap_W
        ok &= (m["peak_kW"] * 1000 <= cap + 1e-3 and m["cap_exceed_h"] == 0.0)
        res.append(f"{pol}: peak {m['peak_kW']:.1f} kW <= cap {cap / 1000:.1f} kW, blocked {m['cap_block_events']}")
    return _res("power cap is never exceeded (admission control)", ok, " | ".join(res))


def check_crn():
    sc = Scenario(util=0.6)
    h = {pol: dcsim.run_one(sc, pol, 3)["workload_hash"] for pol in ["P0", "P2", "P3", "P4"]}
    same = len(set(h.values())) == 1
    h_other = dcsim.run_one(sc, "P0", 4)["workload_hash"]
    h_rep = dcsim.run_one(sc, "P0", 3)["workload_hash"]
    ok = same and h_other != h["P0"] and h_rep == h["P0"]
    return _res("common random numbers: identical job stream across policies, distinct across replications, repeatable",
                ok, f"hashes {sorted(set(h.values()))[0]} (policies), {h_other} (other rep)")


def check_generators():
    cp = ClusterParams()
    r = []
    ok = True
    lam = 20.0 / HOUR
    # Poisson gaps ~ Exp(lam): KS test
    t = arrivals("poisson", lam, 4000 * HOUR, rng_for(1, 0, "arrival"))
    gaps = np.diff(t)
    ks = stats.kstest(gaps * lam, "expon")
    ok &= ks.pvalue > 0.001
    r.append(f"poisson gaps KS p={ks.pvalue:.3f}")
    # long-run rates
    for model in ["poisson", "bursty", "diurnal"]:
        tt = arrivals(model, lam, 4000 * HOUR, rng_for(1, 1, "arrival"))
        rate = len(tt) / 4000
        ok &= abs(rate - 20.0) / 20.0 < 0.05
        r.append(f"{model} rate {rate:.2f}/h (target 20)")
    tt = arrivals("diurnal", lam, 4000 * HOUR, rng_for(1, 2, "arrival"))
    hod = ((tt / HOUR) % 24).astype(int)
    cnt = np.bincount(hod, minlength=24)
    ratio = cnt[13:16].mean() / cnt[1:4].mean()
    ok &= 4.0 < ratio < 20.0
    r.append(f"diurnal peak/trough ≈ {ratio:.1f} (theory 9)")
    # bursty index of dispersion > 1
    tb = arrivals("bursty", lam, 4000 * HOUR, rng_for(1, 3, "arrival"))
    cb = np.bincount((tb // HOUR).astype(int), minlength=4000)
    iod = cb.var() / cb.mean()
    tp = arrivals("poisson", lam, 4000 * HOUR, rng_for(1, 3, "arrival"))
    cpois = np.bincount((tp // HOUR).astype(int), minlength=4000)
    ok &= iod > 1.5 * cpois.var() / cpois.mean()
    r.append(f"hourly index of dispersion bursty {iod:.1f} vs poisson {cpois.var() / cpois.mean():.2f}")
    # CI scenarios: variability ordering
    lo = dcsim.ci_variability(dcsim.make_ci("synthetic_low_var", 0, 24 * 30))
    hi = dcsim.ci_variability(dcsim.make_ci("synthetic_high_var", 0, 24 * 30))
    ok &= hi["daily_range_over_mean"] > 3 * lo["daily_range_over_mean"]
    r.append(f"CI daily-range/mean low {lo['daily_range_over_mean']:.2f} vs high {hi['daily_range_over_mean']:.2f}")
    return _res("generator distributional checks (KS, long-run rates, burstiness, CI variability)", ok, " | ".join(r))


def check_trace_path():
    """Trace-driven workload: identical across policies, different window per replication, every job finishes."""
    import tempfile, os
    rng = np.random.default_rng(5)
    n = 3000
    df = pd.DataFrame(dict(arrival_s=np.cumsum(rng.exponential(300, n)),
                           n_gpus=rng.choice([1, 2, 4, 8, 16, 32, 64], n, p=[.1, .1, .1, .3, .2, .15, .05]),
                           duration_s=np.exp(rng.normal(np.log(3600), 1.0, n))))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "trace.csv")
        df.to_csv(p, index=False)
        sc = Scenario(util=0.6, trace=p)
        hs = {pol: dcsim.run_one(sc, pol, 0) for pol in ["P0", "P2", "P4"]}
        h1 = dcsim.run_one(sc, "P0", 1)["workload_hash"]
    same = len({m["workload_hash"] for m in hs.values()}) == 1
    done = all(m["n_incomplete"] == 0 for m in hs.values())
    ok = same and done and h1 != hs["P0"]["workload_hash"]
    return _res("trace-driven workload (SYNTHETIC test trace): CRN across policies, window per replication, all jobs finish",
                ok, f"same stream {same}, all complete {done}, rep1 differs {h1 != hs['P0']['workload_hash']}")


def check_rng_streams():
    a = rng_for(7, 0, "arrival").random(5)
    b = rng_for(7, 0, "attrs").random(5)
    c = rng_for(7, 1, "arrival").random(5)
    d = rng_for(7, 0, "arrival").random(5)
    ok = (not np.allclose(a, b)) and (not np.allclose(a, c)) and np.allclose(a, d)
    return _res("RNG streams: separated by stream and replication, reproducible (SeedSequence)", ok, "ok" if ok else "stream clash")


def check_tick_insensitivity(reps=3, tol=0.02):
    """Discretisation insensitivity: halving the scheduler tick should not move the carbon result (WARN, not FAIL)."""
    diffs = []
    for pol in ["P2", "P4"]:
        a = [dcsim.run_one(Scenario(util=0.6, tick_s=3600.0), pol, r)["carbon_dyn_kg"] for r in range(reps)]
        b = [dcsim.run_one(Scenario(util=0.6, tick_s=1800.0), pol, r)["carbon_dyn_kg"] for r in range(reps)]
        diffs.append(abs(np.mean(a) - np.mean(b)) / np.mean(a))
    ok = max(diffs) < tol
    return _res("scheduler tick halved (3600s -> 1800s): carbon unchanged within tolerance", ok,
                f"relative change P2 {diffs[0]:.3%}, P4 {diffs[1]:.3%} (tolerance {tol:.0%})", warn=True)


# ---------------------------------------------------------------------------------------------
# Calibration vs validation of the GPU power model (never validate on the calibration data)
# ---------------------------------------------------------------------------------------------

def calibrate_gpu_power(df: pd.DataFrame, holdout_workloads=None):
    """df columns: workload, frac_compute (0..1), power_w (per GPU).  Model: P = P_idle + (P_max-P_idle)*fc.
    Fit on training workloads, report error on HELD-OUT workloads (leave-one-workload-out if holdout is None)."""
    def fit(d):
        A = np.vstack([np.ones(len(d)), d["frac_compute"].to_numpy()]).T
        coef, *_ = np.linalg.lstsq(A, d["power_w"].to_numpy(), rcond=None)
        return coef[0], coef[0] + coef[1]
    ws = sorted(df["workload"].unique())
    if len(ws) < 3:
        raise ValueError("need at least 3 distinct workloads to calibrate and validate separately")
    rows = []
    for w in (holdout_workloads or ws):
        tr, te = df[df.workload != w], df[df.workload == w]
        pi, pm = fit(tr)
        pred = pi + (pm - pi) * te["frac_compute"].to_numpy()
        err = pred - te["power_w"].to_numpy()
        rows.append(dict(heldout=w, P_idle=pi, P_max=pm, mape=float(np.mean(np.abs(err) / te["power_w"].to_numpy())),
                         rmse_w=float(np.sqrt(np.mean(err ** 2)))))
    pi, pm = fit(df)
    return dict(P_idle_all=pi, P_max_all=pm, validation=pd.DataFrame(rows))


def check_calibration_selftest():
    rng = np.random.default_rng(0)
    rows = []
    for w, lo, hi in [("a", 0.5, 0.95), ("b", 0.6, 1.0), ("c", 0.4, 0.9), ("d", 0.7, 1.0)]:
        fc = rng.uniform(lo, hi, 12)
        rows += [dict(workload=w, frac_compute=f, power_w=90 + 560 * f + rng.normal(0, 8)) for f in fc]
    out = calibrate_gpu_power(pd.DataFrame(rows))
    ok = abs(out["P_idle_all"] - 90) < 30 and abs(out["P_max_all"] - 650) < 30 and out["validation"]["mape"].max() < 0.05
    return _res("calibrate/validate split recovers known parameters on SYNTHETIC data (self-test only)", ok,
                f"fit P_idle={out['P_idle_all']:.0f} W, P_max={out['P_max_all']:.0f} W (truth 90, 650); "
                f"worst held-out MAPE {out['validation']['mape'].max():.2%}")


def run_all(quick=False):
    checks = [check_hand_case, check_hand_case_varying_ci, check_carbon_misaligned_tick, check_congestion_degenerate, check_little, check_energy_conservation,
              check_invariants, check_power_cap, check_crn, check_trace_path, check_rng_streams, check_generators,
              check_calibration_selftest, check_k_extension]
    if not quick:
        checks += [check_mmc, check_tick_insensitivity]
    rep = []
    for fn in checks:
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:                      # a crash is a failure, report it
            r = _res(fn.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
        r["seconds"] = round(time.time() - t0, 1)
        rep.append(r)
    return rep


def print_report(rep):
    for r in rep:
        print(f"[{r['status']:4s}] {r['name']}  ({r['seconds']}s)\n         {r['detail']}")
    n_fail = sum(r["status"] == "FAIL" for r in rep)
    n_warn = sum(r["status"] == "WARN" for r in rep)
    print(f"\n{len(rep)} checks: {len(rep) - n_fail - n_warn} PASS, {n_warn} WARN, {n_fail} FAIL")
    return n_fail == 0