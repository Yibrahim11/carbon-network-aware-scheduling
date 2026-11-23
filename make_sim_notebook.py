import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


def module_cell(name):
    src = open(os.path.join(HERE, name)).read()
    code(f"%%writefile {name}\n" + src)


md("""
# Carbon-, power- and network-aware scheduling: simulation notebook (free Colab)

**What this runs:** a two-level simulator. A cluster-level discrete-event simulator (queue, EASY backfill, GPU pods, policies **P0, P2, P3, P4**) consumes a **job profile table** (iteration time, compute fraction, inter-pod traffic). The profile table comes from ASTRA-sim when you have it, and from a clearly labelled synthetic placeholder until then. A fluid pod-uplink congestion model couples jobs that share the network; a phase-dependent power model plus carbon intensity turns it into energy and carbon.

**How to use it (about 3 cells of effort):**
1. Set `MODE` in the first code cell: `"quick"` (about 2 to 4 minutes) or `"full"` (about 15 to 25 minutes on free Colab).
2. **Runtime > Run all.**
3. Copy the text block printed by the last cell (`=== RESULT DIGEST ===`) back to Claude. The zip it also builds contains everything else.

Everything checkpoints to Google Drive, so a disconnect costs you nothing: run all again and finished runs are skipped.

**Honesty box.** Until ASTRA-sim profiles, calibrated GPU power, and real carbon data are plugged in (optional hooks below), the numbers are *pipeline tests with ILLUSTRATIVE parameters*, not findings. The digest says so in its first lines.
""")

code("""
import os, sys, json, time, platform
MODE = "quick"          # "quick" or "full"
MASTER_SEED = 20261004  # recorded in the results; replications use SeedSequence streams derived from it

# ---- persistent storage (falls back to a local folder when not on Colab) ----
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/msc_sim'
    ON_COLAB = True
except Exception:
    BASE = os.path.abspath('./msc_sim_local')
    ON_COLAB = False
for sub in ('results', 'data', 'figures', 'bundles'):
    os.makedirs(f'{BASE}/{sub}', exist_ok=True)
print('storage:', BASE, '| colab:', ON_COLAB, '| python', platform.python_version(), '| cpus', os.cpu_count())

# ---- OPTIONAL real inputs: drop these files in BASE/data (all optional; absent = placeholder used) ----
PROFILE_VARIANT   = 'aligned'   # 'interleaved' = WORST-CASE rank placement (every ring hop crosses pods); 'aligned' = BEST-CASE (one crossing). Report both: they bracket reality.
ASTRA_PROFILE_CSV = f'{BASE}/data/astra_profiles.csv' if PROFILE_VARIANT == 'interleaved' else f'{BASE}/data/astra_profiles_aligned.csv'   # jtype,n_gpus,pclass,iter_time_s,frac_compute,interpod_GB,source
GHANA_CI_CSV      = f'{BASE}/data/ghana_ci.csv'        # timestamp,gco2_per_kwh   (hourly)
TRACE_CSV         = f'{BASE}/data/trace.csv'           # arrival_s,n_gpus,duration_s  (production GPU trace)
TARGET_GPU_IDLE_W = 132    # W/GPU = (1860-800)/8: 8-GPU H100 node idle 1.86 kW (arXiv 2506.14551) minus ASSUMED 800 W host overhead (P_server)
TARGET_GPU_MAX_W  = 874    # W/GPU-slot = (7790-800)/8: highest observed node average 7.79 kW (same paper); includes the GPU's share of node fan/switch load
GPU_POWER_CSV     = f'{BASE}/data/gpu_power.csv'       # workload,frac_compute,power_w   (per-GPU measurements)
""")

md("## 1. Simulator source (written next to the notebook)\nThree small modules. Re-running this section overwrites them and reloads cleanly.")
module_cell("dcsim.py")
module_cell("dcsim_verify.py")
module_cell("dcsim_exp.py")

code("""
import importlib, numpy as np, pandas as pd, scipy, matplotlib
import dcsim, dcsim_verify, dcsim_exp
for m in (dcsim, dcsim_verify, dcsim_exp):
    importlib.reload(m)
import dcsim, dcsim_verify, dcsim_exp
versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, dcsim=dcsim.__version__)
print(versions)
""")

md("""
## 2. Model in one page (what an examiner will ask first)

**Level of abstraction (Ch. 1.3 of the course text):** packet and wavelength detail are *not* simulated. Jobs run at *iteration* granularity; the network is a *fluid* model on pod uplinks; the optical fabric is an abstraction (capacity, port power, circuit set-up delay).

| Symbol | Meaning |
|---|---|
| `it`, `fc`, `G` | uncongested iteration time, compute fraction, per-pod-uplink GB per iteration (from the profile table) |
| `k` | pods a job spans (k=1 compact, k>=2 spread); inter-pod volume scales as 2(k-1)/k |
| `d_j = G_j / it_j` | uncongested uplink demand of job j on each pod it spans (GB/s) |
| `load_p = sum_j d_j` | demand on pod p's uplink; `rho_p = load_p / uplink_GBps` |
| `s_j = max_p curve(rho_p)` | congestion slowdown of job j (default curve = processor sharing, `max(1, rho)`; replace with an ns-3 fit) |
| `comm` | congestible (inter-pod) communication time per iteration at no congestion, hidden or exposed (profile column `comm_time_s`) |
| `it_eff = it + comm * (s-1)` | congested iteration time; conservative: assumes overlap has no slack, so even fully hidden communication becomes exposed once `s>1` |
| `P_gpu = n * (P_idle + (P_max - P_idle) * t_comp/it_eff)` | phase-dependent GPU power; more waiting = lower power but longer runtime |
| `P_total = PUE * (P_gpu_alloc + idle_GPUs*P_idle + servers + network)` | facility power; **idle GPUs and servers are always counted** |
| `Carbon = sum_t P_total(t) * dt * CI(t)` | exact piecewise integral (events include every CI hour boundary) |

**Study type:** *terminating* simulation. The cluster starts empty, 7 days of arrivals, 11 days of accounting window so every job can finish; runs where a job does not finish are flagged *censored* and dropped from paired comparisons. There is no warm-up to delete (course text Ch. 9.5), and confidence intervals come from independent replications.

**Policies.**
- **P0** baseline: FIFO + EASY backfill, first-fit GPU fill, no awareness.
- **P2** carbon-aware: shifts deferrable jobs to the lowest-forecast-CI start within deadline slack (only if the saving is at least 5%, to avoid a straw-man); network- and power-oblivious. *P2raw* = no 5% threshold.
- **P3** network-aware: starts ASAP, picks the placement minimising own runtime plus slowdown imposed on running jobs.
- **P4** joint: cost in gCO2e = carbon (with congestion-adjusted runtime and port power) + externality on running jobs + small delay price `w_delay`; compares start-now against hourly future starts using forecast free GPUs, claims of earlier queued jobs, congestion from jobs still active, and the power cap. *P4nc* = ablation without queue-claim awareness.
- **Power cap:** facility admission control for every policy (a job starts only if the cap holds).

**Randomness (Ch. 2, 10):** `SeedSequence([master, replication, stream])`, separate streams for arrivals, job attributes, carbon, and forecast noise. **All policies see the identical job stream in a replication (common random numbers)**, so differences are *paired* and reported with t-intervals.
""")

md("## 3. Verification battery (run before trusting anything)")
code("""
t0 = time.time()
VERIFY = dcsim_verify.run_all(quick=(MODE == "quick"))
ok = dcsim_verify.print_report(VERIFY)
json.dump(VERIFY, open(f'{BASE}/results/verification_report.json', 'w'), indent=1)
print('\\nverification time %.0fs' % (time.time() - t0))
assert ok, 'A verification check FAILED: do not use results until fixed (see report above).'
""")

md("""
## 4. Job profile source (the GPU-to-network link)
Default = synthetic analytic profile (labelled). **To use ASTRA-sim results**, build `astra_profiles.csv` with one row per `(jtype, n_gpus, pclass)` where `pclass` is `compact` (one pod) or `spread` (two pods): iteration time, compute fraction (`1 - exposed_comm/total`), GB per iteration crossing each pod uplink, and (recommended) `comm_time_s` = inter-pod communication time per iteration, hidden or exposed (if absent, the exposed time is used, which understates congestion sensitivity). `dcsim.profile_row_from_astra(...)` converts ASTRA-sim's per-NPU output. **Assumption to verify:** 1 ASTRA-sim cycle = 1 ns.
""")
code("""
cluster0, power0 = dcsim.ClusterParams(), dcsim.PowerParams()
if os.path.exists(ASTRA_PROFILE_CSV):
    PROFILE_DF = pd.read_csv(ASTRA_PROFILE_CSV)
    print('using ASTRA-sim profile table:', ASTRA_PROFILE_CSV, len(PROFILE_DF), 'rows')
    _syn = dcsim.synthetic_profile(cluster0)
    _have = set(zip(PROFILE_DF.jtype, PROFILE_DF.n_gpus.astype(int), PROFILE_DF.pclass))
    _fill = _syn[[(a, int(b), c) not in _have for a, b, c in zip(_syn.jtype, _syn.n_gpus, _syn.pclass)]].copy()
    if len(_fill):
        _fill['source'] = 'synthetic-fill'
        PROFILE_DF = pd.concat([PROFILE_DF, _fill], ignore_index=True)
        print('WARNING: ASTRA table lacks', len(_fill), 'needed (jtype,n_gpus,pclass) keys -> filled with SYNTHETIC rows:')
        print(_fill[['jtype', 'n_gpus', 'pclass']].to_string(index=False))
else:
    PROFILE_DF = None        # run_one builds the synthetic table per scenario (comm_scale applied)
    print('NO astra_profiles.csv found -> SYNTHETIC placeholder profile (pipeline test only)')
prof_show = PROFILE_DF if PROFILE_DF is not None else dcsim.synthetic_profile(cluster0)
PROFILE_SOURCE = str(prof_show['source'].iloc[0]) if prof_show['source'].nunique() == 1 else 'astra-sim+synthetic-fill (' + ', '.join(f'{k}={v}' for k, v in prof_show['source'].value_counts().items()) + ')'

# Go/no-go Test 1 diagnostic: does placement change iteration time?
piv = prof_show.pivot_table(index=['jtype', 'n_gpus'], columns='pclass', values='iter_time_s')
if 'compact' in piv and 'spread' in piv:
    piv['spread/compact'] = piv['spread'] / piv['compact']
    piv['Test1 (>=1.10)'] = np.where(piv['spread/compact'] >= 1.10, 'PASS', 'no')
print(piv.round(3).to_string())
if PROFILE_SOURCE.startswith('synthetic'):
    print('\\nNOTE: with a synthetic profile this ratio is true BY CONSTRUCTION. Test 1 is only evidence when the table comes from ASTRA-sim.')

# ASTRA-sim output parser self-test on the docs' published sample lines
sample = chr(10).join(f'[workload] [info] sys[{i}] finished, 64440 cycles, exposed communication 64440 cycles.' for i in range(8))
cy, ex = dcsim.parse_astra_output(sample)
print('\\nparser self-test:', dcsim.profile_row_from_astra('docs_allreduce_1MB', 8, 'compact', cy, ex, 1, 0.0))
""")

md("""
## 5. Carbon data and Test 2 (is carbon intensity variable enough to matter?)
Put hourly CSVs in `BASE/data/` named `ghana_ci.csv` or `ci_<anything>.csv` (columns `timestamp,gco2_per_kwh`). Each one becomes an extra scenario in every design, named after the file. The two synthetic scenarios are **structural stand-ins** (low vs high daily variability), not any real grid.
""")
code("""
CI_LIST = ['synthetic_low_var', 'synthetic_high_var']
rows = {k: dcsim.ci_variability(dcsim.make_ci(k, 0, 24 * 28)) for k in CI_LIST}
import glob
_real = sorted(set(([GHANA_CI_CSV] if os.path.exists(GHANA_CI_CSV) else []) + glob.glob(f'{BASE}/data/ci_*.csv')))
for _f in _real:
    _g = pd.read_csv(_f)
    _name = os.path.splitext(os.path.basename(_f))[0]
    rows[_name + ' (real, ' + str(len(_g)) + ' h)'] = dcsim.ci_variability(_g.iloc[:, 1].astype(float).to_numpy())
    CI_LIST.append('csv:' + _f)
if not _real:
    print('no real carbon-intensity CSV found (ghana_ci.csv or ci_*.csv in data/): only synthetic CI scenarios will run')
else:
    print('real carbon-intensity series used:', [os.path.basename(f) for f in _real])
tab = pd.DataFrame(rows).T
print(tab.round(2).to_string())
print('\\nTest 2 reading: daily_range_over_mean well below ~0.1 means temporal shifting has little to exploit.')
""")

md("## 6. GPU power: calibrate on some workloads, validate on held-out ones (optional)")
code("""
POWER_BASE = dcsim.PowerParams()
if os.path.exists(GPU_POWER_CSV):
    cal = dcsim_verify.calibrate_gpu_power(pd.read_csv(GPU_POWER_CSV))
    print(f"fitted P_idle={cal['P_idle_all']:.0f} W, P_max={cal['P_max_all']:.0f} W")
    print(cal['validation'].round(4).to_string())
    _pi, _pm = float(cal['P_idle_all']), float(cal['P_max_all'])
    if TARGET_GPU_MAX_W and TARGET_GPU_IDLE_W:
        print(f'TARGET GPU: idle {TARGET_GPU_IDLE_W} W and max {TARGET_GPU_MAX_W} W from datasheet/published data; the fit above only validates the LINEAR SHAPE (held-out error table)')
        _pi, _pm = float(TARGET_GPU_IDLE_W), float(TARGET_GPU_MAX_W)
    elif TARGET_GPU_MAX_W:
        _k = TARGET_GPU_MAX_W / _pm
        print(f'WARNING: only TARGET_GPU_MAX_W set: the measured idle FRACTION ({_pi/_pm:.0%}) is carried over to the target GPU. Large data-centre GPUs idle at a much lower fraction of max: set TARGET_GPU_IDLE_W too.')
        print(f'SCALING measured GPU ({_pm:.0f} W max) to target GPU max {TARGET_GPU_MAX_W} W (x{_k:.2f}): the linear SHAPE is validated on the measured GPU, the absolute watts of the target come from its datasheet')
        _pi, _pm = _pi * _k, _pm * _k
    else:
        print('TARGET_GPU_MAX_W not set: simulation uses the MEASURED GPU watts as-is (check they match the GPU you are simulating)')
    POWER_BASE.P_gpu_idle, POWER_BASE.P_gpu_max = _pi, _pm
else:
    print('no gpu_power.csv: P_gpu_idle/P_gpu_max stay ILLUSTRATIVE', POWER_BASE.P_gpu_idle, POWER_BASE.P_gpu_max,
          '(calibrate on part of your measurements, report error on the HELD-OUT part)')
""")

md("## 7. Experiments (designed, replicated, checkpointed)")
code("""
R = 3 if MODE == "quick" else 10
RUN_TAG = (f'meas3_{PROFILE_VARIANT}' if PROFILE_DF is not None else 'syn3')      # new result files: old checkpoints (other comm levels / profile) are never mixed in
WORKERS = max(1, os.cpu_count() or 1)
dcsim_exp.set_comm_levels(PROFILE_DF is not None)      # measured table: comm_scale 1.0 = the measured cost
dcsim_exp.set_stress_levels(PROFILE_VARIANT == 'interleaved' and PROFILE_DF is not None)   # worst-case profile: lower load so runs can drain
print('utilisation levels: main', dcsim_exp.UTIL_MAIN, '| sweep', dcsim_exp.UTIL_SWEEP, '| base', dcsim_exp.BASE_UTIL)
print('comm_scale levels: main', dcsim_exp.COMM_MAIN, '| sweep', dcsim_exp.COMM_SWEEP, '| base', dcsim_exp.BASE_COMM)
design_main = dcsim_exp.main_design(R=R, cis=tuple(CI_LIST))
if MODE == "quick":
    design_sens = [r for r in dcsim_exp.sensitivity_design(R=R, policies=dcsim_exp.POLICIES_MAIN)
                   if r['tag'] in ('sens:comm_scale', 'sens:util')]
else:
    design_sens = dcsim_exp.sensitivity_design(R=R, ci_extra=[c for c in CI_LIST if c.startswith('csv:')])
design_trace = dcsim_exp.trace_design(TRACE_CSV, R=R) if os.path.exists(TRACE_CSV) else []
if design_trace:
    print('trace-driven design included:', TRACE_CSV)
t0 = time.time()
kw = dict(workers=WORKERS, profile_df=PROFILE_DF, master_seed=MASTER_SEED, power_base=POWER_BASE)
dcsim_exp.run_design(design_main, f'{BASE}/results/main_{MODE}_{RUN_TAG}.csv', **kw)
if design_trace:
    dcsim_exp.run_design(design_trace, f'{BASE}/results/trace_{MODE}_{RUN_TAG}.csv', **kw)
dcsim_exp.run_design(design_sens, f'{BASE}/results/sens_{MODE}_{RUN_TAG}.csv', **kw)
print('experiment time %.0fs' % (time.time() - t0))
""")

code("""
# Best-known benchmark for the P4 heuristic on a 5-job instance (local search; not a proof of optimality)
t0 = time.time()
ORACLE_DF, _ = dcsim_exp.oracle_benchmark(comm_scale=12.0, restarts=(40 if MODE == 'quick' else 150))
print(ORACLE_DF.round(2).to_string())
print('benchmark time %.0fs' % (time.time() - t0))
""")

md("## 8. Analysis: paired differences with 95% intervals, mechanism plots")
code("""
main = dcsim_exp.load_runs(f'{BASE}/results/main_{MODE}_{RUN_TAG}.csv')
sens = dcsim_exp.load_runs(f'{BASE}/results/sens_{MODE}_{RUN_TAG}.csv')
TRACE = dcsim_exp.load_runs(f'{BASE}/results/trace_{MODE}_{RUN_TAG}.csv') if os.path.exists(f'{BASE}/results/trace_{MODE}_{RUN_TAG}.csv') else None
TABS = dcsim_exp.make_tables(main, sens, TRACE)
for k, v in TABS.items():
    v.to_csv(f'{BASE}/results/paired_{k}_{MODE}.csv', index=False)
FIGS = dcsim_exp.save_figures(main, sens, TABS, f'{BASE}/figures')
def show_img(p):
    try:
        from IPython.display import Image, display
        display(Image(filename=p))
    except Exception:
        print('figure saved:', p)
for p in FIGS:
    show_img(p)
""")

md("## 9. Bundle and digest (copy the digest block back to Claude)")
code("""
import zipfile, subprocess
_ps = ('published H100 node data (arXiv 2506.14551), linear shape validated on a T4' if (os.path.exists(GPU_POWER_CSV) and TARGET_GPU_MAX_W and TARGET_GPU_IDLE_W) else ('MEASURED on the calibration GPU as-is' if os.path.exists(GPU_POWER_CSV) else 'ILLUSTRATIVE placeholders'))
META = dict(power_source=f'GPU idle {POWER_BASE.P_gpu_idle:.0f} W / max {POWER_BASE.P_gpu_max:.0f} W ({_ps}); server {POWER_BASE.P_server:.0f} W ASSUMED; PUE {POWER_BASE.PUE} ASSUMED', code_version=dcsim.__version__, seed=MASTER_SEED, mode=MODE, R=R, profile_source=PROFILE_SOURCE,
            ci_list=', '.join(os.path.basename(c) if c.startswith('csv:') else c for c in CI_LIST))
DIGEST = dcsim_exp.digest(main, sens, ORACLE_DF, VERIFY, META, TABS, TRACE)
open(f'{BASE}/results/DIGEST_{MODE}.txt', 'w').write(DIGEST)

params = dict(cluster=dcsim.asdict(cluster0), power=dcsim.asdict(POWER_BASE), fabrics=dcsim.FABRICS,
              job_types=dcsim.JOB_TYPES, meta=META, versions=versions,
              profile_table=prof_show.to_dict(orient='records'))
json.dump(params, open(f'{BASE}/results/params_{MODE}.json', 'w'), indent=1, default=str)
try:
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout
except Exception:
    freeze = ''
open(f'{BASE}/results/pip_freeze.txt', 'w').write(freeze)

REPRO = f'''# Reproducibility statement (course text Ch. 13.4 checklist), mode={MODE}
1. Model, assumptions, abstractions: terminating DES; fluid pod-uplink congestion; optical fabric abstracted to capacity, port power and set-up delay; constant PUE; no preemption or migration; single site.
2. Simulator and version: dcsim {dcsim.__version__}; every parameter including defaults is in params_{MODE}.json; Python/package versions in pip_freeze.txt.
3. Seeds, replications, warm-up, run length: master seed {MASTER_SEED}; SeedSequence([seed, replication, stream]); R={R} replications; no warm-up (terminating); arrivals over 168 h, accounting window 264 h.
4. Confidence intervals: t-interval over independent replications; paired differences under common random numbers; runs with unfinished jobs are censored and excluded from paired comparisons (counts reported).
5. Validation experiments shown: M/M/c against Erlang-C; Little's law; hand-computed energy and carbon cases (constant and hourly-varying CI); congestion degenerate cases; energy conservation. NOT validated against a real facility; profile source = {PROFILE_SOURCE}.
6. Limitations: see digest header and the notebook limitations section. Workload mix, durations, slack, power and fabric numbers are ILLUSTRATIVE unless replaced by the optional data files.
'''
open(f'{BASE}/results/REPRODUCIBILITY_STATEMENT.md', 'w').write(REPRO)

zpath = f'{BASE}/bundles/results_{MODE}_{time.strftime("%Y%m%d_%H%M%S")}.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for folder in ('results', 'figures'):
        for fn in os.listdir(f'{BASE}/{folder}'):
            z.write(f'{BASE}/{folder}/{fn}', f'{folder}/{fn}')
    for fn in ('dcsim.py', 'dcsim_verify.py', 'dcsim_exp.py'):
        z.write(fn, f'code/{fn}')
print('bundle:', zpath)
try:
    from google.colab import files
    files.download(zpath)
except Exception:
    pass
print()
print(DIGEST)
""")

md("""
## 10. Where each earlier gap is closed

| Gap raised | Closed by |
|---|---|
| "Optical" undefined | Fabric abstraction (`FABRICS`: capacity, port power, set-up delay), no wavelength-level claims |
| No hypothesis or mechanism | Congestion model + metrics `net_inflation`, `spread_frac`, `ci_at_start_gpuw`; onset sweep over `comm_scale` |
| No power-aware element | Facility cap (admission control, tested), peak power, idle GPU and server power always counted |
| P4 undefined | `P4` cost function and algorithm in the source; `P4nc` ablation; best-known benchmark |
| Equation 8 / routing variable | Per-pod uplink load `load_p` and slowdown `s_j` (model table above) |
| Ghana variability unknown | Test 2 table + `ghana_ci.csv` hook; synthetic low/high variability contrast |
| Traces lack communication data | `trace.csv` hook; communication intensity assigned and stated; `comm_scale` sensitivity |
| Validation vs calibration | `calibrate_gpu_power` fits on some workloads and reports error on held-out ones |
| Single seed / no intervals (course text Ch. 10-11) | Replications, SeedSequence streams, t-intervals, CRN pairing, censoring flag |
| Terminating vs steady state (Ch. 9.5) | Declared terminating; no warm-up; censoring handled |
| Verification battery (Ch. 13.2) | 14 checks (12 in quick mode) incl. M/M/c vs Erlang-C, Little's law, hand cases, conservation, degenerate cases, generators |
| Traffic model drives conclusions (Ch. 10.5) | Poisson, bursty (MMPP), diurnal arrivals and trace-driven mode; arrival factor in the designs |
| Factorial design, not wandering (Ch. 12.4) | `main_design` (full factorial, few levels) plus one-factor sweeps; all raw results saved with configuration |
| Idle power missing | Idle GPUs and servers always counted; `carbon_dyn_kg` isolates the policy-dependent part |
| Reporting checklist (Ch. 13.4) | Auto-generated `REPRODUCIBILITY_STATEMENT.md`, `params.json`, `pip_freeze.txt`, versioned code in the zip |
| Discretisation insensitivity | Scheduler tick halving check (full mode) |

## 11. Limitations (written before a reviewer writes them)
- **Profile table.** Synthetic until ASTRA-sim rows are supplied. ASTRA-sim reports cycles; the 1 cycle = 1 ns conversion is an assumption to verify.
- **Congestion** is a fluid, bottleneck-pod approximation using uncongested demand; intra-pod fabric is non-blocking; spreads over k>2 pods only rescale traffic. Calibrate `congestion_pts` from ns-3 runs of concurrent jobs.
- **Optical layer** is an abstraction; there is no wavelength assignment, no physical-layer model, and reconfiguration delay is paid once per spread job.
- **Workloads.** Job mix, durations, deferrable fraction and slack are ILLUSTRATIVE; traces lack communication data, so intensity is assigned.
- **Forecasts** are perfect by default (`forecast_sigma` adds multiplicative noise in the sensitivity design).
- **Scope.** Single site, no preemption, migration or checkpointing, constant PUE, training jobs only.
- **Benchmark** is a best-known solution from local search on 5 jobs, not a proof of optimality.
""")

nb = {"cells": cells, "metadata": {"colab": {"name": "msc_simulation.ipynb"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "msc_simulation.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
