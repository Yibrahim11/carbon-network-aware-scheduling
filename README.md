# Reproducibility package — MSc proposal (Yahaya Ibrahim)

Cross-layer simulation of carbon- and network-aware AI workload scheduling (`dcsim` v0.1.0, seed 20261004).

## Fastest check (a few seconds, needs numpy, pandas, scipy)
    python regenerate_one_row.py
Regenerates one row of the results table from scratch (P2 vs P0, dynamic carbon, Poisson / 50 % load / 1x
communication / synthetic high-variability grid, aligned profile, 10 paired replications) and compares 100
numbers with `results/main_full_meas3_aligned.csv`. Expected: `-20.00 %` and `MATCH` (difference ~1e-16).

## Rows of the proposal's results table (about 10 seconds)
    python regenerate_table_rows.py
Re-simulates the aligned-column rows of the proposal's results table that come from the base cell (P2 vs P0 dynamic
carbon -14.4 %; P4 vs P2 dynamic carbon -0.1 % and mean completion time -23.9 %) plus the row above, checks every
re-simulated run against the stored raw output, and recomputes each paired difference. Expected: `ALL ROWS MATCH`.
`docs/SUPERVISOR_SESSION.md` is the plan for regenerating a row together.

## The proposal's current results table (both columns, about 30 seconds)
    python regenerate_meas6_rows.py
Re-simulates every row of the results table in proposal v5 (runs tagged `meas6_*`, dcsim 0.3.0, `code_v0.3.0/`) and
checks each re-simulated run against the stored output. Expected: `ALL ROWS MATCH`. `any <file> <row>` re-simulates any
single stored run. The earlier scripts check earlier runs: `regenerate_table_rows.py` the aligned run of proposal v3
(dcsim 0.1.0, `code/`), `regenerate_interleaved_rows.py` the interleaved Google Colab run (`meas5`, dcsim 0.2.0).

## Which run is which
| Tag | Simulator | Where | Status |
|---|---|---|---|
| `meas3_aligned` | 0.1.0 (`code/`) | Colab | superseded; behind proposal v3 |
| `meas3_interleaved_LOCALRUN` | 0.1.0 | workstation, 3-decimal profile | superseded |
| `meas5_interleaved` | 0.2.0 (`code_v0.2.0/`) | Colab, official profile | superseded (its pod-size runs at 1x are identical to `meas6`) |
| `meas6_aligned`, `meas6_interleaved` | 0.3.0 (`code_v0.3.0/`) | workstation, notebook v8 in full mode | **current**: behind proposal v5 |
| `meas6_interleaved` (Colab) | 0.3.0 | Google Colab, notebook v8 in full mode, 23 Oct 2026 | **current**: all 3,060 main and sensitivity runs identical to the workstation run (difference 0); adds the trace-driven and pod-size runs (`trace_`, `podtrace_` files) and `DIGEST_full_meas6_interleaved_colab.txt` |

What changed in 0.3.0: communication intensity (`comm_scale`) rescales the whole measured communication time, including
the part inside the uncongested iteration time. Up to 0.2.x it rescaled only inter-pod volume and the congestible time,
so away from 1x the placement penalty stayed at its 1x value and congestion above 1x was overstated. Results at 1x do not
change. 0.3.0 also reports `net_inflation_total` (runtime over the runtime inside one pod), next to `net_inflation`
(congestion only). 17 verification checks.

## Review of 3 November 2026 (proposal v7)
`analysis/review_2026-11/` holds the analyses behind proposal v7: runs at the operating point both placement bounds sustain,
P4 against P3 as well as P2, Holm-corrected primary comparisons, the 15-minute power ramp rate, and the decision-rule
boundary. See its README for the order. Package version 6.

## Layout
- `code/` simulator (`dcsim.py`), experiment/analysis layer (`dcsim_exp.py`), 15 verification checks (`dcsim_verify.py`), notebook generator.
- `notebooks/msc_simulation_v4.ipynb` the pipeline that produced the stored aligned results (Google Colab free CPU).
- `notebooks/msc_simulation_v7.ipynb` current pipeline (dcsim 0.2.0): pod-size experiment, outputs tagged by profile variant,
  input-fingerprint guard, and corrected handling of jobs split over 3 or more pods (measured `spread_k3`/`spread_k4` rows
  first, otherwise a flat-ring extension of the 2-pod row; `k_rule='hier'` keeps the v0.1.0 behaviour). `code/` stays at
  0.1.0, the version that produced the stored results; with it replaced by the 0.2.0 code, both regeneration scripts still
  give `MATCH` (checked 22 Oct 2026), because the default rule is unchanged.
- `colab_cells/astra_profile_cell_kpods.py` measures the 3- and 4-pod splits in ns-3 (run after the 32/64-NPU cell).
- `code_v0.2.0/` the simulator in notebook v7, which produced `meas5_interleaved` (also used by `analysis/k_rule_comparison.py`).
- `code_v0.3.0/` the current simulator (notebook v8), which produced `meas6_*`.
- `notebooks/msc_simulation_v8.ipynb` current pipeline (dcsim 0.3.0).
- `analysis/` the rule comparison behind the corrected values in the proposal's table note (indicative: 3-decimal
  interleaved profile, stand-in trace), with its raw runs, and `make_onset_both.py`, which redraws Figure 1 (`onset_both.png`) from the stored runs.
- `colab_cells/` input builders: ns-3 profiles (8/16 and 32/64 GPUs), merge cell, Alibaba v2026 trace, GPU power, carbon data.
- `data/` `astra_profiles_aligned.csv` (exact table used for the aligned run, exported from `results/params_full.json`),
  `ci_ghana_derived.csv`, `gpu_power.csv` (T4 measurement), plus a 3-decimal reconstruction of the interleaved table.
- `results/` raw per-run outputs (one row per run), digests, parameter snapshot, verification report, `pip_freeze.txt`.
- `docs/` reference check note (15 and 21 Oct 2026), supervisor session plan.
- `MANIFEST.json` SHA-256 of every file.

## Status of each result file
| File | Origin |
|---|---|
| `main_full_meas3_aligned.csv`, `sens_...`, `trace_...` | Colab full run (10 replications), aligned profile. Primary evidence. |
| `*_interleaved_LOCALRUN.csv`, `DIGEST_full_interleaved_local.txt` | Worst-case profile run on a 2-CPU workstation with a 3-decimal reconstruction of the profile table and no trace file. **To be repeated on Colab with the official profile file before it is cited as final.** |

## Not included (add before submission)
- `raw_ns3_runs.csv` (24 raw ns-3 measurements, 40 after the 3/4-pod cell). The profile tables themselves are now in `data/`
  (`astra_profiles_interleaved.csv` exported from the Colab run's parameter snapshot).
- `trace.csv`, needed only to re-simulate the trace-driven and pod-size runs (rebuild with `colab_cells/alibaba_trace_v2026_cell.py`; needs the public Alibaba v2026 job summary, about 1.2 GB).
- The ns-3 / ASTRA-sim commit used (submodule hashes are in the build log; record them here).

## Known assumptions (all labelled in the proposal)
Compute time per iteration (0.5 / 1.0 / 2.0 s), server overhead 800 W, PUE 1.3, Poisson arrivals for the trace,
assumed emission factors for the Ghana scenario, congestion curve and fabric power presets.


## How to cite
See `CITATION.cff`. Code: MIT licence; data (`data/`, `data_build/`, `results/`): CC BY 4.0 (see `LICENSE`).
The archived version and its DOI are on Zenodo; the DOI is given in Section 3.13 of the thesis.

## Final experiments (thesis Chapter IV; dcsim 0.4.0, 22 Nov 2026)
    python code_v0.4.0/dcsim_verify.py          # 20 checks (adds persistence forecast, V-family, power shape)
    python final/final_design.py aligned        # 4,460 + 120 runs, ~10 min on two cores; checkpointed
    python final/final_design.py interleaved
    python final/final_analysis.py              # final/out/*.csv, summary.json (every number in Chapter IV)
    python final/figs_final.py                  # Figures 3.2, 4.1-4.4, 4.6-4.8 (final/figs)
`code_v0.4.0` with default options reproduces `code_v0.3.0` exactly (relative difference 0). Headline forecast:
day-ahead persistence. All carbon-intensity inputs are synthetic or modelled (no measured hourly series). The trace
version of the pod-size experiment is from the preliminary runs (`results/podtrace_*_meas6_*`), because the Alibaba trace
host is not reachable from the environment used for the final runs; since v8.1.0 it is a diagnostic only (see below).

## v8.1.0 — revision after the supervisor's review (27 Nov 2026)
    python final/podsize_design.py aligned      # final pod-size experiment, synthetic mix, pods 32/16/8: 480 runs, <1 min
    python final/podsize_design.py interleaved
    python final/final_analysis.py              # decision points and A* now from FINAL runs only
    python final/figs_final.py                  # Figures 3.2 and 4.1-4.8 (4.5 and 4.6 redrawn)
- Review comment M3: the decision-rule boundary A* was estimated partly from the preliminary trace pod-size runs (dcsim
  0.3.0, perfect forecasts). It is now estimated from the 66 distinct cells of the final runs (main factorial, common point,
  one-factor sweeps and the new final pod-size experiment): A* between 0.044 and 0.074, no cell misordered; rank correlation
  0.93 for A against 0.76 for total inflation. The 12 preliminary trace cells are written to
  `final/out/decision_points_trace_diagnostic.csv`, plotted as hollow markers in Figure 4.6, and all lie on the correct side.
- Nothing else in `final/out/` changed (byte-identical to v8.0.0 except `decision_points.csv` and `summary.json`).
- `analysis/thesis_2026-11/figs_thesis.py` no longer draws Figures 4.5 and 4.6 (now in `final/figs_final.py`); the label of
  the preliminary Ghana series in Figure 3.5 reads "Ghana-derived (preliminary)", as in Table 3.5.
