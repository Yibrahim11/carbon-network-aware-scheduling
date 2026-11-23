# Analyses for the supervisor's review of 3 November 2026 (proposal v7)

Run in this order (about 2 minutes on 2 CPUs; needs numpy, pandas, scipy, matplotlib):

    python analyse.py          # 320 runs at matched operating points, with 15-minute power traces -> matched_runs.csv, traces.json
    python extra.py            # 400 runs: low-variability and Ghana grids at 30% load; aligned communication sweep at 30% -> extra_runs.csv
    python ramp_holm.py        # ramp rates (-> matched_runs_ramp.csv) and the paired primary comparisons
    python primary_final.py    # Table 3: Holm-adjusted p-values, raw and Bonferroni intervals; H3 by Welch test
    python tables.py           # all paired comparisons by cell (P4 vs P2 and vs P3), energy bound, run times
    python decision_points.py  # Figure 2 data: avoidable network inflation A vs P3 completion-time gain
    python figs.py             # Figure 1 (onset_both.png) and Figure 2 (decision_boundary.png)

All runs use dcsim 0.3.0 (`../../code_v0.3.0/`), the packaged profile tables and seed 20261004. The base cells re-simulated
here (aligned 70% load / 1x, interleaved 30% / 0.5x) are identical to the stored meas6 runs (difference 0).
Common operating point of the primary comparisons: Poisson arrivals, 30% load, 1x communication, high-variability grid.
