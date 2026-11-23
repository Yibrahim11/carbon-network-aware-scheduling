# Thesis analysis (10 Nov 2026)

Analytical results of Chapter III, Section 3.10, and every thesis figure. Uses dcsim 0.3.0 (`code_v0.3.0/`), the profile
tables and data in `data/`, and the stored runs in `results/` and `analysis/review_2026-11/`.

    python analytic.py          # about 1 minute: analytic_results.json, analytic_traffic.csv, analytic_energy.csv, analytic_bound.csv
    python figs_thesis.py       # fig3_2 ... fig4_6 as PDF and PNG (needs analytic_bound.csv)

| Result | Check | Outcome |
|---|---|---|
| R1 ring traffic per uplink: aligned 2(n-1)G/n, interleaved (n-1)G | against ns-3 inter-pod traffic, 14 configurations | exact |
| R2 congestion onset m* = C t / g | evaluated from the profile tables | Table 3.6 |
| R3 GPU energy invariance (Eq. 3.16) | 80 runs at the common operating point | max rel. error 1.4e-6; total-energy differences = network energy |
| R4 deferral bound S <= phi (V/2) sinc(pi d/24) | closed form vs exhaustive search; oracle vs simulated P2, 60 replications | within 0.3 points; P2 below oracle in every replication |

After the final experiments, point the two scripts at the new result files and rerun; the figure captions and the
numbers quoted in Chapters III and IV must then be updated from `analytic_results.json` and the printed output.

## Ghana series anchored to the national statistics (11 Nov 2026)

    python ../../data_build/ghana_ci_ec.py ../../data_build   # anchors for 2024 and 2025 + eight bounding hourly series per year
    python ghana_ec_check.py                                  # oracle and simulated P2 at the common operating point

Sources: Energy Commission of Ghana, 2026 Energy Statistics (Tables 3.7, 6.24, 6.25, 6.26); IPCC 2006 default emission
factors; PPTC 2025 Electricity Supply Plan (peak at 20:00, s.2.2; Akosombo lead load-follower, ch.1; off-peak = 85% of peak, s.5.4.1).
2025: 10.7 Mt CO2, thermal 564 g/kWh, average 383 g/kWh; 2024 (revised): 553 and 335 g/kWh.
Headline: 2025, thermal_flat, a = 0.081 (V = 0.16): P2 saves 2.4 % (95 % CI 2.0-2.8) aligned, 1.4 % (0.9-1.8) interleaved.
All 2025 constructions: V = 0.10-0.52; P2 0.9-10.7 % aligned, 0.7-6.3 % interleaved.
The preliminary `data/ci_ghana_derived.csv` (mean 376, V = 0.10) has no recorded construction.
