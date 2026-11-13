"""Ghana grid carbon intensity anchored to the Energy Commission's 2026 Energy Statistics (data years 2024 revised and 2025).

Annual anchors (Energy Commission, 2026 Energy Statistics):
  Table 3.7   generation: 2024 hydro 9,893, thermal 15,506, other RE 181 GWh; 2025 hydro 8,752, thermal 18,996, other RE 197 GWh
  Tables 6.24/6.25  fuel input to electricity plants (ktoe): 2025 gas 3,957, crude 146.15, products 311; 2024 gas 3,575, crude 19.81, products 38
  Table 6.26  grid emission factor (all other projects): 0.35 tCO2/MWh for 2024 and 2025 (2025 provisional) -> cross-check only
Emission factors for fuel combustion: IPCC (2006) defaults, tCO2/TJ: natural gas 56.1, crude oil 73.3, gas/diesel oil 74.1.
1 ktoe = 41.868 TJ.

Hourly shape: the statistics give no hourly data, so two bounding dispatch assumptions are built:
  thermal_flat : thermal output constant, hydro and solar follow the load   -> intensity highest at night
  hydro_flat   : hydro output constant, thermal follows the load            -> intensity highest at the evening peak
Load: L(h) = Lbar [1 + a cos(2 pi (h - 20)/24)], peak at 20:00 (2024 system peak was at 20:00, Supply Plan 2025 s.2.2);
  a = 0.081 from the Supply Plan's off-peak = 85% of peak (s.5.4.1), plus a in {0.10, 0.15, 0.25} as sensitivity.
  The Supply Plan names Akosombo as the lead load-following plant (ch.1), which supports thermal_flat.
Solar: 181 GWh/yr on a half-sine between 06:00 and 18:00. No seasonal variation (the source gives none).

    python ghana_ci_ec.py OUTDIR      # writes ghana_anchor.json, ghana_bounds_summary.csv and ci_ghana_ec<year>_<bound>_a<a>.csv
"""
import os, sys, json
import numpy as np, pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
KTOE_TJ = 41.868
EF = {"natural_gas": 56.1, "crude_oil": 73.3, "petroleum_products": 74.1}           # tCO2/TJ, IPCC 2006
YEARS = {2025: dict(fuel={"natural_gas": 3957.0, "crude_oil": 146.15, "petroleum_products": 311.0},
                    gen={"hydro": 8752.0, "thermal": 18996.0, "other_re": 197.0}),
         2024: dict(fuel={"natural_gas": 3575.0, "crude_oil": 19.81, "petroleum_products": 38.0},
                    gen={"hydro": 9893.0, "thermal": 15506.0, "other_re": 181.0})}
# Daily swing calibrated to the 2025 Electricity Supply Plan (PPTC, GRIDCo/Energy Commission), Section 5.4.1:
# off-peak demand is taken as 85% of peak; with L = Lbar[1 + a cos(.)], min/max = (1-a)/(1+a) = 0.85.
A_PLAN = (1 - 0.85) / (1 + 0.85)
allrows, anchors = [], {}
for YEAR, Y in YEARS.items():
  FUEL_KTOE, GEN_GWH = Y["fuel"], Y["gen"]
  total_gwh = sum(GEN_GWH.values())
  co2_t = sum(FUEL_KTOE[f] * KTOE_TJ * EF[f] for f in EF)
  anchor = dict(co2_Mt=co2_t / 1e6, thermal_g_per_kWh=co2_t / GEN_GWH["thermal"],
                average_g_per_kWh=co2_t / total_gwh, official_grid_EF_all_other=350.0, year=YEAR,
                thermal_efficiency=GEN_GWH["thermal"] * 3.6 / (sum(FUEL_KTOE.values()) * KTOE_TJ),
                mean_load_MW=total_gwh * 1e3 / 8760)
  anchors[YEAR] = anchor

  H = np.arange(8760); hod = H % 24
  solar_shape = np.where((hod >= 6) & (hod < 18), np.sin(np.pi * (hod - 6 + 0.5) / 12), 0.0)
  S = GEN_GWH["other_re"] * 1e3 * solar_shape / solar_shape.sum()                      # MW per hour
  Lbar = anchor["mean_load_MW"]; e_th = anchor["thermal_g_per_kWh"]
  rows = []
  for a in (A_PLAN, 0.10, 0.15, 0.25):
      L = Lbar * (1 + a * np.cos(2 * np.pi * (hod - 20) / 24))
      for bound in ("thermal_flat", "hydro_flat"):
          if bound == "thermal_flat":
              T = np.full(8760, GEN_GWH["thermal"] * 1e3 / 8760)
          else:
              Hy = GEN_GWH["hydro"] * 1e3 / 8760
              T = L - Hy - S
          ci = e_th * T / L
          ts = pd.date_range(f"{YEAR}-01-01", periods=8760, freq="h").strftime("%Y-%m-%d %H:%M")
          pd.DataFrame(dict(timestamp=ts, gco2_per_kwh=ci.round(1))).to_csv(os.path.join(OUT, f"ci_ghana_ec{YEAR}_{bound}_a{a:.3f}.csv"), index=False)
          d = ci.reshape(-1, 24)
          rows.append(dict(year=YEAR, bound=bound, a=a, mean=ci.mean(), gen_weighted_mean=(ci * L).sum() / L.sum(), min=ci.min(), max=ci.max(),
                           V=float(((d.max(1) - d.min(1)) / d.mean(1)).mean()), min_thermal_MW=T.min()))
  print(json.dumps(anchor, indent=1)); print(pd.DataFrame(rows).round(3).to_string(index=False))
  allrows += rows
json.dump(anchors, open(os.path.join(OUT, "ghana_anchor.json"), "w"), indent=1)
pd.DataFrame(allrows).to_csv(os.path.join(OUT, "ghana_bounds_summary.csv"), index=False)
