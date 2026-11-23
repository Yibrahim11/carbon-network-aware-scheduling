# ===== Colab cell: build real hourly carbon-intensity CSVs for the simulation =====
# Output: msc_sim/data/ci_<name>.csv with columns timestamp,gco2_per_kwh (hourly). Each file becomes its own scenario.
import os, time, requests, numpy as np, pandas as pd
D = "/content/drive/MyDrive/msc_sim/data"; os.makedirs(D, exist_ok=True)

# --- A. Great Britain (free API, no key, half-hourly actual intensity, gCO2/kWh): a real HIGH-variability reference grid ---
START, DAYS = "2025-01-01", 365
def gb_year(start, days):
    out, t0 = [], pd.Timestamp(start, tz="UTC")
    for k in range(0, days, 13):                                   # API limits a request to 14 days
        a, b = t0 + pd.Timedelta(days=k), t0 + pd.Timedelta(days=min(k + 13, days))
        url = f"https://api.carbonintensity.org.uk/intensity/{a:%Y-%m-%dT%H:%MZ}/{b:%Y-%m-%dT%H:%MZ}"
        for attempt in range(3):
            r = requests.get(url, timeout=30)
            if r.ok: break
            time.sleep(2)
        r.raise_for_status()
        for x in r.json()["data"]:
            v = x["intensity"].get("actual") or x["intensity"].get("forecast")
            out.append((pd.Timestamp(x["from"]), v))
    s = pd.Series(dict(out)).sort_index()
    return s[~s.index.duplicated()].astype(float)
try:
    s = gb_year(START, DAYS).resample("1h").mean().interpolate(limit=6).dropna()
    pd.DataFrame(dict(timestamp=s.index.strftime("%Y-%m-%d %H:%M"), gco2_per_kwh=s.values)).to_csv(f"{D}/ci_gb_2025.csv", index=False)
    print("GB:", len(s), "hours | mean", round(s.mean()), "| min", round(s.min()), "| max", round(s.max()))
except Exception as e:
    print("GB download failed:", repr(e)[:300])

# --- B. ANY other zone you download yourself (e.g. Electricity Maps data portal, https://portal.electricitymaps.com , free sign-up) ---
# Put the downloaded CSV in Drive and set the path + name here; the converter finds the datetime and carbon-intensity columns.
EXTRA = {}          # example: {"ghana": "/content/drive/MyDrive/downloads/GH_2024_hourly.csv"}
for name, path in EXTRA.items():
    g = pd.read_csv(path); print(name, "columns:", list(g.columns))
    tcol = next(c for c in g.columns if "date" in c.lower() or "time" in c.lower())
    vcols = [c for c in g.columns if "intensity" in c.lower()]
    vcol = next((c for c in vcols if "lca" in c.lower()), vcols[0]); print("using", tcol, "and", vcol)
    out = pd.DataFrame(dict(timestamp=g[tcol], gco2_per_kwh=pd.to_numeric(g[vcol], errors="coerce"))).dropna()
    out.to_csv(f"{D}/ci_{name}.csv", index=False); print(name, len(out), "rows saved")
print(sorted(f for f in os.listdir(D) if f.startswith("ci_")))
