# Build trace.csv from Alibaba cluster-trace-gpu-v2026 (job_execution_summary only, ~1.2 GB zip).
# Run in a SEPARATE Colab notebook (CPU runtime is fine). Do NOT download pod_hourly (350 GB).
import os, subprocess, glob, numpy as np, pandas as pd
from google.colab import drive
drive.mount('/content/drive')
OUTDIR = "/content/drive/MyDrive/msc_sim/data"; os.makedirs(OUTDIR, exist_ok=True)
URL = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/asi_opensource_job_execution_summary.zip"
MIN_GPUS = 4          # keep multi-GPU jobs (the ones that have inter-node traffic)
KEEP_TYPES = ("training", "offline_inference")   # batch-like jobs; add "online_inference"/"dev" to widen
MAX_JOBS = 20000      # random subsample (fixed seed) to keep the simulator fast
SEED = 12345

os.makedirs("/content/a26", exist_ok=True)
if not glob.glob("/content/a26/**/*.parquet", recursive=True):
    subprocess.run(["wget", "-q", "--show-progress", "-O", "/content/a26/s.zip", URL], check=True)
    subprocess.run(["unzip", "-q", "-o", "/content/a26/s.zip", "-d", "/content/a26"], check=True)
fs = glob.glob("/content/a26/**/*.parquet", recursive=True); print("parquet files:", fs)
# MEMORY-SAFE: read only 4 columns, row-group by row-group, filter immediately (the full table crashes free Colab RAM).
import pyarrow.parquet as pq, gc
COLS = ["gpu_request", "duration_hours", "job_type_public"]
parts = []
for f in fs:
    pf = pq.ParquetFile(f); print(f, "rows:", pf.metadata.num_rows, "row groups:", pf.num_row_groups); print(pf.schema_arrow)
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=COLS).to_pandas()
        t = t[(t["gpu_request"] >= MIN_GPUS) & (t["duration_hours"] > 0) & t["job_type_public"].isin(KEEP_TYPES)]
        parts.append(t); del t; gc.collect()
d = pd.concat(parts, ignore_index=True); del parts; gc.collect()
print("after filters:", len(d)); print(d["job_type_public"].value_counts())
assert len(d) > 100, "too few rows after filtering; widen KEEP_TYPES or lower MIN_GPUS"
rng = np.random.default_rng(SEED)
if len(d) > MAX_JOBS: d = d.sample(MAX_JOBS, random_state=SEED)
n = len(d)
# The summary has NO arrival timestamps -> arrivals are a Poisson process (1 job per 60 s on average).
# The simulator rescales arrivals to its horizon anyway, so only the SHAPE (Poisson) is assumed.
arr = np.cumsum(rng.exponential(60.0, n))
out = pd.DataFrame({"arrival_s": arr, "n_gpus": d["gpu_request"].round().astype(int).values,
                    "duration_s": (d["duration_hours"].values * 3600).round(1)})
out = out[out["n_gpus"] >= MIN_GPUS]
out.to_csv(f"{OUTDIR}/trace.csv", index=False)
print("wrote", f"{OUTDIR}/trace.csv", len(out), "jobs")
print(out.describe().T); print("GPU-count mix:\n", out["n_gpus"].value_counts().head(10))
