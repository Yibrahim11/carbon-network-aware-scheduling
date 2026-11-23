# ===== Colab cell: merge the NEW 32/64-NPU profile rows (saved locally in /content) into the Drive files =====
import os, pandas as pd
from google.colab import drive
drive.mount('/content/drive')
D = "/content/drive/MyDrive/msc_sim/data"; assert os.path.isdir(D), "msc_sim/data not found on Drive"
for fn in ("astra_profiles.csv", "astra_profiles_aligned.csv", "raw_ns3_runs.csv"):
    new_p = f"/content/{fn}"
    if not os.path.exists(new_p): print("not found locally (skipped):", new_p); continue
    new = pd.read_csv(new_p); old_p = f"{D}/{fn}"
    old = pd.read_csv(old_p) if os.path.exists(old_p) else pd.DataFrame()
    if fn == "raw_ns3_runs.csv":
        m = pd.concat([old, new], ignore_index=True).drop_duplicates(["npus", "placement", "msg_MB"], keep="last")
    else:
        if len(old):
            key = set(zip(new.jtype, new.n_gpus, new.pclass)); old = old[[k not in key for k in zip(old.jtype, old.n_gpus, old.pclass)]]
        m = pd.concat([old, new], ignore_index=True)
    m.to_csv(old_p, index=False); print(fn, "->", len(m), "rows")
    if fn != "raw_ns3_runs.csv": print(m[["jtype", "n_gpus", "pclass", "iter_time_s", "comm_time_s", "interpod_GB"]].round(3).to_string(index=False))
