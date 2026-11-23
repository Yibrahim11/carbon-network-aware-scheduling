# ===== Colab cell v2: GPU power vs compute duty cycle (T4), with diagnostics. Runtime must be GPU. =====
import os, time, threading, subprocess, numpy as np, pandas as pd, torch
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Runtime > Change runtime type > T4 GPU, then Runtime > Restart session"
print(subprocess.run("nvidia-smi --query-gpu=name,power.draw,power.limit,power.management,clocks.sm,utilization.gpu --format=csv",
                     shell=True, capture_output=True, text=True).stdout)
try:
    import pynvml
except ImportError:
    subprocess.run("pip install -q nvidia-ml-py", shell=True); import pynvml
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def power_w():
    return pynvml.nvmlDeviceGetPowerUsage(H) / 1000.0            # milliwatts -> watts (board power)
try: print("NVML power now:", power_w(), "W")
except Exception as e: raise SystemExit(f"NVML cannot read power on this GPU/driver: {e!r} -> tell me this line")

D = "/content/drive/MyDrive/msc_sim/data"; os.makedirs(D, exist_ok=True)
dev = "cuda"
A16 = torch.randn(4096, 4096, device=dev, dtype=torch.float16); B16 = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
A32 = torch.randn(3072, 3072, device=dev); B32 = torch.randn(3072, 3072, device=dev)
V = torch.randn(64 * 1024 * 1024, device=dev)
WORK = {"fp16_matmul": lambda: A16 @ B16, "fp32_matmul": lambda: A32 @ B32, "membound": lambda: V.mul_(1.0001)}
PERIOD, T_RUN, WARM = 0.4, 20.0, 8.0
# sanity: full load must draw clearly more than idle
t1 = time.time()
while time.time() - t1 < 6: WORK["fp16_matmul"](); torch.cuda.synchronize()
print("power under full load:", round(power_w(), 1), "W")
rows = []
for name, fn in WORK.items():
    for duty in (0.0, 0.25, 0.5, 0.75, 1.0):
        samples, t0, stop = [], time.time(), threading.Event()
        def sampler():
            while not stop.is_set():
                if time.time() - t0 > WARM: samples.append(power_w())
                time.sleep(0.1)
        th = threading.Thread(target=sampler); th.start()
        while time.time() - t0 < T_RUN:
            t2 = time.time()
            while time.time() - t2 < duty * PERIOD: fn(); torch.cuda.synchronize()
            time.sleep(max(0.0, (1 - duty) * PERIOD))
        stop.set(); th.join()
        p = float(np.mean(samples)) if samples else float("nan")
        rows.append(dict(workload=name, frac_compute=duty, power_w=p)); print(f"{name:12s} duty {duty:.2f} -> {p:6.1f} W  (n={len(samples)}, min {min(samples):.1f}, max {max(samples):.1f})")
df = pd.DataFrame(rows)
assert df.power_w.notna().all() and df.power_w.max() > df.power_w.min() + 5, "power did not vary with load: do not use this file; paste the printed lines to me"
df.to_csv(f"{D}/gpu_power.csv", index=False); print("saved", f"{D}/gpu_power.csv"); print(df.round(1).to_string(index=False))
