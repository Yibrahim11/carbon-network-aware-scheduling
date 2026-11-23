# ===== Colab cell: build astra_profiles.csv from ns-3 runs (compact vs spread, 8 and 16 NPUs) =====
import os, re, subprocess, importlib.util, numpy as np, pandas as pd
A = "/content/astra-sim"; NS3 = f"{A}/extern/network_backend/ns-3"
OUT, TOPO, CFG = f"{NS3}/scratch/output", f"{NS3}/scratch/topology", f"{NS3}/scratch/config"
WL = f"{A}/examples/workload/microbenchmarks"
SIZES_MB = [16, 64]            # simulated message sizes; time is fitted as a + b*size and extrapolated
NPUS = [8, 16]
# job types: gradient size G (GB) and compute time per iteration (ASSUMED, same as the synthetic model; overlap = 0)
JT = {"vision": dict(G=0.4, t_comp=0.5, n=[8, 16]), "llm_mid": dict(G=4.0, t_comp=1.0, n=[8, 16])}

def write_topo(path, leaf_of, num_leaves=2, up="400Gbps", g="400Gbps", d="0.0005ms"):
    """Same physical network for every placement: 2 leaf switches + 1 spine; only the host->leaf mapping differs."""
    n = len(leaf_of); assert all(0 <= x < num_leaves for x in leaf_of)
    leaves = [n + i for i in range(num_leaves)]; spine = [n + num_leaves]
    links = [f"{i} {leaves[leaf_of[i]]} {g} {d} 0" for i in range(n)] + [f"{lf} {spine[0]} {up} {d} 0" for lf in leaves]
    sw = leaves + spine
    open(path, "w").write(f"{n+len(sw)} {len(sw)} {len(links)}\n" + " ".join(map(str, sw)) + "\n" + "\n".join(links) + "\n")

def leaf_map(n, kind):
    return {"compact": [0]*n, "spread": [i % 2 for i in range(n)], "aligned": [0]*(n//2) + [1]*(n//2)}[kind]

# workloads (8/16 NPUs x 16/64 MB) were generated in a separate process earlier
for n in NPUS:
    for s_ in SIZES_MB: assert os.path.isdir(f"{WL}/all_reduce/{n}npus_{s_}MB"), f"missing workload {n}npus_{s_}MB"

def run(n, s, kind):
    name = f"{kind}_{n}"; write_topo(f"{TOPO}/{name}.txt", leaf_map(n, kind))
    cfg = open(f"{CFG}/config.txt").read()
    cfg = re.sub(r"(?m)^TOPOLOGY_FILE .*$", f"TOPOLOGY_FILE ../../scratch/topology/{name}.txt", cfg)
    open(f"{CFG}/config_{name}.txt", "w").write(cfg)
    open(f"{OUT}/flow.txt", "w").write("0\n"); open(f"{OUT}/trace.txt", "w").write("0\n")
    for fn in ("fct.txt", "mix.tr", "pfc.txt", "qlen.txt"): open(f"{OUT}/{fn}", "w").close()    # no stale output between runs
    cmd = [f"./ns3.42-AstraSimNetwork-default", f"--workload-configuration={WL}/all_reduce/{n}npus_{s}MB/all_reduce",
           f"--system-configuration={A}/examples/system/native_collectives/Ring_4chunks.json",
           f"--network-configuration={CFG}/config_{name}.txt",
           f"--remote-memory-configuration={A}/examples/remote_memory/analytical/no_memory_expansion.json",
           f"--logical-topology-configuration={A}/examples/network/ns3/sample_{n}nodes_1D.json", "--comm-group-configuration=empty"]
    r = subprocess.run(cmd, cwd=f"{NS3}/build/scratch", capture_output=True, text=True, timeout=3000)
    if r.returncode != 0: raise RuntimeError(f"ns-3 failed: n={n} {s}MB {kind}\n{(r.stdout + r.stderr)[-2000:]}")
    cyc = [int(m) for m in re.findall(r"finished, (\d+) cycles", r.stdout + r.stderr)]
    assert len(cyc) == n, f"{name} {s}MB: got {len(cyc)} finish lines\n{(r.stdout+r.stderr)[-800:]}"
    lf = leaf_map(n, kind); cross = 0; last_end = 0
    for ln in open(f"{OUT}/fct.txt"):
        f = ln.split()
        if len(f) < 5: continue
        a, b = (int(f[0], 16) >> 8) & 0xFFFF, (int(f[1], 16) >> 8) & 0xFFFF
        if a < n and b < n and lf[a] != lf[b]: cross += int(f[4])
        last_end = max(last_end, float(f[5]) + float(f[6]))          # start + fct, ns
    if kind == "compact": assert cross == 0, "compact placement must have no inter-leaf traffic: endpoint decoding is wrong"
    if kind == "spread": assert cross > 0, "interleaved placement must cross leaves: endpoint decoding is wrong"
    if last_end <= 0: raise RuntimeError(f"no valid flow end time parsed for {n} NPUs {s} MB {kind}: check fct.txt format")
    CLOCK.append(max(cyc) / last_end)
    return max(cyc), cross

T = {}; RAW = []; CLOCK = []
for n in NPUS:
    for kind in ("compact", "spread", "aligned"):
        for s in SIZES_MB:
            cy, cb = run(n, s, kind); T[(n, kind, s)] = cy * 1e-9      # ASSUMES 1 cycle = 1 ns (consistent with fct.txt)
            RAW.append(dict(npus=n, placement=kind, msg_MB=s, max_cycles=cy, time_s=T[(n, kind, s)], cross_leaf_bytes_measured=cb))
            print(n, kind, s, "MB ->", round(T[(n, kind, s)]*1e3, 3), "ms | cross-leaf GB (fct.txt):", round(cb/1e9, 4))

def fit(n, kind):                      # t(G_GB) = a + b*G, least squares over the simulated sizes
    x = np.array(SIZES_MB) / 1024.0; y = np.array([T[(n, kind, s)] for s in SIZES_MB])
    b_, a_ = np.polyfit(x, y, 1)
    if len(x) > 2: print(f"  linearity check n={n} {kind}: max rel. residual {np.max(np.abs(a_ + b_*x - y)/y):.3%}")
    return lambda G: a_ + b_ * G

rows = {"spread": [], "aligned": []}
for jt, p in JT.items():
    for n in p["n"]:
        tc = fit(n, "compact")(p["G"]); rows_c = dict(jtype=jt, n_gpus=n, pclass="compact", iter_time_s=p["t_comp"] + tc,
              frac_compute=p["t_comp"] / (p["t_comp"] + tc), interpod_GB=0.0, comm_time_s=tc, source="astra-sim/ns-3 comm fit + ASSUMED compute")
        for var, hops in (("spread", n // 2), ("aligned", 1)):
            ts = fit(n, var)(p["G"]); gb = hops * 2 * (n - 1) / n * p["G"]
            rows[var] += [rows_c, dict(jtype=jt, n_gpus=n, pclass="spread", iter_time_s=p["t_comp"] + ts,
                          frac_compute=p["t_comp"] / (p["t_comp"] + ts), interpod_GB=gb, comm_time_s=ts, source="astra-sim/ns-3 comm fit + ASSUMED compute", placement_variant=("interleaved" if var == "spread" else "aligned"))]
os.makedirs("/content/drive/MyDrive/msc_sim/data", exist_ok=True) if os.path.isdir("/content/drive") else None
D = "/content/drive/MyDrive/msc_sim/data" if os.path.isdir("/content/drive/MyDrive") else "."
pd.DataFrame(rows["spread"]).drop_duplicates().to_csv(f"{D}/astra_profiles.csv", index=False)
pd.DataFrame(rows["aligned"]).drop_duplicates().to_csv(f"{D}/astra_profiles_aligned.csv", index=False)
d = pd.DataFrame(rows["spread"]).drop_duplicates()
pd.DataFrame(RAW).to_csv(f"{D}/raw_ns3_runs.csv", index=False)          # Layer 1: raw simulator measurements
print(d.round(4).to_string(index=False)); print("saved to", D)
print(f"clock check: reported cycles / last flow end (ns) = {min(CLOCK):.3f} .. {max(CLOCK):.3f}  (about 1.0 supports 1 cycle = 1 ns)")
# formula vs measured inter-leaf traffic (spread = interleaved; per-direction GB = total/2)
for r in RAW:
    if r["placement"] in ("spread", "aligned"):
        n, S = r["npus"], r["msg_MB"]; hops = n // 2 if r["placement"] == "spread" else 1
        formula = 2 * hops * 2 * (n - 1) / n * S * 1024 * 1024      # both directions, bytes
        print(f"  traffic check {n} NPUs {r['placement']:8s} {S} MB: measured/formula = {r['cross_leaf_bytes_measured']/formula:.3f}")
for n in NPUS: print(n, "NPUs, largest size: spread/compact =", round(T[(n,'spread',SIZES_MB[-1])]/T[(n,'compact',SIZES_MB[-1])], 2),
      "| aligned/compact =", round(T[(n,'aligned',SIZES_MB[-1])]/T[(n,'compact',SIZES_MB[-1])], 2))
