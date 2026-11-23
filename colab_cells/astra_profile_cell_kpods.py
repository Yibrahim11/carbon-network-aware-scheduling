# ===== Colab cell (v7): ADD measured 3- and 4-pod ns-3 profiles (pclass 'spread_k3', 'spread_k4') to astra_profiles*.csv =====
# Run AFTER the 8/16-NPU and 32/64-NPU cells in the same Colab session (needs the built ns-3 binary, the 32/64-NPU workloads and
# logical topologies those cells created). Safe to re-run: rows with the same jtype/n_gpus/pclass are replaced, others kept.
#
# Why: dcsim measured every 'spread' row on TWO leaf switches. In the synthetic large-job mix (32-GPU pods), 32- and 64-GPU
# jobs are often split over 3 or 4 pods. Without measured rows, dcsim extends the 2-pod row with a rule (ClusterParams.k_rule);
# the old default ('hier', hierarchical ring) is not consistent with the flat ring ASTRA-sim simulates. These rows replace the
# rule with measurements for the splits first-fit actually produces (GPUs are allocated in blocks of 8).
#
# Time: 32-NPU runs are quick; 64-NPU runs are the slowest (8 runs). Set NPUS = [32] first if the session is short.
import os, re, subprocess, numpy as np, pandas as pd
A = "/content/astra-sim"; NS3 = f"{A}/extern/network_backend/ns-3"
OUT, TOPO, CFG = f"{NS3}/scratch/output", f"{NS3}/scratch/topology", f"{NS3}/scratch/config"
WL = f"{A}/examples/workload/microbenchmarks"
SIZES_MB = [16, 64]
NPUS = [32, 64]
SPLITS = {32: {3: [16, 8, 8], 4: [8, 8, 8, 8]}, 64: {3: [24, 24, 16], 4: [16, 16, 16, 16]}}   # GPUs per pod (first-fit, 8-blocks)
JT = {"llm_mid": dict(G=4.0, t_comp=1.0, n=[32]), "llm_large": dict(G=14.0, t_comp=2.0, n=[32, 64])}   # same as dcsim.JOB_TYPES
BIN = f"{NS3}/build/scratch/ns3.42-AstraSimNetwork-default"
assert os.path.exists(BIN), "ns-3 binary missing: run the build and the 8/16-NPU cell first"
for n in NPUS:
    assert os.path.exists(f"{A}/examples/network/ns3/sample_{n}nodes_1D.json"), "run the 32/64-NPU cell first (logical topology)"
    for s_ in SIZES_MB:
        assert os.path.isdir(f"{WL}/all_reduce/{n}npus_{s_}MB"), f"run the 32/64-NPU cell first (workload {n}npus_{s_}MB)"


def write_topo(path, leaf_of, num_leaves, up="400Gbps", g="400Gbps", d="0.0005ms"):
    """Same link rates as the 2-leaf cells: one 400 Gb/s uplink per leaf (= per pod) to a single spine."""
    n = len(leaf_of); assert all(0 <= x < num_leaves for x in leaf_of)
    leaves = [n + i for i in range(num_leaves)]; spine = [n + num_leaves]
    links = [f"{i} {leaves[leaf_of[i]]} {g} {d} 0" for i in range(n)] + [f"{lf} {spine[0]} {up} {d} 0" for lf in leaves]
    sw = leaves + spine
    open(path, "w").write(f"{n+len(sw)} {len(sw)} {len(links)}\n" + " ".join(map(str, sw)) + "\n" + "\n".join(links) + "\n")


def leaf_map(sizes, kind):
    """Rank -> pod. 'interleaved': ranks dealt round-robin over pods that still have room (worst case, as in the
    2-pod cells). 'aligned': contiguous rank blocks, one ring crossing per pod boundary (best case)."""
    if kind == "aligned":
        return [p for p, c in enumerate(sizes) for _ in range(c)]
    left, out, p = list(sizes), [], 0
    while len(out) < sum(sizes):
        if left[p] > 0:
            out.append(p); left[p] -= 1
        p = (p + 1) % len(sizes)
    return out


def ring_egress_formula(lf, S_bytes):
    """Flat ring all-reduce: rank i sends 2(n-1)/n * S to rank i+1. Bytes leaving each pod, per pod."""
    n, k = len(lf), max(lf) + 1
    e = np.zeros(k)
    for i in range(n):
        if lf[i] != lf[(i + 1) % n]:
            e[lf[i]] += 2 * (n - 1) / n * S_bytes
    return e


def run(n, s, k, kind):
    lf = leaf_map(SPLITS[n][k], kind); name = f"{kind}_k{k}_{n}"
    write_topo(f"{TOPO}/{name}.txt", lf, k)
    cfg = open(f"{CFG}/config.txt").read()
    cfg = re.sub(r"(?m)^TOPOLOGY_FILE .*$", f"TOPOLOGY_FILE ../../scratch/topology/{name}.txt", cfg)
    open(f"{CFG}/config_{name}.txt", "w").write(cfg)
    open(f"{OUT}/flow.txt", "w").write("0\n"); open(f"{OUT}/trace.txt", "w").write("0\n")
    for fn in ("fct.txt", "mix.tr", "pfc.txt", "qlen.txt"): open(f"{OUT}/{fn}", "w").close()
    cmd = [BIN, f"--workload-configuration={WL}/all_reduce/{n}npus_{s}MB/all_reduce",
           f"--system-configuration={A}/examples/system/native_collectives/Ring_4chunks.json",
           f"--network-configuration={CFG}/config_{name}.txt",
           f"--remote-memory-configuration={A}/examples/remote_memory/analytical/no_memory_expansion.json",
           f"--logical-topology-configuration={A}/examples/network/ns3/sample_{n}nodes_1D.json", "--comm-group-configuration=empty"]
    r = subprocess.run(cmd, cwd=f"{NS3}/build/scratch", capture_output=True, text=True, timeout=6000)
    if r.returncode != 0: raise RuntimeError(f"ns-3 failed: {name} {s}MB\n{(r.stdout + r.stderr)[-2000:]}")
    cyc = [int(m) for m in re.findall(r"finished, (\d+) cycles", r.stdout + r.stderr)]
    assert len(cyc) == n, f"{name} {s}MB: got {len(cyc)} finish lines"
    egress = np.zeros(k); last_end = 0.0
    for ln in open(f"{OUT}/fct.txt"):
        f = ln.split()
        if len(f) < 7: continue
        a, b = (int(f[0], 16) >> 8) & 0xFFFF, (int(f[1], 16) >> 8) & 0xFFFF
        if a < n and b < n and lf[a] != lf[b]: egress[lf[a]] += int(f[4])
        last_end = max(last_end, float(f[5]) + float(f[6]))
    assert egress.sum() > 0, "spread placement must cross pods: endpoint decoding is wrong"
    CLOCK.append(max(cyc) / last_end)
    return max(cyc) * 1e-9, egress, lf                  # ASSUMES 1 cycle = 1 ns (clock check printed below)


T, EG, RAW, CLOCK = {}, {}, [], []
for n in NPUS:
    for k in (3, 4):
        for kind in ("interleaved", "aligned"):
            for s in SIZES_MB:
                t, eg, lf = run(n, s, k, kind); T[(n, k, kind, s)] = t; EG[(n, k, kind, s)] = eg
                chk = eg / np.maximum(ring_egress_formula(lf, s * 1024 * 1024), 1)
                RAW.append(dict(npus=n, placement=f"{'spread' if kind == 'interleaved' else 'aligned'}_k{k}", msg_MB=s,
                                max_cycles=int(round(t * 1e9)), time_s=t, cross_leaf_bytes_measured=int(eg.sum()),
                                pod_sizes="+".join(map(str, SPLITS[n][k])), max_pod_egress_bytes=int(eg.max())))
                print(f"{n} NPUs, k={k} ({'+'.join(map(str, SPLITS[n][k]))}), {kind:11s} {s} MB -> {t*1e3:8.3f} ms | "
                      f"per-pod egress measured/flat-ring formula = {np.round(chk, 3).tolist()}")


def fit_time(n, k, kind):
    x = np.array(SIZES_MB) / 1024.0; y = np.array([T[(n, k, kind, s)] for s in SIZES_MB])
    b_, a_ = np.polyfit(x, y, 1); return lambda G: a_ + b_ * G


rows = {"interleaved": [], "aligned": []}
for jt, p in JT.items():
    for n in p["n"]:
        if n not in NPUS: continue
        for k in (3, 4):
            for kind in ("interleaved", "aligned"):
                ts = fit_time(n, k, kind)(p["G"])
                gb = EG[(n, k, kind, SIZES_MB[-1])].max() / (SIZES_MB[-1] * 1024 * 1024) * p["G"]   # GB leaving the busiest pod
                rows[kind].append(dict(jtype=jt, n_gpus=n, pclass=f"spread_k{k}", iter_time_s=p["t_comp"] + ts,
                                       frac_compute=p["t_comp"] / (p["t_comp"] + ts), interpod_GB=gb, comm_time_s=ts,
                                       source="astra-sim/ns-3 comm fit + ASSUMED compute", placement_variant=kind,
                                       pod_sizes="+".join(map(str, SPLITS[n][k]))))
D = "/content/drive/MyDrive/msc_sim/data" if os.path.isdir("/content/drive/MyDrive/msc_sim/data") else "."


def merge(fn, new):
    new = pd.DataFrame(new).drop_duplicates(); p_ = f"{D}/{fn}"
    old = pd.read_csv(p_) if os.path.exists(p_) else pd.DataFrame()
    if len(old):
        key = set(zip(new.jtype, new.n_gpus, new.pclass)); old = old[[q not in key for q in zip(old.jtype, old.n_gpus, old.pclass)]]
    m = pd.concat([old, new], ignore_index=True); m.to_csv(p_, index=False); return m


m1 = merge("astra_profiles.csv", rows["interleaved"]); m2 = merge("astra_profiles_aligned.csv", rows["aligned"])
rp = f"{D}/raw_ns3_runs.csv"; raw_new = pd.DataFrame(RAW)
(pd.concat([pd.read_csv(rp), raw_new], ignore_index=True) if os.path.exists(rp) else raw_new) \
    .drop_duplicates(["npus", "placement", "msg_MB"], keep="last").to_csv(rp, index=False)
print(f"\nclock check: cycles / last flow end = {min(CLOCK):.3f} .. {max(CLOCK):.3f} (about 1.0 supports 1 cycle = 1 ns)")
print("saved to", D, "| interleaved table rows:", len(m1), "| aligned table rows:", len(m2))

# How good was each extension rule? Compare measured k=3/4 rows with the 2-pod row extended by each rule.
two = {v: pd.read_csv(f"{D}/{f}").set_index(["jtype", "n_gpus", "pclass"]) for v, f in
       (("interleaved", "astra_profiles.csv"), ("aligned", "astra_profiles_aligned.csv"))}
rule = {"hier": lambda k: 2 * (k - 1) / k, "ring_interleaved": lambda k: 2 / k, "ring_aligned": lambda k: 1.0}
print("\nmeasured / rule-predicted, per-pod volume and communication time (1.00 = rule exact):")
for kind in ("interleaved", "aligned"):
    for r in rows[kind]:
        b = two[kind].loc[(r["jtype"], r["n_gpus"], "spread")]; k = int(r["pclass"][-1])
        own = "ring_interleaved" if kind == "interleaved" else "ring_aligned"
        msg = [f"{nm}: volume {r['interpod_GB'] / (b.interpod_GB * rule[nm](k) / rule[nm](2) if nm == 'hier' else b.interpod_GB * rule[nm](k)):.2f}, "
               f"time {r['comm_time_s'] / (b.comm_time_s * (rule[nm](k) / rule[nm](2) if nm == 'hier' else rule[nm](k))):.2f}"
               for nm in ("hier", own)]
        print(f"  {kind:11s} {r['jtype']:9s} {r['n_gpus']} GPUs k={k}: " + " | ".join(msg))
