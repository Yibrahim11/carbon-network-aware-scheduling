# ===== Colab cell (v4): ADD 32- and 64-NPU ns-3 profiles to the existing astra_profiles*.csv on Drive =====
# Run AFTER the 8/16-NPU cell in the same Colab session (needs the built ns-3 binary + /content/astra-sim). CPU runtime is fine.
# Safe to re-run. It MERGES into the existing CSVs (rows for the same jtype/n_gpus/pclass are replaced, others kept).
import os, re, subprocess, sys, json, textwrap, importlib.util, numpy as np, pandas as pd
A = "/content/astra-sim"; NS3 = f"{A}/extern/network_backend/ns-3"
OUT, TOPO, CFG = f"{NS3}/scratch/output", f"{NS3}/scratch/topology", f"{NS3}/scratch/config"
WL = f"{A}/examples/workload/microbenchmarks"
SIZES_MB = [16, 64]
NPUS = [32, 64]                 # run [32] first if short on time; 64 NPUs is the slowest
JT = {"llm_mid": dict(G=4.0, t_comp=1.0, n=[32]), "llm_large": dict(G=14.0, t_comp=2.0, n=[32, 64])}   # same G/t_comp as dcsim.JOB_TYPES
assert os.path.exists(f"{NS3}/build/scratch/ns3.42-AstraSimNetwork-default"), "ns-3 binary missing: rerun the build/8-16 NPU cell first"

# ---- 1) logical topology files (1-D ring over N NPUs) -------------------------------------------
ex = f"{A}/examples/network/ns3"
print("existing logical-topology samples:", sorted(os.listdir(ex))[:12])
for n in NPUS:
    p = f"{ex}/sample_{n}nodes_1D.json"
    if not os.path.exists(p): json.dump({"logical-dims": [str(n)]}, open(p, "w"))
_s8 = open(f"{ex}/sample_8nodes_1D.json").read().strip(); print("8-node sample content (format check):", _s8)
assert "logical-dims" in _s8, "unexpected logical-topology format: send me the line printed above"

# ---- 2) workloads (Chakra ET files) generated in a SEPARATE process (protobuf version issue) -----
GEN = textwrap.dedent("""
    import sys, os
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node, ALL_REDUCE, COMM_COLL_NODE, AttributeProto as A
    from chakra.src.third_party.utils.protolib import encodeMessage as enc
    root, n, mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    d = f"{root}/all_reduce/{n}npus_{mb}MB"; os.makedirs(d, exist_ok=True)
    for i in range(n):
        with open(f"{d}/all_reduce.{i}.et", "wb") as f:
            enc(f, GlobalMetadata(version="0.0.4"))
            nd = Node(); nd.id = 1; nd.name = "All-Reduce"; nd.type = COMM_COLL_NODE
            nd.attr.append(A(name="is_cpu_op", bool_val=False)); nd.attr.append(A(name="comm_type", int64_val=ALL_REDUCE))
            nd.attr.append(A(name="comm_size", int64_val=mb * 1024 * 1024)); enc(f, nd)
""")
open("/content/gen_ar.py", "w").write(GEN)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "protobuf"], check=False)
for n in NPUS:
    for s_ in SIZES_MB:
        if not os.path.isdir(f"{WL}/all_reduce/{n}npus_{s_}MB"):
            r = subprocess.run([sys.executable, "/content/gen_ar.py", WL, str(n), str(s_)], capture_output=True, text=True)
            assert r.returncode == 0, f"workload generation failed (send me this):\n{r.stderr[-1500:]}"
        assert len(os.listdir(f"{WL}/all_reduce/{n}npus_{s_}MB")) >= n, f"workload {n}npus_{s_}MB incomplete"

def write_topo(path, leaf_of, num_leaves=2, up="400Gbps", g="400Gbps", d="0.0005ms"):
    """Same physical network for every placement: 2 leaf switches + 1 spine; only the host->leaf mapping differs."""
    n = len(leaf_of); assert all(0 <= x < num_leaves for x in leaf_of)
    leaves = [n + i for i in range(num_leaves)]; spine = [n + num_leaves]
    links = [f"{i} {leaves[leaf_of[i]]} {g} {d} 0" for i in range(n)] + [f"{lf} {spine[0]} {up} {d} 0" for lf in leaves]
    sw = leaves + spine
    open(path, "w").write(f"{n+len(sw)} {len(sw)} {len(links)}\n" + " ".join(map(str, sw)) + "\n" + "\n".join(links) + "\n")

def leaf_map(n, kind):
    return {"compact": [0]*n, "spread": [i % 2 for i in range(n)], "aligned": [0]*(n//2) + [1]*(n//2)}[kind]

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
D = "/content/drive/MyDrive/msc_sim/data" if os.path.isdir("/content/drive/MyDrive/msc_sim/data") else "."
def merge(fn, new):
    new = pd.DataFrame(new).drop_duplicates(); p_ = f"{D}/{fn}"
    old = pd.read_csv(p_) if os.path.exists(p_) else pd.DataFrame()
    if len(old):
        key = set(zip(new.jtype, new.n_gpus, new.pclass)); old = old[[k not in key for k in zip(old.jtype, old.n_gpus, old.pclass)]]
    m = pd.concat([old, new], ignore_index=True); m.to_csv(p_, index=False); return m
m1 = merge("astra_profiles.csv", rows["spread"]); merge("astra_profiles_aligned.csv", rows["aligned"])
rp = f"{D}/raw_ns3_runs.csv"; raw_new = pd.DataFrame(RAW)
pd.concat([pd.read_csv(rp), raw_new], ignore_index=True).drop_duplicates(["npus", "placement", "msg_MB"], keep="last").to_csv(rp, index=False) if os.path.exists(rp) else raw_new.to_csv(rp, index=False)
print(m1.round(4).to_string(index=False)); print("saved to", D, "| profile rows now:", len(m1), "(synthetic fill needed only for missing keys)")
print(f"clock check: reported cycles / last flow end (ns) = {min(CLOCK):.3f} .. {max(CLOCK):.3f}  (about 1.0 supports 1 cycle = 1 ns)")
for r in RAW:
    if r["placement"] in ("spread", "aligned"):
        n, S = r["npus"], r["msg_MB"]; hops = n // 2 if r["placement"] == "spread" else 1
        formula = 2 * hops * 2 * (n - 1) / n * S * 1024 * 1024
        print(f"  traffic check {n} NPUs {r['placement']:8s} {S} MB: measured/formula = {r['cross_leaf_bytes_measured']/formula:.3f}")
for n in NPUS: print(n, "NPUs, largest size: spread/compact =", round(T[(n,'spread',SIZES_MB[-1])]/T[(n,'compact',SIZES_MB[-1])], 2),
      "| aligned/compact =", round(T[(n,'aligned',SIZES_MB[-1])]/T[(n,'compact',SIZES_MB[-1])], 2))
