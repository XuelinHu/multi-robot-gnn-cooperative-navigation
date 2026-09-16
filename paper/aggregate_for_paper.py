"""Aggregate frozen result CSVs into the tables used by the IEEE manuscript.

Reads only the frozen final evaluation files under results/ and prints
markdown tables. Standard library only (no pandas on this machine).
"""
import csv
import os
from collections import defaultdict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def load(name):
    with open(os.path.join(RES, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def group(rows, keyfn, valfn):
    buckets = defaultdict(list)
    for r in rows:
        buckets[keyfn(r)].append(valfn(r))
    return buckets


def emit(title, rows, keyfn, metrics, numfmt="{:.3f}"):
    print("\n### " + title)
    keys = sorted(set(keyfn(r) for r in rows))
    header = ["group"] + [m for m, _ in metrics]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for k in keys:
        subset = [r for r in rows if keyfn(r) == k]
        cells = [str(k)]
        for _, fn in metrics:
            vals = [fn(r) for r in subset]
            cells.append(numfmt.format(mean(vals)))
        print("| " + " | ".join(cells) + " |")


def fnum(r, name):
    v = r.get(name, "")
    return float(v) if v not in ("", None) else float("nan")


# ---------------------------------------------------------------- main table
main = load("baselines_final.csv")
emit(
    "Main comparison (4 layouts x 10 seeds)",
    main,
    lambda r: f"{r['method']}/{int(float(r['robots']))}",
    [
        ("success", lambda r: fnum(r, "success")),
        ("coll_events", lambda r: fnum(r, "collision_events")),
        ("min_sep", lambda r: fnum(r, "min_separation")),
        ("path_len", lambda r: fnum(r, "path_length")),
        ("ctrl_ms", lambda r: fnum(r, "control_time_ms")),
    ],
)

# --------------------------------------------------- per-method x robot mean
print("\n### Main success mean +- std by method/robots")
for m in sorted(set(r["method"] for r in main)):
    for n in (5, 10, 20):
        vals = [fnum(r, "success") for r in main if r["method"] == m and int(float(r["robots"])) == n]
        print(f"{m:6s} n={n:2d}  success={mean(vals):.3f} +- {std(vals):.3f}  (n_ep={len(vals)})")

# --------------------------------------------------------- extended learned
ext = load("extended_learned_physical.csv")
emit(
    "Extended learned models (overall, 480 ep)",
    ext,
    lambda r: r["method"],
    [
        ("success", lambda r: fnum(r, "success")),
        ("strict_success", lambda r: fnum(r, "strict_success")),
        ("phys_coll", lambda r: fnum(r, "physical_collision_events")),
        ("sep_viol", lambda r: fnum(r, "collision_events")),
        ("min_sep", lambda r: fnum(r, "min_separation")),
        ("ctrl_ms", lambda r: fnum(r, "control_time_ms")),
    ],
)

# --------------------------------------------------------------- graph ablation
g = load("ablation_graph_final.csv")
emit(
    "Graph ablation",
    g,
    lambda r: f"{r['variant']}/{int(float(r['robots']))}",
    [
        ("success", lambda r: fnum(r, "success")),
        ("coll_events", lambda r: fnum(r, "collision_events")),
        ("min_sep", lambda r: fnum(r, "min_separation")),
    ],
)

# -------------------------------------------------------------- safety ablation
s = load("ablation_safety_final.csv")
emit(
    "Safety ablation (raw GNN vs safety filter)",
    s,
    lambda r: f"{'raw' if r['safety'] in ('0', '0.0', 'False', 'false') else 'safe'}/{int(float(r['robots']))}",
    [
        ("success", lambda r: fnum(r, "success")),
        ("coll_events", lambda r: fnum(r, "collision_events")),
        ("min_sep", lambda r: fnum(r, "min_separation")),
    ],
)

# -------------------------------------------------------------- radius ablation
rad = load("ablation_radius_final.csv")
emit(
    "Communication radius ablation",
    rad,
    lambda r: f"{r['radius']}/{int(float(r['robots']))}",
    [
        ("success", lambda r: fnum(r, "success")),
        ("coll_events", lambda r: fnum(r, "collision_events")),
        ("avg_degree", lambda r: fnum(r, "average_degree")),
        ("ctrl_ms", lambda r: fnum(r, "control_time_ms")),
    ],
)

# ------------------------------------------------------------------ multiscale
ms = load("multiscale_models_test_safety.csv")
emit(
    "Multiscale (safety-filtered), 1800 ep",
    ms,
    lambda r: f"{r['method']}/{int(float(r['robots']))}",
    [
        ("success", lambda r: fnum(r, "success")),
        ("strict", lambda r: fnum(r, "strict_success")),
        ("phys_coll", lambda r: fnum(r, "physical_collision_events")),
        ("min_sep", lambda r: fnum(r, "min_separation")),
        ("ctrl_ms", lambda r: fnum(r, "control_time_ms")),
    ],
)

msraw = load("multiscale_models_test.csv")
emit(
    "Multiscale RAW (no safety filter), 1800 ep",
    msraw,
    lambda r: f"{r['method']}/{int(float(r['robots']))}",
    [
        ("success", lambda r: fnum(r, "success")),
        ("strict", lambda r: fnum(r, "strict_success")),
        ("phys_coll", lambda r: fnum(r, "physical_collision_events")),
    ],
)

# --------------------------------------------------------------- training seeds
ts = load("training_seed_comparison.csv")
print("\n### Training seed stability (3 checkpoints)")
for n in (5, 10, 20):
    per_ck = defaultdict(list)
    for r in ts:
        if int(float(r["robots"])) == n:
            per_ck[r["checkpoint"]].append(fnum(r, "success"))
    ck_means = [mean(v) for v in per_ck.values()]
    print(
        f"n={n:2d}  ckpt means={[round(c, 3) for c in ck_means]}"
        f"  cross_mean={mean(ck_means):.3f} cross_std={std(ck_means):.3f}"
    )
