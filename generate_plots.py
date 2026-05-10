import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HR = Path("hardware_results")
SV = HR / "server_verification"
FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

COLORS = {
    "naive": "#d62728",
    "padded": "#ff7f0e",
    "f2_swizzle_3_0_0": "#2ca02c",
    "hexcute_swizzle_2_3_3": "#1f77b4",
    "cute_swizzle_3_3_3": "#9467bd",
    "isl_affine_a1": "#8c564b",
    "isl_affine_a3": "#e377c2",
    "isl_blocked_affine": "#17becf",
}

SHORT = {
    "naive": "Naive",
    "padded": "Padded",
    "f2_swizzle_3_0_0": "F$_2$ Swizzle",
    "hexcute_swizzle_2_3_3": "HexCute",
    "cute_swizzle_3_3_3": "CuTe",
    "isl_affine_a1": "ISL Skewed",
    "isl_affine_a3": "ISL Skewed$_3$",
    "isl_blocked_affine": "ISL Blocked",
}


def millions(x, _):
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def save(fig, stem):
    fig.savefig(FIGS / f"{stem}.png", bbox_inches="tight", dpi=200)
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {stem}.png")


def parse_raw_bank_metrics(path):
    metrics = {"total": 0, "load": 0, "store": 0, "gpu_time_ns": 0}
    if not path.exists():
        return metrics
    header = None
    with path.open(errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("=="):
                continue
            row = next(csv.reader([line]))
            if "Kernel Name" in row:
                header = row
                continue
            if not header or len(row) < len(header):
                continue
            for idx, col in enumerate(header):
                name = col.strip().lower()
                cell = row[idx].replace(",", "").strip()
                if not cell:
                    continue
                try:
                    value = int(cell)
                except ValueError:
                    continue
                if "bank_conflicts" in name and "op_ld" not in name and "op_st" not in name:
                    metrics["total"] += value
                elif "bank_conflicts" in name and "op_ld" in name:
                    metrics["load"] += value
                elif "bank_conflicts" in name and "op_st" in name:
                    metrics["store"] += value
                elif "gpu__time_duration.sum" in name:
                    metrics["gpu_time_ns"] += value
    return metrics


def parse_per_iter_ms(path):
    if not path.exists():
        return 0.0
    for line in path.read_text().splitlines():
        if line.startswith("per_iter_ms="):
            return float(line.split("=", 1)[1].strip())
    return 0.0


def parse_json_ms(path):
    if not path.exists():
        return 0.0
    return float(json.loads(path.read_text()).get("per_iter_ms", 0.0))


def fig_pass_gemm_bank_conflicts():
    orig = json.loads((SV / "matmul_pass_ncu_summary.json").read_text())["orig"]
    f2 = json.loads((SV / "matmul_pass_ncu_summary.json").read_text())["f2"]
    labels = ["Total", "Load", "Store", "Load + Store"]
    orig_vals = [orig["total"], orig["load"], orig["store"], orig["load"] + orig["store"]]
    f2_vals = [f2["total"], f2["load"], f2["store"], f2["load"] + f2["store"]]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars1 = ax.bar(x - w / 2, orig_vals, w, label="Original TTGIR/PTX", color="#c44e52")
    bars2 = ax.bar(x + w / 2, f2_vals, w, label="Pass-output TTGIR/PTX", color="#55a868")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height())}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bank Conflicts")
    ax.set_title("Pass-Driven GEMM: Input vs Output Bank Conflicts")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_pass_gemm_bank_conflicts")


def fig_pass_gemm_timing():
    orig = parse_per_iter_ms(SV / "matmul_pass_orig_timing_serial.txt")
    f2 = parse_per_iter_ms(SV / "matmul_pass_f2_timing_serial.txt")
    labels = ["Original TTGIR/PTX", "Pass-output TTGIR/PTX"]
    vals = [orig, f2]
    cols = ["#c44e52", "#55a868"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    bars = ax.bar(labels, vals, color=cols)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Per-Iteration Time (ms)")
    ax.set_title("Pass-Driven GEMM: Direct Timing")
    fig.tight_layout()
    save(fig, "fig_pass_gemm_timing")


def fig_isolated_gemm():
    rows = list(csv.DictReader(open(HR / "hardware_only_layout_results.csv")))
    data = defaultdict(lambda: defaultdict(int))
    for r in rows:
        tc = int(r["total_conflicts"])
        if tc > 0:
            data[(r["tile_k"], r["layout"])]["total"] += tc
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, tk in zip(axes, ["32", "64", "128"]):
        layouts = []
        vals = []
        for lay in ["naive", "padded"]:
            key = (tk, lay)
            if key in data and data[key]["total"] > 0:
                layouts.append(SHORT.get(lay, lay))
                vals.append(data[key]["total"])
        if vals:
            cols = [COLORS["naive"], COLORS["padded"]][: len(vals)]
            bars = ax.bar(layouts, vals, color=cols, edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:,.0f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_title(f"tile$_k$ = {tk}", fontsize=11)
        ax.set_ylabel("Total Bank Conflicts" if tk == "32" else "")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
        ax.tick_params(axis="x", rotation=20)
        ax.text(
            0.5,
            0.92,
            "F$_2$, HexCute, CuTe, ISL: all 0",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            fontstyle="italic",
            color="green",
            bbox=dict(facecolor="white", edgecolor="green", alpha=0.7, pad=2),
        )
    fig.suptitle("Isolated GEMM Bank Conflicts (ncu hardware, non-zero only)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_isolated_gemm")


def fig_stress_comparison():
    rows = list(csv.DictReader(open(HR / "hardware_only_stress_results.csv")))
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        agg[(r["tile_k"], r["layout"])]["total"] += int(r["total_conflicts"])
    layout_order = [
        "naive",
        "padded",
        "f2_swizzle_3_0_0",
        "hexcute_swizzle_2_3_3",
        "cute_swizzle_3_3_3",
        "isl_affine_a1",
        "isl_affine_a3",
        "isl_blocked_affine",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, tk in zip(axes, ["32", "64", "128"]):
        lays = []
        vals = []
        cols = []
        for lay in layout_order:
            value = agg[(tk, lay)]["total"]
            if value > 0:
                lays.append(SHORT.get(lay, lay))
                vals.append(value)
                cols.append(COLORS.get(lay, "#999999"))
        bars = ax.bar(lays, vals, color=cols, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v / 1e6:.1f}M",
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_title(f"tile$_k$ = {tk}", fontsize=11)
        ax.set_ylabel("Sum of Bank Conflicts (4 patterns)" if tk == "32" else "")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.suptitle("Stress-Kernel Bank Conflicts: ISL vs F$_2$/HexCute/CuTe (ncu hardware)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_stress_comparison")


def fig_stress_heatmap():
    rows = list(csv.DictReader(open(HR / "hardware_only_stress_results.csv")))
    layout_order = [
        "naive",
        "padded",
        "f2_swizzle_3_0_0",
        "hexcute_swizzle_2_3_3",
        "isl_affine_a1",
        "isl_blocked_affine",
    ]
    patterns = sorted(set(r["tag_name"] for r in rows))
    data = {}
    for r in rows:
        if r["tile_k"] == "64":
            data[(r["layout"], r["tag_name"])] = int(r["total_conflicts"])
    mat = []
    y_labels = []
    for lay in layout_order:
        row_vals = [data.get((lay, pat), 0) for pat in patterns]
        if any(v > 0 for v in row_vals):
            mat.append(row_vals)
            y_labels.append(SHORT.get(lay, lay))
    mat_arr = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat_arr, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels([p.replace("stress_", "") for p in patterns], fontsize=8, rotation=25, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    for i in range(len(y_labels)):
        for j in range(len(patterns)):
            v = mat_arr[i, j]
            label = f"{v / 1e6:.1f}M" if v >= 1e6 else (f"{v:,.0f}" if v > 0 else "0")
            color = "white" if v > mat_arr.max() * 0.6 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label="Bank Conflicts", shrink=0.8)
    ax.set_title("Stress-Kernel Conflicts by Layout and Pattern (tile$_k$=64, ncu)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_stress_heatmap_tk64")


def fig_isl_wins():
    rows = list(csv.DictReader(open(HR / "hardware_isl_wins.csv")))
    labels = []
    ref_vals = []
    isl_vals = []
    for r in rows:
        labels.append(f"tk={r['tile_k']}\\n{r['tag_name'].replace('stress_', '')}")
        ref_vals.append(int(r["best_reference_conflicts"]))
        isl_vals.append(int(r["best_isl_conflicts"]))
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - w / 2, ref_vals, w, label="Best F$_2$/HexCute", color=COLORS["f2_swizzle_3_0_0"])
    bars2 = ax.bar(x + w / 2, isl_vals, w, label="Best ISL", color=COLORS["isl_blocked_affine"])
    for bar, v in zip(bars1, ref_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v / 1e6:.1f}M",
                    ha="center", va="bottom", fontsize=6.5)
    for bar, v in zip(bars2, isl_vals):
        label = f"{v / 1e6:.1f}M" if v >= 1e6 else (f"{v:,}" if v > 0 else "0")
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 50000), label,
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Bank Conflicts")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.legend(fontsize=9)
    ax.set_title("ISL Wins: Per-Pattern Comparison (ncu hardware)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig_isl_wins")


def fig_analytical_verification():
    rows = list(csv.DictReader(open(HR / "stress_bank_model_verification.csv")))
    predicted = []
    measured = []
    for r in rows:
        pr = int(r["predicted_conflicts"])
        hw = int(r["hardware_total_conflicts"])
        if pr > 0 or hw > 0:
            predicted.append(pr)
            measured.append(hw)
    fig, ax = plt.subplots(figsize=(6, 6))
    mx = max(max(predicted), max(measured)) * 1.1
    ax.scatter(predicted, measured, s=18, alpha=0.7, edgecolors="black", linewidth=0.3)
    ax.plot([0, mx], [0, mx], "r--", linewidth=1, label="Predicted = Hardware")
    ax.set_xlabel("Predicted Conflicts (Analytical Model)")
    ax.set_ylabel("Hardware Conflicts (ncu)")
    ax.set_title("Analytical Model vs Hardware Verification", fontsize=11)
    ax.legend()
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
    fig.tight_layout()
    save(fig, "fig_analytical_verification")


def fig_cpp_bert_naive():
    entries = []
    for lid, name in [
        (0, "naive"),
        (1, "padded"),
        (2, "f2_swizzle_3_0_0"),
        (3, "hexcute_swizzle_2_3_3"),
        (5, "isl_affine_a1"),
        (7, "isl_blocked_affine"),
    ]:
        total = 0
        path = HR / f"ncu_model_bert_layout{lid}_tk64_summary.csv"
        if path.exists():
            with path.open() as f:
                for row in csv.DictReader(f):
                    total += int(row.get("total_conflicts", 0))
        if total > 0:
            entries.append((SHORT.get(name, name), total, COLORS.get(name, "#999999")))
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    cols = [e[2] for e in entries]
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v / 1e6:.1f}M",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total Bank Conflicts")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.set_title("C++ BERT Runner (12 layers, tile$_k$=64): Naive vs Optimized Layouts (ncu)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_cpp_bert_naive")


def fig_real_bert():
    entries = []
    for lid, name in [
        (2, "f2_swizzle_3_0_0"),
        (3, "hexcute_swizzle_2_3_3"),
        (5, "isl_affine_a1"),
        (7, "isl_blocked_affine"),
    ]:
        stats = parse_raw_bank_metrics(SV / f"ncu_real_bert_layout{lid}_tk64_raw.csv")
        if stats["total"] > 0:
            entries.append((SHORT.get(name, name), stats["total"], COLORS.get(name, "#999999")))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    cols = [e[2] for e in entries]
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v / 1e6:.1f}M",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total Bank Conflicts")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.set_title("Real HuggingFace BERT: Final 4-Layout Comparison (ncu)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig_real_bert")


def fig_real_bert_five_layout_total():
    specs = [
        ("naive", SV / "ncu_real_bert_layout0_tk64_live_raw.csv"),
        ("f2_swizzle_3_0_0", SV / "ncu_real_bert_layout2_tk64_live_raw.csv"),
        ("hexcute_swizzle_2_3_3", SV / "ncu_real_bert_layout3_tk64_raw.csv"),
        ("isl_affine_a1", SV / "ncu_real_bert_layout5_tk64_raw.csv"),
        ("isl_blocked_affine", SV / "ncu_real_bert_layout7_tk64_raw.csv"),
    ]
    labels = []
    vals = []
    cols = []
    for name, path in specs:
        stats = parse_raw_bank_metrics(path)
        labels.append(SHORT.get(name, name))
        vals.append(stats["total"])
        cols.append(COLORS.get(name, "#999999"))
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v / 1e6:.2f}M",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Total Bank Conflicts")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.set_title("Real HuggingFace BERT: Five-Layout Total Bank Conflicts", fontsize=12)
    fig.tight_layout()
    save(fig, "fig_real_bert_5layout_bank_conflicts")


def fig_real_bert_five_layout_breakdown():
    specs = [
        ("naive", SV / "ncu_real_bert_layout0_tk64_live_raw.csv"),
        ("f2_swizzle_3_0_0", SV / "ncu_real_bert_layout2_tk64_live_raw.csv"),
        ("hexcute_swizzle_2_3_3", SV / "ncu_real_bert_layout3_tk64_raw.csv"),
        ("isl_affine_a1", SV / "ncu_real_bert_layout5_tk64_raw.csv"),
        ("isl_blocked_affine", SV / "ncu_real_bert_layout7_tk64_raw.csv"),
    ]
    labels = []
    loads = []
    stores = []
    for name, path in specs:
        stats = parse_raw_bank_metrics(path)
        labels.append(SHORT.get(name, name))
        loads.append(stats["load"])
        stores.append(stats["store"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bars1 = ax.bar(x, loads, label="Load Conflicts", color="#4c72b0")
    bars2 = ax.bar(x, stores, bottom=loads, label="Store Conflicts", color="#dd8452")
    for i, (ld, st) in enumerate(zip(loads, stores)):
        ax.text(i, ld + st, f"{(ld + st) / 1e6:.2f}M", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Bank Conflicts")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.set_title("Real HuggingFace BERT: Load/Store Conflict Breakdown", fontsize=12)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_real_bert_5layout_breakdown")


def main():
    print("Generating figures...")
    fig_pass_gemm_bank_conflicts()
    fig_pass_gemm_timing()
    fig_isolated_gemm()
    fig_stress_comparison()
    fig_stress_heatmap()
    fig_isl_wins()
    fig_analytical_verification()
    fig_cpp_bert_naive()
    fig_real_bert()
    fig_real_bert_five_layout_total()
    fig_real_bert_five_layout_breakdown()
    print(f"All figures saved to {FIGS}/")


if __name__ == "__main__":
    main()
