"""Cross-check stress-kernel ncu counters against bank-index predictions.

This is not used as final evidence.  It explains whether the hardware counters
are plausible for the synthetic stress access patterns, accounting for shared
memory multicast when multiple lanes read the same address.
"""

from __future__ import annotations

import csv
from pathlib import Path


OUT_DIR = Path("hardware_results")
STRESS_ITERS = 2048
BLOCKS = 64
WARPS_PER_BLOCK = 8
BASE = STRESS_ITERS * BLOCKS * WARPS_PER_BLOCK

LAYOUT_IDS = {
    "naive": 0,
    "padded": 1,
    "f2_swizzle_3_0_0": 2,
    "hexcute_swizzle_2_3_3": 3,
    "cute_swizzle_3_3_3": 4,
    "isl_affine_a1": 5,
    "isl_affine_a3": 6,
    "isl_blocked_affine": 7,
}

PATTERN_IDS = {
    "stress_col0_row_lane": 10,
    "stress_col0_row_2lane": 11,
    "stress_col_lane_row_lane": 12,
    "stress_col_3lane_row_lane": 13,
}


def swizzle_col(tile_k: int, layout: int, row: int, col: int) -> int:
    if layout in (0, 1):
        return col
    if layout == 2:
        return col ^ (((row >> 0) & 7) << 0)
    if layout == 3:
        return col ^ (((row >> 3) & 3) << 3)
    if layout == 4:
        return col ^ (((row >> 3) & 7) << 3)
    if layout == 5:
        return (col + row) % tile_k
    if layout == 6:
        return (col + 3 * row) % tile_k
    return (col + row + (row // 8)) % tile_k


def access_for_pattern(tile_k: int, pattern: int, lane: int) -> tuple[int, int]:
    if pattern == 10:
        return lane, 0
    if pattern == 11:
        return (2 * lane) & 31, 0
    if pattern == 12:
        return lane, lane % tile_k
    if pattern == 13:
        return lane, (3 * lane) % tile_k
    raise ValueError(f"unknown stress pattern {pattern}")


def expected_conflicts(tile_k: int, layout: int, pattern: int) -> tuple[int, int, int]:
    stride = tile_k + 1 if layout == 1 else tile_k
    addresses_by_bank: dict[int, set[int]] = {}
    lanes_by_bank: dict[int, list[int]] = {}

    for lane in range(32):
        row, col = access_for_pattern(tile_k, pattern, lane)
        scol = swizzle_col(tile_k, layout, row, col)
        address = row * stride + scol
        bank = address % 32
        addresses_by_bank.setdefault(bank, set()).add(address)
        lanes_by_bank.setdefault(bank, []).append(lane)

    max_distinct_addresses = max(len(addresses) for addresses in addresses_by_bank.values())
    max_lanes = max(len(lanes) for lanes in lanes_by_bank.values())
    predicted = (max_distinct_addresses - 1) * BASE
    return max_lanes, max_distinct_addresses, predicted


def main() -> None:
    in_path = OUT_DIR / "hardware_only_stress_results.csv"
    out_path = OUT_DIR / "stress_bank_model_verification.csv"
    with in_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = [
        "tile_k",
        "layout",
        "tag_name",
        "max_lanes_per_bank",
        "max_distinct_addresses_per_bank",
        "predicted_conflicts",
        "hardware_total_conflicts",
        "absolute_error",
        "note",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            tile_k = int(row["tile_k"])
            layout = LAYOUT_IDS[row["layout"]]
            pattern = PATTERN_IDS[row["tag_name"]]
            max_lanes, max_distinct, predicted = expected_conflicts(tile_k, layout, pattern)
            hardware = int(row["total_conflicts"])
            note = "matches"
            if predicted == 0 and hardware != 0:
                note = "near_zero_hardware_noise" if hardware < 1024 else "mismatch"
            elif abs(hardware - predicted) > 1024:
                note = "mismatch"
            writer.writerow({
                "tile_k": tile_k,
                "layout": row["layout"],
                "tag_name": row["tag_name"],
                "max_lanes_per_bank": max_lanes,
                "max_distinct_addresses_per_bank": max_distinct,
                "predicted_conflicts": predicted,
                "hardware_total_conflicts": hardware,
                "absolute_error": abs(hardware - predicted),
                "note": note,
            })
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
