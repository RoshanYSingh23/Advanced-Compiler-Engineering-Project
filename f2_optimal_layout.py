import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict





def int_to_f2_vec(val: int, bits: int) -> np.ndarray:
    """Convert an integer to a binary vector in F₂."""
    return np.array([(val >> i) & 1 for i in range(bits)], dtype=np.int32)

def f2_vec_to_int(vec: np.ndarray) -> int:
    """Convert a binary vector back to an integer."""
    return sum(int(v) << i for i, v in enumerate(vec))

def f2_rank(matrix: np.ndarray) -> int:
    """Compute rank of a binary matrix over F₂ using Gaussian elimination."""
    M = matrix.copy().astype(np.int32)
    rows, cols = M.shape
    rank = 0
    for col in range(cols):

        pivot = -1
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue

        M[[rank, pivot]] = M[[pivot, rank]]

        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] ^ M[rank])
        rank += 1
    return rank





@dataclass
class SwizzleConfig:
    B: int
    M: int
    S: int

    def __str__(self):
        return f"Swizzle<{self.B},{self.M},{self.S}>"

def apply_swizzle(row: int, col: int, config: SwizzleConfig) -> int:
    """
    Apply CuTe-style Swizzle<B,M,S> to compute the physical offset.

    The swizzle XORs B bits of the row (starting at bit M) into the column
    (starting at bit S). This redistributes data across banks.

    Unswizzled offset: row * num_cols + col
    Swizzled offset:   row * num_cols + (col ^ ((row >> M) & mask) << S)
    where mask = (1 << B) - 1
    """
    mask = (1 << config.B) - 1
    swizzle_bits = (row >> config.M) & mask
    swizzled_col = col ^ (swizzle_bits << config.S)
    return swizzled_col

def compute_bank(offset: int, num_banks: int = 32) -> int:
    """Compute which shared memory bank an offset maps to."""



    return (offset * 2 // 4) % num_banks





def count_bank_conflicts_for_warp(
    tile_rows: int, tile_cols: int,
    swizzle: SwizzleConfig,
    warp_size: int = 32,
    num_banks: int = 32
) -> Dict[str, any]:
    """
    Simulate a warp of 32 threads loading data from shared memory.
    Count how many bank conflicts occur for row-wise and column-wise access.

    Returns detailed conflict analysis.
    """
    results = {}


    row_conflicts = 0
    row_total_accesses = 0
    for row in range(min(tile_rows, 4)):
        banks_accessed = []
        for tid in range(warp_size):
            col = tid
            if col >= tile_cols:
                continue
            swizzled_col = apply_swizzle(row, col, swizzle)
            offset = row * tile_cols + swizzled_col
            bank = offset % num_banks
            banks_accessed.append(bank)


        if banks_accessed:
            bank_counts = {}
            for b in banks_accessed:
                bank_counts[b] = bank_counts.get(b, 0) + 1
            max_conflict = max(bank_counts.values())
            row_conflicts += max_conflict - 1
            row_total_accesses += 1

    results["row_access_conflicts"] = row_conflicts
    results["row_access_samples"] = row_total_accesses


    col_conflicts = 0
    col_total_accesses = 0
    for col in range(min(tile_cols, 4)):
        banks_accessed = []
        for tid in range(warp_size):
            row = tid
            if row >= tile_rows:
                continue
            swizzled_col = apply_swizzle(row, col, swizzle)
            offset = row * tile_cols + swizzled_col
            bank = offset % num_banks
            banks_accessed.append(bank)

        if banks_accessed:
            bank_counts = {}
            for b in banks_accessed:
                bank_counts[b] = bank_counts.get(b, 0) + 1
            max_conflict = max(bank_counts.values())
            col_conflicts += max_conflict - 1
            col_total_accesses += 1

    results["col_access_conflicts"] = col_conflicts
    results["col_access_samples"] = col_total_accesses
    results["total_conflicts"] = row_conflicts + col_conflicts

    return results

def build_layout_matrix_f2(
    tile_rows: int, tile_cols: int,
    swizzle: SwizzleConfig,
    num_bits: int = 10
) -> np.ndarray:
    """
    Build the F₂ layout matrix for a given tile and swizzle configuration.

    Each column of the matrix represents how one bit of the logical coordinate
    maps to the physical offset bits. The bottom 5 rows correspond to bank index bits.
    """


    row_bits = max(1, int(np.ceil(np.log2(max(tile_rows, 2)))))
    col_bits = max(1, int(np.ceil(np.log2(max(tile_cols, 2)))))
    input_bits = row_bits + col_bits


    L = np.zeros((num_bits, input_bits), dtype=np.int32)

    for bit_idx in range(input_bits):

        if bit_idx < col_bits:
            row, col = 0, 1 << bit_idx
        else:
            row, col = 1 << (bit_idx - col_bits), 0


        swizzled_col = apply_swizzle(row, col, swizzle)
        offset = row * tile_cols + swizzled_col


        L[:, bit_idx] = int_to_f2_vec(offset, num_bits)

    return L

def check_bank_freedom_f2(
    tile_rows: int, tile_cols: int,
    swizzle: SwizzleConfig
) -> Tuple[bool, int]:
    """
    Check if a swizzle configuration is bank-conflict-free using F₂ rank analysis.

    Returns (is_free, rank_of_bank_bits).
    The layout is bank-conflict-free if the bottom 5 rows of the layout matrix
    have rank = 5 (= log₂(32 banks)).
    """
    L = build_layout_matrix_f2(tile_rows, tile_cols, swizzle, num_bits=10)


    bank_rows = L[:5, :]
    rank = f2_rank(bank_rows)

    return rank == 5, rank





def find_optimal_swizzle(
    tile_rows: int, tile_cols: int,
    max_B: int = 3, max_M: int = 3, max_S: int = 3
) -> Tuple[SwizzleConfig, Dict]:
    """
    Search for the optimal Swizzle<B,M,S> that minimizes bank conflicts.

    Strategy:
    1. Enumerate all valid (B, M, S) configurations
    2. For each, compute the F₂ rank of bank-index bits
    3. Additionally count actual simulated bank conflicts
    4. Return the config with best F₂ rank and fewest simulated conflicts
    """
    best_config = SwizzleConfig(0, 0, 0)
    best_score = (-1, float('inf'))
    all_results = []

    for B in range(0, max_B + 1):
        for M_val in range(0, max_M + 1):
            for S in range(0, max_S + 1):
                config = SwizzleConfig(B, M_val, S)
                is_free, rank = check_bank_freedom_f2(tile_rows, tile_cols, config)
                conflicts = count_bank_conflicts_for_warp(tile_rows, tile_cols, config)

                total_conflicts = conflicts["total_conflicts"]
                score = (rank, -total_conflicts)

                result = {
                    "config": str(config),
                    "B": B, "M": M_val, "S": S,
                    "f2_rank": rank,
                    "bank_free": is_free,
                    "row_conflicts": conflicts["row_access_conflicts"],
                    "col_conflicts": conflicts["col_access_conflicts"],
                    "total_conflicts": total_conflicts,
                }
                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_config = config

    return best_config, {
        "best": str(best_config),
        "best_rank": best_score[0],
        "best_conflicts": -best_score[1],
        "all_configs_tested": len(all_results),
        "top_5": sorted(all_results, key=lambda x: (x["f2_rank"], -x["total_conflicts"]), reverse=True)[:5],
    }





BERT_GEMMS = {
    "QKV_Proj":    {"M": 512, "N": 2304, "K": 768,  "tile": (128, 128)},
    "Attn_Score":  {"M": 512, "N": 512,  "K": 64,   "tile": (64, 64)},
    "Attn_Value":  {"M": 512, "N": 64,   "K": 512,  "tile": (64, 64)},
    "Out_Proj":    {"M": 512, "N": 768,  "K": 768,  "tile": (128, 128)},
    "FFN_1":       {"M": 512, "N": 3072, "K": 768,  "tile": (128, 128)},
    "FFN_2":       {"M": 512, "N": 768,  "K": 3072, "tile": (128, 128)},
}

GPT2_GEMMS = {
    "QKV_Proj":    {"M": 1024, "N": 2304, "K": 768,  "tile": (128, 128)},
    "Attn_Score":  {"M": 1024, "N": 1024, "K": 64,   "tile": (64, 64)},
    "Attn_Value":  {"M": 1024, "N": 64,   "K": 1024, "tile": (64, 64)},
    "Out_Proj":    {"M": 1024, "N": 768,  "K": 768,  "tile": (128, 128)},
    "FFN_1":       {"M": 1024, "N": 3072, "K": 768,  "tile": (128, 128)},
    "FFN_2":       {"M": 1024, "N": 768,  "K": 3072, "tile": (128, 128)},
}

def analyze_model_layouts(model_name: str, gemm_configs: dict):
    """Find optimal F₂ swizzle for each GEMM layer in a model."""
    print(f"\n{'='*70}")
    print(f"F₂ OPTIMAL LAYOUT ANALYSIS: {model_name}")
    print(f"{'='*70}")

    results = {}
    for layer_name, cfg in gemm_configs.items():
        tile_r, tile_c = cfg["tile"]
        print(f"\n[{layer_name}] GEMM({cfg['M']}×{cfg['K']}) × ({cfg['K']}×{cfg['N']}), Tile={tile_r}×{tile_c}")


        best, details = find_optimal_swizzle(tile_r, tile_c)


        no_swizzle = count_bank_conflicts_for_warp(tile_r, tile_c, SwizzleConfig(0, 0, 0))

        print(f"  Naive (no swizzle):    {no_swizzle['total_conflicts']} bank conflicts")
        print(f"  F₂ Optimal:            {best} → {details['best_conflicts']} bank conflicts (rank={details['best_rank']})")
        print(f"  Bank-conflict-free:    {'YES' if details['best_rank'] >= 5 else 'NO'}")

        if details['top_5']:
            print(f"  Top configs:")
            for t in details['top_5'][:3]:
                print(f"    {t['config']}: rank={t['f2_rank']}, conflicts={t['total_conflicts']}")

        results[layer_name] = {
            "gemm_shape": f"({cfg['M']},{cfg['N']},{cfg['K']})",
            "tile_shape": f"({tile_r},{tile_c})",
            "naive_conflicts": no_swizzle['total_conflicts'],
            "f2_optimal_config": str(best),
            "f2_optimal_conflicts": details['best_conflicts'],
            "f2_rank": details['best_rank'],
            "bank_conflict_free": details['best_rank'] >= 5,
        }

    return results





def compute_layout_transition_cost(
    src_swizzle: SwizzleConfig,
    dst_swizzle: SwizzleConfig,
    tile_size: Tuple[int, int]
) -> int:
    """
    Compute the cost of transitioning between two layouts.
    If layouts are compatible (same swizzle), cost = 0.
    Otherwise, cost = number of elements that need re-shuffling.
    """
    if str(src_swizzle) == str(dst_swizzle):
        return 0


    rows, cols = tile_size
    mismatches = 0
    for r in range(min(rows, 32)):
        for c in range(min(cols, 32)):
            src_col = apply_swizzle(r, c, src_swizzle)
            dst_col = apply_swizzle(r, c, dst_swizzle)
            if src_col != dst_col:
                mismatches += 1
    return mismatches





if __name__ == "__main__":
    import json


    bert_results = analyze_model_layouts("BERT-Base (seq=512)", BERT_GEMMS)


    gpt2_results = analyze_model_layouts("GPT-2 (seq=1024)", GPT2_GEMMS)


    print(f"\n{'='*70}")
    print("CROSS-OPERATOR LAYOUT COMPATIBILITY (BERT)")
    print(f"{'='*70}")

    layers = list(BERT_GEMMS.keys())
    for i in range(len(layers) - 1):
        src = bert_results[layers[i]]["f2_optimal_config"]
        dst = bert_results[layers[i+1]]["f2_optimal_config"]

        compatible = src == dst
        print(f"  {layers[i]} → {layers[i+1]}: {'COMPATIBLE' if compatible else 'NEEDS CONVERSION'} ({src} → {dst})")


    all_results = {"bert": bert_results, "gpt2": gpt2_results}
    with open("f2_layout_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to f2_layout_results.json")
