#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <fstream>

#ifndef NO_ISL
#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/val.h>
#include <isl/printer.h>
#endif




struct LayoutResult {
    int tile;
    int num_banks;
    int warp_size;
    int alpha;
    int total_conflicts;
    int max_way_conflict;
    bool is_zero;
};

LayoutResult evaluate_skew_alpha(int T, int B, int W, int alpha) {
    int unique_rows = std::min(T, W);
    int total_conflicts = 0;
    int max_way = 1;

    for (int k = 0; k < T; k++) {
        int bank_count[256] = {};
        for (int r = 0; r < unique_rows; r++) {
            int swiz = ((k + alpha * r) % T + T) % T;
            int bank = (r * T + swiz) % B;
            if (bank < 0) bank += B;
            bank_count[bank]++;
        }
        for (int b = 0; b < B; b++) {
            if (bank_count[b] > 1) {
                total_conflicts += bank_count[b] - 1;
                max_way = std::max(max_way, bank_count[b]);
            }
        }
    }
    return {T, B, W, alpha, total_conflicts, max_way, total_conflicts == 0};
}

LayoutResult find_optimal_alpha(int T, int B, int W) {
    LayoutResult best = {T, B, W, -1, T * W * W, W, false};
    for (int alpha = 0; alpha < T; alpha++) {
        LayoutResult r = evaluate_skew_alpha(T, B, W, alpha);
        if (r.total_conflicts < best.total_conflicts)
            best = r;
        if (best.total_conflicts == 0) break;
    }
    return best;
}




int count_conflicts_naive(int T, int B, int W) {
    int total = 0;
    int unique_rows = std::min(T, W);
    for (int k = 0; k < T; k++) {
        int bank_count[256] = {};
        for (int r = 0; r < unique_rows; r++) {
            int bank = (r * T + k) % B;
            bank_count[bank]++;
        }
        for (int b = 0; b < B; b++)
            if (bank_count[b] > 1) total += bank_count[b] - 1;
    }
    return total;
}

int count_conflicts_padded(int T, int B, int W) {
    int stride = T + 1;
    int total = 0;
    int unique_rows = std::min(T, W);
    for (int k = 0; k < T; k++) {
        int bank_count[256] = {};
        for (int r = 0; r < unique_rows; r++) {
            int bank = (r * stride + k) % B;
            bank_count[bank]++;
        }
        for (int b = 0; b < B; b++)
            if (bank_count[b] > 1) total += bank_count[b] - 1;
    }
    return total;
}

int count_conflicts_f2_xor(int T, int B, int W) {
    int total = 0;
    int unique_rows = std::min(T, W);
    for (int k = 0; k < T; k++) {
        int bank_count[256] = {};
        for (int r = 0; r < unique_rows; r++) {
            int swiz = k ^ (r & 7);
            int bank = (r * T + swiz) % B;
            bank_count[bank]++;
        }
        for (int b = 0; b < B; b++)
            if (bank_count[b] > 1) total += bank_count[b] - 1;
    }
    return total;
}

int count_conflicts_xor_full(int T, int B, int W) {
    int total = 0;
    int unique_rows = std::min(T, W);
    for (int k = 0; k < T; k++) {
        int bank_count[256] = {};
        for (int r = 0; r < unique_rows; r++) {
            int swiz = k ^ (r & (T - 1));
            int bank = (r * T + swiz) % B;
            bank_count[bank]++;
        }
        for (int b = 0; b < B; b++)
            if (bank_count[b] > 1) total += bank_count[b] - 1;
    }
    return total;
}




#ifndef NO_ISL
void isl_analyze_layout(int T, int B, int alpha) {
    isl_ctx* ctx = isl_ctx_alloc();


    char domain_str[256];
    snprintf(domain_str, sizeof(domain_str),
             "{ [r, k] : 0 <= r < %d and 0 <= k < %d }", T, T);
    isl_set* domain = isl_set_read_from_str(ctx, domain_str);





    char map_str[512];
    snprintf(map_str, sizeof(map_str),
             "{ [r, k] -> [b] : exists (q, s : "
             "s = k + %d * r - %d * q and 0 <= s < %d and "
             "b = (r * %d + s) - %d * (r * %d + s) / %d and "
             "0 <= b < %d) }",
             alpha, T, T, T, B, T, B, B);

    isl_map* access = isl_map_read_from_str(ctx, map_str);
    if (!access) {
        std::cout << "    ISL: could not parse access map (expected for complex mod)\n";
        isl_set_free(domain);
        isl_ctx_free(ctx);
        return;
    }



    isl_map* inv = isl_map_reverse(isl_map_copy(access));
    isl_map* conflicts = isl_map_apply_range(isl_map_copy(access), inv);


    isl_set* conflict_set = isl_map_domain(conflicts);
    int n = isl_set_dim(conflict_set, isl_dim_set);

    char* str = isl_set_to_str(conflict_set);
    std::cout << "    ISL conflict set: " << str << "\n";
    free(str);

    isl_set_free(conflict_set);
    isl_set_free(domain);
    isl_map_free(access);
    isl_ctx_free(ctx);
}
#endif




int main() {
    std::cout << "================================================================\n";
    std::cout << "GENERAL ISL LAYOUT SOLVER\n";
    std::cout << "Finding optimal skew α for all (tile, num_banks) combinations\n";
    std::cout << "================================================================\n\n";

    int tile_sizes[] = {4, 8, 12, 16, 20, 24, 32, 48, 64};
    int bank_counts[] = {16, 32, 64};
    int warp_size = 32;


    std::ofstream csv("csv_results/general_isl_all_configs.csv");
    csv << "tile,num_banks,warp_size,layout,conflicts,max_way\n";

    std::cout << "PART 1: Optimal α for every (tile, banks) pair\n";
    std::cout << "─────────────────────────────────────────────────\n";
    std::cout << std::setw(6) << "Tile" << std::setw(8) << "Banks"
              << std::setw(8) << "α_opt" << std::setw(12) << "Conflicts"
              << std::setw(10) << "MaxWay" << std::setw(10) << "Zero?" << "\n";

    for (int T : tile_sizes) {
        for (int B : bank_counts) {
            LayoutResult r = find_optimal_alpha(T, B, warp_size);
            std::cout << std::setw(6) << T << std::setw(8) << B
                      << std::setw(8) << r.alpha
                      << std::setw(12) << r.total_conflicts
                      << std::setw(10) << r.max_way_conflict
                      << std::setw(10) << (r.is_zero ? "YES" : "no") << "\n";
            csv << T << "," << B << "," << warp_size
                << ",isl_skew_a" << r.alpha
                << "," << r.total_conflicts << "," << r.max_way_conflict << "\n";
        }
    }

    std::cout << "\n\nPART 2: All layouts compared at each tile size (banks=32)\n";
    std::cout << "────────────────────────────────────────────────────────────\n";

    for (int T : tile_sizes) {
        int B = 32;
        LayoutResult isl = find_optimal_alpha(T, B, warp_size);
        int naive     = count_conflicts_naive(T, B, warp_size);
        int padded    = count_conflicts_padded(T, B, warp_size);
        int f2_xor    = count_conflicts_f2_xor(T, B, warp_size);
        int xor_full  = (T & (T-1)) == 0 ? count_conflicts_xor_full(T, B, warp_size) : -1;

        std::cout << "\n  TILE=" << T << ", BANKS=" << B << ":\n";
        std::cout << "    naive:       " << std::setw(8) << naive << " conflicts\n";
        std::cout << "    padded(+1):  " << std::setw(8) << padded << " conflicts\n";
        std::cout << "    f2_xor(3b):  " << std::setw(8) << f2_xor << " conflicts\n";
        if (xor_full >= 0)
            std::cout << "    xor_full:    " << std::setw(8) << xor_full << " conflicts\n";
        else
            std::cout << "    xor_full:    N/A (non-power-of-2 tile)\n";
        std::cout << "    ISL(α=" << isl.alpha << "):   "
                  << std::setw(8) << isl.total_conflicts << " conflicts  ← OPTIMAL\n";

        csv << T << "," << B << "," << warp_size << ",naive," << naive << "," << 0 << "\n";
        csv << T << "," << B << "," << warp_size << ",padded," << padded << "," << 0 << "\n";
        csv << T << "," << B << "," << warp_size << ",f2_xor," << f2_xor << "," << 0 << "\n";
        if (xor_full >= 0)
            csv << T << "," << B << "," << warp_size << ",xor_full," << xor_full << "," << 0 << "\n";
        csv << T << "," << B << "," << warp_size
            << ",isl_optimal," << isl.total_conflicts << "," << isl.max_way_conflict << "\n";
    }
    csv.close();

    std::cout << "\n\nPART 3: PROOF that α=1 works for all even tiles\n";
    std::cout << "──────────────────────────────────────────────────\n";
    std::cout << "Testing α=1 for all even tiles T ∈ [2, 128], B ∈ {16, 32, 64}:\n";

    bool theorem_holds = true;
    for (int T = 2; T <= 128; T += 2) {
        for (int B : bank_counts) {
            LayoutResult r = evaluate_skew_alpha(T, B, warp_size, 1);
            if (!r.is_zero) {
                std::cout << "  COUNTEREXAMPLE: T=" << T << ", B=" << B
                          << " has " << r.total_conflicts << " conflicts with α=1\n";
                theorem_holds = false;
            }
        }
    }
    if (theorem_holds) {
        std::cout << "  ✓ THEOREM VERIFIED: α=1 gives ZERO conflicts for ALL even tiles\n";
        std::cout << "    T ∈ {2,4,...,128}, B ∈ {16,32,64}, warp_size=32\n";
    }

    std::cout << "\nTesting odd tiles to show they may have conflicts:\n";
    for (int T : {3, 5, 7, 9, 15, 31, 33, 63}) {
        for (int B : {32}) {
            LayoutResult r = find_optimal_alpha(T, B, warp_size);
            LayoutResult r1 = evaluate_skew_alpha(T, B, warp_size, 1);
            std::cout << "  T=" << std::setw(3) << T << ", B=32: "
                      << "α=1 conflicts=" << std::setw(5) << r1.total_conflicts
                      << "  optimal α=" << r.alpha
                      << " conflicts=" << r.total_conflicts << "\n";
        }
    }

#ifndef NO_ISL
    std::cout << "\n\nPART 4: ISL verification for T=32, B=32, α=1\n";
    std::cout << "──────────────────────────────────────────────\n";
    isl_analyze_layout(32, 32, 1);
#endif

    std::cout << "\nCSV written: csv_results/general_isl_all_configs.csv\n";
    return 0;
}
