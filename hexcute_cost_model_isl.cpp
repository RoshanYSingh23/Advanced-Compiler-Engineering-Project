#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <functional>
#include <cassert>


#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/space.h>
#include <isl/val.h>
#include <isl/constraint.h>
#include <isl/aff.h>
#include <isl/printer.h>
#include <isl/schedule.h>





struct HardwareConfig {
    int sm_count = 24;
    double clock_ghz = 1.59;
    double memory_bandwidth_gbs = 288.0;
    double l2_cache_mb = 1.5;
    int smem_banks = 32;
    int warp_size = 32;
    int vec_length = 8;
    double tensor_core_tflops = 5.0;
    int smem_size_kb = 64;
    std::string name = "GTX-1660Ti";
};





struct SwizzleParams {
    int B, M, S;

    std::string str() const {
        return "Swizzle<" + std::to_string(B) + "," +
               std::to_string(M) + "," + std::to_string(S) + ">";
    }

    int apply(int row, int col) const {
        int mask = (1 << B) - 1;
        int swizzle_bits = (row >> M) & mask;
        return col ^ (swizzle_bits << S);
    }
};





class HexCuteConstraintSolver {
    HardwareConfig hw_;
    isl_ctx* ctx_;

public:
    HexCuteConstraintSolver(const HardwareConfig& hw)
        : hw_(hw), ctx_(isl_ctx_alloc()) {}

    ~HexCuteConstraintSolver() {
        isl_ctx_free(ctx_);
    }


    SwizzleParams synthesize_swizzle() {








        int S = (int)std::log2(hw_.vec_length);
        int M = S;
        int B_ideal = (int)std::log2(hw_.smem_banks / hw_.vec_length);
        int B = std::min(B_ideal, 3);


        std::ostringstream ss;
        ss << "{ [b, m, s] : "
           << "s = " << S << " and "
           << "m = " << M << " and "
           << "0 <= b <= " << B << " }";

        isl_set* constraint_space = isl_set_read_from_str(ctx_, ss.str().c_str());


        isl_set* lex_max = isl_set_lexmax(constraint_space);


        isl_printer* printer = isl_printer_to_str(ctx_);
        printer = isl_printer_print_set(printer, lex_max);
        char* result_str = isl_printer_get_str(printer);

        std::cout << "  ISL constraint solution: " << result_str << "\n";

        free(result_str);
        isl_printer_free(printer);
        isl_set_free(lex_max);

        return {B, M, S};
    }


    isl_map* build_mma_thread_layout(int mma_m, int mma_n, int mma_k) {
        int threads_m = std::max(1, mma_m / hw_.vec_length);
        int threads_n = std::max(1, mma_n / hw_.vec_length);



        std::ostringstream ss;
        ss << "{ [lane] -> [r, c] : "
           << "exists (rm, cm : "
           << "lane = rm * " << threads_n << " + cm and "
           << "r = rm * " << hw_.vec_length << " and "
           << "c = cm * " << hw_.vec_length << " and "
           << "0 <= rm < " << threads_m << " and "
           << "0 <= cm < " << threads_n << " and "
           << "0 <= lane < " << hw_.warp_size << ") }";

        return isl_map_read_from_str(ctx_, ss.str().c_str());
    }


    int count_bank_conflicts_for_swizzle(const SwizzleParams& sw,
                                          int tile_rows, int tile_cols) {
        int total_conflicts = 0;


        for (int row = 0; row < std::min(tile_rows, 8); row++) {
            std::map<int, int> bank_counts;
            for (int tid = 0; tid < hw_.warp_size && tid < tile_cols; tid++) {
                int swizzled_col = sw.apply(row, tid);
                int offset = row * tile_cols + swizzled_col;
                int bank = offset % hw_.smem_banks;
                bank_counts[bank]++;
            }
            for (auto& [bank, count] : bank_counts)
                total_conflicts += std::max(0, count - 1);
        }


        for (int col = 0; col < std::min(tile_cols, 8); col++) {
            std::map<int, int> bank_counts;
            for (int tid = 0; tid < hw_.warp_size && tid < tile_rows; tid++) {
                int swizzled_col = sw.apply(tid, col);
                int offset = tid * tile_cols + swizzled_col;
                int bank = offset % hw_.smem_banks;
                bank_counts[bank]++;
            }
            for (auto& [bank, count] : bank_counts)
                total_conflicts += std::max(0, count - 1);
        }

        return total_conflicts;
    }

    isl_ctx* get_ctx() { return ctx_; }
};





struct KernelCharacteristics {
    std::string name;
    double flops;
    double bytes_accessed;
    double bank_conflict_factor;
    double cache_miss_factor;
    double coalescing_factor;
    double reuse_factor;
};





struct CostBreakdown {
    double compute_cycles;
    double memory_cycles;
    double bank_conflict_penalty;
    double cache_miss_penalty;
    double base_cycles;
    double total_cycles;
    double predicted_time_ms;
    double predicted_tflops;
    double predicted_bw_gbs;
};

class HexCuteCostModel {
    HardwareConfig hw_;
    SwizzleParams swizzle_;

public:
    HexCuteCostModel(const HardwareConfig& hw, const SwizzleParams& sw)
        : hw_(hw), swizzle_(sw) {}

    CostBreakdown estimate(const KernelCharacteristics& kernel) const {
        CostBreakdown cost;


        double effective_tflops = hw_.tensor_core_tflops * kernel.reuse_factor;
        if (effective_tflops <= 0) effective_tflops = 0.1;
        double compute_time_s = kernel.flops / (effective_tflops * 1e12);
        cost.compute_cycles = compute_time_s * hw_.clock_ghz * 1e9;


        double effective_bw = hw_.memory_bandwidth_gbs * kernel.coalescing_factor;
        if (effective_bw <= 0) effective_bw = 1.0;
        double memory_time_s = kernel.bytes_accessed / (effective_bw * 1e9);
        cost.memory_cycles = memory_time_s * hw_.clock_ghz * 1e9;


        double swizzle_effectiveness = 1.0 / (1 << swizzle_.B);
        cost.bank_conflict_penalty = std::max(0.0,
            (kernel.bank_conflict_factor - 1.0) * 1000 * swizzle_effectiveness);


        cost.cache_miss_penalty = std::max(0.0,
            (kernel.cache_miss_factor - 1.0) * 2000);


        cost.base_cycles = std::max(cost.compute_cycles, cost.memory_cycles);
        cost.total_cycles = cost.base_cycles + cost.bank_conflict_penalty + cost.cache_miss_penalty;

        cost.predicted_time_ms = cost.total_cycles / (hw_.clock_ghz * 1e6);
        cost.predicted_tflops = kernel.flops / ((cost.predicted_time_ms / 1000.0) * 1e12);
        cost.predicted_bw_gbs = kernel.bytes_accessed / ((cost.predicted_time_ms / 1000.0) * 1e9);

        return cost;
    }
};





struct TileConfig {
    int block_m, block_n, block_k;
    int warp_m, warp_n;
    int stages;
    SwizzleParams swizzle;

    std::string str() const {
        std::ostringstream ss;
        ss << "Tile(" << block_m << "x" << block_n << "x" << block_k
           << "), Warp(" << warp_m << "x" << warp_n
           << "), Stages=" << stages << ", " << swizzle.str();
        return ss.str();
    }
};

class CuTlassTileSearcher {
    isl_ctx* ctx_;
    HardwareConfig hw_;

public:
    CuTlassTileSearcher(isl_ctx* ctx, const HardwareConfig& hw)
        : ctx_(ctx), hw_(hw) {}


    std::vector<TileConfig> search_tile_space(int M, int N, int K) {
        int smem_bytes = hw_.smem_size_kb * 1024;
        int elem_size = 2;



        std::vector<TileConfig> results;
        std::vector<int> bm_vals = {32, 64, 128, 256};
        std::vector<int> bn_vals = {32, 64, 128, 256};
        std::vector<int> bk_vals = {16, 32, 64};
        std::vector<int> stage_vals = {2, 3, 4};

        for (int bm : bm_vals) {
            if (bm > M) continue;
            for (int bn : bn_vals) {
                if (bn > N) continue;
                for (int bk : bk_vals) {
                    if (bk > K) continue;
                    for (int stages : stage_vals) {
                        int smem_needed = 2 * (bm * bk + bk * bn) * stages * elem_size;
                        if (smem_needed > smem_bytes) continue;


                        std::ostringstream ss;
                        ss << "{ [i, j, k] : 0 <= i < " << bm
                           << " and 0 <= j < " << bn
                           << " and 0 <= k < " << bk << " }";
                        isl_set* tile_domain = isl_set_read_from_str(ctx_, ss.str().c_str());
                        bool valid = !isl_set_is_empty(tile_domain);
                        isl_set_free(tile_domain);

                        if (valid) {
                            for (int B : {0, 1, 2, 3}) {
                                for (int Mv : {0, 3}) {
                                    for (int S : {0, 3}) {
                                        results.push_back({
                                            bm, bn, bk,
                                            bm / 2, bn / 2,
                                            stages,
                                            {B, Mv, S}
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return results;
    }


    double score_tile(const TileConfig& tile, int M, int N, int K,
                      HexCuteConstraintSolver& solver) {

        int bank_conflicts = solver.count_bank_conflicts_for_swizzle(
            tile.swizzle, tile.block_m, tile.block_k);


        int smem_used = 2 * (tile.block_m * tile.block_k +
                             tile.block_k * tile.block_n) * tile.stages * 2;
        double smem_util = (double)smem_used / (hw_.smem_size_kb * 1024.0);


        int num_ctas = ((M + tile.block_m - 1) / tile.block_m) *
                       ((N + tile.block_n - 1) / tile.block_n);
        double wave_efficiency = (double)num_ctas / hw_.sm_count;
        if (wave_efficiency > 1.0)
            wave_efficiency = 1.0 - (1.0 / wave_efficiency) * 0.1;


        return bank_conflicts * 10.0 + (1.0 - smem_util) * 100.0 +
               (1.0 - wave_efficiency) * 50.0;
    }
};





struct GemmShape {
    std::string name;
    int M, N, K;
    std::string model;
};

std::vector<GemmShape> get_all_gemm_shapes() {
    return {

        {"QKV_Proj",    512, 2304, 768,  "BERT"},
        {"Attn_Score",  512, 512,  64,   "BERT"},
        {"Attn_Value",  512, 64,   512,  "BERT"},
        {"Out_Proj",    512, 768,  768,  "BERT"},
        {"FFN_Up",      512, 3072, 768,  "BERT"},
        {"FFN_Down",    512, 768,  3072, "BERT"},

        {"QKV_Proj",    1024, 2304, 768,  "GPT2"},
        {"Attn_Score",  1024, 1024, 64,   "GPT2"},
        {"Attn_Value",  1024, 64,   1024, "GPT2"},
        {"Out_Proj",    1024, 768,  768,  "GPT2"},
        {"FFN_Up",      1024, 3072, 768,  "GPT2"},
        {"FFN_Down",    1024, 768,  3072, "GPT2"},
    };
}





int main() {
    HardwareConfig hw;
    std::cout << "================================================================\n";
    std::cout << "HexCute / CuTlass Cost Model — ISL + Boost C++ Implementation\n";
    std::cout << "Hardware: " << hw.name << " (" << hw.sm_count << " SMs, "
              << hw.clock_ghz << " GHz)\n";
    std::cout << "================================================================\n\n";


    std::cout << "--- Step 1: HexCute Swizzle Synthesis (ISL) ---\n";
    HexCuteConstraintSolver solver(hw);
    SwizzleParams hexcute_swizzle = solver.synthesize_swizzle();
    std::cout << "  Synthesized: " << hexcute_swizzle.str() << "\n";
    std::cout << "  B=" << hexcute_swizzle.B << " gives " << (1 << hexcute_swizzle.B)
              << " independent phases across " << hw.smem_banks << " banks\n\n";


    SwizzleParams f2_swizzle{3, 0, 0};
    SwizzleParams naive_swizzle{0, 0, 0};
    SwizzleParams cute_swizzle{3, 3, 3};


    std::cout << "--- Step 2: HexCute Cost Model Evaluation ---\n";

    int matrix_size = 1024;
    double flops = 2.0 * matrix_size * matrix_size * matrix_size;
    double bytes = matrix_size * matrix_size * 3.0 * 2;

    std::vector<KernelCharacteristics> kernels = {
        {"Baseline",  flops, bytes, 1.0,  1.0,  1.0,  1.0},
        {"Rotation",  flops, bytes, 1.15, 1.10, 0.92, 0.95},
        {"Butterfly", flops, bytes, 1.25, 1.18, 0.88, 0.93},
        {"Kronecker", flops, bytes, 1.10, 1.08, 0.95, 1.05},
        {"Toeplitz",  flops, bytes, 1.20, 1.22, 0.85, 0.90},
    };


    struct SwizzleEval {
        std::string name;
        SwizzleParams params;
    };
    std::vector<SwizzleEval> swizzle_configs = {
        {"Naive (no swizzle)", naive_swizzle},
        {"F₂ Optimal",        f2_swizzle},
        {"HexCute",            hexcute_swizzle},
        {"CuTe Default",       cute_swizzle},
    };

    std::ofstream csv("hexcute_cost_model_isl_results.csv");
    csv << "Kernel,Swizzle,ComputeCycles,MemoryCycles,BankPenalty,CachePenalty,"
        << "TotalCycles,PredTimeMs,PredTFLOPS,BankConflicts\n";

    for (auto& sw_eval : swizzle_configs) {
        HexCuteCostModel model(hw, sw_eval.params);
        std::cout << "\n  Swizzle: " << sw_eval.name << " " << sw_eval.params.str() << "\n";

        for (auto& kernel : kernels) {
            CostBreakdown cost = model.estimate(kernel);
            int bank_conflicts = solver.count_bank_conflicts_for_swizzle(
                sw_eval.params, 128, 128);

            std::cout << "    " << std::setw(12) << kernel.name
                      << ": " << std::fixed << std::setprecision(4) << cost.predicted_time_ms
                      << " ms, " << std::setprecision(2) << cost.predicted_tflops
                      << " TFLOPS, bank_conflicts=" << bank_conflicts << "\n";

            csv << kernel.name << "," << sw_eval.name << ","
                << cost.compute_cycles << "," << cost.memory_cycles << ","
                << cost.bank_conflict_penalty << "," << cost.cache_miss_penalty << ","
                << cost.total_cycles << "," << cost.predicted_time_ms << ","
                << cost.predicted_tflops << "," << bank_conflicts << "\n";
        }
    }


    std::cout << "\n--- Step 3: CuTlass Tile Configuration Search (ISL) ---\n";

    CuTlassTileSearcher tile_searcher(solver.get_ctx(), hw);
    auto shapes = get_all_gemm_shapes();

    std::ofstream tile_csv("cutlass_tile_search_results.csv");
    tile_csv << "Model,Layer,M,N,K,BestTile,BestSwizzle,Score,BankConflicts\n";

    for (auto& shape : shapes) {
        auto tiles = tile_searcher.search_tile_space(shape.M, shape.N, shape.K);

        if (tiles.empty()) {
            std::cout << "  [" << shape.model << "/" << shape.name << "] No valid tiles found\n";
            continue;
        }


        double best_score = 1e18;
        TileConfig best_tile = tiles[0];
        for (auto& tile : tiles) {
            double score = tile_searcher.score_tile(tile, shape.M, shape.N, shape.K, solver);
            if (score < best_score) {
                best_score = score;
                best_tile = tile;
            }
        }

        int best_conflicts = solver.count_bank_conflicts_for_swizzle(
            best_tile.swizzle, best_tile.block_m, best_tile.block_k);

        std::cout << "  [" << shape.model << "/" << shape.name << "] "
                  << shape.M << "x" << shape.N << "x" << shape.K << "\n"
                  << "    Best: " << best_tile.str()
                  << " (score=" << std::fixed << std::setprecision(1) << best_score
                  << ", conflicts=" << best_conflicts << ")\n"
                  << "    Configs evaluated: " << tiles.size() << "\n";

        tile_csv << shape.model << "," << shape.name << ","
                 << shape.M << "," << shape.N << "," << shape.K << ","
                 << "\"" << best_tile.str() << "\","
                 << "\"" << best_tile.swizzle.str() << "\","
                 << best_score << "," << best_conflicts << "\n";
    }


    std::cout << "\n--- Step 4: Bank Conflict Comparison Across All Swizzles ---\n";
    std::cout << std::setw(8) << "Model" << std::setw(14) << "Layer"
              << std::setw(10) << "Naive" << std::setw(10) << "F₂"
              << std::setw(10) << "HexCute" << std::setw(10) << "CuTe"
              << std::setw(12) << "Best" << "\n";
    std::cout << std::string(74, '-') << "\n";

    for (auto& shape : shapes) {
        int tile_r = std::min(128, shape.M);
        int tile_c = std::min(128, std::min(shape.K, shape.N));

        int naive_c = solver.count_bank_conflicts_for_swizzle(naive_swizzle, tile_r, tile_c);
        int f2_c = solver.count_bank_conflicts_for_swizzle(f2_swizzle, tile_r, tile_c);
        int hexcute_c = solver.count_bank_conflicts_for_swizzle(hexcute_swizzle, tile_r, tile_c);
        int cute_c = solver.count_bank_conflicts_for_swizzle(cute_swizzle, tile_r, tile_c);

        int min_c = std::min({naive_c, f2_c, hexcute_c, cute_c});
        std::string best = "Naive";
        if (min_c == f2_c) best = "F₂";
        if (min_c == hexcute_c) best = "HexCute";
        if (min_c == cute_c) best = "CuTe";
        if (naive_c == f2_c && f2_c == hexcute_c && hexcute_c == cute_c)
            best = "TIE";

        std::cout << std::setw(8) << shape.model << std::setw(14) << shape.name
                  << std::setw(10) << naive_c << std::setw(10) << f2_c
                  << std::setw(10) << hexcute_c << std::setw(10) << cute_c
                  << std::setw(12) << best << "\n";
    }


    std::cout << "\n================================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "================================================================\n";
    std::cout << "  HexCute synthesized swizzle: " << hexcute_swizzle.str() << "\n";
    std::cout << "  F₂ optimal swizzle:          " << f2_swizzle.str() << "\n";
    std::cout << "  CuTe default swizzle:        " << cute_swizzle.str() << "\n";
    std::cout << "  Results saved to:\n";
    std::cout << "    - hexcute_cost_model_isl_results.csv\n";
    std::cout << "    - cutlass_tile_search_results.csv\n";
    std::cout << "\n  To verify with hardware:\n";
    std::cout << "    ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared \\\n";
    std::cout << "        python3 triton_gemm_bank_conflicts.py --profile\n";

    csv.close();
    tile_csv.close();

    return 0;
}
