#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sstream>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <functional>


#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/space.h>
#include <isl/point.h>
#include <isl/val.h>
#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/printer.h>


class F2Matrix {
    int rows_, cols_;
    std::vector<uint64_t> data_;

public:
    F2Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows, 0) {
        assert(cols <= 64);
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    void set(int r, int c, int val) {
        if (val & 1)
            data_[r] |= (1ULL << c);
        else
            data_[r] &= ~(1ULL << c);
    }

    int get(int r, int c) const {
        return (data_[r] >> c) & 1;
    }

    uint64_t row_bits(int r) const { return data_[r]; }


    int rank() const {
        std::vector<uint64_t> M = data_;
        int rank = 0;
        for (int col = 0; col < cols_; col++) {
            int pivot = -1;
            for (int row = rank; row < rows_; row++) {
                if ((M[row] >> col) & 1) {
                    pivot = row;
                    break;
                }
            }
            if (pivot == -1) continue;
            std::swap(M[rank], M[pivot]);
            for (int row = 0; row < rows_; row++) {
                if (row != rank && ((M[row] >> col) & 1)) {
                    M[row] ^= M[rank];
                }
            }
            rank++;
        }
        return rank;
    }


    F2Matrix sub_rows(int r0, int nr) const {
        F2Matrix sub(nr, cols_);
        for (int i = 0; i < nr; i++)
            sub.data_[i] = data_[r0 + i];
        return sub;
    }

    void print(const std::string& label = "") const {
        if (!label.empty()) std::cout << label << ":\n";
        for (int r = 0; r < rows_; r++) {
            std::cout << "  [";
            for (int c = 0; c < cols_; c++) {
                std::cout << get(r, c);
                if (c < cols_ - 1) std::cout << " ";
            }
            std::cout << "]\n";
        }
    }
};


std::vector<int> int_to_f2_vec(int val, int bits) {
    std::vector<int> v(bits);
    for (int i = 0; i < bits; i++)
        v[i] = (val >> i) & 1;
    return v;
}


struct SwizzleConfig {
    int B;
    int M;
    int S;

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


class ISLLayoutAnalyzer {
    isl_ctx* ctx_;

public:
    ISLLayoutAnalyzer() {
        ctx_ = isl_ctx_alloc();
    }

    ~ISLLayoutAnalyzer() {
        isl_ctx_free(ctx_);
    }

    isl_set* build_tile_domain(int tile_rows, int tile_cols) {
        std::ostringstream ss;
        ss << "{ [i, j] : 0 <= i < " << tile_rows
           << " and 0 <= j < " << tile_cols << " }";
        return isl_set_read_from_str(ctx_, ss.str().c_str());
    }


    isl_map* build_naive_access(int tile_rows, int tile_cols) {
        std::ostringstream ss;
        ss << "{ [i, j] -> [o] : o = i * " << tile_cols << " + j"
           << " and 0 <= i < " << tile_rows
           << " and 0 <= j < " << tile_cols << " }";
        return isl_map_read_from_str(ctx_, ss.str().c_str());
    }


    isl_map* build_skew_access(int tile_rows, int tile_cols, int factor) {
        std::ostringstream ss;
        ss << "{ [i, j] -> [o] : "
           << "exists q : "
           << "o = i * " << tile_cols << " + j + i * " << factor << " - q * " << tile_cols
           << " and 0 <= j + i * " << factor << " - q * " << tile_cols << " < " << tile_cols
           << " and 0 <= i < " << tile_rows
           << " and 0 <= j < " << tile_cols << " }";
        return isl_map_read_from_str(ctx_, ss.str().c_str());
    }


    int count_bank_conflicts_isl(int tile_rows, int tile_cols,
                                  std::function<int(int, int)> offset_fn,
                                  int num_banks = 32) {
        int total_conflicts = 0;


        for (int row = 0; row < std::min(tile_rows, 8); row++) {
            std::map<int, int> bank_counts;
            for (int tid = 0; tid < 32 && tid < tile_cols; tid++) {
                int offset = offset_fn(row, tid);
                int bank = offset % num_banks;
                bank_counts[bank]++;
            }
            for (auto& [bank, count] : bank_counts)
                total_conflicts += std::max(0, count - 1);
        }


        for (int col = 0; col < std::min(tile_cols, 8); col++) {
            std::map<int, int> bank_counts;
            for (int tid = 0; tid < 32 && tid < tile_rows; tid++) {
                int offset = offset_fn(tid, col);
                int bank = offset % num_banks;
                bank_counts[bank]++;
            }
            for (auto& [bank, count] : bank_counts)
                total_conflicts += std::max(0, count - 1);
        }

        return total_conflicts;
    }


    void analyze_warp_access_with_isl(int tile_rows, int tile_cols, int row_idx) {

        isl_set* warp = isl_set_read_from_str(ctx_,
            "{ [tid] : 0 <= tid < 32 }");


        std::ostringstream ss;
        ss << "{ [tid] -> [" << row_idx << ", tid] }";
        isl_map* thread_to_coord = isl_map_read_from_str(ctx_, ss.str().c_str());


        std::ostringstream os;
        os << "{ [i, j] -> [i * " << tile_cols << " + j] }";
        isl_map* coord_to_offset = isl_map_read_from_str(ctx_, os.str().c_str());


        isl_map* thread_to_offset = isl_map_apply_range(
            isl_map_copy(thread_to_coord), isl_map_copy(coord_to_offset));


        isl_set* offsets = isl_map_range(isl_map_copy(thread_to_offset));


        isl_printer* printer = isl_printer_to_str(ctx_);
        printer = isl_printer_print_set(printer, offsets);
        char* str_val = isl_printer_get_str(printer);
        std::cout << "    ISL warp offsets (row=" << row_idx << "): " << str_val << "\n";
        free(str_val);

        isl_printer_free(printer);
        isl_set_free(offsets);
        isl_map_free(thread_to_offset);
        isl_map_free(coord_to_offset);
        isl_map_free(thread_to_coord);
        isl_set_free(warp);
    }

    isl_ctx* get_ctx() { return ctx_; }
};



struct LayoutAnalysisResult {
    SwizzleConfig config;
    int f2_rank;
    bool bank_conflict_free;
    int simulated_conflicts;
    int row_conflicts;
    int col_conflicts;
};

F2Matrix build_layout_matrix(int tile_rows, int tile_cols,
                              const SwizzleConfig& swizzle, int num_bits = 10) {
    int row_bits = std::max(1, (int)std::ceil(std::log2(std::max(tile_rows, 2))));
    int col_bits = std::max(1, (int)std::ceil(std::log2(std::max(tile_cols, 2))));
    int input_bits = row_bits + col_bits;

    F2Matrix L(num_bits, input_bits);

    for (int bit_idx = 0; bit_idx < input_bits; bit_idx++) {
        int row = 0, col = 0;
        if (bit_idx < col_bits)
            col = 1 << bit_idx;
        else
            row = 1 << (bit_idx - col_bits);

        int swizzled_col = swizzle.apply(row, col);
        int offset = row * tile_cols + swizzled_col;

        auto bits = int_to_f2_vec(offset, num_bits);
        for (int b = 0; b < num_bits; b++)
            L.set(b, bit_idx, bits[b]);
    }

    return L;
}

LayoutAnalysisResult analyze_swizzle(int tile_rows, int tile_cols,
                                      const SwizzleConfig& config,
                                      ISLLayoutAnalyzer& isl_analyzer) {

    F2Matrix L = build_layout_matrix(tile_rows, tile_cols, config);
    F2Matrix bank_rows = L.sub_rows(0, 5);
    int rank = bank_rows.rank();


    auto offset_fn = [&](int row, int col) -> int {
        int sc = config.apply(row, col);
        return row * tile_cols + sc;
    };

    int conflicts = isl_analyzer.count_bank_conflicts_isl(
        tile_rows, tile_cols, offset_fn);


    int row_conflicts = 0, col_conflicts = 0;
    for (int r = 0; r < std::min(tile_rows, 8); r++) {
        std::map<int, int> bc;
        for (int tid = 0; tid < 32 && tid < tile_cols; tid++) {
            int off = offset_fn(r, tid);
            bc[off % 32]++;
        }
        for (auto& [_, c] : bc)
            row_conflicts += std::max(0, c - 1);
    }
    for (int c = 0; c < std::min(tile_cols, 8); c++) {
        std::map<int, int> bc;
        for (int tid = 0; tid < 32 && tid < tile_rows; tid++) {
            int off = offset_fn(tid, c);
            bc[off % 32]++;
        }
        for (auto& [_, cnt] : bc)
            col_conflicts += std::max(0, cnt - 1);
    }

    return {config, rank, rank >= 5, conflicts, row_conflicts, col_conflicts};
}



struct OptimalSearchResult {
    SwizzleConfig best_config;
    LayoutAnalysisResult best_result;
    std::vector<LayoutAnalysisResult> all_results;
};

OptimalSearchResult find_optimal_swizzle(int tile_rows, int tile_cols,
                                          ISLLayoutAnalyzer& isl_analyzer,
                                          int max_B = 3, int max_M = 3, int max_S = 3) {
    OptimalSearchResult search;
    int best_rank = -1;
    int best_conflicts = INT32_MAX;

    for (int B = 0; B <= max_B; B++) {
        for (int M = 0; M <= max_M; M++) {
            for (int S = 0; S <= max_S; S++) {
                SwizzleConfig config{B, M, S};
                auto result = analyze_swizzle(tile_rows, tile_cols, config, isl_analyzer);
                search.all_results.push_back(result);


                if (result.f2_rank > best_rank ||
                    (result.f2_rank == best_rank && result.simulated_conflicts < best_conflicts)) {
                    best_rank = result.f2_rank;
                    best_conflicts = result.simulated_conflicts;
                    search.best_config = config;
                    search.best_result = result;
                }
            }
        }
    }


    std::sort(search.all_results.begin(), search.all_results.end(),
        [](const LayoutAnalysisResult& a, const LayoutAnalysisResult& b) {
            if (a.f2_rank != b.f2_rank) return a.f2_rank > b.f2_rank;
            return a.simulated_conflicts < b.simulated_conflicts;
        });

    return search;
}


struct CustomLayoutResult {
    std::string name;
    int conflicts;
    int max_degree;
};

std::vector<CustomLayoutResult> search_custom_isl_mappings(
    int tile_rows, int tile_cols, ISLLayoutAnalyzer& isl_analyzer) {

    std::vector<CustomLayoutResult> results;


    for (int factor : {1, 2, 3, 4, 5, 7, 8, 11, 13, 15, 16, 17, 19, 23, 29, 31}) {
        auto offset_fn = [&](int row, int col) -> int {
            return row * tile_cols + (col + row * factor) % tile_cols;
        };
        int conflicts = isl_analyzer.count_bank_conflicts_isl(
            tile_rows, tile_cols, offset_fn);
        results.push_back({"skew_" + std::to_string(factor), conflicts, 0});
    }


    for (int bits = 1; bits <= 5; bits++) {
        auto offset_fn = [&](int row, int col) -> int {
            int mask = (1 << bits) - 1;
            return row * tile_cols + (col ^ (row & mask));
        };
        int conflicts = isl_analyzer.count_bank_conflicts_isl(
            tile_rows, tile_cols, offset_fn);
        results.push_back({"xor_" + std::to_string(bits) + "bit", conflicts, 0});
    }


    for (int factor : {1, 3, 7}) {
        for (int bits : {2, 3}) {
            auto offset_fn = [&](int row, int col) -> int {
                int skewed = (col + row * factor) % tile_cols;
                int mask = (1 << bits) - 1;
                return row * tile_cols + (skewed ^ (row & mask));
            };
            int conflicts = isl_analyzer.count_bank_conflicts_isl(
                tile_rows, tile_cols, offset_fn);
            results.push_back({"skew" + std::to_string(factor) +
                              "_xor" + std::to_string(bits), conflicts, 0});
        }
    }


    for (int p : {5, 7, 11, 13, 17, 19, 23, 29, 31}) {
        auto offset_fn = [&](int row, int col) -> int {
            return row * tile_cols + (col + row * p) % tile_cols;
        };
        int conflicts = isl_analyzer.count_bank_conflicts_isl(
            tile_rows, tile_cols, offset_fn);
        results.push_back({"prime_" + std::to_string(p), conflicts, 0});
    }

    std::sort(results.begin(), results.end(),
        [](const CustomLayoutResult& a, const CustomLayoutResult& b) {
            return a.conflicts < b.conflicts;
        });

    return results;
}



struct GemmConfig {
    std::string name;
    int M, N, K;
    int tile_r, tile_c;
    std::string desc;
};

std::vector<GemmConfig> get_bert_gemms() {
    return {
        {"QKV_Proj",    512, 2304, 768,  128, 128, "Q,K,V projection"},
        {"Attn_Score",  512, 512,  64,   64,  64,  "Attention scores"},
        {"Attn_Value",  512, 64,   512,  64,  64,  "Attention values"},
        {"Out_Proj",    512, 768,  768,  128, 128, "Output projection"},
        {"FFN_Up",      512, 3072, 768,  128, 128, "FFN up"},
        {"FFN_Down",    512, 768,  3072, 128, 128, "FFN down"},
    };
}

std::vector<GemmConfig> get_gpt2_gemms() {
    return {
        {"QKV_Proj",    1024, 2304, 768,  128, 128, "QKV fused"},
        {"Attn_Score",  1024, 1024, 64,   64,  64,  "Attention scores"},
        {"Attn_Value",  1024, 64,   1024, 64,  64,  "Attention values"},
        {"Out_Proj",    1024, 768,  768,  128, 128, "Output projection"},
        {"FFN_Up",      1024, 3072, 768,  128, 128, "FFN up"},
        {"FFN_Down",    1024, 768,  3072, 128, 128, "FFN down"},
    };
}



void write_json_results(const std::string& filename,
                        const std::map<std::string, std::vector<LayoutAnalysisResult>>& model_results,
                        const std::map<std::string, std::vector<CustomLayoutResult>>& custom_results) {
    std::ofstream out(filename);
    out << "{\n";

    bool first_model = true;
    for (auto& [model_name, results] : model_results) {
        if (!first_model) out << ",\n";
        first_model = false;
        out << "  \"" << model_name << "\": {\n";

        for (size_t i = 0; i < results.size(); i++) {
            auto& r = results[i];
            out << "    \"config_" << i << "\": {\n"
                << "      \"swizzle\": \"" << r.config.str() << "\",\n"
                << "      \"f2_rank\": " << r.f2_rank << ",\n"
                << "      \"bank_conflict_free\": " << (r.bank_conflict_free ? "true" : "false") << ",\n"
                << "      \"total_conflicts\": " << r.simulated_conflicts << ",\n"
                << "      \"row_conflicts\": " << r.row_conflicts << ",\n"
                << "      \"col_conflicts\": " << r.col_conflicts << "\n"
                << "    }";
            if (i < results.size() - 1) out << ",";
            out << "\n";
        }
        out << "  }";
    }

    if (!custom_results.empty()) {
        out << ",\n  \"custom_isl_mappings\": {\n";
        bool first_custom = true;
        for (auto& [layer, customs] : custom_results) {
            if (!first_custom) out << ",\n";
            first_custom = false;
            out << "    \"" << layer << "\": [\n";
            for (size_t i = 0; i < std::min(customs.size(), (size_t)10); i++) {
                out << "      {\"name\": \"" << customs[i].name
                    << "\", \"conflicts\": " << customs[i].conflicts << "}";
                if (i < std::min(customs.size(), (size_t)10) - 1) out << ",";
                out << "\n";
            }
            out << "    ]";
        }
        out << "\n  }";
    }

    out << "\n}\n";
    out.close();
}

=

int main() {
    ISLLayoutAnalyzer isl_analyzer;

    std::cout << "================================================================\n";
    std::cout << "F₂ Optimal Layout Selection — ISL C++ Implementation\n";
    std::cout << "================================================================\n\n";

    std::map<std::string, std::vector<LayoutAnalysisResult>> all_model_results;
    std::map<std::string, std::vector<CustomLayoutResult>> all_custom_results;


    auto bert_gemms = get_bert_gemms();
    std::cout << "--- BERT-Base Transformer Layers ---\n";
    for (auto& gemm : bert_gemms) {
        std::cout << "\n[" << gemm.name << "] GEMM(" << gemm.M << "x" << gemm.K
                  << ") x (" << gemm.K << "x" << gemm.N << "), Tile=" << gemm.tile_r
                  << "x" << gemm.tile_c << "\n";


        auto search = find_optimal_swizzle(gemm.tile_r, gemm.tile_c, isl_analyzer);
        std::cout << "  F₂ Optimal: " << search.best_config.str()
                  << " (rank=" << search.best_result.f2_rank
                  << ", conflicts=" << search.best_result.simulated_conflicts << ")\n";


        SwizzleConfig naive_cfg{0, 0, 0};
        auto naive = analyze_swizzle(gemm.tile_r, gemm.tile_c, naive_cfg, isl_analyzer);
        std::cout << "  Naive:       " << naive.simulated_conflicts << " conflicts\n";


        isl_analyzer.analyze_warp_access_with_isl(gemm.tile_r, gemm.tile_c, 0);


        auto custom = search_custom_isl_mappings(gemm.tile_r, gemm.tile_c, isl_analyzer);
        std::cout << "  Custom ISL mappings (top 5):\n";
        for (int i = 0; i < std::min((int)custom.size(), 5); i++) {
            std::cout << "    " << custom[i].name << ": " << custom[i].conflicts << " conflicts";
            if (custom[i].conflicts < search.best_result.simulated_conflicts) {
                std::cout << " ** BEATS F₂! **";
            }
            std::cout << "\n";
        }

        all_model_results["bert_" + gemm.name].push_back(search.best_result);
        all_custom_results["bert_" + gemm.name] = custom;
    }


    auto gpt2_gemms = get_gpt2_gemms();
    std::cout << "\n--- GPT-2 Transformer Layers ---\n";
    for (auto& gemm : gpt2_gemms) {
        std::cout << "\n[" << gemm.name << "] GEMM(" << gemm.M << "x" << gemm.K
                  << ") x (" << gemm.K << "x" << gemm.N << "), Tile=" << gemm.tile_r
                  << "x" << gemm.tile_c << "\n";

        auto search = find_optimal_swizzle(gemm.tile_r, gemm.tile_c, isl_analyzer);
        std::cout << "  F₂ Optimal: " << search.best_config.str()
                  << " (rank=" << search.best_result.f2_rank
                  << ", conflicts=" << search.best_result.simulated_conflicts << ")\n";

        SwizzleConfig naive_cfg{0, 0, 0};
        auto naive = analyze_swizzle(gemm.tile_r, gemm.tile_c, naive_cfg, isl_analyzer);
        std::cout << "  Naive:       " << naive.simulated_conflicts << " conflicts\n";

        auto custom = search_custom_isl_mappings(gemm.tile_r, gemm.tile_c, isl_analyzer);
        std::cout << "  Custom ISL mappings (top 5):\n";
        for (int i = 0; i < std::min((int)custom.size(), 5); i++) {
            std::cout << "    " << custom[i].name << ": " << custom[i].conflicts << " conflicts";
            if (custom[i].conflicts < search.best_result.simulated_conflicts) {
                std::cout << " ** BEATS F₂! **";
            }
            std::cout << "\n";
        }

        all_model_results["gpt2_" + gemm.name].push_back(search.best_result);
        all_custom_results["gpt2_" + gemm.name] = custom;
    }


    std::cout << "\n================================================================\n";
    std::cout << "SUMMARY: Can Custom ISL Mappings Beat F₂ Optimal?\n";
    std::cout << "================================================================\n";

    bool found_better = false;
    for (auto& [key, customs] : all_custom_results) {
        if (customs.empty()) continue;
        auto& f2_best = all_model_results[key].front();
        if (customs[0].conflicts < f2_best.simulated_conflicts) {
            found_better = true;
            std::cout << "  " << key << ": " << customs[0].name
                      << " (" << customs[0].conflicts << ") BEATS F₂ "
                      << f2_best.config.str() << " (" << f2_best.simulated_conflicts << ")\n";
        }
    }
    if (!found_better) {
        std::cout << "  No custom ISL mapping beat F₂ optimal in simulation.\n";
        std::cout << "  This is expected — F₂ rank=5 is theoretically optimal for 32 banks.\n";
        std::cout << "  However, hardware nsight measurements may reveal different behavior\n";
        std::cout << "  due to cache line effects, vectorized loads, and memory coalescing.\n";
    }


    write_json_results("f2_optimal_layout_isl_results.json", all_model_results, all_custom_results);
    std::cout << "\nResults saved to f2_optimal_layout_isl_results.json\n";

    return 0;
}
