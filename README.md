# ACE Project

This folder is the GitHub-ready subset of the larger workspace, not the entire original directory.
The full workspace contains large vendored trees, local LLVM/Triton toolchains, downloaded archives, and temporary build products that do not belong in the repository. This curated folder keeps only the report, the project-owned source files, the Triton pass patch and source snapshots, the TTGIR/PTX artifacts used in the pass experiment, the raw and summary results used by the report, and the plots that appear in the report.


## What This Repository Contains

This repository covers four layers of the project:

1. Layout-search algorithms for F2, HexCute, and ISL-based shared-memory mappings.
2. CUDA and PyTorch experiments that measure bank conflicts on real workloads.
3. A real TritonGPU pass that rewrites `#triton_gpu.shared` encodings inside TTGIR and can be invoked with `triton-opt`.
4. The final report and the figures/results that support it.

## External Dependencies Not Included

These are intentionally not committed into this repository. They must be installed separately on any machine used to rebuild or rerun the experiments.

- NVIDIA CUDA toolkit with driver support
- Nsight Compute `ncu`
- Python 3
- PyTorch with CUDA
- HuggingFace `transformers`
- Triton source tree, specifically a `release/2.3.x` checkout
- A matching LLVM/MLIR toolchain for that Triton revision
- ISL development headers and libraries for the C++ ISL programs
- PPCG only if you want to reproduce the older PPCG-related exploration outside the final reported path

## Recommended Environment

For the final compiler path and the pass experiment, the easiest reproducible setup is:

- separate Triton checkout on `release/2.3.x`
- matching LLVM/MLIR bundle expected by that Triton branch
- CUDA-enabled machine with a working NVIDIA driver
- `ncu` available in `PATH` or at an absolute location

For the PyTorch BERT experiments, additionally install:

- `torch`
- `transformers`
- a compiler toolchain compatible with `torch.utils.cpp_extension.load_inline`

## Repository Layout

```text
ACE_Project/
├── README.md
├── report.tex
├── generate_plots.py
├── f2_optimal_layout.py
├── f2_optimal_layout_isl.cpp
├── hexcute_constraint_solver.py
├── hexcute_cost_model_impl.py
├── hexcute_cost_model_isl.cpp
├── general_isl_layout.cpp
├── verify_stress_conflicts.py
├── hardware_layout_kernels.cu
├── bert_real_model_with_custom_layouts.py
├── ptx_matmul_runner.cpp
├── figures/
├── hardware_results/
│   └── server_verification/
├── triton_pass/
├── ttgir_inputs/
├── ttgir_outputs/
└── ptx_outputs/
```

## File-By-File Guide

### Top Level Files

| File | Purpose | How To Use It |
| --- | --- | --- |
| `report.tex` | Final detailed project report. | Open directly in Overleaf or compile locally with LaTeX. It expects the PNG files in `figures/`. |
| `generate_plots.py` | Regenerates every plot used in `report.tex`. | Run from the repository root with `python3 generate_plots.py`. It reads `hardware_results/` and writes plots into `figures/`. |
| `f2_optimal_layout.py` | Python implementation of the F2 shared-memory swizzle search. | Use it to inspect or rerun the finite-field swizzle search outside Triton. It is the algorithmic prototype path for F2. |
| `f2_optimal_layout_isl.cpp` | C++ implementation of the F2 search with ISL-assisted analysis support. | Build it against ISL if you want a compiled version of the F2 search and its conflict evaluation logic. |
| `hexcute_constraint_solver.py` | HexCute-style algebraic layout synthesis in Python. | Use it to derive HexCute swizzle parameters from thread-value and bank constraints. |
| `hexcute_cost_model_impl.py` | HexCute cost model and evaluation logic in Python. | Use it to score candidate HexCute layouts and compare them against alternatives. |
| `hexcute_cost_model_isl.cpp` | C++ ISL-backed HexCute analysis path. | Build it if you want the compiled HexCute cost-model path described in the report. |
| `general_isl_layout.cpp` | Main ISL polyhedral layout solver. | Use it to explore affine and blocked-affine layouts, especially the ISL skewed and ISL blocked variants discussed in the report. |
| `verify_stress_conflicts.py` | Analytical verifier for the stress-kernel conflict model. | Run it to check predicted conflicts against hardware-measured conflicts for the stress patterns. |
| `hardware_layout_kernels.cu` | CUDA kernels for the isolated GEMM, stress kernels, and whole-model layout experiments. | This is the main non-Triton hardware benchmark source used for the older and broader layout-family experiments. |
| `bert_real_model_with_custom_layouts.py` | PyTorch + CUDA-extension driver that replaces `nn.Linear` with custom layout-aware GEMM kernels. | Run it directly, or under `ncu`, to profile real HuggingFace BERT using naive, F2, HexCute, or ISL layouts. |
| `ptx_matmul_runner.cpp` | CUDA Driver API launcher for the exact Triton-generated PTX kernels. | Build it and use it for the pass-driven GEMM experiment so the input and output TTGIR/PTX kernels are compared directly. |

### Triton Pass Files

| File | Purpose | How To Use It |
| --- | --- | --- |
| `triton_pass/triton_release_2_3_x_f2_pass.patch` | Patch that captures the Triton source-tree modifications needed to register and build the pass in a real Triton checkout. | Apply it to a clean `release/2.3.x` Triton tree with `git apply`. |
| `triton_pass/F2SharedLayout.cpp` | The pass implementation. | This is the core TTGIR rewrite logic. It searches Triton-representable F2 swizzles and rewrites `SharedEncodingAttr` values. |
| `triton_pass/Passes.td` | Snapshot of the modified TritonGPU TableGen pass definition file. | Use it to inspect how the pass is declared and registered through TableGen. |
| `triton_pass/Passes.h` | Snapshot of the modified TritonGPU pass declaration header. | Use it to inspect the C++ pass factory declaration added for this pass. |

### TTGIR and PTX Artifacts

| File | Purpose | How To Use It |
| --- | --- | --- |
| `ttgir_inputs/matmul_kernel.ttgir.mlir` | Input TTGIR for the standalone GEMM pass experiment. | Feed this into `triton-opt --tritongpu-f2-shared-layout` inside a patched Triton build. |
| `ttgir_outputs/matmul_kernel_f2_from_pass.ttgir.mlir` | Actual TTGIR emitted by the Triton pass. | Use it to confirm that the pass rewrote the shared encodings and still produced valid TTGIR. |
| `ptx_outputs/matmul_kernel_orig.ptx` | PTX translated from the original TTGIR. | Feed it into `ptx_matmul_runner` or inspect it directly. |
| `ptx_outputs/matmul_kernel_f2.ptx` | PTX translated from the pass-output TTGIR. | Use it for the exact pass-output performance and conflict comparison. |

### Figures

| File | What It Shows |
| --- | --- |
| `figures/fig_pass_gemm_bank_conflicts.png` | Exact input-vs-output pass experiment on the standalone Triton GEMM kernel, including total, load, store, and `load + store` subtotals. |
| `figures/fig_pass_gemm_timing.png` | Direct timing comparison for the original and pass-output PTX kernels. |
| `figures/fig_isolated_gemm.png` | Isolated GEMM conflicts for naive and padded layouts across tile sizes, highlighting zero-conflict optimized layouts. |
| `figures/fig_stress_comparison.png` | Aggregate stress-kernel bank conflicts across layout families and tile sizes. |
| `figures/fig_stress_heatmap_tk64.png` | Per-pattern stress-kernel conflict heatmap at `tile_k = 64`. |
| `figures/fig_isl_wins.png` | Cases where the best ISL layout beats the best F2/HexCute layout. |
| `figures/fig_analytical_verification.png` | Analytical-model predictions versus hardware measurements for stress kernels. |
| `figures/fig_cpp_bert_naive.png` | Whole-model C++ BERT runner comparison showing naive versus optimized layouts. |
| `figures/fig_real_bert.png` | Real HuggingFace BERT comparison among the optimized layouts. |
| `figures/fig_real_bert_5layout_bank_conflicts.png` | Final five-layout real-BERT total conflict comparison: naive, F2, HexCute, ISL skewed, ISL blocked affine. |
| `figures/fig_real_bert_5layout_breakdown.png` | Final five-layout real-BERT load/store conflict breakdown. |

### Hardware Result Files

| File | Purpose |
| --- | --- |
| `hardware_results/hardware_only_layout_results.csv` | Isolated GEMM hardware results used for the isolated-GEMM figure and report section. |
| `hardware_results/hardware_only_stress_results.csv` | Raw aggregate stress-kernel conflict data used for the stress comparison and heatmap. |
| `hardware_results/hardware_isl_wins.csv` | Reduced table of the strongest ISL wins over F2/HexCute. |
| `hardware_results/stress_bank_model_verification.csv` | Predicted-versus-measured conflict pairs used in analytical verification. |
| `hardware_results/ncu_model_bert_layout0_tk64_summary.csv` | Summary CSV for the C++ BERT runner using the naive layout. |
| `hardware_results/ncu_model_bert_layout1_tk64_summary.csv` | Summary CSV for the C++ BERT runner using the padded layout. |
| `hardware_results/ncu_model_bert_layout2_tk64_summary.csv` | Summary CSV for the C++ BERT runner using the F2 layout. |
| `hardware_results/ncu_model_bert_layout3_tk64_summary.csv` | Summary CSV for the C++ BERT runner using the HexCute layout. |
| `hardware_results/ncu_model_bert_layout5_tk64_summary.csv` | Summary CSV for the C++ BERT runner using the ISL skewed layout. |
| `hardware_results/ncu_model_bert_layout7_tk64_summary.csv` | Summary CSV for the C++ BERT runner using the ISL blocked-affine layout. |

### Server Verification Files

| File | Purpose |
| --- | --- |
| `hardware_results/server_verification/matmul_pass_ncu_summary.json` | Compact summary of the exact pass-driven GEMM `ncu` run. |
| `hardware_results/server_verification/ncu_matmul_pass_orig_raw.csv` | Raw Nsight Compute CSV for the original TTGIR/PTX GEMM kernel. |
| `hardware_results/server_verification/ncu_matmul_pass_f2_raw.csv` | Raw Nsight Compute CSV for the pass-output TTGIR/PTX GEMM kernel. |
| `hardware_results/server_verification/matmul_pass_orig_timing_serial.txt` | Direct serial timing output for the original PTX kernel. |
| `hardware_results/server_verification/matmul_pass_f2_timing_serial.txt` | Direct serial timing output for the pass-output PTX kernel. |
| `hardware_results/server_verification/real_bert_layout0_tk64_live.json` | Runtime summary for the real BERT naive-layout run. |
| `hardware_results/server_verification/real_bert_layout2_tk64_live.json` | Runtime summary for the real BERT F2-layout run. |
| `hardware_results/server_verification/real_bert_layout3_tk64.json` | Runtime summary for the real BERT HexCute-layout run. |
| `hardware_results/server_verification/real_bert_layout5_tk64.json` | Runtime summary for the real BERT ISL skewed-layout run. |
| `hardware_results/server_verification/real_bert_layout7_tk64.json` | Runtime summary for the real BERT ISL blocked-affine run. |
| `hardware_results/server_verification/ncu_real_bert_layout0_tk64_live_raw.csv` | Raw `ncu` CSV for the real BERT naive-layout run. |
| `hardware_results/server_verification/ncu_real_bert_layout2_tk64_live_raw.csv` | Raw `ncu` CSV for the real BERT F2-layout run. |
| `hardware_results/server_verification/ncu_real_bert_layout3_tk64_raw.csv` | Raw `ncu` CSV for the real BERT HexCute-layout run. |
| `hardware_results/server_verification/ncu_real_bert_layout5_tk64_raw.csv` | Raw `ncu` CSV for the real BERT ISL skewed-layout run. |
| `hardware_results/server_verification/ncu_real_bert_layout7_tk64_raw.csv` | Raw `ncu` CSV for the real BERT ISL blocked-affine run. |

## How To Rebuild The Triton Pass

1. Clone Triton separately outside this repository:
   `git clone --depth 1 --branch release/2.3.x https://github.com/triton-lang/triton.git`
2. Apply the patch:
   `git apply /path/to/ACE-Project/triton_pass/triton_release_2_3_x_f2_pass.patch`
3. Point the Triton build to a matching LLVM/MLIR toolchain.
4. Configure and build `triton-opt` and `triton-translate`.
5. Confirm the pass is registered:
   `triton-opt --help | grep tritongpu-f2-shared-layout`

The pass is registered through TableGen, not through a Python post-processor. The relevant snapshots are already included in `triton_pass/`.

## How To Run The TTGIR Pass Experiment

From a patched Triton build:

1. Start with `ttgir_inputs/matmul_kernel.ttgir.mlir`.
2. Run:
   `triton-opt --tritongpu-f2-shared-layout ttgir_inputs/matmul_kernel.ttgir.mlir -o ttgir_outputs/matmul_kernel_f2_from_pass.ttgir.mlir`
3. Translate both TTGIR files to PTX with `triton-translate`.
4. Build `ptx_matmul_runner.cpp`.
5. Launch the original and pass-output PTX through the runner.
6. Run `ncu` on the runner to collect the shared-memory conflict counters.

This is the exact pass experiment used in the report for the direct input-vs-output TTGIR/PTX comparison.

## How To Run The Whole-BERT Custom-Layout Experiment

1. Install PyTorch with CUDA and `transformers`.
2. Run `bert_real_model_with_custom_layouts.py` from the repository root.
3. Choose the layout id:
   - `0` = naive
   - `2` = F2
   - `3` = HexCute
   - `5` = ISL skewed
   - `7` = ISL blocked affine
4. Wrap the script with `ncu` to collect bank-conflict counters.

Example shape used in the report:

- model = `bert`
- `tile_k = 64`
- `seq_len = 128`
- `batch = 1`

## How To Regenerate The Report Figures

From the repository root:

```bash
python3 generate_plots.py
```

This regenerates every figure referenced by `report.tex`, including:

- pass-driven GEMM bank-conflict plot
- pass-driven GEMM timing plot
- isolated GEMM plot
- stress-comparison plot
- stress heatmap
- ISL-win plot
- analytical verification plot
- C++ BERT runner plot
- optimized-layout real-BERT plot
- five-layout real-BERT total-conflict plot
- five-layout real-BERT breakdown plot

## How To Compile The Report

1. Upload the entire `ACE-Project` folder to Overleaf, or copy its contents into an Overleaf project.
2. Ensure the `figures/` directory is uploaded along with `report.tex`.
3. Compile `report.tex`.

The report is already written to use the packaged PNG figures directly, so no path changes are needed if the directory structure is preserved.