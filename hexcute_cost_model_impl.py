import math
import json
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HardwareConfig:
    """Hardware parameters for cost modeling."""
    sm_count: int = 24
    clock_ghz: float = 1.59
    memory_bandwidth_gbs: float = 288.0
    l2_cache_mb: float = 1.5
    smem_banks: int = 32
    warp_size: int = 32
    vec_length: int = 8
    tensor_core_tflops: float = 5.0

@dataclass
class KernelCharacteristics:
    """Kernel-specific pattern characteristics."""
    name: str
    flops: float
    bytes_accessed: float
    bank_conflict_factor: float
    cache_miss_factor: float
    coalescing_factor: float
    reuse_factor: float

class HexCuteCostModel:
    """HexCute-inspired static cost model."""

    def __init__(self, hw_config: HardwareConfig):
        self.hw = hw_config
        self.swizzle_params = self._synthesize_swizzle()

    def _synthesize_swizzle(self) -> Dict[str, int]:
        """HexCute swizzle synthesis from constraints."""

        S = int(math.log2(self.hw.vec_length))
        M = S
        B = int(math.log2(self.hw.smem_banks // self.hw.vec_length))
        B = min(B, 3)
        return {"B": B, "M": M, "S": S}

    def estimate_compute_cycles(self, kernel: KernelCharacteristics) -> float:
        """Estimate compute cycles based on FLOPs and tensor throughput."""

        effective_tflops = self.hw.tensor_core_tflops * kernel.reuse_factor
        if effective_tflops <= 0:
            effective_tflops = 0.1

        compute_time_s = kernel.flops / (effective_tflops * 1e12)
        compute_cycles = compute_time_s * self.hw.clock_ghz * 1e9
        return compute_cycles

    def estimate_memory_cycles(self, kernel: KernelCharacteristics) -> float:
        """Estimate memory cycles including coalescing penalties."""
        effective_bw = self.hw.memory_bandwidth_gbs * kernel.coalescing_factor
        if effective_bw <= 0:
            effective_bw = 1.0

        memory_time_s = kernel.bytes_accessed / (effective_bw * 1e9)
        memory_cycles = memory_time_s * self.hw.clock_ghz * 1e9
        return memory_cycles

    def estimate_bank_conflict_penalty(self, kernel: KernelCharacteristics) -> float:
        """Estimate SMEM bank conflict penalty."""

        swizzle_effectiveness = 1.0 / (2 ** self.swizzle_params["B"])
        conflict_penalty = (kernel.bank_conflict_factor - 1.0) * 1000
        mitigated_penalty = conflict_penalty * swizzle_effectiveness
        return max(0.0, mitigated_penalty)

    def estimate_cache_miss_penalty(self, kernel: KernelCharacteristics) -> float:
        """Estimate L2 cache miss penalty."""
        miss_penalty = (kernel.cache_miss_factor - 1.0) * 2000
        return max(0.0, miss_penalty)

    def estimate_total_cost(self, kernel: KernelCharacteristics) -> Dict[str, float]:
        """Full HexCute cost decomposition."""
        compute_cycles = self.estimate_compute_cycles(kernel)
        memory_cycles = self.estimate_memory_cycles(kernel)
        bank_penalty = self.estimate_bank_conflict_penalty(kernel)
        cache_penalty = self.estimate_cache_miss_penalty(kernel)


        base_cycles = max(compute_cycles, memory_cycles)
        total_cycles = base_cycles + bank_penalty + cache_penalty


        predicted_time_ms = total_cycles / (self.hw.clock_ghz * 1e6)
        predicted_tflops = kernel.flops / ((predicted_time_ms / 1000) * 1e12)
        predicted_bw_gbs = kernel.bytes_accessed / ((predicted_time_ms / 1000) * 1e9)

        return {
            "compute_cycles": compute_cycles,
            "memory_cycles": memory_cycles,
            "bank_conflict_penalty": bank_penalty,
            "cache_miss_penalty": cache_penalty,
            "base_cycles": base_cycles,
            "total_cycles": total_cycles,
            "predicted_time_ms": predicted_time_ms,
            "predicted_tflops": predicted_tflops,
            "predicted_bw_gbs": predicted_bw_gbs,
        }


def build_kernel_characteristics(matrix_size=1024) -> Dict[str, KernelCharacteristics]:
    """Define characteristics for each matrix pattern."""
    flops = 2 * matrix_size * matrix_size * matrix_size
    bytes_accessed = (matrix_size * matrix_size * 3) * 2

    return {
        "Baseline": KernelCharacteristics(
            name="Baseline",
            flops=flops,
            bytes_accessed=bytes_accessed,
            bank_conflict_factor=1.0,
            cache_miss_factor=1.0,
            coalescing_factor=1.0,
            reuse_factor=1.0,
        ),
        "Rotation": KernelCharacteristics(
            name="Rotation",
            flops=flops,
            bytes_accessed=bytes_accessed,
            bank_conflict_factor=1.15,
            cache_miss_factor=1.10,
            coalescing_factor=0.92,
            reuse_factor=0.95,
        ),
        "Butterfly": KernelCharacteristics(
            name="Butterfly",
            flops=flops,
            bytes_accessed=bytes_accessed,
            bank_conflict_factor=1.25,
            cache_miss_factor=1.18,
            coalescing_factor=0.88,
            reuse_factor=0.93,
        ),
        "Kronecker": KernelCharacteristics(
            name="Kronecker",
            flops=flops,
            bytes_accessed=bytes_accessed,
            bank_conflict_factor=1.10,
            cache_miss_factor=1.08,
            coalescing_factor=0.95,
            reuse_factor=1.05,
        ),
        "Toeplitz": KernelCharacteristics(
            name="Toeplitz",
            flops=flops,
            bytes_accessed=bytes_accessed,
            bank_conflict_factor=1.20,
            cache_miss_factor=1.22,
            coalescing_factor=0.85,
            reuse_factor=0.90,
        ),
    }


def apply_isl_optimizations(kernels: Dict[str, KernelCharacteristics]) -> Dict[str, KernelCharacteristics]:
    """Apply estimated improvements from ISL schedules."""
    optimized = {}

    for name, k in kernels.items():
        if name == "Baseline":
            optimized[name] = k
            continue


        if name == "Rotation":

            optimized[name] = KernelCharacteristics(
                name=f"{name}_ISL",
                flops=k.flops,
                bytes_accessed=k.bytes_accessed,
                bank_conflict_factor=k.bank_conflict_factor * 0.85,
                cache_miss_factor=k.cache_miss_factor * 0.82,
                coalescing_factor=min(1.0, k.coalescing_factor * 1.12),
                reuse_factor=k.reuse_factor * 1.15,
            )
        elif name == "Butterfly":

            optimized[name] = KernelCharacteristics(
                name=f"{name}_ISL",
                flops=k.flops,
                bytes_accessed=k.bytes_accessed,
                bank_conflict_factor=k.bank_conflict_factor * 0.78,
                cache_miss_factor=k.cache_miss_factor * 0.75,
                coalescing_factor=min(1.0, k.coalescing_factor * 1.18),
                reuse_factor=k.reuse_factor * 1.25,
            )
        elif name == "Kronecker":

            optimized[name] = KernelCharacteristics(
                name=f"{name}_ISL",
                flops=k.flops,
                bytes_accessed=k.bytes_accessed,
                bank_conflict_factor=k.bank_conflict_factor * 0.90,
                cache_miss_factor=k.cache_miss_factor * 0.85,
                coalescing_factor=min(1.0, k.coalescing_factor * 1.08),
                reuse_factor=k.reuse_factor * 1.10,
            )
        elif name == "Toeplitz":

            optimized[name] = KernelCharacteristics(
                name=f"{name}_ISL",
                flops=k.flops,
                bytes_accessed=k.bytes_accessed,
                bank_conflict_factor=k.bank_conflict_factor * 0.80,
                cache_miss_factor=k.cache_miss_factor * 0.78,
                coalescing_factor=min(1.0, k.coalescing_factor * 1.20),
                reuse_factor=k.reuse_factor * 1.18,
            )

    return optimized


def run_cost_model_evaluation():
    """Run full cost model for baseline and ISL-optimized kernels."""
    hw = HardwareConfig()
    model = HexCuteCostModel(hw)
    kernels = build_kernel_characteristics(matrix_size=1024)
    kernels_isl = apply_isl_optimizations(kernels)

    results = {
        "hardware": {
            "sm_count": hw.sm_count,
            "clock_ghz": hw.clock_ghz,
            "memory_bandwidth_gbs": hw.memory_bandwidth_gbs,
            "smem_banks": hw.smem_banks,
            "swizzle": model.swizzle_params,
        },
        "baseline": {},
        "isl_optimized": {},
        "improvements": {},
    }

    print("=" * 80)
    print("HEXCUTE COST MODEL EVALUATION")
    print("=" * 80)
    print(f"Swizzle params (B,M,S): ({model.swizzle_params['B']},{model.swizzle_params['M']},{model.swizzle_params['S']})")
    print()


    print("Baseline kernel cost estimates:")
    for name, kernel in kernels.items():
        cost = model.estimate_total_cost(kernel)
        results["baseline"][name] = cost
        print(f"  {name:10s}: {cost['predicted_time_ms']:.4f} ms, {cost['predicted_tflops']:.2f} TFLOPS")
    print()


    print("ISL-optimized kernel cost estimates:")
    for name, kernel in kernels_isl.items():
        if name == "Baseline":
            continue
        cost = model.estimate_total_cost(kernel)
        results["isl_optimized"][name] = cost
        print(f"  {name:10s}: {cost['predicted_time_ms']:.4f} ms, {cost['predicted_tflops']:.2f} TFLOPS")
    print()


    print("Estimated improvements from ISL schedules:")
    for name in ["Rotation", "Butterfly", "Kronecker", "Toeplitz"]:
        base_t = results["baseline"][name]["predicted_time_ms"]
        isl_t = results["isl_optimized"][name]["predicted_time_ms"]
        improvement = (base_t - isl_t) / base_t * 100

        base_tf = results["baseline"][name]["predicted_tflops"]
        isl_tf = results["isl_optimized"][name]["predicted_tflops"]
        tf_gain = (isl_tf - base_tf) / base_tf * 100

        results["improvements"][name] = {
            "time_reduction_percent": improvement,
            "tflops_gain_percent": tf_gain,
            "speedup": base_t / isl_t,
        }
        print(f"  {name:10s}: {improvement:6.2f}% time reduction, {tf_gain:6.2f}% TFLOPS gain ({base_t/isl_t:.2f}x)")

    return results


if __name__ == "__main__":
    results = run_cost_model_evaluation()

    with open("hexcute_cost_model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved to hexcute_cost_model_results.json")
