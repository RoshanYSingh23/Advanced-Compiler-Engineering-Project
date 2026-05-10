#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kM = 128;
constexpr int kN = 256;
constexpr int kK = 32;
constexpr int kBlockX = 128;
constexpr int kDynamicSmemBytes = (2 * 128 * 32 + 2 * 32 * 256) * 2;

void check(CUresult result, const char *what) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = nullptr;
  const char *desc = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &desc);
  std::ostringstream os;
  os << what << " failed";
  if (name)
    os << " [" << name << "]";
  if (desc)
    os << ": " << desc;
  throw std::runtime_error(os.str());
}

std::string readFile(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("failed to open " + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

}

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: " << argv[0] << " <kernel.ptx> [iters]\n";
      return 1;
    }

    const std::string ptxPath = argv[1];
    const int iters = argc == 3 ? std::stoi(argv[2]) : 100;
    if (iters <= 0)
      throw std::runtime_error("iters must be positive");

    check(cuInit(0), "cuInit");

    CUdevice device = 0;
    check(cuDeviceGet(&device, 0), "cuDeviceGet");

    char deviceName[256] = {};
    check(cuDeviceGetName(deviceName, sizeof(deviceName), device),
          "cuDeviceGetName");

    CUcontext context = nullptr;
    check(cuCtxCreate(&context, 0, device), "cuCtxCreate");

    std::string ptx = readFile(ptxPath);

    CUmodule module = nullptr;
    check(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr),
          "cuModuleLoadDataEx");

    CUfunction kernel = nullptr;
    check(cuModuleGetFunction(
              &kernel, module,
              "matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c"),
          "cuModuleGetFunction");

    check(cuFuncSetAttribute(kernel,
                             CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             kDynamicSmemBytes),
          "cuFuncSetAttribute");

    const size_t bytesA = static_cast<size_t>(kM) * kK * sizeof(std::uint16_t);
    const size_t bytesB = static_cast<size_t>(kK) * kN * sizeof(std::uint16_t);
    const size_t bytesC = static_cast<size_t>(kM) * kN * sizeof(float);

    CUdeviceptr dA = 0;
    CUdeviceptr dB = 0;
    CUdeviceptr dC = 0;
    check(cuMemAlloc(&dA, bytesA), "cuMemAlloc(A)");
    check(cuMemAlloc(&dB, bytesB), "cuMemAlloc(B)");
    check(cuMemAlloc(&dC, bytesC), "cuMemAlloc(C)");

    check(cuMemsetD8(dA, 0, bytesA), "cuMemsetD8(A)");
    check(cuMemsetD8(dB, 0, bytesB), "cuMemsetD8(B)");
    check(cuMemsetD8(dC, 0, bytesC), "cuMemsetD8(C)");

    unsigned int M = kM;
    unsigned int N = kN;
    unsigned int K = kK;
    unsigned int strideAm = kK;
    unsigned int strideBk = kN;
    unsigned int strideCm = kN;

    void *params[] = {&dA, &dB, &dC, &M, &N, &K, &strideAm, &strideBk,
                      &strideCm};

    check(cuCtxSynchronize(), "cuCtxSynchronize(prelaunch)");

    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
      check(cuLaunchKernel(kernel, 1, 1, 1, kBlockX, 1, 1, kDynamicSmemBytes,
                           nullptr, params, nullptr),
            "cuLaunchKernel");
    }
    check(cuCtxSynchronize(), "cuCtxSynchronize(postlaunch)");
    const auto end = std::chrono::steady_clock::now();

    const double totalMs =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::vector<float> hostC(static_cast<size_t>(kM) * kN, -1.0f);
    check(cuMemcpyDtoH(hostC.data(), dC, bytesC), "cuMemcpyDtoH");

    double checksum = 0.0;
    for (float value : hostC)
      checksum += value;

    std::cout << "device=" << deviceName << "\n";
    std::cout << "ptx=" << ptxPath << "\n";
    std::cout << "iters=" << iters << "\n";
    std::cout << "grid=1x1x1 block=" << kBlockX << "x1x1 smem="
              << kDynamicSmemBytes << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "total_ms=" << totalMs << "\n";
    std::cout << "per_iter_ms=" << (totalMs / static_cast<double>(iters))
              << "\n";
    std::cout << "checksum=" << checksum << "\n";

    cuMemFree(dC);
    cuMemFree(dB);
    cuMemFree(dA);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
