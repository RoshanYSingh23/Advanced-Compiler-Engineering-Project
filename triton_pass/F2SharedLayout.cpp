#include "llvm/ADT/StringMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

namespace {

struct SwizzleChoice {
  int B = 0;
  int M = 0;
  int S = 0;
  int rank = 0;
  int conflicts = std::numeric_limits<int>::max();
  int perPhase = 1;
  int maxPhase = 1;
};

static int ceilLog2(int64_t value) {
  int bits = 0;
  int64_t x = std::max<int64_t>(value, 1);
  x -= 1;
  while (x > 0) {
    ++bits;
    x >>= 1;
  }
  return std::max(bits, 1);
}

static int applySwizzle(int row, int col, const SwizzleChoice &choice) {
  if (choice.B == 0)
    return col;
  int mask = (1 << choice.B) - 1;
  int swizzleBits = (row >> choice.M) & mask;
  return col ^ (swizzleBits << choice.S);
}

static int f2Rank(std::vector<uint64_t> rows, int cols) {
  int rank = 0;
  for (int col = 0; col < cols; ++col) {
    int pivot = -1;
    for (int row = rank; row < static_cast<int>(rows.size()); ++row) {
      if ((rows[row] >> col) & 1ULL) {
        pivot = row;
        break;
      }
    }
    if (pivot == -1)
      continue;
    std::swap(rows[rank], rows[pivot]);
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
      if (row != rank && ((rows[row] >> col) & 1ULL))
        rows[row] ^= rows[rank];
    }
    ++rank;
  }
  return rank;
}

static int countBankConflicts(int tileRows, int tileCols,
                              const SwizzleChoice &choice) {
  auto offset = [&](int row, int col) {
    return row * tileCols + applySwizzle(row, col, choice);
  };

  int total = 0;
  for (int row = 0; row < std::min(tileRows, 8); ++row) {
    std::map<int, int> bankCounts;
    for (int tid = 0; tid < std::min(tileCols, 32); ++tid)
      bankCounts[offset(row, tid) % 32]++;
    for (const auto &entry : bankCounts)
      total += std::max(0, entry.second - 1);
  }
  for (int col = 0; col < std::min(tileCols, 8); ++col) {
    std::map<int, int> bankCounts;
    for (int tid = 0; tid < std::min(tileRows, 32); ++tid)
      bankCounts[offset(tid, col) % 32]++;
    for (const auto &entry : bankCounts)
      total += std::max(0, entry.second - 1);
  }
  return total;
}

static int computeBankRank(int tileRows, int tileCols, const SwizzleChoice &choice) {
  int rowBits = ceilLog2(tileRows);
  int colBits = ceilLog2(tileCols);
  int inputBits = rowBits + colBits;
  std::vector<uint64_t> bankRows(5, 0);

  for (int bitIdx = 0; bitIdx < inputBits; ++bitIdx) {
    int row = 0;
    int col = 0;
    if (bitIdx < colBits)
      col = 1 << bitIdx;
    else
      row = 1 << (bitIdx - colBits);
    int offset = row * tileCols + applySwizzle(row, col, choice);
    for (int bankBit = 0; bankBit < 5; ++bankBit) {
      if ((offset >> bankBit) & 1)
        bankRows[bankBit] |= (1ULL << bitIdx);
    }
  }
  return f2Rank(bankRows, inputBits);
}

static SwizzleChoice findBestRepresentableF2(int tileRows, int tileCols,
                                             int maxSwizzleBits) {
  SwizzleChoice best;
  bool initialized = false;
  for (int B = 0; B <= maxSwizzleBits; ++B) {
    for (int M = 0; M <= maxSwizzleBits; ++M) {
      SwizzleChoice current;
      current.B = B;
      current.M = M;
      current.S = 0;
      current.rank = computeBankRank(tileRows, tileCols, current);
      current.conflicts = countBankConflicts(tileRows, tileCols, current);
      current.perPhase = std::max(1, 1 << M);
      current.maxPhase = std::max(1, 1 << B);

      if (!initialized || current.rank > best.rank ||
          (current.rank == best.rank && current.conflicts < best.conflicts) ||
          (current.rank == best.rank && current.conflicts == best.conflicts &&
           current.maxPhase > best.maxPhase)) {
        best = current;
        initialized = true;
      }
    }
  }
  return best;
}

static SmallVector<int64_t> getLogicalSharedShape(RankedTensorType tensorType) {
  auto shared =
      tensorType.getEncoding().dyn_cast<mlir::triton::gpu::SharedEncodingAttr>();
  if (!shared)
    return {};

  auto tensorShape = tensorType.getShape();
  unsigned encodedRank = shared.getOrder().size();
  if (tensorShape.size() < encodedRank || encodedRank == 0)
    return {};

  SmallVector<int64_t> logicalShape;
  unsigned begin = tensorShape.size() - encodedRank;
  for (unsigned i = begin; i < tensorShape.size(); ++i)
    logicalShape.push_back(tensorShape[i]);
  return logicalShape;
}

static std::string makeEncodingKey(Attribute encoding) {
  std::string key;
  llvm::raw_string_ostream os(key);
  encoding.print(os);
  return os.str();
}

static int64_t logicalShapeArea(ArrayRef<int64_t> logicalShape) {
  if (logicalShape.empty())
    return 0;
  int64_t area = 1;
  for (int64_t dim : logicalShape)
    area *= std::max<int64_t>(dim, 1);
  return area;
}

class TritonGPUF2SharedLayoutPass
    : public TritonGPUF2SharedLayoutBase<TritonGPUF2SharedLayoutPass> {
public:
  TritonGPUF2SharedLayoutPass() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    llvm::StringMap<Attribute> rewrittenEncodings;
    llvm::StringMap<SwizzleChoice> chosenLayouts;
    llvm::StringMap<Attribute> originalEncodings;
    llvm::StringMap<SmallVector<int64_t>> representativeShapes;

    auto recordTensorType = [&](Type type) {
      auto tensorType = type.dyn_cast<RankedTensorType>();
      if (!tensorType)
        return;
      auto shared =
          tensorType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      if (!shared)
        return;

      auto logicalShape = getLogicalSharedShape(tensorType);
      if (logicalShape.size() != 2)
        return;

      std::string key = makeEncodingKey(shared);
      originalEncodings[key] = shared;
      auto existing = representativeShapes.find(key);
      if (existing == representativeShapes.end() ||
          logicalShapeArea(logicalShape) >
              logicalShapeArea(existing->getValue()))
        representativeShapes[key] = logicalShape;
    };

    module.walk([&](triton::FuncOp func) {
      auto funcType = func.getFunctionType();
      for (Type inputType : funcType.getInputs())
        recordTensorType(inputType);
      for (Type resultType : funcType.getResults())
        recordTensorType(resultType);
    });

    module.walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments())
            recordTensorType(arg.getType());
        }
      }
      for (Value result : op->getResults())
        recordTensorType(result.getType());
    });

    for (const auto &entry : representativeShapes) {
      auto originalEncoding =
          originalEncodings.lookup(entry.getKey())
              .cast<triton::gpu::SharedEncodingAttr>();
      auto logicalShape = entry.getValue();
      auto choice = findBestRepresentableF2(static_cast<int>(logicalShape[0]),
                                            static_cast<int>(logicalShape[1]),
                                            maxSwizzleBits);
      SmallVector<unsigned> order(originalEncoding.getOrder().begin(),
                                  originalEncoding.getOrder().end());
      auto newEncoding = triton::gpu::SharedEncodingAttr::get(
          ctx, originalEncoding.getVec(), choice.perPhase, choice.maxPhase,
          order, originalEncoding.getCTALayout(),
          originalEncoding.getHasLeadingOffset());
      rewrittenEncodings[entry.getKey()] = newEncoding;
      chosenLayouts[entry.getKey()] = choice;
    }

    auto rewriteTensorType = [&](Type type) -> Type {
      auto tensorType = type.dyn_cast<RankedTensorType>();
      if (!tensorType)
        return type;
      auto shared =
          tensorType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      if (!shared)
        return type;

      auto logicalShape = getLogicalSharedShape(tensorType);
      if (logicalShape.size() != 2)
        return type;

      std::string key = makeEncodingKey(shared);
      auto cached = rewrittenEncodings.find(key);
      if (cached == rewrittenEncodings.end())
        return type;

      return RankedTensorType::get(tensorType.getShape(),
                                   tensorType.getElementType(),
                                   cached->second);
    };

    module.walk([&](triton::FuncOp func) {
      auto funcType = func.getFunctionType();
      SmallVector<Type> newInputs;
      SmallVector<Type> newResults;
      bool changed = false;

      newInputs.reserve(funcType.getNumInputs());
      for (Type inputType : funcType.getInputs()) {
        Type rewritten = rewriteTensorType(inputType);
        changed |= rewritten != inputType;
        newInputs.push_back(rewritten);
      }

      newResults.reserve(funcType.getNumResults());
      for (Type resultType : funcType.getResults()) {
        Type rewritten = rewriteTensorType(resultType);
        changed |= rewritten != resultType;
        newResults.push_back(rewritten);
      }

      if (changed)
        func.setType(FunctionType::get(ctx, newInputs, newResults));
    });

    module.walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            Type newType = rewriteTensorType(arg.getType());
            if (newType != arg.getType())
              arg.setType(newType);
          }
        }
      }

      for (Value result : op->getResults()) {
        Type newType = rewriteTensorType(result.getType());
        if (newType != result.getType())
          result.setType(newType);
      }
    });

    if (emitRemarks) {
      module.walk([&](triton::FuncOp func) {
        for (const auto &entry : chosenLayouts) {
          const auto &choice = entry.getValue();
          auto shape = representativeShapes.lookup(entry.getKey());
          func.emitRemark()
              << "F2 shared layout rewrite " << entry.getKey()
              << " representativeShape=[" << shape[0] << "x" << shape[1]
              << "]"
              << " -> Swizzle<" << choice.B << "," << choice.M << ","
              << choice.S << "> with perPhase=" << choice.perPhase
              << ", maxPhase=" << choice.maxPhase << ", rank=" << choice.rank
              << ", simulatedConflicts=" << choice.conflicts;
        }
      });
    }

    if (failed(verify(module)))
      signalPassFailure();
  }
};

}

std::unique_ptr<Pass> mlir::createTritonGPUF2SharedLayoutPass() {
  return std::make_unique<TritonGPUF2SharedLayoutPass>();
}
