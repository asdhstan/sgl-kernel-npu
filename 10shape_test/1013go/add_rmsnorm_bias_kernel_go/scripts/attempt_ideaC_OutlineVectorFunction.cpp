//===--------------------- OutlineVectorFunction.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Transform/IR/TransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
#define GEN_PASS_DEF_OUTLINEVECTORFUNCTION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "outline-vector-function"

using namespace mlir;
using namespace mlir::hfusion;

namespace {
struct OutlineVectorFunctionPass
    : public impl::OutlineVectorFunctionBase<OutlineVectorFunctionPass> {
public:
  void runOnOperation() override;
};

static Value getOpTransformHandle(std::string label, OpBuilder &builder,
                                  transform::SequenceOp seqOp) {
  DictionaryAttr opAttr = builder.getDictionaryAttr(
      builder.getNamedAttr(label, builder.getUnitAttr()));
  Value opHandle =
      builder
          .create<transform::MatchOp>(
              seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
              seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
              transform::MatchInterfaceEnumAttr{}, opAttr, DictionaryAttr{},
              TypeAttr{}, ArrayAttr{})
          .getResults();
  return opHandle;
}

static bool loopHasOutlinedLoopAttr(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().starts_with("outlined-loop-target-")) {
      return true;
    }
  }
  return false;
}

/// `isFill` marks a vector function outlined around a fill, so that a later
/// pass can look for a constant to forward to its readers without having to
/// hunt for fills by inspecting every function body. Whether the fill really
/// does stamp a compile-time constant is for that pass to establish.
static void outlineLoopsIntoVF(SmallVector<Value> &loopHandles, Location loc,
                               OpBuilder &builder, std::string vfName,
                               bool isFill = false) {
  if (loopHandles.size() == 0)
    return;
  auto outlineOp = builder.create<transform::ExtendedLoopOutlineOp>(
      loc,
      SmallVector<Type>{builder.getType<transform::AnyOpType>(),
                        builder.getType<transform::AnyOpType>()},
      loopHandles, vfName);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getFunction(),
                                        mlir::hivm::VectorFunctionAttr::name,
                                        nullptr);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getFunction(),
                                        "no_inline", nullptr);
  if (isFill)
    builder.create<transform::AnnotateOp>(loc, outlineOp.getFunction(),
                                          kFillVFAttrName, nullptr);
  builder.create<transform::AnnotateOp>(
      loc, outlineOp.getCall(), mlir::hivm::VectorFunctionAttr::name, nullptr);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getCall(), "no_inline",
                                        nullptr);
  loopHandles.clear();
}

static void removeLabelsForOutlinedLoops(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().contains("outlined")) {
      op->removeAttr(attr.getName().strref());
    }
  }
}

static void duplicateReusedValuesForInnerSCFForOp(scf::ForOp loop,
                                                  OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  DominanceInfo domInfo;
  auto loc = loop.getLoc();
  DenseSet<Value> set;
  loop.getBody()->walk([&](scf::ForOp innerLoop) {
    for (OpOperand &iterArg : innerLoop.getInitArgsMutable()) {
      Value src = iterArg.get();
      if (!src.getDefiningOp())
        continue;
      auto srcExtractOp = dyn_cast<tensor::ExtractSliceOp>(src.getDefiningOp());
      if (srcExtractOp)
        src = srcExtractOp.getSource();
      if (!src.getDefiningOp() || !domInfo.properlyDominates(src, loop))
        continue;
      if (!set.contains(src)) {
        auto ty = dyn_cast<RankedTensorType>(src.getType());
        if (!ty || !ty.hasStaticShape())
          continue;
        set.insert(src);
      } else {
        builder.setInsertionPoint(loop);
        auto ty = dyn_cast<RankedTensorType>(src.getType());
        Value duplicated = builder.create<tensor::EmptyOp>(loc, ty.getShape(),
                                                           ty.getElementType());
        if (!isEmptyLikeTensor(src)) {
          duplicated =
              builder.create<linalg::CopyOp>(loc, src, duplicated).getResult(0);
        }
        if (srcExtractOp) {
          builder.setInsertionPoint(srcExtractOp);
          auto newSrcExtractOp = builder.create<tensor::ExtractSliceOp>(
              loc, srcExtractOp.getType(), duplicated,
              srcExtractOp.getMixedOffsets(), srcExtractOp.getMixedSizes(),
              srcExtractOp.getMixedStrides());
          iterArg.assign(newSrcExtractOp.getResult());
        } else {
          iterArg.assign(duplicated);
        }
      }
    }
  });
}

/// Returns the position of `value` among `op`'s operands, if any.  The operand
/// order of the vector transfer ops is not stable across MLIR versions, so the
/// rewrite locates the accumulator tensor instead of assuming an index.
static std::optional<unsigned> operandPosition(Operation *op, Value value) {
  for (auto [index, operand] : llvm::enumerate(op->getOperands()))
    if (operand == value)
      return static_cast<unsigned>(index);
  return std::nullopt;
}

/// Rewrites a reduction loop whose accumulator is a rank-0 tensor so that the
/// loop only ever performs an element-wise vector add, leaving a single
/// cross-lane reduction for after the loop.
///
/// `transform.structured.tile_reduction_using_for` splits the reduction into
/// vector-sized tiles and folds every tile's result into the output tensor.
/// Vectorizing that combine step yields, per tile,
///
///   %ri = vector.transfer_read %acc[]                   // UB round trip
///   %sc = vector.extractelement %ri[]
///   %mr = vector.multi_reduction <add>, %tile, %sc [0]  // full cross-lane
///   %bc = vector.broadcast %mr
///   %tw = vector.transfer_write %bc, %acc[]
///
/// A long reduction therefore pays a cross-lane reduction (and a UB round trip)
/// per vector.  The accumulator is kept in a rank-1 tensor because the later
/// AVE lowering does not accept a register-resident loop-carried vector; only
/// the cross-lane step is removed from the loop and hoisted behind it.
static void rewriteScalarAccumulatorReductions(func::FuncOp func) {
  SmallVector<scf::ForOp> candidates;
  func.walk([&](scf::ForOp forOp) {
    if (forOp.getInitArgs().size() == 1 && forOp.getNumResults() == 1)
      candidates.push_back(forOp);
  });

  IRRewriter rewriter(func.getContext());
  for (scf::ForOp forOp : candidates) {
    if (!forOp->getBlock())
      continue;
    Block *body = forOp.getBody();
    auto yieldOp = dyn_cast<scf::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      continue;

    Value iterArg = forOp.getRegionIterArg(0);
    auto iterType = dyn_cast<RankedTensorType>(iterArg.getType());
    if (!iterType || iterType.getNumElements() != 1 ||
        !iterType.getElementType().isF32())
      continue;

    auto writeOp = yieldOp.getOperand(0).getDefiningOp<vector::TransferWriteOp>();
    if (!writeOp)
      continue;
    std::optional<unsigned> writeIdx = operandPosition(writeOp, iterArg);
    if (!writeIdx)
      continue;
    auto broadcastOp =
        writeOp->getOperand(0).getDefiningOp<vector::BroadcastOp>();
    if (!broadcastOp)
      continue;
    auto reduceOp =
        broadcastOp->getOperand(0).getDefiningOp<vector::MultiDimReductionOp>();
    if (!reduceOp || reduceOp.getKind() != vector::CombiningKind::ADD)
      continue;
    auto extractOp =
        reduceOp->getOperand(1).getDefiningOp<vector::ExtractElementOp>();
    if (!extractOp)
      continue;
    auto accReadOp =
        extractOp->getOperand(0).getDefiningOp<vector::TransferReadOp>();
    if (!accReadOp)
      continue;
    std::optional<unsigned> readIdx = operandPosition(accReadOp, iterArg);
    if (!readIdx)
      continue;

    Value tile = reduceOp->getOperand(0);
    auto vecType = dyn_cast<VectorType>(tile.getType());
    if (!vecType || vecType.getRank() != 1 || !vecType.getElementType().isF32())
      continue;

    Operation *tileDef = tile.getDefiningOp();
    if (!tileDef || tileDef->getBlock() != body || tileDef == accReadOp ||
        tileDef == extractOp || tileDef == reduceOp.getOperation())
      continue;

    Location loc = forOp.getLoc();
    Value initTensor = forOp.getInitArgs()[0];
    Type elementType = vecType.getElementType();
    auto elementFloat = [&]() {
      return rewriter.getFloatAttr(elementType, 0.0);
    };

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp);

    // Values cloned out of the loop body must be re-created in front of their
    // user, otherwise they stop dominating.
    auto liftOperands = [&](Operation *cloned) {
      for (OpOperand &operand :
           llvm::make_early_inc_range(cloned->getOpOperands())) {
        Operation *def = operand.get().getDefiningOp();
        if (!def || def->getBlock() != body || def->getNumResults() != 1)
          continue;
        Operation *fresh = rewriter.clone(*def);
        fresh->moveBefore(cloned);
        operand.set(fresh->getResult(0));
      }
    };
    auto advance = [&](Operation *op) {
      rewriter.setInsertionPointAfter(op);
    };

    // (a) Materialize the incoming accumulator as a rank-1 tensor.
    auto initRead = cast<vector::TransferReadOp>(rewriter.clone(*accReadOp));
    initRead->setOperand(*readIdx, initTensor);
    liftOperands(initRead);
    advance(initRead);
    Value initScalar =
        rewriter.create<vector::ExtractElementOp>(loc, initRead.getResult());
    advance(initScalar.getDefiningOp());
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    advance(zeroIndex.getDefiningOp());
    Value zeroElement =
        rewriter.create<arith::ConstantOp>(loc, elementFloat());
    advance(zeroElement.getDefiningOp());
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, vecType.getShape(), elementType);
    advance(emptyTensor.getDefiningOp());
    Value initVectorTensor = rewriter.create<linalg::FillOp>(
        loc, ValueRange{initScalar}, ValueRange{emptyTensor})->getResult(0);
    advance(initVectorTensor.getDefiningOp());

    // (b) Same loop, but only an element-wise add per tile.
    auto newLoop = rewriter.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        ValueRange{initVectorTensor});
    {
      OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(newLoop.getBody());
      IRMapping mapping;
      // Without these the cloned ops would still reference the old loop's block
      // arguments, which then cannot be destroyed with the old loop.
      mapping.map(forOp.getInductionVar(), newLoop.getInductionVar());
      mapping.map(iterArg, newLoop.getRegionIterArg(0));
      for (Operation &nested : *body) {
        if (&nested == accReadOp.getOperation() ||
            &nested == extractOp.getOperation() ||
            &nested == reduceOp.getOperation() ||
            &nested == broadcastOp.getOperation() ||
            &nested == writeOp.getOperation() || &nested == yieldOp)
          continue;
        rewriter.clone(nested, mapping);
      }
      Value clonedTile = mapping.lookupOrNull(tile);
      if (!clonedTile)
        continue;  // checked above, defensive only
      Value current = newLoop.getRegionIterArg(0);
      Value readBack = rewriter.create<vector::TransferReadOp>(
          loc, vecType, current, ValueRange{zeroIndex}, zeroElement);
      Value sum = rewriter.create<arith::AddFOp>(loc, readBack, clonedTile);
      auto written = rewriter.create<vector::TransferWriteOp>(
          loc, sum, current, ValueRange{zeroIndex});
      // The scf::ForOp builder leaves the loop body unterminated.
      rewriter.create<scf::YieldOp>(loc, written.getResult());
    }

    // (c) One cross-lane reduction for the whole reduction, then store it.
    advance(newLoop);
    Value finalRead = rewriter.create<vector::TransferReadOp>(
        loc, vecType, newLoop.getResult(0), ValueRange{zeroIndex}, zeroElement);
    advance(finalRead.getDefiningOp());
    Value finalZero =
        rewriter.create<arith::ConstantOp>(loc, elementFloat());
    advance(finalZero.getDefiningOp());
    SmallVector<bool, 1> reductionMask(vecType.getRank(), true);
    auto finalReduce = rewriter.create<vector::MultiDimReductionOp>(
        loc, finalRead, finalZero, reductionMask, vector::CombiningKind::ADD);
    advance(finalReduce.getOperation());
    auto finalBroadcast = rewriter.create<vector::BroadcastOp>(
        loc, broadcastOp.getResult().getType(), finalReduce.getResult());
    advance(finalBroadcast.getOperation());
    auto finalWrite = cast<vector::TransferWriteOp>(rewriter.clone(*writeOp));
    finalWrite->setOperand(0, finalBroadcast.getResult());
    finalWrite->setOperand(*writeIdx, initTensor);
    liftOperands(finalWrite);

    rewriter.replaceAllUsesWith(forOp.getResult(0), finalWrite.getResult());
    rewriter.eraseOp(forOp);
  }
}

void OutlineVectorFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());
  module.walk([&](func::FuncOp func) {
    func.walk([&](scf::ForOp forOp) {
      if (!loopHasOutlinedLoopAttr(forOp))
        return;
      duplicateReusedValuesForInnerSCFForOp(forOp, builder);
    });
    std::string vfNamePrefix = func.getSymName().str() + "_outlined_vf_";
    int vfIdx = -1;
    builder.setInsertionPointAfter(func);
    transform::SequenceOp seqOp = builder.create<transform::SequenceOp>(
        func.getLoc(), TypeRange(),
        transform::FailurePropagationMode::Propagate,
        builder.getType<transform::AnyOpType>(),
        [](OpBuilder &b, Location nested, Value rootH) {
          b.create<transform::YieldOp>(nested, ValueRange());
        });
    auto loc = seqOp.getLoc();
    builder.setInsertionPointToStart(seqOp.getBodyBlock());
    func.walk([&](Block *block) {
      int loopIdxInVf = -1;
      Operation *prevLoop = nullptr;
      SmallVector<Value> loopHandles;
      for (Operation &operation : block->getOperations()) {
        Operation *op = &operation;
        if (op->hasAttr("outlinedLoopWithFill")) {
          assert(isa<scf::ForOp>(op));
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx));
          vfIdx++;
          std::string loopLabel = vfNamePrefix + std::to_string(vfIdx);
          op->setAttr(loopLabel, builder.getUnitAttr());
          loopHandles.push_back(
              getOpTransformHandle(loopLabel, builder, seqOp));
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx),
                             /*isFill=*/true);
          prevLoop = nullptr;
        } else if (loopHasOutlinedLoopAttr(op)) {
          assert(isa<scf::ForOp>(op));
          if (!prevLoop || op->getPrevNode() != prevLoop) {
            vfIdx++;
            loopIdxInVf = -1;
          }
          loopIdxInVf++;
          std::string loopLabel = vfNamePrefix + std::to_string(vfIdx) + "_" +
                                  std::to_string(loopIdxInVf);
          op->setAttr(loopLabel, builder.getUnitAttr());
          loopHandles.push_back(
              getOpTransformHandle(loopLabel, builder, seqOp));
          prevLoop = op;
        } else {
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx));
          prevLoop = nullptr;
        }
      }
    });
    transform::TransformOptions options;
    options.enableExpensiveChecks(false);
    LogicalResult result = transform::applyTransformNamedSequence(
        func, seqOp, func->getParentOfType<ModuleOp>(), options);
    seqOp->erase();
    if (failed(result))
      func->emitError(
          "Failed to outline vector function from loops of current function.");
  });

  module.walk([&](scf::ForOp forOp) { removeLabelsForOutlinedLoops(forOp); });

  // Reduction loops now live inside the outlined vector functions, so this is
  // the first point where the rewrite can add vector ops without tripping the
  // "vector operation outside vector function" verifier.
  module.walk([&](func::FuncOp func) {
    if (hivm::isVF(func))
      rewriteScalarAccumulatorReductions(func);
  });
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createOutlineVectorFunctionPass() {
  return std::make_unique<OutlineVectorFunctionPass>();
}
