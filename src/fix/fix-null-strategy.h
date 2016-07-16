#ifndef KALDI_FIX_NULL_STRATEGY_H_
#define KALDI_FIX_NULL_STRATEGY_H_

#include "fix/fix-strategy.h"

namespace kaldi {
  namespace fix {
    class NullStrategy : public FixStrategy {
    public:
      virtual StrategyType GetType() const { return kNullStrategy; }
    protected:
      virtual void ReadData(std::istream &is, bool binary) {}
      virtual void WriteData(std::ostream &os, bool binary) const {}
      virtual void DoFixBlob(CuMatrixBase<BaseFloat> &blob, int n) {}
      virtual void DoFixBlob(MatrixBase<BaseFloat> &blob, int n) {}
      virtual void DoFixParam(VectorBase<BaseFloat> &blob,
                              Component::ComponentType comp_type,
                              int n) {}
    };
  }
}
#endif
