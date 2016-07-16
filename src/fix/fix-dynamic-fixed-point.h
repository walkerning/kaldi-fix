#ifndef KALDI_FIX_DYNAMIC_FIXED_POINT_H_
#define KALDI_FIX_DYNAMIC_FIXED_POINT_H_

#include <tr1/unordered_map>
#include <limits>
#include <algorithm>

#include "fix/fix-strategy.h"
#include "fix/fix-kernels-ansi.h"

namespace kaldi {
  namespace fix {
    using namespace kaldi::nnet1;
    typedef std::tr1::unordered_map<int, int> IndexIntMap;

    class DynamicFixedPointStrategy : public FixStrategy {

    public:
      StrategyType GetType() const { return kDynamicFixedPoint; }

      static const int DEFAULT_PARAM_BIT_NUM = 8;
      static const int DEFAULT_BLOB_BIT_NUM = 8;

      int ParamBitNum(int n, Component::ComponentType _type) const {
        IndexIntMap::const_iterator got_index;
        IndexIntMap::const_iterator got_type;
        if ((got_index = param_index_map_.find(n)) != param_index_map_.end()) {
          return got_index->second;
        }
        if ((got_type = param_type_map_.find(static_cast<int> (_type))) != param_type_map_.end()) {
          return got_type->second;
        }
        return DEFAULT_PARAM_BIT_NUM;
      }

      int BlobBitNum(int n) const {
        IndexIntMap::const_iterator got;
        if ((got = blob_index_map_.find(n)) != blob_index_map_.end()) {
          return got->second;
        }
        return DEFAULT_BLOB_BIT_NUM;
      }

      int BlobFragPos(int n, const CuMatrixBase<BaseFloat>& blob, int bit_num) {
        int frag_pos;
        IndexIntMap::const_iterator got;
        if ((got = blob_frag_pos_map_.find(n)) != blob_frag_pos_map_.end()) {
          frag_pos = got->second;
        } else {
          BaseFloat b_max = std::max(fabs(blob.Max()), fabs(blob.Min()));
          blob_frag_pos_map_[n] = frag_pos = bit_num - 1 - ceil(log(b_max) / log(2));
        }
        return frag_pos;
      }

      int BlobFragPos(int n, const MatrixBase<BaseFloat>& blob, int bit_num) {
        int frag_pos;
        IndexIntMap::const_iterator got;
        if ((got = blob_frag_pos_map_.find(n)) != blob_frag_pos_map_.end()) {
          frag_pos = got->second;
        } else {
          BaseFloat b_max = std::max(fabs(blob.Max()), fabs(blob.Min()));
          blob_frag_pos_map_[n] = frag_pos = bit_num - 1 - ceil(log(b_max) / log(2));
        }
        return frag_pos;
      }

      int ParamFragPos(int n, const VectorBase<BaseFloat>& blob, int bit_num) {
        int frag_pos;
        const BaseFloat* data = blob.Data();
        int dim = blob.Dim();

        IndexIntMap::const_iterator got;
        if ((got = param_frag_pos_map_.find(n)) != param_frag_pos_map_.end()) {
          frag_pos = got->second;
        } else {
          // FIXME: Or use VectorBase::Max/Min instead?
          /* BaseFloat max_num = blob.Max(); */
          /* BaseFloat min_num = blob.Min(); */

          BaseFloat max_num = *std::max_element(data, data + dim);
          BaseFloat min_num = *std::min_element(data, data + dim);

          BaseFloat b_max = std::max(fabs(max_num), fabs(min_num));
          param_frag_pos_map_[n] = frag_pos = bit_num - 1 - ceil(log(b_max) / log(2));
        }
        return frag_pos;
      }
     
      static BaseFloat Float2Fix(BaseFloat f, int bit_num, int frag_pos) {
        int bitvalid = bit_num - 1;
        int maxnum = ((1) << bitvalid) - 1;
        int minnum = -(1 << bitvalid);
        BaseFloat result = 0;
        // FIXME: 这个策略实现不确定对不对
        if (frag_pos >= 0) {
          result = f * (1 << frag_pos);
        } else {
          result = f / (1 << -frag_pos);
        }

        if (result > maxnum) {
          result = maxnum;
        } else if (result < minnum) {
          result = minnum;
        }
       
        result = BaseFloat(int(result));
        if (frag_pos >= 0) {
          result = result / (1 << frag_pos);
        } else {
          result = result * (1 << -frag_pos);
        }

        return result;
      }

      virtual void Clear() {
        param_index_map_.clear();
        blob_index_map_.clear();
        param_frag_pos_map_.clear();
        blob_frag_pos_map_.clear();
      }

    protected:
      virtual void ReadData(std::istream &is, bool binary) {
        while ('<' == Peek(is, binary)) {
          std::string token;
          int raw_type;
          int index;

          int first_char = PeekToken(is, binary);
          switch (first_char) {
          case 'B': ExpectToken(is, binary, "<BlobIndexBit>");
            ReadBasicType(is, binary, &index);
            ReadBasicType(is, binary, &blob_index_map_[index]);
            break;
          case 'P': ReadToken(is, false, &token);
            if (token == "<ParamTypeBit>") {
              ReadBasicType(is, binary, &raw_type);
              ReadBasicType(is, binary, &param_type_map_[raw_type]);
            } else if (token == "<ParamIndexBit>") {
              ReadBasicType(is, binary, &index);
              ReadBasicType(is, binary, &param_index_map_[index]);
            }
            break;
          default: ReadToken(is, false, &token);
            KALDI_ERR << "Unknown token: " << token;
          }
        }
      }
     
      virtual void WriteData(std::ostream &os, bool binary) const {
        for( IndexIntMap::const_iterator item = blob_index_map_.begin(); item != blob_index_map_.end(); ++item ) { 
          WriteToken(os, binary, "<BlobIndexBit>");
          WriteBasicType(os, binary, item->first);
          WriteBasicType(os, binary, item->second);
        }
        for( IndexIntMap::const_iterator item = param_index_map_.begin(); item != param_index_map_.end(); ++item ) {
          WriteToken(os, binary, "<ParamIndexBit>");
          WriteBasicType(os, binary, item->first);
          WriteBasicType(os, binary, item->second);
        }
        for( IndexIntMap::const_iterator item = param_type_map_.begin(); item != param_type_map_.end(); ++item ) {
          WriteToken(os, binary, "<ParamTypeBit>");
          WriteBasicType(os, binary, item->first);
          WriteBasicType(os, binary, item->second);
        }

        if (!binary) os << "\n";
      }

      virtual void DoFixBlob(CuMatrixBase<BaseFloat> &blob, int n) {
#if HAVE_CUDA == 1
                          // handle data on GPU
                          int bit_num = BlobBitNum(n);
        int frag_pos = BlobFragPos(n, blob, bit_num);

        // convert float to fix
        BaseFloat multiplier;
        if (frag_pos >= 0) {
          multiplier = (1 << frag_pos);
        } else {
          multiplier = 1. / (1 << -frag_pos);
        }

        blob.Scale(multiplier);

        int bitvalid = bit_num - 1;
        float maxnum = (1 << bitvalid) - 1;
        float minnum = -(1 << bitvalid);
        dim3 dimGrid, dimBlock;
        GetBlockSizesForSimpleMatrixOperation(blob.NumRows(), blob.NumCols(),
                                              &dimGrid, &dimBlock);
        cuda_saturate(dimGrid, dimBlock, blob.NumRows() * blob.NumCols(),
                      blob.Data(), maxnum, minnum);

        blob.Scale(1. / multiplier);
#else
          // Copy to CPU and handled
          Matrix<BaseFloat> blob_cpu = Matrix<BaseFloat>(blob);
        DoFixBlob(blob_cpu, n);
#endif
          }

      virtual void DoFixBlob(MatrixBase<BaseFloat> &blob, int n) {
        int bit_num = BlobBitNum(n);
        int frag_pos = BlobFragPos(n, blob, bit_num);

        BaseFloat* data = blob.Data();
        MatrixIndexT stride = blob.Stride();
       
        // float to fix
        for (MatrixIndexT i = 0; i < blob.NumRows(); i++) {
          for (MatrixIndexT j = 0; j < blob.NumCols(); j++) {
            data[j] = Float2Fix(data[j], bit_num, frag_pos);
            data += stride;
          }
        }
      }

      virtual void DoFixParam(VectorBase<BaseFloat> &blob,
                              Component::ComponentType comp_type,
                              int n) {
        int bit_num = ParamBitNum(n, comp_type);
        int frag_pos = ParamFragPos(n, blob, bit_num);

        BaseFloat* data = blob.Data();
        // float to fix
        for (MatrixIndexT i = 0; i < blob.Dim(); i++) {
          data[i] = Float2Fix(data[i], bit_num, frag_pos);
        }
      }

    private:
      IndexIntMap param_type_map_;
      IndexIntMap param_index_map_;

      IndexIntMap blob_index_map_;

      IndexIntMap param_frag_pos_map_;
      IndexIntMap blob_frag_pos_map_;
    }; // end class DynamicFixedPointStrategy
  } // end namespace fix
} // end namespace kaldi

#endif
