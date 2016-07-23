#ifndef KALDI_FIX_DYNAMIC_FIXED_POINT_H_
#define KALDI_FIX_DYNAMIC_FIXED_POINT_H_

#include <tr1/unordered_map>
#include <limits>
#include <algorithm>
#include <cmath>

#include "fix/fix-strategy.h"
#include "fix/fix-kernels-ansi.h"
#include "fix/fix-nonlinear-kernels-ansi.h"

namespace kaldi {
  namespace fix {
    using namespace kaldi::nnet1;
    typedef std::tr1::unordered_map<int, int> IndexIntMap;

    class DynamicFixedPointStrategy : public FixStrategy {

    public:
    DynamicFixedPointStrategy()
      : default_param_bit_(DEFAULT_PARAM_BIT_NUM),
        default_blob_bit_(DEFAULT_BLOB_BIT_NUM),
        is_table_made(0), sigmoid_xrange_(8),
        tanh_xrange_(5), sigmoid_npoints_(1024),
        tanh_npoints_(1024), sigmoid_expo_(15),
        tanh_expo_(15), sigmoid_amp_(1<<15),
        tanh_amp_(1<<15) {}

      ~DynamicFixedPointStrategy() {
        if (is_table_made) {
          CU_SAFE_CALL(cudaFree(sigmoid_x_));
          CU_SAFE_CALL(cudaFree(sigmoid_y_));
          CU_SAFE_CALL(cudaFree(tanh_x_));
          CU_SAFE_CALL(cudaFree(tanh_y_));
        }
      }

      StrategyType GetType() const { return kDynamicFixedPoint; }

      static const int DEFAULT_PARAM_BIT_NUM = 8;
      static const int DEFAULT_BLOB_BIT_NUM = 8;

      int ParamBitNum(int n, Component::ComponentType _type) {
        int bit_num = default_param_bit_;
        IndexIntMap::const_iterator got;
        IndexIntMap::const_iterator got_index;
        IndexIntMap::const_iterator got_type;
        if ((got = param_bit_num_map_.find(n)) != param_bit_num_map_.end()) {
          return got->second;
        }
        if ((got_index = param_index_map_.find(n)) != param_index_map_.end()) {
          bit_num = got_index->second;
        }
        if ((got_type = param_type_map_.find(static_cast<int> (_type))) != param_type_map_.end()) {
          bit_num = got_type->second;
        }
        param_bit_num_map_[n] = bit_num;
        return bit_num;
      }

      int BlobBitNum(int n) {
        int bit_num = default_blob_bit_;
        IndexIntMap::const_iterator got;
        if ((got = blob_bit_num_map_.find(n)) != param_bit_num_map_.end()) {
          return got->second;
        }
        if ((got = blob_index_map_.find(n)) != blob_index_map_.end()) {
          bit_num = got->second;
        }
        blob_bit_num_map_[n] = bit_num;
        return bit_num;
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

        param_bit_num_map_.clear();
        blob_bit_num_map_.clear();

        param_frag_pos_map_.clear();
        blob_frag_pos_map_.clear();
      }

    protected:
      virtual void Initialize() {
        MakeTable();
      }

      virtual void ReadConfigData(std::istream &is, bool binary) {
        while ('<' == Peek(is, binary)) {
          std::string token;
          int raw_type;
          int index;

          int first_char = PeekToken(is, binary);
          switch (first_char) {
          case 'D': ReadToken(is, binary, &token);
            if (token == "<DefaultBlobBit>") {
              ReadBasicType(is, binary, &default_blob_bit_);
            } else if (token == "<DefaultParamBit>") {
              ReadBasicType(is, binary, &default_param_bit_);
            } else {
              KALDI_ERR << "Unknown token: " << token;
            }
            break;
          case 'B': ExpectToken(is, binary, "<BlobIndexBit>");
            ReadBasicType(is, binary, &index);
            ReadBasicType(is, binary, &blob_index_map_[index]);
            break;
          case 'P': ReadToken(is, binary, &token);
            if (token == "<ParamTypeBit>") {
              ReadBasicType(is, binary, &raw_type);
              ReadBasicType(is, binary, &param_type_map_[raw_type]);
            } else if (token == "<ParamIndexBit>") {
              ReadBasicType(is, binary, &index);
              ReadBasicType(is, binary, &param_index_map_[index]);
            } else {
              KALDI_ERR << "Unknown token: " << token;
            }
            break;
          case 'N': ReadToken(is, binary, &token);
            if (token == "<NonLinearSigmoid>") {
              ReadBasicType(is, binary, &sigmoid_xrange_);  // x range
              ReadBasicType(is, binary, &sigmoid_npoints_); // number of points
              // the exponent to multiply to convert to integers
              ReadBasicType(is, binary, &sigmoid_expo_); 
              sigmoid_amp_ = 1 << sigmoid_expo_;
            } else if (token == "<NonLinearTanh>") {
              ReadBasicType(is, binary, &tanh_xrange_);  // x range
              ReadBasicType(is, binary, &tanh_npoints_); // number of points
              // the exponent to shift to convert to integers
              ReadBasicType(is, binary, &tanh_expo_); 
              tanh_amp_ = 1 << tanh_expo_;
            } else {
              KALDI_ERR << "Unknown token: " << token;
            }
            break;
          default: ReadToken(is, binary, &token);
            KALDI_ERR << "Unknown token: " << token;
          }
        }
      }
     
      virtual void WriteConfigData(std::ostream &os, bool binary) const {
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

        WriteToken(os, binary, "<NonLinearSigmoid>");
        WriteBasicType(os, binary, sigmoid_xrange_);  // x range
        WriteBasicType(os, binary, sigmoid_npoints_); // number of points
        // the exponent to multiply to convert to integers
        WriteBasicType(os, binary, sigmoid_expo_);

        WriteToken(os, binary, "<NonLinearTanh>");
        WriteBasicType(os, binary, tanh_xrange_);  // x range
        WriteBasicType(os, binary, tanh_npoints_); // number of points
        // the exponent to multiply to convert to integers
        WriteBasicType(os, binary, tanh_expo_);

        if (!binary) os << "\n";
      }

      virtual void ReadData(std::istream &is, bool binary) {
        if ('<' == Peek(is, binary) && 'M' == PeekToken(is, binary)) {
          // Only read data when file is not null and first token is <Model>
          ExpectToken(is, binary, "<Model>");
          innerReadData(is, binary);
        } else {
          ReadConfigData(is, binary);
        }
      }

      void innerReadData(std::istream &is, bool binary) {
        while ('<' == Peek(is, binary)) {
          std::string token;
          int index;

          int first_char = PeekToken(is, binary);
          switch (first_char) {
          case 'F': ReadToken(is, binary, &token);
            if (token == "<FragPosBlob>") {
              ReadBasicType(is, binary, &index);
              ReadBasicType(is, binary, &blob_frag_pos_map_[index]);
            } else if (token == "<FragPosParam>") {
              ReadBasicType(is, binary, &index);
              ReadBasicType(is, binary, &param_frag_pos_map_[index]);
            } else {
              KALDI_ERR << "Unknown token: " << token;
            }
            break;
          case 'B': ReadToken(is, binary, &token);
            if (token == "<BitNumBlob>") {
              ReadBasicType(is, binary, &index);
              ReadBasicType(is, binary, &blob_bit_num_map_[index]);
            } else if (token == "<BitNumParam>") {
              ReadBasicType(is, binary, &index);
              ReadBasicType(is, binary, &param_bit_num_map_[index]);
            } else {
              KALDI_ERR << "Unknown token: " << token;
            }
            break;
          case 'N': ReadToken(is, binary, &token);
            if (token == "<NonLinearSigmoid>") {
              ReadBasicType(is, binary, &sigmoid_xrange_);  // x range
              ReadBasicType(is, binary, &sigmoid_npoints_); // number of points
              int tmp_expo;
              // the exponent to multiply to convert to integers
              ReadBasicType(is, binary, &tmp_expo); 
              sigmoid_amp_ = 1 << tmp_expo;
            } else if (token == "<NonLinearTanh>") {
              ReadBasicType(is, binary, &tanh_xrange_);  // x range
              ReadBasicType(is, binary, &tanh_npoints_); // number of points
              int tmp_expo;
              // the exponent to multiply to convert to integers
              ReadBasicType(is, binary, &tmp_expo); 
              tanh_amp_ = 1 << tmp_expo;
            } else {
              KALDI_ERR << "Unknown token: " << token;
            }
            break;
          default: ReadToken(is, binary, &token);
            KALDI_ERR << "Unknown token: " << token;
          }
        }
      }

      void innerWriteData(std::ostream &os, bool binary) const {
        for( IndexIntMap::const_iterator item = param_bit_num_map_.begin(); item != param_bit_num_map_.end(); ++item ) { 
          WriteToken(os, binary, "<BitNumParam>");
          WriteBasicType(os, binary, item->first);
          WriteBasicType(os, binary, item->second);
        }
        for( IndexIntMap::const_iterator item = blob_bit_num_map_.begin(); item != blob_bit_num_map_.end(); ++item ) { 
          WriteToken(os, binary, "<BitNumBlob>");
          WriteBasicType(os, binary, item->first);
          WriteBasicType(os, binary, item->second);
        }

        for( IndexIntMap::const_iterator item = param_frag_pos_map_.begin(); item != param_frag_pos_map_.end(); ++item ) { 
          WriteToken(os, binary, "<FragPosParam>");
          WriteBasicType(os, binary, item->first);
          WriteBasicType(os, binary, item->second);
        }
        for( IndexIntMap::const_iterator item = blob_frag_pos_map_.begin(); item != blob_frag_pos_map_.end(); ++item ) { 
          WriteToken(os, binary, "<FragPosBlob>");
          WriteBasicType(os, binary, item->first);
          WriteBasicType(os, binary, item->second);
        }

        WriteToken(os, binary, "<NonLinearSigmoid>");
        WriteBasicType(os, binary, sigmoid_xrange_);  // x range
        WriteBasicType(os, binary, sigmoid_npoints_); // number of points
        // the exponent to multiply to convert to integers
        WriteBasicType(os, binary, sigmoid_expo_);

        WriteToken(os, binary, "<NonLinearTanh>");
        WriteBasicType(os, binary, tanh_xrange_);  // x range
        WriteBasicType(os, binary, tanh_npoints_); // number of points
        // the exponent to multiply to convert to integers
        WriteBasicType(os, binary, tanh_expo_);

        if (!binary) os << "\n";
      }

      virtual void WriteData(std::ostream &os, bool binary, bool config_only) const {
        if (config_only) {
          WriteConfigData(os, binary);
        } else {
          WriteToken(os, binary, "<Model>");
          innerWriteData(os, binary);
        }
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
        cuda_saturate(dimGrid, dimBlock, blob.Data(), maxnum, minnum, blob.Dim());

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

      void MakeTable() {
        // Make sigmoid table
        BaseFloat *sigmoid_x_array = new BaseFloat[sigmoid_npoints_ + 1];
        int32 * sigmoid_x_bin = new int[sigmoid_npoints_ + 1];
        BaseFloat *sigmoid_y_array = new BaseFloat[sigmoid_npoints_ + 1];
        int32 *sigmoid_y_bin = new int[sigmoid_npoints_ + 1];

        for (int i = 0; i < sigmoid_npoints_ + 1; i++) {
          sigmoid_x_array[i] = -1 + i * 2 / (static_cast<BaseFloat>(sigmoid_npoints_));
        }
        for (int i = 0; i < sigmoid_npoints_ + 1; i++) {
          sigmoid_x_bin[i] = (int)((sigmoid_x_array[i] + 1) * sigmoid_amp_ + 0.5);
        }
        for (int i = 0; i < sigmoid_npoints_ + 1; i++) {
          sigmoid_y_array[i] = 1 / (1 + exp(-sigmoid_x_array[i] * sigmoid_xrange_));
        }
        for (int i = 0; i < sigmoid_npoints_ + 1; i++) {
          sigmoid_y_bin[i] = (int)(sigmoid_y_array[i] * sigmoid_amp_ + 0.5);
        }

        sigmoid_x_ = static_cast<int*>(CuDevice::Instantiate().Malloc((sigmoid_npoints_ + 1) * sizeof(int)));
        CU_SAFE_CALL(cudaMemcpy(sigmoid_x_, sigmoid_x_bin, (sigmoid_npoints_ + 1) * sizeof(int),
                                cudaMemcpyHostToDevice));
        sigmoid_y_ = static_cast<int*>(CuDevice::Instantiate().Malloc((sigmoid_npoints_ + 1) * sizeof(int)));
        CU_SAFE_CALL(cudaMemcpy(sigmoid_y_, sigmoid_y_bin, (sigmoid_npoints_ + 1) * sizeof(int),
                                cudaMemcpyHostToDevice));
        delete[] sigmoid_x_array;
        delete[] sigmoid_y_array;
        delete[] sigmoid_x_bin;
        delete[] sigmoid_y_bin;

        // Make tanh table
        BaseFloat *tanh_x_array = new BaseFloat[tanh_npoints_ + 1];
        int32 * tanh_x_bin = new int[tanh_npoints_ + 1];
        BaseFloat *tanh_y_array = new BaseFloat[tanh_npoints_ + 1];
        int32 *tanh_y_bin = new int[tanh_npoints_ + 1];

        for (int i = 0; i < tanh_npoints_ + 1; i++) {
          tanh_x_array[i] = -1 + i * 2 / ((BaseFloat)tanh_npoints_);
        }
        for (int i = 0; i < tanh_npoints_ + 1; i++) {
          tanh_x_bin[i] = (int)((tanh_x_array[i] + 1) * tanh_amp_ + 0.5);
        }
        for (int i = 0; i < tanh_npoints_ + 1; i++) {
          tanh_y_array[i] = (exp(tanh_x_array[i] * tanh_xrange_) - 
                             exp(-tanh_x_array[i] * tanh_xrange_)) /
            (exp(tanh_x_array[i] * tanh_xrange_) + exp(-tanh_x_array[i] * tanh_xrange_));
        }
        for (int i = 0; i < tanh_npoints_ + 1; i++) {
          tanh_y_bin[i] = (int)(tanh_y_array[i] * tanh_amp_ + 0.5);
        }

        tanh_x_ = static_cast<int*>(CuDevice::Instantiate().Malloc((tanh_npoints_ + 1) * sizeof(int)));
        CU_SAFE_CALL(cudaMemcpy(tanh_x_, tanh_x_bin, (tanh_npoints_ + 1) * sizeof(int),
                                cudaMemcpyHostToDevice));
        tanh_y_ = static_cast<int*>(CuDevice::Instantiate().Malloc((tanh_npoints_ + 1) * sizeof(int)));
        CU_SAFE_CALL(cudaMemcpy(tanh_y_, tanh_y_bin, (tanh_npoints_ + 1) * sizeof(int),
                                cudaMemcpyHostToDevice));

        // Relase all these extra cpu heap memory
        // FIXME: if we want to support cpu calculation (pass in MatrixBase<BaseFloat>,
        //        we'll still need these memory.
        is_table_made = 1;
        delete[] tanh_x_array;
        delete[] tanh_y_array;
        delete[] tanh_x_bin;
        delete[] tanh_y_bin;
      }

      virtual void DoFixSigm(CuMatrixBase<BaseFloat> &blob, int n) {
        dim3 dimGrid, dimBlock;
        GetBlockSizesForSimpleMatrixOperation(blob.NumRows(), blob.NumCols(), &dimGrid, &dimBlock);
        cuda_mapping(dimGrid, dimBlock, blob.Data(), sigmoid_xrange_, sigmoid_x_, sigmoid_y_, sigmoid_npoints_, 0, 1, sigmoid_amp_, blob.Dim());

      }

      virtual void DoFixTanh(CuMatrixBase<BaseFloat> &blob, int n)
      {
        dim3 dimGrid, dimBlock;
        GetBlockSizesForSimpleMatrixOperation(blob.NumRows(), blob.NumCols(), &dimGrid, &dimBlock);
        cuda_mapping(dimGrid, dimBlock, blob.Data(), tanh_xrange_, tanh_x_, tanh_y_, tanh_npoints_, -1, 1, tanh_amp_, blob.Dim());
      }

    private:
      IndexIntMap param_type_map_;
      IndexIntMap param_index_map_;

      IndexIntMap blob_index_map_;

      IndexIntMap param_bit_num_map_;
      IndexIntMap blob_bit_num_map_;
      IndexIntMap param_frag_pos_map_;
      IndexIntMap blob_frag_pos_map_;
      
      int default_param_bit_;
      int default_blob_bit_;

      // For nonlinear table lookup
      int is_table_made;
      int sigmoid_xrange_;
      int tanh_xrange_;
      int sigmoid_npoints_;
      int tanh_npoints_;
      int sigmoid_expo_;
      int tanh_expo_;
      BaseFloat sigmoid_amp_;
      BaseFloat tanh_amp_;

      int* sigmoid_x_; // on device
      int* sigmoid_y_; // on device
      int* tanh_x_;    // on device
      int* tanh_y_;    // on device

    }; // end class DynamicFixedPointStrategy
  } // end namespace fix
} // end namespace kaldi

#endif
