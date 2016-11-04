#ifndef KALDI_FIX_DYNAMIC_FIXED_POINT_H_
#define KALDI_FIX_DYNAMIC_FIXED_POINT_H_

#include <limits>
#include <algorithm>
#include <cmath>

#include "fix/fix-strategy.h"
#include "fix/fix-kernels-ansi.h"
#include "fix/fix-nonlinear-kernels-ansi.h"

namespace kaldi {
  namespace fix {
    using namespace kaldi::nnet1;

    class DynamicFixedPointStrategy : public FixStrategy {

    public:
    DynamicFixedPointStrategy()
      : default_param_bit_(DEFAULT_PARAM_BIT_NUM),
        default_blob_bit_(DEFAULT_BLOB_BIT_NUM),
        is_table_made(0), 
        sigmoid_xrange_(8), tanh_xrange_(4), 
        sigmoid_npoints_(2048), tanh_npoints_(2048), 
        output_amp_(1<<15) {}

      ~DynamicFixedPointStrategy() {
        if (is_table_made) {
          CuDevice::Instantiate().Free(this->sigmoid_y_);
          CuDevice::Instantiate().Free(this->tanh_y_);
        }
      }

      StrategyType GetType() const { return kDynamicFixedPoint; }

      static const int DEFAULT_PARAM_BIT_NUM = 16;
      static const int DEFAULT_BLOB_BIT_NUM = 16;

      std::vector<int> ParamBitNum(int n, Component::ComponentType _type) {
	// if ParamBitNum has been given in the fixconf file
        IndexVectorMap::const_iterator got;
        if ((got = param_bit_num_map_.find(n)) != param_bit_num_map_.end()) {
          return got->second;
        }
	// otherwise
        std::vector<int> bit_num;
        if (_type == kaldi::nnet1::Component::MarkerToType("<LstmProjectedStreams>")) {
          int default_bitnum[7] = {12,12,16,16,16,16,12};
          for (int i = 0; i < 7; ++i) {
            bit_num.push_back(default_bitnum[i]);
          }
        } else if (_type == kaldi::nnet1::Component::MarkerToType("<AffineTransform>")) {
          bit_num.push_back(12);
          bit_num.push_back(16);
        } else {
          bit_num.push_back(16);
        }
        return bit_num;
      }

      int BlobBitNum(int n) {
        IndexIntMap::const_iterator got;
	// if BlobBitNum has been given in the fixconf file
        if ((got = blob_bit_num_map_.find(n)) != blob_bit_num_map_.end()) {
          return got->second;
        }
	// otherwise
        blob_bit_num_map_[n] = default_blob_bit_;
        return default_blob_bit_;
      }

      int BlobFragPos(int n, const CuMatrixBase<BaseFloat>& blob, int bit_num) {
	// if BlobFragPos has been given in the fixconf file
        IndexIntMap::const_iterator got;
        if ((got = blob_frag_pos_map_.find(n)) != blob_frag_pos_map_.end()) {
          return got->second;
        }
	// otherwise
	BaseFloat b_max = std::max(fabs(blob.Max()), fabs(blob.Min()));
        int frag_pos = bit_num - 1 - ceil(log(b_max) / log(2));
	blob_frag_pos_map_[n] = frag_pos;
        return frag_pos;
      }

      int BlobFragPos(int n, const MatrixBase<BaseFloat>& blob, int bit_num) {
	// if BlobFragPos has been given in the fixconf file
        IndexIntMap::const_iterator got;
        if ((got = blob_frag_pos_map_.find(n)) != blob_frag_pos_map_.end()) {
          return got->second;
        }
	// otherwise
	BaseFloat b_max = std::max(fabs(blob.Max()), fabs(blob.Min()));
        int frag_pos = bit_num - 1 - ceil(log(b_max) / log(2));
	blob_frag_pos_map_[n] = frag_pos;
        return frag_pos;
      }

      std::vector<int> ParamFragPos(int n, const VectorBase<BaseFloat>& blob, std::vector<int>& bit_num, std::vector<int>& inner_num_param) {
        // if ParamFragPos has been given in the fixconf file
        IndexVectorMap::const_iterator got;
        if ((got = param_frag_pos_map_.find(n)) != param_frag_pos_map_.end()) {
          return got->second;
        }
	// otherwise
	std::vector<int> frag_pos;
	int pos = 0;
	BaseFloat b_max = 0;
        for (size_t i = 0; i < inner_num_param.size(); ++i) {
          SubVector<BaseFloat> temp(blob.Range(pos, inner_num_param[i]));
          b_max = std::max(fabs(temp.Max()), fabs(temp.Min()));
          frag_pos.push_back( bit_num[i] - 1 - ceil(log(b_max) / log(2)) );
          pos += inner_num_param[i];
        }
        param_frag_pos_map_[n] = frag_pos;
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

        /*
        if (result > maxnum) {
          result = maxnum;
        } else if (result < minnum) {
          result = minnum;
        }
        */
       
        int result_fix = int(result);
        if (result_fix > maxnum) {
          result_fix %= (maxnum + 1);
        } else if (result_fix < minnum) {
          result_fix %= minnum;
        }

        result = BaseFloat(result_fix);
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
            } else if (token == "<NonLinearTanh>") {
              ReadBasicType(is, binary, &tanh_xrange_);  // x range
              ReadBasicType(is, binary, &tanh_npoints_); // number of points
              // the exponent to shift to convert to integers
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

        WriteToken(os, binary, "<NonLinearTanh>");
        WriteBasicType(os, binary, tanh_xrange_);  // x range
        WriteBasicType(os, binary, tanh_npoints_); // number of points
        // the exponent to multiply to convert to integers

        if (!binary) os << "\n";
      }

      virtual void ReadData(std::istream &is, bool binary, kaldi::nnet1::NnetFix& nnet_fix) {
        if ('<' == Peek(is, binary) && 'M' == PeekToken(is, binary)) {
          // Only read data when file is not null and first token is <Model>
          ExpectToken(is, binary, "<Model>");
          innerReadData(is, binary, nnet_fix);
        } else {
          ReadConfigData(is, binary);
        }
      }

      void innerReadData(std::istream &is, bool binary, kaldi::nnet1::NnetFix& nnet_fix) {
        while ('<' == Peek(is, binary)) {
	  std::string token;
          int index;

          int first_char = PeekToken(is, binary);
          switch (first_char) {
          case 'F': ReadToken(is, binary, &token);
            if (token == "<FragPosBlob>") {
	      std::string subtoken;
              ReadToken(is, binary, &subtoken);
              if(subtoken == "<Layer>"){
                ReadBasicType(is, binary, &index);
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
              ReadToken(is, binary, &subtoken);
              if(subtoken == "<Max>"){
                ReadBasicType(is, binary, &blob_frag_pos_map_[index]);
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
            } else if (token == "<FragPosParam>") {
	      std::string subtoken;
              ReadToken(is, binary, &subtoken);
              if(subtoken == "<Component>"){
                ReadBasicType(is, binary, &index);
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
              // ReadBasicType(is, binary, &param_frag_pos_map_[index]);
              if (nnet_fix.GetComponent(index).GetType() == kaldi::nnet1::Component::MarkerToType("<LstmProjectedStreams>")) {
		std::vector<int> temp(7,0);
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<w_gifo_x_>"){
                  ReadBasicType(is, binary, &temp[0]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<w_gifo_r_>"){
                  ReadBasicType(is, binary, &temp[1]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<bias_>"){
                  ReadBasicType(is, binary, &temp[2]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<peephole_i_c_>"){
                  ReadBasicType(is, binary, &temp[3]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<peephole_f_c_>"){
                  ReadBasicType(is, binary, &temp[4]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<peephole_o_c_>"){
                  ReadBasicType(is, binary, &temp[5]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<w_r_m_>"){
                  ReadBasicType(is, binary, &temp[6]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                param_frag_pos_map_[index] = temp;
              } else{
                KALDI_ERR << "Unknown type: " << nnet_fix.GetComponent(index).GetType();
              }
            } else {
              KALDI_ERR << "Unknown token: " << token;
            }
            break;
          case 'B': ReadToken(is, binary, &token);
            if (token == "<BitNumBlob>") {
	      std::string subtoken;
              ReadToken(is, binary, &subtoken);
              if(subtoken == "<Layer>"){
                ReadBasicType(is, binary, &index);
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
              ReadToken(is, binary, &subtoken);
              if(subtoken == "<Max>"){
                ReadBasicType(is, binary, &blob_bit_num_map_[index]);
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
            } else if (token == "<BitNumParam>") {
	      std::string subtoken;
              ReadToken(is, binary, &subtoken);
              if(subtoken == "<Component>"){
                ReadBasicType(is, binary, &index);
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
              // ReadBasicType(is, binary, &param_bit_num_map_[index]);
              if (nnet_fix.GetComponent(index).GetType() == kaldi::nnet1::Component::MarkerToType("<LstmProjectedStreams>")) {
		std::vector<int> temp(7,0);
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<w_gifo_x_>"){
                  ReadBasicType(is, binary, &temp[0]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<w_gifo_r_>"){
                  ReadBasicType(is, binary, &temp[1]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<bias_>"){
                  ReadBasicType(is, binary, &temp[2]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<peephole_i_c_>"){
                  ReadBasicType(is, binary, &temp[3]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<peephole_f_c_>"){
                  ReadBasicType(is, binary, &temp[4]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<peephole_o_c_>"){
                  ReadBasicType(is, binary, &temp[5]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                ReadToken(is, binary, &subtoken);
                if(subtoken == "<w_r_m_>"){
                  ReadBasicType(is, binary, &temp[6]);
                }else{
                  KALDI_ERR << "Unknown subtoken: " << subtoken;
                }
                param_frag_pos_map_[index] = temp;
              } else {
                KALDI_ERR << "Unknown type: " << nnet_fix.GetComponent(index).GetType();
              }
            } else {
              KALDI_ERR << "Unknown token: " << token;
            }
            break;
          case 'N': ReadToken(is, binary, &token);
            if (token == "<NonLinearSigmoid>") {
	      std::string subtoken;
              ReadToken(is, binary, &subtoken);
              if (subtoken == "<x_range>"){
                ReadBasicType(is, binary, &sigmoid_xrange_);  // x range
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
              ReadToken(is, binary, &subtoken);
              if (subtoken == "<n_points>"){
                ReadBasicType(is, binary, &sigmoid_npoints_); // number of points
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
            } else if (token == "<NonLinearTanh>") {
	      std::string subtoken;
              ReadToken(is, binary, &subtoken);
              if (subtoken == "<x_range>"){
                ReadBasicType(is, binary, &tanh_xrange_);  // x range
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
              ReadToken(is, binary, &subtoken);
              if (subtoken == "<n_points>"){
                ReadBasicType(is, binary, &tanh_npoints_); // number of points
              }else{
                KALDI_ERR << "Unknown subtoken: " << subtoken;
              }
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
        for( IndexVectorMap::const_iterator item = param_bit_num_map_.begin(); item != param_bit_num_map_.end(); ++item ) {
          WriteToken(os, binary, "<BitNumParam>");
          os << "\n";
          WriteToken(os, binary, "<Component>");
          WriteBasicType(os, binary, item->first);
          int dis;
          for ( std::vector<int>::const_iterator order = (item->second).begin(); order != (item->second).end(); ++order) {
            dis = std::distance((item->second).begin(), order);
            switch(dis){
	    case 0 : WriteToken(os, binary, "<w_gifo_x_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 1 : WriteToken(os, binary, "<w_gifo_r_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 2 : WriteToken(os, binary, "<bias_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 3 : WriteToken(os, binary, "<peephole_i_c_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 4 : WriteToken(os, binary, "<peephole_f_c_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 5 : WriteToken(os, binary, "<peephole_o_c_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 6 : WriteToken(os, binary, "<w_r_m_>");
	      WriteBasicType(os, binary, *order);
	      os << "\n";
	      break;
	    default : KALDI_ERR << "Overflow";
	      break;
            }
          }
        }
        for( IndexIntMap::const_iterator item = blob_bit_num_map_.begin(); item != blob_bit_num_map_.end(); ++item ) {
          WriteToken(os, binary, "<BitNumBlob>");
          os << "\n";
          WriteToken(os, binary, "<Layer>");
          WriteBasicType(os, binary, item->first);
          WriteToken(os, binary, "<Max>");
          WriteBasicType(os, binary, item->second);
          os << "\n";
        }

        for( IndexVectorMap::const_iterator item = param_frag_pos_map_.begin(); item != param_frag_pos_map_.end(); ++item ) {
          WriteToken(os, binary, "<FragPosParam>");
          os << "\n";
          WriteToken(os, binary, "<Component>");
          WriteBasicType(os, binary, item->first);
          int dis;
          for ( std::vector<int>::const_iterator order = (item->second).begin(); order != (item->second).end(); ++order) {
            dis = std::distance((item->second).begin(), order);
            switch(dis){
	    case 0 : WriteToken(os, binary, "<w_gifo_x_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 1 : WriteToken(os, binary, "<w_gifo_r_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 2 : WriteToken(os, binary, "<bias_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 3 : WriteToken(os, binary, "<peephole_i_c_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 4 : WriteToken(os, binary, "<peephole_f_c_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 5 : WriteToken(os, binary, "<peephole_o_c_>");
	      WriteBasicType(os, binary, *order);
	      break;
	    case 6 : WriteToken(os, binary, "<w_r_m_>");
	      WriteBasicType(os, binary, *order);
	      os << "\n";
	      break;
	    default : KALDI_ERR << "Overflow";
	      break;
            }
          }
        }
        for( IndexIntMap::const_iterator item = blob_frag_pos_map_.begin(); item != blob_frag_pos_map_.end(); ++item ) {
          WriteToken(os, binary, "<FragPosBlob>");
          os << "\n";
          WriteToken(os, binary, "<Layer>");
          WriteBasicType(os, binary, item->first);
          WriteToken(os, binary, "<Max>");
          WriteBasicType(os, binary, item->second);
          os << "\n";
        }

        WriteToken(os, binary, "<NonLinearSigmoid>");
        os << "\n";
        WriteToken(os, binary, "<x_range>");
        WriteBasicType(os, binary, sigmoid_xrange_);  // x range
        WriteToken(os, binary, "<n_points>");
        WriteBasicType(os, binary, sigmoid_npoints_); // number of points
        os << "\n";
        WriteToken(os, binary, "<NonLinearTanh>");
        os << "\n";
        WriteToken(os, binary, "<x_range>");
        WriteBasicType(os, binary, tanh_xrange_);  // x range
        WriteToken(os, binary, "<n_points>");
        WriteBasicType(os, binary, tanh_npoints_); // number of points

        if (!binary) os << "\n";
      }

      virtual void WriteData(std::ostream &os, bool binary, bool config_only) const {
        if (!config_only) {
          WriteToken(os, binary, "<Model>");
          os << "\n";
          innerWriteData(os, binary);
        } else {
          KALDI_ERR << "Cannot config_only!";
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
                              int n,
                              std::vector<int> inner_num_param) {
        std::vector<int> bit_num = ParamBitNum(n, comp_type);
        std::vector<int> frag_pos = ParamFragPos(n, blob, bit_num, inner_num_param);

        BaseFloat* data = blob.Data();
        for (size_t j = 0; j < frag_pos.size(); ++j) {
          // float to fix
          for (MatrixIndexT i = 0; i < inner_num_param[j]; ++i) {
            data[i] = Float2Fix(data[i], bit_num[j], frag_pos[j]);
          }
          data += inner_num_param[j];
        }
      }

      void MakeTable() {
        // Make sigmoid table
        BaseFloat *sigmoid_x_array = new BaseFloat[sigmoid_npoints_];
        BaseFloat *sigmoid_y_array = new BaseFloat[sigmoid_npoints_];
        int32 *sigmoid_y_bin = new int[sigmoid_npoints_];

        for (int i = 0; i < sigmoid_npoints_; ++i) {
          sigmoid_x_array[i] = -1 + i * 2 / (static_cast<BaseFloat>(sigmoid_npoints_));
        }
        for (int i = 0; i < sigmoid_npoints_; ++i) {
          sigmoid_y_array[i] = 1 / (1 + exp(-sigmoid_x_array[i] * sigmoid_xrange_));
        }
        for (int i = 0; i < sigmoid_npoints_; ++i) {
          sigmoid_y_bin[i] = (int)(sigmoid_y_array[i] * output_amp_ + 0.5);
        }

        sigmoid_y_ = static_cast<int*>(CuDevice::Instantiate().Malloc((sigmoid_npoints_) * sizeof(int)));
        CU_SAFE_CALL(cudaMemcpy(sigmoid_y_, sigmoid_y_bin, (sigmoid_npoints_) * sizeof(int),
                                cudaMemcpyHostToDevice));
        delete[] sigmoid_x_array;
        delete[] sigmoid_y_array;
        delete[] sigmoid_y_bin;

        // Make tanh table
        BaseFloat *tanh_x_array = new BaseFloat[tanh_npoints_];
        BaseFloat *tanh_y_array = new BaseFloat[tanh_npoints_];
        int32 *tanh_y_bin = new int[tanh_npoints_];

        for (int i = 0; i < tanh_npoints_; ++i) {
          tanh_x_array[i] = -1 + i * 2 / ((BaseFloat)tanh_npoints_);
        }
        for (int i = 0; i < tanh_npoints_; ++i) {
          tanh_y_array[i] = (exp(tanh_x_array[i] * tanh_xrange_) - 
                             exp(-tanh_x_array[i] * tanh_xrange_)) /
            (exp(tanh_x_array[i] * tanh_xrange_) + exp(-tanh_x_array[i] * tanh_xrange_));
        }
        for (int i = 0; i < tanh_npoints_; ++i) {
          tanh_y_bin[i] = (int)(tanh_y_array[i] * output_amp_ + 0.5);
        }

        tanh_y_ = static_cast<int*>(CuDevice::Instantiate().Malloc((tanh_npoints_) * sizeof(int)));
        CU_SAFE_CALL(cudaMemcpy(tanh_y_, tanh_y_bin, (tanh_npoints_) * sizeof(int),
                                cudaMemcpyHostToDevice));
        delete[] tanh_x_array;
        delete[] tanh_y_array;
        delete[] tanh_y_bin;

        is_table_made = 1;
      }

      virtual void DoFixSigm(CuMatrixBase<BaseFloat> &blob,
                             const CuMatrixBase<BaseFloat> &in,
                             int n) {
        dim3 dimGrid, dimBlock;
        GetBlockSizesForSimpleMatrixOperation(blob.NumRows(), blob.NumCols(), &dimGrid, &dimBlock);
        cuda_mapping(dimGrid, dimBlock, blob.Data(), in.Data(), sigmoid_xrange_, sigmoid_y_, sigmoid_npoints_, 0, 1, output_amp_, blob.Dim(), in.Stride());

      }

      virtual void DoFixTanh(CuMatrixBase<BaseFloat> &blob,
                             const CuMatrixBase<BaseFloat> &in,
                             int n)
      {
        dim3 dimGrid, dimBlock;
        GetBlockSizesForSimpleMatrixOperation(blob.NumRows(), blob.NumCols(), &dimGrid, &dimBlock);
        cuda_mapping(dimGrid, dimBlock, blob.Data(), in.Data(), tanh_xrange_, tanh_y_, tanh_npoints_, -1, 1, output_amp_, blob.Dim(), in.Stride());
      }

      virtual void DoSetupStrategy(std::ostream &os, bool binary, bool config_only) {}

    private:
      IndexIntMap param_type_map_;
      IndexIntMap param_index_map_;

      IndexIntMap blob_index_map_;

      IndexVectorMap param_bit_num_map_;
      IndexIntMap blob_bit_num_map_;
      IndexVectorMap param_frag_pos_map_;
      IndexIntMap blob_frag_pos_map_;
      
      int default_param_bit_;
      int default_blob_bit_;

      // For nonlinear table lookup
      int is_table_made;
      int sigmoid_xrange_;
      int tanh_xrange_;
      int sigmoid_npoints_;
      int tanh_npoints_;
      BaseFloat output_amp_;

      int* sigmoid_y_; // on device
      int* tanh_y_;    // on device

    }; // end class DynamicFixedPointStrategy
  } // end namespace fix
} // end namespace kaldi

#endif
