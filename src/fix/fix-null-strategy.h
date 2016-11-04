#ifndef KALDI_FIX_NULL_STRATEGY_H_
#define KALDI_FIX_NULL_STRATEGY_H_

#include "fix/fix-strategy.h"

namespace kaldi {
  namespace fix {

    typedef std::tr1::unordered_map<int, BaseFloat> IndexFloatMap;

    class NullStrategy : public FixStrategy {
    public:
    NullStrategy()
      : default_blob_bit(16),
	sigmoid_xrange(8),
	tanh_xrange(4),
	sigmoid_npoints(2048),
	tanh_npoints(2048) {}
	  
      virtual StrategyType GetType() const { return kNullStrategy; }

    protected:

      virtual void ReadData(std::istream &is, bool binary, kaldi::nnet1::NnetFix& nnet_fix) {}

      void innerWriteData(std::ostream &os, bool binary) const {
        for( IndexVectorMap::const_iterator item = param_bit_num.begin(); item != param_bit_num.end(); ++item ) {
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
        for( IndexIntMap::const_iterator item = blob_bit_num.begin(); item != blob_bit_num.end(); ++item ) {
          WriteToken(os, binary, "<BitNumBlob>");
          os << "\n";
          WriteToken(os, binary, "<Layer>");
          WriteBasicType(os, binary, item->first);
          WriteToken(os, binary, "<Max>");
          WriteBasicType(os, binary, item->second);
          os << "\n";
        }

        for( IndexVectorMap::const_iterator item = param_frag_pos.begin(); item != param_frag_pos.end(); ++item ) {
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
        for( IndexIntMap::const_iterator item = blob_frag_pos.begin(); item != blob_frag_pos.end(); ++item ) {
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
        WriteBasicType(os, binary, sigmoid_xrange);  // x range
        WriteToken(os, binary, "<n_points>");
        WriteBasicType(os, binary, sigmoid_npoints); // number of points
        os << "\n";
        WriteToken(os, binary, "<NonLinearTanh>");
        os << "\n";
        WriteToken(os, binary, "<x_range>");
        WriteBasicType(os, binary, tanh_xrange);  // x range
        WriteToken(os, binary, "<n_points>");
        WriteBasicType(os, binary, tanh_npoints); // number of points

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
	IndexFloatMap::const_iterator got;
	if ((got = blob_min.find(n)) != blob_min.end()) {
	  BaseFloat min_temp = blob.Min();
	  BaseFloat max_temp = blob.Max();
	  if (min_temp < blob_min[n])
	    blob_min[n] = min_temp;
	  if (max_temp > blob_max[n])
	    blob_max[n] = max_temp;
	} else {
	  blob_min[n] = blob.Min();
	  blob_max[n] = blob.Max();
	}
      }

      virtual void DoFixBlob(MatrixBase<BaseFloat> &blob, int n) {}

      virtual void DoFixParam(VectorBase<BaseFloat> &blob,
                              Component::ComponentType comp_type,
                              int n,
                              std::vector<int> inner_num_param) {
	if (comp_type == kaldi::nnet1::Component::MarkerToType("<LstmProjectedStreams>")) {
	  int bit_num[7] = {12,12,16,16,16,16,12};
	  int pos = 0;
	  BaseFloat b_max = 0;
	  for (size_t i = 0; i < 7; ++i) {
	    param_bit_num[n].push_back(bit_num[i]);

	    SubVector<BaseFloat> temp(blob.Range(pos, inner_num_param[i]));
	    b_max = std::max(fabs(temp.Max()), fabs(temp.Min()));
	    param_frag_pos[n].push_back( bit_num[i] - 1 - ceil(log(b_max) / log(2)));
	    pos += inner_num_param[i];
	  }
	}
      }

      virtual void DoFixSigm(CuMatrixBase<BaseFloat> &blob,
                             const CuMatrixBase<BaseFloat> &in,
                             int n) {
        blob.Sigmoid(in); // default behavior
      }
      virtual void DoFixTanh(CuMatrixBase<BaseFloat> &blob,
                             const CuMatrixBase<BaseFloat> &in,
                             int n) {
        blob.Tanh(in); // default behavior
      }

      virtual void DoSetupStrategy(std::ostream &os, bool binary, bool config_only) {
	int index = 0;
	BaseFloat b_max = 0;
	// confirm fixconf related to blob
	for (IndexFloatMap::const_iterator got = blob_max.begin(); got != blob_max.end(); ++got) {
	  // confirm blob_bit_num
	  index = got->first;
	  blob_bit_num[index] = default_blob_bit;
	  // confirm blob_frag_pos
	  b_max = std::max(fabs(blob_min[index]), fabs(blob_max[index]));
	  blob_frag_pos[index] = default_blob_bit - 1 - ceil(log(b_max) / log(2));
	}

	// fix-point of bias in lstm should be aligned with the fix-point of blob for this layer
	for (IndexVectorMap::iterator got = param_frag_pos.begin(); got != param_frag_pos.end(); ++got) {
	  index = got->first;
	  IndexIntMap::const_iterator item = blob_frag_pos.find(index + 1);
	  (got->second)[2] = item->second;
	}

	Write(os, binary, config_only);
      }

    private:

      IndexVectorMap param_bit_num;
      IndexVectorMap param_frag_pos; 
      IndexIntMap blob_bit_num;
      IndexIntMap blob_frag_pos; 

      IndexFloatMap blob_min;
      IndexFloatMap blob_max;
      
      int default_blob_bit;

      int sigmoid_xrange;
      int tanh_xrange;
      int sigmoid_npoints;
      int tanh_npoints;
    };
  }
}
#endif
