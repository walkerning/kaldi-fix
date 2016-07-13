#include <fix/fix-strategy.h>
#include "nnet/nnet-nnet-fix.h"
#include "fix/fix-dynamic-fixed-point.h"

namespace kaldi {
  namespace fix {

    const struct FixStrategy::key_value FixStrategy::kMarkerMap[] = {
      { FixStrategy::kDynamicFixedPoint, "<DynamicFixedPoint>" },
    };

    const char* FixStrategy::TypeToMarker(StrategyType t) {
      int32 N = sizeof(kMarkerMap) / sizeof(kMarkerMap[0]);
      for (int i = 0; i < N; i++) {
	if (kMarkerMap[i].key == t) return kMarkerMap[i].value;
      }
      KALDI_ERR << "Unknown type : " << t;
      return NULL;
    }
    
    FixStrategy::StrategyType FixStrategy::MarkerToType(const std::string &s) {
      std::string s_lowercase(s);
      std::transform(s.begin(), s.end(), s_lowercase.begin(), ::tolower);  // lc
      int32 N = sizeof(kMarkerMap) / sizeof(kMarkerMap[0]);
      for (int i = 0; i < N; i++) {
	std::string m(kMarkerMap[i].value);
	std::string m_lowercase(m);
	std::transform(m.begin(), m.end(), m_lowercase.begin(), ::tolower);
	if (s_lowercase == m_lowercase) return kMarkerMap[i].key;
      }
      KALDI_ERR << "Unknown marker : '" << s << "'";
      return kUnknown;
    }

    std::shared_ptr<FixStrategy> FixStrategy::Read(std::istream &is, bool binary) {
      std::string token;
      
      int first_char = Peek(is, binary);
      if (first_char == EOF) return NULL;
      
      ReadToken(is, binary, &token);
      std::shared_ptr<FixStrategy> ans = std::shared_ptr<FixStrategy>(NewStrategyOfType(MarkerToType(token)));
      ans -> ReadData(is, binary);

      ExpectToken(is, binary, "<!EndOfStrategy>");
      return ans;
    }

   void FixStrategy::Write(std::ostream &os, bool binary) const {
      WriteToken(os, binary, FixStrategy::TypeToMarker(GetType()));
      if (!binary) os << "\n";
      WriteData(os, binary);
      WriteToken(os, binary, "<!EndOfStrategy>");  // Write component separator.
      if (!binary) os << "\n";
    }

    // std::shared_ptr<FixStrategy> FixStrategy::Init(const std::string &conf_line) {
    //   std::istringstream is(conf_line);
    //   std::string strategy_type_string;
    //   int32 input_dim, output_dim;

    //   // initialize w/o internal data
    //   ReadToken(is, false, &strategy_type_string);
    //   StrategyType strategy_type = MarkerToType(strategy_type_string);
    //   std::shared_ptr<FixStrategy> ans = NewStrategyOfType(strategy_type);

    //   // initialize internal data with the remaining part of config line
    //   ans->InitData(is);

    //   return ans;
    // }

    FixStrategy* FixStrategy::NewStrategyOfType(StrategyType strategy_type) {
      FixStrategy* ans = NULL;
      switch (strategy_type) {
      case FixStrategy::kDynamicFixedPoint:
	ans = new DynamicFixedPointStrategy();
	break;
      default:
	KALDI_ERR << "Missing type: " << TypeToMarker(strategy_type);
      }
      return ans;
    }

    void FixStrategy::FixParam(kaldi::nnet1::NnetFix& nnet_fix) {
      Vector<BaseFloat> vector;
      for (int32 n = 0; n < nnet_fix.NumComponents(); n++) {
	// FIXME: 是不是updatable代表需要定点
	if (nnet_fix.GetComponent(n).IsUpdatable()) {
	  dynamic_cast<kaldi::nnet1::UpdatableComponent&>(nnet_fix.GetComponent(n)).GetParams(&vector);
	}
 	this->DoFixParam(vector, nnet_fix.GetComponent(n).GetType(), n);
	dynamic_cast<kaldi::nnet1::UpdatableComponent&>(nnet_fix.GetComponent(n)).SetParams(vector);
      }
    }

  } // end namespace fix
} // end namespace kaldi
