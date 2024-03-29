#include "fix/fix-strategy.h"
#include "nnet/nnet-nnet-fix.h"
#include "fix/fix-dynamic-fixed-point.h"
#include "fix/fix-null-strategy.h"
#include "nnet/nnet-component.h"

namespace kaldi {
  namespace fix {

    const struct FixStrategy::key_value FixStrategy::kMarkerMap[] = {
      { FixStrategy::kDynamicFixedPoint, "<DynamicFixedPoint>" },
      { FixStrategy::kNullStrategy, "<NullStrategy>" }
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

    std::tr1::shared_ptr<FixStrategy> FixStrategy::NewNullStrategy() {
      return std::tr1::shared_ptr<FixStrategy> (new NullStrategy());
    }

    std::tr1::shared_ptr<FixStrategy> FixStrategy::Read(std::istream &is, bool binary, kaldi::nnet1::NnetFix& nnet_fix) {
      std::string token;
     
      int first_char = Peek(is, binary);
      if (first_char == EOF) return std::tr1::shared_ptr<FixStrategy>(NewStrategyOfType(kNullStrategy));
     
      ReadToken(is, binary, &token);
      std::tr1::shared_ptr<FixStrategy> ans = std::tr1::shared_ptr<FixStrategy>(NewStrategyOfType(MarkerToType(token)));
      ans -> ReadData(is, binary, nnet_fix);
      // ExpectToken(is, binary, "<!EndOfStrategy>");
      ans -> Initialize();
      return ans;
    }

    void FixStrategy::Write(std::ostream &os, bool binary, bool config_only) const {
      WriteToken(os, binary, "<DynamicFixedPoint>");
      if (!binary) os << "\n";
      WriteData(os, binary, config_only);
      if (!binary) os << "\n";
    }

    FixStrategy* FixStrategy::NewStrategyOfType(StrategyType strategy_type) {
      FixStrategy* ans = NULL;
      switch (strategy_type) {
      case FixStrategy::kDynamicFixedPoint:
        ans = new DynamicFixedPointStrategy();
        break;
      case FixStrategy::kNullStrategy:
        ans = new NullStrategy();
        break;
      default:
        KALDI_ERR << "Missing type: " << TypeToMarker(strategy_type);
      }
      return ans;
    }

    void FixStrategy::FixParam(kaldi::nnet1::NnetFix& nnet_fix) {
      for (int32 n = 0; n < nnet_fix.NumComponents(); ++n) {
        if (nnet_fix.GetComponent(n).GetType() == Component::kLstmProjectedStreams) // only fix lstm component
	{
	  Vector<BaseFloat> vector(dynamic_cast<kaldi::nnet1::UpdatableComponent&>(nnet_fix.GetComponent(n)).NumParams());
	  dynamic_cast<kaldi::nnet1::UpdatableComponent&>(nnet_fix.GetComponent(n)).GetParams(&vector);
          this->DoFixParam(vector, nnet_fix.GetComponent(n).GetType(), n, dynamic_cast<kaldi::nnet1::UpdatableComponent&>(nnet_fix.GetComponent(n)).InnerNumParams());
          dynamic_cast<kaldi::nnet1::UpdatableComponent&>(nnet_fix.GetComponent(n)).SetParams(vector);
        }
      }
    }

  } // end namespace fix
} // end namespace kaldi
