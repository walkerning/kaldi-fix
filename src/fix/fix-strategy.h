#ifndef KALDI_FIX_FIX_STRATEGY_H_
#define KALDI_FIX_FIX_STRATEGY_H_

#include <iostream>
#include <memory>
#include <string>

#include "base/kaldi-common.h"
#include "nnet/nnet-component.h"
#include "util/kaldi-io.h"

namespace kaldi {
  namespace nnet1 {
    class NnetFix;
  }
}

namespace kaldi {
  namespace fix {
    class FixStrategy {
    public:
      /// Types of fix strategy
      typedef enum {
	kUnknown = 0x0,

	kDynamicFixedPoint = 0x0100
      } StrategyType;

      /// A pair of type and marker,
      struct key_value {
	const StrategyType key;
	const char *value;
      };

      // static FixStrategy* Init(const std::string &conf_line);
      /// The table with pairs of strategy types and markers
      /// (defined in fix-strategy.cc),
      static const struct key_value kMarkerMap[];

      static const char* TypeToMarker(StrategyType t);
      static StrategyType MarkerToType(const std::string &s);
      
      static std::shared_ptr<FixStrategy> Read(const std::string &rxfilename) {
	bool binary;
	Input in(rxfilename, &binary);
	std::shared_ptr<FixStrategy> strategy = Read(in.Stream(), binary);
	in.Close(); 
	return strategy;
      }

      static std::shared_ptr<FixStrategy> Read(std::istream &is, bool binary);

      /// Write the component to a stream,
      void Write(std::ostream &os, bool binary) const;

      /// Get Type Identification of the strategy type.
      virtual StrategyType GetType() const = 0;

      void FixParam(kaldi::nnet1::NnetFix& nnet_fix);
      
      void FixBlob(CuMatrixBase<BaseFloat> &blob, int n) {
	this->DoFixBlob(blob, n);
      }

      void FixBlob(MatrixBase<BaseFloat> &blob, int n) {
	this->DoFixBlob(blob, n);
      }
      
      virtual void Clear() {
	// A stub as default implementation. 
      }

    protected:
      // virtual functions to be implemented in derived classes
      virtual void ReadData(std::istream &is, bool binary) = 0;

      virtual void WriteData(std::ostream &os, bool binary) const = 0;
 
      virtual void DoFixBlob(CuMatrixBase<BaseFloat> &blob, int n) = 0;

      virtual void DoFixBlob(MatrixBase<BaseFloat> &blob, int n) = 0;

      virtual void DoFixParam(VectorBase<BaseFloat> &blob,
			      kaldi::nnet1::Component::ComponentType comp_type,
			      int n) = 0;

    private:
      static FixStrategy* NewStrategyOfType(StrategyType t);

    }; // end class FixStrategy
  } // end namespace fix
} // end namespace kaldi

#endif
