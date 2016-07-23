#ifndef KALDI_FIX_FIX_STRATEGY_H_
#define KALDI_FIX_FIX_STRATEGY_H_

#include <iostream>
#include <tr1/memory>
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

        kNullStrategy = 0x100,

        kDynamicFixedPoint = 0x0200
      } StrategyType;

      /// A pair of type and marker,
      struct key_value {
        const StrategyType key;
        const char *value;
      };

      /// The table with pairs of strategy types and markers
      /// (defined in fix-strategy.cc),
      static const struct key_value kMarkerMap[];

      static const char* TypeToMarker(StrategyType t);
      static StrategyType MarkerToType(const std::string &s);
     
      static std::tr1::shared_ptr<FixStrategy> Read(const std::string &rxfilename) {
        bool binary;
        Input in(rxfilename, &binary);
        std::tr1::shared_ptr<FixStrategy> strategy = Read(in.Stream(), binary);
        in.Close(); 
        return strategy;
      }

      static std::tr1::shared_ptr<FixStrategy> Init(const std::string &conf_line) {
        bool binary = false;
        std::istringstream is(conf_line);
        std::tr1::shared_ptr<FixStrategy> strategy = Read(is, binary);
        return strategy;
      }

      static std::tr1::shared_ptr<FixStrategy> Read(std::istream &is, bool binary);

      /// Write the component to a stream,
      void Write(std::ostream &os, bool binary, bool config_only=false) const;

      /// Get Type Identification of the strategy type.
      virtual StrategyType GetType() const = 0;

      void FixParam(kaldi::nnet1::NnetFix& nnet_fix);
     
      void FixBlob(CuMatrixBase<BaseFloat> &blob, int n) {
        this->DoFixBlob(blob, n);
      }

      void FixBlob(MatrixBase<BaseFloat> &blob, int n) {
        this->DoFixBlob(blob, n);
      }

      void FixSigm(CuMatrixBase<BaseFloat> &blob, 
                   const CuMatrixBase<BaseFloat> &in,
                   int n)
      {
        this->DoFixSigm(blob, in, n);
      }

      void FixTanh(CuMatrixBase<BaseFloat> &blob, 
                   const CuMatrixBase<BaseFloat> &in,
                   int n)
      {
        this->DoFixTanh(blob, in, n);
      }
     
      virtual void Clear() {
        // A stub as default implementation. 
      }

      static std::tr1::shared_ptr<FixStrategy> NewNullStrategy();

    protected:
      // virtual functions to be implemented in derived classes
      virtual void ReadData(std::istream &is, bool binary) = 0;

      virtual void WriteData(std::ostream &os, bool binary, bool config_only=false) const = 0;

      virtual void DoFixBlob(CuMatrixBase<BaseFloat> &blob, int n) {}

      virtual void DoFixBlob(MatrixBase<BaseFloat> &blob, int n) = 0;

      virtual void DoFixParam(VectorBase<BaseFloat> &blob,
                              kaldi::nnet1::Component::ComponentType comp_type,
                              int n) = 0;

      virtual void DoFixSigm(CuMatrixBase<BaseFloat> &blob,
                             const CuMatrixBase<BaseFloat> &in,
                             int n) = 0;
	  
      virtual void DoFixTanh(CuMatrixBase<BaseFloat> &blob,
                             const CuMatrixBase<BaseFloat> &in,
                             int n) = 0;

      virtual void Initialize() {}

    private:
      static FixStrategy* NewStrategyOfType(StrategyType t);

    }; // end class FixStrategy
  } // end namespace fix
} // end namespace kaldi

#endif
