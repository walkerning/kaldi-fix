// nnet/nnet-nnet.h

// Copyright 2011-2016  Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET_NNET_NNET_FIX_H_
#define KALDI_NNET_NNET_NNET_FIX_H_

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "fix/fix-strategy.h"

namespace kaldi {
namespace nnet1 {

class NnetFix : public Nnet{
public:
    InitFix(std::string fix_config);
    ApplyWeightFix();
    ApplyBlobFix(CuMatrix<BaseFloat> in, int32 blob_index);
    friend class FixStrategy;

private:
    shared_ptr<FixStrategy> fix_strategy_;

};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_NNET_FIX_H_


