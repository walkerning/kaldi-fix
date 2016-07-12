// nnet/nnet-nnet.cc

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

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-parallel-component.h"
#include "nnet/nnet-multibasis-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-various.h"
#include "nnet/nnet-lstm-projected-streams.h"
#include "nnet/nnet-blstm-projected-streams.h"

namespace kaldi {
namespace nnet1 {


void NnetFix::InitFix(std::string fix_config){
    if (fix_config != ""){
        fix_strategy_ = FixStrategy::Read();
    }
    else
    {
        fix_strategy_ = NULL;
    }
}

void NnetFix::ApplyWeightFix(){
    if (fix_strategy_ != NULL){
        self->fix_strategy_->FixWeight(*this);
    }
}

void NnetFix::ApplyBlobFix(CuMatrix<Basefloat> in, int32 blob_index){
    if (fix_strategy_ != NULL){
        fix_strategy_->FixBlob(in, blob_index);
    }
}

/**
 * Forward propagation through the network,
 * (from first component to last).
 */
void NnetFix::Propagate(const CuMatrixBase<BaseFloat> &in,
                     CuMatrix<BaseFloat> *out) {
  // In case of empty network copy input to output,
  if (NumComponents() == 0) {
    (*out) = in;  // copy,
    return;
  }
  // We need C+1 buffers,
  if (propagate_buf_.size() != NumComponents()+1) {
    propagate_buf_.resize(NumComponents()+1);
  }
  // Copy input to first buffer,
  propagate_buf_[0] = in;
  // Propagate through all the components,
  for (int32 i = 0; i < static_cast<int32>(components_.size()); i++) {
    ApplyBlobFix(propagate_buf_[i], i);
    components_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
  }
  ApplyBlobFix(propagate_buf_[NumComponents()], NumComponents());
  // Copy the output from the last buffer,
  (*out) = propagate_buf_[NumComponents()];
}


/**
 * Error back-propagation through the network,
 * (from last component to first).
 */
void NnetFix::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff,
                         CuMatrix<BaseFloat> *in_diff) {
  // Copy the derivative in case of empty network,
  if (NumComponents() == 0) {
    (*in_diff) = out_diff;  // copy,
    return;
  }
  // We need C+1 buffers,
  KALDI_ASSERT(static_cast<int32>(propagate_buf_.size()) == NumComponents()+1);
  if (backpropagate_buf_.size() != NumComponents()+1) {
    backpropagate_buf_.resize(NumComponents()+1);
  }
  // Copy 'out_diff' to last buffer,
  backpropagate_buf_[NumComponents()] = out_diff;
  // Loop from last Component to the first,
  for (int32 i = NumComponents()-1; i >= 0; i--) {
    // Backpropagate through 'Component',
    components_[i]->Backpropagate(propagate_buf_[i],
                                  propagate_buf_[i+1],
                                  backpropagate_buf_[i+1],
                                  &backpropagate_buf_[i]);
    // Update 'Component' (if applicable),
    if (components_[i]->IsUpdatable()) {
      UpdatableComponent* uc =
        dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->Update(propagate_buf_[i], backpropagate_buf_[i+1]);
    }
  }
  // Export the derivative (if applicable),
  if (NULL != in_diff) {
    (*in_diff) = backpropagate_buf_[0];
  }
}


void NnetFix::Feedforward(const CuMatrixBase<BaseFloat> &in,
                       CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);
  (*out) = in;  // works even with 0 components,
  CuMatrix<BaseFloat> tmp_in;
  for (int32 i = 0; i < NumComponents(); i++) {
    out->Swap(&tmp_in);
    // Apply Fix Blob
    ApplyBlobFix(tmp_in, i);
    components_[i]->Propagate(tmp_in, out);
  }
  ApplyBlobFix(*out, NumComponents());

}


/*
std::string Nnet::Info() const {
  // global info
  std::ostringstream ostr;
  ostr << "num-components " << NumComponents() << std::endl;
  if (NumComponents() == 0)
    return ostr.str();
  ostr << "input-dim " << InputDim() << std::endl;
  ostr << "output-dim " << OutputDim() << std::endl;
  ostr << "number-of-parameters " << static_cast<float>(NumParams())/1e6
       << " millions" << std::endl;
  // topology & weight stats
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "component " << i+1 << " : "
         << Component::TypeToMarker(components_[i]->GetType())
         << ", input-dim " << components_[i]->InputDim()
         << ", output-dim " << components_[i]->OutputDim()
         << ", " << components_[i]->Info() << std::endl;
  }
  return ostr.str();
}


std::string Nnet::InfoPropagate(bool header) const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  if (header) ostr << "\n### FORWARD PROPAGATION BUFFER CONTENT :\n";
  ostr << "[0] output of <Input> " << MomentStatistics(propagate_buf_[0])
       << std::endl;
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "[" << 1+i << "] output of "
         << Component::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(propagate_buf_[i+1]) << std::endl;
    // nested networks too...
    if (Component::kParallelComponent == components_[i]->GetType()) {
      ostr <<
        dynamic_cast<ParallelComponent*>(components_[i])->InfoPropagate();
    }
    if (Component::kMultiBasisComponent == components_[i]->GetType()) {
      ostr << dynamic_cast<MultiBasisComponent*>(components_[i])->InfoPropagate();
    }
  }
  if (header) ostr << "### END FORWARD\n";
  return ostr.str();
}
*/

}  // namespace nnet1
}  // namespace kaldi
