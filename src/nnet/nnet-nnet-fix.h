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
#include <memory>
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

class NnetFix {
 public:
  NnetFix();
  ~NnetFix();

  NnetFix(const NnetFix& other);  // Allow copy constructor.
  NnetFix& operator= (const NnetFix& other);  // Allow assignment operator.

 public:
  /// For Nnet Fix
  void InitFix(std::string fix_config);
  void InitFixLine(std::string fix_config_line);
  void ApplyWeightFix();
  void ApplyBlobFix(CuMatrix<BaseFloat> in, int32 blob_index);
  friend class fix::FixStrategy;

 private:
  /// For Nnet Fix strategy
  std::tr1::shared_ptr<fix::FixStrategy> fix_strategy_;

 public:
  /// Perform forward pass through the network,
  void Propagate(const CuMatrixBase<BaseFloat> &in,
                 CuMatrix<BaseFloat> *out);
  /// Perform backward pass through the network,
  void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff,
                     CuMatrix<BaseFloat> *in_diff);
  /// Perform forward pass through the network (with 2 swapping buffers),
  void Feedforward(const CuMatrixBase<BaseFloat> &in,
                   CuMatrix<BaseFloat> *out);

  /// Dimensionality on network input (input feature dim.),
  int32 InputDim() const;
  /// Dimensionality of network outputs (posteriors | bn-features | etc.),
  int32 OutputDim() const;

  /// Returns the number of 'Components' which form the NN.
  /// Typically a NN layer is composed of 2 components:
  /// the <AffineTransform> with trainable parameters
  /// and a non-linearity like <Sigmoid> or <Softmax>.
  /// Usually there are 2x more Components than the NN layers.
  int32 NumComponents() const {
    return components_.size();
  }

  /// Component accessor,
  const Component& GetComponent(int32 c) const;

  /// Component accessor,
  Component& GetComponent(int32 c);

  /// LastComponent accessor,
  const Component& GetLastComponent() const;

  /// LastComponent accessor,
  Component& GetLastComponent();

  /// Replace c'th component in 'this' NnetFix (deep copy),
  void ReplaceComponent(int32 c, const Component& comp);

  /// Swap c'th component with the pointer,
  void SwapComponent(int32 c, Component** comp);

  /// Append Component to 'this' instance of NnetFix (deep copy),
  void AppendComponent(const Component& comp);

  /// Append Component* to 'this' instance of NnetFix by a shallow copy
  /// ('this' instance of NnetFix over-takes the ownership of the pointer).
  void AppendComponentPointer(Component *dynamically_allocated_comp);

  /// Append other NnetFix to the 'this' NnetFix (copy all its components),
  void AppendNnet(const NnetFix& nnet_to_append);

  /// Remove c'th component,
  void RemoveComponent(int32 c);

  /// Remove the last of the Components,
  void RemoveLastComponent();

  /// Access to the forward-pass buffers
  const std::vector<CuMatrix<BaseFloat> >& PropagateBuffer() const {
    return propagate_buf_;
  }
  /// Access to the backward-pass buffers
  const std::vector<CuMatrix<BaseFloat> >& BackpropagateBuffer() const {
    return backpropagate_buf_;
  }

  /// Get the number of parameters in the network,
  int32 NumParams() const;

  /// Get the gradient stored in the network,
  void GetGradient(Vector<BaseFloat>* gradient) const;

  /// Get the network weights in a supervector,
  void GetParams(Vector<BaseFloat>* params) const;

  /// Set the network weights from a supervector,
  void SetParams(const VectorBase<BaseFloat>& params);

  /// Set the dropout rate
  void SetDropoutRetention(BaseFloat r);

  /// Reset streams in LSTM multi-stream training,
  void ResetLstmStreams(const std::vector<int32> &stream_reset_flag);

  /// Set sequence length in LSTM multi-stream training,
  void SetSeqLengths(const std::vector<int32> &sequence_lengths);

  /// Initialize the NnetFix from the prototype,
  void Init(const std::string &proto_file);

  /// Read NnetFix from 'rxfilename',
  void Read(const std::string &rxfilename);
  /// Read NnetFix from 'istream',
  void Read(std::istream &in, bool binary);

  /// Write NnetFix to 'wxfilename',
  void Write(const std::string &wxfilename, bool binary) const;
  /// Write NnetFix to 'ostream',
  void Write(std::ostream &out, bool binary) const;

  /// Create string with human readable description of the nnet,
  std::string Info() const;
  /// Create string with per-component gradient statistics,
  std::string InfoGradient(bool header = true) const;
  /// Create string with propagation-buffer statistics,
  std::string InfoPropagate(bool header = true) const;
  /// Create string with back-propagation-buffer statistics,
  std::string InfoBackPropagate(bool header = true) const;
  /// Consistency check,
  void Check() const;
  /// Relese the memory,
  void Destroy();

  /// Set hyper-parameters of the training (pushes to all UpdatableComponents),
  void SetTrainOptions(const NnetTrainOptions& opts);
  /// Get training hyper-parameters from the network,
  const NnetTrainOptions& GetTrainOptions() const {
    return opts_;
  }

 private:
  /// Vector which contains all the components composing the neural network,
  /// the components are for example: AffineTransform, Sigmoid, Softmax
  std::vector<Component*> components_;

  /// Buffers for forward pass (on demand initialization),
  std::vector<CuMatrix<BaseFloat> > propagate_buf_;
  /// Buffers for backward pass (on demand initialization),
  std::vector<CuMatrix<BaseFloat> > backpropagate_buf_;

  /// Option class with hyper-parameters passed to UpdatableComponent(s)
  NnetTrainOptions opts_;
};

  }  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_NNET_FIX_H_


