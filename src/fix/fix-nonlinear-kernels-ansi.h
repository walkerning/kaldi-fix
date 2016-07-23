#include "cudamatrix/cu-matrixdim.h"

namespace kaldi {
  namespace fix {
	  void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock,float* data, const float x_rang, const int * x_arraybin,
                            const int * y_arraybin, const int num_p, const float LB, const float UB, float amp, MatrixDim d);
	  void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, double* data, const float x_rang, const int * x_arraybin,
                            const int * y_arraybin, const int num_p, const float LB, const float UB, float amp, MatrixDim d);
  }
}
