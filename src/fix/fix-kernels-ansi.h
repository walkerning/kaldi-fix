#include "cudamatrix/cu-matrixdim.h"

namespace kaldi {
  namespace fix {
    void cuda_saturate(const dim3 dimGrid, const dim3 dimBlock, float* data,
                       float maxnum, float minnum, MatrixDim d);
    void cuda_saturate(const dim3 dimGrid, const dim3 dimBlock, double* data,
                       double maxnum, double minnum, MatrixDim d);
  }
}
