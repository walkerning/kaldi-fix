#include "fix/fix-kernels-ansi.h"

namespace kaldi {
  namespace fix {
    template <typename Real>
    __global__ static void _saturate(Real* in, const int maxnum,
                                     const int minnum, MatrixDim d) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j * d.stride;
      if (i < d.cols && j < d.rows) {
        if (in[index] > maxnum) {
          in[index] = maxnum;
        } else if (in[index] < minnum) {
          in[index] = minnum;
        }
        in[index] = Real(int(in[index]));
      }
    }

    void cuda_saturate(const dim3 dimGrid, const dim3 dimBlock, float* data, const float maxnum, const float minnum, MatrixDim d) {
      _saturate <<<dimGrid, dimBlock>>>(data, maxnum, minnum, d);
    }
    void cuda_saturate(const dim3 dimGrid, const dim3 dimBlock, double* data, const double maxnum, const double minnum, MatrixDim d) {
      _saturate <<<dimGrid, dimBlock>>>(data, maxnum, minnum, d);
    }
  }
}
