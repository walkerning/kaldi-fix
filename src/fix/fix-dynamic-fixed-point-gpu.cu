#include "fix/fix-kernels-ansi.h"

namespace kaldi {
  namespace fix {
    template <typename Real>
    __global__ void _saturate(const int count, Real* in,
                              const int maxnum, const int minnum) {
      for (int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < count; index += blockDim.x * gridDim.x) {
        if (in[index] > maxnum) {
          in[index] = maxnum;
        } else if (in[index] < minnum) {
          in[index] = minnum;
        }
        in[index] = Real(int(in[index]));
      }
    }

    void cuda_saturate(const dim3 dimGrid, const dim3 dimBlock, const int count, float* data, const float maxnum, const float minnum) {
      _saturate <<<dimGrid, dimBlock>>>(count, data, maxnum, minnum);
    }
    void cuda_saturate(const dim3 dimGrid, const dim3 dimBlock, const int count, double* data, const double maxnum, const double minnum) {
      _saturate <<<dimGrid, dimBlock>>>(count, data, maxnum, minnum);
    }
  }
}
