#include "fix/fix-nonlinear-kernels-ansi.h"

namespace kaldi {
  namespace fix {
    template <typename Real>
    __global__ void _mapping(Real* in, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int num_p, const float LB, const float UB, float amp, MatrixDim d) 
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j * d.stride;
      if (i < d.cols && j < d.rows) {
        int order =  (int)((in[index] / x_rang + 1) * amp  + 0.5);
        if (order < x_arraybin[0])
          in[index]=LB;
        else if (order >= x_arraybin[num_p])
          in[index]=UB;
        else {
          for (int i = 0; i < num_p; i++) {
            if (x_arraybin[i] <= order && order < x_arraybin[i + 1]) {
              in[index] = Real(((x_arraybin[i+1]-order)*y_arraybin[i]+(order-x_arraybin[i])*y_arraybin[i+1])/(x_arraybin[i+1]-x_arraybin[i])/amp);
              break;
            }
          }
        }
      }
    }

    void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, float* data, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int num_p, const float LB, const float UB, float amp, MatrixDim d) {
      _mapping <<<dimGrid, dimBlock>>>(data, x_rang, x_arraybin, y_arraybin, num_p, LB, UB, amp, d);
    }
    void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, double* data, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int num_p, const float LB, const float UB, float amp, MatrixDim d) {
      _mapping <<<dimGrid, dimBlock>>>(data, x_rang, x_arraybin, y_arraybin, num_p, LB, UB, amp, d);
    }
  }
}
