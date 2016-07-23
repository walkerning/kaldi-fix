#include "fix/fix-nonlinear-kernels-ansi.h"

namespace kaldi {
  namespace fix {
    template <typename Real>
    __global__ void _mapping(Real* out, const Real* in, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int num_p, const float LB, const float UB, float amp, MatrixDim d, int src_stride) 
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int dst_index = i + j * d.stride;
      int src_index = i + j * src_stride;
      if (i < d.cols && j < d.rows) {
        int order =  (int)((in[src_index] / x_rang + 1) * amp  + 0.5);
        if (order < x_arraybin[0])
          out[dst_index]=LB;
        else if (order >= x_arraybin[num_p])
          out[dst_index]=UB;
        else {
          for (int i = 0; i < num_p; i++) {
            if (x_arraybin[i] <= order && order < x_arraybin[i + 1]) {
              out[dst_index] = Real(((x_arraybin[i+1]-order)*y_arraybin[i]+(order-x_arraybin[i])*y_arraybin[i+1])/(x_arraybin[i+1]-x_arraybin[i])/amp);
              break;
            }
          }
        }
      }
    }

    void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, float* data, const float* in, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int num_p, const float LB, const float UB, float amp, MatrixDim d, int src_stride) {
      _mapping <<<dimGrid, dimBlock>>>(data, in, x_rang, x_arraybin, y_arraybin, num_p, LB, UB, amp, d, src_stride);
    }
    void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, double* data, const double* in, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int num_p, const float LB, const float UB, float amp, MatrixDim d, int src_stride) {
      _mapping <<<dimGrid, dimBlock>>>(data, in, x_rang, x_arraybin, y_arraybin, num_p, LB, UB, amp, d, src_stride);
    }
  }
}
