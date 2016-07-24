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
        int pos = (int)((in[src_index] / (2*x_rang) + 0.5)*num_p);
        int order =  (int)((in[src_index] / x_rang + 1) * amp  + 0.5);
        if (pos < 0)
          out[dst_index]=LB;
        else if (pos >= num_p)
          out[dst_index]=UB;
        else {
          out[dst_index] = Real(((x_arraybin[pos+1]-order)*y_arraybin[pos]+(order-x_arraybin[pos])*y_arraybin[pos+1])/(x_arraybin[pos+1]-x_arraybin[pos])/amp);
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
