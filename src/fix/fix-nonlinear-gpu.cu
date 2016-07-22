#include "fix/fix-nonlinear-kernels-ansi.h"

namespace kaldi {
  namespace fix {
    template <typename Real>
    __global__ void _mapping(const int count, Real* in, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int bit_num, const int num_p, const float LB, const float UB, float amp) 
    {
      for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) 
        {
          int order =  (int)((in[index] / x_rang + 1) * amp  + 0.5);
          //in[index] = (int)((in[index] / x_rang + 1) * amp  + 0.5);
          if (order  < x_arraybin[0])
            in[index]=LB;
          else if (order >= x_arraybin[num_p])
            in[index]=UB;
          else
            {
              for (int i = 0; i < num_p; i++)
                {
                  if (x_arraybin[i] <= order && order < x_arraybin[i + 1])
                    {
                      in[index] = Real(((x_arraybin[i+1]-order)*y_arraybin[i]+(order-x_arraybin[i])*y_arraybin[i+1])/(x_arraybin[i+1]-x_arraybin[i])/amp);
                      break;
                    }
                }
            }
        }
    }

    void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, const int count, float* data, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int bit_num, const int num_p, const float LB, const float UB, float amp) {
      _mapping <<<dimGrid, dimBlock>>>(count, data, x_rang, x_arraybin, y_arraybin, bit_num, num_p, LB, UB, amp);
    }
    void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, const int count, double* data, const float x_rang, const int * x_arraybin, const int * y_arraybin, const int bit_num, const int num_p, const float LB, const float UB, float amp) {
      _mapping <<<dimGrid, dimBlock>>>(count, data, x_rang, x_arraybin, y_arraybin, bit_num, num_p, LB, UB, amp);
    }
  }
}
