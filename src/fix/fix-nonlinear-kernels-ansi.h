namespace kaldi {
  namespace fix {
	  void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, const int count, float* data, const float x_rang, const int * x_arraybin,
                            const int * y_arraybin, const int bit_num, const int num_p, const float LB, const float UB, float amp);
	  void cuda_mapping(const dim3 dimGrid, const dim3 dimBlock, const int count, double* data, const float x_rang, const int * x_arraybin,
                            const int * y_arraybin, const int bit_num, const int num_p, const float LB, const float UB, float amp);
  }
}
