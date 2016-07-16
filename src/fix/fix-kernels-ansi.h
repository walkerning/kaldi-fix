namespace kaldi {
  namespace fix {
    void cuda_saturate(const dim3 dimGrid, const dim3 dimBlock, const int count, 
                       float* data, float maxnum, float minnum);
    void cuda_saturate(const dim3 dimGrid, const dim3 dimBlock, const int count, 
                       double* data, double maxnum, double minnum);
  }
}
