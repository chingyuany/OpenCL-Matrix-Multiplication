__kernel void mat_mul(const int N, __global float *inputA,
                      __global float *inputB, __global float *output,
                      __local float *Bwrk) {
  int j, k;
  int i = get_global_id(0);
  int iloc = get_local_id(0);
  int nloc = get_local_size(0);
  float tmp;
  float Awrk[1024];
  for (k = 0; k < N; k++) {
    Awrk[k] = inputA[i * N + k];
  }
  for (j = 0; j < N; j++) {
    for (k = iloc; k < N; k += nloc) {
      Bwrk[k] = inputB[k * N + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    tmp = 0.0f;
    for (k = 0; k < N; k++) {
      tmp += Awrk[k] * Bwrk[k];
    }
    output[i * N + j] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
