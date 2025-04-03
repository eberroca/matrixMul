#ifndef _MATRIXMULCUDA_H_
#define _MATRIXMULCUDA_H_

/* Program parameters */
#define BLOCK_SIZE 32 /* 32*32 = 1024 treads per block */

/* Program Variables */
cudaError_t err;
float *d_A, *d_B, *d_C;

#define CHECK_ERR(x)                                    \
  if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
            cudaGetErrorString(err),__FILE__,__LINE__); \
    exit(-1);                                           \
  }                                                     \


void allocate_matrices_device() {
  /* mem alloc in device */
  err = cudaMalloc ((void **) &d_A, sizeof(float)*N*N);
  CHECK_ERR(err);

  err = cudaMalloc ((void **) &d_B, sizeof(float)*N*N);
  CHECK_ERR(err);

  err = cudaMalloc ((void **) &d_C, sizeof(float)*N*N);
  CHECK_ERR(err);
}

void free_matrices_device() {
  err = cudaFree(d_A);
  CHECK_ERR(err);  
  err = cudaFree(d_B);
  CHECK_ERR(err);
  err = cudaFree(d_C);
  CHECK_ERR(err);
}

void copy_matrices_to_device() {
  err = cudaMemcpy(d_A, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  CHECK_ERR(err);
  err = cudaMemcpy(d_B, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  CHECK_ERR(err);
}

void copy_matrices_to_device2() {
  
}

void copy_result_from_device() {
  err = cudaMemcpy(C, d_C, sizeof(float)*N*N, cudaMemcpyDeviceToHost); 
  CHECK_ERR(err);
}

#endif
