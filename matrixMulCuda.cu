#include "matrixMul.h"
#include "matrixMulCuda.h"

__global__ void matrixMul(float *d_A, float *d_B, float *d_C, int n) {
  int posx, posy, pos;
  int k;

  posx = blockDim.x * blockIdx.x + threadIdx.x;
  posy = blockDim.y * blockIdx.y + threadIdx.y;
  pos = posy * n + posx;

  if (posx < n && posy < n) {
      d_C[pos] = 0.0;
      for (k=0; k<n; k++) {
          d_C[pos] += d_A[posy*n + k] * d_B[k*n + posx];
      }
  }
}

int main(int argc, char **argv) {
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  unsigned long long usecstart, usecstop;

  parameters(argc, argv);
  allocate_matrices();
  allocate_matrices_device();
  initialize_inputs();

  /* Threads' geometry */
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 dimGrid(ceil((1.0*N)/(1.0*BLOCK_SIZE)), ceil((1.0*N)/(1.0*BLOCK_SIZE)), 1); 

  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);

  copy_matrices_to_device();
  printf("Computing Parallely.\n");
  matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
  copy_result_from_device();

  gettimeofday(&etstop, &tzdummy);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  free_matrices();
  free_matrices_device();

  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);
  printf("--------------------------------------------\n"); 
  exit(0);
}

