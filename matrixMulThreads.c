#include "matrixMul.h"

void *matrixMul(void *params) {
  int i, j, k;
  int thread_id = (int)params;

  for (i=thread_id; i < N; i+=NUM_THREADS) {
    for (j=0; j<N; j++) {
      C[i*N + j] = 0.0;
      for (k=0; k<N; k++) {
        C[i*N + j] += A[i*N + k] * B[k*N + j];
      }
    }
  }
}

int main(int argc, char **argv) {
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  unsigned long long usecstart, usecstop;

  srand(time_seed()); /* Randomize matrix elements */
  parameters(argc, argv);
  allocate_matrices();
  initialize_inputs();

  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);

  printf("Computing Parallely.\n");
  start_threads(matrixMul);
  join_threads();

  gettimeofday(&etstop, &tzdummy);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  free_matrices();

  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);
  printf("--------------------------------------------\n");  
  exit(0);
}

