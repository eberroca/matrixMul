#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>


/* Program Parameters */
#define NUM_THREADS 4
int N;  /* Matrix size */

/* Program Variables */
float *A, *B, *C;
pthread_t threads[NUM_THREADS];

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  /* Read command-line arguments */
  if (argc == 2) {
    N = atoi(argv[1]);
    if (N < 1) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension>\n", argv[0]);
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row*N + col] = (float)rand() / 67108864.0;
      B[row*N + col] = (float)rand() / 67108864.0;
      C[row*N + col] = 0.0;
    }
  }

}

void allocate_matrices() {
  A = (float *) malloc(sizeof(int)*N*N);
  B = (float *) malloc(sizeof(int)*N*N);
  C = (float *) malloc(sizeof(int)*N*N);
  if (!A || !B || !C)
    exit(-1);
}	

void free_matrices() {
  free(A);
  free(B);
  free(C);
}

void start_threads(void *(*function)(void*)) {
  int i;
  for (i = 0; i < NUM_THREADS; i++) {
    pthread_create(&threads[i], NULL, function, (void *)i);
  }
}

void join_threads() {
  int i;
  for (i = 0; i < NUM_THREADS; i++)
    pthread_join(threads[i], NULL);
}

#endif
