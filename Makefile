all:
	gcc -o matrixMul matrixMul.c
	gcc -o matrixMulThreads matrixMulThreads.c -pthread
	nvcc -o matrixMulCuda matrixMulCuda.cu

clean:
	rm matrixMul matrixMulThreads matrixMulCuda
