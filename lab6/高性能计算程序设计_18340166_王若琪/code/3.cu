/******************************************
 * Compile:
 * nvcc 3.cu -o 3 -lcublas
 * Run:       
 * ./3 <M> <N> <K>
 ******************************************/

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define GET_TIME(now)                           \
{                                           \
    struct timeval t;                       \
    gettimeofday(&t, NULL);                 \
    now = t.tv_sec + t.tv_usec / 1000000.0; \
}

void init_matrix(float *mat, int r, int l)
{
    int i, j;
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < l; j++)
        {
            mat[i * l + j] = (float)(rand() % 50)/100;
        }
    }
}

void printMat(float *mat, int r, int l){
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < l; j++)
        {
            printf("%f ", mat[i * l + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv)
{
	srand(time(0));
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    //分配内存
	float *a = (float *)malloc(sizeof(float) * M * N);
	float *b = (float *)malloc(sizeof(float) * N * K);
	float *c = (float *)malloc(sizeof(float) * M * K);

    //随机生成矩阵
    init_matrix(a, M, N);
    init_matrix(b, N, K);

    double start, finish, time;
    GET_TIME(start);

	float *cuda_a, *cuda_b, *cuda_c;
    //cudaMalloc 分配空间
	cudaMalloc((void **)&cuda_a, sizeof(float) * M * N);
	cudaMalloc((void **)&cuda_b, sizeof(float) * N * K);
	cudaMalloc((void **)&cuda_c, sizeof(float) * M * K);

    //cudaMemcpy 将矩阵复制到显存中
	cudaMemcpy(cuda_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, N * K * sizeof(float), cudaMemcpyHostToDevice);

    //设置 alpha beta
	float alpha = 1;
	float beta = 0;
    //调用API
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, cuda_b, K, cuda_a, N, &beta, cuda_c, K);
    //将结果复制回内存
    cudaMemcpy(c, cuda_c, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    GET_TIME(finish);
    time = finish - start;

    printMat(a, M, N);
    printMat(b, N, K);
    printMat(c, M, K);
 
    printf("time of gemm: %f s\n", time);

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
	free(a);
	free(b);
	free(c);
	return 0;
}