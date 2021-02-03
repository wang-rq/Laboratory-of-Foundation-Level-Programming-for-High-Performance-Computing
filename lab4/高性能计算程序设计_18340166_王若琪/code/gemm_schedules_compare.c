/******************************************
 * Compile:
 * gcc -o gemm_schedules_compare gemm_schedules_compare.c -fopenmp
 * Run:       
 * ./gemm_schedules_compare <number of threads>
 ******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include <sys/time.h>

#define GET_TIME(now)                           \
	{                                           \
		struct timeval t;                       \
		gettimeofday(&t, NULL);                 \
		now = t.tv_sec + t.tv_usec / 1000000.0; \
	}

int M;
int N;
int K;
int ThreadNumber;
int** matrixA;
int** matrixB;
int** matrixC;

int **initMatrix(int r, int c)
{ //初始化矩阵
	int **temp = (int **)malloc(sizeof(int *) * r);
	for (int i = 0; i < r; ++i)
		temp[i] = (int *)malloc(sizeof(int) * c);

	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
		{
			temp[i][j] = rand() % 50;
		}
	}
	return temp;
}

void gemm_parallel_default()
{ //OpenMP并行矩阵乘法，默认调度方式
#pragma omp parallel for num_threads(ThreadNumber)
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < K; j++)
		{
			matrixC[i][j] = 0;
			for (int l = 0; l < N; l++)
			{
				matrixC[i][j] += matrixA[i][l] * matrixB[l][j];
			}
		}
	}
}

void gemm_parallel_static()
{ //OpenMP并行矩阵乘法，stactic调度方式
#pragma omp parallel for num_threads(ThreadNumber)\
schedule(static, 1)
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < K; j++)
		{
			matrixC[i][j] = 0;
			for (int l = 0; l < N; l++)
			{
				matrixC[i][j] += matrixA[i][l] * matrixB[l][j];
			}
		}
	}
}

void gemm_parallel_dynamic()
{ //OpenMP并行矩阵乘法，dynamic调度方式
#pragma omp parallel for num_threads(ThreadNumber)\
schedule(dynamic, 1)
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < K; j++)
		{
			matrixC[i][j] = 0;
			for (int l = 0; l < N; l++)
			{
				matrixC[i][j] += matrixA[i][l] * matrixB[l][j];
			}
		}
	}
}

void printMat(int **mat, int r, int c)
{ //输出矩阵
	for (int i = 0; i < r; ++i)
	{
		for (int j = 0; j < c; ++j)
		{
			printf("%d ", mat[i][j]);
			if (j == c - 1)
				printf("\n");
		}
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	ThreadNumber = strtol(argv[1], NULL, 10);

	printf("Please enter 3 integers (512~2048) :\n");
	scanf("%d", &M);
	scanf("%d", &N);
	scanf("%d", &K);

	//矩阵分配空间
	matrixC = (int **)malloc(sizeof(int *) * M);
	for (int i = 0; i < M; ++i)
		matrixC[i] = (int *)malloc(sizeof(int) * K);

	matrixA = initMatrix(M, N);
	matrixB = initMatrix(N, K);

	double start, finish, time1, time2, time3;
	GET_TIME(start);
	gemm_parallel_default();
	GET_TIME(finish);
	time1 = finish - start;

	GET_TIME(start);
	gemm_parallel_static();
	GET_TIME(finish);
	time2 = finish - start;

	GET_TIME(start);
	gemm_parallel_dynamic();
	GET_TIME(finish);
	time3 = finish - start;

	printMat(matrixA, M, N);
	printMat(matrixB, N, K);
	printMat(matrixC, M, K);

	printf("time of default schedule: %f s\n", time1);
	printf("time of static schedule:  %f s\n", time2);
	printf("time of dynamic schedule: %f s\n", time3);

	for (int i = 0; i < M; ++i)
		free(matrixA[i]);
	free(matrixA);
	for (int i = 0; i < N; ++i)
		free(matrixB[i]);
	free(matrixB);

	for (int i = 0; i < M; ++i){
		free(matrixC[i]);
	}	
	free(matrixC);
	return 0;
}
