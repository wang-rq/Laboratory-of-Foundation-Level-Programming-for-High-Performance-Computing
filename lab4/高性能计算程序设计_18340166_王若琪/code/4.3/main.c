#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "func.h"
#include <sys/time.h>

#define GET_TIME(now)                           \
	{                                           \
		struct timeval t;                       \
		gettimeofday(&t, NULL);                 \
		now = t.tv_sec + t.tv_usec / 1000000.0; \
	}

struct for_index {
    int start;
    int end;
    int increment;
};

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

void * functor_gemm (void *args){
    struct for_index * index = (struct for_index *) args;
    for (int i = index->start; i < index->end; i = i + index->increment){
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

	matrixC = (int **)malloc(sizeof(int *) * M);
	for (int i = 0; i < M; ++i)
		matrixC[i] = (int *)malloc(sizeof(int) * K);

	matrixA = initMatrix(M, N);
	matrixB = initMatrix(N, K);

	double start, finish, time;
	GET_TIME(start);
	parallel_for(0, M, 1, functor_gemm, NULL, ThreadNumber); //并行矩阵乘法改造
	GET_TIME(finish);
	time = finish - start;

	printMat(matrixA, M, N);
	printMat(matrixB, N, K);
	printMat(matrixC, M, K);
	printf("time of gemm: %f s\n", time);

	for (int i = 0; i < M; ++i)
		free(matrixA[i]);
	free(matrixA);
	for (int i = 0; i < N; ++i)
		free(matrixB[i]);
	free(matrixB);
	for (int i = 0; i < M; ++i)
		free(matrixC[i]);
	free(matrixC);

	return 0;
}
