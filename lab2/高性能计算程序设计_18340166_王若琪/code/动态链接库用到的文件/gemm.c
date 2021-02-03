#include<stdlib.h>
#include "func.h"
int** gemm(int** matrixA, int** matrixB, int M, int N, int K) {   //通用矩阵乘法
	int **matrix=(int**)malloc(sizeof(int*)*M);
	for (int i=0;i<M;++i)
		matrix[i]=(int *)malloc(sizeof(int)*K);
	for (int i = 0; i < M; i++){
		for (int j = 0; j < K; j++){
			matrix[i][j] = 0;
			for (int l = 0; l < N; l++){
				matrix[i][j] += matrixA[i][l] * matrixB[l][j];
			}
		}
	}
	return matrix;
}
