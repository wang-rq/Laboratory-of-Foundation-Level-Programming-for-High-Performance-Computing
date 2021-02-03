#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int M;
int N;
int K;

int** initMatrix(int r, int c) {  //初始化矩阵
	int **temp=(int**)malloc(sizeof(int*)*r);
	for (int i=0;i<r;++i)
		temp[i]=(int *)malloc(sizeof(int)*c);

	for(int i = 0; i < r; i++){
		for (int j = 0; j < c; j++){
			temp[i][j] = rand()%50;
		}
	}
	return temp;
}


int** gemm(int** matrixA, int** matrixB) {   //通用矩阵乘法
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

void printMat(int** mat, int r, int c){  //输出矩阵
	for(int i=0;i<r;++i){
		for(int j=0;j<c;++j){
			printf("%d ",mat[i][j]);
			if(j==c-1) printf("\n");
		}
	}
	printf("\n");
}

int main(){
	printf("Please enter 3 integers (512~2048) :\n");
	scanf("%d",&M);
	scanf("%d",&N);
	scanf("%d",&K);
	
	int** matrixA = initMatrix(M,N);
	int** matrixB = initMatrix(N,K);
    
	clock_t begin, end;
	begin=clock();
	int** matrixC = gemm(matrixA, matrixB);
	end=clock();
	double time=(double)(end-begin)/CLOCKS_PER_SEC;
    
	printMat(matrixA,M,N);
	printMat(matrixB,N,K);
	printMat(matrixC,M,K);
	printf("time of gemm: %f s\n",time);
    
	return 0;
}

