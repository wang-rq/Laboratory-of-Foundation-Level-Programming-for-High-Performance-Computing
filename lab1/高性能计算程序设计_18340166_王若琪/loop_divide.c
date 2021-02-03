#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int M;
int N;
int K;


int** loop_divide_1(int** matrixA, int** matrixB) {   //第一种拆分方法，K维度上进行拆分展开
	int **matrix=(int**)malloc(sizeof(int*)*M);
	for (int i=0;i<M;++i)
		matrix[i]=(int *)malloc(sizeof(int)*K);
	for (int i = 0; i < M; i++){
		for (int j = 0; j < K; j+=4){
			matrix[i][j+0] = 0;
			matrix[i][j+1] = 0;
			matrix[i][j+2] = 0;
			matrix[i][j+3] = 0;
			for (int l = 0; l < N; l++){
				matrix[i][j+0] += matrixA[i][l] * matrixB[l][j+0];
				matrix[i][j+1] += matrixA[i][l] * matrixB[l][j+1];
				matrix[i][j+2] += matrixA[i][l] * matrixB[l][j+2];
				matrix[i][j+3] += matrixA[i][l] * matrixB[l][j+3];
			}
		}
	}
	return matrix;
}


int** loop_divide_2(int** matrixA, int** matrixB) {   //第二种拆分方法
	int **matrix=(int**)malloc(sizeof(int*)*M);
	for (int i=0;i<M;++i)
		matrix[i]=(int *)malloc(sizeof(int)*K);
	for (int i = 0; i < M; i+=4){
		for (int j = 0; j < K; j+=4){
			matrix[i+0][j+0] = 0;
			matrix[i+0][j+1] = 0;
			matrix[i+0][j+2] = 0;
			matrix[i+0][j+3] = 0;
			matrix[i+1][j+0] = 0;
			matrix[i+1][j+1] = 0;
			matrix[i+1][j+2] = 0;
			matrix[i+1][j+3] = 0;
			matrix[i+2][j+0] = 0;
			matrix[i+2][j+1] = 0;
			matrix[i+2][j+2] = 0;
			matrix[i+2][j+3] = 0;
			matrix[i+3][j+0] = 0;
			matrix[i+3][j+1] = 0;
			matrix[i+3][j+2] = 0;
			matrix[i+3][j+3] = 0;
			for (int l = 0; l < N; l++){
				matrix[i+0][j+0] += matrixA[i+0][l] * matrixB[l][j+0];
				matrix[i+0][j+1] += matrixA[i+0][l] * matrixB[l][j+1];
				matrix[i+0][j+2] += matrixA[i+0][l] * matrixB[l][j+2];
				matrix[i+0][j+3] += matrixA[i+0][l] * matrixB[l][j+3];
				matrix[i+1][j+0] += matrixA[i+1][l] * matrixB[l][j+0];
				matrix[i+1][j+1] += matrixA[i+1][l] * matrixB[l][j+1];
				matrix[i+1][j+2] += matrixA[i+1][l] * matrixB[l][j+2];
				matrix[i+1][j+3] += matrixA[i+1][l] * matrixB[l][j+3];
				matrix[i+2][j+0] += matrixA[i+2][l] * matrixB[l][j+0];
				matrix[i+2][j+1] += matrixA[i+2][l] * matrixB[l][j+1];
				matrix[i+2][j+2] += matrixA[i+2][l] * matrixB[l][j+2];
				matrix[i+2][j+3] += matrixA[i+2][l] * matrixB[l][j+3];
				matrix[i+3][j+0] += matrixA[i+3][l] * matrixB[l][j+0];
				matrix[i+3][j+1] += matrixA[i+3][l] * matrixB[l][j+1];
				matrix[i+3][j+2] += matrixA[i+3][l] * matrixB[l][j+2];
				matrix[i+3][j+3] += matrixA[i+3][l] * matrixB[l][j+3];
			}
		}
	}
	return matrix;
}

int** loop_divide_3(int** matrixA, int** matrixB) {    //第三种拆分方法
	int **matrix=(int**)malloc(sizeof(int*)*M);
	for (int i=0;i<M;++i)
		matrix[i]=(int *)malloc(sizeof(int)*K);
	for (int i = 0; i < M; i+=4){
		for (int j = 0; j < K; j+=4){
			matrix[i+0][j+0] = 0;
			matrix[i+0][j+1] = 0;
			matrix[i+0][j+2] = 0;
			matrix[i+0][j+3] = 0;
			matrix[i+1][j+0] = 0;
			matrix[i+1][j+1] = 0;
			matrix[i+1][j+2] = 0;
			matrix[i+1][j+3] = 0;
			matrix[i+2][j+0] = 0;
			matrix[i+2][j+1] = 0;
			matrix[i+2][j+2] = 0;
			matrix[i+2][j+3] = 0;
			matrix[i+3][j+0] = 0;
			matrix[i+3][j+1] = 0;
			matrix[i+3][j+2] = 0;
			matrix[i+3][j+3] = 0;
			for (int l = 0; l < N; l+=4){
				matrix[i+0][j+0] += matrixA[i+0][l+0] * matrixB[l+0][j+0];
				matrix[i+0][j+1] += matrixA[i+0][l+0] * matrixB[l+0][j+1];
				matrix[i+0][j+2] += matrixA[i+0][l+0] * matrixB[l+0][j+2];
				matrix[i+0][j+3] += matrixA[i+0][l+0] * matrixB[l+0][j+3];
				matrix[i+1][j+0] += matrixA[i+1][l+0] * matrixB[l+0][j+0];
				matrix[i+1][j+1] += matrixA[i+1][l+0] * matrixB[l+0][j+1];
				matrix[i+1][j+2] += matrixA[i+1][l+0] * matrixB[l+0][j+2];
				matrix[i+1][j+3] += matrixA[i+1][l+0] * matrixB[l+0][j+3];
				matrix[i+2][j+0] += matrixA[i+2][l+0] * matrixB[l+0][j+0];
				matrix[i+2][j+1] += matrixA[i+2][l+0] * matrixB[l+0][j+1];
				matrix[i+2][j+2] += matrixA[i+2][l+0] * matrixB[l+0][j+2];
				matrix[i+2][j+3] += matrixA[i+2][l+0] * matrixB[l+0][j+3];
				matrix[i+3][j+0] += matrixA[i+3][l+0] * matrixB[l+0][j+0];
				matrix[i+3][j+1] += matrixA[i+3][l+0] * matrixB[l+0][j+1];
				matrix[i+3][j+2] += matrixA[i+3][l+0] * matrixB[l+0][j+2];
				matrix[i+3][j+3] += matrixA[i+3][l+0] * matrixB[l+0][j+3];

				matrix[i+0][j+0] += matrixA[i+0][l+1] * matrixB[l+1][j+0];
				matrix[i+0][j+1] += matrixA[i+0][l+1] * matrixB[l+1][j+1];
				matrix[i+0][j+2] += matrixA[i+0][l+1] * matrixB[l+1][j+2];
				matrix[i+0][j+3] += matrixA[i+0][l+1] * matrixB[l+1][j+3];
				matrix[i+1][j+0] += matrixA[i+1][l+1] * matrixB[l+1][j+0];
				matrix[i+1][j+1] += matrixA[i+1][l+1] * matrixB[l+1][j+1];
				matrix[i+1][j+2] += matrixA[i+1][l+1] * matrixB[l+1][j+2];
				matrix[i+1][j+3] += matrixA[i+1][l+1] * matrixB[l+1][j+3];
				matrix[i+2][j+0] += matrixA[i+2][l+1] * matrixB[l+1][j+0];
				matrix[i+2][j+1] += matrixA[i+2][l+1] * matrixB[l+1][j+1];
				matrix[i+2][j+2] += matrixA[i+2][l+1] * matrixB[l+1][j+2];
				matrix[i+2][j+3] += matrixA[i+2][l+1] * matrixB[l+1][j+3];
				matrix[i+3][j+0] += matrixA[i+3][l+1] * matrixB[l+1][j+0];
				matrix[i+3][j+1] += matrixA[i+3][l+1] * matrixB[l+1][j+1];
				matrix[i+3][j+2] += matrixA[i+3][l+1] * matrixB[l+1][j+2];
				matrix[i+3][j+3] += matrixA[i+3][l+1] * matrixB[l+1][j+3];

				matrix[i+0][j+0] += matrixA[i+0][l+2] * matrixB[l+2][j+0];
				matrix[i+0][j+1] += matrixA[i+0][l+2] * matrixB[l+2][j+1];
				matrix[i+0][j+2] += matrixA[i+0][l+2] * matrixB[l+2][j+2];
				matrix[i+0][j+3] += matrixA[i+0][l+2] * matrixB[l+2][j+3];
				matrix[i+1][j+0] += matrixA[i+1][l+2] * matrixB[l+2][j+0];
				matrix[i+1][j+1] += matrixA[i+1][l+2] * matrixB[l+2][j+1];
				matrix[i+1][j+2] += matrixA[i+1][l+2] * matrixB[l+2][j+2];
				matrix[i+1][j+3] += matrixA[i+1][l+2] * matrixB[l+2][j+3];
				matrix[i+2][j+0] += matrixA[i+2][l+2] * matrixB[l+2][j+0];
				matrix[i+2][j+1] += matrixA[i+2][l+2] * matrixB[l+2][j+1];
				matrix[i+2][j+2] += matrixA[i+2][l+2] * matrixB[l+2][j+2];
				matrix[i+2][j+3] += matrixA[i+2][l+2] * matrixB[l+2][j+3];
				matrix[i+3][j+0] += matrixA[i+3][l+2] * matrixB[l+2][j+0];
				matrix[i+3][j+1] += matrixA[i+3][l+2] * matrixB[l+2][j+1];
				matrix[i+3][j+2] += matrixA[i+3][l+2] * matrixB[l+2][j+2];
				matrix[i+3][j+3] += matrixA[i+3][l+2] * matrixB[l+2][j+3];

				matrix[i+0][j+0] += matrixA[i+0][l+3] * matrixB[l+3][j+0];
				matrix[i+0][j+1] += matrixA[i+0][l+3] * matrixB[l+3][j+1];
				matrix[i+0][j+2] += matrixA[i+0][l+3] * matrixB[l+3][j+2];
				matrix[i+0][j+3] += matrixA[i+0][l+3] * matrixB[l+3][j+3];
				matrix[i+1][j+0] += matrixA[i+1][l+3] * matrixB[l+3][j+0];
				matrix[i+1][j+1] += matrixA[i+1][l+3] * matrixB[l+3][j+1];
				matrix[i+1][j+2] += matrixA[i+1][l+3] * matrixB[l+3][j+2];
				matrix[i+1][j+3] += matrixA[i+1][l+3] * matrixB[l+3][j+3];
				matrix[i+2][j+0] += matrixA[i+2][l+3] * matrixB[l+3][j+0];
				matrix[i+2][j+1] += matrixA[i+2][l+3] * matrixB[l+3][j+1];
				matrix[i+2][j+2] += matrixA[i+2][l+3] * matrixB[l+3][j+2];
				matrix[i+2][j+3] += matrixA[i+2][l+3] * matrixB[l+3][j+3];
				matrix[i+3][j+0] += matrixA[i+3][l+3] * matrixB[l+3][j+0];
				matrix[i+3][j+1] += matrixA[i+3][l+3] * matrixB[l+3][j+1];
				matrix[i+3][j+2] += matrixA[i+3][l+3] * matrixB[l+3][j+2];
				matrix[i+3][j+3] += matrixA[i+3][l+3] * matrixB[l+3][j+3];
			}
		}
	}
	return matrix;
}



int** gemm(int** matrixA, int** matrixB) {    //通用矩阵乘法
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


int** initMatrix(int r, int c) {    //初始化矩阵
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


void isEqual(int** matrixA, int** matrixB) {  //用于检查优化算法的正确性
	for (int i = 0; i < M; i++){
		for (int j = 0; j < K; j++){
			if(matrixA[i][j] != matrixB[i][j]){
				printf("Matrices are not equal!\n");
  				return;
			}
		}
	}
	printf("Matrices are equal!\n");
}


int main(){
	printf("Please enter 3 integers (512~2048) :\n");
	scanf("%d",&M);
	scanf("%d",&N);
	scanf("%d",&K);
	if(M%4||N%4||K%4){
		printf("The integers must be divisible by 4\n");
		return 0;
	} 

	int** matrixA = initMatrix(M,N);
	int** matrixB = initMatrix(N,K);

	clock_t begin, end;
	begin=clock();
	int** matrixF = gemm(matrixA, matrixB);
	end=clock();
	double time4=(double)(end-begin)/CLOCKS_PER_SEC;
	printf("time of gemm:       %f s\n",time4);


	begin=clock();
	int** matrixC = loop_divide_1(matrixA, matrixB);
	end=clock();
	double time1=(double)(end-begin)/CLOCKS_PER_SEC;
	printf("time of loop_divide_1: %f s\n",time1);

	begin=clock();
	int** matrixD = loop_divide_2(matrixA, matrixB);
	end=clock();
	double time2=(double)(end-begin)/CLOCKS_PER_SEC;
	printf("time of loop_divide_2: %f s\n",time2);

	begin=clock();
	int** matrixE = loop_divide_3(matrixA, matrixB);
	end=clock();
	double time3=(double)(end-begin)/CLOCKS_PER_SEC;
	printf("time of loop_divide_3: %f s\n",time3);
	

	//isEqual(matrixF,matrixE);

	
	return 0;
}


