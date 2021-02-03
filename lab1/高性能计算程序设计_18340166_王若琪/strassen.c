#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int M;
int N;
int K;


int tableSizeFor(int M, int N, int K) {
	int cap=M;
	if(cap<N) cap=N;
	if(cap<K) cap=K; //比较得出三者的最大值

	int temp = cap - 1;
	temp |= temp >> 1;
	temp |= temp >> 2;
	temp |= temp >> 4;
	temp |= temp >> 8;
	temp |= temp >> 16;
	return temp + 1; //取到能够使矩阵变为2的n次方的最小值，便于strassen方法分块
}

int** divide(int** matrixA, int size, int pos) {  //给矩阵分块
	int **matrix=(int**)malloc(sizeof(int*)*(size/2));
	if(pos == 1){
		for (int i = 0; i < size/2; i++){
			matrix[i]=(int *)malloc(sizeof(int)*(size/2));
			for (int j = 0; j < size/2; j++){
				matrix[i][j] = matrixA[i][j];
			}
		}
	}
	else if(pos == 2){
		for (int i = 0; i < size/2; i++){
			matrix[i]=(int *)malloc(sizeof(int)*(size/2));
			for (int j = 0; j < size/2; j++){
				matrix[i][j] = matrixA[i][size/2+j];
			}
		}
	}
	else if(pos == 3){
		for (int i = 0; i < size/2; i++) {
			matrix[i]=(int *)malloc(sizeof(int)*(size/2));
			for (int j = 0; j < size / 2; j++) {
				matrix[i][j] = matrixA[i + size / 2][j];
			}
		}
	}
	else{
		for (int i = 0; i < size/2; i++){
			matrix[i]=(int *)malloc(sizeof(int)*(size/2));
			for (int j = 0; j < size/2; j++){
				matrix[i][j] = matrixA[i+size/2][j+size/2];
			}
		}
	}
	return matrix;
}

int** addMat(int** matrixA, int** matrixB, int size) {   //加法
	int **matrix=(int**)malloc(sizeof(int*)*size);
		for (int i = 0; i < size; i++){
			matrix[i]=(int *)malloc(sizeof(int)*size);
			for (int j = 0; j < size; j++){
				matrix[i][j] = matrixA[i][j] + matrixB[i][j];
			}
		}
	return matrix;
}

int** concatMatrices(int** A, int** B, int** C, int** D, int size) {  //合并
	int **matrix=(int**)malloc(sizeof(int*)*size*2);
	for (int i = 0; i < size; i++){
		matrix[i]=(int *)malloc(sizeof(int)*size*2);
		for (int j = 0; j < size; j++){
			matrix[i][j] = A[i][j];
		}
	}
	for (int i = 0; i < size; i++){
		matrix[size + i]=(int *)malloc(sizeof(int)*size*2);
		for (int j = 0; j < size; j++){
			matrix[i][size + j] = B[i][j];
		}
	}
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			matrix[size + i][j] = C[i][j];
		}
	}
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			matrix[size + i][size + j] = D[i][j];
		}
	}
	return matrix;
}


int** subtractMat(int** matrixA, int** matrixB, int size) {  //减法
	int **matrix=(int**)malloc(sizeof(int*)*size);
	for (int i = 0; i < size; i++){
		matrix[i]=(int *)malloc(sizeof(int)*size);
		for (int j = 0; j < size; j++){
			matrix[i][j] = matrixA[i][j] - matrixB[i][j];
		}
	}
	return matrix;
}

int** gemm_sq(int** matrixA, int** matrixB, int size) { //适用于方阵乘法
	int **matrix=(int**)malloc(sizeof(int*)*size);
	for (int i=0;i<size;++i)
		matrix[i]=(int *)malloc(sizeof(int)*size);
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			matrix[i][j] = 0;
			for (int l = 0; l < size; l++){
                		matrix[i][j] += matrixA[i][l] * matrixB[l][j];
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

int** Strassen(int** matrixA, int** matrixB, int size){  //strassen优化方法
	if(size <= 64){
		return gemm_sq(matrixA, matrixB, size);
	}
	else{
		int** a = divide(matrixA, size, 1);
		int** b = divide(matrixA, size, 2);
		int** c = divide(matrixA, size, 3);
		int** d = divide(matrixA, size, 4);
		int** e = divide(matrixB, size, 1);
		int** f = divide(matrixB, size, 2);
		int** g = divide(matrixB, size, 3);
		int** h = divide(matrixB, size, 4);

		int** p1 = Strassen(a, subtractMat(f, h, size/2), size/2);
		int** p2 = Strassen(addMat(a, b, size/2), h, size/2);
		int** p3 = Strassen(addMat(c, d, size/2), e, size/2);
		int** p4 = Strassen(d, subtractMat(g, e, size/2), size/2);
		int** p5 = Strassen(addMat(a, d, size/2), addMat(e, h, size/2), size/2);
		int** p6 = Strassen(subtractMat(b, d, size/2), addMat(g, h, size/2), size/2);
		int** p7 = Strassen(subtractMat(a, c, size/2), addMat(e, f, size/2), size/2);
		int** A = addMat(addMat(p4, p5, size/2), subtractMat(p6, p2, size/2), size/2);
		int** B = addMat(p1, p2, size/2);
		int** C = addMat(p3, p4, size/2);
		int** D = addMat(subtractMat(p1, p3, size/2), subtractMat(p5, p7, size/2), size/2);
		return concatMatrices(A, B, C, D, size/2);
	}
}


int** initMatrix(int r, int c, int extend_size) {  //初始化矩阵
	int **mat=(int**)malloc(sizeof(int*)*extend_size);
	for (int i=0;i<extend_size;++i)
		mat[i]=(int *)malloc(sizeof(int)*extend_size);
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			mat[i][j]=(int)rand()%50;
		}
		for(int j=c;j<extend_size;j++){
			mat[i][j]=0;
		}
	}
	for(int i=r;i<extend_size;i++){
		for(int j=0;j<extend_size;j++){
			mat[i][j]=0;
		}
	}
	return mat;
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

	int extend_len = tableSizeFor(M,N,K);
	int** matrixA = initMatrix(M,N,extend_len);
	int** matrixB = initMatrix(N,K,extend_len);

	clock_t begin, end;

	begin=clock();
	int** matrixD = gemm(matrixA, matrixB);
	end=clock();
	double time1=(double)(end-begin)/CLOCKS_PER_SEC;

	begin=clock();
	int** matrixC = Strassen(matrixA, matrixB, extend_len);
	end=clock();
	double time2=(double)(end-begin)/CLOCKS_PER_SEC;

//	isEqual(matrixC,matrixD);

	printf("time of gemm:     %f s\n",time1);
	printf("time of strassen: %f s\n",time2);
	

	return 0;
}

