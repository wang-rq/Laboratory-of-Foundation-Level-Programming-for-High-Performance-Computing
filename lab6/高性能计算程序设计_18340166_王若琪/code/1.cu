/******************************************
* Compile:
* nvcc 1.cu -o 1
* Run:       
* ./1 <M> <N> <K> <CUDA_BLOCK_SIZE>
******************************************/

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

int BLOCK_SIZE;

//CUDA 初始化
bool InitCUDA()
{
	int count;
	//取得支持Cuda的设备的数目
	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++)
	{

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}
	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

//生成随机矩阵
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

//输出矩阵
void printMat(float *mat, int r, int l)
{
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

// 并行矩阵乘法函数
__global__ static void matMultCUDA(const float *a, const float *b, float *c, int M, int N, int K, int BLOCK_SIZE)
{
	//表示目前的 thread 的编号
	const int tid = threadIdx.x;
	//表示目前在第几个 block 中
	const int bid = blockIdx.x;
	//计算出当前的 row 和 column
	const int idx = bid * BLOCK_SIZE + tid;
	const int row = idx / M;
	int column = idx % M;
	do
	{
		//矩阵乘法
		if (row < M && column < K)
		{
			float t = 0;
			for (int i = 0; i < N; i++)
			{
				t += a[row * N + i] * b[i * K + column];
			}
			c[row * K + column] = t;
		}
		column += M;
	} while (column < K);
}

int main(int argc, char **argv)
{
	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);
	BLOCK_SIZE = atoi(argv[4]);

	const int blocks_num = (M * M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	//CUDA 初始化
	if (!InitCUDA())
		return 0;

	//定义矩阵
	float *a, *b, *c;

	//分配内存
	a = (float *)malloc(sizeof(float) * M * N);
	b = (float *)malloc(sizeof(float) * N * K);
	c = (float *)malloc(sizeof(float) * M * K);

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
	cudaMemcpy(cuda_a, a, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, sizeof(float) * N * K, cudaMemcpyHostToDevice);

	// 在CUDA 中执行函数
	matMultCUDA<<<blocks_num, BLOCK_SIZE, 0>>>(cuda_a, cuda_b, cuda_c, M, N, K, BLOCK_SIZE);

	//cudaMemcpy 将结果从显存中复制回内存
	cudaMemcpy(c, cuda_c, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

	GET_TIME(finish);
	time = finish - start;

	//Free
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);

	printMat(a, M, N);
	printMat(b, N, K);
	printMat(c, M, K);

	printf("time of gemm: %f s\n\n", time);

	return 0;
}