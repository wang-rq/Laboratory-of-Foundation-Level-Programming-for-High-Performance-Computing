/******************************************
 * Compile:
 * nvcc 2.cu -o 2
 * Run:       
 * ./2
*******************************************/

#include <stdio.h>
#include <sys/time.h>

#define height 256
#define width 256
#define filter_size 3
#define stride 1
#define pad 1
#define channels 3

#define block_size_x 32
#define block_size_y 32

#define GET_TIME(now)                           \
    {                                           \
        struct timeval t;                       \
        gettimeofday(&t, NULL);                 \
        now = t.tv_sec + t.tv_usec / 1000000.0; \
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

float im2col_get_data(float *im, int row, int col, int channel){ 
    row -= pad;
    col -= pad;    // padding补0
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;    
    // im[col + width*(row + height*channel)]=im[col+width*row+width*height*channel]
    return im[col + width*(row + height*channel)];
}
    

void im2col(float* data_im, float* data_col) {   
    int c,h,w;    // 计算卷积后的尺寸
    int height_col = (height + 2*pad - filter_size) / stride + 1;    
    int width_col = (width + 2*pad - filter_size) / stride + 1;
    int channels_col = channels * filter_size * filter_size;    
    // 获取对应的值
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % filter_size;        
        int h_offset = (c / filter_size) % filter_size;        
        int c_im = c / filter_size / filter_size;        
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) { 
                // 获取原图中对应的坐标
                int im_row = h_offset + h * stride;                
                int im_col = w_offset + w * stride;                
                // col_index为重排后图像中的索引
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_data(data_im, im_row, im_col, c_im);
            }
        }
    }
}

int main()
{
    double start, finish, time;

    int height_col = (height + 2*pad - filter_size) / stride + 1;    
    int width_col = (width + 2*pad - filter_size) / stride + 1;    /// 卷积核大小：filter_size*filter_size是一个卷积核的大小，通道数channels
    int channels_col = channels * filter_size * filter_size;    
    
    // 动态分配内存
    float *im = (float *)malloc(height * width * channels * sizeof(float));
    float *col = (float *)malloc(channels_col * height_col * width_col * sizeof(float));
    float *filter = (float *)malloc(channels * filter_size * filter_size * sizeof(float));

    // 初始化input矩阵
    for (int i = 0; i < height * width * channels ; i++)
    {
        im[i] = (float)(rand() % 50)/100;
    }
    // 初始化filter
    for (int i = 0; i < filter_size * filter_size * channels; i++)
    {
        filter[i] = (float)(rand() % 50)/100;
    }

    GET_TIME(start);

    im2col(im, col);

    float *cuda_a, *cuda_b, *cuda_c;
    

    float *c = (float *)malloc(channels * (width_col * height_col) * sizeof(float));
    

	//cudaMalloc 分配空间
	cudaMalloc((void **)&cuda_a, sizeof(float) * channels * (filter_size * filter_size) );
	cudaMalloc((void **)&cuda_b, sizeof(float) * channels_col * (width_col * height_col));
	cudaMalloc((void **)&cuda_c, sizeof(float) * channels * (width_col * height_col));
    

	//cudaMemcpy 将矩阵复制到显存中
	cudaMemcpy(cuda_a, filter, sizeof(float) * channels * (filter_size * filter_size), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, col, sizeof(float) * channels_col * (width_col * height_col), cudaMemcpyHostToDevice);

    int BLOCK_SIZE = height;
    const int blocks_num = (channels * (width_col * height_col) + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// 在CUDA 中执行 gemm 函数
	matMultCUDA<<<blocks_num, BLOCK_SIZE, 0>>>(cuda_a, cuda_b, cuda_c, channels, (filter_size * filter_size), (width_col * height_col), BLOCK_SIZE);

    //cudaMemcpy 将结果从显存中复制回内存
	cudaMemcpy(c, cuda_c, sizeof(float) * channels * (width_col * height_col), cudaMemcpyDeviceToHost);
    
	GET_TIME(finish);
	time = finish - start;
    
    // 输出结果到文件result.txt
    FILE *fp = fopen("result.txt", "w");
    for (int i = 0; i < channels * (width_col * height_col); i++)
    {
        fprintf(fp, "%f ", c[i]);
    }

	//Free
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
    free(c);
    free(im);
    free(col);

	printf("time: %f s\n\n", time);
    

    return 0;
}

