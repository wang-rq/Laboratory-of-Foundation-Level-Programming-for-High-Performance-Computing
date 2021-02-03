/******************************************
 * Compile:
 * nvcc 1.cu -o 1
 * Run:       
 * ./1
*******************************************/

#include <stdio.h>
#include <sys/time.h>

#define mat_height 4096
#define mat_width 4096
#define filter_height 3
#define filter_width 3
#define stride 1

#define block_size_x 32
#define block_size_y 32

#define GET_TIME(now)                           \
    {                                           \
        struct timeval t;                       \
        gettimeofday(&t, NULL);                 \
        now = t.tv_sec + t.tv_usec / 1000000.0; \
    }

int check(float *c, float *d, int n);

// 计算
#define padding_height ((filter_height / 2) * 2)
#define padding_width ((filter_width / 2) * 2)
#define input_height (mat_height + padding_height)
#define input_width (mat_width + padding_width)

void cpu_convolution(float *output, float *input, float *filter)
{
    // 矩阵循环 步长 stride
    for (int y = 0; y < mat_height; y += stride)
    {
        for (int x = 0; x < mat_width; x += stride)
        {
            // filter 循环
            for (int i = 0; i < filter_height; i++)
            {
                for (int j = 0; j < filter_width; j++)
                {
                    output[y / stride * mat_width + x / stride] += input[(y + i) * input_width + x + j] * filter[i * filter_width + j];
                }
            }
        }
    }
}

// 将三层的结果相加
void cpu_add(float *arr1, float *arr2, float *arr3, float *result)
{
    for (int y = 0; y < mat_height; y+=stride)
    {
        for(int x = 0; x <  mat_width; x+=stride){
            int temp = y / stride * mat_width + x / stride;
            result[temp] = arr1[temp] + arr2[temp] + arr3[temp];
        }
    }
}

__global__ void cuda_convolution(float *output, float *input, float *filter)
{
    // 计算矩阵的坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    // 如果符合步长要求就计算结果
    if (y % stride == 0 && x % stride == 0)
    {
        for (int i = 0; i < filter_height; i++)
        {
            for (int j = 0; j < filter_width; j++)
            {
                sum += input[(y + i) * input_width + x + j] * filter[i * filter_width + j];
            }
        }
        output[y / stride * mat_width + x / stride] = sum;
    }
}

// 将三层的结果相加
__global__ void cuda_add(float *arr1, float *arr2, float *arr3, float *result){
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y % stride == 0 && x % stride == 0)
        result[y / stride * mat_width + x / stride] = arr1[y / stride * mat_width + x / stride] + arr2[y / stride * mat_width + x / stride] + arr3[y / stride * mat_width + x / stride];
}

int main()
{
    double start, finish, time;

    // 动态分配内存
    float *input1 = (float *)malloc(input_height * input_width * sizeof(float));
    float *input2 = (float *)malloc(input_height * input_width * sizeof(float));
    float *input3 = (float *)malloc(input_height * input_width * sizeof(float));

    float *output11 = (float *)malloc(mat_height * mat_width * sizeof(float));
    float *output12 = (float *)malloc(mat_height * mat_width * sizeof(float));
    float *output13 = (float *)malloc(mat_height * mat_width * sizeof(float));
    float *output21 = (float *)malloc(mat_height * mat_width * sizeof(float));
    float *output22 = (float *)malloc(mat_height * mat_width * sizeof(float));
    float *output23 = (float *)malloc(mat_height * mat_width * sizeof(float));

    float *result1 = (float *)malloc(mat_height * mat_width * sizeof(float));
    float *result2 = (float *)malloc(mat_height * mat_width * sizeof(float));

    float *filter1 = (float *)malloc(filter_height * filter_width * sizeof(float));
    float *filter2 = (float *)malloc(filter_height * filter_width * sizeof(float));
    float *filter3 = (float *)malloc(filter_height * filter_width * sizeof(float));

    // 初始化input矩阵 (padding 部分默认为 0)
    for (int i = 0; i < input_height * input_width; i++)
    {
        input1[i] = (float)(rand() % 50)/100;
        input2[i] = (float)(rand() % 50)/100;
        input3[i] = (float)(rand() % 50)/100;
    }
    // 初始化filter
    for (int i = 0; i < filter_height * filter_width; i++)
    {
        filter1[i] = (float)(rand() % 50)/100;
        filter2[i] = (float)(rand() % 50)/100;
        filter3[i] = (float)(rand() % 50)/100;
    }

    // 结果初始化为0
    memset(output11, 0, mat_height * mat_height * sizeof(float));
    memset(output12, 0, mat_height * mat_height * sizeof(float));
    memset(output13, 0, mat_height * mat_height * sizeof(float));
    memset(output21, 0, mat_height * mat_height * sizeof(float));
    memset(output22, 0, mat_height * mat_height * sizeof(float));
    memset(output23, 0, mat_height * mat_height * sizeof(float));

    // 串行计算并计时
    GET_TIME(start);
    cpu_convolution(output11, input1, filter1);
    cpu_convolution(output12, input2, filter2);
    cpu_convolution(output13, input3, filter3);
    cpu_add(output11, output12, output13, result1);
    GET_TIME(finish);
    time = finish - start;
    printf("Sequential convolution time:  %f s\n", time);

    // 分配CUDA空间
    float *cuda_input1;
    float *cuda_output1;
    float *cuda_filter1;
    cudaMalloc((void **)&cuda_input1, input_height * input_width * sizeof(float));
    cudaMalloc((void **)&cuda_output1, mat_height * mat_width * sizeof(float));
    cudaMalloc((void **)&cuda_filter1, filter_height * filter_width * sizeof(float));
    float *cuda_input2;
    float *cuda_output2;
    float *cuda_filter2;
    cudaMalloc((void **)&cuda_input2, input_height * input_width * sizeof(float));
    cudaMalloc((void **)&cuda_output2, mat_height * mat_width * sizeof(float));
    cudaMalloc((void **)&cuda_filter2, filter_height * filter_width * sizeof(float));
    float *cuda_input3;
    float *cuda_output3;
    float *cuda_filter3;
    cudaMalloc((void **)&cuda_input3, input_height * input_width * sizeof(float));
    cudaMalloc((void **)&cuda_output3, mat_height * mat_width * sizeof(float));
    cudaMalloc((void **)&cuda_filter3, filter_height * filter_width * sizeof(float));
    float *cuda_result;
    cudaMalloc((void **)&cuda_result, mat_height * mat_width * sizeof(float));
    
    // 将矩阵和filter拷贝到gpu中
    cudaMemcpy(cuda_input1, input1, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_filter1, filter1, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_input2, input2, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_filter2, filter2, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_input3, input3, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_filter3, filter3, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);

    //结果初始化为0
    cudaMemset(cuda_output1, 0, mat_height * mat_width * sizeof(float));
    cudaMemset(cuda_output2, 0, mat_height * mat_width * sizeof(float));
    cudaMemset(cuda_output3, 0, mat_height * mat_width * sizeof(float));

    //设置 stread 和 grid
    dim3 threads(block_size_x, block_size_y);
    dim3 grid(int(ceilf(mat_width / (float)threads.x)), int(ceilf(mat_height / (float)threads.y)));

    // CUDA并行计算并计时
    cudaDeviceSynchronize();
    GET_TIME(start);
    cuda_convolution<<<grid, threads>>>(cuda_output1, cuda_input1, cuda_filter1);
    cuda_convolution<<<grid, threads>>>(cuda_output2, cuda_input2, cuda_filter2);
    cuda_convolution<<<grid, threads>>>(cuda_output3, cuda_input3, cuda_filter3);
    cuda_add<<<grid, threads>>>(cuda_output1, cuda_output2, cuda_output3, cuda_result);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    time = finish - start;
    printf("CUDA convolution time:        %f s\n", time);

    // 将结果拷贝到内存空间
    cudaMemcpy(result2, cuda_result, mat_height * mat_width * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果到文件result.txt
    FILE *fp = fopen("result.txt", "w");
    for (int y = 0; y < mat_height; y += stride)
    {
        for(int x = 0; x < mat_width; x += stride){
            fprintf(fp, "%f ", result2[y / stride * mat_width + x / stride]);
        }
        fprintf(fp, "\n");
    }

    // 检测结果是否在误差范围内
    if (check(result1, result2, mat_height * mat_width) > 0)
    {
        printf("The result is wrong.\n");
    }
    else
    {
        printf("The result is right.\n");
    }

    // 释放空间
    cudaFree(cuda_output1);
    cudaFree(cuda_input1);
    cudaFree(cuda_filter1);
    cudaFree(cuda_output2);
    cudaFree(cuda_input2);
    cudaFree(cuda_filter2);
    cudaFree(cuda_output3);
    cudaFree(cuda_input3);
    cudaFree(cuda_filter3);
    cudaFree(cuda_result);
    free(filter1);
    free(input1);
    free(filter2);
    free(input2);
    free(filter3);
    free(input3);
    free(output11);
    free(output12);
    free(output13);
    free(output21);
    free(output22);
    free(output23);
    free(result1);
    free(result2);

    return 0;
}

// 判断结果是否在误差范围内
int check(float *arr1, float *arr2, int n)
{
    int errors = 0;
    for (int y = 0; y < mat_height; y += stride)
    {
        for(int x = 0; x < mat_width; x += stride){
            int i = y / stride * mat_width + x / stride;
            if (isnan(arr1[i]) || isnan(arr2[i]))
                errors++;
            float diff = (arr1[i] - arr2[i]) / arr1[i];
            if (diff > 1e-6f)
                errors++;
        }
    }
    return errors;
}