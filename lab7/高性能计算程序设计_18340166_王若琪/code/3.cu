/******************************************
 * Compile:
 * nvcc 3.cu -o 3 -I/opt/conda/include -L/opt/conda/lib -lcudnn
 * Run:       
 * ./3
*******************************************/

#include <cudnn.h>
#include <cuda.h>
#include <iostream>
using namespace std;

#define input_height 4096
#define input_width 4096
#define filter_size 3
#define stride 3
#define pad 1
#define channels 3
#define output_height (input_height + 2*pad - filter_size) / stride + 1
#define output_width (input_width + 2*pad - filter_size) / stride + 1

#define checkCUDNN(expression)                                     \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS)                        \
        {                                                          \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

int main(int argc, char const *argv[])
{
    cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	
	float *image = (float *)malloc(input_height * input_width * channels * sizeof(float));
	// 初始化input矩阵
	for (int i = 0; i < input_height * input_width * channels ; i++)
	{
		image[i] = (float)(rand() % 50)/100;
	}


    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/3,
                                          /*image_height=*/input_height,
                                          /*image_width=*/input_width));

    //output descriptor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/1,
                                          /*image_height=*/output_height,
                                          /*image_width=*/output_width));

    //descriptor kernel
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/1,
                                          /*in_channels=*/3,
                                          /*kernel_height=*/3,
                                          /*kernel_width=*/3));

    //descriptor convolucion
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                               /*pad_height=*/pad,
                                               /*pad_width=*/pad,
                                               /*vertical_stride=*/stride,
                                               /*horizontal_stride=*/stride,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
        cudnnGetConvolutionForwardAlgorithm(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                            /*memoryLimitInBytes=*/0,
                                            &convolution_algorithm));

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));

    //分配空间
    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);
    int image_bytes = 1 * 3 * input_height * input_width * sizeof(float);
    float *d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image, image_bytes, cudaMemcpyHostToDevice);
    float *d_output{nullptr};
    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    // 初始化 kernel
    float h_kernel[3][1][3][3];
    for (int kernel = 0; kernel < 3; ++kernel)
    {
        for (int channel = 0; channel < 1; ++channel)
        {
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    h_kernel[kernel][channel][row][column] = (float)(rand() % 50)/100;
                }
            }
        }
    }

    float *d_kernel{nullptr};
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    const float alpha = 1, beta = 1;

    // convolution forward
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       d_output));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time : %f\n", milliseconds / (3.0f * 1000.0f));
    
    float *h_output = (float *)malloc(image_bytes);
	cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
    
    // 输出结果到文件result.txt
    FILE *fp = fopen("result.txt", "w");
    for (int i = 0; i < output_width * output_height; i++)
    {
        fprintf(fp, "%f ", h_output[i]);
    }
    
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    free(image);
	free(h_output);
	free(d_workspace);
    
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);

    cudnnDestroy(cudnn);
    
    return 0;
}