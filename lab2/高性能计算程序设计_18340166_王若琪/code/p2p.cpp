#include <iostream>
#include <mpi.h>
#include <stdlib.h>
using namespace std;

void initMatrix(int *A, int rows, int cols);
void gemm(int *A, int *B, int *C, int m, int n, int k);
void printMatrix(int *A, int rows, int cols);

int main(int argc, char **argv)
{
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    double start, end;

    int *A, *B, *C;
    int *buffer_A, *buffer_B, *buffer_C;

    int rank, numprocs;

    MPI_Status status;

    MPI_Init(&argc, &argv); // 并行开始
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_lines = M / numprocs; // 计算分块后每一块的行数

    buffer_A = new int[num_lines * N]; // 每个进程中的A分块
    buffer_C = new int[num_lines * K]; // 每个进程中的答案
    buffer_B = new int[N * K]; // 每个进程中的B


    if (rank == 0)
    {
        A = new int[M * N];
        B = new int[N * K];
        C = new int[M * K];
        initMatrix(A, M, N);
        initMatrix(B, N, K);

        start = MPI_Wtime();

        for (int i = 1; i < numprocs; i++) // 将 B 发送给其他进程
        {
            MPI_Send(B, N * K, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for (int i = 1; i < numprocs; i++) // 将 A 的分块发送给其他进程
        {
            MPI_Send(A + (i - 1) * num_lines * N, N * num_lines, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

        for (int i = 1; i < numprocs; i++)  // 接收答案
        {
            MPI_Recv(buffer_C, num_lines * K, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < num_lines; j++)
            {
                for (int l = 0; l < K; l++)
                {
                    C[((i - 1) * num_lines + j) * K + l] = buffer_C[j * K + l];
                }
            }
        }
       
        int lastRow = (num_lines * (numprocs - 1));
        if (lastRow < M) // 计算完没有除尽的部分
        {
            int remainRows = M - lastRow;
            gemm(A + lastRow * N, B, C + lastRow * K, remainRows, N, K);
        }

        end = MPI_Wtime();

        // printMatrix(A, M, N);
        // printMatrix(B, N, K);
        // printMatrix(C, M, K);
        printf("Runtime = %f s\n", end - start);
        delete[] A;
        delete[] B;
        delete[] C;
    }

    else
    {
        MPI_Recv(buffer_B, N * K, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // 接收来自 0 号进程的矩阵 B
        MPI_Recv(buffer_A, N * num_lines, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // 接收来自 0 号进程的矩阵 A 的分块
        gemm(buffer_A, buffer_B, buffer_C, num_lines, N, K);
        MPI_Send(buffer_C, num_lines * K, MPI_INT, 0, 2, MPI_COMM_WORLD); // 发送这一部分的计算结果到 0 号进程中
    }

    delete[] buffer_A;
    delete[] buffer_B;
    delete[] buffer_C;
    MPI_Finalize();

    return 0;
}

void initMatrix(int *Matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        Matrix[i]=rand()%50;
    }
}

void gemm(int *A, int *B, int *C, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            int temp = 0;
            for (int l = 0; l < n; l++)
            {
                temp += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = temp;
        }
    }
}

void printMatrix(int *A, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", A[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}