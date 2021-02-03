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
    int *buffer_A, *buffer_C;

    int rank, numprocs;

    MPI_Status status;

    MPI_Init(&argc, &argv); // 并行开始
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_lines = M / numprocs; // 计算分块后每一块的行数

    buffer_A = new int[num_lines * N]; // 每个进程中的A分块
    buffer_C = new int[num_lines * K]; // 每个进程中的答案
    B = new int[N * K]; // 每个进程中的B

    start = MPI_Wtime();

    if (rank == 0)
    {
        A = new int[M * N];
        C = new int[M * K];

        initMatrix(A, M, N);
        initMatrix(B, N, K);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(A, num_lines * N, MPI_INT, buffer_A, num_lines * N, MPI_INT, 0, MPI_COMM_WORLD);  // 将 A 的分块发送给其他进程
    MPI_Bcast(B, N * K, MPI_INT, 0, MPI_COMM_WORLD); // 将 B 广播给其他进程

    gemm(buffer_A, B, buffer_C, num_lines, N, K);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(buffer_C, num_lines * K, MPI_INT, C, num_lines * K, MPI_INT, 0, MPI_COMM_WORLD);  // 从其他进程中收集数组

    int lastRow = num_lines * numprocs;
    if (rank == 0 && lastRow < M)  // 计算完没有除尽的部分
    {
        int remainRows = M - lastRow;
        gemm(A + lastRow * N, B, C + lastRow * K, remainRows, N, K);
    }

    delete[] buffer_A;
    delete[] buffer_C;
    delete[] B;

    end = MPI_Wtime();

    if (rank == 0)
    {
        //        printMatrix(A, M, N);
        //        printMatrix(B, N, K);
        //        printMatrix(C, M, K);
        printf("Runtime = %f s\n", end - start);
        delete[] A;
        delete[] C;
    }

    MPI_Finalize();

    return 0;
}

void initMatrix(int *A, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        A[i] = rand() % 50;
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