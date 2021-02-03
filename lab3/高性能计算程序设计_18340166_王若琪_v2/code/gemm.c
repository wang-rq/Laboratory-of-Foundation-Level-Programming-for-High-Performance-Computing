/******************************************
 * Compile:
 * gcc -std=c99 -g -o gemm gemm.c -lpthread
 * Run:       
 * ./gemm <number of threads>
 ******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

int thread_count;
int M, N, K;
int **matrixA;
int **matrixB;
int **matrixC;

void *gemm(void *rank);
int **initMatrix(int r, int c);
void printMat(int **mat, int r, int c);

int main(int argc, char *argv[])
{
    printf("Please enter 3 integers (512~2048) :\n");
    scanf("%d", &M);
    scanf("%d", &N);
    scanf("%d", &K);
    matrixA = initMatrix(M, N);
    matrixB = initMatrix(N, K);
    matrixC = (int **)malloc(sizeof(int *) * M);
    for (int i = 0; i < M; ++i)
        matrixC[i] = (int *)malloc(sizeof(int) * K);

    double start, finish, time;
    GET_TIME(start);

    long thread;
    pthread_t *thread_handles;
    thread_count = strtol(argv[1], NULL, 10);
    thread_handles = malloc(thread_count * sizeof(pthread_t)); //为每个线程的pthread_t对象分配内存
    for (thread = 0; thread < thread_count; thread++)          //生成线程
    {
        pthread_create(&thread_handles[thread], NULL, gemm, (void *)thread);
    }
    for (thread = 0; thread < thread_count; thread++) //停止线程
    {
        pthread_join(thread_handles[thread], NULL);
    }
    free(thread_handles);

    GET_TIME(finish);
    time = finish-start;

    printMat(matrixA, M, N);
    printMat(matrixB, N, K);
    printMat(matrixC, M, K);
    printf("time: %f s\n", time);

    free(matrixA);
    free(matrixB);
    free(matrixC);
}

int **initMatrix(int r, int c) //初始化矩阵
{
    int **temp = (int **)malloc(sizeof(int *) * r);
    for (int i = 0; i < r; ++i)
        temp[i] = (int *)malloc(sizeof(int) * c);

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            temp[i][j] = rand() % 50;
        }
    }
    return temp;
}

void *gemm(void *rank) //每个线程内进行计算
{
    //确定每个线程的计算范围
    int my_rank = (long)rank;
    int my_first_M, my_last_M; //矩阵 A 的开始行数和结束行数
    int quotient = M / thread_count;
    int remainder = M % thread_count;
    int my_n_count;
    if (my_rank < remainder)
    {
        my_n_count = quotient + 1;
        my_first_M = my_rank * my_n_count;
    }
    else
    {
        my_n_count = quotient;
        my_first_M = my_rank * my_n_count + remainder;
    }
    my_last_M = my_first_M + my_n_count;

    //通用矩阵乘法
    for (int i = my_first_M; i < my_last_M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            matrixC[i][j] = 0;
            for (int l = 0; l < N; l++)
            {
                matrixC[i][j] += matrixA[i][l] * matrixB[l][j];
            }
        }
    }
    return NULL;
}

void printMat(int **mat, int r, int c)
{ //输出矩阵
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            printf("%d ", mat[i][j]);
            if (j == c - 1)
                printf("\n");
        }
    }
    printf("\n");
}
