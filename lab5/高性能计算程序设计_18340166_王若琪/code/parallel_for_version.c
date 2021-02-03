/******************************************
 * Compile:
 * gcc -std=c99 -g -o parallel_for_version parallel_for_version.c -lpthread
 * Run:       
 * ./parallel_for_version <number of threads>
 ******************************************/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

int main(int argc, char *argv[]);
void parallel_for(int start, int end, int increment, void *(*functor)(void *), void *arg, int num_thread);
/******************************************************************************/
#define M 2000
#define N 2000

#define GET_TIME(now)                           \
    {                                           \
        struct timeval t;                       \
        gettimeofday(&t, NULL);                 \
        now = t.tv_sec + t.tv_usec / 1000000.0; \
    }

struct for_index
{
    int start;
    int end;
    int increment;
};

void parallel_for(int start, int end, int increment, void *(*functor)(void *), void *arg, int num_thread)
{
    long thread;
    pthread_t *thread_handles;
    thread_handles = malloc(num_thread * sizeof(pthread_t)); //为每个线程的pthread_t对象分配内存

    for (thread = 0; thread < num_thread; thread++) //生成线程
    {
        // 确定每个线程的开始和结束
        int my_rank = thread;
        int my_first, my_last;
        int quotient = (end - start) / num_thread;
        int remainder = (end - start) % num_thread;
        int my_count;
        if (my_rank < remainder)
        {
            my_count = quotient + 1;
            my_first = my_rank * my_count;
        }
        else
        {
            my_count = quotient;
            my_first = my_rank * my_count + remainder;
        }
        my_last = my_first + my_count;
        struct for_index *index;
        index = malloc(sizeof(struct for_index));
        index->start = start + my_first;
        index->end = start + my_last;
        index->increment = increment;
        pthread_create(&thread_handles[thread], NULL, functor, index);
    }

    for (thread = 0; thread < num_thread; thread++) //停止线程
    {
        pthread_join(thread_handles[thread], NULL);
    }
    free(thread_handles);
}

double diff;
double epsilon = 0.001;
int i;
int iterations;
int iterations_print;
int j;
double mean;
double mean;
double my_diff;
double **u;
double **w;
double start, finish;
int ThreadNumber;
pthread_mutex_t mutex;

void *func_1(void *args)
{
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        w[i][0] = 100.0;
    }
}

void *func_2(void *args)
{
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        w[i][N - 1] = 100.0;
    }
}

void *func_3(void *args)
{
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        w[M - 1][i] = 100.0;
    }
}

void *func_4(void *args)
{
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        w[0][i] = 0.0;
    }
}

void *func_5(void *args)
{
    struct for_index *index = (struct for_index *)args;
    pthread_mutex_lock(&mutex); //获得临界区的访问权
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        mean = mean + w[i][0] + w[i][N - 1];
    }
    pthread_mutex_unlock(&mutex); //退出临界区
}

void *func_6(void *args)
{
    struct for_index *index = (struct for_index *)args;
    pthread_mutex_lock(&mutex); //获得临界区的访问权
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        mean = mean + w[M - 1][i] + w[0][i];
    }
    pthread_mutex_unlock(&mutex); //退出临界区
}

void *func_7(void *args)
{
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        for (int j = 1; j < N - 1; j++)
        {
            w[i][j] = mean;
        }
    }
}

void *func_8(void *args)
{
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        for (int j = 0; j < N; j++)
        {
            u[i][j] = w[i][j];
        }
    }
}

void *func_9(void *args)
{
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        for (int j = 1; j < N - 1; j++)
        {
            w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;
        }
    }
}

void *func_10(void *args)
{
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        for (int j = 1; j < N - 1; j++)
        {
            if (my_diff < fabs(w[i][j] - u[i][j]))
            {
                my_diff = fabs(w[i][j] - u[i][j]);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    ThreadNumber = strtol(argv[1], NULL, 10);
    u = (double **)malloc(sizeof(double *) * M);
    for (int i = 0; i < M; ++i)
        u[i] = (double *)malloc(sizeof(double) * N);

    w = (double **)malloc(sizeof(double *) * M);
    for (int i = 0; i < M; ++i)
        w[i] = (double *)malloc(sizeof(double) * N);

    pthread_mutex_init(&mutex, NULL);
    printf("\n");
    printf("HEATED_PLATE_PARALLEL_FOR\n");
    printf("  C/PARALLEL_FOR version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    
    printf("  Number of threads =              %d\n", ThreadNumber);

    mean = 0.0;

    parallel_for(1, M - 1, 1, func_1, NULL, ThreadNumber);
    parallel_for(1, M - 1, 1, func_2, NULL, ThreadNumber);
    parallel_for(0, N, 1, func_3, NULL, ThreadNumber);
    parallel_for(0, N, 1, func_4, NULL, ThreadNumber);
    parallel_for(1, M - 1, 1, func_5, NULL, ThreadNumber);
    parallel_for(0, N, 1, func_6, NULL, ThreadNumber);

    mean = mean / (double)(2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", mean);

    parallel_for(1, M - 1, 1, func_7, NULL, ThreadNumber);

    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
    GET_TIME(start);

    diff = epsilon;

    while (epsilon <= diff)
    {
        parallel_for(0, M, 1, func_8, NULL, ThreadNumber);
        parallel_for(1, M - 1, 1, func_9, NULL, ThreadNumber);

        diff = 0.0;
        my_diff = 0.0;

        parallel_for(1, M - 1, 1, func_10, NULL, ThreadNumber);

        if (diff < my_diff)
        {
            diff = my_diff;
        }

        iterations++;
        if (iterations == iterations_print)
        {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }
    GET_TIME(finish);

    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", finish - start);
    /*
  Terminate.
*/
    printf("\n");
    printf("HEATED_PLATE_PARALLEL:\n");
    printf("  Normal end of execution.\n");

    pthread_mutex_destroy(&mutex);
    for (int i = 0; i < M; ++i)
        free(u[i]);
    free(u);
    for (int i = 0; i < M; ++i)
        free(w[i]);
    free(w);
    return 0;

#undef M
#undef N
}
