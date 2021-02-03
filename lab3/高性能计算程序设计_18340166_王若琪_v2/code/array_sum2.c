/******************************************
 * Compile:
 * gcc -std=c99 -g -o array_sum2 array_sum2.c -lpthread
 * Run:       
 * ./array_sum2 <number of threads>
 ******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define GET_TIME(now)                           \
    {                                           \
        struct timeval t;                       \
        gettimeofday(&t, NULL);                 \
        now = t.tv_sec + t.tv_usec / 1000000.0; \
    }

#define NUM 1000

void *array_sum(void *rank);
void init_array();
int se_add();

int thread_count;
pthread_mutex_t mutex;
int *array;
int global_index = 0;
int sum = 0;

int main(int argc, char *argv[])
{
    array = (int *)malloc(NUM * sizeof(int));
    init_array();
    pthread_mutex_init(&mutex, NULL);

    double start, finish, time;
    GET_TIME(start);

    long thread;
    pthread_t *thread_handles;
    thread_count = strtol(argv[1], NULL, 10);
    thread_handles = malloc(thread_count * sizeof(pthread_t)); //为每个线程的pthread_t对象分配内存
    for (thread = 0; thread < thread_count; thread++)          //生成线程
    {
        pthread_create(&thread_handles[thread], NULL, array_sum, (void *)thread);
    }
    for (thread = 0; thread < thread_count; thread++) //停止线程
    {
        pthread_join(thread_handles[thread], NULL);
    }
    free(thread_handles);

    GET_TIME(finish);
    time = finish - start;

    printf("sun = %d\n", sum);
    printf("time of array_sum: %f s\n", time);

    pthread_mutex_destroy(&mutex);
    free(array);
}

void *array_sum(void *rank)
{
    int temp_index;
    int local_sum;
    while (1)
    {
        local_sum = 0;
        pthread_mutex_lock(&mutex); //获得临界区的访问权
        if (global_index == NUM)
        {
            pthread_mutex_unlock(&mutex); //退出临界区
            break;
        }
        else
        {
            temp_index = global_index;
            global_index += 10;
            pthread_mutex_unlock(&mutex); //退出临界区
        }

        for (int i = 0; i < 10; i++)//在临界区外进行10个一组求和
            local_sum += array[temp_index + i];

        pthread_mutex_lock(&mutex); //获得临界区的访问权
        sum += local_sum;//加到sum里面
        pthread_mutex_unlock(&mutex); //退出临界区
    }
    return NULL;
}

void init_array()
{
    for (int i = 0; i < NUM; i++)
    {
        array[i] = rand() % 50;
    }
}
