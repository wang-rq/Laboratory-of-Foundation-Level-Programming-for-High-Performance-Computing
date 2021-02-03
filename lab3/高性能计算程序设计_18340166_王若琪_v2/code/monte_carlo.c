/******************************************
 * Compile:
 * gcc -std=c99 -g -o monte_carlo monte_carlo.c -lpthread
 * Run:       
 * ./monte_carlo <number of threads> <number of tosses>
 ******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MR_MULTIPLIER 279470273
#define MR_INCREMENT 0
#define MR_MODULUS 4294967291U
#define MR_DIVISOR ((double)4294967291U)

void *Thread_work(void *rank);
unsigned my_rand(unsigned *seed_p);
double my_drand(unsigned *seed_p);

int thread_count;
long long number_in = 0;
long long number_of_tosses;

pthread_mutex_t mutex;

int main(int argc, char *argv[])
{
    long thread;
    pthread_t *thread_handles;
    double estimate;
    thread_count = strtol(argv[1], NULL, 10);
    number_of_tosses = strtoll(argv[2], NULL, 10);
    pthread_mutex_init(&mutex, NULL);

    thread_handles = malloc(thread_count * sizeof(pthread_t)); //为每个线程的pthread_t对象分配内存
    for (thread = 0; thread < thread_count; thread++)          //生成线程
    {
        pthread_create(&thread_handles[thread], NULL, Thread_work, (void *)thread);
    }
    for (thread = 0; thread < thread_count; thread++) //停止线程
    {
        pthread_join(thread_handles[thread], NULL);
    }
    free(thread_handles);

    estimate = number_in / ((double)number_of_tosses); //计算面积
    printf("Estimated value: %e\n", estimate);

    pthread_mutex_destroy(&mutex);
    return 0;
}

void *Thread_work(void *rank)
{
    long my_rank = (long)rank;
    long long toss;
    long long local_number_in = 0;
    long long local_tosses = number_of_tosses / thread_count;
    long long start = local_tosses * my_rank;
    long long finish = start + local_tosses;
    double x, y;
    unsigned seed = my_rank + 1;

    for (toss = start; toss < finish; toss++)
    {
        x = 2 * my_drand(&seed) - 1; //随机生成x
        y = 2 * my_drand(&seed) - 1; //随机生成y
        if (x * x <= y)              //判断落在阴影部分
            local_number_in++;
    }
    pthread_mutex_lock(&mutex); //获得临界区的访问权
    number_in += local_number_in;
    pthread_mutex_unlock(&mutex); //退出临界区

    return NULL;
}

unsigned my_rand(unsigned *seed_p)
{
    long long z = *seed_p;
    z *= MR_MULTIPLIER;
    z %= MR_MODULUS;
    *seed_p = z;
    return *seed_p;
}

double my_drand(unsigned *seed_p)
{
    unsigned x = my_rand(seed_p);
    double y = x / MR_DIVISOR;
    return y;
}