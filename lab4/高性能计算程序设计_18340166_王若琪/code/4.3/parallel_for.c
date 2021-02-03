#include <stdlib.h>
#include <pthread.h>
#include "func.h"

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