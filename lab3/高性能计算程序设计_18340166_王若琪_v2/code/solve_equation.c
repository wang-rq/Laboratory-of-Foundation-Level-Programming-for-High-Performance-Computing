/******************************************
 * Compile:
 * gcc -std=c99 -g -o solve_equation solve_equation.c -lpthread -lm
 * Run:       
 * ./solve_equation
 ******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

void *thread_work_0(void *rank); //求得delta
void *thread_work_1(void *rank); //求得分母
void *thread_work_2(void *rank); //求得最终解

int thread_count = 3;
double a, b, c;
double delta;                      //中间值 delta
double denominator1, denominator2; //中间值 分母
double x1, x2;
pthread_mutex_t mutex;
pthread_cond_t cond_var1, cond_var2;

int main(int argc, char *argv[])
{
    printf("Please enter a: ");
    scanf("%lf", &a);
    printf("Please enter b: ");
    scanf("%lf", &b);
    printf("Please enter c: ");
    scanf("%lf", &c);

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond_var1, NULL);
    pthread_cond_init(&cond_var2, NULL);
    long thread;
    pthread_t *thread_handles;
    thread_handles = malloc(thread_count * sizeof(pthread_t)); //为每个线程的pthread_t对象分配内存

    pthread_create(&thread_handles[0], NULL, thread_work_0, (void *)0);
    pthread_create(&thread_handles[1], NULL, thread_work_1, (void *)1);
    pthread_create(&thread_handles[2], NULL, thread_work_2, (void *)2);

    for (thread = 0; thread < thread_count; thread++) //停止线程
    {
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond_var1);
    pthread_cond_destroy(&cond_var2);

    if (x1 != x2)
        printf("Solution: x1 = %f , x2 = %f\n", x1, x2);
    else
        printf("Solution: x1 = x2 = %f\n", x1);

    return 0;
}

void *thread_work_0(void *rank) //求得delta
{
    pthread_mutex_lock(&mutex);
    delta = b * b - 4 * a * c;
    if (delta < 0)
    {
        printf("No solution!\n");
        exit(0);
    }
    pthread_cond_signal(&cond_var1); //解锁，使得1号线程计算可以进行
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void *thread_work_1(void *rank) //求得分母
{
    pthread_mutex_lock(&mutex);
    while (pthread_cond_wait(&cond_var1, &mutex) != 0); //阻塞线程直到被解锁
    denominator1 = (double)-1 * b - sqrt(delta);
    denominator2 = (double)-1 * b + sqrt(delta);
    pthread_cond_signal(&cond_var2); //解锁，使得2号线程计算可以进行
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void *thread_work_2(void *rank) //求得最终结果
{
    pthread_mutex_lock(&mutex);
    while (pthread_cond_wait(&cond_var2, &mutex) != 0)
        ; //阻塞线程直到被解锁
    x1 = denominator1 / (2 * a);
    x2 = denominator2 / (2 * a);
    pthread_mutex_unlock(&mutex);
    return NULL;
}