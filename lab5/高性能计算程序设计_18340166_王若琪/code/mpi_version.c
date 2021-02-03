/******************************************
 * Compile:
 * mpicc -o mpi_version mpi_version.c
 * Run:       
 * mpirun -np <number of threads> mpi_version
 ******************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <sys/time.h>

#define M 2000
#define N 2000

double **u;
double **w;

#define GET_TIME(now)                           \
    {                                           \
        struct timeval t;                       \
        gettimeofday(&t, NULL);                 \
        now = t.tv_sec + t.tv_usec / 1000000.0; \
    }

int calc_n_from_rank(int rank, int size)
{
    int n;
    n = N / size;
    if ((N % size) != 0)
    {
        if (rank == size - 1)
            n += N % size;
    }
    return n;
}

int main(int argc, char *argv[])
{
    int i, j;
    double start, finish;
    int start_col, end_col;
    int iterations;
    int iterations_print;
    int rank;
    int L;
    int R;
    int size;
    int tag = 0;
    int local_m;
    int local_n;
    double mean;
    double diff = 999;
    double epsilon = 0.001;

    double *sendbuf;
    double *recvbuf;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("\n");
        printf("HEATED_PLATE_MPI\n");
        printf("  C/MPI version\n");
        printf("  A program to solve for the steady state temperature distribution\n");
        printf("  over a rectangular plate.\n");
        printf("\n");
        printf("  Spatial grid of %d by %d points.\n", M, N);
        printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
        printf("  Number of threads =              %d\n", size);
    }

    L = (rank == 0) ? (rank + size - 1) : (rank - 1);
    R = (rank + 1) % size;

    local_m = M;
    local_n = calc_n_from_rank(rank, size);

    u = (double **)malloc(sizeof(double *) * local_m);
    for (i = 0; i < local_m; i++)
    {
        u[i] = (double *)malloc(sizeof(double) * (local_n + 2));
    }
    w = (double **)malloc(sizeof(double *) * local_m);
    for (i = 0; i < local_m; i++)
    {
        w[i] = (double *)malloc(sizeof(double) * (local_n + 2));
    }
    sendbuf = (double *)malloc(sizeof(double) * local_m);
    recvbuf = (double *)malloc(sizeof(double) * local_m);

    mean = ((M - 2) * 100.0 * 2 + (N - 2) * 100.0) / (double)((2 * M) + (2 * N) - 4);

    if (rank == 0)
    {
        printf("\n");
        printf("  MEAN = %f\n", mean);
        printf("\n");
        printf(" Iteration  Change\n");
        printf("\n");
    }

    for (i = 0; i < local_m; i++)
    {
        for (j = 1; j < local_n + 1; j++)
        {
            if (i == 0)
                w[i][j] = 0.0;
            else if (i == local_m - 1)
                w[i][j] = 100.0;
            else if ((rank == 0) && j == 1)
                w[i][j] = 100.0;
            else if ((rank == size - 1) && j == local_n)
                w[i][j] = 100.0;
            else
                w[i][j] = mean;
        }
    }

    iterations = 0;
    iterations_print = 1;

    GET_TIME(start);

    while (diff >= epsilon)
    {

        for (i = 0; i < local_m; i++)
            sendbuf[i] = w[i][1];
        MPI_Sendrecv(sendbuf, local_m, MPI_DOUBLE, L, tag,
                     recvbuf, local_m, MPI_DOUBLE, R, tag,
                     MPI_COMM_WORLD, &status);
        for (i = 0; i < local_m; i++)
            w[i][local_n + 1] = recvbuf[i];
        for (i = 0; i < local_m; i++)
            sendbuf[i] = w[i][local_n];
        MPI_Sendrecv(sendbuf, local_m, MPI_DOUBLE, R, tag,
                     recvbuf, local_m, MPI_DOUBLE, L, tag,
                     MPI_COMM_WORLD, &status);
        for (i = 0; i < local_m; i++)
            w[i][0] = recvbuf[i];

        for (i = 0; i < local_m; i++)
        {
            for (j = 0; j < local_n + 2; j++)
            {
                u[i][j] = w[i][j];
            }
        }

        for (i = 1; i < local_m - 1; i++)
        {
            if (rank == 0)
            {
                start_col = 2;
                end_col = local_n;
            }
            else if (rank == size - 1)
            {
                start_col = 1;
                end_col = local_n - 1;
            }
            else
            {
                start_col = 1;
                end_col = local_n;
            }
            for (j = start_col; j < end_col + 1; j++)
            {
                w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;
            }
        }

        diff = 0.0;
        for (i = 1; i < local_m - 1; i++)
        {
            if (rank == 0)
            {
                start_col = 2;
                end_col = local_n;
            }
            else if (rank == size - 1)
            {
                start_col = 1;
                end_col = local_n - 1;
            }
            else
            {
                start_col = 1;
                end_col = local_n;
            }
            for (j = start_col; j < end_col + 1; j++)
            {
                if (diff < fabs(w[i][j] - u[i][j]))
                {
                    diff = fabs(w[i][j] - u[i][j]);
                }
            }
        }

        if (rank == 0)
        {
            int i;
            double temp_diff = 999;
            for (i = 1; i < size; i++)
            {
                MPI_Recv(&temp_diff, 1, MPI_DOUBLE, i, 20, MPI_COMM_WORLD, &status);
                if (temp_diff > diff)
                    diff = temp_diff;
            }
        }
        if (rank != 0)
        {
            MPI_Send(&diff, 1, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD);
        }

        MPI_Bcast(&diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        iterations++;
        if (rank == 0)
        {
            if (iterations == iterations_print)
            {
                printf("  %8d  %f\n", iterations, diff);
                iterations_print = 2 * iterations_print;
            }
        }
    }

    GET_TIME(finish);

    if (rank == 0)
    {
        printf("\n");
        printf("  %8d  %f\n", iterations, diff);
        printf("\n");
        printf("  Error tolerance achieved.\n");
        printf("  Wallclock time = %f\n", finish - start);
        printf("\n");
        printf("\n");
        printf("HEATED_PLATE_MPI:\n");
        printf("  Normal end of execution.\n");
    }

    MPI_Finalize();

    for (i = 0; i < local_m; i++)
    {
        free(u[i]);
        free(w[i]);
    }
    free(u);
    free(w);
    free(sendbuf);
    free(recvbuf);

    return 0;
}
