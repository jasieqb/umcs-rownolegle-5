#include <mpi.h>
#include <cstdlib>
#include <cmath>
#include <iostream>

inline double generate_random()
{
    return rand() / (double)RAND_MAX;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long total_points = 1000000; 
    long points_per_process = total_points / size;
    long inside_circle = 0;
    long global_inside_circle = 0;

    srand(rank + 1);

    for (long i = 0; i < points_per_process; ++i)
    {
        double x = generate_random();
        double y = generate_random();
        if (x * x + y * y <= 1.0)
        {
            inside_circle++;
        }
    }

    MPI_Reduce(&inside_circle, &global_inside_circle, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double pi_estimate = 4.0 * ((double)global_inside_circle) / total_points;
        std::cout << "Szacowana wartość liczby π: " << pi_estimate << std::endl;
    }

    MPI_Finalize();
    return 0;
}
