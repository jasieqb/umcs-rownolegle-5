#include <iostream>
#include <omp.h>

double compute_sequence(int n)
{
    double result = -1.0;
    int i;
#pragma omp parallel for reduction(- : result)
    for (i = 2; i <= n; i++)
    {
        result -= 1.0 / i;
    }

    return result;
}

int main()
{
    int n = 1000;
    double result;

    omp_set_num_threads(4);

    result = compute_sequence(n);

    std::cout << "Wynik ciągu dla n = " << n << ": " << result << std::endl;

    return 0;
}
