#include <iostream>
#include <cmath>
#include <omp.h>

double f(double x) {
    return x + sin(1.0 / x);
}

double calka(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;

        #pragma omp for
        for (int i = 1; i < n; ++i) {
            double x = a + i * h;
            local_sum += f(x);
        }
        #pragma omp critical
        sum += local_sum;
    }
    sum += (f(a) + f(b)) / 2.0;
    return sum * h;
}

int main() {
    double a = 0.01;
    double b = 1.0;
    int n = 1000000; // Liczba przedziałów
    double result = calka
(a, b, n);
    std::cout << "Wynik całkowania: " << result << std::endl;

    return 0;
}
