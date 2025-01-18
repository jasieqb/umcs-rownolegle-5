#include <iostream>
#include <omp.h>

double sredniaArytm(double a[], int n) {
    double suma = 0.0; 

    #pragma omp parallel
    {
        double local_sum = 0.0; 
        #pragma omp for
        for (int i = 0; i < n; i++) {
            local_sum += a[i];
        }
        #pragma omp atomic
        suma += local_sum;
    }
    return suma / n;
}

int main() {
    int n = 10; 
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    double srednia = sredniaArytm(a, n);

    printf("Średnia arytmetyczna: %.2f\n", srednia);

    return 0;
}
