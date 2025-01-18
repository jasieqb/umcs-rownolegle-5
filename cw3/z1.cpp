#include <omp.h>
#include <iostream>

double sredniaArytm(double a[], int n)
{
    double suma = 0.0;

#pragma omp parallel for reduction(+ : suma)
    for (int i = 0; i < n; ++i)
    {
        suma += a[i];
    }

    return suma / n;
}

int main()
{
    const int rozmiar = 10;
    double tablica[rozmiar] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    double wynik = sredniaArytm(tablica, rozmiar);
    std::cout << "Średnia arytmetyczna: " << wynik << std::endl;

    return 0;
}
