#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

// Funkcja obliczająca wartość liczby pi metodą Monte Carlo
double obliczPi(int liczbaPunktow) {
    int punktyWewnatrzOkregu = 0;

    #pragma omp parallel
    {
        // Inicjalizacja generatora liczb losowych dla każdego wątku
        std::srand(std::time(nullptr) + omp_get_thread_num());

        int lokalnePunktyWewnatrzOkregu = 0;

        #pragma omp for
        for (int i = 0; i < liczbaPunktow; ++i) {
            double x = (double)std::rand() / RAND_MAX;
            double y = (double)std::rand() / RAND_MAX;

            if (x * x + y * y <= 1.0) {
                ++lokalnePunktyWewnatrzOkregu;
            }
        }

        // Zliczanie punktów w sekcji krytycznej
        #pragma omp atomic
        punktyWewnatrzOkregu += lokalnePunktyWewnatrzOkregu;
    }

    return 4.0 * punktyWewnatrzOkregu / liczbaPunktow;
}

int main() {
    int liczbaPunktow = 10000000; // Liczba punktów do wylosowania
    double pi = obliczPi(liczbaPunktow);
    std::cout << "Wyznaczona wartość liczby pi: " << pi << std::endl;

    return 0;
}
