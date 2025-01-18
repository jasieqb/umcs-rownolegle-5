#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

double obliczPi(int liczbaPunktow) {
    int punktyWewnatrzOkregu = 0;

    #pragma omp parallel
    {
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
        #pragma omp atomic
        punktyWewnatrzOkregu += lokalnePunktyWewnatrzOkregu;
    }

    return 4.0 * punktyWewnatrzOkregu / liczbaPunktow;
}

int main() {
    int liczbaPunktow = 10000000; 
    double pi = obliczPi(liczbaPunktow);
    std::cout << "Wyznaczona wartość liczby pi: " << pi << std::endl;

    return 0;
}
