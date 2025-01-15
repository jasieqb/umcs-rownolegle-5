#include <omp.h>
#include <iostream>

using namespace std;

int main() {
    int threads = 4; // liczba wątków
    int n = 20;      // rozmiar tablic
    int* a = new int[n];
    int* b = new int[n];
    int* c = new int[n];
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }
    // print tablice
    for (int i = 0; i < n; i++) {
        cout << i << " " << a[i] << " " << b[i] << endl;
    }
    #pragma omp parallel num_threads(threads)
    {
        int thread_id = omp_get_thread_num();
        int thread_count = omp_get_num_threads();

        for (int i = thread_id; i < n; i += thread_count) {
            c[i] = a[i] + b[i];
        }
    }
    cout << "Wynik:" << endl;
    for (int i = 0; i < n; i++) {
        cout << i << ": " << c[i] << endl;
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
