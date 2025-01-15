#include <iostream>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstdlib>

// Sekwencyjne wyszukiwanie maksymalnej wartości
int find_max_sequential(const std::vector<int>& arr, int start, int end) {
    return *std::max_element(arr.begin() + start, arr.begin() + end);
}

// Rekurencyjne równoległe wyszukiwanie maksymalnej wartości
int find_max_parallel(const std::vector<int>& arr, int start, int end) {
    if (end - start <= 1000) { // Gdy segment jest mały, wykonaj sekwencyjnie
        return find_max_sequential(arr, start, end);
    }

    int mid = start + (end - start) / 2;
    int max_left, max_right;

    #pragma omp task shared(max_left) // Rekurencja na lewej części
    max_left = find_max_parallel(arr, start, mid);

    #pragma omp task shared(max_right) // Rekurencja na prawej części
    max_right = find_max_parallel(arr, mid, end);

    #pragma omp taskwait // Synchronizacja tasków

    return std::max(max_left, max_right);
}

int main() {
    const int SIZE = 10000000; // Rozmiar tablicy
    std::vector<int> data(SIZE);

    // Inicjalizacja tablicy losowymi liczbami
    #pragma omp parallel for
    for (int i = 0; i < SIZE; ++i) {
        data[i] = rand() % 1000000;
    }

    // Pomiar czasu dla wersji sekwencyjnej
    double start_time_seq = omp_get_wtime();
    int max_sequential = find_max_sequential(data, 0, SIZE);
    double end_time_seq = omp_get_wtime();

    // Pomiar czasu dla wersji równoległej
    int max_parallel;
    double start_time_par = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            max_parallel = find_max_parallel(data, 0, SIZE);
        }
    }
    double end_time_par = omp_get_wtime();

    // Wyniki
    std::cout << "Sekwencyjne:\n";
    std::cout << "  Maksymalna wartość: " << max_sequential << "\n";
    std::cout << "  Czas wykonania: " << end_time_seq - start_time_seq << " sekund\n";

    std::cout << "\nRównoległe:\n";
    std::cout << "  Maksymalna wartość: " << max_parallel << "\n";
    std::cout << "  Czas wykonania: " << end_time_par - start_time_par << " sekund\n";

    std::cout << "\nPrzyspieszenie: " 
              << (end_time_seq - start_time_seq) / (end_time_par - start_time_par)
              << "x\n";

    return 0;
}
