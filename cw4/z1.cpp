#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <algorithm>

int partition(std::vector<double> &array, int low, int high)
{
    double pivot = array[high];
    int i = low - 1;

    for (int j = low; j < high; ++j)
    {
        if (array[j] <= pivot)
        {
            ++i;
            std::swap(array[i], array[j]);
        }
    }
    std::swap(array[i + 1], array[high]);
    return i + 1;
}

void parallelQuickSort(std::vector<double> &array, int low, int high, int threshold)
{
    if (low < high)
    {

        if (high - low < threshold)
        {
            std::sort(array.begin() + low, array.begin() + high + 1);
            return;
        }

        int pivot = partition(array, low, high);

#pragma omp task shared(array)
        {
            parallelQuickSort(array, low, pivot - 1, threshold);
        }

#pragma omp task shared(array)
        {
            parallelQuickSort(array, pivot + 1, high, threshold);
        }

#pragma omp taskwait
    }
}

int main()
{
    const int SIZE = 10000000;
    const int THRESHOLD = 1000;
    std::vector<double> array(SIZE);

    std::srand(std::time(0));
    for (int i = 0; i < SIZE; ++i)
    {
        array[i] = (rand() % 1000) + 1;
    }

    std::cout << "Sortowanie równoległe QuickSort..." << std::endl;

#pragma omp parallel
    {
#pragma omp single
        {
            parallelQuickSort(array, 0, SIZE - 1, THRESHOLD);
        }
    }

    std::cout << "Sortowanie zakończone. Przykładowe wartości: " << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "10 największych wartości: " << std::endl;
    for (int i = SIZE - 10; i < SIZE; ++i)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
