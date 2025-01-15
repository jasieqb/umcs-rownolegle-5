#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>

// Przykladowe funkcje logiczne
bool f1() { 
    // Symulacja pracy funkcji
    return rand() % 2 == 0; 
}

bool f2() { 
    // Symulacja pracy funkcji
    return rand() % 3 != 0; 
}

bool f3() { 
    // Symulacja pracy funkcji
    return rand() % 4 == 0; 
}

bool f4() { 
    // Symulacja pracy funkcji
    return rand() % 5 != 0; 
}

bool f5() { 
    // Symulacja pracy funkcji
    return rand() % 6 == 0; 
}

// Główna funkcja
bool executeFunctions() {
    std::vector<bool (*)(void)> functions = {f1, f2, f3, f4, f5};
    bool allSuccessful = true;

    #pragma omp parallel for shared(allSuccessful)
    for (size_t i = 0; i < functions.size(); ++i) {
        if (!functions[i]()) {
            #pragma omp critical
            {
                allSuccessful = false;
            }
        }
    }

    return allSuccessful;
}

int main() {
    if (executeFunctions()) {
        std::cout << "All functions succeeded.\n";
    } else {
        std::cout << "At least one function failed.\n";
    }

    return 0;
}
