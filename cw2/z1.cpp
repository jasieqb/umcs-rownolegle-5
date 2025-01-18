#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>

bool f1()
{
    return rand() % 2 == 0;
}

bool f2()
{
    return rand() % 3 != 0;
}

bool f3()
{
    return rand() % 4 == 0;
}

bool f4()
{
    return rand() % 5 != 0;
}

bool f5()
{
    return rand() % 6 == 0;
}

bool executeFunctions()
{
    std::vector<bool (*)(void)> functions = {f1, f2, f3, f4, f5};
    bool allSuccessful = true;

#pragma omp parallel for shared(allSuccessful)
    for (size_t i = 0; i < functions.size(); ++i)
    {
        if (!functions[i]())
        {
#pragma omp critical
            {
                allSuccessful = false;
            }
        }
    }

    return allSuccessful;
}

int main()
{
    if (executeFunctions())
    {
        std::cout << "All functions succeeded.\n";
    }
    else
    {
        std::cout << "At least one function failed.\n";
    }

    return 0;
}
