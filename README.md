## Description
This project is focused on parallel computing techniques and their applications. It includes various algorithms and implementations that demonstrate the principles of parallel processing.

## openMP
OpenMP is an API that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It consists of a set of compiler directives, library routines, and environment variables that influence run-time behavior.

### How to compile and run
```bash
gcc -fopenmp -o <output_file> <source_file>.c
./<output_file>
```

## MPI

MPI (Message Passing Interface) is a standardized and portable message-passing system designed by a group of researchers from academia and industry to function on a wide variety of parallel computers.

### How to compile and run

```bash
mpicc -o <output_file> <source_file>.c
mpirun -np <number_of_processes> ./<output_file>
```

## CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia. It allows software developers to use a CUDA-enabled graphics processing unit (GPU) for general-purpose processing.

### How to compile and run

```bash
nvcc -o <output_file> <source_file>.cu
./<output_file>
```


## Zadania 


### Spis Zadań z Podziałem na Daty

#### **14.10.2024**
- Zadania ze skryptu:
  - 4.6
  - 4.8

#### **21.10.2024**
- Zadania ze skryptu:
  - 4.4
  - 4.5

#### **28.10.2024**
- Zadania ze skryptu:
  - ~~4.4~~
  - ~~4.19 (oba z samodzielną implementacją redukcji)~~

#### **04.11.2024**
- Znaleźć lub wymyślić problem, który da się zrównoleglić za pomocą tasków w OpenMP (i zaimplementować).
  - Propozycja: rozważyć problemy rekurencyjne (z wyłączeniem Fibonacciego).
  - Ważna poprawna struktura programu oraz poprawny końcowy wynik.
  - Przyspieszenie dzięki taskom jest mile widziane, ale nieobowiązkowe.

#### **25.11.2024**
- CUDA: Sumowanie tablic na GPU:
  1. Zaalokować i zainicjalizować dwie tablice (int) na CPU.
  2. Skopiować dane z CPU do GPU.
  3. Napisać kernel sumujący dwie tablice (wynik w trzeciej tablicy).
  4. Skopiować wyniki z GPU do CPU i wyświetlić.

#### **02.12.2024**
- Implementacja redukcji na GPU (CUDA):
  - Zadanie: Sumowanie relatywnie dużej tablicy (np. 100.000 elementów).
  - Wersja redukcji:
    - Wyniki kumulowane po lewej stronie tablicy.
    - Jednowymiarowa tablica i kernel, uruchamiany w wielu blokach (np. 100 bloków po 1024 wątki).

#### **09.12.2024**
- Zadanie na wykorzystanie pamięci **shared** (CUDA):
  1. Zaalokować tablicę na GPU (1024 liczby zmiennoprzecinkowe) i zainicjalizować ją.
  2. Napisać kernel:
     - Przemnożyć każdy element przez jego 10 prawych i 10 lewych sąsiadów.
     - Operacje wykonywać w pamięci shared.
  3. Zwrócić wyniki do tablicy globalnej (uwaga na bariery synchronizacji).

#### **16.12.2024** *(opcjonalne / dodatkowe / na wymianę za inne zadanie)*
- Zadanie na wykorzystanie strumieni w CUDA:
  1. Wybrać problem, który można podzielić na porcje (np. podnoszenie elementów dużej tablicy do kwadratu).
  2. Struktura programu:
     - Jednoczesne przesyłanie i przetwarzanie danych na GPU za pomocą dwóch strumieni.
     - Realizacja etapów A, B, C jednocześnie:
       - A: Przesyłanie danych na GPU.
       - B: Przetwarzanie pierwszej porcji (A).
       - C: Przesyłanie kolejnej porcji (B).
  3. Cele:
     - Skrócenie czasu wykonania programu dzięki jednoczesnemu przetwarzaniu i przesyłaniu danych.

#### **Zadanie MPI**
- Implementacja programu wykorzystującego MPI do wyznaczenia liczby PI metodą Monte Carlo:
  1. Proces 0 rozdziela pracę między procesy.
  2. Procesy wykonują losowania i przesyłają wyniki do procesu 0.
  3. Proces 0 wyznacza wartość liczby PI na podstawie wyników.
