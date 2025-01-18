Pamięć w karcie graficznej (GPU) jest zorganizowana hierarchicznie, co pozwala na efektywne zarządzanie danymi w obliczeniach równoległych. Oto szczegółowy podział pamięci w GPU:

---

### **1. Pamięć globalna (Global Memory)**
- **Zakres:** Dostępna dla wszystkich wątków na GPU, w każdym bloku i siatce.
- **Charakterystyka:**
  - Główna pamięć GPU (RAM).
  - Stosunkowo wolna (latencja dostępu rzędu setek cykli zegara).
  - Duża pojemność (gigabajty).
  - Służy do przechowywania danych pomiędzy hostem (CPU) a urządzeniem (GPU).
- **Zastosowanie:**
  - Dane wejściowe i wyjściowe przetwarzane przez GPU.
  - Tablice, macierze, bufory tekstur.
- **Optymalizacja:**
  - Dostęp powinien być współbieżny i ułożony liniowo (coalesced), aby minimalizować opóźnienia.

---

### **2. Pamięć współdzielona (Shared Memory)**
- **Zakres:** Wspólna dla wszystkich wątków w jednym bloku.
- **Charakterystyka:**
  - Znajduje się na chipie (on-chip), co czyni ją szybszą niż pamięć globalna.
  - Ograniczona pojemność (zwykle od kilkudziesięciu do kilkuset kilobajtów na blok).
  - Dostęp do tej pamięci jest szybki, ale wymaga synchronizacji między wątkami (np. `__syncthreads()` w CUDA).
- **Zastosowanie:**
  - Tymczasowe przechowywanie danych potrzebnych dla wątków jednego bloku.
  - Optymalizacja algorytmów wymagających wielokrotnego odczytu tych samych danych.
- **Optymalizacja:**
  - Należy unikać konfliktów bankowych (ang. bank conflicts), które mogą zmniejszyć przepustowość.

---

### **3. Pamięć lokalna (Local Memory)**
- **Zakres:** Prywatna dla każdego wątku.
- **Charakterystyka:**
  - Przechowuje zmienne lokalne danego wątku, które nie mieszczą się w rejestrach.
  - Implementowana w pamięci globalnej, więc dostęp do niej jest wolny.
- **Zastosowanie:**
  - Zmienne o dużych rozmiarach lub indeksowane w sposób dynamiczny.
- **Optymalizacja:**
  - Unikać nadmiernego korzystania z pamięci lokalnej, gdyż prowadzi to do wzrostu latencji.

---

### **4. Rejestry (Registers)**
- **Zakres:** Prywatna pamięć dla każdego wątku.
- **Charakterystyka:**
  - Najszybszy rodzaj pamięci, ale bardzo ograniczony (kilkadziesiąt do kilkuset rejestrów na wątek).
  - Wysoka przepustowość.
- **Zastosowanie:**
  - Przechowywanie zmiennych tymczasowych używanych przez wątek.
- **Optymalizacja:**
  - Zbyt duże użycie rejestrów na wątek prowadzi do rozlania do pamięci lokalnej, co spowalnia obliczenia.

---

### **5. Pamięć stała (Constant Memory)**
- **Zakres:** Dostępna dla wszystkich wątków.
- **Charakterystyka:**
  - Przechowuje dane stałe, które nie zmieniają się w trakcie wykonania kernela.
  - Ograniczona pojemność (zwykle 64 kB).
  - Szybki dostęp, gdy wszystkie wątki odczytują tę samą wartość (coalesced).
- **Zastosowanie:**
  - Stałe parametry, np. macierze transformacji lub współczynniki filtrów.
- **Optymalizacja:**
  - Powinna być używana do przechowywania rzadko zmieniających się danych.

---

### **6. Pamięć tekstur i powierzchni (Texture and Surface Memory)**
- **Zakres:** Dostępna dla wszystkich wątków.
- **Charakterystyka:**
  - Służy do obsługi danych w aplikacjach graficznych i obliczeniowych (np. przetwarzanie obrazów).
  - Zoptymalizowana pod kątem odczytu danych o wysokiej lokalności przestrzennej.
- **Zastosowanie:**
  - Dane dwuwymiarowe, takie jak obrazy, tekstury, macierze.

---

### **7. Pamięć stała rejestrowa (L1 Cache) i pamięć cache L2**
- **L1 Cache:**
  - Dostępna dla wątków w jednym bloku.
  - Znajduje się na poziomie jednostki wieloprocesorowej (SM).
  - Ograniczona pojemność (16-48 kB na SM).
- **L2 Cache:**
  - Dostępna dla wszystkich wątków i bloków.
  - Znajduje się pomiędzy pamięcią globalną a L1.
  - Służy do przyspieszenia dostępu do danych w pamięci globalnej.
- **Optymalizacja:**
  - Dane często używane powinny być umieszczane w pamięci współdzielonej lub cache.

---

### **8. Podsumowanie hierarchii pamięci**
| **Rodzaj pamięci** | **Zakres dostępu**       | **Prędkość**       | **Rozmiar**       | **Zastosowanie**                                   |
|---------------------|--------------------------|--------------------|-------------------|---------------------------------------------------|
| **Rejestry**        | Dla jednego wątku        | Najszybsza         | Bardzo mały       | Zmienne lokalne wątku                            |
| **Pamięć współdzielona** | Wspólna dla bloku        | Bardzo szybka      | Kilkadziesiąt kB  | Tymczasowe dane współdzielone przez wątki bloku  |
| **Pamięć globalna** | Dostępna dla wszystkich  | Wolna              | Gigabajty         | Dane wejściowe i wyjściowe kernela               |
| **Pamięć lokalna**  | Dla jednego wątku        | Wolna (globalna)   | Ograniczona       | Zmienne tymczasowe niewchodzące do rejestrów     |
| **Pamięć stała**    | Dostępna dla wszystkich  | Szybka             | 64 kB             | Stałe dane używane przez wszystkie wątki         |
| **L1 Cache**        | Lokalna dla jednostki SM | Szybka             | Kilkadziesiąt kB  | Przyspieszenie dostępu do danych lokalnych       |
| **L2 Cache**        | Globalna dla GPU         | Szybka             | Kilka MB          | Przyspieszenie dostępu do pamięci globalnej      |

Hierarchia ta pozwala na optymalizację dostępu do pamięci, co jest kluczowe dla wydajności GPU w obliczeniach równoległych.
