#include <cuda.h>
#include <stdio.h>

#define N 1024             // Rozmiar danych
#define CHUNK_SIZE (N / 2) // Rozmiar jednej porcji danych

__global__ void kernel_square(int *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] *= data[idx];
    }
}

int main()
{
    int *host_data = (int *)malloc(N * sizeof(int));
    int *device_data1;
    int *device_data2;

    cudaMalloc(&device_data1, CHUNK_SIZE * sizeof(int));
    cudaMalloc(&device_data2, CHUNK_SIZE * sizeof(int));

    // Inicjalizacja danych
    for (int i = 0; i < N; ++i)
    {
        host_data[i] = i + 1;
    }

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemcpyAsync(device_data1, host_data, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);              // A1
    kernel_square<<<(CHUNK_SIZE + 255) / 256, 256, 0, stream1>>>(device_data1, CHUNK_SIZE);                           // B1
    cudaMemcpyAsync(device_data2, host_data + CHUNK_SIZE, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2); // A2

    kernel_square<<<(CHUNK_SIZE + 255) / 256, 256, 0, stream2>>>(device_data2, CHUNK_SIZE);              // B2
    cudaMemcpyAsync(host_data, device_data1, CHUNK_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream1); // C1

    cudaMemcpyAsync(host_data + CHUNK_SIZE, device_data2, CHUNK_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream2); // C2

    // Synchronizacja strumieni
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Wypisanie wyników
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", host_data[i]);
    }

    // Zwolnienie zasobów
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(device_data1);
    cudaFree(device_data2);
    free(host_data);

    return 0;
}
