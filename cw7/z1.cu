#include <cuda.h>
#include <stdio.h>

#define N 1024
#define RADIUS 10

__global__ void process_with_shared(float *array_gpu, float *result_gpu)
{
    __shared__ float shared_array[N];

    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    shared_array[threadIdx.x] = array_gpu[thread_index];
    __syncthreads();

    float product = 1.0f;
    for (int offset = -RADIUS; offset <= RADIUS; ++offset)
    {
        if (threadIdx.x + offset >= 0 && threadIdx.x + offset < blockDim.x)
        {
            product *= shared_array[threadIdx.x + offset];
        }
    }
    result_gpu[thread_index] = product;
}

int main()
{
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    float *array_host = (float *)malloc(N * sizeof(float));
    float *result_host = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i)
    {
        array_host[i] = 1.0f + i % 10; // Dla uproszczenia: 1, 2, 3, ..., 10
    }

    float *array_gpu, *result_gpu;
    cudaMalloc(&array_gpu, N * sizeof(float));
    cudaMalloc(&result_gpu, N * sizeof(float));

    cudaMemcpy(array_gpu, array_host, N * sizeof(float), cudaMemcpyHostToDevice);

    process_with_shared<<<grid_size, block_size>>>(array_gpu, result_gpu);

    cudaDeviceSynchronize();

    cudaMemcpy(result_host, result_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        printf("%f ", result_host[i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }

    free(array_host);
    free(result_host);
    cudaFree(array_gpu);
    cudaFree(result_gpu);

    return 0;
}
