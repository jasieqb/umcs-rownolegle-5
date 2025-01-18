#include <cuda.h>
#include <stdio.h>

__global__ void reduce_sum(float *input, float *output, int n)
{
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (global_idx + s < n && tid < s)
        {
            input[global_idx] += input[global_idx + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        output[blockIdx.x] = input[global_idx];
    }
}

int main()
{
    const int size = 100000;
    const int block_size = 1024;
    const int grid_size = (size + block_size - 1) / block_size;

    float *h_input = (float *)malloc(size * sizeof(float));
    float *h_output = (float *)malloc(grid_size * sizeof(float));
    float *d_input, *d_output;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, grid_size * sizeof(float));

    for (int i = 0; i < size; ++i)
    {
        h_input[i] = 1.0f; // Dla uproszczenia: same jedynki
    }

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum<<<grid_size, block_size>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    float total_sum = 0.0f;
    for (int i = 0; i < grid_size; ++i)
    {
        total_sum += h_output[i];
    }

    printf("Total sum: %f\n", total_sum);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
