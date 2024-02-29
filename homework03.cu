#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMul(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int width;
    for (width = 512; width <= 4096; width += 512) {
        int size = width * width * sizeof(float);
        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C = (float*)malloc(size);

        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);

        // Initialize matrices h_A and h_B with random values
        for (int i = 0; i < width * width; ++i) {
            h_A[i] = rand() / (float)RAND_MAX;
            h_B[i] = rand() / (float)RAND_MAX;
        }

        // Transfer data from host to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Define grid and block dimensions
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

        // Execute the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Transfer results from device to host
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Free host memory
        free(h_A);
        free(h_B);
        free(h_C);

        printf("Matrix size: %dx%d, Execution time: %f milliseconds\n", width, width, milliseconds);
    }

    return 0;
}
