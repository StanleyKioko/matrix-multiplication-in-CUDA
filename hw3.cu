#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication
__global__ void matrixMulCUDA(float *a, float *b, float *c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < size && col < size) {
        for (int k = 0; k < size; ++k) {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

// Host function to perform matrix multiplication on CPU
void matrixMulCPU(float *a, float *b, float *c, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k) {
                sum += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = sum;
        }
    }
}

int main() {
    int minSize = 512;
    int maxSize = 4096;
    int interval = 512;

    printf("Matrix Size\tCPU Time (ms)\tGPU Time (ms)\n");
    for (int size = minSize; size <= maxSize; size += interval) {
        size_t matrix_bytes = size * size * sizeof(float);

        // Allocate memory on host
        float *h_a = (float*)malloc(matrix_bytes);
        float *h_b = (float*)malloc(matrix_bytes);
        float *h_c_cpu = (float*)malloc(matrix_bytes);
        float *h_c_gpu = (float*)malloc(matrix_bytes);

        // Initialize matrices
        for (int i = 0; i < size * size; ++i) {
            h_a[i] = static_cast<float>(rand()) / RAND_MAX;
            h_b[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Allocate memory on device
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, matrix_bytes);
        cudaMalloc(&d_b, matrix_bytes);
        cudaMalloc(&d_c, matrix_bytes);

        // Transfer data from host to device
        cudaMemcpy(d_a, h_a, matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, matrix_bytes, cudaMemcpyHostToDevice);

        // Define grid and block dimensions
        dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 numBlocks((size + TILE_WIDTH - 1) / TILE_WIDTH, (size + TILE_WIDTH - 1) / TILE_WIDTH);

        // Launch CUDA kernel
        clock_t start_gpu = clock();
        matrixMulCUDA<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);
        cudaDeviceSynchronize();
        clock_t end_gpu = clock();
        float gpu_time_ms = 1000.0f * (end_gpu - start_gpu) / CLOCKS_PER_SEC;

        // Transfer results from device to host
        cudaMemcpy(h_c_gpu, d_c, matrix_bytes, cudaMemcpyDeviceToHost);

        // Perform matrix multiplication on CPU for comparison
        clock_t start_cpu = clock();
        matrixMulCPU(h_a, h_b, h_c_cpu, size);
        clock_t end_cpu = clock();
        float cpu_time_ms = 1000.0f * (end_cpu - start_cpu) / CLOCKS_PER_SEC;

        // Compare CPU and GPU results
        bool passed = true;
        for (int i = 0; i < size * size; ++i) {
            if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
                passed = false;
                break;
            }
        }

        if (passed) {
            printf("%dx%d\t%.2f\t%.2f\n", size, size, cpu_time_ms, gpu_time_ms);
        } else {
            printf("Error: Results do not match for size %dx%d\n", size, size);
        }

        // Free memory
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    return 0;
}
