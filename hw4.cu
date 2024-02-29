#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication with shared memory
__global__ void matrixMulCUDA(float *a, float *b, float *c, int width, int height) {
    __shared__ float shared_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        shared_a[ty][tx] = a[Row * width + ph * TILE_WIDTH + tx];
        shared_b[ty][tx] = b[(ph * TILE_WIDTH + ty) * width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Cvalue += shared_a[ty][k] * shared_b[k][tx];
        __syncthreads();
    }

    if (Row < height && Col < width)
        c[Row * width + Col] = Cvalue;
}

// Host function to perform matrix multiplication on CPU
void matrixMulCPU(float *a, float *b, float *c, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += a[i * width + k] * b[k * width + j];
            }
            c[i * width + j] = sum;
        }
    }
}

int main() {
    int width = 1024; // Matrix width
    int height = 512; // Matrix height

    size_t matrix_bytes = width * height * sizeof(float);

    // Allocate memory on host
    float *h_a = (float*)malloc(matrix_bytes);
    float *h_b = (float*)malloc(matrix_bytes);
    float *h_c_cpu = (float*)malloc(matrix_bytes);
    float *h_c_gpu = (float*)malloc(matrix_bytes);

    // Initialize matrices
    for (int i = 0; i < width * height; ++i) {
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
    dim3 numBlocks((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch CUDA kernel
    matrixMulCUDA<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, width, height);
    cudaDeviceSynchronize();

    // Transfer results from device to host
    cudaMemcpy(h_c_gpu, d_c, matrix_bytes, cudaMemcpyDeviceToHost);

    // Perform matrix multiplication on CPU for comparison
    matrixMulCPU(h_a, h_b, h_c_cpu, width, height);

    // Compare CPU and GPU results
    bool passed = true;
    for (int i = 0; i < width * height; ++i) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("GPU and CPU results match.\n");
    } else {
        printf("Error: Results do not match.\n");
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
