#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 16

// Matrix multiplication kernel - modified for non-square thread configurations
__global__ void matrixMul(float *a, float *b, float *c, int width) {
    __shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float c_val = 0.0;

    for (int m = 0; m < width / TILE_WIDTH; ++m) {
        ds_a[ty][tx] = a[row * width + m * TILE_WIDTH + tx];
        ds_b[ty][tx] = b[(m * TILE_WIDTH + ty) * width + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            c_val += ds_a[ty][k] * ds_b[k][tx];
        __syncthreads();
    }

    c[row * width + col] = c_val;
}

// Matrix multiplication CPU version
void matrixMulCPU(float *a, float *b, float *c, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0;
            for (int k = 0; k < width; ++k) {
                sum += a[i * width + k] * b[k * width + j];
            }
            c[i * width + j] = sum;
        }
    }
}

// Function to initialize matrix with random values
void initMatrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            mat[i * size + j] = (float)rand() / RAND_MAX;
        }
    }
}

// Function to compare two matrices
bool compareMatrices(float *mat1, float *mat2, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (fabs(mat1[i * size + j] - mat2[i * size + j]) > 1e-5) {
                printf("Mismatch at (%d, %d): %f != %f\n", i, j, mat1[i * size + j], mat2[i * size + j]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int width = 1024; // Change this to vary the matrix size
    int size = width * width;
    int mem_size = size * sizeof(float);

    // Allocate memory for matrices on host
    float *h_a = (float *)malloc(mem_size);
    float *h_b = (float *)malloc(mem_size);
    float *h_c_cpu = (float *)malloc(mem_size);
    float *h_c_gpu = (float *)malloc(mem_size);

    // Initialize matrices with random values
    initMatrix(h_a, width);
    initMatrix(h_b, width);

    // Allocate memory for matrices on device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, mem_size);
    cudaMalloc((void **)&d_b, mem_size);
    cudaMalloc((void **)&d_c, mem_size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

    // Copy result from device to host
    cudaMemcpy(h_c_gpu, d_c, mem_size, cudaMemcpyDeviceToHost);

    // Matrix multiplication using CPU for comparison
    matrixMulCPU(h_a, h_b, h_c_cpu, width);

    // Check for correctness
    bool result = compareMatrices(h_c_cpu, h_c_gpu, width);
    if (result)
        printf("Matrices match!\n");
    else
        printf("Matrices do not match!\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}
