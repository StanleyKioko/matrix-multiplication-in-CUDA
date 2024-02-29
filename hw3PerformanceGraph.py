import matplotlib.pyplot as plt

# Matrix sizes
matrix_sizes = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]

# CPU times (in milliseconds)
cpu_times = [50, 200, 800, 2500, 6000, 12000, 20000, 32000]

# GPU times (in milliseconds)
gpu_times = [10, 40, 150, 500, 1200, 2400, 4000, 6400]

# Plotting the performance graph
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, cpu_times, marker='o', label='CPU Time')
plt.plot(matrix_sizes, gpu_times, marker='s', label='GPU Time')
plt.title('Matrix Multiplication Performance')
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (ms)')
plt.xticks(matrix_sizes, rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
