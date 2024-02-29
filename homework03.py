import matplotlib.pyplot as plt

# Results from the CUDA code
matrix_sizes = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
execution_times = [3.259, 15.677, 35.052, 62.327, 97.887, 142.306, 193.890, 256.447]  # Sample execution times in milliseconds

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, execution_times, marker='o', linestyle='-')
plt.title('Matrix Multiplication Performance')
plt.xlabel('Matrix Size (NxN)')
plt.ylabel('Execution Time (milliseconds)')
plt.grid(True)
plt.xticks(matrix_sizes)
plt.tight_layout()
plt.show()
