import matplotlib.pyplot as plt
import numpy as np

# Data: Speedup relative to synchronous MPI
grid_sizes = ['128x128', '128x256', '256x256', '1024x1024']
mpi_speedup = [1.0, 1.0, 1.0, 1.0]  # Baseline
waitall_speedup = [0.907/0.859, 2.845/2.511, 6.520/5.388, 16.666/13.731]  # ~1.06, 1.13, 1.21, 1.21
testall_speedup = [0.907/0.413, 2.845/1.421, 6.520/3.425, 16.666/11.675]  # ~2.19, 2.00, 1.90, 1.43

x = np.arange(len(grid_sizes))
fig, ax = plt.subplots()
ax.plot(x, mpi_speedup, marker='o', label='Sync MPI (Baseline)')
ax.plot(x, waitall_speedup, marker='s', label='Semi-Async (MPI_Waitall)')
ax.plot(x, testall_speedup, marker='^', label='Fully Async (MPI_Testall)')

ax.set_xlabel('Grid Size')
ax.set_ylabel('Speedup Relative to Sync MPI')
ax.set_title('Asynchronous Speedup vs. Grid Size (80 Cores)')
ax.set_xticks(x)
ax.set_xticklabels(grid_sizes)
ax.legend()

plt.savefig('async_speedup.png')  # Save as image for paper insertion
plt.show()