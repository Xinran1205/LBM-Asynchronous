import matplotlib.pyplot as plt
import numpy as np

# 数据：相对于同步MPI的加速比
grid_sizes = ['128x128', '256x256', '1024x1024']
mpi_speedup = [1.0, 1.0, 1.0]  # 基线
waitall_speedup = [0.907/0.859, 6.520/5.388, 16.666/13.731]  # 约1.06, 1.21, 1.21
testall_speedup = [0.907/0.413, 6.520/3.425, 16.666/11.675]  # 约2.19, 1.90, 1.43

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

plt.savefig('async_speedup.png')  # 保存为图片，插入论文
plt.show()