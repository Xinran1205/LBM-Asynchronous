# LBM-Asynchronous: 完全异步格子玻尔兹曼方法实现

[English Version](README.md)

## 项目概述

本项目实现了完全异步的格子玻尔兹曼方法（LBM）算法，用于计算流体力学仿真。该工作展示了异步计算如何在保持数值精度的同时，在高性能计算环境中实现显著的加速。

### 主要成就
- **2.2倍加速**：在128×128网格上相比同步MPI实现
- **<1%最终状态误差**：相比同步参考实现
- **完全异步算法**：使用过时边界区域数据
- **全面验证**：在IRIDIS 5集群上进行测试

## 项目结构

项目采用系统化的开发方法，从基线实现逐步发展到完全异步算法：

### 1. 基线实现

#### `SerialCode/`
- **目的**：串行基线实现，作为参考
- **内容**：
  - `d2q9-bgk.c`：单线程LBM实现
  - `Makefile`：串行编译构建配置
- **用途**：提供数值验证的基准真值
- **引用**：基于布里斯托大学高性能计算课程代码 [UoB-HPC/advanced-hpc-lbm](https://github.com/UoB-HPC/advanced-hpc-lbm)

#### `OpenMP/`
- **目的**：OpenMP并行基线实现
- **内容**：
  - `d2q9-bgk.c`：使用OpenMP指令的多线程LBM
  - `Makefile`：包含OpenMP标志的构建配置
  - `job_submit_d2q9-bgk`：集群作业提交脚本
  - `env.sh`：环境设置脚本
- **用途**：共享内存并行基线，用于性能比较

#### `MPI/`
- **目的**：同步MPI基线实现
- **内容**：
  - `d2q9-bgk.c`：使用阻塞MPI调用的分布式内存LBM
  - `Makefile`：MPI编译构建配置
  - `job_submit_d2q9-bgk`：集群作业提交脚本
- **用途**：使用传统同步通信的分布式内存基线

### 2. 异步实现

#### `MPI_Waitall/`
- **目的**：使用非阻塞通信的半异步实现
- **内容**：
  - `d2q9-bgk.c`：使用MPI_Isend/MPI_Irecv和MPI_Waitall的LBM
  - `Makefile`：构建配置
  - `job_submit_d2q9-bgk`：作业提交脚本
- **关键特性**：将内部计算与通信重叠
- **用途**：同步和完全异步之间的中间步骤

#### `MPI_Testall_OptimizedVersion/`
- **目的**：完全异步实现（最终版本）
- **内容**：
  - `d2q9-bgk.c`：使用MPI_Testall和过时边界区域的LBM
  - `Makefile`：构建配置
  - `job_submit_d2q9-bgk`：作业提交脚本
- **关键特性**：
  - 用MPI_Testall替换MPI_Waitall
  - 允许使用过时边界数据进行计算
  - 实现最大异步性
- **用途**：性能评估的主要实现

#### `MPI_Testall_ComplexVersion/`
- **目的**：初始复杂实现（已废弃）
- **内容**：
  - `d2q9-bgk.c`：异步算法的早期复杂版本
  - `Makefile`：构建配置
- **状态**：已被优化版本取代
- **用途**：仅作为历史参考

### 3. 验证和测试

#### `check/`
- **目的**：数值验证和正确性测试
- **内容**：
  - `check.py`：比较仿真结果的Python脚本
  - `*.av_vels.dat`：不同网格大小的平均速度数据文件
  - `*.final_state.dat`：用于验证的最终状态数据文件
- **用途**：验证所有实现都产生数值正确的结果
- **网格大小**：128×128, 128×256, 256×256, 1024×1024
- **引用**：布里斯托大学HPC课程的验证脚本 [UoB-HPC/advanced-hpc-lbm](https://github.com/UoB-HPC/advanced-hpc-lbm)

#### `dataSet/`
- **目的**：输入数据和配置文件
- **内容**：
  - `input_*.params`：不同网格大小的参数文件
  - `obstacles_*.dat`：障碍物配置文件
- **用途**：提供仿真参数和几何定义

### 4. 可视化和分析

#### `Visualization/`
- **目的**：动画和性能分析工具
- **内容**：
  - `animation.py`：创建流体流动的动画可视化
  - `visualize_4plots.py`：生成多面板可视化图表
  - `plo.py`：性能绘图和加速分析
  - `async_speedup.png`：性能比较图表
- **用途**：仿真结果和性能指标的可视化分析

## 算法演进

1. **串行基线**：单线程参考实现
2. **OpenMP并行**：共享内存并行化
3. **同步MPI**：使用阻塞通信的分布式内存
4. **半异步**：使用MPI_Waitall的非阻塞通信
5. **完全异步**：使用MPI_Testall和过时数据的非阻塞通信

## 关键技术创新

- **过时边界区域**：允许使用过时的边界数据进行计算
- **非阻塞通信**：重叠计算和通信
- **MPI_Testall优化**：用轮询替换阻塞同步
- **可视化驱动优化**：使用动画分析改进算法稳定性

## 性能结果

| 网格大小 | 同步MPI | 半异步 | 完全异步 | 加速比 |
|----------|---------|--------|----------|--------|
| 128×128  | 0.907s  | 0.859s | 0.413s   | 2.2×   |
| 128×256  | 2.845s  | 2.511s | 1.421s   | 2.0×   |
| 256×256  | 6.520s  | 5.388s | 3.425s   | 1.9×   |
| 1024×1024| 16.666s | 13.731s| 11.675s  | 1.4×   |

## 使用说明

### 构建和运行

1. **串行版本**：
   ```bash
   cd SerialCode
   make
   ./d2q9-bgk ../dataSet/input_128x128.params ../dataSet/obstacles_128x128.dat
   ```

2. **OpenMP版本**：
   ```bash
   cd OpenMP
   make
   ./d2q9-bgk ../dataSet/input_128x128.params ../dataSet/obstacles_128x128.dat
   ```

3. **MPI版本**：
   ```bash
   cd MPI_Testall_OptimizedVersion
   make
   mpirun -np 4 ./d2q9-bgk ../dataSet/input_128x128.params ../dataSet/obstacles_128x128.dat
   ```

### 验证

```bash
# 运行验证检查
make check
```

### 可视化

```bash
cd Visualization
python3 animation.py
python3 plo.py
```

## 研究影响

这项工作展示了异步算法在计算流体力学中的实际可行性，提供了：

- **性能提升**：在保持精度的同时实现显著加速
- **可扩展性**：在不同网格大小和核心数上有效
- **实用见解**：HPC应用的现实世界实现策略
- **验证框架**：异步算法的综合测试方法

## 未来工作

项目为以下方面奠定了基础：
- 异步LBM算法的进一步优化
- 扩展到3D仿真
- 与现代HPC框架的集成
- 应用于其他计算物理问题

## 致谢

串行基线实现（`SerialCode/`）和验证框架（`check/`）基于布里斯托大学高性能计算课程材料：

> UoB-HPC. advanced-hpc-lbm: COMS30006- Advanced High Performance Computing- Lattice Boltzmann, n.d. Computer software. URL: https://github.com/UoB-HPC/advanced-hpc-lbm

原始代码采用D2Q9-BGK方案模拟盖驱动腔体问题，已在Blue Crystal Phase 4集群上验证。仿真结果符合预期的守恒特性和收敛行为。

## 引用

如果您在研究中使用了此代码，请引用相关论文并承认此实现。

---

*本项目作为高性能计算研究的一部分开发，展示了异步算法在计算流体力学中的实际优势。*
