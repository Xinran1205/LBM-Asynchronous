# LBM-Asynchronous: Fully Asynchronous Lattice Boltzmann Method Implementation

[中文版本 / Chinese Version](README_CN.md)

## Project Overview

This project implements a fully asynchronous Lattice Boltzmann Method (LBM) algorithm for computational fluid dynamics simulations. The work demonstrates how asynchronous computing can achieve significant speedup while maintaining numerical accuracy in high-performance computing environments.

### Key Achievements
- **2.2× speedup** on 128×128 grid compared to synchronous MPI
- **<1% final-state error** compared to synchronous reference
- **Fully asynchronous algorithm** using stale halo regions
- **Comprehensive validation** on IRIDIS 5 cluster

## Project Structure

The project follows a systematic development approach, progressing from baseline implementations to fully asynchronous algorithms:

### 1. Baseline Implementations

#### `SerialCode/`
- **Purpose**: Serial baseline implementation for reference
- **Contents**: 
  - `d2q9-bgk.c`: Single-threaded LBM implementation
  - `Makefile`: Build configuration for serial compilation
- **Usage**: Provides the ground truth for numerical validation
- **Attribution**: Based on the University of Bristol's high-performance computing course code [UoB-HPC/advanced-hpc-lbm](https://github.com/UoB-HPC/advanced-hpc-lbm)

#### `OpenMP/`
- **Purpose**: OpenMP parallel baseline implementation
- **Contents**:
  - `d2q9-bgk.c`: Multi-threaded LBM using OpenMP directives
  - `Makefile`: Build configuration with OpenMP flags
  - `job_submit_d2q9-bgk`: Job submission script for cluster
  - `env.sh`: Environment setup script
- **Usage**: Shared-memory parallel baseline for performance comparison

#### `MPI/`
- **Purpose**: Synchronous MPI baseline implementation
- **Contents**:
  - `d2q9-bgk.c`: Distributed-memory LBM using blocking MPI calls
  - `Makefile`: Build configuration for MPI compilation
  - `job_submit_d2q9-bgk`: Job submission script for cluster
- **Usage**: Distributed-memory baseline using traditional synchronous communication

### 2. Asynchronous Implementations

#### `MPI_Waitall/`
- **Purpose**: Semi-asynchronous implementation using non-blocking communication
- **Contents**:
  - `d2q9-bgk.c`: LBM with MPI_Isend/MPI_Irecv and MPI_Waitall
  - `Makefile`: Build configuration
  - `job_submit_d2q9-bgk`: Job submission script
- **Key Features**: Overlaps interior computation with communication
- **Usage**: Intermediate step between synchronous and fully asynchronous

#### `MPI_Testall_OptimizedVersion/`
- **Purpose**: Fully asynchronous implementation (final version)
- **Contents**:
  - `d2q9-bgk.c`: LBM with MPI_Testall and stale halo regions
  - `Makefile`: Build configuration
  - `job_submit_d2q9-bgk`: Job submission script
- **Key Features**: 
  - Replaces MPI_Waitall with MPI_Testall
  - Allows computation with stale boundary data
  - Achieves maximum asynchrony
- **Usage**: Main implementation for performance evaluation

#### `MPI_Testall_ComplexVersion/`
- **Purpose**: Initial complex implementation (deprecated)
- **Contents**:
  - `d2q9-bgk.c`: Early complex version of asynchronous algorithm
  - `Makefile`: Build configuration
- **Status**: Superseded by optimized version
- **Usage**: Historical reference only

### 3. Validation and Testing

#### `check/`
- **Purpose**: Numerical validation and correctness testing
- **Contents**:
  - `check.py`: Python script for comparing simulation results
  - `*.av_vels.dat`: Average velocity data files for different grid sizes
  - `*.final_state.dat`: Final state data files for validation
- **Usage**: Validates that all implementations produce numerically correct results
- **Grid Sizes**: 128×128, 128×256, 256×256, 1024×1024
- **Attribution**: Validation script from University of Bristol's HPC course [UoB-HPC/advanced-hpc-lbm](https://github.com/UoB-HPC/advanced-hpc-lbm)

#### `dataSet/`
- **Purpose**: Input data and configuration files
- **Contents**:
  - `input_*.params`: Parameter files for different grid sizes
  - `obstacles_*.dat`: Obstacle configuration files
- **Usage**: Provides simulation parameters and geometry definitions

### 4. Visualization and Analysis

#### `Visualization/`
- **Purpose**: Animation and performance analysis tools
- **Contents**:
  - `animation.py`: Creates animated visualizations of fluid flow
  - `visualize_4plots.py`: Generates multi-panel visualization plots
  - `plo.py`: Performance plotting and speedup analysis
  - `async_speedup.png`: Performance comparison chart
- **Usage**: Visual analysis of simulation results and performance metrics

## Algorithm Progression

1. **Serial Baseline**: Single-threaded reference implementation
2. **OpenMP Parallel**: Shared-memory parallelization
3. **Synchronous MPI**: Distributed-memory with blocking communication
4. **Semi-Asynchronous**: Non-blocking communication with MPI_Waitall
5. **Fully Asynchronous**: Non-blocking communication with MPI_Testall and stale data

## Key Technical Innovations

- **Stale Halo Regions**: Allows computation to proceed with outdated boundary data
- **Non-blocking Communication**: Overlaps computation and communication
- **MPI_Testall Optimization**: Replaces blocking synchronization with polling
- **Visualization-Driven Optimization**: Uses animated analysis to improve algorithm stability

## Performance Results

| Grid Size | Synchronous MPI | Semi-Async | Fully Async | Speedup |
|-----------|----------------|------------|-------------|---------|
| 128×128   | 0.907s         | 0.859s     | 0.413s      | 2.2×    |
| 128×256   | 2.845s         | 2.511s     | 1.421s      | 2.0×    |
| 256×256   | 6.520s         | 5.388s     | 3.425s      | 1.9×    |
| 1024×1024 | 16.666s        | 13.731s    | 11.675s     | 1.4×    |

## Usage Instructions

### Building and Running

1. **Serial Version**:
   ```bash
   cd SerialCode
   make
   ./d2q9-bgk ../dataSet/input_128x128.params ../dataSet/obstacles_128x128.dat
   ```

2. **OpenMP Version**:
   ```bash
   cd OpenMP
   make
   ./d2q9-bgk ../dataSet/input_128x128.params ../dataSet/obstacles_128x128.dat
   ```

3. **MPI Versions**:
   ```bash
   cd MPI_Testall_OptimizedVersion
   make
   mpirun -np 4 ./d2q9-bgk ../dataSet/input_128x128.params ../dataSet/obstacles_128x128.dat
   ```

### Validation

```bash
# Run validation check
make check
```

### Visualization

```bash
cd Visualization
python3 animation.py
python3 plo.py
```

## Research Impact

This work demonstrates the practical viability of asynchronous algorithms in computational fluid dynamics, providing:

- **Performance Gains**: Significant speedup while maintaining accuracy
- **Scalability**: Effective across different grid sizes and core counts
- **Practical Insights**: Real-world implementation strategies for HPC applications
- **Validation Framework**: Comprehensive testing methodology for asynchronous algorithms

## Future Work

The project establishes a foundation for:
- Further optimization of asynchronous LBM algorithms
- Extension to 3D simulations
- Integration with modern HPC frameworks
- Application to other computational physics problems

## Acknowledgments

The serial baseline implementation (`SerialCode/`) and validation framework (`check/`) are based on the University of Bristol's high-performance computing course materials:

> UoB-HPC. advanced-hpc-lbm: COMS30006- Advanced High Performance Computing- Lattice Boltzmann, n.d. Computer software. URL: https://github.com/UoB-HPC/advanced-hpc-lbm

The original code employs the D2Q9-BGK scheme to simulate the lid-driven cavity problem and has been verified on the Blue Crystal Phase 4 cluster. The simulation results conform to expected conservation properties and convergence behavior.

## Citation

If you use this code in your research, please cite the associated paper and acknowledge this implementation.

---

*This project was developed as part of high-performance computing research, demonstrating the practical benefits of asynchronous algorithms in computational fluid dynamics.*