# repo759

ECE 759 — High Performance Computing in Engineering Applications (UW–Madison, Fall 2024)

Coursework for ECE 759. The assignments build up from serial C++ baselines, to shared-memory parallelism with **OpenMP**, and finally to GPU programming with **CUDA**.

## Homework overview

| HW | Description | Parallel model |
| --- | --- | --- |
| [HW01](./HW01) | Intro to the cluster: a basic serial C++ program plus a first SLURM batch script. | Serial |
| [HW02](./HW02) | Serial C++ baselines — inclusive scan, matrix multiply (4 loop-order variants), and 2D convolution. | Serial |
| [HW03](./HW03) | Multithreaded matrix multiply, 2D convolution, and a task-based parallel merge sort. | OpenMP |
| [HW04](./HW04) | N-body gravitational simulation parallelized over particles (serial baseline + threaded version). | OpenMP |
| [HW05](./HW05) | Intro CUDA kernels — per-thread factorial, global-memory writes, and vector scaling (`vscale`). | CUDA |
| [HW06](./HW06) | Tiled matrix multiply and a 1D stencil/convolution using shared memory with halo cells. | CUDA |
| [HW07](./HW07) | Shared-memory tiled matrix multiply (int/float/double) and a parallel sum reduction. | CUDA |

## Tally

- **OpenMP:** 2 (HW03, HW04)
- **CUDA:** 3 (HW05, HW06, HW07)
- **Serial C++ baselines:** 2 (HW01, HW02)
