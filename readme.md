Goal:
Benchmark and optimize PySCF workloads for quantum chemistry calculations.

Methods:
- CPU profiling with cProfile and line_profiler
- Thread scaling with OMP_NUM_THREADS
- Memory profiling
- Optional GPU comparison with GPU4PySCF

Workloads:
- RHF
- DFT
- FCIDUMP generation
- AO-to-MO integral transformation

Metrics:
- wall time
- SCF iteration time
- memory usage
- speedup
- scaling with basis size

Data:
- from CCCBDB experimental geometry data