# Supplementary Material: AILS for Min-Max DC-GmTSP

This repository contains the benchmark instances, detailed solution certificates, and statistical summary data for the paper:

**"An Adaptive Iterated Local Search for the Min-Max Multiple Generalized Traveling Salesman Problem"**

## 📂 Repository Structure

The repository is organized as follows:

```text
├── Benchmark_Instances/       # Contains 83 transformed benchmark instances
└── Detailed_Results/          # Experimental results and statistical summaries
    ├── AILS_results/          # Solution files for AILS (10 independent runs per instance)
    ├── ILS_results/           # Solution files for the Baseline ILS (10 independent runs per instance)
    ├── gurobi_results/        # Optimal solutions found by Gurobi (for small-scale instances)
    └── experiment_summary.csv # Aggregated metrics and comparison table
