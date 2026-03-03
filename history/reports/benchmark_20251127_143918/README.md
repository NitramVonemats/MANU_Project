# Benchmark Report

Generated: 2025-11-27 14:39:31

## Summary

| dataset                 | model    | algorithm   |   test_rmse |    test_mae |   test_r2 |    val_rmse |   train_time |
|:------------------------|:---------|:------------|------------:|------------:|----------:|------------:|-------------:|
| Caco2_Wang              | Best HPO | abc         |  0.00257962 |  0.00192529 |  0.529017 |  0.00381198 |     435.536  |
| Clearance_Hepatocyte_AZ | Best HPO | sa          | 50.2916     | 35.0083     | -0.097495 | 50.836      |      75.0571 |
| Clearance_Microsome_AZ  | Best HPO | sa          | 40.8605     | 24.7475     |  0.100402 | 38.0799     |      60.1326 |
| Half_Life_Obach         | Best HPO | abc         | 20.3678     |  8.47892    |  0.118939 | 15.6429     |      50.7804 |

## Files

- `full_results.csv` - Complete results from all runs
- `summary.csv` - Summary comparison table
- `detailed_comparison.csv` - Detailed comparison sorted by performance
- `performance_comparison.png` - Performance comparison plots
- `dataset_performance.png` - Dataset-specific performance
- `algorithm_comparison.png` - HPO algorithm comparison
