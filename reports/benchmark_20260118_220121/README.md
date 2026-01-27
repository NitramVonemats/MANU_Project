# Benchmark Report

Generated: 2026-01-18 22:01:23

## Summary

| dataset                 | task_type      | model    | algorithm   |    test_rmse |     test_mae |    test_r2 |     val_rmse |   train_time |    test_f1 |   test_auc_roc |   test_accuracy |     val_f1 |   val_auc_roc |
|:------------------------|:---------------|:---------|:------------|-------------:|-------------:|-----------:|-------------:|-------------:|-----------:|---------------:|----------------:|-----------:|--------------:|
| Caco2_Wang              | regression     | Best HPO | abc         |   0.00257962 |   0.00192529 |   0.529017 |   0.00381198 |     435.536  | nan        |     nan        |      nan        | nan        |    nan        |
| Clearance_Hepatocyte_AZ | regression     | Best HPO | sa          |  50.2916     |  35.0083     |  -0.097495 |  50.836      |      75.0571 | nan        |     nan        |      nan        | nan        |    nan        |
| Clearance_Microsome_AZ  | regression     | Best HPO | sa          |  40.8605     |  24.7475     |   0.100402 |  38.0799     |      60.1326 | nan        |     nan        |      nan        | nan        |    nan        |
| Half_Life_Obach         | regression     | Best HPO | abc         |  20.3678     |   8.47892    |   0.118939 |  15.6429     |      50.7804 | nan        |     nan        |      nan        | nan        |    nan        |
| herg                    | classification | Best HPO | hc          | nan          | nan          | nan        | nan          |      17.1171 |   0.884422 |       0.813991 |        0.825758 |   0.851852 |      0.854251 |
| tox21                   | classification | Best HPO | random      | nan          | nan          | nan        | nan          |      66.691  |   0.530612 |       0.73536  |        0.968341 |   0.521739 |      0.697634 |

## Files

- `full_results.csv` - Complete results from all runs
- `summary.csv` - Summary comparison table
- `detailed_comparison.csv` - Detailed comparison sorted by performance
- `performance_comparison.png` - Performance comparison plots
- `dataset_performance.png` - Dataset-specific performance
- `algorithm_comparison.png` - HPO algorithm comparison
