#!/bin/bash
# Foundation Models Benchmark Runner
# Runs on all 7 datasets (4 ADME + 3 Tox)

echo "========================================================================"
echo "FOUNDATION MODELS BENCHMARK - Starting"
echo "========================================================================"
echo ""
echo "Datasets: 7 (Caco2_Wang, Half_Life_Obach, Clearance_Hepatocyte_AZ,"
echo "              Clearance_Microsome_AZ, tox21, herg, clintox)"
echo "Models: 2 (Morgan Fingerprint, ChemBERTa)"
echo ""
echo "Output: results/foundation_benchmark/benchmark_results_*.csv"
echo "Log: foundation_benchmark.log"
echo ""
echo "========================================================================"
echo ""

python scripts/analyses/benchmark_foundation_models.py 2>&1 | tee foundation_benchmark.log

echo ""
echo "========================================================================"
echo "FOUNDATION MODELS BENCHMARK - Completed"
echo "========================================================================"
echo ""
echo "Check results in: results/foundation_benchmark/"
echo "Check log: foundation_benchmark.log"
