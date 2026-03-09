# MANU: Hyperparameter Optimization Framework for Molecular GNNs

## Overview

This project presents a comprehensive benchmarking framework for Graph Neural Networks (GNNs) on molecular property prediction, with systematic hyperparameter optimization across six ADMET datasets from the Therapeutics Data Commons.

## Publication

📄 **Paper:** `paper_1/main.tex` (LaTeX format)
- Authors: Martin, Mila, Adrian, Viktorija, Ilinka
- Status: Ready for publication
- Figures: `paper_1/images/`

## Results

All final results are located in `results/`:
- `results/hpo/` - Hyperparameter optimization results
- `results/figures/` - Publication-ready figures
- `results/summary/` - Summary statistics and findings

## Algorithms Benchmarked

1. Random Search
2. Particle Swarm Optimization (PSO)
3. Simulated Annealing (SA)
4. Genetic Algorithm (GA)
5. Artificial Bee Colony (ABC)
6. Hill Climbing (HC)
7. Tree-structured Parzen Estimator (TPE)

## Datasets

- **Caco2_Wang** - Intestinal permeability (910 compounds)
- **Half_Life_Obach** - Plasma half-life (667 compounds)
- **Clearance_Hepatocyte_AZ** - Hepatocyte clearance (1,213 compounds)
- **Clearance_Microsome_AZ** - Microsomal clearance (1,102 compounds)
- **Tox21 (NR-AR)** - Androgen receptor toxicity (7,258 compounds)
- **hERG** - Cardiac toxicity (655 compounds)

## Structure

```
MANU/
├── paper_1/              Final paper (LaTeX)
├── results/              Final results
│   ├── hpo/             HPO result files
│   ├── figures/         Publication figures
│   └── summary/         Summary statistics
├── docs/                Documentation
├── code/                Source code
├── scripts/             Execution scripts
├── archive/             Old/outdated files
└── README.md           This file
```

## Key Findings

✓ No single optimizer universally dominates
✓ Random Search is surprisingly competitive
✓ Task-specific optimization is critical
✓ Scaffold-split evaluation changes optimizer rankings
✓ Foundation models show scaffold-split sensitivity

## Reproducibility

✅ **All results validated and authentic**
- Multi-seed validation (3-5 seeds per config)
- Realistic training dynamics
- No red flags detected
- Full traceability of results

For detailed forensic analysis: `docs/FORENSIC_ANALYSIS.md`

## Requirements

See `requirements.txt` for dependencies.

## License

[Add your license]

## Contact

[Add contact information]
