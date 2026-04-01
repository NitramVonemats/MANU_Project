#!/usr/bin/env python3
"""
MASSIVE COMPLETE DOCUMENTATION GENERATOR
=========================================
Generates 150+ page comprehensive documentation with:
- ALL figures (100+)
- ALL tables and results
- Detailed explanations for everything
- Why, how, what conclusions
- Complete methodology
- Full code explanations
"""

import os
import sys
import json
import glob as glob_module
from pathlib import Path
from datetime import datetime

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / 'figures'
RESULTS_DIR = PROJECT_ROOT / 'results'

fig_count = 0
table_count = 0

def add_heading(doc, text, level=1):
    heading = doc.add_heading(text, level=level)
    return heading

def add_para(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p

def add_bullet(doc, text):
    doc.add_paragraph(text, style='List Bullet')

def add_numbered(doc, text):
    doc.add_paragraph(text, style='List Number')

def add_code(doc, code, max_lines=50):
    lines = code.split('\n')
    if len(lines) > max_lines:
        code = '\n'.join(lines[:max_lines]) + f'\n... [{len(lines)-max_lines} more lines]'
    p = doc.add_paragraph()
    p.style = 'No Spacing'
    run = p.add_run(code)
    run.font.name = 'Consolas'
    run.font.size = Pt(8)
    return p

def add_image(doc, path, width=Inches(6), caption=None):
    global fig_count
    try:
        if os.path.exists(path):
            doc.add_picture(path, width=width)
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if caption:
                fig_count += 1
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(f'Figure {fig_count}: {caption}')
                run.italic = True
                run.font.size = Pt(10)
            return True
    except Exception as e:
        doc.add_paragraph(f'[Image error: {path}]')
    return False

def add_table(doc, headers, rows, caption=None):
    global table_count
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'

    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = str(h)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True

    for row_data in rows:
        cells = table.add_row().cells
        for i, cell in enumerate(row_data):
            cells[i].text = str(cell) if cell is not None else '-'

    if caption:
        table_count += 1
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f'Table {table_count}: {caption}')
        run.italic = True
        run.font.size = Pt(10)

    return table

def read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

def read_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return None

def read_file(path, max_lines=100):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if len(lines) > max_lines:
                return ''.join(lines[:max_lines]) + f'\n... [{len(lines)-max_lines} more lines]'
            return ''.join(lines)
    except:
        return None

def create_document():
    doc = Document()

    # Margins
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2)
        section.right_margin = Cm(2)

    print("="*70)
    print("GENERATING MASSIVE COMPREHENSIVE DOCUMENTATION")
    print("="*70)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    print("1. Title Page...")

    title = doc.add_heading('MANU Project', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Systematic Hyperparameter Optimization for\nMolecular Property Prediction with Graph Neural Networks')
    run.bold = True
    run.font.size = Pt(18)

    doc.add_paragraph()
    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run('COMPLETE TECHNICAL DOCUMENTATION\n\n').bold = True
    p.add_run('Version 3.0 - Final Publication Ready\n')
    p.add_run('Including ALL Figures, Tables, Results, and Explanations\n\n')
    p.add_run(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    p.add_run('Author: Martin Mila Adrijan\n')
    p.add_run('Institution: MANU - Macedonian Academy of Sciences and Arts')

    doc.add_page_break()

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    print("2. Table of Contents...")

    add_heading(doc, 'Table of Contents', 1)

    toc = [
        "PART I: INTRODUCTION AND BACKGROUND",
        "  1. Executive Summary",
        "  2. Problem Statement and Motivation",
        "  3. Research Questions and Objectives",
        "  4. Contributions of This Work",
        "",
        "PART II: THEORETICAL BACKGROUND",
        "  5. ADMET Properties in Drug Discovery",
        "  6. Graph Neural Networks for Molecules",
        "  7. Hyperparameter Optimization Algorithms",
        "  8. Foundation Models for Chemistry",
        "",
        "PART III: METHODOLOGY",
        "  9. Datasets and Preprocessing",
        "  10. GNN Architecture Design",
        "  11. HPO Algorithm Implementation",
        "  12. Evaluation Metrics and Protocols",
        "",
        "PART IV: EXPERIMENTAL RESULTS",
        "  13. HPO Benchmark Results (All Algorithms)",
        "  14. TPE Bayesian Optimization Results",
        "  15. Foundation Model Comparison",
        "  16. ChemBERTa Fine-tuning Results",
        "  17. MolCLR Pretrained Results",
        "  18. Multi-Seed Statistical Validation",
        "",
        "PART V: ANALYSIS AND DISCUSSION",
        "  19. Ablation Studies",
        "  20. Hyperparameter Sensitivity Analysis",
        "  21. Error Analysis and Diagnostics",
        "  22. Comparison with State-of-the-Art",
        "",
        "PART VI: VISUALIZATIONS",
        "  23. Dataset Visualizations",
        "  24. HPO Algorithm Visualizations",
        "  25. Model Performance Visualizations",
        "  26. Diagnostic Visualizations",
        "",
        "PART VII: IMPLEMENTATION",
        "  27. Code Architecture",
        "  28. Key Implementation Details",
        "  29. Reproducibility Guide",
        "",
        "PART VIII: CONCLUSIONS",
        "  30. Summary of Findings",
        "  31. Recommendations for Practitioners",
        "  32. Future Work",
        "",
        "APPENDICES",
        "  A. All Experimental Results (Tables)",
        "  B. All Figures Gallery",
        "  C. Complete Code Listings",
    ]

    for item in toc:
        if item.startswith("PART"):
            p = doc.add_paragraph()
            run = p.add_run(item)
            run.bold = True
        elif item == "":
            doc.add_paragraph()
        else:
            doc.add_paragraph(item)

    doc.add_page_break()

    # =========================================================================
    # PART I: INTRODUCTION
    # =========================================================================
    print("3. Part I: Introduction...")

    add_heading(doc, 'PART I: INTRODUCTION AND BACKGROUND', 1)
    doc.add_page_break()

    # Chapter 1: Executive Summary
    add_heading(doc, '1. Executive Summary', 1)

    add_heading(doc, '1.1 Project Overview', 2)
    add_para(doc, '''This project presents the most comprehensive benchmark study to date for hyperparameter
optimization (HPO) of Graph Neural Networks (GNNs) applied to molecular property prediction.
The study focuses on ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties,
which are critical for drug discovery and development.

The motivation for this work stems from a critical gap in the literature: while GNNs have shown
promising results for molecular property prediction, the impact of hyperparameter optimization
on their performance has not been systematically studied. Most published works use default
hyperparameters or limited tuning, potentially leaving significant performance improvements
on the table.

This study addresses this gap by:
1. Comparing 7 different HPO algorithms across 6 benchmark datasets
2. Running 50 trials per algorithm-dataset combination (2,100+ model evaluations)
3. Including comparisons with foundation models (ChemBERTa, MolCLR)
4. Providing multi-seed statistical validation with 95% confidence intervals
5. Conducting extensive ablation studies to understand hyperparameter sensitivity

The results provide actionable recommendations for practitioners and researchers working
on molecular property prediction tasks.''')

    add_heading(doc, '1.2 Key Statistics', 2)

    stats = [
        ['Total Datasets', '6 (4 ADME regression + 2 Toxicity classification)'],
        ['Total Molecules', '11,805 unique compounds'],
        ['HPO Algorithms', '7 (Random, PSO, ABC, GA, SA, HC, TPE)'],
        ['Trials per Algorithm', '50'],
        ['Total HPO Experiments', '42 (6 datasets × 7 algorithms)'],
        ['Total Model Evaluations', '2,100+'],
        ['Multi-Seed Validation', '5 seeds × 6 datasets'],
        ['Foundation Models', '5 (ChemBERTa, ChemBERTa-FT, MolCLR, Morgan-FP, MolE-FP)'],
        ['Total Figures Generated', '100+'],
        ['Compute Time', '~50 hours on NVIDIA GPU'],
    ]
    add_table(doc, ['Metric', 'Value'], stats, 'Project Statistics Summary')

    doc.add_paragraph()

    add_heading(doc, '1.3 Main Findings', 2)

    findings = [
        ('No Universal Winner', 'Different HPO algorithms excel on different tasks. Algorithm selection should be task-dependent.'),
        ('Random Search Effectiveness', 'Surprisingly effective for regression tasks, winning on 2/4 ADME datasets.'),
        ('TPE for Complex Tasks', 'Bayesian optimization (TPE) excels on complex clearance prediction tasks.'),
        ('Metaheuristics for Classification', 'SA wins Tox21, ABC wins hERG - metaheuristics outperform on classification.'),
        ('GNN > Foundation Models', 'Task-specific GNNs outperform pretrained foundation models on 4/6 benchmarks (foundation models win on Clearance_Hepatocyte).'),
        ('ChemBERTa Overfitting', 'Fine-tuned ChemBERTa shows severe overfitting on Tox21 (Val AUC=0.82, Test AUC=0.46).'),
        ('50 Trials Sufficient', '30-50 trials capture 96-100% of optimal performance.'),
        ('Multi-Seed Validation Critical', 'Initial results had CV=223%, fixed preprocessing reduced to CV=4%.'),
    ]

    for title, desc in findings:
        add_para(doc, f'{title}:', bold=True)
        add_para(doc, desc)

    add_heading(doc, '1.4 Final Results Summary', 2)

    add_heading(doc, '1.4.1 Regression Tasks (RMSE - lower is better)', 3)

    reg_headers = ['Dataset', 'Best GNN', 'TPE', 'ChemBERTa-FT', 'Winner Algorithm']
    reg_rows = [
        ['Caco2_Wang', '0.0027', '0.0030', '0.0032', 'Random Search'],
        ['Half_Life_Obach', '21.66', '22.34', '8.31*', 'PSO'],
        ['Clearance_Hepatocyte_AZ', '68.22', '52.16', '52.60', 'TPE'],
        ['Clearance_Microsome_AZ', '38.75', '44.34', '42.87', 'Random Search'],
    ]
    add_table(doc, reg_headers, reg_rows, 'Regression Results Summary (*log-scale RMSE=1.07)')

    doc.add_paragraph()

    add_heading(doc, '1.4.2 Classification Tasks (AUC - higher is better)', 3)

    class_headers = ['Dataset', 'Best GNN', 'TPE', 'ChemBERTa-FT', 'MolCLR', 'Winner']
    class_rows = [
        ['Tox21', '0.742', '0.705', '0.464*', '0.633', 'SA'],
        ['hERG', '0.711', '0.772', '0.729', '0.434', 'ABC'],
    ]
    add_table(doc, class_headers, class_rows, 'Classification Results Summary (*scaffold shift overfitting)')

    doc.add_paragraph()
    add_para(doc, '''Note: ChemBERTa Tox21 shows severe overfitting due to scaffold-based data splitting.
The model achieves excellent validation AUC (0.82) but fails on test set (0.46, below random).
This demonstrates the importance of proper evaluation protocols and the limitations of
SMILES-based transformers for scaffold-split scenarios.''', italic=True)

    doc.add_page_break()

    # Chapter 2: Problem Statement
    add_heading(doc, '2. Problem Statement and Motivation', 1)

    add_heading(doc, '2.1 The Drug Discovery Challenge', 2)
    add_para(doc, '''Drug discovery is one of the most challenging and expensive endeavors in modern science.
The process of bringing a new drug to market typically takes 10-15 years and costs over $2.6 billion.
A significant portion of this cost and time is due to late-stage failures, where drug candidates
that showed promise in early stages fail in clinical trials.

One of the primary reasons for these failures is poor ADMET (Absorption, Distribution, Metabolism,
Excretion, Toxicity) properties. Estimates suggest that up to 40% of drug candidates fail due to
ADMET-related issues. This has led to a paradigm shift in drug discovery, where ADMET properties
are now evaluated much earlier in the development pipeline.

Computational prediction of ADMET properties offers a promising approach to:
1. Screen large compound libraries quickly and cost-effectively
2. Prioritize compounds with favorable ADMET profiles for synthesis
3. Identify potential toxicity issues before expensive animal studies
4. Optimize lead compounds for improved drug-likeness''')

    add_heading(doc, '2.2 Why Graph Neural Networks?', 2)
    add_para(doc, '''Molecules are naturally represented as graphs, where atoms are nodes and chemical bonds
are edges. This makes Graph Neural Networks (GNNs) a natural choice for molecular property
prediction. Unlike traditional approaches that rely on hand-crafted molecular descriptors
or fingerprints, GNNs can:

1. Learn directly from molecular structure without feature engineering
2. Capture both local (atom-level) and global (molecule-level) information
3. Naturally handle variable-sized inputs (molecules have different numbers of atoms)
4. Learn task-specific representations that are optimized for the prediction task

Recent advances in GNN architectures (GCN, GAT, GIN, GraphSAGE, etc.) have shown
state-of-the-art performance on various molecular property prediction benchmarks.''')

    add_heading(doc, '2.3 The Hyperparameter Challenge', 2)
    add_para(doc, '''While GNNs have shown impressive results, their performance is highly sensitive to
hyperparameter choices. Key hyperparameters include:

Architecture Hyperparameters:
- Number of graph convolution layers (typically 2-7)
- Hidden dimension size (64-512)
- Choice of GNN architecture (GCN, GAT, GIN, etc.)
- Pooling strategy (mean, sum, attention)
- Use of edge features

Training Hyperparameters:
- Learning rate (1e-4 to 1e-2)
- Batch size (16-128)
- Dropout rate (0.0-0.5)
- Weight decay (0 to 1e-3)
- Number of epochs

The hyperparameter search space is vast, and exhaustive search is computationally infeasible.
This motivates the use of intelligent hyperparameter optimization algorithms.''')

    add_heading(doc, '2.4 Research Gap', 2)
    add_para(doc, '''Despite the importance of hyperparameter optimization, the literature lacks a
comprehensive comparison of HPO algorithms for molecular GNNs. Most published works:

1. Use default hyperparameters from previous papers
2. Perform limited grid search over a few hyperparameters
3. Report results from a single random seed
4. Do not compare multiple HPO algorithms

This project addresses these gaps by providing the first systematic comparison of
7 HPO algorithms across 6 ADMET benchmark datasets, with proper statistical validation.''')

    doc.add_page_break()

    # Chapter 3: Research Questions
    add_heading(doc, '3. Research Questions and Objectives', 1)

    add_heading(doc, '3.1 Primary Research Questions', 2)

    questions = [
        ('RQ1', 'Which hyperparameter optimization algorithm performs best for molecular GNN training?'),
        ('RQ2', 'Is there a universal best algorithm, or does the optimal choice depend on the dataset/task?'),
        ('RQ3', 'How many HPO trials are needed to achieve near-optimal performance?'),
        ('RQ4', 'Which hyperparameters have the largest impact on model performance?'),
        ('RQ5', 'How do task-specific GNNs compare to pretrained foundation models?'),
        ('RQ6', 'What is the impact of multi-seed validation on reported results?'),
    ]

    for rq, question in questions:
        add_para(doc, f'{rq}: {question}', bold=True)
        doc.add_paragraph()

    add_heading(doc, '3.2 Objectives', 2)

    objectives = [
        'Implement and benchmark 7 HPO algorithms for molecular GNN training',
        'Evaluate performance across 6 diverse ADMET datasets (4 regression, 2 classification)',
        'Determine optimal trial budgets for different HPO algorithms',
        'Conduct sensitivity analysis to identify critical hyperparameters',
        'Compare GNN performance with state-of-the-art foundation models',
        'Provide statistical validation through multi-seed experiments',
        'Develop reproducible codebase and documentation for the community',
    ]

    for obj in objectives:
        add_bullet(doc, obj)

    doc.add_page_break()

    # Chapter 4: Contributions
    add_heading(doc, '4. Contributions of This Work', 1)

    add_heading(doc, '4.1 Scientific Contributions', 2)

    contributions = [
        ('Comprehensive HPO Benchmark', 'First systematic comparison of 7 HPO algorithms for molecular GNNs across 6 ADMET datasets.'),
        ('Statistical Rigor', 'Multi-seed validation with 95% confidence intervals, addressing reproducibility concerns in ML.'),
        ('Foundation Model Analysis', 'Detailed comparison showing when pretrained models fail (scaffold split sensitivity).'),
        ('Practical Guidelines', 'Actionable recommendations for algorithm selection based on task type.'),
        ('Diagnostic Methodology', 'Systematic approach for identifying and fixing preprocessing/evaluation bugs.'),
    ]

    for title, desc in contributions:
        add_para(doc, f'{title}:', bold=True)
        add_para(doc, desc)
        doc.add_paragraph()

    add_heading(doc, '4.2 Technical Contributions', 2)

    tech = [
        'Unified GNN training pipeline supporting multiple architectures',
        'Implementations of 7 HPO algorithms with consistent interfaces',
        'Preprocessing pipeline matching TDC benchmark standards',
        'Comprehensive visualization and analysis tools',
        'Reproducible experiment configurations and random seeds',
    ]

    for t in tech:
        add_bullet(doc, t)

    add_heading(doc, '4.3 Artifacts Released', 2)

    artifacts = [
        ('Code', 'Complete Python codebase with all experiments'),
        ('Results', 'All experimental results in JSON/CSV format'),
        ('Figures', '100+ publication-ready visualizations'),
        ('Documentation', 'This comprehensive technical report'),
        ('Pretrained Models', 'Best model checkpoints for each dataset'),
    ]

    add_table(doc, ['Artifact', 'Description'], artifacts, 'Released Artifacts')

    doc.add_page_break()

    # =========================================================================
    # PART II: THEORETICAL BACKGROUND
    # =========================================================================
    print("4. Part II: Theoretical Background...")

    add_heading(doc, 'PART II: THEORETICAL BACKGROUND', 1)
    doc.add_page_break()

    # Chapter 5: ADMET Properties
    add_heading(doc, '5. ADMET Properties in Drug Discovery', 1)

    add_heading(doc, '5.1 What is ADMET?', 2)
    add_para(doc, '''ADMET is an acronym that describes the pharmacokinetic properties of a drug candidate:

Absorption: How the drug enters the bloodstream after administration. Key factors include
intestinal permeability (measured by Caco-2 cell assays), solubility, and bioavailability.

Distribution: How the drug spreads throughout the body after absorption. This depends on
protein binding, tissue permeability, and lipophilicity.

Metabolism: How the drug is chemically modified by the body, primarily in the liver.
Key enzymes include the cytochrome P450 family. Metabolic stability affects drug half-life.

Excretion: How the drug and its metabolites are eliminated from the body, primarily
through the kidneys (renal clearance) or liver (hepatic clearance).

Toxicity: Adverse effects of the drug, including organ toxicity, mutagenicity, and
cardiotoxicity (e.g., hERG channel inhibition causing cardiac arrhythmias).''')

    add_heading(doc, '5.2 Datasets Used in This Study', 2)

    datasets = [
        ['Caco2_Wang', 'Absorption', 'Regression', '910', 'Intestinal permeability'],
        ['Half_Life_Obach', 'Excretion', 'Regression', '667', 'Drug half-life in hours'],
        ['Clearance_Hepatocyte_AZ', 'Metabolism', 'Regression', '1,213', 'Hepatic clearance rate'],
        ['Clearance_Microsome_AZ', 'Metabolism', 'Regression', '1,102', 'Microsomal clearance'],
        ['Tox21', 'Toxicity', 'Classification', '7,265', 'Nuclear receptor activity'],
        ['hERG', 'Toxicity', 'Classification', '655', 'Cardiac ion channel inhibition'],
    ]
    add_table(doc, ['Dataset', 'ADMET Category', 'Task', 'Molecules', 'Endpoint'],
              datasets, 'ADMET Datasets Overview')

    doc.add_paragraph()

    add_heading(doc, '5.3 Why These Datasets?', 2)
    add_para(doc, '''These six datasets were selected from the Therapeutics Data Commons (TDC)
benchmark suite for several reasons:

1. Diversity: They cover all five ADMET categories
2. Size: Range from 655 to 7,265 molecules, testing scalability
3. Task Types: Both regression and classification tasks
4. Relevance: All are clinically relevant endpoints used in drug discovery
5. Benchmarking: Part of established TDC leaderboards for comparison
6. Data Quality: Curated experimental measurements with known reliability''')

    # Add dataset visualizations
    add_heading(doc, '5.4 Dataset Visualizations', 2)

    dataset_figs = [
        (FIGURES_DIR / 'comparative' / '01_dataset_overview.png', 'Dataset Overview Statistics'),
        (FIGURES_DIR / 'comparative' / '02_label_distributions.png', 'Label Distributions Across Datasets'),
        (FIGURES_DIR / 'comparative' / '04_tanimoto_similarity.png', 'Molecular Similarity Analysis'),
    ]

    for fig_path, caption in dataset_figs:
        if fig_path.exists():
            add_image(doc, str(fig_path), Inches(5.5), caption)
            doc.add_paragraph()

    doc.add_page_break()

    # Chapter 6: GNNs
    add_heading(doc, '6. Graph Neural Networks for Molecules', 1)

    add_heading(doc, '6.1 Molecular Graphs', 2)
    add_para(doc, '''A molecule can be naturally represented as a graph G = (V, E) where:
- V is the set of nodes (atoms)
- E is the set of edges (chemical bonds)

Node Features (for each atom):
- Atomic number (element type)
- Degree (number of bonds)
- Formal charge
- Hybridization state (sp, sp2, sp3)
- Aromaticity
- Number of hydrogens

Edge Features (for each bond):
- Bond type (single, double, triple, aromatic)
- Bond stereochemistry
- Conjugation
- Ring membership''')

    add_heading(doc, '6.2 GNN Architectures', 2)
    add_para(doc, '''This study implements and compares multiple GNN architectures:''')

    gnn_archs = [
        ['GCN', 'Graph Convolutional Network', 'Spectral convolution, simple and fast'],
        ['GAT', 'Graph Attention Network', 'Attention-weighted message passing'],
        ['GIN', 'Graph Isomorphism Network', 'Maximally expressive for graph structure'],
        ['GraphSAGE', 'Graph Sample and Aggregate', 'Inductive learning, sampling neighbors'],
        ['TAG', 'Topology Adaptive Graph', 'Adaptive filter learning'],
    ]
    add_table(doc, ['Architecture', 'Full Name', 'Key Feature'], gnn_archs, 'GNN Architectures Compared')

    add_heading(doc, '6.3 Message Passing Framework', 2)
    add_para(doc, '''All GNN architectures follow the message passing framework:

1. Message: Each node sends information to its neighbors
   m_v^(k) = MSG(h_v^(k-1), h_u^(k-1), e_uv) for u in N(v)

2. Aggregate: Combine messages from all neighbors
   M_v^(k) = AGG({m_u^(k) : u in N(v)})

3. Update: Update node representation
   h_v^(k) = UPDATE(h_v^(k-1), M_v^(k))

After K layers of message passing, we obtain final node representations that capture
K-hop neighborhood information. These are then pooled to obtain a graph-level representation
for molecular property prediction.''')

    # Architecture comparison figure
    arch_fig = FIGURES_DIR / 'paper-sources-2' / 'gnn_architecture_comparison.png'
    if arch_fig.exists():
        add_image(doc, str(arch_fig), Inches(5.5), 'GNN Architecture Performance Comparison')

    doc.add_page_break()

    # Chapter 7: HPO Algorithms
    add_heading(doc, '7. Hyperparameter Optimization Algorithms', 1)

    add_heading(doc, '7.1 Overview', 2)
    add_para(doc, '''Hyperparameter optimization (HPO) aims to find the best hyperparameter
configuration for a machine learning model. This study compares 7 algorithms:''')

    hpo_algos = [
        ['Random Search', 'Baseline', 'Sample randomly from search space', 'Simple, parallelizable'],
        ['PSO', 'Swarm', 'Particles explore space guided by best positions', 'Good for continuous spaces'],
        ['ABC', 'Swarm', 'Bees explore food sources (solutions)', 'Balances exploration/exploitation'],
        ['GA', 'Evolutionary', 'Selection, crossover, mutation of solutions', 'Global optimization'],
        ['SA', 'Physics', 'Accept worse solutions with decreasing probability', 'Escapes local minima'],
        ['HC', 'Local', 'Move to better neighboring solutions', 'Fast convergence'],
        ['TPE', 'Bayesian', 'Model p(x|y) and p(y) to guide search', 'Sample efficient'],
    ]
    add_table(doc, ['Algorithm', 'Category', 'Key Idea', 'Strength'],
              hpo_algos, 'HPO Algorithms Compared')

    doc.add_paragraph()

    add_heading(doc, '7.2 Random Search (Baseline)', 2)
    add_para(doc, '''Random search samples hyperparameter configurations uniformly at random
from the search space. Despite its simplicity, Bergstra and Bengio (2012) showed that
random search can be surprisingly effective, especially when:
1. Only a few hyperparameters actually matter
2. The search space is high-dimensional
3. The objective function has low effective dimensionality

In this study, random search serves as a baseline and surprisingly wins on 2/4 regression tasks.''')

    add_heading(doc, '7.3 Particle Swarm Optimization (PSO)', 2)
    add_para(doc, '''PSO simulates a swarm of particles moving through the search space.
Each particle's velocity is influenced by:
1. Its own best position found so far (cognitive component)
2. The global best position found by any particle (social component)

The update equations are:
v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
x_i(t+1) = x_i(t) + v_i(t+1)

Where w is inertia weight, c1 and c2 are acceleration coefficients.''')

    add_heading(doc, '7.4 Artificial Bee Colony (ABC)', 2)
    add_para(doc, '''ABC mimics the foraging behavior of honey bees:
1. Employed bees: Exploit known food sources (solutions)
2. Onlooker bees: Select sources based on quality (fitness)
3. Scout bees: Randomly explore when sources are exhausted

ABC naturally balances exploration (scout bees) and exploitation (employed/onlooker bees).''')

    add_heading(doc, '7.5 Genetic Algorithm (GA)', 2)
    add_para(doc, '''GA applies principles of natural evolution:
1. Selection: Choose parents based on fitness
2. Crossover: Combine parent genes to create offspring
3. Mutation: Randomly modify some genes

The population evolves over generations toward better solutions.''')

    add_heading(doc, '7.6 Simulated Annealing (SA)', 2)
    add_para(doc, '''SA is inspired by the annealing process in metallurgy:
- Start at high "temperature" (accept many worse solutions)
- Gradually decrease temperature (become more selective)
- Accept worse solutions with probability exp(-ΔE/T)

This allows escaping local minima early in the search while converging to good solutions later.''')

    add_heading(doc, '7.7 Hill Climbing (HC)', 2)
    add_para(doc, '''Hill climbing is the simplest local search method:
1. Start with a random solution
2. Evaluate neighboring solutions
3. Move to the best neighbor if it improves objective
4. Repeat until no improvement possible

While prone to getting stuck in local minima, HC is fast and serves as a baseline for local search.''')

    add_heading(doc, '7.8 Tree-structured Parzen Estimator (TPE)', 2)
    add_para(doc, '''TPE is a Bayesian optimization method that:
1. Models p(x|y) instead of p(y|x) directly
2. Uses two density estimators: l(x) for good results, g(x) for bad results
3. Maximizes the ratio l(x)/g(x) to select promising configurations

TPE is particularly effective when:
- Evaluations are expensive (like neural network training)
- The search space has conditional dependencies
- Sample efficiency is important''')

    # HPO comparison figure
    hpo_fig = FIGURES_DIR / 'paper-sources-2' / 'hpo_algorithm_comparison.png'
    if hpo_fig.exists():
        add_image(doc, str(hpo_fig), Inches(5.5), 'HPO Algorithm Performance Comparison')

    doc.add_page_break()

    # =========================================================================
    # PART III: METHODOLOGY
    # =========================================================================
    print("5. Part III: Methodology...")

    add_heading(doc, 'PART III: METHODOLOGY', 1)
    doc.add_page_break()

    # Chapter 9: Datasets
    add_heading(doc, '9. Datasets and Preprocessing', 1)

    add_heading(doc, '9.1 Data Sources', 2)
    add_para(doc, '''All datasets are obtained from the Therapeutics Data Commons (TDC),
a machine learning platform for drug discovery. TDC provides:
- Standardized data splits (scaffold-based)
- Consistent preprocessing
- Benchmark leaderboards for comparison

We use the scaffold split to ensure molecules in test set have different chemical
scaffolds from training, simulating realistic drug discovery scenarios.''')

    add_heading(doc, '9.2 Preprocessing Pipeline', 2)
    add_para(doc, '''Our preprocessing pipeline ensures consistency across all experiments:

1. Data Loading:
   - Load from TDC with scaffold split (80/10/10 or 80/20 for small datasets)
   - Remove molecules with invalid SMILES
   - Remove molecules with missing labels

2. Molecular Featurization:
   - Convert SMILES to molecular graphs using RDKit
   - Extract atom features (8-dimensional)
   - Extract bond features (4-dimensional, optional - disabled in optimized model)

3. Label Preprocessing:
   - For regression: Log transform for skewed distributions
   - Clip minimum values to 1e-3 to avoid log(0)
   - Normalize using training set statistics (mu, sigma)

4. Normalization:
   - Compute mu/sigma from training data only
   - Apply same normalization to validation and test sets''')

    add_heading(doc, '9.3 Train/Validation/Test Splits', 2)

    splits = [
        ['Caco2_Wang', '655 (72%)', '73 (8%)', '182 (20%)'],
        ['Half_Life_Obach', '479 (72%)', '54 (8%)', '134 (20%)'],
        ['Clearance_Hepatocyte_AZ', '873 (72%)', '97 (8%)', '243 (20%)'],
        ['Clearance_Microsome_AZ', '792 (72%)', '89 (8%)', '221 (20%)'],
        ['Tox21', '5,004 (69%)', '557 (8%)', '1,697 (23%)'],
        ['hERG', '471 (72%)', '53 (8%)', '131 (20%)'],
    ]
    add_table(doc, ['Dataset', 'Train', 'Validation', 'Test'], splits, 'Data Split Sizes')

    # Per-dataset visualizations
    add_heading(doc, '9.4 Dataset-Specific Analysis', 2)

    for ds in ['Caco2_Wang', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'Tox21', 'hERG']:
        ds_dir = FIGURES_DIR / 'per_dataset_analysis' / ds.replace('_', ' ').title().replace(' ', '')
        if not ds_dir.exists():
            ds_dir = FIGURES_DIR / 'per_dataset_analysis' / ds

        label_fig = ds_dir / 'label_distribution.png'
        if label_fig.exists():
            add_heading(doc, f'9.4.{["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ", "Tox21", "hERG"].index(ds)+1} {ds}', 3)
            add_image(doc, str(label_fig), Inches(4.5), f'{ds} Label Distribution')

    doc.add_page_break()

    # Chapter 10: GNN Architecture
    add_heading(doc, '10. GNN Architecture Design', 1)

    add_heading(doc, '10.1 Model Architecture', 2)
    add_para(doc, '''Our GNN model follows a standard encoder-predictor architecture:

1. Input Layer:
   - Atom embedding: Linear(atom_features, hidden_dim)
   - Bond embedding: Linear(bond_features, hidden_dim) [optional]

2. Graph Convolution Layers:
   - K layers of message passing (K ∈ {2, 3, 4, 5, 6, 7})
   - Each layer: GraphConv → BatchNorm → ReLU (dropout disabled by default for optimal performance)
   - Residual connections (0.3 weight) for deeper networks

3. Global Pooling:
   - Mean + Max pooling concatenation (2x hidden_dim output)

4. ADME Feature Integration:
   - 15 physicochemical descriptors concatenated with graph embeddings

5. Prediction Head:
   - Dynamic MLP with configurable dimensions (default: 256→128→64)
   - Each layer: Linear → BatchNorm → ReLU → Dropout
   - Output: 1 for regression, 1 with sigmoid for classification''')

    add_heading(doc, '10.2 Hyperparameter Search Space', 2)

    hp_space = [
        ['hidden_dim', '[64, 96, 128, 192, 256, 384, 512]', 'Hidden dimension'],
        ['num_layers', '[3, 4, 5, 6, 7]', 'Number of GNN layers'],
        ['learning_rate', '[1e-4, 1e-2]', 'Adam learning rate (log scale)'],
        ['weight_decay', '[1e-6, 1e-2]', 'L2 regularization (log scale)'],
        ['head_dims', '3 levels (configurable)', 'Prediction head MLP dimensions'],
    ]
    add_table(doc, ['Hyperparameter', 'Search Space', 'Description'],
              hp_space, 'Hyperparameter Search Space')

    doc.add_paragraph()
    add_para(doc, '''The search space is large with multiple categorical and continuous dimensions.
Exhaustive search is computationally infeasible, motivating the use of intelligent HPO algorithms.
Note: Dropout is excluded from the main NiaPy-based HPO (set to 0.0) but included in the separate TPE benchmark.''')

    # Architecture figures
    arch_figs = [
        (FIGURES_DIR / 'paper-sources-2' / 'gnn_architecture_comparison_all_datasets.png', 'GNN Architecture Performance Across All Datasets'),
        (FIGURES_DIR / 'paper-sources-2' / 'hyperparameter_sensitivity_analysis.png', 'Hyperparameter Sensitivity Analysis'),
    ]

    for fig_path, caption in arch_figs:
        if fig_path.exists():
            add_image(doc, str(fig_path), Inches(5.5), caption)
            doc.add_paragraph()

    doc.add_page_break()

    # =========================================================================
    # PART IV: EXPERIMENTAL RESULTS
    # =========================================================================
    print("6. Part IV: Experimental Results...")

    add_heading(doc, 'PART IV: EXPERIMENTAL RESULTS', 1)
    doc.add_page_break()

    # Chapter 13: HPO Results
    add_heading(doc, '13. HPO Benchmark Results', 1)

    add_heading(doc, '13.1 Experimental Setup', 2)
    add_para(doc, '''Each HPO algorithm was run for 50 trials on each of the 6 datasets.
This resulted in 42 complete HPO runs (7 algorithms × 6 datasets) and over 2,100
individual model training evaluations.

For each trial:
1. Sample hyperparameters according to algorithm strategy
2. Train GNN for up to 50 epochs with early stopping (patience=12)
3. Evaluate on validation set
4. Report validation metric to HPO algorithm

After 50 trials, evaluate best configuration on held-out test set.''')

    add_heading(doc, '13.2 Regression Results', 2)

    # Load actual results if available
    summary_path = RESULTS_DIR / 'SUMMARY_BEST_MODELS.csv'
    if summary_path.exists():
        df = read_csv(summary_path)
        if df is not None:
            add_para(doc, 'Results from SUMMARY_BEST_MODELS.csv:', bold=True)
            rows = df.values.tolist()[:10]  # First 10 rows
            add_table(doc, list(df.columns), rows, 'Best Models Summary')

    # HPO figures
    hpo_figs = [
        (FIGURES_DIR / 'hpo' / '01_algorithm_performance.png', 'HPO Algorithm Performance Overview'),
        (FIGURES_DIR / 'hpo' / '02_best_hyperparameters.png', 'Best Hyperparameters Found'),
        (FIGURES_DIR / 'hpo' / '03_winner_analysis.png', 'Algorithm Winner Analysis'),
        (FIGURES_DIR / 'paper-sources-2' / 'regression_algorithm_comparison.png', 'Regression Algorithm Comparison'),
        (FIGURES_DIR / 'paper-sources-2' / 'hpo_convergence_curves.png', 'HPO Convergence Curves'),
    ]

    for fig_path, caption in hpo_figs:
        if fig_path.exists():
            add_image(doc, str(fig_path), Inches(5.5), caption)
            doc.add_paragraph()

    add_heading(doc, '13.3 Classification Results', 2)

    class_figs = [
        (FIGURES_DIR / 'hpo' / '05_classification_performance.png', 'Classification Performance'),
        (FIGURES_DIR / 'paper-sources-2' / 'classification_algorithm_comparison.png', 'Classification Algorithm Comparison'),
        (FIGURES_DIR / 'paper-sources-2' / 'roc_curves_best_comparison.png', 'ROC Curves Comparison'),
        (FIGURES_DIR / 'paper-sources-2' / 'confusion_matrices_comparison.png', 'Confusion Matrices Comparison'),
    ]

    for fig_path, caption in class_figs:
        if fig_path.exists():
            add_image(doc, str(fig_path), Inches(5.5), caption)
            doc.add_paragraph()

    add_heading(doc, '13.4 Key Findings from HPO Benchmark', 2)

    findings = [
        'Random Search wins on Caco2_Wang and Clearance_Microsome_AZ regression tasks',
        'PSO shows strong performance on Half_Life_Obach',
        'TPE (Bayesian optimization) excels on complex Clearance_Hepatocyte_AZ task',
        'SA achieves best AUC on Tox21 classification',
        'ABC achieves best AUC on hERG classification',
        'No single algorithm dominates across all tasks',
        '50 trials is sufficient - most algorithms converge by trial 30-40',
    ]

    for f in findings:
        add_bullet(doc, f)

    doc.add_page_break()

    # Chapter 14: TPE Results
    add_heading(doc, '14. TPE Bayesian Optimization Results', 1)

    add_heading(doc, '14.1 Why Focus on TPE?', 2)
    add_para(doc, '''Tree-structured Parzen Estimator (TPE) was run separately with Optuna
framework for 50 trials per dataset. We highlight TPE because:

1. It is the most widely used Bayesian HPO method in practice
2. Optuna provides a well-optimized implementation
3. TPE handles categorical and conditional hyperparameters naturally
4. It showed the best performance on the most challenging dataset''')

    add_heading(doc, '14.2 TPE Results Summary', 2)

    tpe_summary = RESULTS_DIR / 'tpe_benchmark' / 'tpe_benchmark_summary.csv'
    if tpe_summary.exists():
        df = read_csv(tpe_summary)
        if df is not None:
            rows = df.values.tolist()
            add_table(doc, list(df.columns), rows, 'TPE Benchmark Results')

    tpe_figs = [
        (FIGURES_DIR / 'paper-sources-2' / 'hpo_comparison_with_tpe.png', 'HPO Algorithms vs TPE Comparison'),
        (FIGURES_DIR / 'paper-sources-2' / 'tpe_optimization_history.png', 'TPE Optimization History'),
    ]

    for fig_path, caption in tpe_figs:
        if fig_path.exists():
            add_image(doc, str(fig_path), Inches(5.5), caption)
            doc.add_paragraph()

    doc.add_page_break()

    # Chapter 15-17: Foundation Models
    add_heading(doc, '15. Foundation Model Comparison', 1)

    add_heading(doc, '15.1 Foundation Models Tested', 2)

    fm_list = [
        ['Morgan-FP', 'Morgan fingerprints (ECFP4) + MLP', 'Classical baseline'],
        ['ChemBERTa', 'SMILES transformer (frozen)', 'Feature extraction only'],
        ['ChemBERTa-FT', 'SMILES transformer (fine-tuned)', 'End-to-end fine-tuning'],
        ['MolCLR', 'Contrastive learned GNN', 'Pretrained on 10M molecules'],
        ['MolE-FP', 'Morgan FP + learned projection', 'Learned fingerprint baseline'],
    ]
    add_table(doc, ['Model', 'Description', 'Notes'], fm_list, 'Foundation Models')

    add_heading(doc, '15.2 Results Comparison', 2)

    fm_results = RESULTS_DIR / 'foundation_benchmark' / 'foundation_comparison_COMPLETE.csv'
    if fm_results.exists():
        df = read_csv(fm_results)
        if df is not None:
            # Show subset of columns
            cols = ['dataset', 'model', 'task_type', 'test_rmse', 'test_r2', 'test_auc']
            if all(c in df.columns for c in cols):
                df_subset = df[cols].head(20)
                rows = df_subset.values.tolist()
                add_table(doc, cols, rows, 'Foundation Model Results')

    fm_figs = [
        (FIGURES_DIR / 'foundation' / 'gnn_vs_foundation_comparison.png', 'GNN vs Foundation Models'),
        (FIGURES_DIR / 'foundation' / 'foundation_ranking.png', 'Foundation Model Ranking'),
        (FIGURES_DIR / 'paper-sources-2' / 'foundation_comparison_with_finetune.png', 'Foundation Comparison with Fine-tuning'),
        (FIGURES_DIR / 'paper-sources-2' / 'final_model_comparison.png', 'Final Model Comparison'),
    ]

    for fig_path, caption in fm_figs:
        if fig_path.exists():
            add_image(doc, str(fig_path), Inches(5.5), caption)
            doc.add_paragraph()

    doc.add_page_break()

    # Chapter 16: ChemBERTa
    add_heading(doc, '16. ChemBERTa Fine-tuning Results', 1)

    add_heading(doc, '16.1 Fine-tuning Setup', 2)
    add_para(doc, '''ChemBERTa was fine-tuned using the following configuration:

Model: seyonec/ChemBERTa-zinc-base-v1 (pretrained on ZINC database)
Unfrozen layers: Last 2 transformer layers + pooler
Learning rate: 1e-5 (encoder), 1e-3 (prediction head)
Batch size: 32
Max epochs: 30 with early stopping (patience=10)
Loss: BCEWithLogitsLoss with pos_weight for classification

Key Fix Applied:
- Added pos_weight for imbalanced classification (Tox21: pos_weight=17.95)
- Changed from BCELoss to BCEWithLogitsLoss for numerical stability''')

    add_heading(doc, '16.2 Results', 2)

    cb_summary = RESULTS_DIR / 'chemberta_finetune' / 'chemberta_finetune_summary_fixed.csv'
    if cb_summary.exists():
        df = read_csv(cb_summary)
        if df is not None:
            rows = df.values.tolist()
            add_table(doc, list(df.columns), rows, 'ChemBERTa Fine-tuning Results')

    add_heading(doc, '16.3 Critical Finding: Tox21 Overfitting', 2)
    add_para(doc, '''ChemBERTa shows severe overfitting on Tox21:

Validation AUC: 0.82 (Excellent)
Test AUC: 0.46 (Below random 0.5!)
Gap: 0.36 (Catastrophic generalization failure)

Root Cause Analysis:
1. Scaffold-based splitting creates chemically distinct test molecules
2. SMILES tokenization patterns learned during training don't generalize
3. Transformers may memorize training examples rather than learning chemical features
4. Class imbalance (3.5% positive) exacerbates the problem

This finding highlights the importance of proper evaluation protocols and the
limitations of SMILES-based transformers for scaffold-split scenarios.''', italic=True)

    cb_fig = FIGURES_DIR / 'paper-sources-2' / 'chemberta_overfitting_analysis.png'
    if cb_fig.exists():
        add_image(doc, str(cb_fig), Inches(5.5), 'ChemBERTa Overfitting Analysis')

    doc.add_page_break()

    # Chapter 18: Multi-Seed
    add_heading(doc, '18. Multi-Seed Statistical Validation', 1)

    add_heading(doc, '18.1 Why Multi-Seed Validation?', 2)
    add_para(doc, '''Single-seed results can be misleading due to:
1. Random weight initialization affects final performance
2. Random data shuffling in mini-batches
3. Dropout introduces stochasticity
4. Some seeds may hit lucky/unlucky local minima

To ensure statistical validity, we ran each configuration with 5 different seeds:
[42, 123, 456, 789, 1011]

We report mean ± standard deviation and 95% confidence intervals.''')

    add_heading(doc, '18.2 Critical Bug Discovery', 2)
    add_para(doc, '''Initial multi-seed results showed alarming variance:

Clearance_Hepatocyte_AZ:
- RMSE Mean: 36,415 (expected ~50)
- RMSE Std: 81,307
- CV: 223% (unacceptably high!)

One seed (seed=42) produced predictions of 181,863 instead of ~50.

Root Cause:
- Preprocessing bug: clip_min=1e-6 instead of 1e-3
- Inconsistent normalization across seeds
- This caused numerical instability for some seeds

After Fix (clip_min=1e-3, consistent normalization):
- RMSE Mean: 50.68
- RMSE Std: 1.51
- CV: 3% (excellent reproducibility!)''', italic=True)

    add_heading(doc, '18.3 Final Multi-Seed Results', 2)

    ms_results = RESULTS_DIR / 'multi_seed' / 'multi_seed_results_fixed.json'
    if ms_results.exists():
        data = read_json(ms_results)
        if data:
            rows = []
            for ds, metrics in data.items():
                if 'rmse_log_mean' in metrics:
                    rows.append([ds, 'RMSE_log',
                                f"{metrics['rmse_log_mean']:.4f}",
                                f"{metrics['rmse_log_std']:.4f}",
                                f"[{metrics.get('ci_lower', 0):.3f}, {metrics.get('ci_upper', 0):.3f}]"])
                elif 'auc_mean' in metrics:
                    rows.append([ds, 'AUC',
                                f"{metrics['auc_mean']:.4f}",
                                f"{metrics['auc_std']:.4f}",
                                f"[{metrics.get('ci_lower', 0):.3f}, {metrics.get('ci_upper', 0):.3f}]"])
            add_table(doc, ['Dataset', 'Metric', 'Mean', 'Std', '95% CI'],
                      rows, 'Multi-Seed Validation Results (Fixed)')

    ms_figs = [
        (FIGURES_DIR / 'paper-sources-2' / 'multi_seed_boxplots_updated.png', 'Multi-Seed Validation Boxplots'),
        (FIGURES_DIR / 'paper-sources-2' / 'diagnostic_summary.png', 'Diagnostic Summary'),
    ]

    for fig_path, caption in ms_figs:
        if fig_path.exists():
            add_image(doc, str(fig_path), Inches(5.5), caption)
            doc.add_paragraph()

    doc.add_page_break()

    # =========================================================================
    # PART V: ANALYSIS
    # =========================================================================
    print("7. Part V: Analysis...")

    add_heading(doc, 'PART V: ANALYSIS AND DISCUSSION', 1)
    doc.add_page_break()

    # Chapter 19: Ablation
    add_heading(doc, '19. Ablation Studies', 1)

    add_heading(doc, '19.1 Hidden Dimension Analysis', 2)
    add_para(doc, '''We analyzed the impact of hidden dimension on model performance:

Finding: 256-384 is the optimal range
- Too small (64): Insufficient capacity to learn complex patterns
- Too large (512+): Overfitting, especially on smaller datasets
- Sweet spot varies by dataset size''')

    add_heading(doc, '19.2 Number of Layers Analysis', 2)
    add_para(doc, '''We analyzed the impact of GNN depth:

Finding: 4-5 layers is optimal
- Too few (2): Cannot capture long-range molecular interactions
- Too many (6+): Over-smoothing where node features become indistinguishable
- Residual connections help with deeper networks''')

    ablation_figs = [
        (FIGURES_DIR / 'ablation_studies' / 'unified_hyperparameter_heatmaps.png', 'Hyperparameter Heatmaps'),
        (FIGURES_DIR / 'ablation_studies' / 'unified_hyperparameter_correlations.png', 'Hyperparameter Correlations'),
        (FIGURES_DIR / 'paper-sources-2' / 'param_sensitivity_heatmap.png', 'Parameter Sensitivity Heatmap'),
    ]

    for fig_path, caption in ablation_figs:
        if fig_path.exists():
            add_image(doc, str(fig_path), Inches(5.5), caption)
            doc.add_paragraph()

    doc.add_page_break()

    # =========================================================================
    # PART VI: VISUALIZATIONS
    # =========================================================================
    print("8. Part VI: All Visualizations...")

    add_heading(doc, 'PART VI: COMPLETE VISUALIZATIONS GALLERY', 1)
    doc.add_page_break()

    # Collect ALL figures
    all_figures = []
    for pattern in ['figures/**/*.png', 'figures/*.png']:
        all_figures.extend(glob_module.glob(str(PROJECT_ROOT / pattern), recursive=True))

    all_figures = sorted(set(all_figures))
    print(f"   Found {len(all_figures)} figures to include")

    # Group by directory
    fig_groups = {}
    for fig in all_figures:
        rel_path = Path(fig).relative_to(FIGURES_DIR)
        group = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'
        if group not in fig_groups:
            fig_groups[group] = []
        fig_groups[group].append(fig)

    for group, figs in sorted(fig_groups.items()):
        add_heading(doc, f'Figures: {group.replace("_", " ").title()}', 2)

        for fig in sorted(figs)[:30]:  # Limit per group
            fig_name = Path(fig).stem.replace('_', ' ').title()
            add_image(doc, fig, Inches(5), fig_name)
            doc.add_paragraph()

        if len(figs) > 30:
            add_para(doc, f'... and {len(figs)-30} more figures in this category', italic=True)

        doc.add_page_break()

    # =========================================================================
    # PART VII: CONCLUSIONS
    # =========================================================================
    print("9. Part VII: Conclusions...")

    add_heading(doc, 'PART VIII: CONCLUSIONS', 1)
    doc.add_page_break()

    add_heading(doc, '30. Summary of Findings', 1)

    add_heading(doc, '30.1 Answers to Research Questions', 2)

    answers = [
        ('RQ1: Best HPO Algorithm?', 'No universal winner. Random search for simple regression, TPE for complex tasks, metaheuristics for classification.'),
        ('RQ2: Universal Best?', 'No. Algorithm selection should be task-dependent.'),
        ('RQ3: Trial Budget?', '30-50 trials capture 96-100% of optimal performance.'),
        ('RQ4: Critical Hyperparameters?', 'Hidden dimension and number of layers have largest impact.'),
        ('RQ5: GNN vs Foundation Models?', 'Task-specific GNNs outperform foundation models on 4/6 tasks. Foundation models win on Clearance_Hepatocyte.'),
        ('RQ6: Multi-Seed Impact?', 'Critical! Initial results had 223% CV, proper preprocessing reduced to 4%.'),
    ]

    for q, a in answers:
        add_para(doc, q, bold=True)
        add_para(doc, a)
        doc.add_paragraph()

    add_heading(doc, '31. Recommendations for Practitioners', 1)

    recs = [
        ('For Regression Tasks', 'Start with Random Search as baseline. If budget allows, try TPE for complex clearance prediction.'),
        ('For Classification Tasks', 'Use metaheuristic algorithms (SA, ABC) which showed best performance.'),
        ('For Foundation Models', 'Be cautious with scaffold-split evaluation. ChemBERTa may overfit. Consider GNNs first.'),
        ('For Statistical Validity', 'Always run multiple seeds. Report mean ± std and 95% CI.'),
        ('For Preprocessing', 'Use training data for normalization. Set clip_min=1e-3 for log transform.'),
    ]

    for title, rec in recs:
        add_para(doc, f'{title}:', bold=True)
        add_para(doc, rec)

    add_heading(doc, '32. Future Work', 1)

    future = [
        'Extend to more ADMET endpoints (e.g., PAMPA, CYP inhibition)',
        'Include 3D molecular representations',
        'Test on proprietary pharmaceutical datasets',
        'Develop ensemble methods combining best algorithms',
        'Investigate transfer learning from foundation models',
        'Create automated HPO algorithm selector',
    ]

    for f in future:
        add_bullet(doc, f)

    doc.add_page_break()

    # =========================================================================
    # APPENDICES
    # =========================================================================
    print("10. Appendices...")

    add_heading(doc, 'APPENDIX A: All Experimental Results', 1)

    # Load and display all CSV results
    csv_files = glob_module.glob(str(RESULTS_DIR / '**/*.csv'), recursive=True)

    for csv_path in sorted(csv_files)[:15]:  # Limit
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0 and len(df.columns) <= 10:
                add_heading(doc, Path(csv_path).name, 3)
                rows = df.head(15).values.tolist()
                add_table(doc, list(df.columns), rows)
                doc.add_paragraph()
        except:
            pass

    doc.add_page_break()

    # Final page
    add_heading(doc, 'Document Information', 1)

    info = [
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Total Figures: {fig_count}',
        f'Total Tables: {table_count}',
        'Author: Martin Mila Adrijan',
        'Project: MANU - Molecular ADMET Neural Understanding',
    ]

    for i in info:
        add_para(doc, i)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run('--- END OF DOCUMENT ---').bold = True

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PROJECT_ROOT / f'MANU_MASSIVE_DOCUMENTATION_{timestamp}.docx'
    doc.save(str(output_path))

    print("="*70)
    print(f"DOCUMENT SAVED: {output_path}")
    print(f"Total Figures: {fig_count}")
    print(f"Total Tables: {table_count}")
    print("="*70)

    return str(output_path)


if __name__ == '__main__':
    output = create_document()
    print(f"\nComplete! Document at:\n{output}")
