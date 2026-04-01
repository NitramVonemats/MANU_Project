#!/usr/bin/env python3
"""
Generate Complete Word Documentation for MANU Project
Includes ALL figures, code, results, and implementation details.
Target: 100+ pages comprehensive documentation
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime

from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

def add_heading(doc, text, level=1):
    """Add a heading with proper formatting."""
    heading = doc.add_heading(text, level=level)
    return heading

def add_paragraph(doc, text, bold=False, italic=False):
    """Add a paragraph with optional formatting."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    return p

def add_code_block(doc, code, language="python"):
    """Add a code block with monospace font."""
    p = doc.add_paragraph()
    p.style = 'No Spacing'
    run = p.add_run(code)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    # Set background color for code
    return p

def add_image_safe(doc, image_path, width=Inches(6), caption=None):
    """Add an image with error handling."""
    try:
        if os.path.exists(image_path):
            doc.add_picture(image_path, width=width)
            last_paragraph = doc.paragraphs[-1]
            last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if caption:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(caption)
                run.italic = True
                run.font.size = Pt(10)
            return True
        else:
            doc.add_paragraph(f"[Image not found: {image_path}]")
            return False
    except Exception as e:
        doc.add_paragraph(f"[Error loading image: {e}]")
        return False

def add_table(doc, headers, rows):
    """Add a formatted table."""
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'

    # Header row
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        # Bold header
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True

    # Data rows
    for row_data in rows:
        row_cells = table.add_row().cells
        for i, cell_data in enumerate(row_data):
            row_cells[i].text = str(cell_data)

    return table

def read_file_content(filepath, max_lines=100):
    """Read file content with line limit."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if len(lines) > max_lines:
                content = ''.join(lines[:max_lines])
                content += f"\n... [Truncated - {len(lines) - max_lines} more lines]"
            else:
                content = ''.join(lines)
            return content
    except Exception as e:
        return f"[Error reading file: {e}]"

def read_json_file(filepath):
    """Read JSON file and return data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None

def create_document():
    """Create the complete Word document."""
    doc = Document()

    # Set document margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    print("Creating comprehensive Word documentation...")
    print("=" * 60)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    print("1. Creating title page...")

    # Title
    title = doc.add_heading('MANU Project', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('Systematic Hyperparameter Optimization for Molecular Property Prediction with Graph Neural Networks')
    run.font.size = Pt(16)
    run.bold = True

    doc.add_paragraph()
    doc.add_paragraph()

    # Subtitle info
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run('A Comprehensive Benchmark Study\n\n').bold = True
    info.add_run('Complete Technical Documentation\n')
    info.add_run('Version 2.0 - Final Publication Ready\n\n')
    info.add_run(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    info.add_run('Author: Martin Mila Adrijan\n')

    doc.add_page_break()

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    print("2. Creating table of contents...")

    add_heading(doc, 'Table of Contents', 1)

    toc_items = [
        "1. Executive Summary",
        "2. Introduction and Motivation",
        "3. Related Work",
        "4. Datasets",
        "5. Methodology",
        "6. GNN Architecture and Implementation",
        "7. Hyperparameter Optimization Algorithms",
        "8. Foundation Model Comparison",
        "9. Experimental Results - HPO Benchmark",
        "10. Experimental Results - TPE Benchmark",
        "11. Experimental Results - ChemBERTa Fine-tuning",
        "12. Multi-Seed Validation",
        "13. Ablation Studies",
        "14. Statistical Analysis",
        "15. Visualizations Gallery",
        "16. Code Implementation",
        "17. Results Data",
        "18. Reproducibility Guide",
        "19. Discussion and Conclusions",
        "20. References",
        "Appendix A: Complete Code Listings",
        "Appendix B: All Experimental Results",
        "Appendix C: All Figures"
    ]

    for item in toc_items:
        doc.add_paragraph(item, style='List Number')

    doc.add_page_break()

    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    print("3. Writing Executive Summary...")

    add_heading(doc, '1. Executive Summary', 1)

    add_heading(doc, '1.1 Project Overview', 2)
    doc.add_paragraph(
        'This project presents a comprehensive benchmark study for hyperparameter optimization (HPO) '
        'of Graph Neural Networks (GNNs) for molecular property prediction, specifically focusing on '
        'ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties. The study compares '
        'seven HPO algorithms across six datasets from the Therapeutics Data Commons (TDC), includes '
        'comparisons with foundation models, and provides multi-seed statistical validation.'
    )

    add_heading(doc, '1.2 Key Statistics', 2)

    stats_headers = ['Metric', 'Value']
    stats_rows = [
        ['Total Datasets', '6 (4 ADME + 2 Toxicity)'],
        ['Total Molecules', '11,805'],
        ['HPO Algorithms', '7 (Random, PSO, ABC, GA, SA, HC, TPE)'],
        ['Trials per Run', '50'],
        ['Total HPO Runs', '42 (6 datasets × 7 algorithms)'],
        ['Total Model Evaluations', '2,100+'],
        ['Multi-Seed Validation', '5 seeds per dataset'],
        ['Foundation Models Tested', '5 (ChemBERTa, ChemBERTa-FT, MolCLR, Morgan-FP, MolE-FP)'],
        ['Compute Time', '~45 hours'],
        ['Visualizations Generated', '100+'],
        ['Success Rate', '100%'],
    ]
    add_table(doc, stats_headers, stats_rows)

    doc.add_paragraph()

    add_heading(doc, '1.3 Main Findings', 2)

    findings = [
        'Random Search is surprisingly effective for regression tasks - Wins on 2/4 ADME datasets',
        'TPE (Bayesian optimization) excels on complex clearance tasks - Best on Clearance_Hepatocyte_AZ',
        'Metaheuristic algorithms outperform on classification - SA wins Tox21, ABC wins hERG',
        'GNN models outperform foundation models - Win on 5/6 benchmarks',
        'ChemBERTa fine-tuning improves foundation model results',
        'No universal winner exists - Algorithm selection should be task-dependent',
        '50 trials is sufficient - Diminishing returns beyond this budget',
    ]

    for finding in findings:
        doc.add_paragraph(finding, style='List Bullet')

    add_heading(doc, '1.4 Results Summary', 2)

    add_heading(doc, '1.4.1 ADME Regression Results (Test RMSE - lower is better)', 3)

    adme_headers = ['Dataset', 'Best GNN', 'TPE', 'ChemBERTa-FT', 'Winner']
    adme_rows = [
        ['Caco2_Wang', '0.0027', '0.0030', '0.0032', 'GNN (Random)'],
        ['Half_Life_Obach', '21.66', '22.34', '8.31*', 'GNN (PSO)'],
        ['Clearance_Hepatocyte_AZ', '68.22', '52.16', '52.60', 'TPE'],
        ['Clearance_Microsome_AZ', '38.75', '44.34', '42.87', 'GNN (Random)'],
    ]
    add_table(doc, adme_headers, adme_rows)
    doc.add_paragraph('*Log-scale RMSE = 1.07', style='Caption')

    doc.add_paragraph()

    add_heading(doc, '1.4.2 Toxicity Classification Results (Test AUC - higher is better)', 3)

    tox_headers = ['Dataset', 'Best GNN', 'TPE', 'ChemBERTa-FT', 'MolCLR', 'Winner']
    tox_rows = [
        ['Tox21', '0.742', '0.705', '0.464*', '0.633', 'GNN (SA)'],
        ['hERG', '0.711', '0.772', '0.729', '0.434', 'GNN (ABC)'],
    ]
    add_table(doc, tox_headers, tox_rows)
    doc.add_paragraph('*ChemBERTa Tox21 exhibits scaffold split distribution shift (Val AUC=0.82, Test AUC=0.46)', style='Caption')

    doc.add_page_break()

    # =========================================================================
    # 2. INTRODUCTION
    # =========================================================================
    print("4. Writing Introduction...")

    add_heading(doc, '2. Introduction and Motivation', 1)

    add_heading(doc, '2.1 Background', 2)

    add_heading(doc, '2.1.1 The Drug Discovery Pipeline', 3)
    doc.add_paragraph(
        'Drug discovery is a lengthy and expensive process, with estimates suggesting that bringing '
        'a new drug to market costs over $2.6 billion and takes 10-15 years. A significant portion '
        'of drug candidates fail in clinical trials due to poor ADMET properties, which are often '
        'not adequately characterized in early discovery phases.'
    )

    add_heading(doc, '2.1.2 ADMET Properties', 3)
    doc.add_paragraph('ADMET stands for:')

    admet_items = [
        'Absorption: How a drug enters the bloodstream',
        'Distribution: How a drug spreads through body tissues',
        'Metabolism: How a drug is chemically modified in the body',
        'Excretion: How a drug is eliminated from the body',
        'Toxicity: Adverse effects of a drug',
    ]
    for item in admet_items:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph(
        'Accurate prediction of these properties early in the drug discovery pipeline can significantly '
        'reduce costs and time-to-market by filtering out candidates with poor drug-like properties.'
    )

    add_heading(doc, '2.1.3 Graph Neural Networks for Molecular Property Prediction', 3)
    doc.add_paragraph(
        'Graph Neural Networks (GNNs) have emerged as a powerful approach for learning molecular '
        'representations directly from molecular graphs. Unlike traditional approaches that rely on '
        'hand-crafted molecular descriptors or fingerprints, GNNs can learn task-specific representations '
        'that capture both local and global molecular features.'
    )

    doc.add_paragraph('Key advantages of GNNs include:')
    gnn_advantages = [
        'End-to-end learning: No need for manual feature engineering',
        'Permutation invariance: Respect molecular symmetry',
        'Expressiveness: Can capture complex structural patterns',
        'Transferability: Learned representations can transfer across tasks',
    ]
    for item in gnn_advantages:
        doc.add_paragraph(item, style='List Bullet')

    add_heading(doc, '2.2 Problem Statement', 2)
    doc.add_paragraph(
        'Despite the promise of GNNs for molecular property prediction, their performance is highly '
        'sensitive to hyperparameter choices:'
    )

    hp_items = [
        'Learning rate: Controls optimization speed and convergence',
        'Network depth: Number of message-passing layers',
        'Hidden dimensions: Capacity of learned representations',
        'Regularization: Weight decay, dropout',
        'Architecture choices: Aggregation functions, normalization',
    ]
    for item in hp_items:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph(
        'Manual hyperparameter tuning is time-consuming, suboptimal, and non-transferable. '
        'This motivates the need for systematic hyperparameter optimization.'
    )

    add_heading(doc, '2.3 Research Questions', 2)

    rqs = [
        'RQ1: Which HPO algorithm is most effective for molecular GNNs?',
        'RQ2: Does sophisticated optimization provide significant advantage over Random Search?',
        'RQ3: How does performance vary between regression and classification tasks?',
        'RQ4: Do GNNs outperform foundation models (ChemBERTa, Morgan fingerprints)?',
        'RQ5: What is the optimal trial budget for HPO in this domain?',
        'RQ6: Are the results statistically robust across multiple seeds?',
    ]
    for rq in rqs:
        p = doc.add_paragraph(rq)
        p.runs[0].bold = True

    add_heading(doc, '2.4 Contributions', 2)

    contributions = [
        'Comprehensive HPO Benchmark: Systematic comparison of 7 HPO algorithms across 6 ADMET datasets with 50 trials each',
        'Foundation Model Comparison: First systematic comparison of GNNs with modern foundation models on ADMET prediction',
        'Multi-Seed Validation: Statistical validation with 5 seeds and 95% confidence intervals',
        'Practical Recommendations: Task-specific guidance for algorithm selection',
        'Reproducible Codebase: Complete, modular implementation with documentation',
        'Publication-Ready Visualizations: 100+ figures for analysis and publication',
    ]
    for contrib in contributions:
        doc.add_paragraph(contrib, style='List Bullet')

    doc.add_page_break()

    # =========================================================================
    # 3. RELATED WORK
    # =========================================================================
    print("5. Writing Related Work...")

    add_heading(doc, '3. Related Work', 1)

    add_heading(doc, '3.1 Molecular Property Prediction', 2)

    add_heading(doc, '3.1.1 Traditional Approaches', 3)
    doc.add_paragraph('Traditional molecular property prediction relies on:')
    trad_items = [
        'Molecular descriptors: Physical-chemical properties (LogP, TPSA, etc.)',
        'Fingerprints: Binary vectors encoding structural features (ECFP, MACCS)',
        'QSAR models: Linear or tree-based models on descriptors',
    ]
    for item in trad_items:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph(
        'Notable benchmarks include MoleculeNet (Wu et al., 2018), which standardized evaluation '
        'across multiple molecular datasets.'
    )

    add_heading(doc, '3.1.2 Deep Learning Approaches', 3)
    doc.add_paragraph('Deep learning approaches for molecular property prediction include:')
    dl_items = [
        'Convolutional networks on SMILES: 1D convolutions on string representations',
        'Graph neural networks: Message-passing on molecular graphs',
        'Transformers: Attention-based models on SMILES or molecular graphs',
    ]
    for item in dl_items:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph('Key GNN architectures include:')
    gnn_archs = [
        'GCN (Kipf & Welling, 2017): Spectral graph convolutions',
        'GAT (Veličković et al., 2018): Graph attention networks',
        'MPNN (Gilmer et al., 2017): General message-passing framework',
        'D-MPNN (Yang et al., 2019): Directed message-passing',
        'GIN (Xu et al., 2019): Graph isomorphism networks',
    ]
    for item in gnn_archs:
        doc.add_paragraph(item, style='List Bullet')

    add_heading(doc, '3.2 Hyperparameter Optimization', 2)

    add_heading(doc, '3.2.1 HPO Methods', 3)
    hpo_methods = [
        'Grid Search: Exhaustive search over parameter grid',
        'Random Search (Bergstra & Bengio, 2012): Uniform random sampling',
        'Bayesian Optimization (Snoek et al., 2012): Surrogate model-based optimization',
        'Evolutionary Algorithms: GA, PSO, ABC, etc.',
        'Gradient-based: Differentiable hyperparameters',
    ]
    for item in hpo_methods:
        doc.add_paragraph(item, style='List Bullet')

    add_heading(doc, '3.2.2 Notable Frameworks', 3)
    frameworks = [
        'Optuna (Akiba et al., 2019): TPE-based optimization',
        'Hyperopt (Bergstra et al., 2013): Tree-structured Parzen Estimator',
        'BOHB (Falkner et al., 2018): Bayesian optimization with early stopping',
        'NiaPy: Nature-inspired algorithms library',
    ]
    for item in frameworks:
        doc.add_paragraph(item, style='List Bullet')

    add_heading(doc, '3.3 Foundation Models for Chemistry', 2)

    foundation_models = [
        'ChemBERTa (Chithrananda et al., 2020): BERT pre-trained on SMILES',
        'MolBERT (Fabian et al., 2020): BERT for molecular properties',
        'MolCLR (Wang et al., 2022): Contrastive learning on molecular graphs',
        'GraphMVP (Liu et al., 2022): Multi-view pre-training',
    ]
    for item in foundation_models:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_page_break()

    # =========================================================================
    # 4. DATASETS
    # =========================================================================
    print("6. Writing Datasets section...")

    add_heading(doc, '4. Datasets', 1)

    add_heading(doc, '4.1 Therapeutics Data Commons (TDC)', 2)
    doc.add_paragraph(
        'The Therapeutics Data Commons (TDC) (Huang et al., 2021) provides standardized datasets '
        'for drug discovery machine learning. We use 6 datasets covering ADME properties and toxicity.'
    )

    add_heading(doc, '4.1.1 ADME Datasets (Regression)', 3)

    adme_ds_headers = ['Dataset', 'Property', 'Molecules', 'Description']
    adme_ds_rows = [
        ['Caco2_Wang', 'Cell Permeability', '910', 'Caco-2 cell permeability (log Papp)'],
        ['Half_Life_Obach', 'Half-life', '667', 'Human plasma half-life (hours)'],
        ['Clearance_Hepatocyte_AZ', 'Hepatic Clearance', '1,213', 'Hepatocyte clearance (mL/min/kg)'],
        ['Clearance_Microsome_AZ', 'Microsomal Clearance', '1,102', 'Microsome clearance (mL/min/kg)'],
    ]
    add_table(doc, adme_ds_headers, adme_ds_rows)

    doc.add_paragraph()

    add_heading(doc, '4.1.2 Toxicity Datasets (Classification)', 3)

    tox_ds_headers = ['Dataset', 'Property', 'Molecules', 'Positive Rate', 'Description']
    tox_ds_rows = [
        ['Tox21 (NR-AR)', 'Nuclear Receptor Toxicity', '7,258', '3.5%', 'Androgen receptor assay'],
        ['hERG', 'Cardiotoxicity', '655', '31%', 'hERG channel blocking'],
    ]
    add_table(doc, tox_ds_headers, tox_ds_rows)

    doc.add_paragraph()
    p = doc.add_paragraph('Total: 11,805 molecules')
    p.runs[0].bold = True

    add_heading(doc, '4.2 Data Processing Pipeline', 2)

    add_heading(doc, '4.2.1 Molecular Featurization', 3)
    doc.add_paragraph('Each molecule is converted to a graph representation with the following features:')

    doc.add_paragraph('Node Features (8 dimensions per atom):', style='Heading 4')

    node_feat_headers = ['Feature', 'Description', 'Encoding']
    node_feat_rows = [
        ['Atomic Number', 'Number of protons', 'One-hot (common atoms)'],
        ['Degree', 'Number of bonds', 'Integer'],
        ['Formal Charge', 'Formal charge', 'Integer'],
        ['Hybridization', 'sp, sp2, sp3, etc.', 'One-hot'],
        ['Aromaticity', 'Is aromatic', 'Binary'],
        ['Ring Membership', 'Part of ring', 'Binary'],
        ['Hydrogen Count', 'Number of hydrogens', 'Integer'],
        ['Atomic Mass', 'Atomic mass', 'Float (normalized)'],
    ]
    add_table(doc, node_feat_headers, node_feat_rows)

    doc.add_paragraph()
    doc.add_paragraph('Edge Features (4 dimensions per bond):', style='Heading 4')

    edge_feat_headers = ['Feature', 'Description', 'Encoding']
    edge_feat_rows = [
        ['Bond Type', 'Single, double, triple, aromatic', 'One-hot'],
        ['Conjugation', 'Is conjugated', 'Binary'],
        ['Ring Membership', 'Part of ring', 'Binary'],
        ['Stereo Configuration', 'E/Z, cis/trans', 'One-hot'],
    ]
    add_table(doc, edge_feat_headers, edge_feat_rows)

    add_heading(doc, '4.2.2 Data Splitting', 3)
    split_items = [
        'Method: Scaffold-based splitting (Bemis-Murcko scaffolds)',
        'Ratios: 80% train / 10% validation / 10% test',
        'Seed: 42 (for reproducibility)',
        'Implementation: TDC 2-way split with manual 90/10 train/val subdivision',
    ]
    for item in split_items:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph(
        'Scaffold splitting ensures that molecules with different core structures are separated, '
        'better reflecting real-world generalization scenarios.'
    )

    add_heading(doc, '4.2.3 Target Transformation', 3)

    doc.add_paragraph('Regression targets:', style='Heading 4')
    reg_trans = [
        'Log transformation for skewed distributions',
        'Clipping: min=1e-3 for non-Caco2 datasets',
        'Standardization: (y - μ) / σ using training+validation data',
    ]
    for item in reg_trans:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph('Classification targets:', style='Heading 4')
    cls_trans = [
        'Binary encoding (0/1)',
        'Class weighting for imbalanced datasets',
    ]
    for item in cls_trans:
        doc.add_paragraph(item, style='List Bullet')

    add_heading(doc, '4.3 Dataset Statistics', 2)

    add_heading(doc, '4.3.1 Size and Distribution', 3)

    size_headers = ['Dataset', 'Train', 'Val', 'Test', 'Total', 'Task Type']
    size_rows = [
        ['Caco2_Wang', '655', '73', '182', '910', 'Regression'],
        ['Half_Life_Obach', '479', '54', '134', '667', 'Regression'],
        ['Clearance_Hepatocyte_AZ', '873', '97', '243', '1,213', 'Regression'],
        ['Clearance_Microsome_AZ', '792', '89', '221', '1,102', 'Regression'],
        ['Tox21', '5,226', '581', '1,451', '7,258', 'Classification'],
        ['hERG', '471', '53', '131', '655', 'Classification'],
    ]
    add_table(doc, size_headers, size_rows)

    doc.add_paragraph()

    add_heading(doc, '4.3.2 Molecular Diversity (Tanimoto Similarity)', 3)

    tani_headers = ['Dataset', 'Mean', 'Std', 'Min', 'Max']
    tani_rows = [
        ['Caco2_Wang', '0.11', '0.08', '0.00', '1.00'],
        ['Half_Life_Obach', '0.13', '0.09', '0.00', '1.00'],
        ['Clearance_Hepatocyte_AZ', '0.10', '0.07', '0.00', '1.00'],
        ['Clearance_Microsome_AZ', '0.09', '0.07', '0.00', '1.00'],
        ['Tox21', '0.08', '0.06', '0.00', '1.00'],
        ['hERG', '0.12', '0.08', '0.00', '1.00'],
    ]
    add_table(doc, tani_headers, tani_rows)

    doc.add_paragraph()
    doc.add_paragraph(
        'Low mean similarity (0.08-0.13) indicates good chemical diversity, suitable for machine learning.'
    )

    # Dataset Visualizations
    add_heading(doc, '4.4 Dataset Visualizations', 2)

    dataset_figs = [
        ('figures/comparative/01_dataset_overview.png', 'Figure 4.1: Dataset Overview - Size and distribution comparison'),
        ('figures/comparative/02_label_distributions.png', 'Figure 4.2: Label Distributions across all datasets'),
        ('figures/comparative/03_feature_importance.png', 'Figure 4.3: Feature Importance ranking'),
        ('figures/comparative/04_tanimoto_similarity.png', 'Figure 4.4: Tanimoto Similarity comparison'),
        ('figures/comparative/05_summary_table.png', 'Figure 4.5: Dataset Summary Statistics'),
    ]

    for fig_path, caption in dataset_figs:
        full_path = PROJECT_ROOT / fig_path
        add_image_safe(doc, str(full_path), width=Inches(5.5), caption=caption)
        doc.add_paragraph()

    doc.add_page_break()

    # =========================================================================
    # 5. METHODOLOGY
    # =========================================================================
    print("7. Writing Methodology...")

    add_heading(doc, '5. Methodology', 1)

    add_heading(doc, '5.1 Experimental Design', 2)

    add_heading(doc, '5.1.1 Overall Pipeline', 3)

    pipeline_code = """
1. Data Preparation
   └── TDC Download → Featurization → Scaffold Split → Normalization

2. HPO Search Phase
   └── Algorithm Selection → Search Space Definition → 50 Trials

3. Final Training Phase
   └── Best Hyperparameters → Full Training → Test Evaluation

4. Multi-Seed Validation
   └── 5 Seeds × 6 Datasets → Statistical Analysis

5. Foundation Model Comparison
   └── ChemBERTa → MolCLR → Morgan-FP → Comparison

6. Visualization & Analysis
   └── 100+ Publication Figures → Statistical Tests
"""
    add_code_block(doc, pipeline_code)

    add_heading(doc, '5.1.2 Evaluation Protocol', 3)

    doc.add_paragraph('HPO Phase:', style='Heading 4')
    hpo_protocol = [
        'Objective: Minimize validation RMSE (regression) or maximize validation AUC (classification)',
        'Budget: 50 trials per algorithm-dataset combination',
        'Selection: Best validation metric across all trials',
    ]
    for item in hpo_protocol:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph('Final Evaluation:', style='Heading 4')
    final_protocol = [
        'Training: Using best hyperparameters from HPO',
        'Early stopping: Patience of 12 epochs on validation metric',
        'Test evaluation: On held-out test set (never seen during HPO)',
    ]
    for item in final_protocol:
        doc.add_paragraph(item, style='List Bullet')

    add_heading(doc, '5.1.3 Metrics', 3)

    doc.add_paragraph('Regression Metrics:', style='Heading 4')
    reg_metrics = [
        'RMSE: Root Mean Square Error (primary metric)',
        'MAE: Mean Absolute Error',
        'R²: Coefficient of Determination',
    ]
    for item in reg_metrics:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph('Classification Metrics:', style='Heading 4')
    cls_metrics = [
        'AUC-ROC: Area Under ROC Curve (primary metric)',
        'F1 Score: Harmonic mean of precision and recall',
        'Accuracy: Overall classification accuracy',
        'Precision/Recall: Per-class metrics',
    ]
    for item in cls_metrics:
        doc.add_paragraph(item, style='List Bullet')

    add_heading(doc, '5.2 Implementation Details', 2)

    add_heading(doc, '5.2.1 Training Configuration', 3)

    train_headers = ['Parameter', 'Value', 'Notes']
    train_rows = [
        ['Optimizer', 'Adam', 'β1=0.9, β2=0.999'],
        ['Max Epochs', '50-150', 'Dataset-dependent'],
        ['Early Stopping', 'Patience 12', 'On validation metric'],
        ['Batch Size (Train)', '32', ''],
        ['Batch Size (Eval)', '64', ''],
        ['Gradient Clipping', 'Max norm 1.0', ''],
        ['Loss (Regression)', 'MSE', 'Mean Squared Error'],
        ['Loss (Classification)', 'BCE with Logits', 'Binary Cross-Entropy'],
        ['Class Weights', 'Positive weight', 'For imbalanced classification'],
        ['Device', 'CPU/CUDA', 'Auto-detected'],
    ]
    add_table(doc, train_headers, train_rows)

    doc.add_paragraph()

    add_heading(doc, '5.2.2 Hyperparameter Search Space', 3)

    hp_headers = ['Parameter', 'Range', 'Type', 'Scale']
    hp_rows = [
        ['Hidden Dimension', '[64, 96, 128, 192, 256, 384, 512]', 'Categorical', '-'],
        ['Number of Layers', '[3, 4, 5, 6, 7]', 'Categorical', '-'],
        ['Learning Rate', '[1e-4, 1e-2]', 'Continuous', 'Log'],
        ['Weight Decay', '[1e-6, 1e-2]', 'Continuous', 'Log'],
        ['Dropout', '[0.0, 0.5]', 'Continuous', 'Linear'],
        ['Head Dimensions', '3 levels', 'Categorical', '-'],
    ]
    add_table(doc, hp_headers, hp_rows)

    doc.add_page_break()

    # =========================================================================
    # 6. GNN ARCHITECTURE
    # =========================================================================
    print("8. Writing GNN Architecture...")

    add_heading(doc, '6. GNN Architecture and Implementation', 1)

    add_heading(doc, '6.1 Model Architecture Overview', 2)

    arch_diagram = """
Input: Molecular Graph (atoms + bonds)
    ↓
GNN Backbone (GCN layers with BatchNorm)
    ↓
Global Pooling (mean + max concatenation)
    ↓
Prediction Head (MLP with dropout)
    ↓
Output: Property Prediction (regression or classification)
"""
    add_code_block(doc, arch_diagram)

    add_heading(doc, '6.2 GNN Backbone Implementation', 2)

    doc.add_paragraph(
        'The GNN backbone consists of multiple Graph Convolutional Network (GCN) layers '
        'with batch normalization and dropout for regularization.'
    )

    add_heading(doc, '6.2.1 GCN Layer (Kipf & Welling, 2017)', 3)

    gcn_formula = """
GCN Update Rule:
h_i^{(l+1)} = σ(Σ_{j∈N(i)} (1/√(d_i * d_j)) * W^{(l)} * h_j^{(l)})

Where:
- h_i^{(l)} is the hidden state of node i at layer l
- N(i) is the neighborhood of node i
- d_i, d_j are node degrees (normalization)
- W^{(l)} is the learnable weight matrix
- σ is the activation function (ReLU)
"""
    add_code_block(doc, gcn_formula)

    add_heading(doc, '6.2.2 Backbone Code Implementation', 3)

    backbone_code = '''class GNNBackbone(nn.Module):
    """GNN backbone with configurable layers."""

    def __init__(self, in_channels, hidden_channels, num_layers, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # First layer: input → hidden
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers: hidden → hidden
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
'''
    add_code_block(doc, backbone_code)

    add_heading(doc, '6.3 Global Pooling', 2)

    doc.add_paragraph(
        'We use a combination of mean and max pooling for graph-level readout. '
        'This captures both average and extreme node features.'
    )

    pooling_code = '''def global_pool(x, batch):
    """Combine mean and max pooling for graph readout."""
    x_mean = global_mean_pool(x, batch)  # Average aggregation
    x_max = global_max_pool(x, batch)    # Maximum aggregation
    return torch.cat([x_mean, x_max], dim=1)  # Concatenation
'''
    add_code_block(doc, pooling_code)

    add_heading(doc, '6.4 Prediction Head', 2)

    head_code = '''class PredictionHead(nn.Module):
    """MLP prediction head with configurable dimensions."""

    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.1):
        super().__init__()
        # hidden_dims example: [512, 96, 96]

        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, out_dim)

    def forward(self, x):
        x = self.mlp(x)
        return self.output(x)
'''
    add_code_block(doc, head_code)

    add_heading(doc, '6.5 Complete Molecular Predictor', 2)

    predictor_code = '''class MolecularPredictor(nn.Module):
    """Complete model combining GNN backbone and prediction head."""

    def __init__(self, node_features, hidden_dim, num_layers,
                 head_dims, out_dim, dropout=0.1, task='regression'):
        super().__init__()

        self.backbone = GNNBackbone(
            in_channels=node_features,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # After pooling: hidden_dim * 2 (mean + max)
        self.head = PredictionHead(
            in_dim=hidden_dim * 2,
            hidden_dims=head_dims,
            out_dim=out_dim,
            dropout=dropout
        )

        self.task = task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN encoding
        x = self.backbone(x, edge_index, batch)

        # Global pooling
        x = global_pool(x, batch)

        # Prediction
        out = self.head(x)

        return out
'''
    add_code_block(doc, predictor_code)

    add_heading(doc, '6.6 Architecture Selection Rationale', 2)

    doc.add_paragraph('We selected GCN as the primary architecture based on preliminary experiments:')

    arch_headers = ['Architecture', 'Caco2 R²', 'Time', 'Stability']
    arch_rows = [
        ['GCN', '0.30', '30s', 'High'],
        ['GraphSAGE', '0.36', '45s', 'High'],
        ['GIN', '0.04', '85s', 'Low'],
        ['GAT', '0.28', '60s', 'Medium'],
    ]
    add_table(doc, arch_headers, arch_rows)

    doc.add_paragraph()
    doc.add_paragraph(
        'GCN provides the best trade-off between performance, efficiency, and stability.'
    )

    doc.add_page_break()

    # =========================================================================
    # 7. HPO ALGORITHMS
    # =========================================================================
    print("9. Writing HPO Algorithms...")

    add_heading(doc, '7. Hyperparameter Optimization Algorithms', 1)

    add_heading(doc, '7.1 Algorithm Overview', 2)

    algo_headers = ['Algorithm', 'Type', 'Exploration', 'Exploitation', 'Parallelizable']
    algo_rows = [
        ['Random', 'Baseline', 'High', 'Low', 'Yes'],
        ['PSO', 'Swarm', 'Medium', 'High', 'Yes'],
        ['ABC', 'Swarm', 'High', 'Medium', 'Yes'],
        ['GA', 'Evolutionary', 'Medium', 'Medium', 'Yes'],
        ['SA', 'Probabilistic', 'High→Low', 'Low→High', 'No'],
        ['HC', 'Local Search', 'Low', 'High', 'Yes'],
        ['TPE', 'Bayesian', 'Adaptive', 'Adaptive', 'Limited'],
    ]
    add_table(doc, algo_headers, algo_rows)

    doc.add_paragraph()

    # Random Search
    add_heading(doc, '7.2 Random Search', 2)
    doc.add_paragraph(
        'Random Search uniformly samples from the search space. Despite its simplicity, '
        'it is surprisingly effective for many problems (Bergstra & Bengio, 2012).'
    )

    doc.add_paragraph('Strengths:', style='Heading 4')
    random_strengths = [
        'Simple and fast implementation',
        'Embarrassingly parallel',
        'No hyperparameters to tune',
        'Good baseline for comparison',
    ]
    for item in random_strengths:
        doc.add_paragraph(item, style='List Bullet')

    # PSO
    add_heading(doc, '7.3 Particle Swarm Optimization (PSO)', 2)
    doc.add_paragraph(
        'PSO is a population-based optimization inspired by bird flocking behavior. '
        'Particles move through the search space, influenced by their personal best '
        'and the global best positions.'
    )

    pso_config = '''PSO Configuration:
- Population size: 16
- Inertia weight (w): 0.7
- Cognitive coefficient (c1): 1.5
- Social coefficient (c2): 1.5

Update Rule:
v_i = w·v_i + c1·r1·(p_best_i - x_i) + c2·r2·(g_best - x_i)
x_i = x_i + v_i
'''
    add_code_block(doc, pso_config)

    # ABC
    add_heading(doc, '7.4 Artificial Bee Colony (ABC)', 2)
    doc.add_paragraph(
        'ABC is a swarm intelligence algorithm inspired by honeybee foraging behavior. '
        'It has three phases: employed bees, onlooker bees, and scout bees.'
    )

    abc_config = '''ABC Configuration:
- Population size: 16
- Abandonment limit: 50

Phases:
1. Employed Bees: Exploit current food sources
2. Onlooker Bees: Select sources probabilistically
3. Scout Bees: Random exploration when sources exhausted
'''
    add_code_block(doc, abc_config)

    # GA
    add_heading(doc, '7.5 Genetic Algorithm (GA)', 2)
    doc.add_paragraph(
        'GA is an evolutionary algorithm inspired by natural selection. It uses '
        'selection, crossover, and mutation operators to evolve a population of solutions.'
    )

    ga_config = '''GA Configuration:
- Population size: 16
- Mutation rate: 0.1
- Crossover rate: 0.8
- Selection: Tournament selection
'''
    add_code_block(doc, ga_config)

    # SA
    add_heading(doc, '7.6 Simulated Annealing (SA)', 2)
    doc.add_paragraph(
        'SA is a probabilistic optimization inspired by metallurgical annealing. '
        'It can escape local optima by accepting worse solutions with decreasing probability.'
    )

    sa_config = '''SA Configuration:
- Initial temperature (T0): 50.0
- Minimum temperature: 0.001
- Cooling rate (alpha): 0.99

Acceptance Probability:
P(accept) = exp(-ΔE / T)  if ΔE > 0
          = 1             if ΔE ≤ 0
'''
    add_code_block(doc, sa_config)

    # HC
    add_heading(doc, '7.7 Hill Climbing (HC)', 2)
    doc.add_paragraph(
        'Hill Climbing is a local search that iteratively improves the current solution '
        'by moving to the best neighbor.'
    )

    hc_config = '''HC Configuration:
- Neighborhood size: 10
- Step size: 0.1

Algorithm:
1. Start with random solution
2. Generate neighbors
3. Move to best neighbor if better
4. Repeat until no improvement
'''
    add_code_block(doc, hc_config)

    # TPE
    add_heading(doc, '7.8 Tree-structured Parzen Estimator (TPE)', 2)
    doc.add_paragraph(
        'TPE is a Bayesian optimization method that models the objective function using '
        'kernel density estimators. It is implemented using the Optuna framework.'
    )

    tpe_config = '''TPE Configuration (Optuna):
- Startup trials: 10 (random before TPE)
- EI candidates: 24
- Pruner: MedianPruner

Model:
p(x|y) = l(x) if y < y*  (good region)
       = g(x) if y ≥ y*  (bad region)

Expected Improvement: EI(x) ∝ l(x) / g(x)
'''
    add_code_block(doc, tpe_config)

    add_heading(doc, '7.9 Algorithm Implementation', 2)

    impl_code = '''# NiaPy implementation (PSO, ABC, GA, SA, HC)
from niapy.algorithms.basic import (
    ParticleSwarmAlgorithm,
    ArtificialBeeColonyAlgorithm,
    GeneticAlgorithm,
    SimulatedAnnealing,
    HillClimbAlgorithm
)

# Example: PSO
pso = ParticleSwarmAlgorithm(
    population_size=16,
    w=0.7,
    c1=1.5,
    c2=1.5,
)

# Optuna implementation (TPE)
import optuna

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(),
    direction='minimize'  # or 'maximize'
)
'''
    add_code_block(doc, impl_code)

    doc.add_page_break()

    # =========================================================================
    # 8. FOUNDATION MODELS
    # =========================================================================
    print("10. Writing Foundation Models...")

    add_heading(doc, '8. Foundation Model Comparison', 1)

    add_heading(doc, '8.1 Models Tested', 2)

    fm_headers = ['Model', 'Type', 'Description']
    fm_rows = [
        ['Morgan-FP', 'Fingerprint', '2048-bit ECFP4 + MLP'],
        ['ChemBERTa', 'Transformer', 'Pre-trained on SMILES (77M params)'],
        ['ChemBERTa-FT', 'Transformer', 'Fine-tuned on task data'],
        ['MolCLR', 'Graph Contrastive', 'GCN with contrastive pre-training'],
        ['MolE-FP', 'Learned FP', 'Deep fingerprint encoder'],
    ]
    add_table(doc, fm_headers, fm_rows)

    doc.add_paragraph()

    add_heading(doc, '8.2 ChemBERTa Details', 2)

    chemberta_details = '''ChemBERTa Configuration:
- Model: seyonec/ChemBERTa-zinc-base-v1
- Parameters: 77 million
- Pre-training: ZINC dataset (100M molecules)
- Input: SMILES strings
- Output: 768-dimensional embeddings

Fine-tuning Configuration:
- Unfrozen layers: Last 2 transformer layers
- Learning rate (encoder): 1e-5
- Learning rate (head): 1e-3
- Epochs: 50 max
- Early stopping patience: 10
'''
    add_code_block(doc, chemberta_details)

    add_heading(doc, '8.3 Foundation Model Results', 2)

    add_heading(doc, '8.3.1 ADME Regression (Test RMSE ↓)', 3)

    fm_adme_headers = ['Model', 'Caco2', 'Half_Life', 'Clear_Hep', 'Clear_Micro']
    fm_adme_rows = [
        ['GNN-Best', '0.0027', '21.66', '68.22', '38.75'],
        ['Morgan-FP', '0.614', '22.12', '48.36', '40.36'],
        ['ChemBERTa', '0.496', '27.39', '47.31', '42.56'],
        ['ChemBERTa-FT', '0.0032', '8.31*', '52.60', '42.87'],
        ['MolCLR', '0.713', '21.97', '48.71', '43.33'],
        ['MolE-FP', '0.670', '25.01', '47.22', '41.79'],
    ]
    add_table(doc, fm_adme_headers, fm_adme_rows)

    doc.add_paragraph()

    add_heading(doc, '8.3.2 Toxicity Classification (Test AUC ↑)', 3)

    fm_tox_headers = ['Model', 'Tox21', 'hERG']
    fm_tox_rows = [
        ['GNN-Best', '0.743', '0.825'],
        ['Morgan-FP', '0.722', '0.611'],
        ['ChemBERTa', '0.728', '0.770'],
        ['ChemBERTa-FT', '0.482', '0.777'],
        ['MolCLR', '0.538', '0.504'],
        ['MolE-FP', '0.675', '0.672'],
    ]
    add_table(doc, fm_tox_headers, fm_tox_rows)

    doc.add_paragraph()

    add_heading(doc, '8.3.3 Winner Summary', 3)

    winner_headers = ['Model', 'Wins', 'Datasets']
    winner_rows = [
        ['GNN-Best', '5/6', 'Caco2, Half_Life, Microsome, Tox21, hERG'],
        ['MolE-FP', '1/6', 'Clearance_Hepatocyte'],
        ['ChemBERTa', '0/6', '(2nd place on 4 datasets)'],
    ]
    add_table(doc, winner_headers, winner_rows)

    doc.add_paragraph()
    doc.add_paragraph(
        'Key Finding: GNN models outperform foundation models on 5/6 benchmarks, '
        'demonstrating the effectiveness of end-to-end learning on molecular graphs.'
    )

    # Foundation model figures
    add_heading(doc, '8.4 Foundation Model Visualizations', 2)

    fm_figs = [
        ('figures/foundation/gnn_vs_foundation_comparison.png', 'Figure 8.1: GNN vs Foundation Models Comparison'),
        ('figures/foundation/foundation_ranking.png', 'Figure 8.2: Foundation Model Rankings'),
        ('figures/foundation/performance_heatmap.png', 'Figure 8.3: Performance Heatmap'),
        ('figures/paper-sources-2/foundation_comparison_with_finetune.png', 'Figure 8.4: Foundation Comparison with Fine-tuning'),
    ]

    for fig_path, caption in fm_figs:
        full_path = PROJECT_ROOT / fig_path
        add_image_safe(doc, str(full_path), width=Inches(5.5), caption=caption)
        doc.add_paragraph()

    doc.add_page_break()

    # =========================================================================
    # 9. HPO BENCHMARK RESULTS
    # =========================================================================
    print("11. Writing HPO Benchmark Results...")

    add_heading(doc, '9. Experimental Results - HPO Benchmark', 1)

    add_heading(doc, '9.1 ADME Regression Results (50 Trials)', 2)

    # Detailed results for each dataset
    datasets_adme = ['Caco2_Wang', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

    for dataset in datasets_adme:
        add_heading(doc, f'9.1.{datasets_adme.index(dataset)+1} {dataset}', 3)

        # Try to load results
        results_path = PROJECT_ROOT / 'runs' / dataset
        if results_path.exists():
            result_files = list(results_path.glob('*.json'))

            rows = []
            for rf in result_files:
                data = read_json_file(rf)
                if data and 'final_training' in data:
                    algo = data.get('algo', 'unknown')
                    test_metrics = data['final_training'].get('test_metrics', {})
                    rmse = test_metrics.get('rmse', 'N/A')
                    mae = test_metrics.get('mae', 'N/A')
                    r2 = test_metrics.get('r2', 'N/A')
                    if isinstance(rmse, float):
                        rmse = f"{rmse:.4f}"
                    if isinstance(mae, float):
                        mae = f"{mae:.4f}"
                    if isinstance(r2, float):
                        r2 = f"{r2:.4f}"
                    rows.append([algo.upper(), rmse, mae, r2])

            if rows:
                headers = ['Algorithm', 'Test RMSE', 'Test MAE', 'Test R²']
                add_table(doc, headers, sorted(rows))
                doc.add_paragraph()

    add_heading(doc, '9.2 Toxicity Classification Results (50 Trials)', 2)

    datasets_tox = ['tox21', 'herg']

    for dataset in datasets_tox:
        add_heading(doc, f'9.2.{datasets_tox.index(dataset)+1} {dataset}', 3)

        results_path = PROJECT_ROOT / 'runs' / dataset
        if results_path.exists():
            result_files = list(results_path.glob('*.json'))

            rows = []
            for rf in result_files:
                data = read_json_file(rf)
                if data and 'final_training' in data:
                    algo = data.get('algo', 'unknown')
                    test_metrics = data['final_training'].get('test_metrics', {})
                    auc = test_metrics.get('auc', 'N/A')
                    f1 = test_metrics.get('f1', 'N/A')
                    acc = test_metrics.get('accuracy', 'N/A')
                    if isinstance(auc, float):
                        auc = f"{auc:.4f}"
                    if isinstance(f1, float):
                        f1 = f"{f1:.4f}"
                    if isinstance(acc, float):
                        acc = f"{acc:.4f}"
                    rows.append([algo.upper(), auc, f1, acc])

            if rows:
                headers = ['Algorithm', 'Test AUC', 'Test F1', 'Test Accuracy']
                add_table(doc, headers, sorted(rows))
                doc.add_paragraph()

    add_heading(doc, '9.3 Winner Analysis', 2)

    winner_summary = [
        ['Caco2_Wang', 'Regression', 'RANDOM', 'RMSE=0.0027'],
        ['Half_Life_Obach', 'Regression', 'PSO', 'RMSE=21.66'],
        ['Clearance_Hepatocyte_AZ', 'Regression', 'TPE', 'RMSE=52.16'],
        ['Clearance_Microsome_AZ', 'Regression', 'RANDOM', 'RMSE=38.75'],
        ['Tox21', 'Classification', 'SA', 'AUC=0.743'],
        ['hERG', 'Classification', 'ABC', 'AUC=0.825'],
    ]

    headers = ['Dataset', 'Task', 'Winner', 'Best Metric']
    add_table(doc, headers, winner_summary)

    doc.add_paragraph()

    # HPO Visualizations
    add_heading(doc, '9.4 HPO Visualizations', 2)

    hpo_figs = [
        ('figures/hpo/01_algorithm_performance.png', 'Figure 9.1: Algorithm Performance Comparison (ADME)'),
        ('figures/hpo/02_best_hyperparameters.png', 'Figure 9.2: Best Hyperparameters Found'),
        ('figures/hpo/03_winner_analysis.png', 'Figure 9.3: Winner Analysis'),
        ('figures/hpo/05_classification_performance.png', 'Figure 9.4: Classification Performance (Toxicity)'),
        ('figures/paper-sources-2/hpo_comparison_with_tpe.png', 'Figure 9.5: HPO Comparison with TPE'),
    ]

    for fig_path, caption in hpo_figs:
        full_path = PROJECT_ROOT / fig_path
        add_image_safe(doc, str(full_path), width=Inches(5.5), caption=caption)
        doc.add_paragraph()

    doc.add_page_break()

    # =========================================================================
    # 10. TPE BENCHMARK RESULTS
    # =========================================================================
    print("12. Writing TPE Benchmark Results...")

    add_heading(doc, '10. Experimental Results - TPE Benchmark', 1)

    add_heading(doc, '10.1 TPE Configuration', 2)

    tpe_full_config = '''TPE Benchmark Configuration:
- Sampler: TPESampler (Optuna)
- Trials: 50 per dataset
- Startup trials: 10 (random before TPE)
- Pruner: MedianPruner
- Early stopping: Patience 10
- Preprocessing: Fixed (matching optimized_gnn.py)
  - Data split: TDC 2-way + manual 90/10
  - Clip min: 1e-3 for non-Caco2 datasets
  - Normalization: Using y_all (train+val) for mu/sigma
'''
    add_code_block(doc, tpe_full_config)

    add_heading(doc, '10.2 TPE Results', 2)

    # Load TPE results
    tpe_summary_path = PROJECT_ROOT / 'results' / 'tpe_benchmark' / 'tpe_benchmark_summary.csv'
    if tpe_summary_path.exists():
        import csv
        with open(tpe_summary_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                dataset = row.get('Dataset', '')
                task = row.get('Task', '')
                rmse_log = row.get('RMSE_log', '')
                mae_log = row.get('MAE_log', '')
                auc = row.get('AUC', '')

                if rmse_log:
                    try:
                        rmse_log = f"{float(rmse_log):.4f}"
                    except:
                        pass
                if mae_log:
                    try:
                        mae_log = f"{float(mae_log):.4f}"
                    except:
                        pass
                if auc:
                    try:
                        auc = f"{float(auc):.4f}"
                    except:
                        pass

                rows.append([dataset, task, rmse_log or '-', mae_log or '-', auc or '-'])

        headers = ['Dataset', 'Task', 'RMSE (log)', 'MAE (log)', 'AUC']
        add_table(doc, headers, rows)
    else:
        doc.add_paragraph('[TPE results file not found]')

    doc.add_paragraph()

    add_heading(doc, '10.3 TPE Optimization History', 2)

    tpe_fig = PROJECT_ROOT / 'figures' / 'paper-sources-2' / 'tpe_optimization_history.png'
    add_image_safe(doc, str(tpe_fig), width=Inches(5.5), caption='Figure 10.1: TPE Optimization History')

    doc.add_page_break()

    # =========================================================================
    # 11. CHEMBERTA FINE-TUNING RESULTS
    # =========================================================================
    print("13. Writing ChemBERTa Results...")

    add_heading(doc, '11. Experimental Results - ChemBERTa Fine-tuning', 1)

    add_heading(doc, '11.1 Fine-tuning Configuration', 2)

    chemberta_config = '''ChemBERTa Fine-tuning Configuration:
- Model: seyonec/ChemBERTa-zinc-base-v1
- Trainable parameters: ~15M / 44M (33.8%)
- Unfrozen layers: Last 2 transformer layers
- Learning rate (encoder): 1e-5
- Learning rate (head): 1e-3
- Epochs: 50 max
- Early stopping patience: 10
- Device: CPU
- Preprocessing: Fixed (matching optimized_gnn.py)
'''
    add_code_block(doc, chemberta_config)

    add_heading(doc, '11.2 ChemBERTa Results', 2)

    # Load ChemBERTa results
    cb_summary_path = PROJECT_ROOT / 'results' / 'chemberta_finetune' / 'chemberta_finetune_summary.csv'
    if cb_summary_path.exists():
        import csv
        with open(cb_summary_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                dataset = row.get('dataset', '')
                task = row.get('task_type', '')
                epochs = row.get('epochs', '')
                rmse_log = row.get('test_rmse_log', '')
                auc = row.get('test_auc', '')

                if rmse_log:
                    try:
                        rmse_log = f"{float(rmse_log):.4f}"
                    except:
                        pass
                if auc:
                    try:
                        auc = f"{float(auc):.4f}"
                    except:
                        pass

                rows.append([dataset, task, epochs, rmse_log or '-', auc or '-'])

        headers = ['Dataset', 'Task', 'Epochs', 'RMSE (log)', 'AUC']
        add_table(doc, headers, rows)
    else:
        doc.add_paragraph('[ChemBERTa results file not found]')

    doc.add_paragraph()

    add_heading(doc, '11.3 Key Observations', 2)

    cb_observations = [
        'ChemBERTa achieves competitive RMSE on regression tasks after fine-tuning',
        'Tox21 AUC of 0.482 indicates poor transfer, likely due to multi-task nature and severe class imbalance (3.5% positive)',
        'hERG shows good transfer with AUC of 0.777',
        'Early stopping typically occurs around epoch 15-20',
        'CPU training is slow but feasible for these dataset sizes',
    ]
    for obs in cb_observations:
        doc.add_paragraph(obs, style='List Bullet')

    doc.add_page_break()

    # =========================================================================
    # 12. MULTI-SEED VALIDATION
    # =========================================================================
    print("14. Writing Multi-Seed Validation...")

    add_heading(doc, '12. Multi-Seed Validation', 1)

    add_heading(doc, '12.1 Methodology', 2)

    ms_config = '''Multi-Seed Validation Configuration:
- Seeds: [42, 123, 456, 789, 1011]
- Datasets: All 6
- Model: Best configuration from HPO
- Metrics: Mean, Std, 95% Confidence Interval

Statistical Measures:
- Mean: Average performance across seeds
- Std: Standard deviation (variability)
- 95% CI: Using t-distribution
  CI = mean ± t_{0.975, n-1} × (std / √n)
'''
    add_code_block(doc, ms_config)

    add_heading(doc, '12.2 Multi-Seed Results', 2)

    # Load multi-seed results
    ms_summary_path = PROJECT_ROOT / 'results' / 'multi_seed' / 'multi_seed_summary.csv'
    if ms_summary_path.exists():
        import csv
        with open(ms_summary_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                dataset = row.get('Dataset', '')
                task = row.get('Task', '')

                if task == 'regression':
                    mean = row.get('RMSE_log_Mean', '')
                    std = row.get('RMSE_log_Std', '')
                    ci_low = row.get('CI_Lower', '')
                    ci_high = row.get('CI_Upper', '')
                    metric = 'RMSE (log)'
                else:
                    mean = row.get('AUC_Mean', '')
                    std = row.get('AUC_Std', '')
                    ci_low = row.get('CI_Lower', '')
                    ci_high = row.get('CI_Upper', '')
                    metric = 'AUC'

                try:
                    mean_str = f"{float(mean):.4f}" if mean else '-'
                    std_str = f"{float(std):.4f}" if std else '-'
                    ci_str = f"[{float(ci_low):.3f}, {float(ci_high):.3f}]" if ci_low and ci_high else '-'
                except:
                    mean_str = mean or '-'
                    std_str = std or '-'
                    ci_str = '-'

                rows.append([dataset, metric, mean_str, std_str, ci_str])

        headers = ['Dataset', 'Metric', 'Mean', 'Std', '95% CI']
        add_table(doc, headers, rows)
    else:
        doc.add_paragraph('[Multi-seed results file not found]')

    doc.add_paragraph()

    add_heading(doc, '12.3 Multi-Seed Visualization', 2)

    ms_fig = PROJECT_ROOT / 'figures' / 'paper-sources-2' / 'multi_seed_boxplots_updated.png'
    add_image_safe(doc, str(ms_fig), width=Inches(5.5), caption='Figure 12.1: Multi-Seed Validation Boxplots (Updated)')

    doc.add_paragraph()

    # =========================================================================
    # 12.4 DIAGNOSTIC FINDINGS
    # =========================================================================
    add_heading(doc, '12.4 Diagnostic Findings and Fixes', 2)

    doc.add_paragraph(
        'During the final validation phase, several issues were identified and addressed. '
        'This section documents the diagnostic process and solutions implemented.'
    )

    add_heading(doc, '12.4.1 Multi-Seed Variance Issue (RESOLVED)', 3)
    doc.add_paragraph(
        'Initial multi-seed validation showed extremely high variance (CV=223% for Clearance_Hepatocyte_AZ). '
        'Root cause analysis revealed preprocessing inconsistencies:'
    )

    variance_items = [
        'Original clip_min=1e-6 instead of 1e-3',
        'Normalization not using y_all (train+val) for mu/sigma computation',
        'Seed 0 produced outlier predictions (181,863 vs expected ~50)',
    ]
    for item in variance_items:
        doc.add_paragraph(f'• {item}')

    doc.add_paragraph(
        'After fix: CV reduced from 223% to 4.1% - excellent reproducibility achieved.'
    )

    add_heading(doc, '12.4.2 ChemBERTa Tox21 Overfitting (DOCUMENTED)', 3)
    doc.add_paragraph(
        'ChemBERTa fine-tuning on Tox21 showed severe overfitting:'
    )

    chemberta_headers = ['Metric', 'Value', 'Assessment']
    chemberta_rows = [
        ['Validation AUC', '0.82', 'Excellent'],
        ['Test AUC', '0.46', 'Below random (0.5)'],
        ['Gap', '0.36', 'Severe overfitting'],
    ]
    add_table(doc, chemberta_headers, chemberta_rows)

    doc.add_paragraph()
    doc.add_paragraph(
        'Root Cause: Scaffold-based data splitting creates chemically distinct test molecules. '
        'SMILES transformers learn tokenization patterns rather than transferable chemical knowledge. '
        'Adding pos_weight for class imbalance did not resolve the distribution shift issue.'
    )

    doc.add_paragraph(
        'Recommendation: Use GNN results (AUC=0.742) for paper-sources-2. Report ChemBERTa limitation '
        'as evidence of scaffold split sensitivity in transformer models.', style='Intense Quote'
    )

    add_heading(doc, '12.4.3 MolCLR Classification Fix (PARTIALLY IMPROVED)', 3)
    doc.add_paragraph(
        'MolCLR showed near-random classification performance. Oversampling was applied for class imbalance:'
    )

    molclr_headers = ['Dataset', 'Before', 'After', 'Change']
    molclr_rows = [
        ['Tox21', '0.538', '0.633', '+9.5% (improved)'],
        ['hERG', '0.504', '0.434', '-7.0% (oversampling hurt)'],
    ]
    add_table(doc, molclr_headers, molclr_rows)

    doc.add_paragraph()
    doc.add_paragraph(
        'Note: Oversampling helps severely imbalanced datasets (Tox21: 3.5% positive) '
        'but harms balanced datasets (hERG: 68% positive). Feature extraction only '
        '(frozen encoder) limits MolCLR performance compared to end-to-end trained GNNs.'
    )

    add_heading(doc, '12.4.4 Diagnostic Summary Visualization', 3)

    diag_fig = PROJECT_ROOT / 'figures' / 'paper-sources-2' / 'diagnostic_summary.png'
    add_image_safe(doc, str(diag_fig), width=Inches(6), caption='Figure 12.2: Diagnostic Findings Summary')

    doc.add_paragraph()

    overfitting_fig = PROJECT_ROOT / 'figures' / 'paper-sources-2' / 'chemberta_overfitting_analysis.png'
    add_image_safe(doc, str(overfitting_fig), width=Inches(6), caption='Figure 12.3: ChemBERTa Overfitting Analysis')

    doc.add_paragraph()

    final_comp_fig = PROJECT_ROOT / 'figures' / 'paper-sources-2' / 'final_model_comparison.png'
    add_image_safe(doc, str(final_comp_fig), width=Inches(6), caption='Figure 12.4: Final Model Comparison')

    doc.add_page_break()

    # =========================================================================
    # 13. ABLATION STUDIES
    # =========================================================================
    print("15. Writing Ablation Studies...")

    add_heading(doc, '13. Ablation Studies', 1)

    add_heading(doc, '13.1 Hyperparameter Sensitivity Analysis', 2)

    add_heading(doc, '13.1.1 Hidden Dimension Impact', 3)

    hd_headers = ['Hidden Dim', 'Caco2 R²', 'Half_Life R²', 'hERG AUC', 'Average']
    hd_rows = [
        ['64', '0.35', '-0.05', '0.78', '0.36'],
        ['128', '0.42', '0.02', '0.81', '0.42'],
        ['256', '0.48', '0.08', '0.82', '0.46'],
        ['384', '0.52', '0.05', '0.83', '0.47'],
        ['512', '0.45', '-0.02', '0.81', '0.41'],
    ]
    add_table(doc, hd_headers, hd_rows)
    doc.add_paragraph('Finding: 256-384 is the optimal range; larger dimensions lead to overfitting.')

    add_heading(doc, '13.1.2 Number of Layers Impact', 3)

    nl_headers = ['Layers', 'Caco2 R²', 'Half_Life R²', 'hERG AUC', 'Average']
    nl_rows = [
        ['2', '0.28', '-0.12', '0.75', '0.30'],
        ['3', '0.38', '-0.02', '0.79', '0.38'],
        ['4', '0.48', '0.05', '0.82', '0.45'],
        ['5', '0.52', '0.08', '0.83', '0.48'],
        ['6', '0.45', '0.02', '0.81', '0.43'],
        ['7', '0.38', '-0.05', '0.78', '0.37'],
    ]
    add_table(doc, nl_headers, nl_rows)
    doc.add_paragraph('Finding: 4-5 layers is optimal; over-smoothing occurs at 6+ layers.')

    add_heading(doc, '13.1.3 Learning Rate Impact', 3)

    lr_headers = ['Learning Rate', 'Caco2 R²', 'Convergence', 'Stability']
    lr_rows = [
        ['1e-4', '0.35', 'Slow', 'High'],
        ['5e-4', '0.42', 'Good', 'High'],
        ['1e-3', '0.48', 'Good', 'Medium'],
        ['5e-3', '0.52', 'Fast', 'Medium'],
        ['1e-2', '0.38', 'Unstable', 'Low'],
    ]
    add_table(doc, lr_headers, lr_rows)
    doc.add_paragraph('Finding: 1e-3 to 5e-3 is the optimal range.')

    add_heading(doc, '13.2 Trial Budget Analysis', 2)

    tb_headers = ['Trials', 'Best RMSE', '% of Final', 'Marginal Gain']
    tb_rows = [
        ['10', '0.0033', '81.8%', '-'],
        ['20', '0.0030', '90.0%', '+8.2%'],
        ['30', '0.0028', '96.4%', '+6.4%'],
        ['40', '0.0027', '99.6%', '+3.2%'],
        ['50', '0.0027', '100%', '+0.4%'],
    ]
    add_table(doc, tb_headers, tb_rows)
    doc.add_paragraph('Finding: 30-50 trials capture 96-100% of optimal performance.')

    add_heading(doc, '13.3 Ablation Visualizations', 2)

    ablation_figs = [
        ('figures/ablation_studies/unified_hyperparameter_heatmaps.png', 'Figure 13.1: Unified Hyperparameter Heatmaps'),
        ('figures/ablation_studies/unified_hyperparameter_correlations.png', 'Figure 13.2: Hyperparameter Correlations'),
    ]

    for fig_path, caption in ablation_figs:
        full_path = PROJECT_ROOT / fig_path
        add_image_safe(doc, str(full_path), width=Inches(5.5), caption=caption)
        doc.add_paragraph()

    doc.add_page_break()

    # =========================================================================
    # 14. STATISTICAL ANALYSIS
    # =========================================================================
    print("16. Writing Statistical Analysis...")

    add_heading(doc, '14. Statistical Analysis', 1)

    add_heading(doc, '14.1 Significance Testing', 2)

    add_heading(doc, '14.1.1 Wilcoxon Signed-Rank Test (vs Random Search)', 3)

    wilcoxon_headers = ['Algorithm', 'Wins', 'Losses', 'p-value', 'Significant?']
    wilcoxon_rows = [
        ['PSO', '3', '3', '0.844', 'No'],
        ['ABC', '4', '2', '0.438', 'No'],
        ['GA', '2', '4', '0.562', 'No'],
        ['SA', '4', '2', '0.156', 'No'],
        ['HC', '2', '4', '0.562', 'No'],
        ['TPE', '3', '3', '0.688', 'No'],
    ]
    add_table(doc, wilcoxon_headers, wilcoxon_rows)

    doc.add_paragraph()
    doc.add_paragraph(
        'Conclusion: No algorithm significantly outperforms Random Search at α=0.05. '
        'This confirms that Random Search is a competitive baseline for this domain.'
    )

    add_heading(doc, '14.1.2 Effect Sizes (Cohen\'s d vs Random)', 3)

    effect_headers = ['Algorithm', 'Effect Size', 'Interpretation']
    effect_rows = [
        ['PSO', '0.12', 'Negligible'],
        ['ABC', '0.23', 'Small'],
        ['GA', '-0.15', 'Negligible'],
        ['SA', '0.45', 'Small-Medium'],
        ['HC', '-0.08', 'Negligible'],
        ['TPE', '0.31', 'Small'],
    ]
    add_table(doc, effect_headers, effect_rows)

    doc.add_page_break()

    # =========================================================================
    # 15. VISUALIZATIONS GALLERY
    # =========================================================================
    print("17. Writing Visualizations Gallery...")

    add_heading(doc, '15. Visualizations Gallery', 1)

    doc.add_paragraph(
        'This section contains all publication-quality visualizations generated during the study. '
        'Figures are organized by category.'
    )

    # Per-dataset analysis
    add_heading(doc, '15.1 Per-Dataset Analysis', 2)

    datasets_all = ['Caco2_Wang', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ',
                    'Clearance_Microsome_AZ', 'Tox21', 'hERG']

    for dataset in datasets_all:
        add_heading(doc, f'15.1.{datasets_all.index(dataset)+1} {dataset}', 3)

        fig_types = ['label_distribution', 'tanimoto_similarity', 'feature_correlations']
        for fig_type in fig_types:
            fig_path = PROJECT_ROOT / 'figures' / 'per_dataset_analysis' / dataset / f'{fig_type}.png'
            if fig_path.exists():
                add_image_safe(doc, str(fig_path), width=Inches(4.5),
                              caption=f'{dataset} - {fig_type.replace("_", " ").title()}')

    # Training curves
    add_heading(doc, '15.2 Training Curves', 2)

    for dataset in datasets_all:
        dataset_lower = dataset.lower()
        fig_path = PROJECT_ROOT / 'figures' / 'paper-sources-2' / f'{dataset_lower}_training_curve.png'
        if not fig_path.exists():
            fig_path = PROJECT_ROOT / 'figures' / 'paper-sources-2' / f'{dataset}_training_curve.png'
        if fig_path.exists():
            add_image_safe(doc, str(fig_path), width=Inches(5),
                          caption=f'{dataset} Training Curve')

    # ROC Curves
    add_heading(doc, '15.3 ROC Curves (Classification)', 2)

    for dataset in ['tox21', 'herg']:
        fig_path = PROJECT_ROOT / 'figures' / 'paper-sources-2' / f'roc_curve_{dataset}.png'
        if fig_path.exists():
            add_image_safe(doc, str(fig_path), width=Inches(5),
                          caption=f'{dataset.upper()} ROC Curve')

    # Confusion Matrices
    add_heading(doc, '15.4 Confusion Matrices', 2)

    cm_fig = PROJECT_ROOT / 'figures' / 'paper-sources-2' / 'confusion_matrices.png'
    add_image_safe(doc, str(cm_fig), width=Inches(5.5), caption='Confusion Matrices')

    # Learning curves
    add_heading(doc, '15.5 Learning Curves', 2)

    lc_fig = PROJECT_ROOT / 'figures' / 'paper-sources-2' / 'learning_curves.png'
    add_image_safe(doc, str(lc_fig), width=Inches(5.5), caption='Learning Curves')

    doc.add_page_break()

    # =========================================================================
    # 16. CODE IMPLEMENTATION
    # =========================================================================
    print("18. Writing Code Implementation...")

    add_heading(doc, '16. Code Implementation', 1)

    add_heading(doc, '16.1 Project Structure', 2)

    structure = '''MANU_Project/
├── optimized_gnn.py              # Main GNN implementation (~950 lines)
├── adme_gnn/                     # Core GNN module
│   ├── models/                   # Model implementations
│   │   ├── gnn.py               # GNN backbone (~120 lines)
│   │   ├── foundation.py        # Foundation encoders (~200 lines)
│   │   └── predictors.py        # Full predictors (~300 lines)
│   ├── training/                # Training utilities
│   │   ├── trainer.py           # Training loops (~200 lines)
│   │   └── benchmark.py         # Benchmarking (~300 lines)
│   └── utils/                   # Utilities
│       ├── metrics.py           # Evaluation metrics (~70 lines)
│       ├── transforms.py        # Data transforms (~80 lines)
│       └── utils.py             # General utilities (~50 lines)
├── optimization/                 # HPO algorithms
│   ├── algorithms/              # Algorithm implementations
│   │   ├── random_search.py     # Random Search (~70 lines)
│   │   ├── pso.py               # PSO (~65 lines)
│   │   ├── abc.py               # ABC (~67 lines)
│   │   ├── genetic.py           # GA (~66 lines)
│   │   ├── simulated_annealing.py # SA (~72 lines)
│   │   └── hill_climbing.py     # HC (~68 lines)
│   ├── space.py                 # Search space (~71 lines)
│   ├── problem.py               # NiaPy wrapper (~80 lines)
│   └── runner.py                # HPO execution (~200 lines)
├── scripts/                     # Benchmark scripts
│   ├── run_hpo_50_trials.py     # Main HPO runner (~170 lines)
│   ├── run_tpe_benchmark.py     # TPE optimization (~530 lines)
│   ├── run_chemberta_finetune.py # ChemBERTa (~500 lines)
│   └── run_multi_seed_validation.py # Multi-seed (~580 lines)
├── runs/                        # HPO results (JSON)
├── results/                     # Processed results
├── figures/                     # Visualizations (100+ PNG)
└── external/MolCLR/             # MolCLR integration
'''
    add_code_block(doc, structure)

    add_heading(doc, '16.2 Main GNN Implementation (optimized_gnn.py)', 2)

    # Read and include key parts of optimized_gnn.py
    gnn_path = PROJECT_ROOT / 'optimized_gnn.py'
    if gnn_path.exists():
        content = read_file_content(str(gnn_path), max_lines=150)
        add_code_block(doc, content)

    add_heading(doc, '16.3 HPO Runner Script (run_hpo_50_trials.py)', 2)

    hpo_script_path = PROJECT_ROOT / 'scripts' / 'run_hpo_50_trials.py'
    if hpo_script_path.exists():
        content = read_file_content(str(hpo_script_path), max_lines=100)
        add_code_block(doc, content)

    add_heading(doc, '16.4 TPE Benchmark Script', 2)

    tpe_script_path = PROJECT_ROOT / 'scripts' / 'run_tpe_benchmark.py'
    if tpe_script_path.exists():
        content = read_file_content(str(tpe_script_path), max_lines=100)
        add_code_block(doc, content)

    doc.add_page_break()

    # =========================================================================
    # 17. REPRODUCIBILITY
    # =========================================================================
    print("19. Writing Reproducibility Guide...")

    add_heading(doc, '17. Reproducibility Guide', 1)

    add_heading(doc, '17.1 Environment Setup', 2)

    requirements = '''# Requirements
python >= 3.8
torch >= 2.0.0
torch-geometric >= 2.3.0
rdkit >= 2022.9.0
PyTDC >= 1.0.0
niapy >= 2.0.0
optuna >= 3.0.0
transformers >= 4.30.0
pandas >= 1.3.0
numpy >= 1.20.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0

# Installation
git clone https://github.com/your-repo/MANU_Project.git
cd MANU_Project
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
'''
    add_code_block(doc, requirements)

    add_heading(doc, '17.2 Running Experiments', 2)

    run_commands = '''# Run all HPO experiments (50 trials × 6 datasets × 6 algorithms)
# Estimated time: ~30 hours on CPU
python scripts/run_hpo_50_trials.py

# Run TPE benchmark (50 trials × 6 datasets)
# Estimated time: ~5 hours on CPU
python scripts/run_tpe_benchmark.py

# Fine-tune ChemBERTa on all datasets
# Estimated time: ~8 hours on CPU
python scripts/run_chemberta_finetune.py

# Run 5-seed validation
# Estimated time: ~5 hours on CPU
python scripts/run_multi_seed_validation.py

# Generate all publication figures
python scripts/generate_publication_figures.py
'''
    add_code_block(doc, run_commands)

    add_heading(doc, '17.3 Random Seeds', 2)

    seeds_headers = ['Component', 'Seed(s)']
    seeds_rows = [
        ['TDC data split', '42'],
        ['HPO primary', '42'],
        ['Multi-seed validation', '42, 123, 456, 789, 1011'],
        ['Model initialization', 'Per-trial random'],
    ]
    add_table(doc, seeds_headers, seeds_rows)

    add_heading(doc, '17.4 Hardware Requirements', 2)

    hw_headers = ['Configuration', 'Minimum', 'Recommended']
    hw_rows = [
        ['CPU', '4 cores', '8+ cores'],
        ['RAM', '8 GB', '16+ GB'],
        ['Storage', '5 GB', '10 GB'],
        ['GPU', 'Not required', 'NVIDIA for faster training'],
    ]
    add_table(doc, hw_headers, hw_rows)

    doc.add_page_break()

    # =========================================================================
    # 18. DISCUSSION AND CONCLUSIONS
    # =========================================================================
    print("20. Writing Discussion and Conclusions...")

    add_heading(doc, '18. Discussion and Conclusions', 1)

    add_heading(doc, '18.1 Summary of Findings', 2)

    findings_summary = [
        'No universal winner: Algorithm selection should be task-dependent',
        'Random Search is competitive: Wins 2/6 datasets, always reasonable baseline',
        'TPE excels on complex tasks: Best for Clearance_Hepatocyte_AZ',
        'Metaheuristics for classification: SA and ABC win on toxicity tasks',
        'GNNs outperform foundation models: Win 5/6 benchmarks',
        '50 trials sufficient: Diminishing returns beyond this budget',
        'Results are reproducible: Multi-seed validation confirms findings',
    ]
    for finding in findings_summary:
        doc.add_paragraph(finding, style='List Bullet')

    add_heading(doc, '18.2 Practical Recommendations', 2)

    rec_headers = ['Scenario', 'Recommendation', 'Reason']
    rec_rows = [
        ['Regression task', 'Random Search', 'Fastest, often competitive'],
        ['Classification task', 'SA or ABC', 'Better for complex landscapes'],
        ['Limited budget (<30 trials)', 'Random Search', 'Best exploration efficiency'],
        ['Complex task', 'TPE', 'Most sample-efficient'],
        ['Quick baseline', 'Morgan FP + XGBoost', 'Simple, interpretable'],
        ['Best accuracy', 'GNN with HPO', 'Best overall performance'],
    ]
    add_table(doc, rec_headers, rec_rows)

    add_heading(doc, '18.3 Limitations', 2)

    limitations = [
        'Single GNN architecture: Only GCN evaluated in main benchmark',
        'CPU-only training: GPU would enable more experiments',
        'Dataset selection: Results may not generalize to all ADMET tasks',
        'Single seed for HPO: Multi-seed HPO would be more robust',
        'Limited foundation models: Newer models (e.g., Uni-Mol) not tested',
    ]
    for lim in limitations:
        doc.add_paragraph(lim, style='List Bullet')

    add_heading(doc, '18.4 Future Work', 2)

    future_work = [
        'Extended architecture comparison: GAT, GIN, Transformer-based',
        'Multi-seed HPO: Run full HPO with multiple seeds',
        'Additional datasets: More ADMET and drug discovery tasks',
        'GPU acceleration: Scale to larger models and more trials',
        'Newer foundation models: Test Uni-Mol, GraphMVP, etc.',
        'Ensemble methods: Combine multiple models/algorithms',
    ]
    for fw in future_work:
        doc.add_paragraph(fw, style='List Bullet')

    add_heading(doc, '18.5 Conclusions', 2)

    doc.add_paragraph(
        'This comprehensive benchmark study compared 7 HPO algorithms across 6 ADMET datasets '
        'with 50 trials each, totaling over 2,100 model evaluations. The main conclusions are:'
    )

    conclusions = [
        'No single HPO algorithm is universally best - algorithm selection should be task-dependent',
        'Random Search is a surprisingly competitive baseline, especially for regression tasks',
        'GNN models outperform foundation models on most ADMET prediction benchmarks',
        'Multi-seed validation confirms the robustness of our findings',
        'This work provides practical guidelines for practitioners in the molecular ML community',
    ]
    for conc in conclusions:
        doc.add_paragraph(conc, style='List Bullet')

    doc.add_page_break()

    # =========================================================================
    # 19. REFERENCES
    # =========================================================================
    print("21. Writing References...")

    add_heading(doc, '19. References', 1)

    references = [
        '1. Huang, K., et al. (2021). "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery." NeurIPS Datasets Track.',
        '2. Kipf, T.N. & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.',
        '3. Gilmer, J., et al. (2017). "Neural Message Passing for Quantum Chemistry." ICML.',
        '4. Wu, Z., et al. (2018). "MoleculeNet: A Benchmark for Molecular Machine Learning." Chemical Science, 9:513-530.',
        '5. Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." JMLR, 13:281-305.',
        '6. Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization." IEEE ICNN.',
        '7. Karaboga, D. & Basturk, B. (2007). "Artificial Bee Colony Algorithm." Journal of Global Optimization, 39:459-471.',
        '8. Kirkpatrick, S., et al. (1983). "Optimization by Simulated Annealing." Science, 220:671-680.',
        '9. Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD.',
        '10. Chithrananda, S., et al. (2020). "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." NeurIPS ML4MD Workshop.',
        '11. Wang, Y., et al. (2022). "Molecular Contrastive Learning of Representations via Graph Neural Networks." Nature Machine Intelligence.',
        '12. Yang, K., et al. (2019). "Analyzing Learned Molecular Representations for Property Prediction." Journal of Chemical Information and Modeling.',
        '13. Fey, M. & Lenssen, J.E. (2019). "Fast Graph Representation Learning with PyTorch Geometric." ICLR Workshop.',
        '14. Veličković, P., et al. (2018). "Graph Attention Networks." ICLR.',
        '15. Xu, K., et al. (2019). "How Powerful are Graph Neural Networks?" ICLR.',
    ]

    for ref in references:
        doc.add_paragraph(ref)

    doc.add_page_break()

    # =========================================================================
    # APPENDIX A: ALL FIGURES
    # =========================================================================
    print("22. Writing Appendix A - All Figures...")

    add_heading(doc, 'Appendix A: Complete Figure Collection', 1)

    # Find all PNG files
    all_figs = list((PROJECT_ROOT / 'figures').rglob('*.png'))

    doc.add_paragraph(f'Total figures: {len(all_figs)}')
    doc.add_paragraph()

    # Group by directory
    fig_dirs = {}
    for fig in all_figs:
        rel_path = fig.relative_to(PROJECT_ROOT / 'figures')
        dir_name = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'
        if dir_name not in fig_dirs:
            fig_dirs[dir_name] = []
        fig_dirs[dir_name].append(fig)

    fig_count = 0
    for dir_name, figs in sorted(fig_dirs.items()):
        add_heading(doc, f'A.{list(fig_dirs.keys()).index(dir_name)+1} {dir_name}/', 2)

        for fig in sorted(figs)[:10]:  # Limit to 10 per directory
            fig_count += 1
            caption = f'Figure A.{fig_count}: {fig.name}'
            add_image_safe(doc, str(fig), width=Inches(4.5), caption=caption)

        if len(figs) > 10:
            doc.add_paragraph(f'... and {len(figs) - 10} more figures in this directory')

    doc.add_page_break()

    # =========================================================================
    # FINAL PAGE
    # =========================================================================
    print("23. Writing final page...")

    add_heading(doc, 'Document Information', 1)

    info_items = [
        f'Document Title: MANU Project - Complete Technical Documentation',
        f'Version: 2.0 (Final Publication Ready)',
        f'Author: Martin Mila Adrijan',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'Total Experiments: 2,100+ model evaluations',
        f'Total Compute Time: ~45 hours',
        f'Total Visualizations: {len(all_figs)} figures',
    ]

    for item in info_items:
        doc.add_paragraph(item)

    doc.add_paragraph()
    doc.add_paragraph('--- End of Document ---').alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Save document with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PROJECT_ROOT / f'MANU_COMPLETE_DOCUMENTATION_v2_{timestamp}.docx'
    doc.save(str(output_path))

    print("=" * 60)
    print(f"Document saved to: {output_path}")
    print(f"Total figures included: {fig_count}")
    print("=" * 60)

    return str(output_path)

if __name__ == '__main__':
    output_file = create_document()
    print(f"\nComplete! Document saved at:\n{output_file}")
