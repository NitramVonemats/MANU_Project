"""
Create unified comparative visualizations across all datasets.
All datasets on one plot for easy comparison.
"""
import os
import pandas as pd
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
os.makedirs('figures/comparative', exist_ok=True)


def load_all_datasets():
    """Load all ADME and Toxicity datasets."""
    datasets = {}

    # ADME datasets (regression)
    adme_datasets = ['Caco2_Wang', 'Clearance_Hepatocyte_AZ',
                     'Clearance_Microsome_AZ', 'Half_Life_Obach']
    for name in adme_datasets:
        path = f'datasets/adme/{name}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Dataset'] = name
            df['Type'] = 'ADME'
            df['Task'] = 'Regression'
            datasets[name] = df
            print(f"Loaded {name}: {len(df)} molecules")

    # Toxicity datasets (classification)
    tox_datasets = ['Tox21', 'hERG', 'ClinTox']
    for name in tox_datasets:
        path = f'datasets/toxicity/{name}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Dataset'] = name
            df['Type'] = 'Toxicity'
            df['Task'] = 'Classification'
            datasets[name] = df
            print(f"Loaded {name}: {len(df)} molecules")

    return datasets


def plot_dataset_overview(datasets):
    """Create unified dataset overview comparison."""
    print("\n[1/5] Creating dataset overview comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Collect statistics
    stats = []
    for name, df in datasets.items():
        stats.append({
            'Dataset': name,
            'Type': df['Type'].iloc[0],
            'Task': df['Task'].iloc[0],
            'Size': len(df),
            'MW': df['MW'].mean(),
            'LogP': df['LogP'].mean(),
            'TPSA': df['TPSA'].mean(),
        })
    stats_df = pd.DataFrame(stats)

    # Subplot 1: Dataset sizes
    ax = axes[0, 0]
    colors = {'ADME': '#4472C4', 'Toxicity': '#ED7D31'}
    bar_colors = [colors[t] for t in stats_df['Type']]
    bars = ax.bar(range(len(stats_df)), stats_df['Size'], color=bar_colors, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df['Dataset'], rotation=45, ha='right')
    ax.set_ylabel('Number of Molecules', fontweight='bold')
    ax.set_title('(A) Dataset Sizes', fontsize=12, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, size) in enumerate(zip(bars, stats_df['Size'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(size)}', ha='center', va='bottom', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['ADME'], label='ADME'),
                      Patch(facecolor=colors['Toxicity'], label='Toxicity')]
    ax.legend(handles=legend_elements, loc='upper right')

    # Subplot 2: Molecular Weight distributions
    ax = axes[0, 1]
    for name, df in datasets.items():
        dtype = df['Type'].iloc[0]
        color = colors[dtype]
        ax.hist(df['MW'], bins=40, alpha=0.4, label=name, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Molecular Weight (g/mol)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('(B) Molecular Weight Distributions', fontsize=12, fontweight='bold', loc='left')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    # Subplot 3: Chemical space (LogP vs TPSA)
    ax = axes[1, 0]
    for name, df in datasets.items():
        dtype = df['Type'].iloc[0]
        color = colors[dtype]
        ax.scatter(df['LogP'], df['TPSA'], alpha=0.3, s=20, label=name, color=color)
    ax.set_xlabel('LogP (Lipophilicity)', fontweight='bold')
    ax.set_ylabel('TPSA (Å²)', fontweight='bold')
    ax.set_title('(C) Chemical Space Coverage', fontsize=12, fontweight='bold', loc='left')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Subplot 4: Property comparison (boxplots)
    ax = axes[1, 1]
    props_data = []
    for name, df in datasets.items():
        for prop in ['MW', 'LogP', 'TPSA']:
            props_data.extend([{
                'Dataset': name,
                'Type': df['Type'].iloc[0],
                'Property': prop,
                'Value': val
            } for val in df[prop]])
    props_df = pd.DataFrame(props_data)

    # Plot grouped boxplot
    property_order = ['MW', 'LogP', 'TPSA']
    positions = []
    labels = []
    for i, prop in enumerate(property_order):
        prop_data = props_df[props_df['Property'] == prop]
        for j, dataset in enumerate(stats_df['Dataset']):
            dataset_data = prop_data[prop_data['Dataset'] == dataset]['Value']
            if len(dataset_data) > 0:
                pos = i * (len(stats_df) + 1) + j
                dtype = stats_df[stats_df['Dataset'] == dataset]['Type'].iloc[0]
                bp = ax.boxplot([dataset_data], positions=[pos], widths=0.6,
                               patch_artist=True, showfliers=False)
                bp['boxes'][0].set_facecolor(colors[dtype])
                bp['boxes'][0].set_alpha(0.7)
                if j == 0:
                    positions.append(pos + len(stats_df)/2 - 0.5)
                    labels.append(prop)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Normalized Value', fontweight='bold')
    ax.set_title('(D) Property Distributions by Dataset', fontsize=12, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Dataset Overview Comparison (All Datasets)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/comparative/01_dataset_overview.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/comparative/01_dataset_overview.png")
    plt.close()


def plot_label_distributions(datasets):
    """Create unified label distribution comparison."""
    print("\n[2/5] Creating label distribution comparison...")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Regression datasets (ADME) - top row (full width)
    ax_reg = fig.add_subplot(gs[0, :])

    adme_datasets = {k: v for k, v in datasets.items() if v['Type'].iloc[0] == 'ADME'}
    positions = []
    labels = []
    colors_list = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']

    for i, (name, df) in enumerate(adme_datasets.items()):
        bp = ax_reg.violinplot([df['Y']], positions=[i], widths=0.7, showmeans=True, showmedians=True)
        for pc in bp['bodies']:
            pc.set_facecolor(colors_list[i % len(colors_list)])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        positions.append(i)
        # Add statistics text
        mean_val = df['Y'].mean()
        std_val = df['Y'].std()
        ax_reg.text(i, ax_reg.get_ylim()[1] * 0.95,
                   f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                   ha='center', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        labels.append(name.replace('_', '\n'))

    ax_reg.set_xticks(positions)
    ax_reg.set_xticklabels(labels, fontsize=10)
    ax_reg.set_ylabel('Target Value', fontweight='bold', fontsize=12)
    ax_reg.set_title('(A) ADME Datasets: Label Distributions (Regression Tasks)',
                     fontsize=14, fontweight='bold', loc='left', pad=15)
    ax_reg.grid(axis='y', alpha=0.3)

    # Classification datasets (Tox) - bottom half
    tox_datasets = {k: v for k, v in datasets.items() if v['Type'].iloc[0] == 'Toxicity'}

    for idx, (name, df) in enumerate(tox_datasets.items()):
        ax = fig.add_subplot(gs[1, idx])

        # Class balance
        class_counts = df['Y'].value_counts().sort_index()
        total = class_counts.sum()

        bars = ax.bar(['Negative', 'Positive'], class_counts.values,
                     color=['#70AD47', '#C55A11'], edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add percentage labels
        for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
            height = bar.get_height()
            percentage = 100 * count / total
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f'({chr(66+idx)}) {name}\n(Balance: {class_counts.values[1]}/{total})',
                    fontsize=12, fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(class_counts.values) * 1.15)

    plt.suptitle('Label Distribution Comparison (All Datasets)', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('figures/comparative/02_label_distributions.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/comparative/02_label_distributions.png")
    plt.close()


def plot_feature_importance(datasets):
    """Create unified feature importance comparison."""
    print("\n[3/5] Creating feature importance comparison...")

    feature_names = ['MW', 'LogP', 'TPSA', 'HBA', 'HBD',
                    'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3']

    importance_data = {}

    for name, df in datasets.items():
        print(f"  Computing feature importance for {name}...")
        X = df[feature_names].fillna(0)
        y = df['Y']

        # Train RF to get feature importance
        if df['Task'].iloc[0] == 'Regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X, y)
        importance_data[name] = model.feature_importances_

    importance_df = pd.DataFrame(importance_data, index=feature_names).T

    # Create two plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Heatmap
    ax = axes[0]
    im = ax.imshow(importance_df.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.4)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df.index)

    # Add text annotations
    for i in range(len(importance_df)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{importance_df.values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feature Importance', fontweight='bold')
    ax.set_title('(A) Feature Importance Heatmap', fontsize=12, fontweight='bold', loc='left', pad=15)

    # Plot 2: Grouped bar chart
    ax = axes[1]
    x = np.arange(len(importance_df))
    width = 0.1

    colors_features = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))

    for i, feature in enumerate(feature_names):
        offset = (i - len(feature_names)/2) * width
        ax.bar(x + offset, importance_df[feature], width,
              label=feature, color=colors_features[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Importance Score', fontweight='bold')
    ax.set_title('(B) Feature Importance by Dataset', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(importance_df.index, rotation=45, ha='right')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Importance Comparison (All Datasets)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/comparative/03_feature_importance.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/comparative/03_feature_importance.png")
    plt.close()


def compute_tanimoto_similarities(smiles_list, sample_size=1000):
    """Compute Tanimoto similarities for a dataset."""
    # Sample if too large
    if len(smiles_list) > sample_size:
        smiles_list = np.random.choice(smiles_list, sample_size, replace=False)

    # Generate fingerprints
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)

    # Compute pairwise similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    return np.array(similarities)


def plot_tanimoto_similarity(datasets):
    """Create unified Tanimoto similarity comparison."""
    print("\n[4/5] Creating Tanimoto similarity comparison...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    similarity_stats = []
    all_similarities = []

    for name, df in datasets.items():
        print(f"  Computing similarities for {name}...")
        sims = compute_tanimoto_similarities(df['SMILES'].values, sample_size=800)

        similarity_stats.append({
            'Dataset': name,
            'Type': df['Type'].iloc[0],
            'Mean': sims.mean(),
            'Median': np.median(sims),
            'Std': sims.std()
        })

        all_similarities.append({
            'dataset': name,
            'type': df['Type'].iloc[0],
            'similarities': sims
        })

    stats_df = pd.DataFrame(similarity_stats)

    # Plot 1: Mean similarity comparison
    ax = axes[0]
    colors = {'ADME': '#4472C4', 'Toxicity': '#ED7D31'}
    bar_colors = [colors[t] for t in stats_df['Type']]

    bars = ax.bar(range(len(stats_df)), stats_df['Mean'], color=bar_colors,
                 edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add error bars (std)
    ax.errorbar(range(len(stats_df)), stats_df['Mean'], yerr=stats_df['Std'],
               fmt='none', ecolor='black', capsize=5, linewidth=2)

    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df['Dataset'], rotation=45, ha='right')
    ax.set_ylabel('Mean Tanimoto Similarity', fontweight='bold')
    ax.set_title('(A) Mean Tanimoto Similarity Across Datasets', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.8)

    # Add value labels
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, stats_df['Mean'], stats_df['Std'])):
        ax.text(bar.get_x() + bar.get_width()/2., mean_val + std_val + 0.02,
               f'{mean_val:.3f}±{std_val:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Similarity distributions (violin plot)
    ax = axes[1]

    violin_data = []
    positions = []
    dataset_labels = []

    for i, item in enumerate(all_similarities):
        # Sample for visualization
        sample_sims = item['similarities'][:500] if len(item['similarities']) > 500 else item['similarities']
        violin_data.append(sample_sims)
        positions.append(i)
        dataset_labels.append(item['dataset'])

    parts = ax.violinplot(violin_data, positions=positions, widths=0.7,
                         showmeans=True, showmedians=True)

    # Color by type
    for i, (pc, item) in enumerate(zip(parts['bodies'], all_similarities)):
        pc.set_facecolor(colors[item['type']])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(dataset_labels, rotation=45, ha='right')
    ax.set_ylabel('Tanimoto Similarity', fontweight='bold')
    ax.set_title('(B) Similarity Distribution Comparison', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Tanimoto Similarity Comparison (All Datasets)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/comparative/04_tanimoto_similarity.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/comparative/04_tanimoto_similarity.png")
    plt.close()


def create_summary_table(datasets):
    """Create comprehensive summary statistics table."""
    print("\n[5/5] Creating summary statistics table...")

    summary_data = []

    for name, df in datasets.items():
        dtype = df['Type'].iloc[0]
        task = df['Task'].iloc[0]

        summary_data.append({
            'Dataset': name,
            'Type': dtype,
            'Task': task,
            'Size': len(df),
            'Avg MW': f"{df['MW'].mean():.1f}",
            'Avg LogP': f"{df['LogP'].mean():.2f}",
            'Avg TPSA': f"{df['TPSA'].mean():.1f}",
            'Label Range': f"{df['Y'].min():.2f} - {df['Y'].max():.2f}" if task == 'Regression' else "Binary (0/1)"
        })

    summary_df = pd.DataFrame(summary_data)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12]*len(summary_df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows by type
    for i in range(1, len(summary_df) + 1):
        color = '#E7E6E6' if summary_df.iloc[i-1]['Type'] == 'ADME' else '#F4B084'
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(1)

    plt.title('Dataset Summary Statistics (All Datasets)',
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig('figures/comparative/05_summary_table.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/comparative/05_summary_table.png")
    plt.close()

    # Also save as CSV
    summary_df.to_csv('figures/comparative/summary_statistics.csv', index=False)
    print("  Saved: figures/comparative/summary_statistics.csv")


def main():
    """Main function to create all unified visualizations."""
    print("="*80)
    print("CREATING UNIFIED COMPARATIVE VISUALIZATIONS")
    print("All datasets on unified plots for easy comparison")
    print("="*80)

    # Load all datasets
    print("\nLoading all datasets...")
    datasets = load_all_datasets()

    if not datasets:
        print("\n[ERROR] No datasets found!")
        print("Please run download_adme_datasets.py and download_tox_datasets.py first.")
        return

    print(f"\nLoaded {len(datasets)} datasets:")
    for name in datasets.keys():
        print(f"  - {name}")

    # Create all visualizations
    plot_dataset_overview(datasets)
    plot_label_distributions(datasets)
    plot_feature_importance(datasets)
    plot_tanimoto_similarity(datasets)
    create_summary_table(datasets)

    print("\n" + "="*80)
    print("[OK] ALL UNIFIED VISUALIZATIONS CREATED!")
    print("="*80)
    print("\nGenerated files in figures/comparative/:")
    print("  1. 01_dataset_overview.png")
    print("  2. 02_label_distributions.png")
    print("  3. 03_feature_importance.png")
    print("  4. 04_tanimoto_similarity.png")
    print("  5. 05_summary_table.png")
    print("  6. summary_statistics.csv")
    print("\n[OK] All datasets are now on unified plots for easy comparison!")


if __name__ == "__main__":
    main()
