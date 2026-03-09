"""
MODEL ARCHITECTURE DIAGRAMS
===========================
Create visual diagrams of different model architectures:
1. GNN-only architecture
2. Foundation-only (ChemBERTa/MolFormer) architecture
3. Hybrid (GNN + Foundation) architecture
4. Complete pipeline flowchart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUTPUT_DIR = Path("visualizations/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CREATING ARCHITECTURE DIAGRAMS")
print("="*80)

# ===================== DIAGRAM 1: GNN-ONLY ARCHITECTURE =====================

def draw_gnn_architecture():
    """Draw GNN-only model architecture"""

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'GNN-Only Architecture (Phase 1 Baseline)',
            fontsize=18, fontweight='bold', ha='center')

    # Input layer
    input_box = FancyBboxPatch((1, 10), 3, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 10.4, 'Molecular Graph\n(Atoms + Bonds)', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Arrow down
    arrow1 = FancyArrowPatch((2.5, 10), (2.5, 9.2),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)

    # Atom features
    feat_box = FancyBboxPatch((0.5, 8.5), 4, 0.6, boxstyle="round,pad=0.05",
                              edgecolor='darkblue', facecolor='#E6F3FF', linewidth=1.5)
    ax.add_patch(feat_box)
    ax.text(2.5, 8.8, 'Atom Features (27d): atomic number, degree, hybridization...',
            ha='center', va='center', fontsize=9)

    # GNN layers
    gnn_layers = ['GNN Layer 1\n(SAGEConv/GINEConv)', 'GNN Layer 2', 'GNN Layer 3']
    y_positions = [7.5, 6.3, 5.1]

    for i, (layer, y_pos) in enumerate(zip(gnn_layers, y_positions)):
        # Layer box
        layer_box = FancyBboxPatch((1, y_pos), 3, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
        ax.add_patch(layer_box)
        ax.text(2.5, y_pos + 0.4, layer, ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Arrow down
        if i < len(gnn_layers) - 1:
            arrow = FancyArrowPatch((2.5, y_pos), (2.5, y_pos - 0.4),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax.add_patch(arrow)

    # Pooling
    arrow_pool = FancyArrowPatch((2.5, 5.1), (2.5, 4.3),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_pool)

    pool_box = FancyBboxPatch((0.8, 3.5), 3.4, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(pool_box)
    ax.text(2.5, 3.9, 'Graph Pooling\n(Mean + Max)', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # ADME features (separate branch)
    adme_box = FancyBboxPatch((5.5, 7), 3.5, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='orange', facecolor='#FFE6CC', linewidth=2)
    ax.add_patch(adme_box)
    ax.text(7.25, 7.4, 'ADME Features (20d)\nMW, LogP, HBD, HBA...',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows to concatenation
    arrow_graph = FancyArrowPatch((2.5, 3.5), (4, 2.5),
                                 arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_graph)

    arrow_adme = FancyArrowPatch((7.25, 7), (6, 2.5),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_adme)

    # Concatenation
    concat_box = FancyBboxPatch((3.5, 1.8), 3, 0.6, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='yellow', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(5, 2.1, 'Concatenate [Graph + ADME]', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # MLP predictor
    arrow_mlp = FancyArrowPatch((5, 1.8), (5, 1.1),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_mlp)

    mlp_box = FancyBboxPatch((3.5, 0.3), 3, 0.8, boxstyle="round,pad=0.1",
                             edgecolor='darkred', facecolor='#FFB3B3', linewidth=2)
    ax.add_patch(mlp_box)
    ax.text(5, 0.7, 'MLP Predictor\n[148 → 128 → 64 → 1]', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Output
    arrow_out = FancyArrowPatch((5, 0.3), (5, -0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_out)

    ax.text(5, -0.7, 'Predicted Value', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

    plt.tight_layout()
    filename = OUTPUT_DIR / 'architecture_gnn_only.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()

# ===================== DIAGRAM 2: FOUNDATION-ONLY ARCHITECTURE =====================

def draw_foundation_architecture():
    """Draw Foundation model (ChemBERTa/MolFormer) architecture"""

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'Foundation Model Architecture (ChemBERTa/MolFormer)',
            fontsize=18, fontweight='bold', ha='center')

    # Input SMILES
    input_box = FancyBboxPatch((1, 10), 3, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 10.4, 'SMILES String\n"CCO"', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Arrow down
    arrow1 = FancyArrowPatch((2.5, 10), (2.5, 9.2),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)

    # Tokenizer
    tok_box = FancyBboxPatch((1, 8.5), 3, 0.6, boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='#CCE5FF', linewidth=2)
    ax.add_patch(tok_box)
    ax.text(2.5, 8.8, 'Tokenizer (BPE)\n["C", "C", "O"]', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Arrow down
    arrow2 = FancyArrowPatch((2.5, 8.5), (2.5, 7.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)

    # Pretrained Transformer
    trans_box = FancyBboxPatch((0.5, 5.5), 4, 2, boxstyle="round,pad=0.1",
                               edgecolor='darkgreen', facecolor='#CCFFCC', linewidth=3)
    ax.add_patch(trans_box)
    ax.text(2.5, 6.8, 'Pretrained Transformer', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(2.5, 6.3, 'ChemBERTa: 77M molecules', ha='center', va='center', fontsize=9)
    ax.text(2.5, 5.9, 'MolFormer: 1.1B molecules', ha='center', va='center', fontsize=9)

    # Arrow down
    arrow3 = FancyArrowPatch((2.5, 5.5), (2.5, 4.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)

    # Mean pooling
    pool_box = FancyBboxPatch((1, 4), 3, 0.6, boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(pool_box)
    ax.text(2.5, 4.3, 'Mean Pooling\n(384d/768d → 256d)', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # ADME features
    adme_box = FancyBboxPatch((5.5, 6.5), 3.5, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='orange', facecolor='#FFE6CC', linewidth=2)
    ax.add_patch(adme_box)
    ax.text(7.25, 6.9, 'ADME Features (20d)\nMW, LogP, HBD, HBA...',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows to concatenation
    arrow_text = FancyArrowPatch((2.5, 4), (4, 3),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_text)

    arrow_adme = FancyArrowPatch((7.25, 6.5), (6, 3),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_adme)

    # Concatenation
    concat_box = FancyBboxPatch((3.5, 2.3), 3, 0.6, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='yellow', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(5, 2.6, 'Concatenate [Text + ADME]', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # MLP predictor
    arrow_mlp = FancyArrowPatch((5, 2.3), (5, 1.6),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_mlp)

    mlp_box = FancyBboxPatch((3.5, 0.8), 3, 0.8, boxstyle="round,pad=0.1",
                             edgecolor='darkred', facecolor='#FFB3B3', linewidth=2)
    ax.add_patch(mlp_box)
    ax.text(5, 1.2, 'MLP Predictor\n[276 → 128 → 64 → 1]', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Output
    arrow_out = FancyArrowPatch((5, 0.8), (5, 0.2),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_out)

    ax.text(5, -0.2, 'Predicted Value', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

    plt.tight_layout()
    filename = OUTPUT_DIR / 'architecture_foundation_only.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()

# ===================== DIAGRAM 3: HYBRID ARCHITECTURE =====================

def draw_hybrid_architecture():
    """Draw Hybrid (GNN + Foundation) architecture"""

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Title
    ax.text(7, 13.5, 'Hybrid Architecture (GNN + Foundation Model)',
            fontsize=20, fontweight='bold', ha='center')

    # Left branch: GNN
    ax.text(3, 12.5, 'GNN Branch', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Molecular graph input
    graph_input = FancyBboxPatch((1.5, 11.3), 3, 0.7, boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(graph_input)
    ax.text(3, 11.65, 'Molecular Graph', ha='center', va='center', fontsize=10, fontweight='bold')

    # GNN layers (compact)
    gnn_box = FancyBboxPatch((1.5, 9.5), 3, 1.5, boxstyle="round,pad=0.1",
                             edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(gnn_box)
    ax.text(3, 10.5, 'GNN Layers', ha='center', fontweight='bold', fontsize=11)
    ax.text(3, 10.1, 'SAGEConv', ha='center', fontsize=9)
    ax.text(3, 9.8, 'or GINEConv', ha='center', fontsize=9)

    # Graph pooling
    pool_box = FancyBboxPatch((1.5, 8.3), 3, 0.7, boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(pool_box)
    ax.text(3, 8.65, 'Mean + Max Pool\n(128d)', ha='center', va='center', fontsize=9)

    # Right branch: Foundation Model
    ax.text(11, 12.5, 'Foundation Branch', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='#CCE5FF', alpha=0.5))

    # SMILES input
    smiles_input = FancyBboxPatch((9.5, 11.3), 3, 0.7, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(smiles_input)
    ax.text(11, 11.65, 'SMILES String', ha='center', va='center', fontsize=10, fontweight='bold')

    # Foundation model (compact)
    found_box = FancyBboxPatch((9.5, 9.5), 3, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='darkblue', facecolor='#CCE5FF', linewidth=2)
    ax.add_patch(found_box)
    ax.text(11, 10.5, 'Foundation Model', ha='center', fontweight='bold', fontsize=11)
    ax.text(11, 10.1, 'ChemBERTa', ha='center', fontsize=9)
    ax.text(11, 9.8, 'or MolFormer', ha='center', fontsize=9)

    # Text embedding
    text_emb = FancyBboxPatch((9.5, 8.3), 3, 0.7, boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(text_emb)
    ax.text(11, 8.65, 'Text Embedding\n(256d)', ha='center', va='center', fontsize=9)

    # ADME features (middle)
    adme_box = FancyBboxPatch((5.75, 6.5), 2.5, 0.7, boxstyle="round,pad=0.1",
                              edgecolor='orange', facecolor='#FFE6CC', linewidth=2)
    ax.add_patch(adme_box)
    ax.text(7, 6.85, 'ADME (20d)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows to fusion
    arrow_gnn = FancyArrowPatch((3, 8.3), (5.5, 5.5),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5, color='darkgreen')
    ax.add_patch(arrow_gnn)

    arrow_found = FancyArrowPatch((11, 8.3), (8.5, 5.5),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='darkblue')
    ax.add_patch(arrow_found)

    arrow_adme = FancyArrowPatch((7, 6.5), (7, 5.7),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5, color='orange')
    ax.add_patch(arrow_adme)

    # Fusion layer
    fusion_box = FancyBboxPatch((4.5, 4.5), 5, 1, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='yellow', linewidth=3)
    ax.add_patch(fusion_box)
    ax.text(7, 5.3, 'Multimodal Fusion', ha='center', fontweight='bold', fontsize=12)
    ax.text(7, 4.85, '[Graph(128d) + Text(256d) + ADME(20d)] = 404d',
            ha='center', fontsize=9)

    # Deep MLP
    arrow_mlp = FancyArrowPatch((7, 4.5), (7, 3.7),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5, color='black')
    ax.add_patch(arrow_mlp)

    mlp_box = FancyBboxPatch((4.5, 2.2), 5, 1.5, boxstyle="round,pad=0.1",
                             edgecolor='darkred', facecolor='#FFB3B3', linewidth=3)
    ax.add_patch(mlp_box)
    ax.text(7, 3.3, 'Deep MLP Predictor', ha='center', fontweight='bold', fontsize=12)
    ax.text(7, 2.9, '404 → 256 → 128 → 64 → 1', ha='center', fontsize=10)
    ax.text(7, 2.5, '(with BatchNorm + Dropout)', ha='center', fontsize=8, style='italic')

    # Output
    arrow_out = FancyArrowPatch((7, 2.2), (7, 1.4),
                               arrowstyle='->', mutation_scale=25, linewidth=3, color='black')
    ax.add_patch(arrow_out)

    output_box = FancyBboxPatch((5.5, 0.5), 3, 0.8, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='gold', linewidth=3)
    ax.add_patch(output_box)
    ax.text(7, 0.9, 'Predicted ADME Value', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Add advantage note
    ax.text(7, -0.3, 'Best of Both Worlds: Graph Structure + Pretrained Knowledge',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    filename = OUTPUT_DIR / 'architecture_hybrid.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()

# Generate all architecture diagrams
print("\n1. Creating GNN-only architecture diagram...")
draw_gnn_architecture()

print("\n2. Creating Foundation-only architecture diagram...")
draw_foundation_architecture()

print("\n3. Creating Hybrid architecture diagram...")
draw_hybrid_architecture()

print("\n" + "="*80)
print("ARCHITECTURE DIAGRAMS COMPLETE!")
print("="*80)
print(f"\nSaved to: {OUTPUT_DIR}/")
print("\nGenerated diagrams:")
print("  1. architecture_gnn_only.png")
print("  2. architecture_foundation_only.png")
print("  3. architecture_hybrid.png")
print("\nUse these for explaining your model architectures to the professor!")
print("="*80)
