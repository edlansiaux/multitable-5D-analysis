import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import dgl
from typing import Dict, List, Optional

def plot_relational_hypergraph(g: dgl.DGLGraph, 
                               output_path: str = "hypergraph.png", 
                               title: str = "Relational Hypergraph"):
    """
    Visualise une projection 2D de l'hypergraphe relationnel.
    Les nœuds sont colorés par type (Table vs Hyperarête).
    """
    plt.figure(figsize=(12, 8))
    
    # Conversion vers NetworkX pour le tracé
    # DGL -> NetworkX (homogène pour simplification visuelle)
    if g.is_heterogeneous:
        g_homo = dgl.to_homogeneous(g)
        nx_g = dgl.to_networkx(g_homo)
    else:
        nx_g = dgl.to_networkx(g)
        
    pos = nx.spring_layout(nx_g, k=0.15, iterations=20)
    
    # Distinction visuelle (si les attributs sont préservés)
    # Ici, simulation : on suppose les N premiers sont des entités
    num_nodes = nx_g.number_of_nodes()
    node_colors = ['#3498db'] * num_nodes # Bleu par défaut
    
    nx.draw_networkx_nodes(nx_g, pos, node_color=node_colors, node_size=50, alpha=0.8)
    nx.draw_networkx_edges(nx_g, pos, alpha=0.2, edge_color='gray')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Graphique sauvegardé : {output_path}")

def plot_attention_heatmap(attention_weights: np.ndarray, 
                           labels: List[str], 
                           output_path: str = "attention.png"):
    """
    Affiche la matrice d'attention entre les tables ou entités (Explicabilité).
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    
    if len(labels) <= 20: # Affiche les labels seulement si lisible
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        
    plt.title("Sparse Relational Attention Weights")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
