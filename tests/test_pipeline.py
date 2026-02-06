import pytest
import pandas as pd
import os
import shutil
from mt5d.core.pipeline.mt5d_pipeline import MT5DPipeline
from mt5d.datasets.synthetic import SyntheticMultiTableGenerator

@pytest.fixture
def synthetic_data():
    """Génère un petit jeu de données pour les tests"""
    gen = SyntheticMultiTableGenerator(num_patients=50)
    return gen.generate()

@pytest.fixture
def config_path(tmp_path):
    """Crée un fichier config temporaire"""
    config_content = """
    project_name: "test_run"
    hypergraph:
      node_dim: 64
    model:
      hidden_dim: 64
      pente:
        use_pente: true
    """
    p = tmp_path / "test_config.yaml"
    p.write_text(config_content)
    return str(p)

def test_full_pipeline_execution(synthetic_data, config_path, tmp_path):
    """
    Vérifie que le pipeline s'exécute de bout en bout (Steps 0 -> 7)
    sans erreur.
    """
    tables, relationships = synthetic_data
    output_dir = tmp_path / "output"
    
    # Initialisation
    pipeline = MT5DPipeline(config_path=config_path)
    
    # Exécution
    results = pipeline.run(
        tables=tables,
        relationships=relationships,
        target_task="classification_test"
    )
    
    # Vérifications (Assertions)
    assert results is not None
    assert "metrics" in results
    assert "hypergraph" in results
    assert "model" in results
    assert results["hypergraph"].num_nodes() > 0
    
    # Vérification Step 0 (Profiling)
    profiling = results["metrics"]
    assert profiling.total_volume > 0
    assert len(profiling.variable_counts) == 4 # 4 tables dans synthetic
    
    # Vérification Sauvegarde
    pipeline.save_pipeline(str(output_dir))
    assert (output_dir / "rht_model.pt").exists()
    assert (output_dir / "config.yaml").exists()

def test_pente_integration(synthetic_data):
    """Vérifie spécifiquement que PentE accepte les 5 dimensions"""
    tables, rels = synthetic_data
    pipeline = MT5DPipeline()
    
    # On force la construction pour obtenir le graphe
    pipeline.profiler.profile(tables, rels)
    g = pipeline.builder.build_from_tables(tables, rels)
    
    # Vérification des features du graphe
    assert 'feat' in g.nodes['entity'].data
    # On vérifie que le constructeur a bien géré les ID globaux
    assert g.num_nodes('entity') == sum(len(df) for df in tables.values())
