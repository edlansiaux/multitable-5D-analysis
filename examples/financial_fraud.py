import sys
from pathlib import Path
import pandas as pd
import torch

# Ajout de la racine au path
sys.path.append(str(Path(__file__).parent.parent))

from mt5d.datasets.financial import FinancialDataset
from mt5d.core.pipeline.mt5d_pipeline import MT5DPipeline
from mt5d.core.ops.drift import RelationalDriftDetector

def main():
    print("=== EXEMPLE : DÉTECTION DE FRAUDE FINANCIÈRE 5D ===")
    
    # 1. Chargement des données (Stub)
    # Dans un cas réel, pointez vers votre dossier de données
    # dataset = FinancialDataset(root_dir="data/finance")
    # tables, rels = dataset.load()
    
    # Génération synthétique pour la démo
    print("Génération de transactions synthétiques...")
    tables = {
        'accounts': pd.DataFrame({'acc_id': range(100), 'balance': [1000.0]*100}),
        'transactions': pd.DataFrame({
            'tx_id': range(1000), 
            'acc_id': [i%100 for i in range(1000)],
            'amount': torch.randn(1000).abs().numpy() * 100,
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='T')
        })
    }
    rels = [('accounts', 'acc_id', 'transactions', 'acc_id', 'one_to_many')]
    
    # 2. Configuration Spécifique Finance
    # On met l'accent sur la dimension temporelle (Dim 5)
    config_path = "configs/financial_config.yaml"
    pipeline = MT5DPipeline(config_path=config_path) if Path(config_path).exists() else MT5DPipeline()
    
    # 3. Exécution du Pipeline
    print("Lancement de l'analyse RHT...")
    results = pipeline.run(tables, rels, target_task="fraud_detection")
    
    # 4. Monitoring de Dérive (Section 5.8)
    # Simulation: Arrivée de nouvelles données avec un pattern de fraude
    print("\n[Monitoring] Analyse de dérive sur le flux temps réel...")
    drift_detector = RelationalDriftDetector()
    
    # On suppose que 'results' contient les embeddings du modèle entraîné
    # Simulation d'embeddings
    ref_emb = torch.randn(100, 64) 
    drift_detector.set_reference(ref_emb)
    
    # Nouveaux embeddings (simulant une attaque)
    new_emb = torch.randn(100, 64) + 0.5 
    drift_report = drift_detector.detect_drift(new_emb)
    
    print(f"Statut du flux : {drift_report['status'].upper()}")
    print(f"Score de dérive : {drift_report['drift_score']:.4f}")
    
    if drift_report['is_drifting']:
        print("ALERTE : Modification structurelle des transactions détectée !")

if __name__ == "__main__":
    main()
