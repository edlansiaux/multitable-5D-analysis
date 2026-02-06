import argparse
import os
from pathlib import Path
import sys

# Ajout de la racine au path
sys.path.append(str(Path(__file__).parent.parent))

from mt5d.datasets.synthetic import SyntheticMultiTableGenerator

def main():
    parser = argparse.ArgumentParser(description="Téléchargement ou génération de données exemples")
    parser.add_argument("--dataset", type=str, default="medical", choices=["medical", "financial", "amazon"],
                        help="Type de dataset à générer")
    parser.add_argument("--output_dir", type=str, default="data/example",
                        help="Dossier de destination")
    parser.add_argument("--size", type=int, default=1000,
                        help="Taille de l'échantillon (nombre d'entités principales)")
    
    args = parser.parse_args()
    output_path = Path(args.output_dir) / args.dataset
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Préparation du dataset '{args.dataset}' ({args.size} échantillons)...")
    
    if args.dataset == "medical":
        print("Génération de données synthétiques MIMIC-IV...")
        gen = SyntheticMultiTableGenerator(num_patients=args.size)
        tables, rels = gen.generate()
        
        # Sauvegarde en CSV
        for name, df in tables.items():
            file_path = output_path / f"{name}.csv"
            df.to_csv(file_path, index=False)
            print(f"  - Sauvegardé : {file_path}")
            
    elif args.dataset == "financial":
        print("Génération de données financières (Stub)...")
        # Ici on pourrait appeler un générateur spécifique
        print("  - [TODO] Implémenter le générateur financier complet.")
        
    elif args.dataset == "amazon":
        print("Téléchargement des données Amazon (Stub)...")
        print("  - Veuillez télécharger les données depuis http://jmcauley.ucsd.edu/data/amazon/")
        
    print(f"\nTerminé. Données disponibles dans : {output_path}")

if __name__ == "__main__":
    main()
