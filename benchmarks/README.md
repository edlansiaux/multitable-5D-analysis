# Benchmarks MT5D

Ce répertoire contient les scripts et résultats des benchmarks.

## Structure
```bash
benchmarks/
├── README.md # Ce fichier
├── runner.py # Script principal de benchmark
├── configs/ # Configurations de benchmark
├── results/ # Résultats stockés
├── plots/ # Visualisations
└── datasets/ # Données de benchmark
```

## Exécution des Benchmarks

### Benchmarks Rapides
```bash
python benchmarks/runner.py --sizes small medium
```
### Benchmarks Complets
```bash
python benchmarks/runner.py --all --output results/full_benchmark.json
```
### Visualization
```bash
python benchmarks/plot_results.py results/full_benchmark.json
```

## Métriques Mesurées
### Performance
- Temps d'exécution par étape
- Utilisation mémoire
- Scalabilité avec la taille des données

### Qualité
- Précision des relations découvertes
- Qualité des embeddings
- Performance des prédictions

### Robustesse
- Stabilité avec données bruitées
- Performance avec données incomplètes
- Généralisation à nouveaux schémas

## Datasets de Benchmark
### Synthétiques
- small: 1K-10K lignes
- medium: 10K-100K lignes
- large: 100K-1M lignes
- xlarge: 1M+ lignes

### Réels (à venir)
- MIMIC-IV (santé)
- Amazon Reviews (e-commerce)
- Financial Transactions (finance)

Contribuer aux Benchmarks
1. Ajoutez votre benchmark dans benchmarks/
2. Incluez un script d'exécution
3. Documentez les métriques
4. Ajoutez des visualisations
5. Soumettez une Pull Request
