#!/usr/bin/env python3
"""
Exécute les benchmarks du framework MT5D.
"""
import argparse
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from mt5d import MT5DPipeline
from mt5d.core.profiling import DimensionalProfiler

def generate_benchmark_data(table_sizes):
    """Génère des données de benchmark de différentes tailles"""
    datasets = {}
    
    for size_name, sizes in table_sizes.items():
        n_rows_main = sizes['main']
        n_rows_secondary = sizes['secondary']
        
        # Table principale
        main_table = pd.DataFrame({
            'id': range(n_rows_main),
            'feature1': np.random.randn(n_rows_main),
            'feature2': np.random.randn(n_rows_main),
            'category': np.random.choice([f'cat_{i}' for i in range(100)], n_rows_main)
        })
        
        # Table secondaire (relations many-to-one)
        secondary_table = pd.DataFrame({
            'id': range(n_rows_secondary),
            'main_id': np.random.choice(range(n_rows_main), n_rows_secondary),
            'value': np.random.randn(n_rows_secondary),
            'timestamp': pd.date_range('2020-01-01', periods=n_rows_secondary, freq='H')
        })
        
        datasets[size_name] = {
            'main': main_table,
            'secondary': secondary_table
        }
    
    return datasets

def run_profiling_benchmark(tables):
    """Benchmark du profilage"""
    results = {}
    
    profiler = DimensionalProfiler()
    
    for size_name, tables_dict in tables.items():
        print(f"  Profilage: {size_name}...")
        
        start_time = time.time()
        
        metrics = profiler.profile(tables_dict)
        
        elapsed = time.time() - start_time
        
        results[size_name] = {
            'time_seconds': elapsed,
            'total_rows': metrics.volume['total_rows'],
            'total_columns': metrics.many_variables['total_columns'],
            'rows_per_second': metrics.volume['total_rows'] / elapsed
        }
    
    return results

def run_hypergraph_benchmark(tables):
    """Benchmark de la construction d'hypergraphe"""
    from mt5d.core.hypergraph import RelationalHypergraphBuilder
    
    results = {}
    builder = RelationalHypergraphBuilder({})
    
    for size_name, tables_dict in tables.items():
        print(f"  Hypergraphe: {size_name}...")
        
        # Définir des relations simples
        relationships = [('secondary', 'main_id', 'main', 'id', 'related_to')]
        
        start_time = time.time()
        
        hypergraph = builder.build_from_tables(tables_dict, relationships)
        
        elapsed = time.time() - start_time
        
        results[size_name] = {
            'time_seconds': elapsed,
            'num_nodes': hypergraph.num_nodes(),
            'num_edges': hypergraph.num_edges(),
            'nodes_per_second': hypergraph.num_nodes() / elapsed
        }
    
    return results

def run_pipeline_benchmark(tables):
    """Benchmark du pipeline complet"""
    results = {}
    
    for size_name, tables_dict in tables.items():
        print(f"  Pipeline: {size_name}...")
        
        pipeline = MT5DPipeline()
        
        start_time = time.time()
        
        try:
            results_dict = pipeline.run(tables_dict)
            success = True
        except Exception as e:
            print(f"    Erreur: {e}")
            success = False
        
        elapsed = time.time() - start_time
        
        if success:
            results[size_name] = {
                'time_seconds': elapsed,
                'success': True,
                'steps_completed': len(results_dict)
            }
        else:
            results[size_name] = {
                'time_seconds': elapsed,
                'success': False,
                'steps_completed': 0
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Exécute les benchmarks MT5D")
    parser.add_argument('--output', '-o', default='benchmark_results.json',
                       help='Fichier de sortie pour les résultats')
    parser.add_argument('--sizes', nargs='+', default=['small', 'medium', 'large'],
                       help='Tailles à benchmarker')
    args = parser.parse_args()
    
    # Définir les tailles
    size_definitions = {
        'small': {'main': 1000, 'secondary': 5000},
        'medium': {'main': 10000, 'secondary': 50000},
        'large': {'main': 50000, 'secondary': 250000}
    }
    
    # Filtrer selon les arguments
    sizes_to_run = {k: v for k, v in size_definitions.items() if k in args.sizes}
    
    print("MT5D Benchmark Suite")
    print("=" * 60)
    
    # Générer les données
    print("Génération des données de benchmark...")
    benchmark_data = generate_benchmark_data(sizes_to_run)
    
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmarks': {}
    }
    
    # 1. Benchmark de profilage
    print("\n1. Benchmark de profilage 5D...")
    profiling_results = run_profiling_benchmark(benchmark_data)
    results['benchmarks']['profiling'] = profiling_results
    
    # 2. Benchmark d'hypergraphe
    print("\n2. Benchmark de construction d'hypergraphe...")
    hypergraph_results = run_hypergraph_benchmark(benchmark_data)
    results['benchmarks']['hypergraph'] = hypergraph_results
    
    # 3. Benchmark de pipeline (uniquement petites tailles)
    print("\n3. Benchmark de pipeline complet...")
    # Limiter aux petites tailles pour le pipeline complet
    small_data = {k: v for k, v in benchmark_data.items() if k == 'small'}
    if small_data:
        pipeline_results = run_pipeline_benchmark(small_data)
        results['benchmarks']['pipeline'] = pipeline_results
    
    # Sauvegarder les résultats
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Afficher un résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("=" * 60)
    
    for benchmark_name, benchmark_results in results['benchmarks'].items():
        print(f"\n{benchmark_name.upper()}:")
        for size_name, size_results in benchmark_results.items():
            print(f"  {size_name}:")
            for metric, value in size_results.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.2f}")
                else:
                    print(f"    {metric}: {value}")
    
    print(f"\nRésultats détaillés sauvegardés dans: {output_path}")
    
    # Générer un rapport CSV
    csv_path = output_path.with_suffix('.csv')
    report_data = []
    
    for benchmark_name, benchmark_results in results['benchmarks'].items():
        for size_name, size_results in benchmark_results.items():
            row = {
                'benchmark': benchmark_name,
                'size': size_name
            }
            row.update(size_results)
            report_data.append(row)
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(csv_path, index=False)
        print(f"Rapport CSV généré: {csv_path}")

if __name__ == "__main__":
    main()
