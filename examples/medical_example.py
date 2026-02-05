import pandas as pd
import numpy as np
from mt5d import MT5DPipeline

# Charger les données (exemple simplifié)
def load_medical_data():
    # Patients
    patients = pd.DataFrame({
        'patient_id': range(1000),
        'age': np.random.randint(18, 90, 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'admission_date': pd.date_range('2020-01-01', periods=1000, freq='D')
    })
    
    # Diagnostics
    diagnoses = pd.DataFrame({
        'diagnosis_id': range(5000),
        'patient_id': np.random.choice(patients['patient_id'], 5000),
        'code': np.random.choice([f'I{i}' for i in range(50)] + 
                                [f'K{i}' for i in range(30)], 5000),
        'date': pd.date_range('2020-01-01', periods=5000, freq='H')
    })
    
    # Mesures de laboratoire
    labs = pd.DataFrame({
        'lab_id': range(20000),
        'patient_id': np.random.choice(patients['patient_id'], 20000),
        'parameter': np.random.choice(['K+', 'Na+', 'Cl-', 'Glucose', 'Creatinine'], 20000),
        'value': np.random.normal(0, 1, 20000),
        'timestamp': pd.date_range('2020-01-01', periods=20000, freq='T')
    })
    
    return {
        'patients': patients,
        'diagnoses': diagnoses,
        'labs': labs
    }

# Définir les relations
relationships = [
    ('diagnoses', 'patient_id', 'patients', 'patient_id', 'has_diagnosis'),
    ('labs', 'patient_id', 'patients', 'patient_id', 'has_lab')
]

# Initialiser et exécuter le pipeline
if __name__ == "__main__":
    # Charger les données
    tables = load_medical_data()
    
    # Créer le pipeline
    pipeline = MT5DPipeline(config_path="configs/medical_config.yaml")
    
    # Exécuter
    results = pipeline.run(
        tables=tables,
        relationships=relationships,
        target_task="patient_readmission_prediction"
    )
    
    # Afficher les insights
    print("\n=== Insights Générés ===")
    for insight in results['results'].get('insights', []):
        print(f"- {insight}")
    
    # Sauvegarder
    pipeline.save_pipeline("outputs/medical_analysis")
