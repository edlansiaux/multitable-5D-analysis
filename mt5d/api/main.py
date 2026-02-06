from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import io
import uvicorn

from mt5d.core.pipeline.mt5d_pipeline import MT5DPipeline
from mt5d.datasets.synthetic import SyntheticMultiTableGenerator

app = FastAPI(
    title="MT5D Analysis API",
    description="API pour l'analyse multi-tables 5D (RHT Framework)",
    version="1.0.0"
)

# Instance globale du pipeline
pipeline = MT5DPipeline()

class TrainingConfig(BaseModel):
    task_name: str
    epochs: int = 10
    target_column: Optional[str] = None

@app.get("/")
def root():
    return {"message": "MT5D API is running. See /docs for usage."}

@app.post("/generate_demo_data")
def generate_demo():
    """Génère des données synthétiques et les charge en mémoire pour tester."""
    gen = SyntheticMultiTableGenerator(num_patients=100)
    tables, rels = gen.generate()
    # Stockage temporaire (en prod, utiliser une DB ou stockage objet)
    app.state.tables = tables
    app.state.rels = rels
    return {"message": "Données synthétiques générées", "tables": list(tables.keys())}

@app.post("/profile")
def run_profiling():
    """Lance l'étape 0 : Profilage Dimensionnel"""
    if not hasattr(app.state, 'tables'):
        return {"error": "Aucune donnée chargée. Utilisez /generate_demo_data d'abord."}
    
    metrics = pipeline.profiler.profile(app.state.tables, app.state.rels)
    return {"metrics": metrics}

@app.post("/train")
async def train_model(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Lance l'entraînement du pipeline RHT en arrière-plan"""
    if not hasattr(app.state, 'tables'):
        return {"error": "No data available"}
    
    def _train():
        print(f"Démarrage entraînement tâche: {config.task_name}")
        pipeline.run(app.state.tables, app.state.rels, target_task=config.task_name)
        
    background_tasks.add_task(_train)
    return {"status": "Training started in background", "config": config}

@app.get("/predict")
def predict(entity_id: str):
    """Inférence pour une entité donnée (simulation)"""
    if pipeline.model is None:
        return {"error": "Model not trained yet"}
    # Logique d'inférence...
    return {"entity_id": entity_id, "prediction": 0.85, "confidence": "high"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
