"""
Example: Running the RHT pipeline on a MIMIC-IV-like dataset.

This creates synthetic data that mimics the structure of MIMIC-IV
(Section 7.1.1) and demonstrates the full 8-step pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from rht.model import RHTConfig
from pipeline import MultiTablePipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def create_synthetic_mimic(
    n_patients: int = 200,
    n_admissions: int = 500,
    n_diagnoses: int = 1500,
    n_prescriptions: int = 2000,
    n_lab_results: int = 5000,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str, str, str]]]:
    """Generate synthetic tables mimicking the MIMIC-IV schema."""
    rng = np.random.default_rng(seed)

    patients = pd.DataFrame(
        {
            "patient_id": range(1, n_patients + 1),
            "age": rng.integers(18, 95, n_patients),
            "gender": rng.choice([0, 1], n_patients),
            "weight": rng.normal(75, 15, n_patients).round(1),
        }
    )

    admissions = pd.DataFrame(
        {
            "hadm_id": range(1, n_admissions + 1),
            "patient_id": rng.integers(1, n_patients + 1, n_admissions),
            "los_days": rng.exponential(5, n_admissions).round(1),
            "mortality": rng.choice([0, 1], n_admissions, p=[0.9, 0.1]),
            "admit_time": pd.date_range(
                "2020-01-01", periods=n_admissions, freq="4h"
            ),
        }
    )

    diagnoses = pd.DataFrame(
        {
            "hadm_id": rng.integers(1, n_admissions + 1, n_diagnoses),
            "icd_code": rng.integers(1, 10_000, n_diagnoses),  # high cardinality
            "seq_num": rng.integers(1, 10, n_diagnoses),
        }
    )

    prescriptions = pd.DataFrame(
        {
            "hadm_id": rng.integers(1, n_admissions + 1, n_prescriptions),
            "drug_code": rng.integers(1, 5000, n_prescriptions),
            "dose_mg": rng.exponential(50, n_prescriptions).round(1),
        }
    )

    lab_results = pd.DataFrame(
        {
            "hadm_id": rng.integers(1, n_admissions + 1, n_lab_results),
            "lab_item_id": rng.integers(1, 500, n_lab_results),
            "value": rng.normal(100, 30, n_lab_results).round(2),
            "charttime": pd.date_range(
                "2020-01-01", periods=n_lab_results, freq="30min"
            ),
        }
    )

    tables = {
        "patients": patients,
        "admissions": admissions,
        "diagnoses": diagnoses,
        "prescriptions": prescriptions,
        "lab_results": lab_results,
    }

    relations = [
        ("patients", "patient_id", "admissions", "patient_id"),
        ("admissions", "hadm_id", "diagnoses", "hadm_id"),
        ("admissions", "hadm_id", "prescriptions", "hadm_id"),
        ("admissions", "hadm_id", "lab_results", "hadm_id"),
    ]

    return tables, relations


def main():
    print("=" * 60)
    print("  MIMIC-IV Synthetic Example — RHT Pipeline")
    print("=" * 60)

    # 1. Create synthetic data
    tables, relations = create_synthetic_mimic(n_patients=50, n_admissions=100)

    for name, df in tables.items():
        print(f"  {name:20s}  {df.shape[0]:>6d} rows x {df.shape[1]} cols")

    # 2. Configure a small model for demonstration
    config = RHTConfig(
        semantic_dim=32,
        relational_dim=16,
        temporal_dim=16,
        categorical_dim=16,
        volume_dim=8,
        num_attention_heads=4,
        num_message_passing_layers=2,
    )

    # 3. Run the full 8-step pipeline
    pipeline = MultiTablePipeline(tables, relations, config=config)
    pipeline.run(contrastive_epochs=10)

    # 4. Inspect outputs
    embeddings = pipeline.get_pente_embeddings()
    print(f"\nPentE embeddings shape: {embeddings.shape}")
    print(
        f"Embedding dim = {config.pente_dim}  "
        f"(sem={config.semantic_dim} + rel={config.relational_dim} "
        f"+ temp={config.temporal_dim} + cat={config.categorical_dim} "
        f"+ vol={config.volume_dim})"
    )

    preds = pipeline.predict()
    print(f"Prediction output shape: {preds.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
