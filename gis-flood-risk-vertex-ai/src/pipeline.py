# =============================================================
# GIS Flood Risk Prediction Pipeline — Vertex AI
# Author: Venkataramanaiah Poliboyina
# Project: L&T MAHSR Bullet Train | GIS Freelancing Portfolio
# GitHub: github.com/Venkataramanaiah-P
# =============================================================
# Features: NDWI, NDVI, NDBI, Elevation, Slope,
#           Distance to River, Rainfall, Population Density
# Model:    Random Forest Classifier
# Platform: Google Cloud Vertex AI Pipelines (KFP v1.8.22)
# CRS:      EPSG:32643 (UTM Zone 43N — Maharashtra/India)
# =============================================================

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from kfp.v2 import compiler
from kfp.v2.dsl import (
    component, pipeline,
    Output, Input,
    Dataset, Model, Metrics
)
from google.cloud import aiplatform

# ── Configuration ─────────────────────────────────────────────
PROJECT_ID    = "your-project-id"       # Replace with your GCP project
LOCATION      = "us-central1"
BUCKET        = "gs://your-bucket"     # Replace with your GCS bucket
PIPELINE_NAME = "gis-flood-risk-pipeline"
PIPELINE_FILE = "gis_flood_pipeline.json"


# ── Component 1: Create GIS Dataset ───────────────────────────
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9"
)
def create_gis_dataset(dataset: Output[Dataset]):
    """
    Creates a synthetic GIS dataset simulating Maharashtra
    district-level flood risk features derived from:
    - Sentinel-2 spectral indices (NDWI, NDVI, NDBI)
    - Terrain analysis (elevation, slope)
    - Hydrological features (distance to river)
    - Climatological data (rainfall)
    - Demographic data (population density)

    Flood label logic (field-standard thresholds):
    - NDWI > 0.3  : water presence detected
    - Elevation < 50m : low-lying flood prone area
    - Distance to river < 2000m : proximity risk
    - Rainfall > 1500mm : high precipitation zone
    """
    import pandas as pd
    import numpy as np
    import os

    np.random.seed(42)
    n = 500  # districts/sample points

    df = pd.DataFrame({
        # Remote Sensing Indices (Sentinel-2)
        "NDWI":         np.random.uniform(-0.3, 0.8, n),   # Water Index
        "NDVI":         np.random.uniform(0.1,  0.7, n),   # Vegetation
        "NDBI":         np.random.uniform(-0.2, 0.5, n),   # Built-up

        # Terrain Features (from DEM — EPSG:32643)
        "elevation_m":  np.random.uniform(5,   800, n),    # metres
        "slope_deg":    np.random.uniform(0,    35, n),    # degrees

        # Hydrological Features
        "dist_river_m": np.random.uniform(50, 10000, n),   # metres

        # Climatological
        "rainfall_mm":  np.random.uniform(200, 3000, n),   # mm/year

        # Demographic
        "pop_density":  np.random.uniform(100, 5000, n),   # per sq km
    })

    # Flood risk label using field-standard GIS thresholds
    df["flood_risk"] = (
        (df["NDWI"]         >  0.3) &
        (df["elevation_m"]  < 50.0) &
        (df["dist_river_m"] < 2000) &
        (df["rainfall_mm"]  > 1500)
    ).astype(int)

    os.makedirs(os.path.dirname(dataset.path), exist_ok=True)
    df.to_csv(dataset.path, index=False)

    total      = len(df)
    flood_cnt  = df["flood_risk"].sum()
    print("GIS Dataset created successfully")
    print("Total samples    : " + str(total))
    print("High risk areas  : " + str(flood_cnt) +
          " (" + str(round(flood_cnt/total*100, 1)) + "%)")
    print("CRS standard     : EPSG:32643 (UTM Zone 43N)")
    print("Features         : NDWI, NDVI, NDBI, elevation, " +
          "slope, dist_river, rainfall, pop_density")


# ── Component 2: Train Flood Risk Model ───────────────────────
@component(
    packages_to_install=["pandas", "scikit-learn", "joblib"],
    base_image="python:3.9"
)
def train_flood_model(
    dataset: Input[Dataset],
    model:   Output[Model]
):
    """
    Trains a Random Forest Classifier for flood risk prediction.

    Design choices:
    - class_weight='balanced': handles imbalanced flood data
    - n_estimators=150: sufficient for tabular GIS features
    - max_depth=8: prevents overfitting on synthetic data

    Feature importance reveals which GIS indicator
    contributes most to flood prediction — critical for
    crisis response prioritization.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    import os

    df = pd.read_csv(dataset.path)
    features = [
        "NDWI", "NDVI", "NDBI",
        "elevation_m", "slope_deg",
        "dist_river_m", "rainfall_mm",
        "pop_density"
    ]

    X = df[features]
    y = df["flood_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model.path), exist_ok=True)
    joblib.dump(clf, model.path)

    # Feature importance — GIS interpretation
    importances = sorted(
        zip(features, clf.feature_importances_),
        key=lambda x: -x[1]
    )
    print("Flood Risk Model trained successfully")
    print("Training samples : " + str(len(X_train)))
    print("Test samples     : " + str(len(X_test)))
    print("")
    print("Feature Importances (GIS significance):")
    for feat, imp in importances:
        bar = "#" * int(imp * 40)
        print("  " + feat.ljust(15) + " : " +
              str(round(imp, 3)) + " " + bar)


# ── Component 3: Evaluate with Crisis Metrics ─────────────────
@component(
    packages_to_install=["pandas", "scikit-learn", "joblib"],
    base_image="python:3.9"
)
def evaluate_flood_model(
    dataset:   Input[Dataset],
    model:     Input[Model],
    threshold: float,
    metrics:   Output[Metrics]
):
    """
    Evaluates flood risk model using crisis-specific metrics.

    Crisis Decision Rules:
    - Recall    > 0.90 : MANDATORY for evacuation decisions
                         Missing a real flood = lives lost
    - Kappa     > 0.75 : MANDATORY for official/legal reports
                         Ensures agreement beyond chance
    - Precision        : Acceptable to be lower (false alarms
                         are costly but not catastrophic)
    - F1 Score  > 0.85 : Balance for operational deployment

    Reference: Meridial Geospatial Crisis Response Standards
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, recall_score,
        precision_score, f1_score,
        cohen_kappa_score, confusion_matrix
    )
    import joblib

    df = pd.read_csv(dataset.path)
    features = [
        "NDWI", "NDVI", "NDBI",
        "elevation_m", "slope_deg",
        "dist_river_m", "rainfall_mm",
        "pop_density"
    ]

    X = df[features]
    y = df["flood_risk"]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf    = joblib.load(model.path)
    y_pred = clf.predict(X_test)

    acc   = accuracy_score(y_test,  y_pred)
    rec   = recall_score(y_test,    y_pred, zero_division=0)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test,        y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm    = confusion_matrix(y_test, y_pred)

    # Log to Vertex AI Experiments
    metrics.log_metric("accuracy",  round(acc,   3))
    metrics.log_metric("recall",    round(rec,   3))
    metrics.log_metric("precision", round(prec,  3))
    metrics.log_metric("f1_score",  round(f1,    3))
    metrics.log_metric("kappa",     round(kappa, 3))

    # Crisis evaluation report
    print("=" * 52)
    print("  FLOOD RISK MODEL — CRISIS EVALUATION REPORT")
    print("=" * 52)
    print("  Accuracy  : " + str(round(acc,  3)))
    print("  Recall    : " + str(round(rec,  3)) +
          ("  [PASS - Crisis Ready]" if rec  >= 0.90 else
           "  [FAIL - Below 0.90 threshold]"))
    print("  Precision : " + str(round(prec, 3)))
    print("  F1 Score  : " + str(round(f1,   3)))
    print("  Kappa     : " + str(round(kappa,3)) +
          ("  [PASS - Official reports OK]" if kappa >= 0.75 else
           "  [WARN - Below 0.75 threshold]"))
    print("")
    print("  Confusion Matrix:")
    print("  TN=" + str(cm[0][0]) +
          " FP=" + str(cm[0][1]) +
          " FN=" + str(cm[1][0]) +
          " TP=" + str(cm[1][1]))
    print("=" * 52)

    if rec >= 0.90 and kappa >= 0.75:
        print("  STATUS: CRISIS DEPLOYMENT APPROVED")
        print("  Safe for evacuation and official decisions")
    elif rec >= 0.90:
        print("  STATUS: PARTIAL APPROVAL")
        print("  Recall OK but Kappa needs improvement")
    else:
        print("  STATUS: NOT APPROVED FOR CRISIS USE")
        print("  Recall below 0.90 — retrain before deployment")
    print("=" * 52)


# ── Pipeline Definition ────────────────────────────────────────
@pipeline(
    name=PIPELINE_NAME,
    description="GIS Flood Risk Prediction Pipeline for Maharashtra — Vertex AI",
    pipeline_root=f"{BUCKET}/pipeline_root"
)
def gis_flood_pipeline(threshold: float = 0.85):
    """
    3-component Vertex AI Pipeline:
    create_gis_dataset → train_flood_model → evaluate_flood_model

    Demonstrates end-to-end MLOps for geospatial crisis mapping.
    """
    dataset_task = create_gis_dataset()

    train_task = train_flood_model(
        dataset=dataset_task.outputs["dataset"]
    ).after(dataset_task)

    evaluate_flood_model(
        dataset=dataset_task.outputs["dataset"],
        model=train_task.outputs["model"],
        threshold=threshold
    ).after(train_task)


# ── Compile & Submit ───────────────────────────────────────────
if __name__ == "__main__":
    # Compile pipeline to JSON
    compiler.Compiler().compile(
        pipeline_func=gis_flood_pipeline,
        package_path=PIPELINE_FILE
    )
    print("Pipeline compiled: " + PIPELINE_FILE)

    # Initialize Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=BUCKET
    )

    # Submit to Vertex AI Pipelines
    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=PIPELINE_FILE,
        pipeline_root=BUCKET + "/pipeline_root",
        parameter_values={"threshold": 0.85},
        enable_caching=False
    )
    job.submit()
    print("Pipeline submitted to Vertex AI!")
    print("Check: GCP Console -> Vertex AI -> Pipelines")
