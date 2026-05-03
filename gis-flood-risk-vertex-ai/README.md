# GIS Flood Risk Prediction Pipeline — Vertex AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI-orange.svg)](https://cloud.google.com/vertex-ai)
[![KFP](https://img.shields.io/badge/KFP-1.8.22-green.svg)](https://kubeflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Author:** Venkataramanaiah Poliboyina  
> **Role:** Survey Manager — L&T MAHSR Bullet Train Project  
> **Stack:** GEE | QGIS | PostGIS | Python | Vertex AI  
> **CRS:** EPSG:32643 (UTM Zone 43N — Maharashtra, India)

---

## Overview

End-to-end **MLOps pipeline** for flood risk prediction using geospatial features derived from Sentinel-2 satellite imagery and terrain analysis. Built on **Google Cloud Vertex AI Pipelines** with 3 components following crisis mapping standards.

This project demonstrates combining **25 years of survey and spatial data expertise** with modern cloud ML for real-world geospatial crisis response applications.

---

## Pipeline Architecture

```
create_gis_dataset
      |
      v
train_flood_model
      |
      v
evaluate_flood_model
```

### Component 1 — Create GIS Dataset
Generates Maharashtra district-level flood risk features:

| Feature | Source | Description |
|---|---|---|
| NDWI | Sentinel-2 B3/B8 | Water presence index |
| NDVI | Sentinel-2 B8/B4 | Vegetation density |
| NDBI | Sentinel-2 B11/B8 | Built-up area index |
| elevation_m | DEM (EPSG:32643) | Terrain elevation in metres |
| slope_deg | DEM derived | Slope in degrees |
| dist_river_m | OSM rivers | Distance to nearest river |
| rainfall_mm | IMD data | Annual precipitation |
| pop_density | Census | Population per sq km |

**Flood Label Thresholds (Field Standard):**
```
NDWI > 0.3        Water detected
Elevation < 50m   Low-lying flood prone zone
Dist River < 2km  High proximity risk
Rainfall > 1500mm High precipitation zone
```

### Component 2 — Train Flood Risk Model
- **Algorithm:** Random Forest Classifier
- **class_weight='balanced':** handles imbalanced flood data
- **n_estimators=150, max_depth=8**
- Outputs feature importance for GIS interpretation

### Component 3 — Evaluate with Crisis Metrics

| Metric | Threshold | Reason |
|---|---|---|
| **Recall** | **> 0.90** | Missing real flood = lives lost |
| **Kappa** | **> 0.75** | Required for official reports |
| Precision | flexible | False alarms acceptable |
| F1 Score | > 0.85 | Operational balance |

> In crisis systems, **Recall is the most critical metric**.  
> A false alarm causes unnecessary evacuation.  
> A missed flood event causes loss of life.

---

## Quick Start

### Prerequisites
```bash
pip install kfp==1.8.22 protobuf==3.20.3 google-cloud-aiplatform
```

### Configure
Edit `src/pipeline.py`:
```python
PROJECT_ID = "your-gcp-project-id"
BUCKET     = "gs://your-bucket-name"
LOCATION   = "us-central1"
```

### Run
```bash
# Set environment
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Execute pipeline
python src/pipeline.py
```

### Monitor
```
GCP Console → Vertex AI → Pipelines → gis-flood-risk-pipeline
```

---

## GIS Context

This pipeline reflects real-world geospatial crisis mapping workflows:

**Flood Detection Workflow:**
1. Sentinel-2 imagery acquisition
2. NDWI computation: `(B3 - B8) / (B3 + B8)`
3. Threshold at NDWI > 0.3 for water pixels
4. DEM-based elevation filtering (< 50m)
5. River proximity analysis (< 2km buffer)
6. ML-based risk classification
7. Validation with Recall > 90% for crisis deployment

**Tools Used in Practice:**
- Google Earth Engine (GEE) — imagery processing
- QGIS — spatial analysis and visualization
- PostGIS — spatial database queries
- Vertex AI — ML pipeline deployment
- Python — GeoPandas, Rasterio, Shapely

---

## Project Portfolio

| Project | Description | Status |
|---|---|---|
| [GEE NDVI Pune](https://github.com/Venkataramanaiah-P) | NDVI time series analysis | Live |
| [Hospital Accessibility](https://github.com/Venkataramanaiah-P) | PyQGIS Maharashtra analysis | Live |
| [PostGIS Urban Risk](https://github.com/Venkataramanaiah-P) | Urban flood risk ETL pipeline | Live |
| **GIS Flood Risk Vertex AI** | **This repository** | **Live** |
| [Cloud Vision API](https://github.com/Venkataramanaiah-P) | Land cover classification | Live |

---

## Repository Structure

```
gis-flood-risk-vertex-ai/
├── src/
│   └── pipeline.py          # Main Vertex AI pipeline
├── notebooks/
│   └── exploration.ipynb    # Data exploration notebook
├── data/
│   └── sample_features.csv  # Sample GIS features
├── docs/
│   └── crisis_metrics.md    # Crisis evaluation standards
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Skills Demonstrated

```
Google Cloud Platform    Vertex AI Pipelines, GCS, KFP SDK
Remote Sensing           Sentinel-2, NDWI, NDVI, NDBI, SAR
GIS Analysis             EPSG:32643, Buffer, Spatial Join
Machine Learning         Random Forest, Imbalanced Data
Crisis Mapping           Recall > 90%, Kappa > 0.75 standards
MLOps                    CI/CD, Pipeline automation, Monitoring
Python                   Pandas, Scikit-learn, GeoPandas
```

---

## Contact

**Venkataramanaiah Poliboyina**  
Survey Manager — L&T MAHSR Bullet Train Project  
GIS & Remote Sensing Specialist

- LinkedIn: [linkedin.com/in/venkataramanaiahpoliboyina](https://linkedin.com/in/venkataramanaiahpoliboyina)
- GitHub: [github.com/Venkataramanaiah-P](https://github.com/Venkataramanaiah-P)
- Open to: Remote GIS Analyst | Crisis Mapping | GEE | PostGIS contracts

---

*Built with 25 years of survey expertise + modern cloud ML*
