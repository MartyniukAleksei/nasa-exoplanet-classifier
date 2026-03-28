# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monorepo for a NASA exoplanet classification web app. Python/Flask ML backend deploys to Render; vanilla JS frontend deploys to GitHub Pages from the `docs/` folder.

- **Live backend**: `https://nasa-exoplanet-api.onrender.com`
- **Live frontend**: `https://martyniukaleksei.github.io/nasa-exoplanet-classifier`

## Common Commands

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py                    # Run Flask dev server on port 5000
gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:5000  # Production-style
```

### Frontend
```bash
cd docs
npm install
npm run dev        # Vite dev server
npm run build      # Production build
npm run preview    # Preview production build
```

There are no tests or linting configured in this project.

## Architecture

### Backend (`backend/`)

Flask API with four endpoints: `/predict` (POST, single prediction), `/predict/batch` (POST), `/analyze` (alias for predict with CORS preflight), `/health` (GET).

**ML pipeline flow**: `nasa_data_loader.py` fetches from NASA Exoplanet Archive TAP API → `data_preprocessing.py` (outlier removal, imputation, RobustScaler) → `feature_engineering.py` (physics-based features: transit ratios, SNR features, stellar characteristics) → `ensemble_models.py` trains/predicts.

**Ensemble**: 7 classifiers (Random Forest, Gradient Boosting, XGBoost, LightGBM, Neural Network, SVM, Logistic Regression) combined via voting. Separate Gradient Boosting regressors predict planet properties (radius, temperature, semi-major axis, impact parameter).

**Pre-trained models** are committed as `.pkl` files in `backend/models/` and loaded at startup by `app.py`. The main orchestrator is `exoplanet_classifier.py` which ties together all pipeline stages.

**Output**: 3-class classification (confirmed exoplanet, planetary candidate, false positive) with confidence scores and predicted physical properties.

### Frontend (`docs/`)

Vanilla JS, no framework. Two HTML pages:
- `index.html` — input form for 7 transit/stellar parameters
- `exoplanet.html` — Canvas-based 3D planet visualization (loaded via iframe in results)

JS is split into `form-submission.js` (validation, sample data) and `result-fetching.js` (API polling, result display, fallback mock data when API is unavailable).

### CORS

Backend allows origins: GitHub Pages URL and localhost variants. Configured in `app.py` via Flask-CORS.

### Deployment

- Backend: Render via `Procfile` (gunicorn, single worker, 120s timeout)
- Frontend: GitHub Pages from `/docs` directory
- No CI/CD pipelines
