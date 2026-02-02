# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NKREPO is a malware repository system by NKAMG (Nankai Anti-Malware Group) for collecting, storing, and analyzing malware samples with multi-model detection capabilities.

## Tech Stack

**Backend:** Python 3.x, Flask 3.1+, PyMySQL, scikit-learn, XGBoost, PyTorch, torch-geometric
**Frontend:** Vue.js 2.6, Element-UI, Vuex, Vue-Router, ECharts
**Database:** MySQL (256 sharded tables by SHA256 prefix)

## Directory Structure

```
nkrepo/
├── data/samples/          # 5-level SHA256-based sample storage
├── data/trails/           # Malicious domain information
├── db/init_db.py          # Database schema initialization
├── download/              # Dataset downloaders (androzoo, virusshare, maltrail)
├── multi_scan/            # Multi-AV scanning (Kaspersky, VirusTotal)
├── utils/                 # Sample management scripts
├── web/flask/             # Backend API server
│   ├── app.py             # Main Flask application (port 5005)
│   ├── config.ini         # Required configuration file
│   ├── feeds/             # ML classifiers (SVM, KNN, RF, XGBoost, LSTM, etc.)
│   └── models/            # Deep learning models (MalConv, Transformer, EMBER, etc.)
└── web/vue/               # Frontend SPA (port 9528)
    ├── src/views/detect/  # Detection pages (ensemble ML, DGA)
    ├── src/views/file_search/  # Sample query pages
    └── tests/unit/        # Jest unit tests
```

## Commands

### Backend
```bash
cd web/flask
pip install -r requirements.txt
python app.py              # Runs on http://{HOST_IP}:5005
```

### Frontend
```bash
cd web/vue
npm install
npm run dev                # Dev server with hot reload (port 9528)
npm run build:prod         # Production build
npm run test:unit          # Run Jest tests
npm lint                   # Lint Vue/JS files
```

### Database Setup
```bash
cd db
python init_db.py -u {user} -p {password} -h {host}
```

### Sample Management
```bash
cd utils
python init_repo.py                    # Initialize directory structure
python add_sample.py -d /path/to/samples
python count_sample.py
```

## Configuration

**Required:** Create `web/flask/config.ini` before running Flask:
```ini
[ini]
ip = 0.0.0.0
row_per_page = 20

[mysql]
host = localhost
user = root
passwd = password
db_category = nkrepo_category
db_family = nkrepo_family
db_platform = nkrepo_platform
charset = utf8mb4

[API]
vt_key = YOUR_VIRUSTOTAL_API_KEY

[files]
# Model and training data paths
```

**Vue Environment:**
- `.env.development` - Dev API proxy (localhost:5005)
- `.env.production` - Production API endpoint

## Code Conventions

### Python
- UTF-8 encoding header: `# -*- coding: utf-8 -*-`
- snake_case for functions/variables, PascalCase for classes
- ConfigParser for .ini files, PyMySQL with context managers

### Vue.js
- PascalCase for components, kebab-case for filenames
- 2-space indentation, ESLint enforced
- Component structure: `<template>` → `<script>` → `<style scoped>`
- Axios with interceptors for API calls

## Architecture

### Detection Pipeline
- **DGA Detection:** `dga_detection.py` - Multi-model domain classification using classifiers in `feeds/`
- **File Detection:** `ensemble_predict.py` - Ensemble of 10 deep learning models:
  - MalConv, MalConv2, Transformer, EMBER, 1D-CNN, InceptionV3, RCNF, Attention-RCNN, RCNN, VGG16
- **ML Classifiers:** `feeds/` - SVM, KNN, Random Forest, XGBoost, AdaBoost, GBDT, Decision Tree, Logistic Regression, Naive Bayes, LSTM

### Database Schema
- 256 sharded sample tables: `sample_00` through `sample_ff` (by SHA256 prefix)
- Separate databases for category (`db_category`), family (`db_family`), platform (`db_platform`)
- Daily domain tables: `domain_YYYYMMDD`

### API Endpoints (Flask)
- `POST /api/detect` - DGA domain detection
- `POST /upload` - File upload and ensemble prediction
- `POST /query_category|family|platform|sha256` - Sample search
- `GET /detail_*/<sha256>` - Sample details
- `GET /download_*/<sha256>` - Download sample (7z encrypted, password: "infected")
- `GET /detection_API/<sha256>` - VirusTotal detection results
- `GET /behaviour_API/<sha256>` - VirusTotal behavior analysis

## Testing

Frontend tests in `web/vue/tests/unit/` using Jest. Run `npm run test:ci` for lint + tests.
