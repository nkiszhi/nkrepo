# NKREPO - Malware Repository Query & Detection System

NKAMG (Nankai Anti-Malware Group) malware repository system for collecting, storing, and analyzing malware samples with multi-model detection capabilities.

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
- `[mysql]`: host, user, passwd, db settings
- `[API]`: vt_key (VirusTotal API key)
- `[files]`: model paths, training data paths

**Vue Environment:**
- `.env.development` - Dev API proxy
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

## Key Features

- **File Detection:** 10 ensemble ML classifiers + 10 deep learning models
- **DGA Detection:** Multi-model malicious domain classification
- **Sample Search:** Query by SHA256, category, family, or platform
- **VirusTotal Integration:** External scanning via API v3

## Testing

Frontend tests in `web/vue/tests/unit/` using Jest. Run `npm run test:ci` for lint + tests.
