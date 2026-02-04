# NKREPO - Malware Analysis Repository

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1+-green.svg)](https://flask.palletsprojects.com/)
[![Vue.js](https://img.shields.io/badge/Vue.js-2.6-brightgreen.svg)](https://vuejs.org/)
[![License](https://img.shields.io/badge/License-GPL-yellow.svg)](LICENSE)

**NKAMG - Nankai Anti-Malware Group**

## Overview

NKREPO is a comprehensive malware repository and analysis system that provides:

- **Malware Sample Storage** - Organized repository with SHA256-based 5-level directory structure
- **Multi-Model Detection** - 10+ ML/DL models for PE file malware detection
- **DGA Detection** - Domain Generation Algorithm detection using ensemble ML classifiers
- **VirusTotal Integration** - Automated scanning and result retrieval
- **Web Interface** - Vue.js frontend with Flask REST API backend
- **Sample Search** - Search by hash, category, family, or platform

## Project Structure

```
nkrepo/
├── data/                    # Sample storage
│   ├── samples/             # 5-level SHA256-based sample repository
│   └── zips/                # Encrypted ZIP downloads (password: infected)
│
├── db/                      # Database module
│   ├── __init__.py
│   ├── db_operations.py     # Database CRUD operations
│   ├── init_db.py           # Database initialization script
│   └── README.md
│
├── multi_scan/              # Detection module
│   ├── __init__.py
│   ├── dga_detection.py     # DGA detection with ensemble ML
│   ├── ensemble_predict.py  # Multi-model PE detection
│   ├── file_detect.py       # File detection API
│   ├── data/                # Training/test data
│   │   └── features/        # Feature CSV files
│   ├── feeds/               # DGA ML classifiers
│   │   ├── knn.py, svm.py, randomforest.py, ...
│   └── models/              # Deep learning models
│       ├── m_2017_malconv/
│       ├── m_2017_transformer/
│       ├── m_2018_ember/
│       ├── m_2019_1d_cnn/
│       ├── m_2020_inceptionv3/
│       ├── m_2021_malconv2/
│       ├── m_2021_rcnf/
│       ├── m_attention_rcnn/
│       ├── m_rcnn/
│       └── m_vgg16/
│
├── web/                     # Web application
│   ├── flask/               # Backend API
│   │   ├── app.py           # Flask application
│   │   ├── config.py        # Configuration module
│   │   ├── api_vt.py        # VirusTotal API client
│   │   └── config.ini       # User configuration
│   └── vue/                 # Frontend
│       ├── src/
│       └── uploads/         # File upload directory
│
├── utils/                   # Utility scripts
│   ├── packer_detector.py   # PE packer detection
│   ├── pe_hash_calculator.py # Multi-hash calculator
│   └── ...
│
├── yara_rules/              # YARA rules for detection
└── docs/                    # Documentation
```

## Features

### 1. Multi-Model Malware Detection

Ensemble prediction using 10 deep learning models:

| Model | Year | Description |
|-------|------|-------------|
| MalConv | 2017 | Raw bytes CNN |
| Transformer | 2017 | Attention-based model |
| EMBER | 2018 | Feature-based classifier |
| 1D-CNN | 2019 | 1D Convolutional network |
| InceptionV3 | 2020 | Image-based detection |
| MalConv2 | 2021 | Improved MalConv |
| RCNF | 2021 | Capsule network |
| Attention-RCNN | - | Attention + RNN |
| RCNN | - | Recurrent CNN |
| VGG16 | - | VGG-based classifier |

### 2. DGA Detection

Multi-classifier ensemble for domain analysis:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- Naive Bayes
- Logistic Regression
- AdaBoost
- Gradient Boosting (GBDT)
- XGBoost

### 3. Database Sharding

Samples distributed across 256 tables based on SHA256 prefix:
- Hash `12abc...` → `sample_12` table
- Hash `ff123...` → `sample_ff` table

### 4. Sample Storage

5-level directory structure for efficient storage:
```
data/samples/a/b/c/d/e/abcdef1234567890...
```

## Installation

### Prerequisites

- Python 3.8+
- MySQL 5.7+
- Node.js 14+ (for frontend)
- 7-Zip (for encrypted downloads)

### Backend Setup

```bash
# Clone repository
git clone https://github.com/nkamg/nkrepo.git
cd nkrepo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize database
python db/init_db.py -u root -p <password> -h localhost

# Configure
cp web/flask/config.ini.example web/flask/config.ini
# Edit config.ini with your MySQL credentials and VirusTotal API key

# Run Flask server
cd web/flask
python app.py
```

### Frontend Setup

```bash
cd web/vue

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Configuration

Edit `web/flask/config.ini`:

```ini
[ini]
ip = 127.0.0.1
port = 5005

[mysql]
host = localhost
port = 3306
user = root
passwd = your_password
db = nkrepo

[API]
vt_key = YOUR_VIRUSTOTAL_API_KEY

[security]
secret_key = change_this_to_random_string
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | POST | DGA domain detection |
| `/upload` | POST | File upload and analysis |
| `/detection_API/<sha256>` | GET | Get VirusTotal detection results |
| `/behaviour_API/<sha256>` | GET | Get VirusTotal behaviour analysis |
| `/query_category` | POST | Search by malware category |
| `/query_family` | POST | Search by malware family |
| `/query_platform` | POST | Search by platform |
| `/query_sha256` | POST | Search by SHA256 hash |
| `/download_*/<sha256>` | GET | Download sample (encrypted ZIP) |

## Usage

### Python API

```python
# File detection
from multi_scan import run_ensemble_prediction
result = run_ensemble_prediction("/path/to/sample.exe")

# DGA detection
from multi_scan import DGADetection
detector = DGADetection()
result = detector.multi_predict_single_dname("suspicious-domain.com")

# Database operations
from db import DatabaseOperation
db = DatabaseOperation()
sample = db.mysqlsha256s("abcdef1234...")
```

### Command Line

```bash
# Check configuration
python web/flask/config.py

# Run ensemble prediction
python multi_scan/ensemble_predict.py

# Calculate file hashes
python utils/pe_hash_calculator.py sample.exe

# Detect packers
python utils/packer_detector.py sample.exe
```

## Hash Types

The system calculates 11 hash types for malware analysis:

- **Cryptographic**: MD5, SHA1, SHA256, SHA512
- **PE-specific**: Imphash, Authentihash, Rich Header Hash, vHash, PEhash
- **Fuzzy**: SSDEEP, TLSH

## Security Notes

**WARNING**: This is a malware analysis system. Files in `data/samples/` are malicious.

- Never execute samples directly on your host machine
- Use VMs or sandboxes for dynamic analysis
- Keep API keys secure in `config.ini`
- Downloaded samples are encrypted ZIP files (password: `infected`)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details.

## Authors

**NKAMG - Nankai Anti-Malware Group**

## Acknowledgments

- [VirusTotal](https://www.virustotal.com/) for malware intelligence API
- [EMBER](https://github.com/elastic/ember) for feature extraction
- [MalConv](https://arxiv.org/abs/1710.09435) for raw bytes CNN architecture
