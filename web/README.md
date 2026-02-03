# NKREPO Web Application

Complete malware repository and analysis web application with Flask backend and Vue.js frontend.

## Quick Start (Windows)

### Prerequisites
- Python 3.8+
- Node.js 8.9+ (LTS recommended)
- MySQL Server 5.7+

### Installation

**1. Database Setup**
```cmd
cd db
setup_database_windows.bat
```

**2. Backend Setup**
```cmd
cd web\flask
setup_windows.bat
notepad config.ini  # Configure MySQL credentials
```

**3. Frontend Setup**
```cmd
cd web\vue
setup_windows.bat
```

**4. Run Application**
```cmd
# Terminal 1 - Backend
cd web\flask
run_flask.bat

# Terminal 2 - Frontend
cd web\vue
run_dev.bat
```

**5. Access**
- Frontend: http://localhost:9528
- Backend API: http://127.0.0.1:5005

For detailed installation instructions, see **[INSTALLATION_WINDOWS.md](INSTALLATION_WINDOWS.md)**

---

## Architecture

### Backend (Flask)
- **Location:** `web/flask/`
- **Port:** 5005
- **Framework:** Flask 3.1+
- **Database:** MySQL (3 databases, 256 sharded tables)
- **Features:**
  - RESTful API endpoints
  - File upload and malware detection
  - Multi-model ML/DL classification
  - VirusTotal integration
  - DGA (Domain Generation Algorithm) detection

**Key Files:**
- `app.py` - Main Flask application
- `config.ini` - Configuration file (create from config.ini.example)
- `requirements.txt` - Python dependencies
- `feeds/` - ML classifiers (SVM, KNN, RF, XGBoost, LSTM)
- `models/` - Deep learning models (MalConv, Transformer, EMBER)

**Setup Scripts:**
- `setup_windows.bat` - Automated Windows installation
- `run_flask.bat` - Start development server

---

### Frontend (Vue.js)
- **Location:** `web/vue/`
- **Port:** 9528 (development)
- **Framework:** Vue 2.6.10 + Vue Element Admin 4.4.0
- **UI Library:** Element UI 2.13.2
- **Features:**
  - Malware sample search and analysis
  - File detection with ensemble ML models
  - DGA domain detection
  - VirusTotal integration
  - Interactive data visualization (ECharts)

**Key Files:**
- `src/views/detect/` - Detection pages
- `src/views/file_search/` - Sample search pages
- `src/utils/request.js` - Axios API client
- `vue.config.js` - Vue CLI configuration (includes proxy setup)
- `.env.development.local` - Development API endpoint configuration
- `.env.production.local` - Production API endpoint configuration

**Setup Scripts:**
- `setup_windows.bat` - Automated Windows installation
- `run_dev.bat` - Start development server
- `build_prod.bat` - Build for production

---

## Technology Stack

### Backend
| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Runtime |
| Flask | 3.1+ | Web framework |
| PyMySQL | Latest | MySQL connector |
| PyTorch | Latest | Deep learning |
| scikit-learn | Latest | Machine learning |
| XGBoost | Latest | Gradient boosting |
| LIEF | 0.14+ | PE parsing |
| yara-python | 4.3+ | YARA scanning |
| pefile | 2024.8.26 | PE analysis |

### Frontend
| Component | Version | Purpose |
|-----------|---------|---------|
| Vue.js | 2.6.10 | Frontend framework |
| Element UI | 2.13.2 | UI components |
| Axios | 0.18.1 | HTTP client |
| Vuex | 3.1.0 | State management |
| Vue Router | 3.0.2 | Routing |
| ECharts | 4.2.1 | Data visualization |

### Database
| Component | Version | Purpose |
|-----------|---------|---------|
| MySQL | 5.7+ / 8.0+ | Main database |
| nkrepo | - | Sample repository (256 sharded tables) |
| nkrepo_category | - | Malware categories |
| nkrepo_family | - | Malware families |
| nkrepo_platform | - | Target platforms |

---

## Database Schema

### Main Database: `nkrepo`

**Sample Tables (256 sharded):** `sample_00` to `sample_ff`
```sql
CREATE TABLE sample_XX (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    md5 VARCHAR(255),
    sha256 VARCHAR(255),
    src_file VARCHAR(255),
    date VARCHAR(255),
    category VARCHAR(255),
    platform VARCHAR(255),
    family VARCHAR(255),
    result VARCHAR(255),
    filetype VARCHAR(255),
    packer VARCHAR(255),
    ssdeep VARCHAR(255)
);
```

**Domain Table:** `domain_YYYYMMDD`
```sql
CREATE TABLE domain_YYYYMMDD (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    category VARCHAR(255),
    source VARCHAR(255)
);
```

**User Table:** `user`
```sql
CREATE TABLE user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password_hash VARCHAR(128) NOT NULL,
    is_active TINYINT(1) NOT NULL DEFAULT 1
);
```

---

## API Endpoints

### File Detection
```
POST /api/detect
Content-Type: multipart/form-data

Request:
  - file: Binary file upload

Response:
{
  "code": 20000,
  "data": {
    "md5": "...",
    "sha256": "...",
    "is_malware": true,
    "ml_results": {...},
    "dl_results": {...},
    "vt_result": {...}
  }
}
```

### DGA Detection
```
POST /api/dga/detect
Content-Type: application/json

Request:
{
  "domain": "example.com"
}

Response:
{
  "code": 20000,
  "data": {
    "domain": "example.com",
    "is_dga": true,
    "confidence": 0.95,
    "model_results": {...}
  }
}
```

### Sample Search
```
GET /api/samples?sha256=...
GET /api/samples?md5=...
GET /api/samples?category=...
GET /api/samples?family=...

Response:
{
  "code": 20000,
  "data": {
    "total": 100,
    "samples": [...]
  }
}
```

---

## Configuration

### Backend Configuration (`web/flask/config.ini`)

```ini
[ini]
ip = 127.0.0.1
port = 5005
row_per_page = 20

[mysql]
host = localhost
port = 3306
user = root
passwd = YOUR_PASSWORD
db_category = nkrepo_category
db_family = nkrepo_family
db_platform = nkrepo_platform
charset = utf8mb4

[API]
vt_key = YOUR_VIRUSTOTAL_API_KEY

[files]
sample_repo = C:/nkrepo/data/samples
model_path = C:/nkrepo/web/flask/models
training_data = C:/nkrepo/web/flask/training_data
upload_folder = ../vue/uploads

[security]
secret_key = CHANGE_TO_RANDOM_SECRET
cors_origins = http://localhost:9528,http://127.0.0.1:9528
```

### Frontend Configuration (`.env.development.local`)

```env
ENV = 'development'

# Direct connection to Flask backend
VUE_APP_BASE_API = 'http://127.0.0.1:5005'

# Or use proxy mode
# VUE_APP_BASE_API = '/dev-api'
```

---

## Development

### Backend Development

**Start Development Server:**
```cmd
cd web\flask
venv\Scripts\activate.bat
python app.py
```

**Run Tests:**
```cmd
pytest tests/
```

**Add New Endpoint:**
1. Add route in `app.py`
2. Implement handler function
3. Update CORS settings if needed
4. Test with Postman or curl

### Frontend Development

**Start Dev Server:**
```cmd
cd web\vue
npm run dev
```

**Lint Code:**
```cmd
npm run lint
```

**Run Unit Tests:**
```cmd
npm run test:unit
```

**Add New Page:**
1. Create component in `src/views/`
2. Add route in `src/router/index.js`
3. Add menu item in layout
4. Test in browser

---

## Production Deployment

### Build Frontend
```cmd
cd web\vue
build_prod.bat
```
Output: `dist/` folder with optimized static files

### Configure Production Backend
1. Update `config.ini` with production settings
2. Set `ip = 0.0.0.0` to listen on all interfaces
3. Use strong secret key
4. Configure proper CORS origins

### Serve with Production Server

**Windows (Waitress):**
```cmd
pip install waitress
waitress-serve --port=5005 app:app
```

**Linux (Gunicorn):**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5005 app:app
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Vue frontend
    location / {
        root /path/to/nkrepo/web/vue/dist;
        try_files $uri $uri/ /index.html;
    }

    # Flask backend API
    location /api {
        proxy_pass http://127.0.0.1:5005;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Troubleshooting

### Backend Issues

**Problem:** Flask won't start - "config.ini not found"
```cmd
cd web\flask
copy config.ini.example config.ini
notepad config.ini
```

**Problem:** Database connection error
- Check MySQL is running: `net start MySQL80`
- Verify credentials in config.ini
- Test connection: `mysql -u root -p`

**Problem:** Missing Python packages
```cmd
cd web\flask
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Frontend Issues

**Problem:** npm install fails
```cmd
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Problem:** Port 9528 in use
```cmd
# Change port
set PORT=9529 && npm run dev
```

**Problem:** CORS errors
- Use proxy mode in `.env.development.local`
- Verify CORS settings in Flask config.ini

### Database Issues

**Problem:** Database not found
```cmd
cd db
setup_database_windows.bat
```

**Problem:** Access denied
- Verify MySQL credentials
- Check user has proper privileges
- Reset MySQL password if needed

For detailed troubleshooting, see **[INSTALLATION_WINDOWS.md](INSTALLATION_WINDOWS.md#troubleshooting)**

---

## Project Structure

```
web/
├── flask/                    # Backend API
│   ├── venv/                 # Python virtual environment
│   ├── feeds/                # ML classifiers
│   │   ├── svm_classifier.py
│   │   ├── knn_classifier.py
│   │   ├── rf_classifier.py
│   │   ├── xgboost_classifier.py
│   │   └── lstm_classifier.py
│   ├── models/               # DL models
│   │   ├── malconv_model.py
│   │   ├── transformer_model.py
│   │   └── ember_model.py
│   ├── app.py                # Main Flask app
│   ├── config.ini            # Configuration (create from template)
│   ├── config.ini.example    # Configuration template
│   ├── requirements.txt      # Python dependencies
│   ├── setup_windows.bat     # Windows installation script
│   └── run_flask.bat         # Windows run script
│
└── vue/                      # Frontend SPA
    ├── node_modules/         # npm packages
    ├── public/               # Static assets
    ├── src/
    │   ├── api/              # API service modules
    │   ├── assets/           # Images, fonts
    │   ├── components/       # Reusable components
    │   ├── layout/           # Layout components
    │   ├── router/           # Vue Router config
    │   ├── store/            # Vuex store
    │   ├── styles/           # Global styles
    │   ├── utils/            # Utility functions
    │   ├── views/            # Page components
    │   │   ├── detect/       # Detection pages
    │   │   │   ├── ensemble-ml.vue    # Ensemble ML detection
    │   │   │   ├── deep-learning.vue  # DL detection
    │   │   │   └── dga-detection.vue  # DGA detection
    │   │   ├── file_search/  # Sample search pages
    │   │   │   ├── search-by-hash.vue
    │   │   │   ├── search-by-category.vue
    │   │   │   └── search-by-family.vue
    │   │   └── dashboard/    # Dashboard
    │   ├── App.vue           # Root component
    │   └── main.js           # Entry point
    ├── tests/                # Unit tests
    ├── .env.development      # Development config
    ├── .env.production       # Production config
    ├── .env.development.local # Local dev config (create from template)
    ├── .env.production.local  # Local prod config (create from template)
    ├── package.json          # npm dependencies
    ├── vue.config.js         # Vue CLI config
    ├── setup_windows.bat     # Windows installation script
    ├── run_dev.bat           # Windows dev run script
    └── build_prod.bat        # Windows build script
```

---

## Features

### Malware Detection
- **Ensemble ML:** 10 machine learning models (SVM, KNN, RF, XGBoost, etc.)
- **Deep Learning:** 10 neural network models (MalConv, Transformer, EMBER, etc.)
- **Packer Detection:** LIEF + YARA based detection for 40+ packers
- **Hash Analysis:** 11 hash types including imphash, authentihash, SSDEEP, TLSH

### Sample Management
- **SHA256-based Storage:** 5-level directory structure
- **Sharded Database:** 256 tables for performance
- **Search Capabilities:** By hash, category, family, platform
- **Batch Operations:** Import/export, bulk analysis

### Integration
- **VirusTotal API:** External scanning and validation
- **YARA Rules:** 40+ packer detection rules
- **Multi-AV Scanning:** Kaspersky integration
- **DGA Detection:** Malicious domain classification

---

## Documentation

- **[INSTALLATION_WINDOWS.md](INSTALLATION_WINDOWS.md)** - Complete Windows installation guide
- **[../DEPENDENCIES_GUIDE.md](../DEPENDENCIES_GUIDE.md)** - Python dependencies and troubleshooting
- **[../utils/PE_HASH_GUIDE.md](../utils/PE_HASH_GUIDE.md)** - PE hash calculator documentation
- **[../yara_rules/YARA_USAGE_GUIDE.md](../yara_rules/YARA_USAGE_GUIDE.md)** - YARA rules documentation

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

---

## License

NKAMG (Nankai Anti-Malware Group)

---

## Support

For issues and questions:
- Check documentation in `web/INSTALLATION_WINDOWS.md`
- Run verification: `python utils/verify_installation.py`
- Review logs in `web/flask/logs/`
- Check browser console (F12) for frontend errors

---

**NKREPO Web Application**
Version 1.0 | NKAMG © 2026
