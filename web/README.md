# NKREPO Web Application

Complete malware repository and analysis web application with FastAPI backend and Vue.js frontend.

## 🎉 Recent Updates (2026-04-29)

### Security Fixes
- ✅ Fixed 12 Dependabot security vulnerabilities
- ✅ Fixed 26 CodeQL security issues
- ✅ Upgraded vulnerable dependencies (xlsx, Pillow, lxml, PyJWT, aiohttp, etc.)
- ✅ Added input validation for path traversal prevention
- ✅ Fixed SQL injection vulnerabilities
- ✅ Removed sensitive information from logs
- ✅ Fixed ReDoS vulnerabilities in regex patterns
- ✅ Improved error handling to prevent information disclosure

### Dependency Updates
- `xlsx` → `xlsx-js-style` (fixed ReDoS vulnerability)
- `Pillow` upgraded to 12.2.0 (fixed PSD/GZIP vulnerabilities)
- `python-multipart` upgraded to 0.0.27
- `lxml` upgraded to 6.1.0 (fixed XXE vulnerability)
- `PyJWT` upgraded to 2.12.1
- `aiohttp` upgraded to 3.13.5 (fixed multiple DoS vulnerabilities)

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+ (LTS recommended)
- MySQL Server 5.7+

### Installation

**1. Database Setup**
```bash
cd db
python init_db.py
```

**2. Backend Setup (FastAPI)**
```bash
cd web/fastapi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure
cp config.ini.example config.ini
# Edit config.ini with your settings
```

**3. Frontend Setup (Vue.js)**
```bash
cd web/vue
npm install
```

**4. Run Application**
```bash
# Terminal 1 - Backend
cd web/fastapi
source venv/bin/activate
python main.py

# Terminal 2 - Frontend
cd web/vue
npm run dev
```

**5. Access**
- Frontend: http://localhost:9528
- Backend API: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

---

## Architecture

### Backend (FastAPI)
- **Location:** `web/fastapi/`
- **Port:** 8000
- **Framework:** FastAPI 0.115+
- **Database:** MySQL (3 databases, 256 sharded tables)
- **Features:**
  - RESTful API endpoints with automatic documentation
  - File upload and malware detection
  - Multi-model ML/DL classification
  - VirusTotal integration
  - DGA (Domain Generation Algorithm) detection
  - ATT&CK technique mapping
  - FlowViz attack flow visualization

**Key Files:**
- `main.py` - Main FastAPI application
- `config.ini` - Configuration file
- `requirements.txt` - Python dependencies
- `app/api/` - API endpoints
- `app/services/` - Business logic
- `app/services/flowviz/` - AI-powered attack flow analysis

**Key Directories:**
- `app/api/` - API route handlers
- `app/services/av_detection/` - Antivirus scanning
- `app/services/flowviz/` - FlowViz AI analysis
- `data/` - ML models and features
- `logs/` - Application logs

---

### Frontend (Vue.js)
- **Location:** `web/vue/`
- **Port:** 9528 (development)
- **Framework:** Vue 3 + Vite
- **UI Library:** Element Plus
- **Features:**
  - Malware sample search and analysis
  - File detection with ensemble ML models
  - DGA domain detection
  - VirusTotal integration
  - ATT&CK matrix visualization
  - Interactive data visualization (ECharts)
  - FlowViz attack flow visualization

**Key Files:**
- `src/views/detect/` - Detection pages
- `src/views/file_search/` - Sample search pages
- `src/utils/request.js` - Axios API client
- `vite.config.js` - Vite configuration
- `.env.development` - Development API endpoint configuration

---

## Technology Stack

### Backend
| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ | Runtime |
| FastAPI | 0.115+ | Web framework |
| PyMySQL | Latest | MySQL connector |
| PyTorch | Latest | Deep learning |
| scikit-learn | Latest | Machine learning |
| XGBoost | Latest | Gradient boosting |
| LIEF | 0.14+ | PE parsing |
| yara-python | 4.3+ | YARA scanning |
| aiohttp | 3.13.5+ | Async HTTP client |

### Frontend
| Component | Version | Purpose |
|-----------|---------|---------|
| Vue.js | 3.x | Frontend framework |
| Element Plus | Latest | UI components |
| Axios | Latest | HTTP client |
| Pinia | Latest | State management |
| Vue Router | 4.x | Routing |
| ECharts | 5.x | Data visualization |

### Database
| Component | Version | Purpose |
|-----------|---------|---------|
| MySQL | 5.7+ / 8.0+ | Main database |
| nkrepo | - | Sample repository (256 sharded tables) |
| nkrepo_category | - | Malware categories |
| nkrepo_family | - | Malware families |
| nkrepo_platform | - | Target platforms |

---

## Project Structure

```
web/
├── fastapi/                    # Backend API
│   ├── venv/                   # Python virtual environment
│   ├── app/
│   │   ├── api/                # API endpoints
│   │   │   ├── query.py        # Sample query
│   │   │   ├── detect.py       # Malware detection
│   │   │   ├── av_scan.py      # AV scanning
│   │   │   └── attck.py        # ATT&CK mapping
│   │   ├── services/
│   │   │   ├── av_detection/   # AV detection service
│   │   │   └── flowviz/        # FlowViz AI analysis
│   │   └── core/               # Core utilities
│   ├── data/                   # ML models and features
│   ├── logs/                   # Application logs
│   ├── main.py                 # Main FastAPI app
│   ├── config.ini              # Configuration
│   └── requirements.txt        # Python dependencies
│
└── vue/                        # Frontend SPA
    ├── node_modules/           # npm packages
    ├── public/                 # Static assets
    ├── src/
    │   ├── api/                # API service modules
    │   ├── components/         # Reusable components
    │   ├── views/              # Page components
    │   │   ├── detect/         # Detection pages
    │   │   ├── file_search/    # Sample search pages
    │   │   └── dashboard/      # Dashboard
    │   ├── App.vue             # Root component
    │   └── main.js             # Entry point
    ├── package.json            # npm dependencies
    └── vite.config.js          # Vite config
```

---

## Configuration

### Backend Configuration (`web/fastapi/config.ini`)

```ini
[ini]
ip = 127.0.0.1
port = 8000

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

[paths]
sample_root = ../../../data/samples
web_upload_dir = ../../../data/web_upload_file
zips_dir = ../../../data/zips

[security]
secret_key = CHANGE_TO_RANDOM_SECRET
```

### Frontend Configuration (`.env.development`)

```env
ENV = 'development'
VUE_APP_BASE_API = 'http://127.0.0.1:8000'
```

---

## Security Features

### Input Validation
- SHA256 format validation (prevents path traversal)
- Task ID validation (prevents directory traversal)
- SQL injection prevention (parameterized queries)
- Command injection prevention (input sanitization)

### Information Security
- Sensitive data removed from logs
- Generic error messages to users
- Debug mode controlled by environment variable
- API keys not logged

### Dependency Security
- Regular dependency updates
- Automated vulnerability scanning (Dependabot)
- Static code analysis (CodeQL)
- Security-first configuration

---

## Development

### Backend Development

**Start Development Server:**
```bash
cd web/fastapi
source venv/bin/activate
python main.py
```

**Run Tests:**
```bash
pytest tests/
```

### Frontend Development

**Start Dev Server:**
```bash
cd web/vue
npm run dev
```

**Build for Production:**
```bash
npm run build
```

---

## API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

---

## Features

### Malware Detection
- **Local ML Models:** 8 machine learning models (AdaBoost, DT, GBDT, GNB, KNN, LR, RF, XGBoost)
- **Docker Models:** Containerized deep learning models via Docker API
- **Packer Detection:** LIEF + YARA based detection
- **Hash Analysis:** Multiple hash types

### Sample Management
- **SHA256-based Storage:** 5-level directory structure
- **Sharded Database:** 256 tables for performance
- **Search Capabilities:** By hash, category, family, platform

### Integration
- **VirusTotal API:** External scanning
- **YARA Rules:** Packer detection
- **ATT&CK Framework:** Technique mapping
- **FlowViz:** AI-powered attack flow visualization
- **Docker Models:** Containerized model inference

---

## Troubleshooting

### Backend Issues

**Problem:** FastAPI won't start
```bash
# Check Python version
python --version  # Should be 3.9+

# Check dependencies
pip install -r requirements.txt
```

**Problem:** Database connection error
- Check MySQL is running
- Verify credentials in config.ini
- Test connection: `mysql -u root -p`

### Frontend Issues

**Problem:** npm install fails
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Problem:** Port 9528 in use
```bash
# Change port in vite.config.js
```

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
- Check API documentation at `/docs`
- Review logs in `web/fastapi/logs/`
- Check browser console (F12) for frontend errors

---

**NKREPO Web Application**
Version 2.0 | NKAMG © 2026
