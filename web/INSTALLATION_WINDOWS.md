# NKREPO Web Application - Windows Installation Guide

Complete step-by-step guide for installing and running the NKREPO web application on Windows.

## Table of Contents

- [System Requirements](#system-requirements)
- [Prerequisites Installation](#prerequisites-installation)
- [Database Setup](#database-setup)
- [Backend (Flask) Setup](#backend-flask-setup)
- [Frontend (Vue.js) Setup](#frontend-vuejs-setup)
- [Running the Application](#running-the-application)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

---

## System Requirements

### Minimum Requirements
- **OS:** Windows 10/11 (64-bit)
- **RAM:** 4 GB minimum, 8 GB recommended
- **Storage:** 10 GB free space
- **Internet:** Required for package installation

### Software Requirements
- Python 3.8+ (tested with 3.8-3.13)
- Node.js 8.9+ (LTS version recommended, tested with Node 14-20)
- MySQL Server 5.7+ or 8.0+
- Git (optional, for cloning repository)

---

## Prerequisites Installation

### 1. Install Python

**Download and Install:**
1. Visit https://www.python.org/downloads/
2. Download Python 3.8 or later (3.10+ recommended)
3. **IMPORTANT:** Check "Add Python to PATH" during installation
4. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

**Expected output:**
```
Python 3.10.x
pip 23.x.x
```

---

### 2. Install Node.js

**Download and Install:**
1. Visit https://nodejs.org/
2. Download LTS version (recommended)
3. Run installer with default settings
4. Verify installation:
   ```cmd
   node --version
   npm --version
   ```

**Expected output:**
```
v14.x.x (or later)
6.x.x (or later)
```

---

### 3. Install MySQL Server

**Download and Install:**
1. Visit https://dev.mysql.com/downloads/mysql/
2. Download MySQL Installer for Windows
3. Run installer and select "Server only" or "Full" installation
4. During setup:
   - Choose "Development Computer" configuration
   - Set root password (remember this!)
   - Use default port 3306
   - Configure as Windows Service (auto-start)

5. Verify installation:
   ```cmd
   mysql --version
   ```

**Expected output:**
```
mysql  Ver 8.0.x for Win64
```

**Test MySQL Connection:**
```cmd
mysql -u root -p
```
Enter your password and ensure you can connect.

---

## Database Setup

### Step 1: Navigate to Database Directory
```cmd
cd C:\nkrepo\db
```

### Step 2: Run Database Setup Script
```cmd
setup_database_windows.bat
```

This script will:
1. Check Python and MySQL installation
2. Install `mysql-connector-python` package
3. Prompt for MySQL credentials
4. Create the `nkrepo` database
5. Create 256 sharded sample tables (sample_00 to sample_ff)
6. Create domain and user tables
7. Optionally create additional databases (category, family, platform)

**Example Session:**
```
MySQL Host (default: localhost): localhost
MySQL Username (default: root): root
MySQL Password: ********

Creating Database and Tables...
[o] Database nkrepo connected successfully.
[o] Table sample_00 created successfully.
...
[o] All sample_xy tables created successfully!
```

### Manual Database Creation (Alternative)

If the batch script fails, create databases manually:

```sql
-- Connect to MySQL
mysql -u root -p

-- Create databases
CREATE DATABASE IF NOT EXISTS nkrepo DEFAULT CHARSET utf8mb4;
CREATE DATABASE IF NOT EXISTS nkrepo_category DEFAULT CHARSET utf8mb4;
CREATE DATABASE IF NOT EXISTS nkrepo_family DEFAULT CHARSET utf8mb4;
CREATE DATABASE IF NOT EXISTS nkrepo_platform DEFAULT CHARSET utf8mb4;

-- Verify
SHOW DATABASES;
```

Then run the Python script directly:
```cmd
python init_db.py -u root -p YOUR_PASSWORD -h localhost
```

---

## Backend (Flask) Setup

### Step 1: Navigate to Flask Directory
```cmd
cd C:\nkrepo\web\flask
```

### Step 2: Run Flask Setup Script
```cmd
setup_windows.bat
```

This script will:
1. Check Python and pip installation
2. Create Python virtual environment (`venv/`)
3. Activate virtual environment
4. Upgrade pip
5. Install all dependencies from `requirements.txt`
6. Create `config.ini` from template
7. Create necessary directories (uploads, training_data, logs)
8. Open `config.ini` in Notepad for configuration

**Expected Output:**
```
[*] Python found:
Python 3.10.x

[*] Creating virtual environment...
[+] Virtual environment created

[*] Activating virtual environment...
[*] Upgrading pip...
[*] Installing dependencies from requirements.txt...
...
Setup Complete!
```

### Step 3: Configure config.ini

The script will open `config.ini` in Notepad. Update the following sections:

#### [mysql] Section
```ini
[mysql]
host = localhost
port = 3306
user = root
passwd = YOUR_MYSQL_PASSWORD_HERE
db_category = nkrepo_category
db_family = nkrepo_family
db_platform = nkrepo_platform
charset = utf8mb4
```

#### [API] Section (VirusTotal)
```ini
[API]
# Get your API key from https://www.virustotal.com/gui/my-apikey
vt_key = YOUR_VIRUSTOTAL_API_KEY_HERE
```

If you don't have a VirusTotal API key, you can skip this for now. Some features will be disabled.

#### [files] Section
```ini
[files]
# Use forward slashes or double backslashes on Windows
sample_repo = C:/nkrepo/data/samples
model_path = C:/nkrepo/web/flask/models
training_data = C:/nkrepo/web/flask/training_data
upload_folder = ../vue/uploads
```

#### [security] Section
```ini
[security]
# Change this to a random string for production!
secret_key = change_this_to_a_random_secret_key_$(date)
cors_origins = http://localhost:9528,http://127.0.0.1:9528
```

**Generate a secure secret key:**
```cmd
python -c "import secrets; print(secrets.token_hex(32))"
```

### Step 4: Verify Backend Installation

Check that all dependencies are installed:
```cmd
cd C:\nkrepo
python utils\verify_installation.py
```

---

## Frontend (Vue.js) Setup

### Step 1: Navigate to Vue Directory
```cmd
cd C:\nkrepo\web\vue
```

### Step 2: Run Vue Setup Script
```cmd
setup_windows.bat
```

This script will:
1. Check Node.js and npm installation
2. Optionally use Taobao npm mirror (faster in China)
3. Install all dependencies from `package.json`
4. Create `.env.development.local` configuration file
5. Create `.env.production.local` configuration file
6. Create uploads directory

**Expected Output:**
```
[*] Node.js found:
v14.x.x

[*] npm found:
6.x.x

[*] Installing dependencies from package.json...
[*] This may take several minutes...
...
[+] Dependencies installed successfully!
```

**Note:** The first `npm install` may take 5-10 minutes depending on your internet connection.

### Step 3: Configure Frontend API Endpoint

The setup script creates `.env.development.local`. Open it and verify:

```env
# Development Environment Configuration
ENV = 'development'

# Flask backend API URL
# Option 1: Direct connection
VUE_APP_BASE_API = 'http://127.0.0.1:5005'

# Option 2: Use proxy (see vue.config.js)
# VUE_APP_BASE_API = '/dev-api'
```

**Choosing API Configuration:**

- **Direct Connection:** Use `http://127.0.0.1:5005` if Flask backend has CORS enabled (already configured)
- **Proxy Mode:** Use `/dev-api` to let Vue dev server proxy requests (configured in `vue.config.js`)

Proxy mode is recommended for development to avoid CORS issues.

---

## Running the Application

### Step 1: Start MySQL Server

Ensure MySQL is running:
```cmd
# Check if MySQL service is running
sc query MySQL80
```

If not running, start it:
```cmd
net start MySQL80
```

(Replace `MySQL80` with your MySQL service name)

### Step 2: Start Flask Backend

Open a new Command Prompt window:
```cmd
cd C:\nkrepo\web\flask
run_flask.bat
```

**Expected Output:**
```
========================================================================
Starting NKREPO Flask Backend Server
========================================================================

[*] Activating virtual environment...
[*] Starting Flask server...
[*] Server will be available at: http://127.0.0.1:5005
[*] Press Ctrl+C to stop the server

 * Serving Flask app 'app.py'
 * Running on http://127.0.0.1:5005
```

**Keep this window open!** The Flask server must remain running.

### Step 3: Start Vue Frontend

Open **another** Command Prompt window:
```cmd
cd C:\nkrepo\web\vue
run_dev.bat
```

**Expected Output:**
```
========================================================================
Starting NKREPO Vue Frontend Development Server
========================================================================

[*] Starting development server...
[*] Frontend will be available at: http://localhost:9528

App running at:
  - Local:   http://localhost:9528/
  - Network: http://192.168.x.x:9528/
```

The browser will automatically open to http://localhost:9528

### Step 4: Access the Application

1. **Frontend URL:** http://localhost:9528
2. **Backend API:** http://127.0.0.1:5005

**Default Login (if authentication is enabled):**
- Username: admin
- Password: (check database or create user)

---

## Troubleshooting

### Common Issues

#### 1. Flask Server Won't Start

**Error: `config.ini not found`**
```
[ERROR] config.ini not found!
```

**Solution:**
```cmd
cd web\flask
copy config.ini.example config.ini
notepad config.ini
# Configure MySQL credentials
```

---

**Error: `Failed to connect to MySQL`**
```
pymysql.err.OperationalError: (2003, "Can't connect to MySQL server...")
```

**Solution:**
- Check MySQL is running: `net start MySQL80`
- Verify credentials in `config.ini`
- Test connection: `mysql -u root -p`

---

**Error: `ModuleNotFoundError: No module named 'flask'`**

**Solution:**
Virtual environment not activated. Run:
```cmd
cd web\flask
venv\Scripts\activate.bat
python app.py
```

Or use `run_flask.bat` which handles activation automatically.

---

#### 2. Vue Dev Server Won't Start

**Error: `node_modules not found`**

**Solution:**
```cmd
cd web\vue
npm install
```

---

**Error: `EADDRINUSE: Port 9528 already in use`**

**Solution:**
Kill existing process or change port:
```cmd
# Find process using port 9528
netstat -ano | findstr :9528

# Kill process (replace PID)
taskkill /PID <PID> /F

# Or change port
set PORT=9529 && npm run dev
```

---

**Error: `Module build failed: Error: Node Sass does not yet support your current environment`**

**Solution:**
```cmd
npm rebuild node-sass
# Or
npm install sass --save-dev
```

---

#### 3. Database Connection Issues

**Error: `Access denied for user 'root'@'localhost'`**

**Solution:**
Reset MySQL password or verify credentials:
```cmd
mysql -u root -p
# Enter password
# If successful, update config.ini with same password
```

---

**Error: `Unknown database 'nkrepo'`**

**Solution:**
Database not created. Run:
```cmd
cd db
setup_database_windows.bat
```

---

#### 4. CORS Errors in Browser Console

**Error:**
```
Access to XMLHttpRequest at 'http://127.0.0.1:5005' from origin 'http://localhost:9528'
has been blocked by CORS policy
```

**Solution 1:** Use proxy mode in `.env.development.local`:
```env
VUE_APP_BASE_API = '/dev-api'
```

**Solution 2:** Verify CORS configuration in `web/flask/app.py`:
```python
CORS(app, origins=['http://localhost:9528', 'http://127.0.0.1:9528'])
```

---

#### 5. Missing Dependencies

**Error: `ImportError: DLL load failed while importing yara`**

**Solution:**
Install yara-python properly:
```cmd
pip uninstall yara-python
pip install --upgrade yara-python
```

For Windows, you may need to install from wheel:
1. Download from: https://github.com/VirusTotal/yara-python/releases
2. Install: `pip install yara_python-X.X.X-cpXX-cpXX-win_amd64.whl`

---

**Error: `No module named 'lief'`**

**Solution:**
```cmd
cd web\flask
venv\Scripts\activate.bat
pip install lief>=0.14.0
```

---

### Performance Issues

#### Slow Frontend Build

**Symptom:** `npm install` takes very long

**Solution:**
Use npm mirror (Taobao for China):
```cmd
npm config set registry https://registry.npmmirror.com
npm install
```

Or use `yarn` instead:
```cmd
npm install -g yarn
yarn install
```

---

#### Slow Backend Startup

**Symptom:** Flask takes long to start

**Solution:**
- Disable ML model loading during development
- Check antivirus isn't scanning Python files
- Use SSD instead of HDD

---

### Log Files

Check log files for detailed error messages:

**Flask Backend Logs:**
- Location: `web/flask/logs/`
- Flask stdout in terminal window

**Vue Frontend Logs:**
- Browser Developer Console (F12)
- Terminal window running `npm run dev`

**MySQL Logs:**
- Location: `C:\ProgramData\MySQL\MySQL Server 8.0\Data\`
- File: `hostname.err`

---

## Production Deployment

### Building for Production

#### 1. Build Vue Frontend
```cmd
cd web\vue
build_prod.bat
```

This creates `dist/` folder with optimized static files.

#### 2. Configure Production Backend

Update `config.ini` for production:
```ini
[ini]
ip = 0.0.0.0          # Listen on all interfaces
port = 5005

[security]
secret_key = CHANGE_TO_SECURE_RANDOM_KEY
cors_origins = http://your-production-domain.com
```

#### 3. Use Production Server (Gunicorn)

Install Gunicorn:
```cmd
pip install gunicorn
```

**Note:** Gunicorn doesn't work on Windows. Use alternative:

**Option 1: Waitress (Windows-compatible)**
```cmd
pip install waitress
waitress-serve --port=5005 app:app
```

**Option 2: Use IIS with wfastcgi**
```cmd
pip install wfastcgi
wfastcgi-enable
# Configure IIS with FastCGI
```

**Option 3: Deploy on Linux**
Recommended for production. See Linux deployment guide.

#### 4. Serve Vue Static Files

**Option 1: Nginx**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Serve Vue static files
    location / {
        root C:/nkrepo/web/vue/dist;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to Flask
    location /api {
        proxy_pass http://127.0.0.1:5005;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Option 2: IIS**
1. Install IIS and URL Rewrite module
2. Copy `dist/` to `C:\inetpub\wwwroot\nkrepo`
3. Configure reverse proxy to Flask backend

#### 5. Database Production Settings

```sql
-- Create production user with limited privileges
CREATE USER 'nkrepo'@'localhost' IDENTIFIED BY 'secure_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON nkrepo.* TO 'nkrepo'@'localhost';
FLUSH PRIVILEGES;
```

Update `config.ini`:
```ini
[mysql]
user = nkrepo
passwd = secure_password
```

#### 6. Security Checklist

- [ ] Change default secret key
- [ ] Use strong MySQL password
- [ ] Enable MySQL firewall (only localhost)
- [ ] Set proper CORS origins
- [ ] Disable Flask debug mode
- [ ] Use HTTPS with SSL certificate
- [ ] Set up regular database backups
- [ ] Configure Windows Firewall rules
- [ ] Use environment variables for secrets

---

## Directory Structure

After complete installation:

```
C:\nkrepo\
├── data\
│   └── samples\          # Malware sample storage (5-level SHA256 structure)
├── db\
│   ├── init_db.py        # Database initialization script
│   └── setup_database_windows.bat
├── web\
│   ├── flask\            # Backend API
│   │   ├── venv\         # Python virtual environment
│   │   ├── config.ini    # Flask configuration (CREATED)
│   │   ├── app.py        # Main Flask application
│   │   ├── logs\         # Application logs (CREATED)
│   │   ├── training_data\  # ML training data (CREATED)
│   │   ├── setup_windows.bat
│   │   └── run_flask.bat
│   └── vue\              # Frontend SPA
│       ├── node_modules\ # npm packages (CREATED)
│       ├── uploads\      # File upload directory (CREATED)
│       ├── dist\         # Production build output
│       ├── .env.development.local  # Dev config (CREATED)
│       ├── .env.production.local   # Prod config (CREATED)
│       ├── setup_windows.bat
│       ├── run_dev.bat
│       └── build_prod.bat
├── utils\                # Malware analysis utilities
└── yara_rules\           # YARA detection rules
```

---

## Quick Start Summary

**Complete installation in 4 steps:**

```cmd
# 1. Database Setup
cd C:\nkrepo\db
setup_database_windows.bat

# 2. Flask Backend Setup
cd ..\web\flask
setup_windows.bat
# Edit config.ini with MySQL credentials

# 3. Vue Frontend Setup
cd ..\vue
setup_windows.bat

# 4. Run Application
# Terminal 1 - Backend:
cd C:\nkrepo\web\flask
run_flask.bat

# Terminal 2 - Frontend:
cd C:\nkrepo\web\vue
run_dev.bat

# Open browser: http://localhost:9528
```

---

## Additional Resources

### Documentation
- Flask Backend API: `web/flask/README.md`
- Vue Frontend: `web/vue/README.md`
- Database Schema: `db/README.md`
- Malware Analysis Tools: `DEPENDENCIES_GUIDE.md`

### External Links
- Python: https://www.python.org/
- Node.js: https://nodejs.org/
- MySQL: https://dev.mysql.com/
- Flask Documentation: https://flask.palletsprojects.com/
- Vue.js Guide: https://vuejs.org/guide/
- Element UI: https://element.eleme.io/

---

## Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review error logs in terminal and log files
3. Verify all dependencies with `python utils/verify_installation.py`
4. Check GitHub issues: https://github.com/your-repo/nkrepo/issues

---

## Version Information

- **NKREPO Version:** 1.0
- **Flask:** 3.1+
- **Vue.js:** 2.6.10
- **Vue Element Admin:** 4.4.0
- **Python:** 3.8+
- **Node.js:** 8.9+
- **MySQL:** 5.7+

---

## License

NKAMG (Nankai Anti-Malware Group)

---

**Installation Guide Last Updated:** 2026-02-01
