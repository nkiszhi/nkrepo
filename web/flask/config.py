# -*- coding: utf-8 -*-
"""
NKREPO Configuration Module

This module provides dynamic path resolution and centralized configuration
for the NKREPO malware analysis system. All paths are computed relative to
the project root directory, making the system portable across environments.

Usage:
    from config import Config

    # Access paths
    sample_path = Config.SAMPLE_REPO
    model_path = Config.MODEL_PATH

    # Access database settings
    db_host = Config.MYSQL_HOST

    # Access API keys
    vt_key = Config.VT_API_KEY
"""

import os
import sys
from configparser import ConfigParser
from pathlib import Path


class Config:
    """
    Centralized configuration class for NKREPO.

    All paths are dynamically computed based on the project root directory.
    User-configurable settings are read from config.ini if it exists.
    """

    # ==========================================================================
    # Path Resolution
    # ==========================================================================

    # Flask app directory: web/flask/
    FLASK_DIR = Path(__file__).resolve().parent

    # Web directory: web/
    WEB_DIR = FLASK_DIR.parent

    # Project root directory: nkrepo/
    PROJECT_ROOT = WEB_DIR.parent

    # ==========================================================================
    # Directory Structure (computed from PROJECT_ROOT)
    # ==========================================================================

    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    SAMPLE_REPO = DATA_DIR / "samples"
    ZIP_STORAGE = DATA_DIR / "zips"

    # Database directory
    DB_DIR = PROJECT_ROOT / "db"

    # Utils directory
    UTILS_DIR = PROJECT_ROOT / "utils"

    # Web directories
    VUE_DIR = WEB_DIR / "vue"
    UPLOAD_FOLDER = VUE_DIR / "uploads"

    # Flask subdirectories
    MODEL_PATH = FLASK_DIR / "models"
    TRAINING_DATA = FLASK_DIR / "training_data"
    FEEDS_DIR = FLASK_DIR / "feeds"
    DATA_FILES_DIR = FLASK_DIR / "data"

    # DGA detection data files
    HMM_ADD = DATA_FILES_DIR / "hmm_matrix.txt"
    GIB_ADD = DATA_FILES_DIR / "gib_model.pkl"
    GRAMFILE_ADD = DATA_FILES_DIR / "2grams.txt"
    TLD_ADD = DATA_FILES_DIR / "tlds-alpha-by-domain.txt"

    # Feature files for DGA detection
    TRAIN_ADD = DATA_FILES_DIR / "features" / "train_features.csv"
    TEST_ADD = DATA_FILES_DIR / "features" / "test_features.csv"

    # ==========================================================================
    # Config File Handling
    # ==========================================================================

    CONFIG_FILE = FLASK_DIR / "config.ini"
    CONFIG_EXAMPLE = FLASK_DIR / "config.ini.example"

    # Load config parser
    _cp = ConfigParser()
    _config_loaded = False

    if CONFIG_FILE.exists():
        _cp.read(str(CONFIG_FILE))
        _config_loaded = True
    elif CONFIG_EXAMPLE.exists():
        # Fall back to example config if no config.ini
        _cp.read(str(CONFIG_EXAMPLE))
        _config_loaded = True

    # ==========================================================================
    # Server Settings
    # ==========================================================================

    HOST_IP = _cp.get('ini', 'ip', fallback='127.0.0.1')
    PORT = _cp.getint('ini', 'port', fallback=5005)
    ROW_PER_PAGE = _cp.getint('ini', 'row_per_page', fallback=20)

    # ==========================================================================
    # MySQL Settings
    # ==========================================================================

    MYSQL_HOST = _cp.get('mysql', 'host', fallback='localhost')
    MYSQL_PORT = _cp.getint('mysql', 'port', fallback=3306)
    MYSQL_USER = _cp.get('mysql', 'user', fallback='root')
    MYSQL_PASSWORD = _cp.get('mysql', 'passwd', fallback='')
    MYSQL_CHARSET = _cp.get('mysql', 'charset', fallback='utf8mb4')

    # Database names
    DB_MAIN = _cp.get('mysql', 'db', fallback='nkrepo')
    DB_CATEGORY = _cp.get('mysql', 'db_category', fallback='nkrepo_category')
    DB_FAMILY = _cp.get('mysql', 'db_family', fallback='nkrepo_family')
    DB_PLATFORM = _cp.get('mysql', 'db_platform', fallback='nkrepo_platform')

    # ==========================================================================
    # API Keys
    # ==========================================================================

    VT_API_KEY = _cp.get('API', 'vt_key', fallback='')

    # ==========================================================================
    # Security Settings
    # ==========================================================================

    SECRET_KEY = _cp.get('security', 'secret_key', fallback='change_this_to_a_random_secret_key')
    CORS_ORIGINS = _cp.get('security', 'cors_origins', fallback='http://localhost:9528,http://127.0.0.1:9528')

    # ==========================================================================
    # DGA Detection / ML Settings
    # ==========================================================================

    # Algorithm and classifier lists for DGA detection
    ALGORITHM_LIST = _cp.get('feeds', 'algorithm_list',
                             fallback='knn,svm,randomforest,decisiontree,naivebayes,logisticregression,adaboost,gbdt,xgboost').split(',')
    CLASSIFIER_LIST = _cp.get('feeds', 'classifier_list',
                              fallback='KNNClassifier,SVMClassifier,RandomForestClassifier,DecisionTreeClassifier,NaiveBayesClassifier,LogisticRegressionClassifier,AdaBoostClassifier,GBDTClassifier,XGBoostClassifier').split(',')

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    @classmethod
    def get_sample_path(cls, sha256: str) -> Path:
        """
        Get the full path to a sample file based on its SHA256 hash.

        Uses 5-level directory structure: ab/cd/ef/12/abcdef1234567890...

        Args:
            sha256: The SHA256 hash of the sample

        Returns:
            Path object pointing to the sample file
        """
        return cls.SAMPLE_REPO / sha256[0:2] / sha256[2:4] / sha256[4:6] / sha256[6:8] / sha256

    @classmethod
    def get_sample_dir(cls, sha256: str) -> Path:
        """
        Get the directory path for a sample based on its SHA256 hash.

        Args:
            sha256: The SHA256 hash of the sample

        Returns:
            Path object pointing to the sample directory
        """
        return cls.SAMPLE_REPO / sha256[0:2] / sha256[2:4] / sha256[4:6] / sha256[6:8]

    @classmethod
    def get_table_name(cls, sha256: str) -> str:
        """
        Get the database table name for a sample based on its SHA256 prefix.

        Uses sharding strategy: first 2 chars of SHA256 -> sample_XX table

        Args:
            sha256: The SHA256 hash of the sample

        Returns:
            Table name string (e.g., 'sample_ab')
        """
        return f"sample_{sha256[:2]}"

    @classmethod
    def get_zip_path(cls, sha256: str) -> Path:
        """
        Get the path to the encrypted ZIP file for a sample.

        Args:
            sha256: The SHA256 hash of the sample

        Returns:
            Path object pointing to the ZIP file
        """
        return cls.ZIP_STORAGE / f"{sha256}.zip"

    @classmethod
    def ensure_directories(cls):
        """
        Create all required directories if they don't exist.
        """
        directories = [
            cls.DATA_DIR,
            cls.SAMPLE_REPO,
            cls.ZIP_STORAGE,
            cls.UPLOAD_FOLDER,
            cls.MODEL_PATH,
            cls.TRAINING_DATA,
            cls.DATA_FILES_DIR,
            cls.DATA_FILES_DIR / "features",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def as_str(cls, path: Path) -> str:
        """
        Convert a Path object to string for compatibility with older code.

        Args:
            path: Path object

        Returns:
            String representation of the path
        """
        return str(path)

    @classmethod
    def print_config(cls):
        """
        Print the current configuration for debugging.
        """
        print("=" * 60)
        print("NKREPO Configuration")
        print("=" * 60)
        print(f"\nProject Root: {cls.PROJECT_ROOT}")
        print(f"Config File:  {cls.CONFIG_FILE} ({'loaded' if cls._config_loaded else 'not found'})")
        print(f"\n--- Directories ---")
        print(f"Sample Repo:  {cls.SAMPLE_REPO}")
        print(f"ZIP Storage:  {cls.ZIP_STORAGE}")
        print(f"Upload Dir:   {cls.UPLOAD_FOLDER}")
        print(f"Model Path:   {cls.MODEL_PATH}")
        print(f"Data Files:   {cls.DATA_FILES_DIR}")
        print(f"\n--- Server ---")
        print(f"Host:         {cls.HOST_IP}:{cls.PORT}")
        print(f"\n--- Database ---")
        print(f"MySQL Host:   {cls.MYSQL_HOST}:{cls.MYSQL_PORT}")
        print(f"MySQL User:   {cls.MYSQL_USER}")
        print(f"Main DB:      {cls.DB_MAIN}")
        print("=" * 60)


# ==========================================================================
# Backward Compatibility - Module-level variables
# ==========================================================================
# These allow existing code to import directly without changes
# Example: from config import SAMPLE_REPO, MYSQL_HOST

PROJECT_ROOT = Config.PROJECT_ROOT
FLASK_DIR = Config.FLASK_DIR
WEB_DIR = Config.WEB_DIR
DATA_DIR = Config.DATA_DIR

# File paths (as strings for compatibility)
SAMPLE_REPO = str(Config.SAMPLE_REPO)
ZIP_STORAGE = str(Config.ZIP_STORAGE)
UPLOAD_FOLDER = str(Config.UPLOAD_FOLDER)
MODEL_PATH = str(Config.MODEL_PATH)
TRAINING_DATA = str(Config.TRAINING_DATA)
DATA_FILES_DIR = str(Config.DATA_FILES_DIR)

# DGA detection data files
HMM_ADD = str(Config.HMM_ADD)
GIB_ADD = str(Config.GIB_ADD)
GRAMFILE_ADD = str(Config.GRAMFILE_ADD)
TLD_ADD = str(Config.TLD_ADD)
TRAIN_ADD = str(Config.TRAIN_ADD)
TEST_ADD = str(Config.TEST_ADD)

# Server settings
HOST_IP = Config.HOST_IP
PORT = Config.PORT
ROW_PER_PAGE = Config.ROW_PER_PAGE

# MySQL settings
MYSQL_HOST = Config.MYSQL_HOST
MYSQL_PORT = Config.MYSQL_PORT
MYSQL_USER = Config.MYSQL_USER
MYSQL_PASSWORD = Config.MYSQL_PASSWORD
MYSQL_CHARSET = Config.MYSQL_CHARSET

# Database names
DB_MAIN = Config.DB_MAIN
DB_CATEGORY = Config.DB_CATEGORY
DB_FAMILY = Config.DB_FAMILY
DB_PLATFORM = Config.DB_PLATFORM

# Aliases for backward compatibility with existing code
host = MYSQL_HOST
user = MYSQL_USER
passwd = MYSQL_PASSWORD
charset = MYSQL_CHARSET
db = DB_MAIN
db1 = DB_CATEGORY
db2 = DB_FAMILY
db3 = DB_PLATFORM

# API keys
VT_API_KEY = Config.VT_API_KEY
api_key = VT_API_KEY

# Security
SECRET_KEY = Config.SECRET_KEY
CORS_ORIGINS = Config.CORS_ORIGINS

# DGA/ML settings
ALGORITHM_LIST = Config.ALGORITHM_LIST
CLASSIFIER_LIST = Config.CLASSIFIER_LIST


# ==========================================================================
# Main - Print config when run directly
# ==========================================================================

if __name__ == "__main__":
    Config.print_config()
