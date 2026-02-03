# -*- coding: utf-8 -*-
"""
NKREPO Multi-Scan Module

Provides malware detection capabilities using multiple ML/DL models:
- File-based malware detection (PE analysis)
- DGA (Domain Generation Algorithm) detection
- Ensemble prediction combining multiple models
"""

from .ensemble_predict import run_ensemble_prediction
from .file_detect import EXEDetection
from .dga_detection import DGADetection

__all__ = [
    'run_ensemble_prediction',
    'EXEDetection',
    'DGADetection',
]
