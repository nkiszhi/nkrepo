# -*- coding: utf-8 -*-
"""
NKREPO Multi-Scan Module

Provides malware detection capabilities using multiple ML/DL models:
- File-based malware detection (PE analysis)
- DGA (Domain Generation Algorithm) detection
- Ensemble prediction combining multiple models

Note: Uses lazy imports to avoid loading ML/DL dependencies at module import time.
"""

__all__ = [
    'run_ensemble_prediction',
    'EXEDetection',
    'DGADetection',
]


def __getattr__(name):
    """Lazy import module members to avoid loading ML models at import time."""
    if name == 'run_ensemble_prediction':
        from .ensemble_predict import run_ensemble_prediction
        return run_ensemble_prediction
    elif name == 'EXEDetection':
        from .file_detect import EXEDetection
        return EXEDetection
    elif name == 'DGADetection':
        from .dga_detection import DGADetection
        return DGADetection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
