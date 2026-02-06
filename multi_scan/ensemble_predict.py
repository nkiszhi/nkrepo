# -*- coding: utf-8 -*-
"""
Ensemble Prediction Module

Combines predictions from multiple ML/DL models for malware detection.
Supports 10 different deep learning models for PE file analysis.

Uses lazy loading to avoid import errors at module load time.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

# Multi-scan directory (where this file is located)
MULTI_SCAN_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = MULTI_SCAN_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = MULTI_SCAN_DIR / "models"

# Feature files for training/testing
TRAIN_FEATURES = FEATURES_DIR / "train_features.csv"
TEST_FEATURES = FEATURES_DIR / "test_features.csv"

# =============================================================================
# Lazy Model Loading
# =============================================================================

# Cache for loaded model prediction functions
_model_cache = {}

# Model configuration: (name, module_path, function_name)
MODEL_CONFIG = [
    ('MalConv', 'models.m_2017_malconv.exec_malconv', 'run_prediction'),
    ('Transformer', 'models.m_2017_transformer.exec_transformer', 'run_prediction'),
    ('EMBER', 'models.m_2018_ember.exec_ember', 'run_prediction'),
    ('1D-CNN', 'models.m_2019_1d_cnn.exec_1d_cnn', 'run_prediction'),
    ('InceptionV3', 'models.m_2020_inceptionv3.exec_InceptionV3', 'run_prediction'),
    ('MalConv2', 'models.m_2021_malconv2.exec_malconv2', 'run_prediction'),
    ('RCNF', 'models.m_2021_rcnf.exec_rcnf', 'run_prediction'),
    ('Attention-RCNN', 'models.m_attention_rcnn.attention_rcnn', 'run_prediction'),
    ('RCNN', 'models.m_rcnn.rcnn', 'run_prediction'),
    ('VGG16', 'models.m_vgg16.vgg16', 'run_prediction'),
]


def _ensure_path():
    """Ensure multi_scan directory is in sys.path for model imports."""
    path_str = str(MULTI_SCAN_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_model(model_name, module_path, func_name):
    """
    Lazily load a model's prediction function.

    Args:
        model_name: Display name for the model
        module_path: Import path for the module
        func_name: Name of the prediction function

    Returns:
        The prediction function, or None if loading failed
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    _ensure_path()

    try:
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        _model_cache[model_name] = func
        logger.debug(f"Successfully loaded model: {model_name}")
        return func
    except Exception as e:
        logger.warning(f"Failed to load model {model_name}: {e}")
        _model_cache[model_name] = None
        return None


def get_available_models():
    """
    Get list of models that can be loaded successfully.

    Returns:
        List of (model_name, predict_function) tuples for available models
    """
    available = []
    for name, module_path, func_name in MODEL_CONFIG:
        func = _load_model(name, module_path, func_name)
        if func is not None:
            available.append((name, func))
    return available

# =============================================================================
# Result Labels
# =============================================================================

LABEL_MALICIOUS = '恶意'
LABEL_SAFE = '安全'
LABEL_PREDICTION_FAILED = '预测失败'
LABEL_NO_VALID_PREDICTION = '无有效预测'
LABEL_MODEL_NOT_LOADED = '模型未加载'
LABEL_ENSEMBLE = '集成结果'


# =============================================================================
# Prediction Functions
# =============================================================================

def ensemble_prediction(file_path):
    """
    Run ensemble prediction using all available models.

    Args:
        file_path: Path to the PE file to analyze

    Returns:
        Dictionary containing results from each model and ensemble result
    """
    results = {}
    malicious_probs = []
    safe_probs = []

    for model_name, module_path, func_name in MODEL_CONFIG:
        predict_func = _load_model(model_name, module_path, func_name)

        if predict_func is None:
            results[model_name] = {
                "probability": None,
                "result": LABEL_MODEL_NOT_LOADED
            }
            continue

        try:
            score = predict_func(file_path)
            if score is not None:
                score = float(score)
                result = LABEL_MALICIOUS if score > 0.5 else LABEL_SAFE
                results[model_name] = {
                    "probability": round(score, 4),
                    "result": result
                }
                if result == LABEL_MALICIOUS:
                    malicious_probs.append(score)
                else:
                    safe_probs.append(score)
            else:
                results[model_name] = {
                    "probability": None,
                    "result": LABEL_PREDICTION_FAILED
                }
        except Exception as e:
            logger.error(f"Error running {model_name}: {e}")
            results[model_name] = {
                "probability": None,
                "result": f"错误: {str(e)[:50]}"
            }

    # Ensemble decision: majority voting with probability weighting
    if malicious_probs and len(malicious_probs) > len(safe_probs):
        ensemble_prob = max(malicious_probs)
        ensemble_result = LABEL_MALICIOUS
    elif safe_probs:
        ensemble_prob = min(safe_probs)
        ensemble_result = LABEL_SAFE
    else:
        ensemble_prob = None
        ensemble_result = LABEL_NO_VALID_PREDICTION

    results[LABEL_ENSEMBLE] = {
        "probability": ensemble_prob,
        "result": ensemble_result
    }
    return results


def run_ensemble_prediction(file_path: str) -> dict:
    """
    External API for ensemble prediction.

    Args:
        file_path: Path to the PE file to analyze

    Returns:
        Dictionary containing prediction results or error message
    """
    if not os.path.isfile(file_path):
        return {"error": "文件不存在"}
    return ensemble_prediction(file_path)


def get_feature_paths() -> dict:
    """
    Get paths to feature files.

    Returns:
        Dictionary with train and test feature file paths
    """
    return {
        "train_features": str(TRAIN_FEATURES),
        "test_features": str(TEST_FEATURES),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR)
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Print configuration
    print("=" * 60)
    print("Ensemble Prediction Configuration")
    print("=" * 60)
    print(f"Multi-Scan Dir:   {MULTI_SCAN_DIR}")
    print(f"Data Dir:         {DATA_DIR}")
    print(f"Features Dir:     {FEATURES_DIR}")
    print(f"Models Dir:       {MODELS_DIR}")
    print(f"Train Features:   {TRAIN_FEATURES}")
    print(f"Test Features:    {TEST_FEATURES}")
    print(f"\nConfigured Models: {len(MODEL_CONFIG)}")
    for name, module_path, _ in MODEL_CONFIG:
        print(f"  - {name} ({module_path})")

    print("\nTesting model loading...")
    available = get_available_models()
    print(f"Successfully loaded: {len(available)}/{len(MODEL_CONFIG)} models")
    for name, _ in available:
        print(f"  ✓ {name}")

    failed = set(m[0] for m in MODEL_CONFIG) - set(m[0] for m in available)
    if failed:
        print(f"Failed to load:")
        for name in failed:
            print(f"  ✗ {name}")
    print("=" * 60)
