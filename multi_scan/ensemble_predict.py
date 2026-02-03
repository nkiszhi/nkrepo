# -*- coding: utf-8 -*-
"""
Ensemble Prediction Module

Combines predictions from multiple ML/DL models for malware detection.
Supports 10 different deep learning models for PE file analysis.
"""

import os
import sys
from pathlib import Path

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

# Add multi_scan directory to path for model imports
sys.path.insert(0, str(MULTI_SCAN_DIR))

# =============================================================================
# Model Imports
# =============================================================================

from models.m_2017_malconv.exec_malconv import run_prediction as malconv_predict
from models.m_2017_transformer.exec_transformer import run_prediction as transformer_predict
from models.m_2018_ember.exec_ember import run_prediction as ember_predict
from models.m_2019_1d_cnn.exec_1d_cnn import run_prediction as cnn_1d_predict
from models.m_2020_inceptionv3.exec_InceptionV3 import run_prediction as inceptionv3_predict
from models.m_2021_malconv2.exec_malconv2 import run_prediction as malconv2_predict
from models.m_2021_rcnf.exec_rcnf import run_prediction as rcnf_predict
from models.m_attention_rcnn.attention_rcnn import run_prediction as attention_rcnn_predict
from models.m_rcnn.rcnn import run_prediction as rcnn_predict
from models.m_vgg16.vgg16 import run_prediction as vgg16_predict

# =============================================================================
# Model Registry
# =============================================================================

MODELS = [
    ('MalConv', malconv_predict),
    ('Transformer', transformer_predict),
    ('EMBER', ember_predict),
    ('1D-CNN', cnn_1d_predict),
    ('InceptionV3', inceptionv3_predict),
    ('MalConv2', malconv2_predict),
    ('RCNF', rcnf_predict),
    ('Attention-RCNN', attention_rcnn_predict),
    ('RCNN', rcnn_predict),
    ('VGG16', vgg16_predict)
]

# Result labels
LABEL_MALICIOUS = '恶意'
LABEL_SAFE = '安全'
LABEL_PREDICTION_FAILED = '预测失败'
LABEL_NO_VALID_PREDICTION = '无有效预测'
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

    for model_name, predict_func in MODELS:
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
    print(f"\nRegistered Models: {len(MODELS)}")
    for name, _ in MODELS:
        print(f"  - {name}")
    print("=" * 60)
