# -*- coding: utf-8 -*-
"""
DGA Detection Module

Multi-model detection system for Domain Generation Algorithm (DGA) domains.
Uses ensemble of ML classifiers to detect malicious domains.

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"
"""

import os
import warnings
from importlib import import_module

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import (
    MODEL_PATH, TRAIN_ADD, TEST_ADD,
    ALGORITHM_LIST, CLASSIFIER_LIST
)


class MultiModelDetection:
    """
    Multi-model DGA detection system.

    Uses ensemble of ML classifiers to detect malicious domains.
    Results are aggregated using a voting mechanism.
    """

    def __init__(self):
        """Initialize the multi-model detection system."""
        self._config = {
            "model_path": MODEL_PATH,
            "train_add": TRAIN_ADD,
            "test_add": TEST_ADD,
            "algorithm_list": ALGORITHM_LIST,
            "classifier_list": CLASSIFIER_LIST
        }
        self._classifiers = []
        self._load_models()

    def _load_models(self):
        """Load all trained models into memory."""
        for i, algorithm in enumerate(self._config["algorithm_list"]):
            module = import_module(f'feeds.{algorithm}')
            classifier_class = getattr(module, self._config["classifier_list"][i])
            classifier = classifier_class()
            classifier.load(self._config["model_path"])
            self._classifiers.append(classifier)

    def multi_predict_single_dname(self, domain_name):
        """
        Perform multi-model detection on a single domain name.

        Args:
            domain_name: The domain name to analyze

        Returns:
            Tuple of (base_results_dict, final_result)
            - base_results_dict: Results from each classifier
            - final_result: 0=safe, 1=malicious, 2=suspicious
        """
        base_result = {}
        base_result_temp = {}

        # Run prediction on all classifiers
        for i, classifier in enumerate(self._classifiers):
            prediction = classifier.predict_single_dname(
                self._config["model_path"], domain_name
            )
            classifier_name = self._config["classifier_list"][i][:-10]

            base_result[classifier_name] = [
                prediction[0],
                format(prediction[1], '.4f'),
                prediction[2]
            ]

            # Mark low-confidence predictions as suspicious
            if prediction[2] > 0.01:
                base_result_temp[classifier_name] = prediction
            else:
                base_result_temp[classifier_name] = (2, prediction[1], prediction[2])

        # Aggregate results
        result_list = [v[0] for v in base_result_temp.values()]

        # All classifiers agree
        if len(set(result_list)) == 1:
            first_result = list(base_result_temp.values())[0][0]
            if first_result != 2:
                return base_result, first_result
            else:
                # All models show low confidence - use best confidence
                sorted_results = sorted(
                    base_result_temp.items(),
                    key=lambda x: x[1][2],
                    reverse=True
                )
                if sorted_results[0][1][2] <= 0.5:
                    return base_result, 2
                else:
                    return base_result, sorted_results[0][1][0]

        # Classifiers disagree - filter out suspicious and use best confidence
        confident_results = {
            k: v for k, v in base_result_temp.items() if v[0] != 2
        }
        sorted_results = sorted(
            confident_results.items(),
            key=lambda x: x[1][2],
            reverse=True
        )

        if sorted_results[0][1][2] <= 0.5:
            final_result = 2
        else:
            final_result = sorted_results[0][1][0]

        return base_result, final_result


if __name__ == "__main__":
    from feeds.knn import KNNClassifier

    clf = KNNClassifier()
    clf.train("./data/model", "./data/features/train_features.csv")
