"""
Unit tests for NetworkAnomalyDetector class
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
import joblib
from train_model import NetworkAnomalyDetector


class TestNetworkAnomalyDetector:
    """Test suite for NetworkAnomalyDetector"""

    def test_detector_initialization(self, detector):
        """Test that detector initializes with correct default values"""
        assert detector.svm1 is None
        assert detector.svm2 is None
        assert detector.rf1 is None
        assert detector.rf2 is None
        assert detector.models == {}
        assert detector.feature_names == []
        assert detector.model_type == 'ensemble'
        assert detector.loaded_model is None

    def test_train_binary_classifiers(self, detector, sample_binary_data, tmp_path):
        """Test binary classifier training"""
        csv_path = tmp_path / "binary_data.csv"
        sample_binary_data.to_csv(csv_path, index=False)

        svm1, svm2 = detector.train_binary_classifiers(str(csv_path))

        assert detector.svm1 is not None
        assert detector.svm2 is not None
        assert svm1 == detector.svm1
        assert svm2 == detector.svm2

        assert 'svm1' in detector.models
        assert 'svm2' in detector.models
        assert 'accuracy' in detector.models['svm1']
        assert 'cv_score' in detector.models['svm1']

        assert hasattr(detector.svm1, 'predict')
        assert hasattr(detector.svm2, 'predict')

    def test_predict_with_trained_model(self, trained_detector):
        """Test prediction with a trained model"""
        test_data = pd.DataFrame(
            np.random.randn(5, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )

        predictions, confidence_scores = trained_detector.predict(test_data)

        assert len(predictions) == 5
        assert len(confidence_scores) == 5
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= c <= 1 for c in confidence_scores)

    def test_predict_without_loaded_model(self, detector):
        """Test that prediction fails without a loaded model"""
        test_data = pd.DataFrame(np.random.randn(5, 10))

        with pytest.raises(ValueError, match="No model loaded"):
            detector.predict(test_data)

    def test_save_and_load_models(self, detector, sample_binary_data, tmp_path):
        """Test saving and loading models"""
        csv_path = tmp_path / "binary_data.csv"
        sample_binary_data.to_csv(csv_path, index=False)

        detector.train_binary_classifiers(str(csv_path))

        model_dir = tmp_path / "models"
        detector.save_models(str(model_dir))

        assert os.path.exists(model_dir / "svm_binary_model1.pkl")
        assert os.path.exists(model_dir / "svm_binary_model2.pkl")
        assert os.path.exists(model_dir / "metadata.json")

        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        assert 'models' in metadata
        assert 'svm1_accuracy' in metadata
        assert 'svm2_accuracy' in metadata

        new_detector = NetworkAnomalyDetector()
        new_detector.load_model(str(model_dir))

        assert new_detector.loaded_model is not None
        assert new_detector.svm1 is not None
        assert new_detector.svm2 is not None
        assert len(new_detector.feature_names) > 0

    def test_model_metadata_after_training(self, detector, sample_binary_data, tmp_path):
        """Test that model metadata is correctly stored"""
        csv_path = tmp_path / "binary_data.csv"
        sample_binary_data.to_csv(csv_path, index=False)

        detector.train_binary_classifiers(str(csv_path))

        assert 'svm1' in detector.models
        assert 'svm2' in detector.models

        svm1_info = detector.models['svm1']
        assert 'model' in svm1_info
        assert 'accuracy' in svm1_info
        assert 'cv_score' in svm1_info
        assert 'test_data' in svm1_info

        assert isinstance(svm1_info['accuracy'], float)
        assert isinstance(svm1_info['cv_score'], float)
        assert 0 <= svm1_info['accuracy'] <= 1
        assert 0 <= svm1_info['cv_score'] <= 1

    def test_load_nonexistent_model(self, detector, tmp_path):
        """Test that loading from nonexistent directory raises error"""
        nonexistent_dir = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            detector.load_model(str(nonexistent_dir))
