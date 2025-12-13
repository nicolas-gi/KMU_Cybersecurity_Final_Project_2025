"""
Pytest configuration and fixtures for ML service tests
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_model import NetworkAnomalyDetector


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), name='Attack Type')

    data = pd.concat([X, y], axis=1)
    return data


@pytest.fixture
def sample_multiclass_data():
    """Generate sample multiclass classification data"""
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    n_classes = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    attack_types = ['Normal', 'DDoS', 'PortScan', 'BruteForce', 'Infiltration']
    y = pd.Series(
        [attack_types[i % n_classes] for i in range(n_samples)],
        name='Attack Type'
    )

    data = pd.concat([X, y], axis=1)
    return data


@pytest.fixture
def detector():
    """Create a fresh NetworkAnomalyDetector instance"""
    return NetworkAnomalyDetector()


@pytest.fixture
def trained_detector(sample_binary_data, tmp_path):
    """Create a detector with a trained model"""
    detector = NetworkAnomalyDetector()

    csv_path = tmp_path / "test_data.csv"
    sample_binary_data.to_csv(csv_path, index=False)

    X = sample_binary_data.drop('Attack Type', axis=1)
    y = sample_binary_data['Attack Type']

    detector.svm1 = SVC(kernel='poly', C=1, random_state=0, probability=True)
    detector.svm1.fit(X, y)
    detector.loaded_model = detector.svm1
    detector.feature_names = X.columns.tolist()

    return detector


@pytest.fixture
def api_client():
    """Create a Flask test client"""
    import api
    api.app.config['TESTING'] = True

    with api.app.test_client() as client:
        yield client


@pytest.fixture
def mock_detector(monkeypatch):
    """Mock the detector for API tests"""
    class MockDetector:
        def __init__(self):
            self.loaded_model = True
            self.model_type = 'ensemble'
            self.feature_names = [f'feature_{i}' for i in range(10)]
            self.rf2 = MockRF()

        def predict(self, X):
            predictions = np.ones(len(X))
            confidence_scores = np.array([0.95] * len(X))
            return predictions, confidence_scores

    class MockRF:
        def predict(self, X):
            return np.array(['DDoS'] * len(X))

    mock = MockDetector()
    return mock
