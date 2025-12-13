"""
E2E tests for Flask API endpoints
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd


class TestAPIEndpoints:
    """Test suite for Flask API endpoints"""

    @patch('api.BINARY_DETECTOR')
    def test_health_endpoint(self, mock_detector, api_client):
        """Test /health endpoint returns correct status"""
        mock_detector.loaded_model = True

        response = api_client.get('/health')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['models_loaded'] is True

    @patch('api.BINARY_DETECTOR')
    def test_stats_endpoint(self, mock_detector, api_client):
        """Test /stats endpoint returns model information"""
        mock_detector.model_type = 'ensemble'
        mock_detector.feature_names = ['feature_0', 'feature_1', 'feature_2']

        response = api_client.get('/stats')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['model_type'] == 'ensemble'
        assert 'features' in data
        assert data['feature_count'] == 3

    @patch('api.BINARY_DETECTOR')
    @patch('api.MULTICLASS_DETECTOR')
    def test_predict_endpoint_normal_traffic(self, mock_multi, mock_binary, api_client):
        """Test /predict endpoint with normal (non-anomalous) traffic"""
        mock_binary.feature_names = ['feature_0', 'feature_1']
        mock_binary.predict = Mock(return_value=(np.array([0]), np.array([0.95])))

        payload = {
            'feature_0': 1.5,
            'feature_1': -0.5
        }

        response = api_client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['is_anomaly'] is False
        assert data['threat_level'] == 'normal'
        assert data['prediction'] == 'Normal'
        assert 'confidence' in data

    @patch('api.BINARY_DETECTOR')
    @patch('api.MULTICLASS_DETECTOR')
    def test_predict_endpoint_anomaly_critical(self, mock_multi, mock_binary, api_client):
        """Test /predict endpoint with critical threat level anomaly"""
        mock_binary.feature_names = ['feature_0', 'feature_1']
        mock_binary.predict = Mock(return_value=(np.array([1]), np.array([0.95])))

        mock_rf2 = Mock()
        mock_rf2.predict = Mock(return_value=np.array(['DDoS']))
        mock_multi.rf2 = mock_rf2

        payload = {
            'feature_0': 5.0,
            'feature_1': -3.2
        }

        response = api_client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['is_anomaly'] is True
        assert data['threat_level'] == 'critical'
        assert data['prediction'] == 'DDoS'
        assert data['confidence'] == 0.95

    @patch('api.BINARY_DETECTOR')
    @patch('api.MULTICLASS_DETECTOR')
    def test_predict_endpoint_anomaly_high(self, mock_multi, mock_binary, api_client):
        """Test /predict endpoint with high threat level anomaly"""
        mock_binary.feature_names = ['feature_0']
        mock_binary.predict = Mock(return_value=(np.array([1]), np.array([0.85])))

        mock_rf2 = Mock()
        mock_rf2.predict = Mock(return_value=np.array(['PortScan']))
        mock_multi.rf2 = mock_rf2

        payload = {'feature_0': 2.5}

        response = api_client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['is_anomaly'] is True
        assert data['threat_level'] == 'high'
        assert data['prediction'] == 'PortScan'

    @patch('api.BINARY_DETECTOR')
    @patch('api.MULTICLASS_DETECTOR')
    def test_predict_batch_endpoint(self, mock_multi, mock_binary, api_client):
        """Test /predict/batch endpoint with multiple samples"""
        mock_binary.feature_names = ['feature_0', 'feature_1']
        mock_binary.predict = Mock(return_value=(
            np.array([0, 1, 1]),
            np.array([0.80, 0.95, 0.65])
        ))

        mock_rf2 = Mock()
        mock_rf2.predict = Mock(return_value=np.array(['Normal', 'DDoS', 'BruteForce']))
        mock_multi.rf2 = mock_rf2

        payload = {
            'samples': [
                {'feature_0': 1.0, 'feature_1': 0.5},
                {'feature_0': 5.0, 'feature_1': -2.0},
                {'feature_0': 3.0, 'feature_1': -1.5}
            ]
        }

        response = api_client.post(
            '/predict/batch',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'results' in data
        assert data['total'] == 3
        assert data['anomalies_detected'] == 2

        results = data['results']
        assert results[0]['is_anomaly'] is False
        assert results[0]['threat_level'] == 'normal'
        assert results[1]['is_anomaly'] is True
        assert results[1]['threat_level'] == 'critical'
        assert results[2]['is_anomaly'] is True
        assert results[2]['threat_level'] == 'medium'

    @patch('api.BINARY_DETECTOR')
    def test_predict_batch_empty_samples(self, mock_binary, api_client):
        """Test /predict/batch endpoint with empty samples list"""
        payload = {'samples': []}

        response = api_client.post(
            '/predict/batch',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'No samples provided'

    @patch('api.load_traffic_samples')
    def test_simulate_endpoint_normal(self, mock_load, api_client):
        """Test /simulate endpoint returns simulated traffic"""
        normal_df = pd.DataFrame({
            'feature_0': [1.0, 2.0],
            'feature_1': [0.5, 1.0],
            'Attack Type': [0, 0]
        })
        attack_df = pd.DataFrame({
            'feature_0': [5.0],
            'feature_1': [-2.0],
            'Attack Type': [1]
        })

        mock_load.return_value = (normal_df, attack_df)

        response = api_client.get('/simulate')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'sample' in data
        assert 'is_attack' in data
        assert isinstance(data['is_attack'], bool)
        assert 'count' in data['sample']

    @patch('api.BINARY_DETECTOR')
    def test_predict_endpoint_missing_features(self, mock_binary, api_client):
        """Test /predict endpoint handles missing features gracefully"""
        mock_binary.feature_names = ['feature_0', 'feature_1', 'feature_2']
        mock_binary.predict = Mock(return_value=(np.array([0]), np.array([0.85])))

        payload = {'feature_0': 1.5}

        response = api_client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200

        call_args = mock_binary.predict.call_args[0][0]
        assert 'feature_1' in call_args.columns
        assert 'feature_2' in call_args.columns
        assert call_args['feature_1'].iloc[0] == 0
        assert call_args['feature_2'].iloc[0] == 0
