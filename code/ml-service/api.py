"""
Flask API for Network Anomaly Detection ML Service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from train_model import NetworkAnomalyDetector
import os

app = Flask(__name__)
CORS(app)

BINARY_DETECTOR = NetworkAnomalyDetector()
MULTICLASS_DETECTOR = NetworkAnomalyDetector()

MODEL_PATH = 'models'
DATA_PATH = '../data/CICIDS2017/PCA_balanced.csv'

if os.path.exists(f'{MODEL_PATH}/metadata.json'):
    BINARY_DETECTOR.load_model(MODEL_PATH)
    MULTICLASS_DETECTOR.load_model(MODEL_PATH)
    print("Models loaded successfully")
else:
    print("No trained model found. Training new models...")
    from train_model import main as train_main
    train_main()
    BINARY_DETECTOR.load_model(MODEL_PATH)
    MULTICLASS_DETECTOR.load_model(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if network traffic is anomalous and classify attack type
    """
    try:
        data = request.json
        df = pd.DataFrame([data])

        for feature in BINARY_DETECTOR.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        df = df[BINARY_DETECTOR.feature_names]

        binary_predictions, binary_scores = BINARY_DETECTOR.predict(df)
        is_anomaly = int(binary_predictions[0])
        confidence = float(binary_scores[0])

        if is_anomaly == 1:
            if MULTICLASS_DETECTOR.rf2:
                multiclass_predictions, _ = MULTICLASS_DETECTOR.rf2.predict(df), None
                attack_type = str(multiclass_predictions[0])
            else:
                attack_type = 'Unknown Attack'

            if confidence > 0.9:
                threat_level = 'critical'
            elif confidence > 0.7:
                threat_level = 'high'
            else:
                threat_level = 'medium'

            prediction_label = attack_type
        else:
            threat_level = 'normal'
            prediction_label = 'Normal'

        return jsonify({
            'is_anomaly': bool(is_anomaly),
            'confidence': confidence,
            'threat_level': threat_level,
            'prediction': prediction_label
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple network traffic samples
    """
    try:
        data = request.json
        samples = data.get('samples', [])

        if not samples:
            return jsonify({'error': 'No samples provided'}), 400

        df = pd.DataFrame(samples)

        for feature in BINARY_DETECTOR.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        df = df[BINARY_DETECTOR.feature_names]

        binary_predictions, binary_scores = BINARY_DETECTOR.predict(df)

        if MULTICLASS_DETECTOR.rf2:
            multiclass_predictions, _ = MULTICLASS_DETECTOR.rf2.predict(df), None
        else:
            multiclass_predictions = ['Unknown Attack'] * len(df)

        results = []
        for i, (is_anomaly, confidence) in enumerate(zip(binary_predictions, binary_scores)):
            is_anomaly = int(is_anomaly)
            confidence = float(confidence)

            if is_anomaly == 1:
                attack_type = str(multiclass_predictions[i])
                if confidence > 0.9:
                    threat_level = 'critical'
                elif confidence > 0.7:
                    threat_level = 'high'
                else:
                    threat_level = 'medium'
                prediction_label = attack_type
            else:
                threat_level = 'normal'
                prediction_label = 'Normal'

            results.append({
                'is_anomaly': bool(is_anomaly),
                'confidence': confidence,
                'threat_level': threat_level,
                'prediction': prediction_label
            })

        return jsonify({
            'results': results,
            'total': len(results),
            'anomalies_detected': sum(r['is_anomaly'] for r in results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': bool(BINARY_DETECTOR.loaded_model)})


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    return jsonify({
        'model_type': BINARY_DETECTOR.model_type,
        'features': BINARY_DETECTOR.feature_names,
        'feature_count': len(BINARY_DETECTOR.feature_names)
    })


NORMAL_SAMPLES = None
ATTACK_SAMPLES = None

def load_traffic_samples():
    """Load and separate normal and attack traffic samples"""
    global NORMAL_SAMPLES, ATTACK_SAMPLES

    if NORMAL_SAMPLES is None or ATTACK_SAMPLES is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        NORMAL_SAMPLES = df[df['Attack Type'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
        ATTACK_SAMPLES = df[df['Attack Type'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Loaded {len(NORMAL_SAMPLES)} normal and {len(ATTACK_SAMPLES)} attack samples")

    return NORMAL_SAMPLES, ATTACK_SAMPLES


@app.route('/simulate', methods=['GET'])
def simulate_traffic():
    """
    Simulate realistic network traffic with 85% normal, 15% anomalies
    """
    normal_samples, attack_samples = load_traffic_samples()

    if normal_samples.empty and attack_samples.empty:
        return jsonify({'error': 'No dataset available'}), 500

    is_attack = np.random.random() > 0.85

    if is_attack and not attack_samples.empty:
        sample = attack_samples.sample(n=1, random_state=None).iloc[0]
        connection_volume = int(np.random.uniform(50, 200))
    elif not normal_samples.empty:
        sample = normal_samples.sample(n=1, random_state=None).iloc[0]
        connection_volume = int(np.random.uniform(10, 80))
    else:
        sample = attack_samples.sample(n=1, random_state=None).iloc[0]
        connection_volume = int(np.random.uniform(50, 200))

    sample_dict = sample.drop('Attack Type').to_dict()
    sample_dict['count'] = connection_volume
    actual_is_attack = int(sample['Attack Type'])

    return jsonify({
        'sample': sample_dict,
        'is_attack': bool(actual_is_attack)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
