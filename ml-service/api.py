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

# Initialize detector and load model
detector = NetworkAnomalyDetector()

# Try to load existing model, train if not available
model_path = 'models'
if os.path.exists(f'{model_path}/anomaly_model.pkl'):
    detector.load_model(model_path)
    print("✓ Model loaded successfully")
else:
    print("⚠ No trained model found. Training new model...")
    # Train a new model with mock data
    from train_model import main as train_main
    train_main()
    detector.load_model(model_path)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_type': detector.model_type,
        'features': len(detector.feature_names)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if network traffic is anomalous
    Expected JSON format:
    {
        "duration": 0.5,
        "src_bytes": 450,
        "dst_bytes": 300,
        "count": 5,
        "srv_count": 3,
        "serror_rate": 0.1,
        "rerror_rate": 0.05,
        "same_srv_rate": 0.8,
        "diff_srv_rate": 0.2
    }
    """
    try:
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required features are present
        for feature in detector.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training
        df = df[detector.feature_names]
        
        # Make prediction
        predictions, scores = detector.predict(df)
        
        prediction = int(predictions[0])
        score = float(scores[0])
        
        # Determine threat level
        if prediction == 1:
            if score > 0.9:
                threat_level = 'critical'
            elif score > 0.7:
                threat_level = 'high'
            else:
                threat_level = 'medium'
        else:
            threat_level = 'normal'
        
        return jsonify({
            'is_anomaly': bool(prediction),
            'confidence': score,
            'threat_level': threat_level,
            'prediction': 'attack' if prediction == 1 else 'normal'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple network traffic samples
    Expected JSON format:
    {
        "samples": [
            {"duration": 0.5, "src_bytes": 450, ...},
            {"duration": 2.0, "src_bytes": 120, ...}
        ]
    }
    """
    try:
        data = request.json
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({'error': 'No samples provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(samples)
        
        # Ensure all required features are present
        for feature in detector.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training
        df = df[detector.feature_names]
        
        # Make predictions
        predictions, scores = detector.predict(df)
        
        results = []
        for pred, score in zip(predictions, scores):
            prediction = int(pred)
            confidence = float(score)
            
            if prediction == 1:
                if confidence > 0.9:
                    threat_level = 'critical'
                elif confidence > 0.7:
                    threat_level = 'high'
                else:
                    threat_level = 'medium'
            else:
                threat_level = 'normal'
            
            results.append({
                'is_anomaly': bool(prediction),
                'confidence': confidence,
                'threat_level': threat_level,
                'prediction': 'attack' if prediction == 1 else 'normal'
            })
        
        return jsonify({
            'results': results,
            'total': len(results),
            'anomalies_detected': sum(r['is_anomaly'] for r in results)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    return jsonify({
        'model_type': detector.model_type,
        'features': detector.feature_names,
        'feature_count': len(detector.feature_names)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
