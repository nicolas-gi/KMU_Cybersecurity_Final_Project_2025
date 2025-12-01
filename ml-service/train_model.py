"""
Machine Learning Model Training for Network Anomaly Detection
Supports CICIDS2017 and NSL-KDD datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os

class NetworkAnomalyDetector:
    def __init__(self, model_type='random_forest'):
        """
        Initialize the anomaly detector
        model_type: 'random_forest' or 'isolation_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def load_nsl_kdd_data(self, file_path):
        """Load and preprocess NSL-KDD dataset"""
        # NSL-KDD column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'attack_type', 'difficulty'
        ]
        
        try:
            df = pd.read_csv(file_path, names=columns)
            return df
        except FileNotFoundError:
            print(f"File {file_path} not found. Using mock data for demonstration.")
            return self._create_mock_data()
    
    def _create_mock_data(self):
        """Create mock network traffic data for demonstration"""
        np.random.seed(42)
        n_samples = 5000
        
        # Normal traffic
        normal = pd.DataFrame({
            'duration': np.random.exponential(2, n_samples // 2),
            'src_bytes': np.random.normal(500, 200, n_samples // 2),
            'dst_bytes': np.random.normal(300, 150, n_samples // 2),
            'count': np.random.poisson(5, n_samples // 2),
            'srv_count': np.random.poisson(3, n_samples // 2),
            'serror_rate': np.random.beta(1, 10, n_samples // 2),
            'rerror_rate': np.random.beta(1, 10, n_samples // 2),
            'same_srv_rate': np.random.beta(8, 2, n_samples // 2),
            'diff_srv_rate': np.random.beta(2, 8, n_samples // 2),
            'attack_type': ['normal'] * (n_samples // 2)
        })
        
        # Attack traffic (anomalies)
        attacks = []
        attack_types = ['dos', 'probe', 'r2l', 'u2r']
        
        for attack in attack_types:
            n_attack = n_samples // 8
            if attack == 'dos':
                # DDoS: high packet count, low duration
                attack_df = pd.DataFrame({
                    'duration': np.random.exponential(0.1, n_attack),
                    'src_bytes': np.random.normal(100, 50, n_attack),
                    'dst_bytes': np.random.normal(50, 30, n_attack),
                    'count': np.random.poisson(50, n_attack),
                    'srv_count': np.random.poisson(40, n_attack),
                    'serror_rate': np.random.beta(8, 2, n_attack),
                    'rerror_rate': np.random.beta(1, 10, n_attack),
                    'same_srv_rate': np.random.beta(2, 8, n_attack),
                    'diff_srv_rate': np.random.beta(8, 2, n_attack),
                    'attack_type': [attack] * n_attack
                })
            elif attack == 'probe':
                # Port scan: many connections, low data
                attack_df = pd.DataFrame({
                    'duration': np.random.exponential(1, n_attack),
                    'src_bytes': np.random.normal(50, 20, n_attack),
                    'dst_bytes': np.random.normal(20, 10, n_attack),
                    'count': np.random.poisson(100, n_attack),
                    'srv_count': np.random.poisson(80, n_attack),
                    'serror_rate': np.random.beta(5, 5, n_attack),
                    'rerror_rate': np.random.beta(5, 5, n_attack),
                    'same_srv_rate': np.random.beta(1, 10, n_attack),
                    'diff_srv_rate': np.random.beta(10, 1, n_attack),
                    'attack_type': [attack] * n_attack
                })
            else:
                # Other attacks
                attack_df = pd.DataFrame({
                    'duration': np.random.exponential(5, n_attack),
                    'src_bytes': np.random.normal(1000, 500, n_attack),
                    'dst_bytes': np.random.normal(800, 400, n_attack),
                    'count': np.random.poisson(10, n_attack),
                    'srv_count': np.random.poisson(8, n_attack),
                    'serror_rate': np.random.beta(3, 7, n_attack),
                    'rerror_rate': np.random.beta(3, 7, n_attack),
                    'same_srv_rate': np.random.beta(5, 5, n_attack),
                    'diff_srv_rate': np.random.beta(5, 5, n_attack),
                    'attack_type': [attack] * n_attack
                })
            attacks.append(attack_df)
        
        return pd.concat([normal] + attacks, ignore_index=True)
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # For mock data, we already have numeric features
        # In real scenario, would encode categorical features
        
        # Separate features and labels
        X = df.drop('attack_type', axis=1)
        y = df['attack_type']
        
        # Convert attack types to binary (normal vs attack)
        y_binary = (y != 'normal').astype(int)
        
        self.feature_names = X.columns.tolist()
        
        return X, y, y_binary
    
    def train(self, X_train, y_train):
        """Train the anomaly detection model"""
        # Scale features
        x_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(x_train_scaled, y_train)
        elif self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(x_train_scaled)
        
        print(f"✓ Model trained: {self.model_type}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        x_test_scaled = self.scaler.transform(X_test)
        
        if self.model_type == 'random_forest':
            y_pred = self.model.predict(x_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        else:
            y_pred = self.model.predict(x_test_scaled)
            y_pred_binary = (y_pred == -1).astype(int)
            accuracy = accuracy_score(y_test, y_pred_binary)
            print(f"\nAccuracy: {accuracy:.4f}")
    
    def predict(self, X):
        """Predict if traffic is anomalous"""
        x_scaled = self.scaler.transform(X)
        
        if self.model_type == 'random_forest':
            predictions = self.model.predict(x_scaled)
            probabilities = self.model.predict_proba(x_scaled)[:, 1]
            return predictions, probabilities
        else:
            predictions = self.model.predict(x_scaled)
            # Convert -1 (anomaly) to 1, and 1 (normal) to 0
            predictions_binary = (predictions == -1).astype(int)
            scores = self.model.score_samples(x_scaled)
            return predictions_binary, scores
    
    def save_model(self, model_dir='models'):
        """Save trained model and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/anomaly_model.pkl')
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        
        # Save feature names and metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        with open(f'{model_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        print(f"\n✓ Model saved to {model_dir}/")
    
    def load_model(self, model_dir='models'):
        """Load trained model and scaler"""
        self.model = joblib.load(f'{model_dir}/anomaly_model.pkl')
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        
        with open(f'{model_dir}/metadata.json', 'r') as f:
            metadata = json.load(f)
            self.model_type = metadata['model_type']
            self.feature_names = metadata['feature_names']
        
        print(f"✓ Model loaded from {model_dir}/")


def main():
    print("="*60)
    print("Network Anomaly Detection - Model Training")
    print("="*60)
    
    # Initialize detector
    detector = NetworkAnomalyDetector(model_type='random_forest')
    
    # Load data (will use mock data if dataset not found)
    print("\n📊 Loading dataset...")
    df = detector.load_nsl_kdd_data('data/KDDTrain+.txt')
    print(f"Dataset loaded: {len(df)} samples")
    
    # Preprocess
    print("\n🔧 Preprocessing data...")
    X, _, y_binary = detector.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Attack ratio in training: {y_train.sum() / len(y_train):.2%}")
    
    # Train
    print("\n🤖 Training model...")
    detector.train(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    detector.evaluate(X_test, y_test)
    
    # Save
    print("\n💾 Saving model...")
    detector.save_model('models')
    
    print("\n✅ Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
