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
        
    def load_cicids_data(self, file_path):
        """Load and preprocess CICIDS2017 or NSL-KDD processed datasets"""
        try:
            print(f"Loading data from: {file_path}")
            df = pd.read_csv(file_path, low_memory=False)

            # Check if this is a processed dataset with expected columns
            if 'Attack Type' in df.columns:
                print(f"✓ Loaded {len(df)} samples from CICIDS/NSL-KDD processed dataset")
                print(f"  Features: {len(df.columns)} columns")

                # Show attack distribution
                if 'Attack Type' in df.columns:
                    print("\n  Attack Type Distribution:")
                    attack_counts = df['Attack Type'].value_counts()
                    for attack, count in attack_counts.head(10).items():
                        print(f"    {attack}: {count:,} ({count/len(df)*100:.2f}%)")
                    if len(attack_counts) > 10:
                        print(f"    ... and {len(attack_counts)-10} more types")

                return df
            else:
                print(f"⚠ Warning: No 'Attack Type' column found in {file_path}")
                print(f"  Available columns: {', '.join(df.columns[:5])}...")
                return None

        except FileNotFoundError:
            print(f"✗ File {file_path} not found.")
            return None
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return None

    def load_nsl_kdd_data(self, file_path):
        """Load and preprocess NSL-KDD dataset (kept for backward compatibility)"""
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
    
    def preprocess_data(self, df, sample_size=None):
        """Preprocess the dataset"""
        print("\n🔧 Preprocessing data...")

        # Sample data if dataset is too large (for faster training)
        if sample_size and len(df) > sample_size:
            print(f"  Sampling {sample_size:,} rows from {len(df):,} total...")
            # Stratified sampling to maintain attack/normal ratio
            if 'Attack Type' in df.columns:
                benign = df[df['Attack Type'] == 'BENIGN'].sample(n=min(int(sample_size*0.8), len(df[df['Attack Type'] == 'BENIGN'])), random_state=42)
                attack = df[df['Attack Type'] != 'BENIGN'].sample(n=min(int(sample_size*0.2), len(df[df['Attack Type'] != 'BENIGN'])), random_state=42)
                df = pd.concat([benign, attack], ignore_index=True).sample(frac=1, random_state=42)
            else:
                df = df.sample(n=sample_size, random_state=42)

        # Identify label columns to exclude from features
        label_columns = ['Attack Type', 'attack_type', 'anomaly_bool', 'Attack Number', ' Label']

        # Separate features and labels
        if 'Attack Type' in df.columns:
            y = df['Attack Type']
            # Convert to binary (BENIGN vs ATTACK)
            y_binary = (y != 'BENIGN').astype(int)
        elif 'attack_type' in df.columns:
            y = df['attack_type']
            y_binary = (y != 'normal').astype(int)
        else:
            print("  ⚠ Warning: No attack type column found, using mock labels")
            y = pd.Series(['normal'] * len(df))
            y_binary = pd.Series([0] * len(df))

        # Drop label columns from features
        X = df.drop(columns=[col for col in label_columns if col in df.columns], errors='ignore')

        # Handle non-numeric columns (encode categorical features)
        non_numeric_cols = X.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            print(f"  Encoding {len(non_numeric_cols)} categorical columns...")
            for col in non_numeric_cols:
                X[col] = self.label_encoder.fit_transform(X[col].astype(str))

        # Handle missing values
        if X.isnull().any().any():
            print(f"  Filling {X.isnull().sum().sum()} missing values...")
            X = X.fillna(0)

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)

        # Store feature names
        self.feature_names = X.columns.tolist()

        print(f"  ✓ Preprocessed: {len(X)} samples, {len(self.feature_names)} features")
        print(f"  ✓ Normal samples: {(y_binary==0).sum():,} ({(y_binary==0).sum()/len(y_binary)*100:.2f}%)")
        print(f"  ✓ Attack samples: {(y_binary==1).sum():,} ({(y_binary==1).sum()/len(y_binary)*100:.2f}%)")

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
    import argparse
    import sys

    print("="*60)
    print("Network Anomaly Detection - Model Training")
    print("="*60)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train network anomaly detection model')
    parser.add_argument('--dataset', type=str, default='auto',
                        help='Path to dataset CSV file, or "auto" to auto-detect')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'isolation_forest'],
                        help='Model type to train')
    parser.add_argument('--sample-size', type=int, default=100000,
                        help='Number of samples to use (default: 100000, use 0 for all data)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Test set split ratio (default: 0.2)')

    args = parser.parse_args()

    # Initialize detector
    detector = NetworkAnomalyDetector(model_type=args.model_type)

    # Auto-detect available datasets
    if args.dataset == 'auto':
        dataset_paths = [
            'data/CICIDS2017/SCA_processed.csv',
            'data/CICIDS2017/PCA_processed.csv',
            'data/NSL-KDD/SCA_processed.csv',
            '../data/CICIDS2017/SCA_processed.csv',
            '../data/CICIDS2017/PCA_processed.csv',
            '../data/NSL-KDD/SCA_processed.csv',
        ]

        dataset_path = None
        for path in dataset_paths:
            if os.path.exists(path):
                dataset_path = path
                print(f"\n✓ Auto-detected dataset: {path}")
                break

        if dataset_path is None:
            print("\n⚠ No dataset found in expected locations:")
            for path in dataset_paths[:3]:
                print(f"  - {path}")
            print("\nCreating mock data for demonstration...")
            df = detector._create_mock_data()
        else:
            df = detector.load_cicids_data(dataset_path)
    else:
        # Load specified dataset
        print(f"\n📊 Loading dataset from: {args.dataset}")
        if not os.path.exists(args.dataset):
            print(f"✗ File not found: {args.dataset}")
            print("Using mock data for demonstration...")
            df = detector._create_mock_data()
        else:
            df = detector.load_cicids_data(args.dataset)

    if df is None:
        print("✗ Failed to load dataset. Exiting.")
        sys.exit(1)

    # Preprocess
    sample_size = args.sample_size if args.sample_size > 0 else None
    X, _, y_binary = detector.preprocess_data(df, sample_size=sample_size)

    # Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=args.test_split, random_state=42, stratify=y_binary
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Attack ratio: {y_train.sum() / len(y_train):.2%}")

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
    print(f"   Model type: {args.model_type}")
    print(f"   Features: {len(detector.feature_names)}")
    print(f"   Training samples: {len(X_train):,}")
    print("="*60)


if __name__ == "__main__":
    main()
