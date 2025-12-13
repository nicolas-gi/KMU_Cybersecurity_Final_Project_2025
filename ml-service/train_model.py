import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os


class NetworkAnomalyDetector:
    def __init__(self):
        self.svm1 = None
        self.svm2 = None
        self.rf1 = None
        self.rf2 = None
        self.models = {}
        self.feature_names = []
        self.model_type = 'ensemble'
        self.loaded_model = None

    def train_binary_classifiers(self, data_path='../data/CICIDS2017/PCA_balanced.csv'):
        d1 = pd.read_csv(data_path)
        X_bc = d1.drop('Attack Type', axis=1)
        y_bc = d1['Attack Type']

        X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
            X_bc, y_bc, test_size=0.2, random_state=0
        )

        self.svm1 = SVC(kernel='poly', C=1, random_state=0, probability=True)
        self.svm1.fit(X_train_bc, y_train_bc)

        cv_svm1 = cross_val_score(self.svm1, X_train_bc, y_train_bc, cv=5)
        print('Support Vector Machine Model 1')
        print(f'Cross-validation scores: {", ".join(map(str, cv_svm1))}')
        print(f'Mean cross-validation score: {cv_svm1.mean():.2f}')
        print(f'SVM Model 1 intercept: {self.svm1.intercept_[0]}')

        self.svm2 = SVC(kernel='rbf', C=1, gamma=0.1, random_state=0, probability=True)
        self.svm2.fit(X_train_bc, y_train_bc)

        cv_svm2 = cross_val_score(self.svm2, X_train_bc, y_train_bc, cv=5)
        print('\nSupport Vector Machine Model 2')
        print(f'Cross-validation scores: {", ".join(map(str, cv_svm2))}')
        print(f'Mean cross-validation score: {cv_svm2.mean():.2f}')
        print(f'SVM Model 2 intercept: {self.svm2.intercept_[0]}')

        y_pred_svm1 = self.svm1.predict(X_test_bc)
        acc1 = accuracy_score(y_test_bc, y_pred_svm1)
        print(f'\nAccuracy SVM Binary Classifier Model 1: {acc1:.4f}')

        y_pred_svm2 = self.svm2.predict(X_test_bc)
        acc2 = accuracy_score(y_test_bc, y_pred_svm2)
        print(f'Accuracy SVM Binary Classifier Model 2: {acc2:.4f}')

        self.models['svm1'] = {
            'model': self.svm1,
            'accuracy': acc1,
            'cv_score': cv_svm1.mean(),
            'test_data': (X_test_bc, y_test_bc)
        }
        self.models['svm2'] = {
            'model': self.svm2,
            'accuracy': acc2,
            'cv_score': cv_svm2.mean(),
            'test_data': (X_test_bc, y_test_bc)
        }

        return self.svm1, self.svm2

    def train_multiclass_classifiers(self, data_path='../data/CICIDS2017/PCA_processed.csv'):
        d2 = pd.read_csv(data_path)

        class_counts = d2['Attack Type'].value_counts()
        selected_classes = class_counts[class_counts > 1950]
        class_names = selected_classes.index
        selected = d2[d2['Attack Type'].isin(class_names)]

        dfs = []
        for name in class_names:
            df = selected[selected['Attack Type'] == name]
            if len(df) > 2500:
                df = df.sample(n=5000, random_state=0)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        X = df.drop('Attack Type', axis=1)
        y = df['Attack Type']

        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(sampling_strategy='auto', random_state=0)
            X_upsampled, y_upsampled = smote.fit_resample(X, y)
        except (ImportError, Exception):
            max_count = y.value_counts().max()
            balanced_dfs = []
            for attack_type in y.unique():
                class_df = df[df['Attack Type'] == attack_type]
                if len(class_df) < max_count:
                    class_df = class_df.sample(n=max_count, replace=True, random_state=0)
                balanced_dfs.append(class_df)
            df = pd.concat(balanced_dfs, ignore_index=True)
            X_upsampled = df.drop('Attack Type', axis=1).values
            y_upsampled = df['Attack Type'].values

        blnc_data = pd.DataFrame(X_upsampled)
        blnc_data['Attack Type'] = y_upsampled
        blnc_data = blnc_data.sample(frac=1, random_state=0)

        features = blnc_data.drop('Attack Type', axis=1)
        labels = blnc_data['Attack Type']

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=0
        )

        self.rf1 = RandomForestClassifier(
            n_estimators=10,
            max_depth=6,
            max_features=None,
            random_state=0
        )
        self.rf1.fit(X_train, y_train)

        cv_rf1 = cross_val_score(self.rf1, X_train, y_train, cv=5)
        print('\nRandom Forest Model 1')
        print(f'Cross-validation scores: {", ".join(map(str, cv_rf1))}')
        print(f'Mean cross-validation score: {cv_rf1.mean():.2f}')

        self.rf2 = RandomForestClassifier(
            n_estimators=15,
            max_depth=8,
            max_features=20,
            random_state=0
        )
        self.rf2.fit(X_train, y_train)

        cv_rf2 = cross_val_score(self.rf2, X_train, y_train, cv=5)
        print('\nRandom Forest Model 2')
        print(f'Cross-validation scores: {", ".join(map(str, cv_rf2))}')
        print(f'Mean cross-validation score: {cv_rf2.mean():.2f}')

        y_pred_rf1 = self.rf1.predict(X_test)
        acc1 = accuracy_score(y_pred_rf1, y_test)
        print(f'\nAccuracy RF Model 1: {acc1:.4f}')

        y_pred_rf2 = self.rf2.predict(X_test)
        acc2 = accuracy_score(y_pred_rf2, y_test)
        print(f'Accuracy RF Model 2: {acc2:.4f}')

        self.models['rf1'] = {
            'model': self.rf1,
            'accuracy': acc1,
            'cv_score': cv_rf1.mean(),
            'test_data': (X_test, y_test)
        }
        self.models['rf2'] = {
            'model': self.rf2,
            'accuracy': acc2,
            'cv_score': cv_rf2.mean(),
            'test_data': (X_test, y_test)
        }

        return self.rf1, self.rf2

    def save_models(self, model_dir='ml-service/models'):
        os.makedirs(model_dir, exist_ok=True)

        if self.svm1:
            joblib.dump(self.svm1, f'{model_dir}/svm_binary_model1.pkl')
        if self.svm2:
            joblib.dump(self.svm2, f'{model_dir}/svm_binary_model2.pkl')
        if self.rf1:
            joblib.dump(self.rf1, f'{model_dir}/rf_multiclass_model1.pkl')
        if self.rf2:
            joblib.dump(self.rf2, f'{model_dir}/rf_multiclass_model2.pkl')

        metadata = {
            'models': list(self.models.keys()),
            'svm1_accuracy': self.models.get('svm1', {}).get('accuracy'),
            'svm2_accuracy': self.models.get('svm2', {}).get('accuracy'),
            'rf1_accuracy': self.models.get('rf1', {}).get('accuracy'),
            'rf2_accuracy': self.models.get('rf2', {}).get('accuracy'),
            'svm1_cv_score': self.models.get('svm1', {}).get('cv_score'),
            'svm2_cv_score': self.models.get('svm2', {}).get('cv_score'),
            'rf1_cv_score': self.models.get('rf1', {}).get('cv_score'),
            'rf2_cv_score': self.models.get('rf2', {}).get('cv_score'),
        }

        with open(f'{model_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, model_dir='models'):
        """Load trained models from disk"""
        metadata_path = f'{model_dir}/metadata.json'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        svm1_path = f'{model_dir}/svm_binary_model1.pkl'
        if os.path.exists(svm1_path):
            self.svm1 = joblib.load(svm1_path)
            print(f"Loaded SVM Model 1 (accuracy: {metadata.get('svm1_accuracy', 'N/A')})")

        svm2_path = f'{model_dir}/svm_binary_model2.pkl'
        if os.path.exists(svm2_path):
            self.svm2 = joblib.load(svm2_path)
            print(f"Loaded SVM Model 2 (accuracy: {metadata.get('svm2_accuracy', 'N/A')})")

        rf1_path = f'{model_dir}/rf_multiclass_model1.pkl'
        if os.path.exists(rf1_path):
            self.rf1 = joblib.load(rf1_path)
            print(f"Loaded RF Model 1 (accuracy: {metadata.get('rf1_accuracy', 'N/A')})")

        rf2_path = f'{model_dir}/rf_multiclass_model2.pkl'
        if os.path.exists(rf2_path):
            self.rf2 = joblib.load(rf2_path)
            print(f"Loaded RF Model 2 (accuracy: {metadata.get('rf2_accuracy', 'N/A')})")

        if self.svm1:
            self.loaded_model = self.svm1
            data_path = '../data/CICIDS2017/PCA_balanced.csv'

            if os.path.exists(data_path):
                sample_df = pd.read_csv(data_path, nrows=1)
                self.feature_names = [col for col in sample_df.columns if col != 'Attack Type']
            else:
                self.feature_names = [f'feature_{i}' for i in range(self.svm1.n_features_in_)]

        return self

    def predict(self, X):
        """
        Make predictions on input data
        Returns: (predictions, confidence_scores)
        """
        if self.loaded_model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        predictions = self.loaded_model.predict(X)

        if hasattr(self.loaded_model, 'predict_proba'):
            probas = self.loaded_model.predict_proba(X)
            confidence_scores = np.max(probas, axis=1)
        else:
            confidence_scores = np.ones(len(predictions))

        return predictions, confidence_scores


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--binary-only', action='store_true')
    parser.add_argument('--multiclass-only', action='store_true')
    parser.add_argument('--binary-data', type=str, default='../data/CICIDS2017/PCA_balanced.csv')
    parser.add_argument('--multiclass-data', type=str, default='../data/CICIDS2017/PCA_processed.csv')

    args = parser.parse_args()

    detector = NetworkAnomalyDetector()

    if not args.multiclass_only:
        if os.path.exists(args.binary_data):
            print(f"Training binary classifiers on {args.binary_data}...")
            detector.train_binary_classifiers(args.binary_data)
        else:
            print(f"Binary data file not found: {args.binary_data}")

    if not args.binary_only:
        if os.path.exists(args.multiclass_data):
            print(f"\nTraining multiclass classifiers on {args.multiclass_data}...")
            detector.train_multiclass_classifiers(args.multiclass_data)
        else:
            print(f"Multiclass data file not found: {args.multiclass_data}")

    if detector.models:
        detector.save_models()
        print('\n' + '='*50)
        print('Training Summary:')
        print('='*50)
        for model_name, model_info in detector.models.items():
            print(f'{model_name}: Accuracy={model_info["accuracy"]:.4f}, CV={model_info["cv_score"]:.4f}')
    else:
        print("\nNo models were trained. Check that data files exist.")


if __name__ == "__main__":
    main()
