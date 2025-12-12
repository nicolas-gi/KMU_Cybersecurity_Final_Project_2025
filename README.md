# KMU Cybersecurity Final Project 2025

## Network Anomaly Detection Using Machine Learning

A machine learning-powered network intrusion detection system using Support Vector Machines (SVM) and Random Forest classifiers to detect cyberattacks in network traffic.

---

## 👥 Team

- **Nicolas Gillard** - Alerting Lead
- **Mohammed JBILOU** - AI Lead
- **Frederik Lind** - QA Lead
- **Lukas W. Blochmann** - Data Analyst

---

## 📋 Project Overview

This project implements machine learning models for network anomaly detection using the CICIDS2017 dataset. It includes both binary classification (normal vs. attack) and multi-class classification (specific attack types).

### Models Implemented

**Binary Classification (SVM):**
- SVM Model 1: Polynomial kernel (CV: 0.97)
- SVM Model 2: RBF kernel (CV: 1.00)

**Multi-class Classification (Random Forest):**
- RF Model 1: 10 estimators, depth 6 (CV: 0.99)
- RF Model 2: 15 estimators, depth 8 (CV: 1.00)

### Attack Types Detected

- BENIGN (Normal traffic)
- DoS (Denial of Service)
- DDoS (Distributed Denial of Service)
- PortScan
- BruteForce
- WebAttack
- Bot

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+ (for frontend, if applicable)

### Setup

```bash
# Clone the repository
git clone https://github.com/nicolas-gi/KMU_Cybersecurity_Final_Project_2025.git
cd Final_Project

# Install Python dependencies
cd ml-service
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### Train Models

**Important:** Run from the project root directory, not from inside ml-service/

```bash
# From project root
python3 ml-service/train_model.py
```

This will train all 4 models:
- 2 SVM models for binary classification
- 2 Random Forest models for multi-class classification

**Options:**

```bash
# Train only binary SVM classifiers
python3 ml-service/train_model.py --binary-only

# Train only multi-class Random Forest classifiers
python3 ml-service/train_model.py --multiclass-only
```

---

## 📁 Project Structure

```
Final_Project/
├── ml-service/
│   ├── train_model.py          # Model training script
│   ├── api.py                  # Flask API server
│   ├── requirements.txt        # Python dependencies
│   └── models/                 # Trained models (.pkl files)
│       ├── svm_binary_model1.pkl
│       ├── svm_binary_model2.pkl
│       ├── rf_multiclass_model1.pkl
│       ├── rf_multiclass_model2.pkl
│       └── metadata.json
├── data/
│   └── CICIDS2017/
│       ├── PCA_balanced.csv    # Binary classification dataset
│       └── PCA_processed.csv   # Multi-class dataset
├── trained_data/               # Original data from Data Analyst
├── frontend/                   # Next.js dashboard (if applicable)
└── README.md
```

---

## 🛠️ Model Training Details

### Dataset

Uses CICIDS2017 dataset with PCA-transformed features:
- **PCA_balanced.csv**: 15,000 samples for binary classification
- **PCA_processed.csv**: 2.5M+ samples for multi-class classification

### Training Process

1. **Binary SVM Models** (PCA_balanced.csv):
   - Train/test split: 80/20
   - Cross-validation: 5-fold
   - Model 1: Polynomial kernel
   - Model 2: RBF kernel (gamma=0.1)

2. **Multi-class Random Forest** (PCA_processed.csv):
   - Class filtering: Keep classes with >1950 samples
   - Sampling: 5000 per class (or all if <2500)
   - Balancing: SMOTE oversampling
   - Train/test split: 80/20
   - Cross-validation: 5-fold

### Expected Results

```
SVM Model 1: CV=0.97, Accuracy=0.9843
SVM Model 2: CV=1.00, Accuracy=0.9980
RF Model 1:  CV=0.99, Accuracy=0.9923
RF Model 2:  CV=1.00, Accuracy=0.9959
```

---

## 🔌 API Usage (if running Flask server)

### Start ML API Server

```bash
cd ml-service
source venv/bin/activate
python3 api.py
```

### Health Check

```bash
GET http://localhost:5000/health
```

### Prediction

```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "features": [0.5, 450, 300, 5, 3, 0.1, 0.05, 0.8, 0.2, ...]
}
```

---

## 📊 Dataset Information

**CICIDS2017** - Canadian Institute for Cybersecurity Intrusion Detection Dataset

- Source: University of New Brunswick
- Website: https://www.unb.ca/cic/datasets/ids-2017.html
- Preprocessed with PCA for dimensionality reduction
- Balanced using SMOTE for multi-class classification

---

## 🧪 Verification

To verify your models match the expected results, check:

1. Training output shows correct CV scores
2. Model intercepts match (SVM models)
3. All 4 model files saved to `ml-service/models/`
4. `metadata.json` contains accuracy metrics

See `TRAINING_VERIFICATION.md` for detailed comparison.

---

## 📝 Notes

- **Critical:** Always run `train_model.py` from the project root, not from `ml-service/`
- Training uses `random_state=0` for reproducibility
- Models are saved as `.pkl` files using joblib
- SMOTE balancing ensures equal representation of attack types

---

## 📚 Resources

- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

---