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
- Node.js 18+

### 5-Minute Setup

**Step 1: Run Setup Script**

```bash
chmod +x setup.sh
./setup.sh
```

**Step 2: Start Services**

Terminal 1 - Start ML API:
```bash
npm run ml:serve
```

Terminal 2 - Start Web Dashboard:
```bash
npm run dev
```

**Step 3: Open Browser**

Visit: http://localhost:3000/monitoring

Click "Start Monitoring" to begin real-time anomaly detection!

### Manual Setup (Alternative)

```bash
git clone https://github.com/nicolas-gi/KMU_Cybersecurity_Final_Project_2025.git
cd KMU_Cybersecurity_Final_Project_2025

cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Install Node.js dependencies
npm install
```

### Train Models

**Important:** Run from the project root directory, not from inside ml-service/

```bash
python3 ml-service/train_model.py
```

This will train all 4 models:
- 2 SVM models for binary classification
- 2 Random Forest models for multi-class classification

**Options:**

```bash
python3 ml-service/train_model.py --binary-only

python3 ml-service/train_model.py --multiclass-only
```

---

## 📊 Real-Time Monitoring Dashboard

### What You'll See

1. **ML Service Status** - Green indicator = ready
2. **Statistics Cards** - Total samples, normal traffic, anomalies, critical alerts
3. **Traffic Distribution Chart** - Normal vs Anomalous traffic bar chart
4. **Real-time Timeline** - Live connection volume graph
5. **Security Alerts Feed** - Real-time threat notifications with severity levels

### Understanding Threat Levels

- 🟢 **Normal** - Regular network traffic (confidence < 0.7)
- 🟡 **Medium** - Suspicious activity (confidence 0.7-0.9)
- 🟠 **High** - Likely attack detected (confidence 0.9+)
- 🔴 **Critical** - Confirmed threat (confidence 0.9+, high severity)

### Test the System

The monitoring dashboard automatically generates realistic network traffic patterns:

- **85% Normal Traffic** - Typical network behavior
- **15% Anomalies** - Simulated attacks (DDoS, port scans, etc.)

Watch as the ML model detects and classifies threats in real-time!

### Available Pages

- `/` - Home page with project info
- `/attack-chart` - Static attack type visualization
- `/monitoring` - Real-time ML-powered dashboard ⭐

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

## 🧪 Running Tests

The project includes comprehensive test coverage for the ML service models and API.

### Run All Tests

```bash
cd ml-service
source venv/bin/activate
./run_tests.sh
```

The test script will:
- Run all unit tests with pytest
- Generate coverage reports
- Display test results with color-coded output
- Create an HTML coverage report in `htmlcov/index.html`

### Manual Test Execution

You can also run tests manually:

```bash
cd ml-service
python3 -m pytest tests/ -v
python3 -m pytest tests/ --cov=. --cov-report=term-missing
```

### Test Coverage

The test suite includes:
- Model training and prediction tests
- API endpoint tests (health check, prediction)
- Configuration validation tests

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

## � Automated CI/CD Pipeline

### Setup Status: ✅ READY FOR DEPLOYMENT

All automated workflows are configured and ready to run on every push and pull request.

### GitHub Secrets Required

Before pushing to GitHub, you **MUST** add your SonarQube token:

1. Go to: GitHub repo → **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Add:
   - **Name:** `SONAR_TOKEN`
   - **Value:** Get from [SonarCloud](https://sonarcloud.io/account/security)

> Without this token, the SonarQube workflow will fail with "Not authorized"

### Automated Workflows

The following workflows run automatically on push/PR to `main` or `develop` branches:

| Workflow | Purpose | Status |
|----------|---------|--------|
| `sonarqube.yml` | Code quality analysis | ✅ Ready |
| `tests.yml` | Next.js build, Python tests, dependencies | ✅ Ready |
| `lint.yml` | ESLint, TypeScript, Prettier, Python linting | ✅ Ready |
| `build.yml` | Legacy SonarQube build workflow | ✅ Ready |

### Local Development Commands

```bash
# Check code before pushing
bash pre-push-check.sh

# Run linter
npm run lint

# Build project
npm run build

# Run SonarQube (after adding token to .env.local)
npm run sonar

# Train ML model
npm run ml:train

# Serve ML API
npm run ml:serve

# Start dev server
npm run dev
```

### Pre-Push Verification

Run this before each push to catch issues early:

```bash
bash pre-push-check.sh
```

This checks:
- ✅ Node.js version (18+)
- ✅ ESLint
- ✅ TypeScript compilation
- ✅ Next.js build
- ⚠️ Python ML Service (requires venv)

### View Pipeline Results

- **GitHub Actions:** Your repo → Actions tab
- **SonarCloud:** [sonarcloud.io](https://sonarcloud.io) → Your Project

### Troubleshooting CI/CD

| Issue | Solution |
|-------|----------|
| SonarQube "Not authorized" | Add `SONAR_TOKEN` to GitHub secrets |
| Build fails | Run `npm run build` locally and fix errors |
| Lint errors | Run `npm run lint` and fix issues |
| Python errors | Run `bash setup.sh` to install venv |

### Configuration Details

**Project Key:** `nicolas-gi_KMU_Cybersecurity_Final_Project_2025`  
**Organization:** `nicolas-gi`  
**Python Version:** 3.8+  
**Node.js:** 18+ (tests on 18.x and 20.x)  
**Quality Gate:** Enabled

---

## 🛠️ Advanced Dataset Processing

### Use Real NSL-KDD Dataset

```bash
cd ml-service
source venv/bin/activate
python3 process_datasets.py --download-nsl-kdd
python3 train_model.py
```

### Use CICIDS2017 Dataset

1. Download from: https://www.unb.ca/cic/datasets/ids-2017.html
2. Process: `python3 process_datasets.py --process-cicids path/to/dataset.csv`
3. Retrain: `python3 train_model.py`

---

## 🐛 Troubleshooting

**ML Service won't start?**

- Check Python 3.8+ is installed: `python3 --version`
- Activate venv: `cd ml-service && source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**"ML Service: unhealthy" error?**

- Make sure ML service is running on port 5000
- Check: `curl http://localhost:5000/health`

**No predictions happening?**

- Verify both services are running (ports 3000 and 5000)
- Check browser console for errors (F12)
- Restart both services

**Build fails?**

- Run `npm run build` locally to see detailed errors
- Check that all dependencies are installed: `npm install`
- Verify Node.js version: `node --version` (should be 18+)

**Lint errors?**

- Run `npm run lint` to see all issues
- Most issues can be auto-fixed with `npm run lint -- --fix`

---

## �📝 Notes

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