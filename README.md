# KMU Cybersecurity Final Project 2025

## Network Anomaly Detection Using Machine Learning

A comprehensive ML-powered network intrusion detection system with real-time monitoring, anomaly detection, and interactive dashboards. Built with Next.js, Python, and scikit-learn.

---

## 📋 Project Overview

This project implements a machine learning-based network anomaly detection system that identifies cyberattacks (DDoS, port scans, intrusion attempts) in real-time. It combines a Python ML backend with a modern Next.js web dashboard for monitoring and visualization.

### Key Features

- **🤖 ML-Powered Detection**: Random Forest and Isolation Forest algorithms for anomaly detection
- **📊 Real-Time Monitoring**: Live dashboard showing normal vs. anomalous traffic
- **🚨 Intelligent Alerts**: Automatic threat detection with severity classification (Normal, Medium, High, Critical)
- **📈 Interactive Visualizations**: Dynamic charts for attack patterns and traffic analysis
- **🎯 Dataset Support**: Compatible with CICIDS2017 and NSL-KDD datasets
- **⚡ RESTful API**: Python Flask API for ML predictions
- **🌓 Dark Mode**: Modern UI with automatic theme switching
- **📱 Responsive Design**: Mobile-friendly interface

### Tech Stack

**Frontend:**

- [Next.js 16](https://nextjs.org/) (React 19) - Web framework
- TypeScript - Type safety
- Tailwind CSS 4 - Styling
- [Nivo](https://nivo.rocks/) - Data visualization

**Backend & ML:**

- Python 3.8+ - ML service
- scikit-learn - Machine learning algorithms
- pandas & NumPy - Data processing
- Flask - API server
- joblib - Model persistence

### Supported Datasets

1. **NSL-KDD** - Classic intrusion detection dataset
2. **CICIDS2017** - Comprehensive labeled network traffic (normal + attacks)

### Attack Types Detected

- **DoS** (Denial of Service): back, land, neptune, pod, smurf, teardrop
- **Probe** (Reconnaissance): ipsweep, nmap, portsweep, satan
- **R2L** (Remote to Local): ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster
- **U2R** (User to Root): buffer_overflow, loadmodule, perl, rootkit

---

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+** and npm
- **Python 3.8+** and pip
- Git

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/nicolas-gi/KMU_Cybersecurity_Final_Project_2025.git
cd final_proj

# Run automated setup script
chmod +x setup.sh
./setup.sh
```

The script will:

- Install Python dependencies
- Create virtual environment
- Train the ML model
- Install Node.js dependencies
- Set up configuration files

### Manual Setup

## 1. Install Python Dependencies

```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the ML Model

```bash
# Train with mock data (for demo)
python3 train_model.py

# Or download and use real datasets
python3 process_datasets.py --download-nsl-kdd
python3 train_model.py
```

## 3. Install Node.js Dependencies

```bash
cd ..
npm install
```

## 4. Configure Environment

Create `.env.local`:

```env
ML_API_URL=http://localhost:5000
NEXT_PUBLIC_API_URL=http://localhost:3000
```

---

## 🎮 Running the Application

### Option 1: Start Everything at Once (Recommended)

```bash
./start.sh
```

This single command will:

- Check and train the ML model if needed
- Start the ML API server on port 5000
- Start the Next.js dashboard on port 3000
- Show status and access URLs
- Display logs in real-time

**To stop all services:**

```bash
./stop.sh
# Or press Ctrl+C in the terminal running start.sh
```

### Option 2: Start Services Separately

**Terminal 1 - ML API Server:**

```bash
npm run ml:serve
# Or: cd ml-service && source venv/bin/activate && python3 api.py
```

**Terminal 2 - Next.js Dashboard:**

```bash
npm run dev
```

### Access the Application

- 🏠 **Home**: <http://localhost:3000>
- 📊 **Real-Time Monitoring**: <http://localhost:3000/monitoring>
- 📈 **Attack Charts**: <http://localhost:3000/attack-chart>
- 🔌 **ML API**: <http://localhost:5000/health>

---

## 📁 Project Structure

```shell
final_proj/
├── app/
│   ├── page.tsx                    # Home page
│   ├── monitoring/
│   │   └── page.tsx                # Real-time monitoring dashboard
│   ├── attack-chart/
│   │   └── page.tsx                # Attack visualization
│   ├── api/
│   │   ├── predict/route.ts        # Single prediction endpoint
│   │   ├── predict/batch/route.ts  # Batch prediction endpoint
│   │   └── ml-health/route.ts      # ML service health check
│   ├── layout.tsx                  # Root layout
│   └── globals.css                 # Global styles
├── ml-service/
│   ├── train_model.py              # ML model training script
│   ├── api.py                      # Flask API server
│   ├── process_datasets.py         # Dataset processing utilities
│   ├── requirements.txt            # Python dependencies
│   ├── models/                     # Trained models directory
│   └── data/                       # Datasets directory
├── public/                         # Static assets
├── setup.sh                        # Automated setup script
├── package.json                    # Node.js dependencies
└── README.md                       # This file
```

---

## 🛠️ Development

### Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Next.js development server |
| `npm run build` | Build production bundle |
| `npm start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm run ml:train` | Train ML model |
| `npm run ml:serve` | Start ML API server |

### ML API Endpoints

## Health Check

```bash
GET http://localhost:5000/health
```

## Single Prediction

```bash
POST http://localhost:5000/predict
Content-Type: application/json

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
```

**Response:**

```json
{
  "is_anomaly": false,
  "confidence": 0.95,
  "threat_level": "normal",
  "prediction": "normal"
}
```

### Batch Prediction

```bash
POST http://localhost:5000/predict/batch
Content-Type: application/json

{
  "samples": [
    { "duration": 0.5, "src_bytes": 450, ... },
    { "duration": 0.1, "src_bytes": 100, ... }
  ]
}
```

---

## 📊 Using Real Datasets

### Download NSL-KDD Dataset

```bash
cd ml-service
python3 process_datasets.py --download-nsl-kdd
```

### Process CICIDS2017 Dataset

1. Download CICIDS2017 from: <https://www.unb.ca/cic/datasets/ids-2017.html>
2. Process the CSV file:

```bash
cd ml-service
python3 process_datasets.py --process-cicids path/to/CICIDS2017.csv
```

### Retrain Model with Real Data

```bash
cd ml-service
source venv/bin/activate
python3 train_model.py
```

The model will automatically use datasets from the `data/` directory if available.

---

## 🎯 Dashboard Features

### Real-Time Monitoring (`/monitoring`)

- **Live Traffic Analysis**: Monitors network traffic every 2 seconds
- **Anomaly Detection**: ML-powered identification of suspicious patterns
- **Alert System**: Real-time security alerts with severity levels
- **Statistics Dashboard**:
  - Total samples processed
  - Normal vs. anomalous traffic
  - Critical alerts count
- **Interactive Charts**:
  - Traffic distribution (Normal vs. Anomalies)
  - Real-time connection volume timeline

### Attack Visualization (`/attack-chart`)

- **Category-Based Analysis**: Groups attacks by type (DoS, Probe, R2L, U2R)
- **Detailed Tooltips**: Shows specific attack types in each category
- **Statistical Overview**: Summary cards with key metrics
- **Color-Coded Threats**: Visual severity indicators

---

## 🔧 Customization

### Changing ML Model Type

Edit `ml-service/train_model.py`:

```python
# Use Random Forest (default)
detector = NetworkAnomalyDetector(model_type='random_forest')

# Or use Isolation Forest
detector = NetworkAnomalyDetector(model_type='isolation_forest')
```

### Adjusting Detection Sensitivity

Modify model parameters in `train_model.py`:

```python
# Random Forest
self.model = RandomForestClassifier(
    n_estimators=100,      # Increase for better accuracy
    max_depth=10,          # Adjust tree depth
    random_state=42
)

# Isolation Forest
self.model = IsolationForest(
    contamination=0.1,     # Expected anomaly ratio (0-0.5)
    random_state=42
)
```

### Custom Alert Thresholds

Edit `app/monitoring/page.tsx`:

```typescript
// Determine threat level based on confidence score
if (prediction == 1) {
  if (score > 0.9) threat_level = 'critical';    // Adjust threshold
  elif (score > 0.7) threat_level = 'high';       // Adjust threshold
  else threat_level = 'medium';
}
```

---

## 🧪 Testing

### Test ML API

```bash
# Check health
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0.1,
    "src_bytes": 100,
    "dst_bytes": 50,
    "count": 50,
    "srv_count": 40,
    "serror_rate": 0.8,
    "rerror_rate": 0.1,
    "same_srv_rate": 0.2,
    "diff_srv_rate": 0.8
  }'
```

---

## 📝 License

This project is part of the KMU Cybersecurity Final Project 2025.

---

## 👥 Contributors

- **Repository**: [nicolas-gi/KMU_Cybersecurity_Final_Project_2025](https://github.com/nicolas-gi/KMU_Cybersecurity_Final_Project_2025)

---

## 🔗 Resources

### Documentation

- [Next.js Documentation](https://nextjs.org/docs)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Nivo Chart Library](https://nivo.rocks/)
- [Tailwind CSS](https://tailwindcss.com/docs)

### Datasets

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

### Research Papers

- KDD Cup 1999 Data
- NSL-KDD: A Modern Intrusion Detection Dataset
- CICIDS2017: A Network Intrusion Detection Dataset
