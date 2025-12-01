# Quick Start Guide - Network Anomaly Detection System

## 🚀 5-Minute Setup

### Step 1: Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

### Step 2: Start Services

**Terminal 1 - Start ML API:**

```bash
npm run ml:serve
```

**Terminal 2 - Start Web Dashboard:**

```bash
npm run dev
```

### Step 3: Open Browser

Visit: <http://localhost:3000/monitoring>

Click "Start Monitoring" to begin real-time anomaly detection!

---

## 📊 What You'll See

1. **ML Service Status** - Green indicator = ready
2. **Statistics Cards** - Total samples, normal traffic, anomalies, critical alerts
3. **Traffic Distribution Chart** - Normal vs Anomalous traffic bar chart
4. **Real-time Timeline** - Live connection volume graph
5. **Security Alerts Feed** - Real-time threat notifications with severity levels

---

## 🎯 Understanding Threat Levels

- **🟢 Normal** - Regular network traffic (confidence < 0.7)
- **🟡 Medium** - Suspicious activity (confidence 0.7-0.9)
- **🟠 High** - Likely attack detected (confidence 0.9+)
- **🔴 Critical** - Confirmed threat (confidence 0.9+, high severity)

---

## 🧪 Test the System

The monitoring dashboard automatically generates realistic network traffic patterns including:

- **85% Normal Traffic** - Typical network behavior
- **15% Anomalies** - Simulated attacks (DDoS, port scans, etc.)

Watch as the ML model detects and classifies threats in real-time!

---

## 📈 Other Pages

- `/` - Home page with project info
- `/attack-chart` - Static attack type visualization
- `/monitoring` - Real-time ML-powered dashboard ⭐

---

## ⚙️ Advanced: Use Real Datasets

### Download NSL-KDD

```bash
cd ml-service
source venv/bin/activate
python3 process_datasets.py --download-nsl-kdd
python3 train_model.py
```

### Use CICIDS2017

1. Download from: <https://www.unb.ca/cic/datasets/ids-2017.html>
2. Process: `python3 process_datasets.py --process-cicids path/to/dataset.csv`
3. Retrain: `python3 train_model.py`

---

## 🛠️ Troubleshooting

**ML Service won't start?**

- Check Python 3.8+ is installed: `python3 --version`
- Activate venv: `cd ml-service && source venv/bin/activate`
- Reinstall deps: `pip install -r requirements.txt`

**"ML Service: unhealthy" error?**

- Make sure ML service is running on port 5000
- Check: `curl http://localhost:5000/health`

**No predictions happening?**

- Verify both services are running (ports 3000 and 5000)
- Check browser console for errors (F12)
- Restart both services

---

## 📚 Need More Help?

See the full README.md for:

- Detailed architecture
- API documentation
- Customization options
- Dataset processing
- Model tuning

---

Happy threat hunting! 🔐
