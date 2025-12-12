#!/bin/bash

if ! command -v python3 &> /dev/null; then
    exit 1
fi

if ! command -v node &> /dev/null; then
    exit 1
fi

cd ml-service
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p models
mkdir -p data

python3 train_model.py

cd ../frontend

npm install

if [ ! -f ".env.local" ]; then
    cat > .env.local << EOF
ML_API_URL=http://localhost:5001
NEXT_PUBLIC_API_URL=http://localhost:3000
EOF
fi
