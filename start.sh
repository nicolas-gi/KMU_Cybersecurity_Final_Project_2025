#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

mkdir -p logs

if [ ! -f "ml-service/models/anomaly_model.pkl" ]; then
    cd ml-service
    if [ -d "venv" ]; then
        venv/bin/python train_model.py
    else
        python3 train_model.py
    fi
    cd ..
fi

cleanup() {
    kill $ML_PID 2>/dev/null
    kill $NEXT_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

cd ml-service
if [ -d "venv" ]; then
    venv/bin/python api.py > ../logs/ml-service.log 2>&1 &
else
    python3 api.py > ../logs/ml-service.log 2>&1 &
fi
ML_PID=$!
cd ..

sleep 3

cd frontend
npm run dev > ../logs/next.log 2>&1 &
NEXT_PID=$!
cd ..

sleep 5

tail -f logs/ml-service.log logs/next.log &
TAIL_PID=$!

wait $ML_PID $NEXT_PID
