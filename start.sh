#!/bin/bash

echo "=========================================="
echo "🚀 Starting Network Anomaly Detection System"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if ML model exists
if [ ! -f "ml-service/models/anomaly_model.pkl" ]; then
    echo -e "${YELLOW}⚠ ML model not found. Training model first...${NC}"
    cd ml-service
    if [ -d "venv" ]; then
        venv/bin/python train_model.py
    else
        python3 train_model.py
    fi
    cd ..
    echo -e "${GREEN}✓ Model trained${NC}\n"
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $ML_PID 2>/dev/null
    kill $NEXT_PID 2>/dev/null
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

# Set up trap to catch Ctrl+C
trap cleanup SIGINT SIGTERM

# Start ML API Server
echo -e "${BLUE}Starting ML API Server on port 5000...${NC}"
cd ml-service
if [ -d "venv" ]; then
    venv/bin/python api.py > ../logs/ml-service.log 2>&1 &
else
    python3 api.py > ../logs/ml-service.log 2>&1 &
fi
ML_PID=$!
cd ..

# Wait for ML service to start
echo -e "${YELLOW}Waiting for ML service to initialize...${NC}"
sleep 3

# Check if ML service is running
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ML API Server running (PID: $ML_PID)${NC}"
else
    echo -e "${RED}✗ ML API Server failed to start${NC}"
    echo -e "${YELLOW}Check logs/ml-service.log for details${NC}"
fi

# Start Next.js Server
echo -e "\n${BLUE}Starting Next.js Dashboard on port 3000...${NC}"
npm run dev > logs/next.log 2>&1 &
NEXT_PID=$!

# Wait for Next.js to start
echo -e "${YELLOW}Waiting for Next.js to initialize...${NC}"
sleep 5

# Check if Next.js is running
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Next.js Dashboard running (PID: $NEXT_PID)${NC}"
else
    echo -e "${RED}✗ Next.js Dashboard failed to start${NC}"
    echo -e "${YELLOW}Check logs/next.log for details${NC}"
fi

echo -e "\n=========================================="
echo -e "${GREEN}✅ System is running!${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}📊 Access the application:${NC}"
echo "  🏠 Home:              http://localhost:3000"
echo "  📊 Real-Time Monitor: http://localhost:3000/monitoring"
echo "  📈 Attack Charts:     http://localhost:3000/attack-chart"
echo "  🔌 ML API:            http://localhost:5000/health"
echo ""
echo -e "${YELLOW}📋 Logs:${NC}"
echo "  ML Service: logs/ml-service.log"
echo "  Next.js:    logs/next.log"
echo ""
echo -e "${RED}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and show logs
tail -f logs/ml-service.log logs/next.log &
TAIL_PID=$!

# Wait for processes
wait $ML_PID $NEXT_PID
