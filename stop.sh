#!/bin/bash

echo "=========================================="
echo "🛑 Stopping Network Anomaly Detection System"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Kill processes on ports 3000 and 5000
echo -e "${YELLOW}Stopping services...${NC}"

# Stop Next.js (port 3000)
NEXT_PID=$(lsof -ti:3000)
if [ ! -z "$NEXT_PID" ]; then
    kill -9 $NEXT_PID 2>/dev/null
    echo -e "${GREEN}✓ Next.js stopped${NC}"
else
    echo "Next.js not running"
fi

# Stop ML API (port 5000)
ML_PID=$(lsof -ti:5000)
if [ ! -z "$ML_PID" ]; then
    kill -9 $ML_PID 2>/dev/null
    echo -e "${GREEN}✓ ML API stopped${NC}"
else
    echo "ML API not running"
fi

# Kill any remaining node/python processes related to the project
pkill -f "next dev" 2>/dev/null
pkill -f "api.py" 2>/dev/null

echo -e "\n${GREEN}✅ All services stopped${NC}"
