#!/bin/bash

echo "=========================================="
echo "Network Anomaly Detection System Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python3
echo -e "\n${YELLOW}Checking Python3...${NC}"
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Python3 found: $(python3 --version)${NC}"
else
    echo "✗ Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check Node.js
echo -e "\n${YELLOW}Checking Node.js...${NC}"
if command -v node &> /dev/null; then
    echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"
else
    echo "✗ Node.js not found. Please install Node.js 18 or higher."
    exit 1
fi

# Setup Python virtual environment
echo -e "\n${YELLOW}Setting up Python environment...${NC}"
cd ml-service
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment and install dependencies
source venv/bin/activate
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Create directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p models
mkdir -p data
echo -e "${GREEN}✓ Directories created${NC}"

# Train the model
echo -e "\n${YELLOW}Training ML model...${NC}"
python3 train_model.py
echo -e "${GREEN}✓ Model trained successfully${NC}"

cd ..

# Install Node.js dependencies
echo -e "\n${YELLOW}Installing Node.js dependencies...${NC}"
npm install
echo -e "${GREEN}✓ Node.js dependencies installed${NC}"

# Create .env.local file if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo -e "\n${YELLOW}Creating .env.local file...${NC}"
    cat > .env.local << EOF
# ML Service URL
ML_API_URL=http://localhost:5000

# Next.js
NEXT_PUBLIC_API_URL=http://localhost:3000
EOF
    echo -e "${GREEN}✓ .env.local created${NC}"
fi

echo -e "\n=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=========================================="
echo ""
echo "To start the system:"
echo "  1. Start ML service:  npm run ml:serve"
echo "  2. Start Next.js app: npm run dev"
echo ""
echo "Then visit:"
echo "  - Dashboard: http://localhost:3000"
echo "  - Monitoring: http://localhost:3000/monitoring"
echo "  - Attack Charts: http://localhost:3000/attack-chart"
echo ""
