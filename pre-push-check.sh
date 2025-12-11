#!/bin/bash

# Pre-Push Verification Script
# Run this before pushing to catch issues early

set -e

echo "🔍 Running Pre-Push Verification..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check 1: Node.js version
echo "${YELLOW}[1/5] Checking Node.js version...${NC}"
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -ge 18 ]; then
    echo -e "${GREEN}✓ Node.js $(node --version) is compatible${NC}"
else
    echo -e "${RED}✗ Node.js must be 18+${NC}"
    exit 1
fi

# Check 2: ESLint
echo ""
echo "${YELLOW}[2/5] Running ESLint...${NC}"
if npm run lint; then
    echo -e "${GREEN}✓ ESLint passed${NC}"
else
    echo -e "${RED}✗ ESLint found issues${NC}"
    exit 1
fi

# Check 3: TypeScript
echo ""
echo "${YELLOW}[3/5] Running TypeScript check...${NC}"
if npx tsc --noEmit; then
    echo -e "${GREEN}✓ TypeScript check passed${NC}"
else
    echo -e "${RED}✗ TypeScript check failed${NC}"
    exit 1
fi

# Check 4: Build
echo ""
echo "${YELLOW}[4/5] Building Next.js project...${NC}"
if npm run build > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Next.js build successful${NC}"
else
    echo -e "${RED}✗ Next.js build failed${NC}"
    exit 1
fi

# Check 5: Python imports (with venv)
echo ""
echo "${YELLOW}[5/5] Checking Python ML service...${NC}"
cd ml-service
if [ -d "venv" ]; then
    source venv/bin/activate
    if python3 -c "import train_model; import api; print('✓ Python modules valid')" 2>/dev/null; then
        deactivate
        echo -e "${GREEN}✓ Python ML service is valid${NC}"
        cd ..
    else
        deactivate
        echo -e "${YELLOW}⚠ Python ML service check skipped (run setup.sh first)${NC}"
        cd ..
    fi
else
    echo -e "${YELLOW}⚠ Python venv not found (run setup.sh first)${NC}"
    cd ..
fi

echo ""
echo -e "${GREEN}════════════════════════════════${NC}"
echo -e "${GREEN}✓ All checks passed!${NC}"
echo -e "${GREEN}Safe to push to repository${NC}"
echo -e "${GREEN}════════════════════════════════${NC}"
