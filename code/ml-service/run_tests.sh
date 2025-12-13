#!/bin/bash
# Test runner script for ML service

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running ML Service Tests...${NC}\n"

# Run tests with coverage, disabling the problematic anchorpy plugin
python3 -m pytest tests/ -p no:anchorpy -v --cov=. --cov-report=term-missing --cov-report=html

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
    echo -e "\nCoverage report generated in htmlcov/index.html"
else
    echo -e "\n${YELLOW}Some tests failed. Please review the output above.${NC}"
    exit 1
fi
