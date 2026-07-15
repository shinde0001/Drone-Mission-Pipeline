#!/usr/bin/env bash
# ============================================================
# setup.sh — Automated Setup for Drone Mission Pipeline
#
# Installs Python dependencies (offline-compatible via requirements.txt).
# Ollama local installation is handled automatically by the backend
# via the dashboard UI on first use when offline AI mode is selected.
# ============================================================

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   🚁 Drone Mission Pipeline Setup            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Installing Python dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}✅ Python dependencies installed successfully!${NC}"
echo ""
echo -e "${CYAN}To launch the Web Dashboard:${NC}"
echo -e "  python3 web_dashboard/app.py"
echo ""
echo -e "${CYAN}Note: If you plan to use local AI models offline without internet connection during flights,${NC}"
echo -e "${CYAN}the dashboard will prompt you to install Ollama (~1.5 GB) if needed when you select Offline mode.${NC}"
echo ""
