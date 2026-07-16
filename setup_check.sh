#!/usr/bin/env bash
# ============================================================
# setup_check.sh — Preflight System Check
#
# Verifies all dependencies are installed and configured
# before running the drone pipeline.
# ============================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

PASS=0
FAIL=0

check() {
    local desc="$1"
    local cmd="$2"
    
    if eval "$cmd" &> /dev/null; then
        echo -e "  ${GREEN}✅ ${desc}${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}❌ ${desc}${NC}"
        FAIL=$((FAIL + 1))
    fi
}

echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║   🔍 Drone Pipeline — Preflight Check       ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BOLD}System:${NC}"
check "Python 3.8+" "python3 -c 'import sys; assert sys.version_info >= (3,8)'"
check "pip3" "pip3 --version"

echo ""
echo -e "${BOLD}Python packages:${NC}"
check "mavsdk" "python3 -c 'import mavsdk'"
check "jsonschema" "python3 -c 'import jsonschema'"
check "pyyaml" "python3 -c 'import yaml'"
check "pymavlink" "python3 -c 'import pymavlink'"
check "fastapi" "python3 -c 'import fastapi'"

echo ""
echo -e "${BOLD}ROS 2 & Simulation:${NC}"
check "ROS 2 (ros2 CLI)" "which ros2"
check "Gazebo Classic" "which gazebo"
check "PX4-Autopilot directory" "test -d ${HOME}/PX4-Autopilot"
check "PX4 SITL build" "test -d ${HOME}/PX4-Autopilot/build/px4_sitl_default"

echo ""
echo -e "${BOLD}LLM Providers (Ollama required):${NC}"
has_ollama=0
# Check if ollama is running and has llama3.2:1b model
if curl -s http://localhost:11434/api/tags | grep -q "llama3.2:1b"; then
    has_ollama=1
fi

if [ $has_ollama -eq 1 ]; then
    echo -e "  ${GREEN}✅ Local Ollama with llama3.2:1b available${NC}"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}❌ Local Ollama with llama3.2:1b is not running or not installed${NC}"
    FAIL=$((FAIL + 1))
fi

echo ""
echo -e "${BOLD}ROS 2 Packages:${NC}"
check "gazebo_ros" "ros2 pkg list 2>/dev/null | grep -q gazebo_ros"
check "gazebo_msgs" "ros2 pkg list 2>/dev/null | grep -q gazebo_msgs"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All ${PASS} checks passed! Ready to fly. 🚁${NC}"
else
    echo -e "${YELLOW}${BOLD}${PASS} passed, ${FAIL} failed.${NC}"
    echo -e "${YELLOW}Fix the failed items above before running the pipeline.${NC}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
