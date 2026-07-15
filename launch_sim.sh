#!/usr/bin/env bash
# ============================================================
# launch_sim.sh — Launch PX4 SITL + Gazebo Classic
#
# This script starts the PX4 Software-In-The-Loop (SITL)
# simulation with Gazebo Classic 11 and the Iris quadcopter.
#
# MAVLink will be available on:
#   - udp://localhost:14540 (MAVSDK / offboard control)
#   - udp://localhost:14550 (QGroundControl)
#
# Usage:
#   bash launch_sim.sh
#
# To stop:
#   Ctrl+C or close the terminal
# ============================================================

set -e

# ── Configuration ──
PX4_DIR="${HOME}/PX4-Autopilot"
VEHICLE="gazebo-classic"

# ── Colour output ──
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   🚁 PX4 SITL + Gazebo Classic Launcher     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── Cleanup existing processes and ports ──
echo -e "${YELLOW}🧹 Cleaning up existing simulation processes and ports...${NC}"
killall -9 px4 gzserver gzclient ruby make sitl_multiple_run.sh 2>/dev/null || true
fuser -k -9 14560/udp 14561/udp 14562/udp 14580/udp 18570/udp 2>/dev/null || true
sleep 1
echo ""

# ── Check PX4 directory ──
if [ ! -d "$PX4_DIR" ]; then
    echo -e "${RED}❌ PX4-Autopilot not found at ${PX4_DIR}${NC}"
    echo -e "${YELLOW}   Clone it with:${NC}"
    echo -e "${YELLOW}   git clone https://github.com/PX4/PX4-Autopilot.git --recursive${NC}"
    exit 1
fi

echo -e "${GREEN}✅ PX4-Autopilot found at ${PX4_DIR}${NC}"

# ── Check Gazebo ──
if ! command -v gazebo &> /dev/null; then
    echo -e "${RED}❌ Gazebo Classic not found in PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Gazebo Classic found${NC}"

# ── Source ROS 2 if available ──
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}✅ ROS 2 Humble sourced${NC}"
fi

echo ""
echo -e "${CYAN}Starting PX4 SITL with Iris quadcopter...${NC}"
echo -e "${CYAN}MAVLink endpoints:${NC}"
echo -e "  • MAVSDK:  udp://localhost:14540"
echo -e "  • QGC:     udp://localhost:14550"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the simulation${NC}"
echo ""

# ── Launch PX4 SITL + Gazebo ──
MODE="standard"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cd "$PX4_DIR"

# (HEADLESS is handled by environment variables)

if [ "$MODE" == "swarm" ]; then
    NUM_DRONES=${NUM_DRONES:-3}
    echo -e "${GREEN}Launching Multi-Agent Swarm ($NUM_DRONES drones)...${NC}"

    # PX4 multi-instance: instance N uses offboard UDP port 14540+N
    # We need instances 0, 1, 2, ... so ports are 14540, 14541, 14542...
    # First, build PX4 if needed
    cd "$PX4_DIR"
    if [ ! -d "build/px4_sitl_default" ]; then
        echo -e "${YELLOW}Building PX4 SITL...${NC}"
        DONT_RUN=1 make px4_sitl gazebo-classic
    fi

    build_path="${PX4_DIR}/build/px4_sitl_default"
    src_path="${PX4_DIR}"
    
    # Source gazebo setup
    source "${src_path}/Tools/simulation/gazebo-classic/setup_gazebo.bash" "${src_path}" "${build_path}" 2>/dev/null || true

    # Start Gazebo server (headless or with GUI)
    if [ -n "${HEADLESS}" ]; then
        gzserver "${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/empty.world" &
    else
        gazebo "${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/empty.world" &
    fi
    GAZEBO_PID=$!
    sleep 3

    # Spawn each drone as PX4 instance 0, 1, 2, ...
    for ((i=0; i<NUM_DRONES; i++)); do
        working_dir="$build_path/rootfs/$i"
        [ ! -d "$working_dir" ] && mkdir -p "$working_dir"

        pushd "$working_dir" &>/dev/null

        # Start PX4 with instance number i → offboard port will be 14540+i
        $build_path/bin/px4 -i $i -d "$build_path/etc" >out.log 2>err.log &

        # Generate SDF model for this instance
        Y=$((3 * i))
        python3 ${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/scripts/jinja_gen.py \
            ${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf.jinja \
            ${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic \
            --mavlink_tcp_port $((4560+i)) \
            --mavlink_udp_port $((14560+i)) \
            --mavlink_id $((1+i)) \
            --gst_udp_port $((5600+i)) \
            --video_uri $((5600+i)) \
            --mavlink_cam_udp_port $((14530+i)) \
            --output-file /tmp/iris_${i}.sdf

        echo -e "${GREEN}Spawning iris_${i} at x=0 y=${Y}${NC}"
        gz model --spawn-file=/tmp/iris_${i}.sdf --model-name=iris_${i} -x 0 -y ${Y} -z 0.83

        popd &>/dev/null
        sleep 1
    done

    echo -e "${GREEN}All ${NUM_DRONES} drones spawned! Ports: 14540-$((14540+NUM_DRONES-1))${NC}"

    # Keep script alive until killed
    trap "pkill -x px4; pkill gzclient; pkill gzserver" SIGINT SIGTERM EXIT
    wait $GAZEBO_PID
else
    echo -e "${GREEN}Launching Standard Single Drone headlessly...${NC}"
    make px4_sitl gazebo-classic
fi
