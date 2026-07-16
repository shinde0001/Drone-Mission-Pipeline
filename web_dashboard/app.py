"""
app.py — FastAPI Web Dashboard Backend

Provides a web interface for the drone mission pipeline with:
  - REST API endpoints for plan/validate/execute
  - WebSocket for live telemetry streaming
  - Serves the frontend static files
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_planner import plan_mission
from src.mission_validator import validate_mission, validate_hardware_safety
from src.utils import load_waypoint_library, AuditLog, load_safety_limits
from mavsdk import System

app = FastAPI(
    title="🚁 Drone Mission Pipeline",
    description="AI-powered drone mission planning and execution",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ──
pipeline_state = {
    "status": "idle",
    "manual_formation": "wedge",  # idle, planning, validating, executing, error
    "last_mission": None,
    "last_validation": None,
    "execution_log": [],
    "drone_logs": [[] for _ in range(5)],
    "telemetry": {
        "connected": False,
        "armed": False,
        "altitude_m": 0,
        "speed_mps": 0,
        "latitude": 0,
        "longitude": 0,
        "heading_deg": 0,
        "battery_pct": 100,
        "drones": [
            {
                "connected": False,
                "armed": False,
                "altitude_m": 0.0,
                "speed_mps": 0.0,
                "latitude": 0.0,
                "longitude": 0.0,
                "heading_deg": 0.0,
                "battery_pct": 100,
            } for _ in range(5)
        ]
    },
    "last_telemetry_update": datetime.now(timezone.utc),
    "connection_mode": "simulation",
    "hardware_ports": [],
    "baud_rate": 57600
}

# Shared drone system instance
shared_drone = System()
swarm_drones = {}

# Shared telemetry state for executor
shared_telemetry_state = {
    "position": None,
    "home": None
}

def is_simulator_running():
    import subprocess
    try:
        subprocess.check_output(["pgrep", "-f", "px4"])
        return True
    except subprocess.CalledProcessError:
        return False

def update_telemetry_timestamp():
    pipeline_state["last_telemetry_update"] = datetime.now(timezone.utc)

async def monitor_position(drone):
    try:
        async for position in drone.telemetry.position():
            update_telemetry_timestamp()
            shared_telemetry_state["position"] = position
            pipeline_state["telemetry"]["altitude_m"] = position.relative_altitude_m
            pipeline_state["telemetry"]["latitude"] = position.latitude_deg
            pipeline_state["telemetry"]["longitude"] = position.longitude_deg
    except Exception as e:
        print(f"Error monitoring position: {e}")

async def monitor_armed(drone):
    try:
        async for armed in drone.telemetry.armed():
            update_telemetry_timestamp()
            pipeline_state["telemetry"]["armed"] = armed
    except Exception as e:
        print(f"Error monitoring armed state: {e}")

async def monitor_heading(drone):
    try:
        async for heading in drone.telemetry.heading():
            update_telemetry_timestamp()
            pipeline_state["telemetry"]["heading_deg"] = heading.heading_deg
    except Exception as e:
        print(f"Error monitoring heading: {e}")

async def monitor_speed(drone):
    try:
        async for velocity in drone.telemetry.velocity_ned():
            update_telemetry_timestamp()
            speed = (velocity.north_m_s**2 + velocity.east_m_s**2 + velocity.down_m_s**2)**0.5
            pipeline_state["telemetry"]["speed_mps"] = speed
    except Exception as e:
        print(f"Error monitoring speed: {e}")

async def monitor_battery(drone):
    try:
        async for battery in drone.telemetry.battery():
            update_telemetry_timestamp()
            pct = battery.remaining_percent
            if pct <= 1.0:
                pct *= 100
            pipeline_state["telemetry"]["battery_pct"] = pct
    except Exception as e:
        print(f"Error monitoring battery: {e}")

async def monitor_single_drone_task(drone_idx=0, port=14540):
    import random
    
    while True:
        try:
            is_hw = pipeline_state.get("connection_mode") == "hardware"
            if not is_hw and not is_simulator_running():
                pipeline_state["telemetry"]["drones"][drone_idx]["connected"] = False
                if drone_idx == 0:
                    pipeline_state["telemetry"]["connected"] = False
                await asyncio.sleep(2)
                continue

            # Use a random port to prevent mavsdk_server port conflicts across retries
            mavsdk_port = random.randint(50000, 60000)
            drone = System(port=mavsdk_port)
            
            connected = False
            if is_hw:
                ports = pipeline_state.get("hardware_ports", [])
                if drone_idx < len(ports):
                    port_path = ports[drone_idx]
                    baud_rate = pipeline_state.get("baud_rate", 57600)
                    
                    # Smart baud negotiation
                    baud_sequence = [(baud_rate, 3), (115200, 1), (9600, 1)]
                    for b_rate, max_tries in baud_sequence:
                        if connected: break
                        
                        for attempt in range(max_tries):
                            try:
                                await drone.connect(system_address=f"serial://{port_path}:{b_rate}")
                                
                                async def check_conn():
                                    async for state in drone.core.connection_state():
                                        if state.is_connected:
                                            return True
                                
                                # Wait up to 3s for connection
                                connected = await asyncio.wait_for(check_conn(), timeout=3.0)
                            except (asyncio.TimeoutError, Exception):
                                connected = False
                            
                            if connected:
                                print(f"Hardware drone {drone_idx} connected on {port_path} at {b_rate} baud (attempt {attempt + 1})")
                                pipeline_state["baud_rate"] = b_rate
                                break
                            else:
                                # Re-instantiate MAVSDK System on failure to avoid backend state corruption
                                try:
                                    drone._stop_mavsdk_server()
                                except Exception:
                                    pass
                                await asyncio.sleep(0.5) # Allow OS to free the serial port lock
                                drone = System(port=random.randint(50000, 60000))
                            
                if not connected:
                    pipeline_state["telemetry"]["drones"][drone_idx]["connected"] = False
                    if drone_idx == 0:
                        pipeline_state["telemetry"]["connected"] = False
                    
                    try:
                        drone._stop_mavsdk_server()
                    except Exception:
                        pass
                        
                    await asyncio.sleep(2)
                    continue
                    
            else:
                try:
                    await drone.connect(system_address=f"udp://:{port}")
                    async def check_conn_udp():
                        async for state in drone.core.connection_state():
                            if state.is_connected:
                                return True
                    connected = await asyncio.wait_for(check_conn_udp(), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    connected = False
                
                if not connected:
                    pipeline_state["telemetry"]["drones"][drone_idx]["connected"] = False
                    if drone_idx == 0:
                        pipeline_state["telemetry"]["connected"] = False
                    
                    try:
                        drone._stop_mavsdk_server()
                    except Exception:
                        pass
                        
                    await asyncio.sleep(2)
                    continue
            
            pipeline_state["telemetry"]["drones"][drone_idx]["connected"] = True
            
            # Reset stale telemetry data from previous connection
            pipeline_state["telemetry"]["drones"][drone_idx]["latitude"] = 0.0
            pipeline_state["telemetry"]["drones"][drone_idx]["longitude"] = 0.0
            pipeline_state["telemetry"]["drones"][drone_idx]["altitude_m"] = 0.0
            pipeline_state["telemetry"]["drones"][drone_idx]["speed_mps"] = 0.0
            pipeline_state["telemetry"]["drones"][drone_idx]["heading_deg"] = 0.0
            pipeline_state["telemetry"]["drones"][drone_idx]["battery_pct"] = 0.0
            
            swarm_drones[drone_idx] = drone
            
            if drone_idx == 0:
                pipeline_state["telemetry"]["connected"] = True
                pipeline_state["telemetry"]["latitude"] = 0.0
                pipeline_state["telemetry"]["longitude"] = 0.0
                pipeline_state["telemetry"]["altitude_m"] = 0.0
                pipeline_state["telemetry"]["speed_mps"] = 0.0
                pipeline_state["telemetry"]["heading_deg"] = 0.0
                pipeline_state["telemetry"]["battery_pct"] = 0.0
                
            async def run_position():
                try:
                    async for pos in drone.telemetry.position():
                        pipeline_state["telemetry"]["drones"][drone_idx]["latitude"] = pos.latitude_deg
                        pipeline_state["telemetry"]["drones"][drone_idx]["longitude"] = pos.longitude_deg
                        pipeline_state["telemetry"]["drones"][drone_idx]["altitude_m"] = pos.relative_altitude_m
                        if drone_idx == 0:
                            pipeline_state["telemetry"]["latitude"] = pos.latitude_deg
                            pipeline_state["telemetry"]["longitude"] = pos.longitude_deg
                            pipeline_state["telemetry"]["altitude_m"] = pos.relative_altitude_m
                            shared_telemetry_state["position"] = pos
                            update_telemetry_timestamp()
                except Exception:
                    pass

            async def run_home():
                try:
                    async for home_pos in drone.telemetry.home():
                        if drone_idx == 0:
                            shared_telemetry_state["home"] = home_pos
                except Exception:
                    pass

            async def run_armed():
                try:
                    async for armed in drone.telemetry.armed():
                        pipeline_state["telemetry"]["drones"][drone_idx]["armed"] = armed
                        if drone_idx == 0:
                            pipeline_state["telemetry"]["armed"] = armed
                            update_telemetry_timestamp()
                except Exception:
                    pass

            async def run_heading():
                try:
                    async for heading in drone.telemetry.heading():
                        pipeline_state["telemetry"]["drones"][drone_idx]["heading_deg"] = heading.heading_deg
                        if drone_idx == 0:
                            pipeline_state["telemetry"]["heading_deg"] = heading.heading_deg
                            update_telemetry_timestamp()
                except Exception:
                    pass

            async def run_speed():
                try:
                    async for velocity in drone.telemetry.velocity_ned():
                        speed = (velocity.north_m_s**2 + velocity.east_m_s**2 + velocity.down_m_s**2)**0.5
                        pipeline_state["telemetry"]["drones"][drone_idx]["speed_mps"] = speed
                        if drone_idx == 0:
                            pipeline_state["telemetry"]["speed_mps"] = speed
                            update_telemetry_timestamp()
                except Exception:
                    pass

            async def run_battery():
                try:
                    async for battery in drone.telemetry.battery():
                        pct = battery.remaining_percent
                        if pct <= 1.0:
                            pct *= 100
                        pipeline_state["telemetry"]["drones"][drone_idx]["battery_pct"] = pct
                        if drone_idx == 0:
                            pipeline_state["telemetry"]["battery_pct"] = pct
                            update_telemetry_timestamp()
                except Exception:
                    pass

            async def run_status_text():
                try:
                    async for status_text in drone.telemetry.status_text():
                        update_telemetry_timestamp()
                        severity = str(status_text.type)
                        message = status_text.text
                        status_val = "info"
                        if "WARN" in severity or "NOTICE" in severity:
                            status_val = "warning"
                        elif "ERROR" in severity or "CRITICAL" in severity or "EMERGENCY" in severity or "ALERT" in severity:
                            status_val = "error"
                        
                        pipeline_state["drone_logs"][drone_idx].append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "status": status_val,
                            "message": f"[PX4] {severity}: {message}"
                        })
                        if len(pipeline_state["drone_logs"][drone_idx]) > 100:
                            pipeline_state["drone_logs"][drone_idx] = pipeline_state["drone_logs"][drone_idx][-100:]
                except Exception as e:
                    print(f"Error monitoring status text for drone {drone_idx}: {e}")

            async def run_continuous_log():
                try:
                    while True:
                        await asyncio.sleep(2.0)
                        if pipeline_state["telemetry"]["drones"][drone_idx]["connected"]:
                            alt = pipeline_state["telemetry"]["drones"][drone_idx]["altitude_m"]
                            spd = pipeline_state["telemetry"]["drones"][drone_idx]["speed_mps"]
                            hdg = pipeline_state["telemetry"]["drones"][drone_idx]["heading_deg"]
                            pipeline_state["drone_logs"][drone_idx].append({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "status": "info",
                                "message": f"[Live Telemetry] Alt: {alt:.1f}m | Speed: {spd:.1f}m/s | Heading: {hdg:.0f}°"
                            })
                            if len(pipeline_state["drone_logs"][drone_idx]) > 100:
                                pipeline_state["drone_logs"][drone_idx] = pipeline_state["drone_logs"][drone_idx][-100:]
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            tasks = [
                asyncio.create_task(run_position()),
                asyncio.create_task(run_armed()),
                asyncio.create_task(run_heading()),
                asyncio.create_task(run_speed()),
                asyncio.create_task(run_battery()),
                asyncio.create_task(run_status_text()),
                asyncio.create_task(run_continuous_log()),
            ]
            if drone_idx == 0:
                tasks.append(asyncio.create_task(run_home()))
                global shared_drone
                shared_drone = drone

            # Monitor connection mode changes or disconnection
            initial_mode = pipeline_state.get("connection_mode")
            while True:
                current_mode = pipeline_state.get("connection_mode")
                if current_mode != initial_mode:
                    break # Connection mode changed, reconnect
                
                if not is_hw and not is_simulator_running():
                    break # Simulator killed
                
                await asyncio.sleep(1)
                
            # Cleanup when breaking out of the loop
            for t in tasks:
                t.cancel()
            
            try:
                drone._stop_mavsdk_server()
            except Exception:
                pass
            
            pipeline_state["telemetry"]["drones"][drone_idx]["connected"] = False
            if drone_idx == 0:
                pipeline_state["telemetry"]["connected"] = False

        except Exception as e:
            print(f"Error in monitor task for drone {drone_idx}: {e}")
            try:
                drone._stop_mavsdk_server()
            except Exception:
                pass
            await asyncio.sleep(2)


async def telemetry_background_task():
    tasks = [
        asyncio.create_task(monitor_single_drone_task(i, 14540 + i))
        for i in range(5)
    ]
    await asyncio.gather(*tasks)

@app.on_event("startup")
async def startup_event():
    # Kill any orphaned processes from previous runs on startup
    import subprocess
    try:
        subprocess.run(["killall", "-9", "px4", "gzserver", "gzclient", "ruby", "make", "sitl_multiple_run.sh", "mavsdk_server"], capture_output=True)
        subprocess.run(["fuser", "-k", "-9", "14560/udp", "14561/udp", "14562/udp", "14580/udp", "18570/udp"], capture_output=True)
    except Exception:
        pass
    asyncio.create_task(telemetry_background_task())


# ── Serve frontend ──
DASHBOARD_DIR = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse(
        DASHBOARD_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


@app.get("/style.css")
async def serve_css():
    return FileResponse(
        DASHBOARD_DIR / "style.css",
        media_type="text/css",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


# ── API Endpoints ──

@app.get("/api/status")
async def get_status():
    """Get current pipeline status and telemetry."""
    return pipeline_state


@app.get("/api/routes")
async def get_routes():
    """Get available predefined routes."""
    library = load_waypoint_library()
    routes = {}
    for name, data in library.items():
        routes[name] = {
            "description": data["description"],
            "default_altitude_m": data["default_altitude_m"],
            "default_speed_mps": data["default_speed_mps"],
            "waypoint_count": len(data["waypoints"]),
            "waypoints": data["waypoints"],
        }
    return routes


@app.get("/api/safety")
async def get_safety():
    """Get current safety limits."""
    return load_safety_limits()


@app.post("/api/safety")
async def update_safety(body: dict):
    """Update safety limits."""
    import yaml
    from pathlib import Path
    
    current_limits = load_safety_limits()
    # Update only the allowed editable numeric parameters
    editable_keys = [
        "max_altitude_m", "max_speed_mps", "geofence_radius_m",
        "max_flight_time_s", "max_leg_distance_m", "min_waypoint_spacing_m"
    ]
    for key in editable_keys:
        if key in body:
            current_limits[key] = float(body[key])

    config_path = Path(__file__).resolve().parent.parent / "config" / "safety_limits.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(current_limits, f, default_flow_style=False, sort_keys=False)
        
    return {"success": True, "limits": current_limits}


@app.get("/api/detect-ports")
async def detect_ports():
    """Scan and return available serial ports for telemetry."""
    import serial.tools.list_ports
    ports = []
    
    # Typically telemetry radios or pixhawks show up as ttyUSB or ttyACM
    for p in serial.tools.list_ports.comports():
        device = p.device
        desc = p.description
        if "ttyUSB" in device or "ttyACM" in device or "COM" in device:
            accessible = os.access(device, os.R_OK | os.W_OK)
            port_info = {
                "device": device,
                "description": desc,
                "accessible": accessible
            }
            if not accessible and os.name == 'posix':
                port_info["error"] = "permission_denied"
            ports.append(port_info)
            
    return {
        "ports": ports,
        "fix_hint": f"Run: sudo usermod -aG dialout {os.environ.get('USER', '$USER')}" if any(p.get("error") == "permission_denied" for p in ports) else ""
    }

@app.post("/api/set-connection-mode")
async def set_connection_mode(body: dict):
    """Switch between simulation and hardware connection mode."""
    mode = body.get("mode", "simulation")
    ports = body.get("ports", [])
    baud_rate = body.get("baud_rate", 57600)
    
    pipeline_state["connection_mode"] = mode
    if mode == "hardware":
        pipeline_state["hardware_ports"] = ports
        pipeline_state["baud_rate"] = baud_rate
    else:
        pipeline_state["hardware_ports"] = []
        
    return {"success": True, "mode": mode}


@app.post("/api/plan")
async def plan(body: dict):
    """Stage 1: Generate mission plan from prompt."""
    prompt = body.get("prompt", "")
    mode = body.get("mode", "standard")
    ai_engine = body.get("ai_engine", "offline")
    api_key = body.get("api_key")
    num_drones = int(body.get("num_drones", 3))

    pipeline_state["status"] = "planning"
    pipeline_state["current_mode"] = mode
    pipeline_state["num_drones"] = num_drones

    try:
        mission = plan_mission(prompt, mode=mode, ai_engine=ai_engine, api_key=api_key, num_drones=num_drones)

        pipeline_state["last_mission"] = mission
        pipeline_state["status"] = "planned"

        return {
            "success": True,
            "mission": mission,
        }
    except Exception as e:
        pipeline_state["status"] = "error"
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/validate")
async def validate(body: dict):
    """Stage 2: Validate mission JSON."""
    mission = body.get("mission") or pipeline_state.get("last_mission")
    connection_mode = pipeline_state.get("connection_mode", "simulation")
    if not mission:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No mission to validate"},
        )

    pipeline_state["status"] = "validating"

    result = validate_mission(mission)
    pipeline_state["last_validation"] = {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
    }

    pipeline_state["status"] = "validated" if result.valid else "validation_failed"

    if result.valid:
        if connection_mode == "hardware":
            hw_result = validate_hardware_safety(mission, pipeline_state["telemetry"])
            result.valid = hw_result.valid
            result.errors.extend(hw_result.errors)
            result.warnings.extend(hw_result.warnings)
            
            pipeline_state["last_validation"] = {
                "valid": result.valid,
                "errors": result.errors,
                "warnings": result.warnings,
            }
            pipeline_state["status"] = "validated" if result.valid else "validation_failed"

    if result.valid:
        mode = pipeline_state.get("current_mode", "standard")
        num_drones = pipeline_state.get("num_drones", 3)
        
        if connection_mode == "hardware":
            pass # Simulator not needed for hardware mode
        else:
            # Only start if not already running in this mode
            if not (is_simulator_running() and pipeline_state.get("active_sim_mode") == mode and pipeline_state["telemetry"]["connected"]):
                pipeline_state["telemetry"]["connected"] = False
                ensure_simulator_running(mode, num_drones)
                pipeline_state["active_sim_mode"] = mode
                
                # Wait for telemetry to connect
                for _ in range(120):  # Wait up to 60 seconds (120 * 0.5s)
                    if pipeline_state["telemetry"]["connected"]:
                        break
                    await asyncio.sleep(0.5)

    return {
        "success": result.valid,
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
    }


# Moved is_simulator_running to the top of the file


def ensure_simulator_running(mode="standard", num_drones=3):
    if pipeline_state.get("connection_mode") == "hardware":
        return

    import subprocess
    import os
    try:
        subprocess.run(["killall", "-9", "px4", "gzserver", "gzclient", "ruby", "make", "sitl_multiple_run.sh"], capture_output=True)
        subprocess.run(["fuser", "-k", "-9", "14560/udp", "14561/udp", "14562/udp", "14580/udp", "18570/udp"], capture_output=True)
    except Exception:
        pass
    project_root = Path(__file__).resolve().parent.parent
    launch_script = project_root / "launch_sim.sh"
    env = os.environ.copy()
    if "DISPLAY" in env:
        env.pop("HEADLESS", None)
    else:
        env["HEADLESS"] = "1"
    env["NUM_DRONES"] = str(num_drones)
    sim_process = subprocess.Popen(
        ["bash", str(launch_script), "--mode", mode],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    pipeline_state["sim_process"] = sim_process


async def run_mission_background(mission: dict):
    pipeline_state["status"] = "executing"
    pipeline_state["execution_log"] = []
    pipeline_state["drone_logs"] = [[] for _ in range(5)]

    def on_audit_record(entry):
        pipeline_state["execution_log"].append(entry)

    audit_log = AuditLog(callback=on_audit_record)
    mode = pipeline_state.get("current_mode", "standard")
    num_drones = pipeline_state.get("num_drones", 3)

    connection_mode = pipeline_state.get("connection_mode", "simulation")

    if connection_mode == "hardware":
        # Check if hardware is connected
        if not pipeline_state["telemetry"]["connected"]:
            pipeline_state["status"] = "error"
            pipeline_state["execution_log"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": 0.0,
                "action": "error",
                "status": "error",
                "details": {"message": "Hardware drone is not connected via serial."},
            })
            return
        audit_log.record("hardware_connected", {"message": "Executing mission on hardware via serial link."})
    else:
        # Check if simulator is already running and connected
        if is_simulator_running() and pipeline_state["telemetry"]["connected"] and pipeline_state.get("active_sim_mode") == mode:
            audit_log.record("simulator_connected", {"message": "Simulator already running and connected."})
        else:
            pipeline_state["telemetry"]["connected"] = False
            audit_log.record("start_simulator", {"message": f"Starting simulator in '{mode}' mode with {num_drones} drones..."})
            ensure_simulator_running(mode, num_drones)
            pipeline_state["active_sim_mode"] = mode
            
            # Wait for the simulator to launch and telemetry to connect
            connected = False
            for _ in range(240):  # Wait up to 120 seconds (240 * 0.5s)
                if pipeline_state["telemetry"]["connected"]:
                    connected = True
                    break
                await asyncio.sleep(0.5)
    
            if not connected:
                pipeline_state["status"] = "error"
                pipeline_state["execution_log"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": 0.0,
                    "action": "error",
                    "status": "error",
                    "details": {"message": f"Failed to connect to simulator in '{mode}' mode."},
                })
                return
            audit_log.record("simulator_connected", {"message": f"Simulator connected successfully in '{mode}' mode."})

    try:
        if mode == "swarm":
            from src.executors.swarm_executor import execute_swarm_mission
            # Ensure mission has drone_count from user selection (overrides LLM if different)
            mission["drone_count"] = num_drones
            await execute_swarm_mission(mission, drone=shared_drone, audit=audit_log, telemetry_state=shared_telemetry_state)
        else:
            from src.executor import execute_mission_async
            await execute_mission_async(mission, drone=shared_drone, audit=audit_log, telemetry_state=shared_telemetry_state)
        pipeline_state["status"] = "completed"
    except Exception as e:
        pipeline_state["status"] = "error"
        pipeline_state["execution_log"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": 0.0,
            "action": "error",
            "status": "error",
            "details": {"message": str(e)},
        })


active_mission_task = None

@app.post("/api/execute")
async def execute(body: dict):
    """Stage 3: Execute validated mission asynchronously."""
    global active_mission_task
    
    mission = body.get("mission") or pipeline_state.get("last_mission")
    if not mission:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No mission to execute"},
        )
    if pipeline_state.get("connection_mode") == "hardware" and not body.get("hardware_confirmed"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Hardware execution must be explicitly confirmed by the operator."},
        )
    
    # Inject num_drones from pipeline state
    mission["num_drones"] = pipeline_state.get("num_drones", 3)

    # Check validation
    validation = pipeline_state.get("last_validation")
    if not validation or not validation.get("valid"):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Mission must be validated before execution",
            },
        )

    # Cancel previous running mission if it exists
    if active_mission_task and not active_mission_task.done():
        print("Cancelling previous background mission execution...")
        active_mission_task.cancel()

    active_mission_task = asyncio.create_task(run_mission_background(mission))
    return {"success": True, "message": "Mission execution started"}


@app.post("/api/terminate")
async def terminate_mission():
    """Terminate the current mission and kill simulator."""
    global active_mission_task
    if active_mission_task and not active_mission_task.done():
        print("Cancelling active mission task...")
        active_mission_task.cancel()
    
    import subprocess
    import os
    import signal
    print("Killing simulator processes...")
    sim_process = pipeline_state.get("sim_process")
    if sim_process:
        try:
            os.killpg(os.getpgid(sim_process.pid), signal.SIGTERM)
        except Exception:
            pass
        pipeline_state["sim_process"] = None

    try:
        subprocess.run(["killall", "-9", "px4", "gzserver", "gzclient", "ruby", "make", "sitl_multiple_run.sh", "mavsdk_server"], capture_output=True)
    except Exception:
        pass
    
    pipeline_state["status"] = "idle"
    pipeline_state["drone_logs"] = [[] for _ in range(5)]
    pipeline_state["execution_log"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": 0.0,
        "action": "terminated",
        "status": "error",
        "details": {"message": "Mission terminated by user. Simulator killed."},
    })
    
    return {"success": True, "message": "Mission terminated successfully"}



@app.post("/api/action/formation")
async def action_formation(body: dict):
    '''Set manual formation.'''
    formation = body.get("formation", "wedge")
    pipeline_state["manual_formation"] = formation
    pipeline_state["execution_log"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": 0.0,
        "action": "set formation",
        "status": "success",
        "details": {"message": f"Manual formation set to {formation}"},
    })
    return {"success": True}

@app.post("/api/action/hold")
async def action_hold():
    """Live hold command."""
    global active_mission_task
    if 'hold' in ('hold', 'rtl'):
        if active_mission_task and not active_mission_task.done():
            active_mission_task.cancel()
    
    try:
        if pipeline_state.get("num_drones", 1) > 1 and len(swarm_drones) > 1:
            tasks = [d.action.hold() for d in swarm_drones.values() if d]
            await asyncio.gather(*tasks)
        else:
            await shared_drone.action.hold()
            
        if 'hold' in ('hold', 'rtl'):
            pipeline_state["status"] = 'holding' if 'hold' == 'hold' else 'returning'
            
        pipeline_state["execution_log"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": 0.0,
            "action": "hold (Live)",
            "status": "success",
            "details": {"message": "Holding position"},
        })
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/action/rtl")
async def action_rtl():
    """Live rtl command."""
    global active_mission_task
    if 'rtl' in ('hold', 'rtl'):
        if active_mission_task and not active_mission_task.done():
            active_mission_task.cancel()
    
    try:
        if pipeline_state.get("num_drones", 1) > 1 and len(swarm_drones) > 1:
            tasks = [d.action.return_to_launch() for d in swarm_drones.values() if d]
            await asyncio.gather(*tasks)
        else:
            await shared_drone.action.return_to_launch()
            
        if 'rtl' in ('hold', 'rtl'):
            pipeline_state["status"] = 'rtling' if 'rtl' == 'hold' else 'returning'
            
        pipeline_state["execution_log"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": 0.0,
            "action": "rtl (Live)",
            "status": "success",
            "details": {"message": "Returning to launch"},
        })
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/action/arm")
async def action_arm():
    """Live arm command."""
    global active_mission_task
    if 'arm' in ('hold', 'rtl'):
        if active_mission_task and not active_mission_task.done():
            active_mission_task.cancel()
    
    try:
        if pipeline_state.get("num_drones", 1) > 1 and len(swarm_drones) > 1:
            tasks = [d.action.arm() for d in swarm_drones.values() if d]
            await asyncio.gather(*tasks)
        else:
            await shared_drone.action.arm()
            
        if 'arm' in ('hold', 'rtl'):
            pipeline_state["status"] = 'arming' if 'arm' == 'hold' else 'returning'
            
        pipeline_state["execution_log"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": 0.0,
            "action": "arm (Live)",
            "status": "success",
            "details": {"message": "Drones armed manually"},
        })
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/action/disarm")
async def action_disarm():
    """Live disarm command."""
    global active_mission_task
    if 'disarm' in ('hold', 'rtl'):
        if active_mission_task and not active_mission_task.done():
            active_mission_task.cancel()
    
    try:
        if pipeline_state.get("num_drones", 1) > 1 and len(swarm_drones) > 1:
            tasks = [d.action.disarm() for d in swarm_drones.values() if d]
            await asyncio.gather(*tasks)
        else:
            await shared_drone.action.disarm()
            
        if 'disarm' in ('hold', 'rtl'):
            pipeline_state["status"] = 'disarming' if 'disarm' == 'hold' else 'returning'
            
        pipeline_state["execution_log"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": 0.0,
            "action": "disarm (Live)",
            "status": "success",
            "details": {"message": "Drones disarmed manually"},
        })
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/action/move")
async def action_move(body: dict):
    '''Move drone in a specific direction.'''
    import math
    direction = body.get("direction")
    distance = body.get("distance", 5.0)
    
    pos = shared_telemetry_state.get("position")
    if not pos:
        return {"success": False, "error": "No telemetry position available"}
        
    lat = pos.latitude_deg
    lon = pos.longitude_deg
    alt = pos.absolute_altitude_m
    
    heading = pipeline_state["telemetry"]["heading_deg"]
    
    d_lat = 0
    d_lon = 0
    d_alt = 0
    
    if direction == "up":
        d_alt = distance
    elif direction == "down":
        d_alt = -distance
    elif direction == "front":
        d_lat = (distance * math.cos(math.radians(heading))) / 111111.0
        d_lon = (distance * math.sin(math.radians(heading))) / (111111.0 * math.cos(math.radians(lat)))
    elif direction == "back":
        d_lat = -(distance * math.cos(math.radians(heading))) / 111111.0
        d_lon = -(distance * math.sin(math.radians(heading))) / (111111.0 * math.cos(math.radians(lat)))
    elif direction == "right":
        d_lat = (distance * math.cos(math.radians(heading + 90))) / 111111.0
        d_lon = (distance * math.sin(math.radians(heading + 90))) / (111111.0 * math.cos(math.radians(lat)))
    elif direction == "left":
        d_lat = (distance * math.cos(math.radians(heading - 90))) / 111111.0
        d_lon = (distance * math.sin(math.radians(heading - 90))) / (111111.0 * math.cos(math.radians(lat)))
    else:
        return {"success": False, "error": "Unknown direction"}
        
    new_lat = lat + d_lat
    new_lon = lon + d_lon
    new_alt = alt + d_alt
    
    try:
        tasks = []
        num_drones = pipeline_state.get("num_drones", 1)
        if num_drones > 1 and len(swarm_drones) > 1:
            formation = pipeline_state.get("manual_formation", "wedge")
            offsets = {
                "wedge": [(-5.0, -5.0, 0.0), (-5.0, 5.0, 0.0)],
                "line": [(0.0, -5.0, 0.0), (0.0, 5.0, 0.0)],
                "column": [(-5.0, 0.0, 0.0), (-10.0, 0.0, 0.0)]
            }.get(formation, [(-5.0, -5.0, 0.0), (-5.0, 5.0, 0.0)])
            
            if 0 in swarm_drones:
                tasks.append(swarm_drones[0].action.goto_location(new_lat, new_lon, new_alt, heading))
                
            for i in range(1, num_drones):
                if i in swarm_drones:
                    dx, dy, dz = offsets[i - 1] if i - 1 < len(offsets) else (0, 0, 0)
                    cos_h = math.cos(math.radians(heading))
                    sin_h = math.sin(math.radians(heading))
                    rotated_n = dx * cos_h - dy * sin_h
                    rotated_e = dx * sin_h + dy * cos_h
                    
                    d_lat_f = rotated_n / 111111.0
                    d_lon_f = rotated_e / (111111.0 * math.cos(math.radians(lat)))
                    
                    tasks.append(swarm_drones[i].action.goto_location(new_lat + d_lat_f, new_lon + d_lon_f, new_alt - dz, heading))
                    
            await asyncio.gather(*tasks)
        else:
            await shared_drone.action.goto_location(new_lat, new_lon, new_alt, heading)
            
        pipeline_state["execution_log"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": 0.0,
            "action": f"move {direction}",
            "status": "success",
            "details": {"message": f"Moving {direction} by {distance}m"},
        })
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/plan-validate")
async def plan_and_validate(body: dict):
    """Combined: Plan + Validate in one call."""
    # Plan
    plan_result = await plan(body)
    if isinstance(plan_result, JSONResponse):
        return plan_result

    plan_data = plan_result
    if not plan_data.get("success"):
        return plan_data

    # Validate
    validate_result = await validate({"mission": plan_data["mission"]})

    return {
        "mission": plan_data["mission"],
        "validation": validate_result,
    }


@app.get("/api/check-ollama")
async def check_ollama():
    """Check if Ollama is installed and/or running, and report disk space."""
    import shutil
    import urllib.request
    
    ollama_path = shutil.which("ollama")
    is_installed = ollama_path is not None
    is_running = False
    models = []
    
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
        data = json.loads(req.read().decode())
        models = [m.get("name") for m in data.get("models", [])]
        is_running = True
        is_installed = True
    except Exception:
        pass

    usage = shutil.disk_usage("/")
    free_gb = round(usage.free / (1024**3), 1)
    
    return {
        "installed": is_installed,
        "running": is_running,
        "models": models,
        "free_disk_gb": free_gb,
        "required_gb": 1.5
    }


@app.post("/api/install-ollama")
async def install_ollama():
    """Install Ollama via official installer script using subprocess."""
    import shutil
    import subprocess
    if shutil.which("ollama"):
        return {"success": True, "message": "Ollama is already installed"}
        
    try:
        proc = subprocess.run(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"], capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            return {"success": True, "message": "Ollama installed successfully!"}
        else:
            return {"success": False, "error": f"Install failed: {proc.stderr or proc.stdout}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribes an uploaded WAV audio file using speech_recognition."""
    import tempfile
    import speech_recognition as sr
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source)
            
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
            
        # Transcribe using Google Web Speech (free, no keys required)
        text = r.recognize_google(audio)
        return {"success": True, "text": text}
    except sr.UnknownValueError:
        return {"success": False, "error": "Speech was unintelligible"}
    except sr.RequestError as e:
        return {"success": False, "error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── WebSocket for live telemetry ──

@app.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    """WebSocket endpoint for streaming live telemetry data and execution logs."""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "telemetry": pipeline_state["telemetry"],
                "status": pipeline_state["status"],
                "execution_log": pipeline_state["execution_log"],
                "drone_logs": pipeline_state["drone_logs"],
            })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    import subprocess
    import time
    
    print("🧹 Cleaning up existing web server processes on port 8000...")
    try:
        subprocess.run(["fuser", "-k", "-9", "8000/tcp"], capture_output=True)
        time.sleep(1)
    except Exception:
        pass
        
    print("🚁 Starting Drone Pipeline Dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
