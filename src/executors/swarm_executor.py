"""
swarm_executor.py — Real MAVSDK Multi-Agent Swarm Executor

Connects to 3 PX4 instances (ports 14541, 14542, 14543) spawned by Gazebo SITL
and coordinates them in a Wedge, Line, or Column formation.
"""

import asyncio
import math
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, OffboardError
from ..utils import AuditLog, print_info, print_success, print_error, print_warning, setup_logger

logger = setup_logger("swarm_executor")

WAYPOINT_TOLERANCE_M = 2.0
POSITION_CHECK_INTERVAL_S = 0.5

FORMATION_OFFSETS = {
    "wedge": [
        (0.0, 0.0, 0.0),      # Drone 1 (Leader)
        (-5.0, -5.0, 0.0),    # Drone 2 (Wingman 1 - left & back)
        (-5.0, 5.0, 0.0),     # Drone 3 (Wingman 2 - right & back)
    ],
    "line": [
        (0.0, 0.0, 0.0),      # Drone 1
        (0.0, -5.0, 0.0),     # Drone 2 (Left)
        (0.0, 5.0, 0.0),      # Drone 3 (Right)
    ],
    "column": [
        (0.0, 0.0, 0.0),      # Drone 1
        (-5.0, 0.0, 0.0),     # Drone 2 (Behind)
        (-10.0, 0.0, 0.0),    # Drone 3 (Far Behind)
    ]
}

async def connect_drone(port: int, timeout_s: float = 60.0) -> System:
    """Connect to a PX4 instance on the given UDP port with a timeout."""
    drone = System()
    addr = f"udp://:{port}"
    logger.info(f"Connecting to drone at {addr}...")
    await drone.connect(system_address=addr)
    
    async def _wait_for_ready():
        # Wait for connection
        async for state in drone.core.connection_state():
            if state.is_connected:
                logger.info(f"Drone on port {port} connected!")
                break
                
        # Wait for GPS fix and arming readiness
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok and health.is_armable:
                logger.info(f"Drone on port {port} GPS fix OK and armable")
                break
    
    try:
        await asyncio.wait_for(_wait_for_ready(), timeout=timeout_s)
    except asyncio.TimeoutError:
        raise RuntimeError(f"Drone on port {port} did not become ready within {timeout_s}s")
            
    return drone

async def execute_swarm_mission(mission: dict, drone: System, audit: AuditLog, telemetry_state: dict):
    print_info("Starting Real Swarm Execution...")
    audit.record("swarm_start", {"mission_name": mission.get("mission_name", "swarm_mission")})
    
    formation_name = mission.get("formation", "wedge").lower()
    # LLM produces "drone_count", but also support "num_drones" as fallback
    num_drones = int(mission.get("drone_count", mission.get("num_drones", 3)))
    
    print_info(f"Swarm config: {num_drones} drones, formation={formation_name}")
    
    # Generate dynamic offsets based on num_drones
    if formation_name == "line":
        offsets = [(0.0, 0.0, 0.0)]
        for i in range(1, num_drones):
            sign = -1 if i % 2 == 1 else 1
            step = (i + 1) // 2
            offsets.append((0.0, sign * 5.0 * step, 0.0))
    elif formation_name == "column":
        offsets = [(0.0, 0.0, 0.0)]
        for i in range(1, num_drones):
            offsets.append((-5.0 * i, 0.0, 0.0))
    else: # wedge (default)
        offsets = [(0.0, 0.0, 0.0)]
        for i in range(1, num_drones):
            sign = -1 if i % 2 == 1 else 1
            step = (i + 1) // 2
            offsets.append((-5.0 * step, sign * 5.0 * step, 0.0))
            
    # Connect to all drones
    # PX4 multi-instance: instance i → offboard UDP port 14540+i
    # Leader = instance 0 → port 14540 (already connected as `drone` param)
    # Wingmen = instances 1..N-1 → ports 14541, 14542, ...
    try:
        if num_drones > 1:
            print_info(f"Connecting to {num_drones - 1} wingmen on ports {[14540 + i for i in range(1, num_drones)]}...")
            wingmen = await asyncio.gather(*(
                connect_drone(14540 + i) for i in range(1, num_drones)
            ))
            drones = [drone] + list(wingmen)
        else:
            drones = [drone]
        print_success(f"All {num_drones} drones connected and ready!")
    except Exception as e:
        print_error(f"Failed to connect to wingmen drones: {e}")
        audit.record("swarm_error", {"message": f"Connection error: {e}"})
        raise e
        
    try:
        actions = mission.get("actions", [])
        for step_idx, action in enumerate(actions):
            action_type = action.get("type")
            params = action.get("params", {})
            
            print_info(f"Executing step {step_idx + 1}/{len(actions)}: {action_type}")
            
            if action_type == "takeoff":
                alt = params.get("altitude_m", 10)
                audit.record("swarm_takeoff", {"altitude_m": alt})
                
                # Arm all drones
                print_info("Arming all drones...")
                await asyncio.gather(*(d.action.arm() for d in drones))
                print_success("All drones armed.")
                
                # Set takeoff altitudes and trigger takeoff
                # Wingmen fly slightly different altitudes if needed, but same here for simple formation
                await asyncio.gather(*(d.action.set_takeoff_altitude(alt) for d in drones))
                await asyncio.gather(*(d.action.takeoff() for d in drones))
                print_info("All drones taking off...")
                
                # Wait for takeoff completion
                while True:
                    alts = []
                    for idx, d in enumerate(drones):
                        async for pos in d.telemetry.position():
                            alts.append(pos.relative_altitude_m)
                            break
                    if all(a >= alt * 0.90 for a in alts):
                        print_success("All drones reached takeoff altitude")
                        break
                    await asyncio.sleep(POSITION_CHECK_INTERVAL_S)
                
            elif action_type == "goto":
                leader_n = params.get("north_m", 0.0)
                leader_e = params.get("east_m", 0.0)
                leader_alt = params.get("altitude_m", 10.0)
                
                audit.record("swarm_goto", {"north_m": leader_n, "east_m": leader_e, "altitude_m": leader_alt})
                
                # Calculate headings for rotation of formation if leader is moving
                heading = math.degrees(math.atan2(leader_e, leader_n)) % 360
                heading_rad = math.radians(heading)
                cos_h = math.cos(heading_rad)
                sin_h = math.sin(heading_rad)
                
                targets = []
                for idx, (dx, dy, dz) in enumerate(offsets):
                    # Rotate the offsets based on heading to keep formation aligned with path
                    rotated_n = dx * cos_h - dy * sin_h
                    rotated_e = dx * sin_h + dy * cos_h
                    
                    target_n = leader_n + rotated_n
                    target_e = leader_e + rotated_e
                    target_d = -(leader_alt + dz)
                    targets.append((target_n, target_e, target_d, heading))
                
                # Send setpoint and start offboard mode for all drones
                for idx, d in enumerate(drones):
                    n, e, d_alt, hdg = targets[idx]
                    await d.offboard.set_position_ned(PositionNedYaw(n, e, d_alt, hdg))
                    
                for idx, d in enumerate(drones):
                    try:
                        await d.offboard.start()
                    except OffboardError as oe:
                        if "already active" not in str(oe).lower():
                            raise oe
                            
                # Command target positions continuously
                for idx, d in enumerate(drones):
                    n, e, d_alt, hdg = targets[idx]
                    await d.offboard.set_position_ned(PositionNedYaw(n, e, d_alt, hdg))
                    
                # Monitor progress
                while True:
                    reached = []
                    for idx, d in enumerate(drones):
                        target_n, target_e, target_d, _ = targets[idx]
                        async for pos in d.telemetry.position():
                            # We need home position to get NED position
                            # For simplicity we query home position or calculate relative offset
                            # Actually, we can get offboard position or NED position from velocity_ned
                            # telemetry.position() gives lat/lon/alt.
                            # MAVSDK telemetry has 'position_velocity_ned' which is very convenient!
                            break
                        
                        async for pos_ned in d.telemetry.position_velocity_ned():
                            curr_n = pos_ned.position.north_m
                            curr_e = pos_ned.position.east_m
                            dist = math.sqrt((curr_n - target_n)**2 + (curr_e - target_e)**2)
                            reached.append(dist <= WAYPOINT_TOLERANCE_M)
                            break
                            
                    if all(reached) and len(reached) == len(drones):
                        print_success("Swarm reached waypoint!")
                        break
                    await asyncio.sleep(POSITION_CHECK_INTERVAL_S)
                    
            elif action_type == "land":
                audit.record("swarm_land")
                print_info("Landing all drones...")
                await asyncio.gather(*(d.action.land() for d in drones))
                
                # Wait for disarm
                while True:
                    armed_states = []
                    for d in drones:
                        async for armed in d.telemetry.armed():
                            armed_states.append(armed)
                            break
                    if not any(armed_states):
                        print_success("All drones landed and disarmed")
                        break
                    await asyncio.sleep(POSITION_CHECK_INTERVAL_S)
                    
            elif action_type == "return_to_launch":
                audit.record("swarm_rtl")
                print_info("Returning to launch for all drones...")
                await asyncio.gather(*(d.action.return_to_launch() for d in drones))
                
                # Wait for disarm
                while True:
                    armed_states = []
                    for d in drones:
                        async for armed in d.telemetry.armed():
                            armed_states.append(armed)
                            break
                    if not any(armed_states):
                        print_success("All drones returned home and disarmed")
                        break
                    await asyncio.sleep(POSITION_CHECK_INTERVAL_S)
                    
    except Exception as e:
        print_error(f"Swarm mission error: {e}")
        audit.record("swarm_error", {"details": str(e)})
        # Attempt to land all drones safely in case of failure
        print_warning("Emergency landing swarm...")
        await asyncio.gather(*(d.action.land() for d in drones if d))
        raise e
        
    audit.record("swarm_completed")
    print_success("Swarm Mission Completed Successfully!")
