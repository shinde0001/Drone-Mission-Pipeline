"""
executor.py — Stage 3: Deterministic Mission Executor

This module reads a VALIDATED mission JSON and issues concrete commands
to the PX4 drone via MAVSDK-Python. It is the bridge between the
validated plan and the actual vehicle.

KEY GUARANTEES:
  1. DETERMINISTIC — The same JSON always produces the exact same
     command sequence. No randomness, no LLM, no AI in this layer.
  2. AUDITABLE — Every command is logged with timestamps in an
     audit trail (AuditLog).
  3. SEQUENTIAL — Actions execute one-by-one in order. The executor
     waits for each action to complete before starting the next.

The LLM is NEVER in the control loop here. This module only reads
the validated JSON and maps action types to MAVSDK API calls.
"""

import asyncio
import math
from mavsdk import System
from mavsdk.offboard import (
    OffboardError,
    PositionNedYaw,
    VelocityNedYaw,
)

from .utils import (
    AuditLog,
    print_stage,
    print_success,
    print_error,
    print_info,
    print_warning,
    setup_logger,
)

logger = setup_logger("executor")

# ── How close (metres) we must be to a waypoint to consider it "reached" ──
WAYPOINT_TOLERANCE_M = 2.0
# ── How often to check position (seconds) ──
POSITION_CHECK_INTERVAL_S = 0.5


async def _connect_drone(system_address: str = "udp://:14540") -> System:
    """Connect to the drone via MAVSDK."""
    drone = System()
    logger.info(f"Connecting to drone at {system_address}...")
    await drone.connect(system_address=system_address)

    # Wait for the drone to connect
    print_info("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print_success("Drone connected!")
            break

    # Wait for GPS fix and armable state
    print_info("Waiting for global position estimate and armable state...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok and health.is_armable:
            print_success("Drone is fully ready and armable!")
            break

    return drone


async def _execute_takeoff(drone: System, params: dict,
                           audit: AuditLog, *args, **kwargs) -> None:
    """Execute a takeoff action."""
    altitude = params.get("altitude_m", 10)
    logger.info(f"Takeoff to {altitude}m")
    audit.record("takeoff", {"altitude_m": altitude})

    # Arm the drone
    print_info("Arming drone...")
    await drone.action.arm()
    print_success("Drone armed")
    audit.record("arm")

    # Set takeoff altitude and take off
    await drone.action.set_takeoff_altitude(altitude)
    await drone.action.takeoff()
    print_info(f"Taking off to {altitude}m...")

    # Wait until we reach roughly the target altitude
    telemetry_state = kwargs.get("telemetry_state")
    if telemetry_state:
        while True:
            pos = telemetry_state.get("position")
            if pos and pos.relative_altitude_m >= altitude * 0.90:
                print_success(f"Reached {pos.relative_altitude_m:.1f}m "
                              f"(target: {altitude}m)")
                break
            await asyncio.sleep(POSITION_CHECK_INTERVAL_S)
    else:
        async for position in drone.telemetry.position():
            if position.relative_altitude_m >= altitude * 0.90:
                print_success(f"Reached {position.relative_altitude_m:.1f}m "
                              f"(target: {altitude}m)")
                break
    await asyncio.sleep(2)  # Stabilize


async def _execute_goto(drone: System, params: dict,
                        audit: AuditLog, *args, **kwargs) -> None:
    """Execute a goto action using offboard mode with NED coordinates."""
    north = params.get("north_m", 0)
    east = params.get("east_m", 0)
    altitude = params.get("altitude_m", 10)
    speed = params.get("speed_mps", 5)

    # NED uses Down, so altitude is negative
    down = -altitude

    logger.info(f"Goto N={north}m E={east}m Alt={altitude}m @ {speed}m/s")
    audit.record("goto", {
        "north_m": north, "east_m": east,
        "altitude_m": altitude, "speed_mps": speed
    })

    # Set an initial setpoint before starting offboard mode
    await drone.offboard.set_position_ned(
        PositionNedYaw(north, east, down, 0.0)
    )

    # Start offboard mode
    try:
        await drone.offboard.start()
        print_info("Offboard mode started")
    except OffboardError as e:
        # May already be in offboard mode
        if "already active" not in str(e).lower():
            logger.warning(f"Offboard start note: {e}")

    # Send position command
    # Calculate heading towards the target
    heading = math.degrees(math.atan2(east, north)) % 360
    await drone.offboard.set_position_ned(
        PositionNedYaw(north, east, down, heading)
    )

    print_info(f"  → Flying to N={north} E={east} Alt={altitude}m "
               f"Hdg={heading:.0f}°")

    telemetry_state = kwargs.get("telemetry_state")
    if telemetry_state:
        while True:
            pause_event = telemetry_state.get("pause_event")
            if pause_event and not pause_event.is_set():
                print_info("Mission paused. Waiting for resume...")
                await pause_event.wait()
                print_info("Mission resumed. Re-engaging offboard mode...")
                await drone.offboard.set_position_ned(PositionNedYaw(north, east, down, heading))
                try:
                    await drone.offboard.start()
                except OffboardError:
                    pass

            pos = telemetry_state.get("position")
            home = telemetry_state.get("home")
            if pos and home:
                # Calculate NED coordinates from GPS (WGS-84 approximation)
                R = 6378137.0
                d_lat = math.radians(pos.latitude_deg - home.latitude_deg)
                d_lon = math.radians(pos.longitude_deg - home.longitude_deg)
                
                curr_north = d_lat * R
                curr_east = d_lon * R * math.cos(math.radians(home.latitude_deg))
                curr_down = -pos.relative_altitude_m
                
                dist = math.sqrt(
                    (curr_north - north) ** 2 +
                    (curr_east - east) ** 2 +
                    (curr_down - down) ** 2
                )
                if dist < WAYPOINT_TOLERANCE_M:
                    print_success(
                        f"  ✓ Reached waypoint (dist={dist:.1f}m)"
                    )
                    break
            await asyncio.sleep(POSITION_CHECK_INTERVAL_S)
    else:
        # Get home position
        home = None
        async for h in drone.telemetry.home():
            home = h
            break

        if home is None:
            raise RuntimeError("Could not retrieve home position for coordinate conversion.")

        # Wait until we reach the waypoint (check NED distance calculated from GPS)
        async for pos in drone.telemetry.position():
            # Calculate NED coordinates from GPS (WGS-84 approximation)
            R = 6378137.0  # Earth's equatorial radius in meters
            d_lat = math.radians(pos.latitude_deg - home.latitude_deg)
            d_lon = math.radians(pos.longitude_deg - home.longitude_deg)
            
            curr_north = d_lat * R
            curr_east = d_lon * R * math.cos(math.radians(home.latitude_deg))
            curr_down = -pos.relative_altitude_m
            
            dist = math.sqrt(
                (curr_north - north) ** 2 +
                (curr_east - east) ** 2 +
                (curr_down - down) ** 2
            )
            if dist < WAYPOINT_TOLERANCE_M:
                print_success(
                    f"  ✓ Reached waypoint (dist={dist:.1f}m)"
                )
                break
            await asyncio.sleep(POSITION_CHECK_INTERVAL_S)

    audit.record("waypoint_reached", {"north_m": north, "east_m": east})


async def _execute_loiter(drone: System, params: dict,
                          audit: AuditLog, *args, **kwargs) -> None:
    """Hold current position for a specified duration."""
    duration = params.get("duration_s", 5)
    logger.info(f"Loiter for {duration}s")
    audit.record("loiter", {"duration_s": duration})

    print_info(f"  ⏱ Hovering for {duration}s...")
    await asyncio.sleep(duration)
    print_success(f"  ✓ Loiter complete ({duration}s)")


async def _execute_land(drone: System, params: dict,
                        audit: AuditLog, *args, **kwargs) -> None:
    """Land the drone at current position."""
    logger.info("Landing")
    audit.record("land")

    # Stop offboard mode if active
    try:
        await drone.offboard.stop()
    except Exception:
        pass  # May not be in offboard mode

    await drone.action.land()
    print_info("Landing...")

    # Wait for disarm (landed)
    async for armed in drone.telemetry.armed():
        if not armed:
            print_success("Drone landed and disarmed")
            break

    audit.record("landed")


async def _execute_return_to_launch(drone: System, params: dict,
                                    audit: AuditLog, *args, **kwargs) -> None:
    """Return to launch position and land."""
    logger.info("Return to launch")
    audit.record("return_to_launch")

    try:
        await drone.offboard.stop()
    except Exception:
        pass

    await drone.action.return_to_launch()
    print_info("Returning to launch point...")

    # Wait for disarm (landed)
    async for armed in drone.telemetry.armed():
        if not armed:
            print_success("Returned to launch and landed")
            break

    audit.record("rtl_complete")


# ── Action dispatcher ────────────────────────────────────────────────
ACTION_HANDLERS = {
    "takeoff": _execute_takeoff,
    "goto": _execute_goto,
    "loiter": _execute_loiter,
    "land": _execute_land,
    "return_to_launch": _execute_return_to_launch,
}


async def execute_mission_async(mission: dict,
                                system_address: str = "udp://:14540",
                                drone: System = None,
                                audit: AuditLog = None,
                                telemetry_state: dict = None
                                ) -> AuditLog:
    """
    Stage 3: Execute a validated mission JSON on the drone.

    This function is DETERMINISTIC: the same mission JSON will always
    produce the exact same sequence of MAVSDK commands. The LLM is
    never consulted during execution.

    Args:
        mission: Validated mission JSON dict.
        system_address: MAVSDK connection string to PX4 SITL.
        drone: Optional pre-connected MAVSDK System instance.
        audit: Optional pre-configured AuditLog instance.
        telemetry_state: Optional shared telemetry state dictionary.

    Returns:
        AuditLog with timestamped record of every command issued.
    """
    if audit is None:
        audit = AuditLog()
    mission_name = mission.get("mission_name", "unnamed")
    repeat_count = mission.get("repeat_count", 1)
    actions = mission.get("actions", [])

    print_stage(3, f"Executing Mission: {mission_name}")
    audit.record("mission_start", {
        "name": mission_name,
        "repeat_count": repeat_count,
        "total_actions": len(actions),
    })

    # Connect to drone
    if drone is None:
        drone = await _connect_drone(system_address)
    else:
        # Wait for GPS fix and armability even if pre-connected drone is passed
        print_info("Waiting for global position estimate and armable state...")
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok and health.is_armable:
                print_success("Drone is fully ready and armable!")
                break

    # Start telemetry monitoring tasks once for the duration of the mission
    pos_task = None
    home_task = None

    if telemetry_state is None:
        telemetry_state = {"position": None, "home": None}
        
        async def _update_position_stream():
            try:
                async for pos in drone.telemetry.position():
                    telemetry_state["position"] = pos
            except asyncio.CancelledError:
                pass

        async def _update_home_stream():
            try:
                async for home_pos in drone.telemetry.home():
                    telemetry_state["home"] = home_pos
            except asyncio.CancelledError:
                pass

        pos_task = asyncio.create_task(_update_position_stream())
        home_task = asyncio.create_task(_update_home_stream())

    try:
        # Separate takeoff/land from repeatable core actions
        takeoff_action = actions[0] if actions[0]["type"] == "takeoff" else None
        land_action = (actions[-1] if actions[-1]["type"] in
                       ("land", "return_to_launch") else None)
        
        # Keep only goto and loiter in the core repeatable loop to prevent mid-mission landing/re-arming hangs
        raw_core_actions = actions[
            (1 if takeoff_action else 0):
            (-1 if land_action else len(actions))
        ]
        core_actions = [a for a in raw_core_actions if a["type"] not in ("takeoff", "land", "return_to_launch")]

        # Execute takeoff (once)
        if takeoff_action:
            handler = ACTION_HANDLERS[takeoff_action["type"]]
            await handler(drone, takeoff_action.get("params", {}), audit, telemetry_state=telemetry_state)

        # Execute core actions (repeated)
        for loop in range(repeat_count):
            if repeat_count > 1:
                print_info(f"\n── Loop {loop + 1}/{repeat_count} ──")
                audit.record("loop_start", {"loop": loop + 1,
                                            "of": repeat_count})

            for i, action in enumerate(core_actions):
                action_type = action["type"]
                params = action.get("params", {})

                print_info(
                    f"  Action {i + 1}/{len(core_actions)}: {action_type}"
                )

                handler = ACTION_HANDLERS.get(action_type)
                if handler is None:
                    msg = f"Unknown action type: {action_type}"
                    print_error(msg)
                    audit.record("error", {"message": msg}, status="error")
                    continue

                try:
                    await handler(drone, params, audit, telemetry_state=telemetry_state)
                except Exception as e:
                    msg = f"Action {action_type} failed: {e}"
                    print_error(msg)
                    audit.record(action_type, {"error": str(e)},
                                 status="error")
                    # Attempt emergency land
                    print_warning("Attempting emergency landing...")
                    try:
                        await drone.action.land()
                    except Exception:
                        pass
                    raise RuntimeError(msg) from e

        # Execute landing (once)
        if land_action:
            handler = ACTION_HANDLERS[land_action["type"]]
            await handler(drone, land_action.get("params", {}), audit, telemetry_state=telemetry_state)

    except Exception as e:
        audit.record("mission_error", {"error": str(e)}, status="error")
        raise
    finally:
        if pos_task:
            pos_task.cancel()
        if home_task:
            home_task.cancel()
        audit.record("mission_end")

    print_success(f"\nMission '{mission_name}' completed successfully!")
    print_info(audit.summary())

    return audit


def execute_mission(mission: dict,
                    system_address: str = "udp://:14540") -> AuditLog:
    """
    Synchronous wrapper for execute_mission_async.
    Same JSON always produces the same behaviour.
    """
    return asyncio.run(execute_mission_async(mission, system_address))
