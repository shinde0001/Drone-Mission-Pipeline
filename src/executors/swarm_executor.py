"""
swarm_executor.py — Real MAVSDK Multi-Agent Swarm Executor with Real-Time Formation Control

Features:
- Connects to N PX4 instances (ports 14540, 14541, 14542, ...)
- FormationController runs a continuous 10Hz control loop for followers (wingmen)
- Rotates formation offsets dynamically based on leader heading
- Active 1m collision avoidance via altitude stacking (+1m for Follower 1, -1m for Follower 2)
- Supports FORMATION, INDEPENDENT, and REGROUP operational modes
"""

import asyncio
import math
import time
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, OffboardError
from ..utils import AuditLog, print_info, print_success, print_error, print_warning, setup_logger

logger = setup_logger("swarm_executor")

WAYPOINT_TOLERANCE_M = 2.0
POSITION_CHECK_INTERVAL_S = 0.5

FORMATION_OFFSETS = {
    "wedge": [
        (-5.0, -5.0, 0.0),    # Wingman 1 (left & back)
        (-5.0, 5.0, 0.0),     # Wingman 2 (right & back)
    ],
    "line": [
        (0.0, -5.0, 0.0),     # Wingman 1 (left)
        (0.0, 5.0, 0.0),      # Wingman 2 (right)
    ],
    "column": [
        (-5.0, 0.0, 0.0),     # Wingman 1 (behind)
        (-10.0, 0.0, 0.0),    # Wingman 2 (far behind)
    ]
}


class FormationController:
    """
    Maintains formation by tracking the leader's real-time position, velocity, and heading,
    and commanding wingmen at 20Hz with velocity feedforward and collision avoidance.
    """
    def __init__(self, leader_drone: System, wingmen: list, offsets: list, collision_radius_m: float = 1.0):
        self.leader = leader_drone
        self.wingmen = wingmen
        self.offsets = offsets
        self.collision_radius = collision_radius_m
        self.running = False
        self.leader_pos = (0.0, 0.0, -10.0)  # (n, e, d)
        self.leader_vel = (0.0, 0.0, 0.0)    # (vn, ve, vd)
        self.leader_last_time = time.time()
        self.leader_heading = 0.0
        self.all_positions = {}   # drone_idx -> (n, e, d)
        self.all_velocities = {}  # drone_idx -> (vn, ve, vd)
        self._tasks = []

    async def start(self):
        """Start high-frequency telemetry monitoring and 20Hz follower control loops."""
        if self.running:
            return
        self.running = True
        logger.info("Starting high-frequency (20Hz) FormationController loops...")

        # Request high-frequency (30Hz) telemetry streams on all drones to minimize sensing lag
        for d in [self.leader] + self.wingmen:
            try:
                await d.telemetry.set_rate_position_velocity_ned(30.0)
                await d.telemetry.set_rate_heading(30.0)
            except Exception as e:
                logger.debug(f"Could not set high telemetry rates: {e}")

        # Monitor leader telemetry (position, velocity & heading)
        self._tasks.append(asyncio.create_task(self._monitor_drone_pos(0, self.leader)))
        self._tasks.append(asyncio.create_task(self._monitor_leader_heading()))

        # Monitor wingmen positions/velocities and start offboard control
        for idx, wingman in enumerate(self.wingmen):
            drone_id = idx + 1
            self._tasks.append(asyncio.create_task(self._monitor_drone_pos(drone_id, wingman)))
            await self._init_offboard(wingman)
            self._tasks.append(asyncio.create_task(self._follower_loop(idx, wingman)))

        print_success("FormationController active (20Hz low-latency tracking + velocity feedforward).")

    async def stop(self):
        """Stop all control and monitoring loops."""
        self.running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        for wingman in self.wingmen:
            try:
                await wingman.offboard.stop()
            except Exception:
                pass
        logger.info("FormationController stopped.")

    async def _monitor_drone_pos(self, drone_idx: int, drone: System):
        """Continuously update the NED position and velocity of a drone."""
        try:
            async for pos_ned in drone.telemetry.position_velocity_ned():
                if not self.running:
                    break
                n = pos_ned.position.north_m
                e = pos_ned.position.east_m
                d = pos_ned.position.down_m
                vn = pos_ned.velocity.north_m_s
                ve = pos_ned.velocity.east_m_s
                vd = pos_ned.velocity.down_m_s
                self.all_positions[drone_idx] = (n, e, d)
                self.all_velocities[drone_idx] = (vn, ve, vd)
                if drone_idx == 0:
                    self.leader_pos = (n, e, d)
                    self.leader_vel = (vn, ve, vd)
                    self.leader_last_time = time.time()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Position monitor loop error for drone {drone_idx}: {e}")

    async def _monitor_leader_heading(self):
        """Continuously update the leader's yaw heading."""
        try:
            async for hdg in self.leader.telemetry.heading():
                if not self.running:
                    break
                self.leader_heading = hdg.heading_deg
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Leader heading monitor error: {e}")

    async def _init_offboard(self, drone: System):
        """Initialize offboard mode by sending a dummy setpoint first."""
        try:
            # Send initial position+velocity setpoint so offboard starts smoothly without jump
            await drone.offboard.set_position_velocity_ned(
                PositionNedYaw(0.0, 0.0, -10.0, 0.0),
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            )
            await drone.offboard.start()
        except OffboardError as oe:
            if "already active" not in str(oe).lower():
                logger.warning(f"Offboard init error: {oe}")

    async def _follower_loop(self, wingman_idx: int, wingman: System):
        """20Hz low-latency control loop for a single wingman with velocity feedforward."""
        drone_id = wingman_idx + 1
        try:
            while self.running:
                if 0 not in self.all_positions:
                    await asyncio.sleep(0.05)
                    continue

                # 1. Dead-reckoning: predict leader position at exact current instant
                dt = min(0.2, max(0.0, time.time() - self.leader_last_time))
                pred_leader_n = self.leader_pos[0] + self.leader_vel[0] * dt
                pred_leader_e = self.leader_pos[1] + self.leader_vel[1] * dt
                pred_leader_d = self.leader_pos[2] + self.leader_vel[2] * dt

                dx, dy, dz = self.offsets[wingman_idx] if wingman_idx < len(self.offsets) else (-5.0 * (wingman_idx + 1), 0.0, 0.0)
                cos_h = math.cos(math.radians(self.leader_heading))
                sin_h = math.sin(math.radians(self.leader_heading))

                # Rotate offset by leader heading
                rotated_n = dx * cos_h - dy * sin_h
                rotated_e = dx * sin_h + dy * cos_h

                target_n = pred_leader_n + rotated_n
                target_e = pred_leader_e + rotated_e
                target_d = pred_leader_d + dz

                # Collision avoidance (Artificial Potential Field / Altitude Stacking)
                for other_id, other_pos in self.all_positions.items():
                    if other_id == drone_id:
                        continue
                    horiz_dist = math.sqrt((target_n - other_pos[0])**2 + (target_e - other_pos[1])**2)
                    if horiz_dist < self.collision_radius:
                        # Close proximity detected! Apply altitude stacking:
                        if wingman_idx == 0:
                            target_d -= 1.0
                        else:
                            target_d += 1.0

                # 2. Velocity Feedforward + Lead Correction: eliminate position tracking lag
                curr_pos = self.all_positions.get(drone_id, (target_n, target_e, target_d))
                pos_err_n = target_n - curr_pos[0]
                pos_err_e = target_e - curr_pos[1]
                pos_err_d = target_d - curr_pos[2]

                target_vn = self.leader_vel[0] + 1.0 * pos_err_n
                target_ve = self.leader_vel[1] + 1.0 * pos_err_e
                target_vd = self.leader_vel[2] + 1.0 * pos_err_d

                await wingman.offboard.set_position_velocity_ned(
                    PositionNedYaw(target_n, target_e, target_d, self.leader_heading),
                    VelocityNedYaw(target_vn, target_ve, target_vd, self.leader_heading)
                )
                await asyncio.sleep(0.05)  # 20Hz control loop
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Follower loop error for wingman {wingman_idx}: {e}")


async def connect_drone(port: int, timeout_s: float = 60.0) -> System:
    """Connect to a PX4 instance on the given UDP port with a timeout."""
    drone_idx = port - 14540
    drone = System(port=50051 + drone_idx)
    addr = f"udp://:{port}"
    logger.info(f"Connecting to drone at {addr}...")
    await drone.connect(system_address=addr)
    
    async def _wait_for_ready():
        async for state in drone.core.connection_state():
            if state.is_connected:
                logger.info(f"Drone on port {port} connected!")
                break
                
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
    
    formation_raw = mission.get("formation", "wedge")
    formation_name = (formation_raw.get("type", "wedge") if isinstance(formation_raw, dict) else str(formation_raw)).lower()
    num_drones = int(mission.get("drone_count", mission.get("num_drones", 3)))
    mode = mission.get("mode", "FORMATION").upper()
    spacing_m = float(mission.get("spacing_m", 5.0))
    collision_radius_m = float(mission.get("collision_radius_m", 1.0))
    
    print_info(f"Swarm config: {num_drones} drones, mode={mode}, formation={formation_name}, spacing={spacing_m}m")
    
    # Generate wingmen offsets based on formation and spacing
    if formation_name == "line":
        wingmen_offsets = [(0.0, -spacing_m, 0.0), (0.0, spacing_m, 0.0)]
    elif formation_name == "column":
        wingmen_offsets = [(-spacing_m, 0.0, 0.0), (-2.0 * spacing_m, 0.0, 0.0)]
    else:  # wedge (default)
        wingmen_offsets = [(-spacing_m, -spacing_m, 0.0), (-spacing_m, spacing_m, 0.0)]
            
    # Connect to all drones
    try:
        if num_drones > 1:
            print_info(f"Connecting to {num_drones - 1} wingmen on ports {[14540 + i for i in range(1, num_drones)]}...")
            wingmen = await asyncio.gather(*(
                connect_drone(14540 + i) for i in range(1, num_drones)
            ))
            drones = [drone] + list(wingmen)
        else:
            drones = [drone]
            wingmen = []
        print_success(f"All {num_drones} drones connected and ready!")
    except Exception as e:
        print_error(f"Failed to connect to wingmen drones: {e}")
        audit.record("swarm_error", {"message": f"Connection error: {e}"})
        raise e
        
    try:
        from swarm_backend.config.schema import Mission as SwarmMission
        try:
            mission_obj = SwarmMission.model_validate(mission)
        except Exception as e:
            print_info(f"Adapting mission dict to Pydantic Mission model ({e})...")
            from swarm_backend.config.schema import (
                MissionMode, FormationConfig, SafetyConfig, AgentConfig, LeaderMission, Action as SchemaAction
            )
            mode_str = str(mission.get("mode", "FORMATION")).upper()
            if mode_str not in ("FORMATION", "INDEPENDENT", "MIXED", "REGROUP"):
                mode_str = "FORMATION"
            
            agents_list = [AgentConfig(id="leader", role="leader")]
            f_config = mission.get("follower_config", {})
            for i in range(1, num_drones):
                fid = f"follower_{i}"
                fval = f_config.get(fid, {}) if isinstance(f_config, dict) else {}
                role_str = fval.get("role", "wingman_left" if i % 2 != 0 else "wingman_right")
                ind_acts = fval.get("independent_actions")
                ind_act_objs = [SchemaAction(**a) if isinstance(a, dict) else a for a in ind_acts] if ind_acts else None
                agents_list.append(AgentConfig(id=fid, role=role_str, slot=i, independent_actions=ind_act_objs))

            actions_raw = mission.get("actions", [])
            if not actions_raw and "leader_mission" in mission and isinstance(mission["leader_mission"], dict):
                actions_raw = mission["leader_mission"].get("actions", [])
            leader_acts = [SchemaAction(**a) if isinstance(a, dict) else a for a in actions_raw]

            formation_cfg = FormationConfig(
                type=formation_name,
                spacing_m=spacing_m,
                angle_deg=135.0 if formation_name == "wedge" else 90.0,
                frame="body_relative"
            ) if mode_str in ("FORMATION", "MIXED", "REGROUP") else None

            mission_obj = SwarmMission(
                mission_id=str(mission.get("mission_name", f"swarm_{num_drones}")),
                mode=MissionMode(mode_str),
                formation=formation_cfg,
                safety=SafetyConfig(collision_radius_m=collision_radius_m, min_separation_m=max(3.0, spacing_m)),
                agents=agents_list,
                leader_mission=LeaderMission(actions=leader_acts)
            )

        # Map connected drones to vehicle_instances for SwarmOrchestrator
        vehicle_instances = {"leader": drones[0]}
        for i, wing in enumerate(drones[1:], start=1):
            vehicle_instances[f"follower_{i}"] = wing

        from swarm_backend.core.orchestrator import SwarmOrchestrator
        orchestrator = SwarmOrchestrator(mission_obj, vehicle_instances=vehicle_instances)
        print_info("Initializing SwarmOrchestrator with connected vehicles...")
        await orchestrator.initialize()
        print_info("Running unified SwarmOrchestrator control loop...")
        await orchestrator.run()

    except Exception as e:
        print_error(f"Swarm mission execution error: {e}")
        audit.record("swarm_error", {"details": str(e)})
        print_warning("Initiating emergency landing for swarm...")
        await asyncio.gather(*(d.action.land() for d in drones if d), return_exceptions=True)
        raise e
        
    audit.record("swarm_completed")
    print_success("Swarm Mission Completed Successfully!")

