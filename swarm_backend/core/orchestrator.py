import asyncio
import logging
import math
from typing import Dict, List, Tuple, Any, Optional
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw

from swarm_backend.config.schema import Mission, Action
from swarm_backend.core.swarm_state import SwarmState
from swarm_backend.core.drone_agent import DroneAgent
from swarm_backend.core.formation_controller import FormationController
from swarm_backend.core.collision_avoidance import CollisionAvoidance
from swarm_backend.core.safety_envelope import SafetyEnvelope

logger = logging.getLogger("orchestrator")

WAYPOINT_TOLERANCE_M = 2.0
LOOP_INTERVAL_S = 0.05  # 20 Hz control loop

class SwarmOrchestrator:
    """
    Main swarm orchestration class that coordinates startup, high-frequency control loops,
    formation geometry, dynamic collision avoidance, safety envelope limits, and mission state transitions.
    """
    def __init__(self, mission: Mission, vehicle_instances: Optional[Dict[str, Any]] = None):
        self.mission = mission
        self.state_bus = SwarmState()
        self.agents: Dict[str, DroneAgent] = {}
        self.vehicle_instances = vehicle_instances or {}
        
        self._running = False
        self._control_task = None
        self._mode = mission.mode

        # Initialize core swarm modules
        self.formation_controller = FormationController(
            formation_type=mission.formation.type,
            spacing_m=mission.formation.spacing_m,
            angle_deg=mission.formation.angle_deg,
            frame=mission.formation.frame
        )

        self.collision_avoidance = CollisionAvoidance(
            min_separation_m=mission.safety.min_separation_m,
            avoidance_gain=mission.safety.avoidance_gain
        )

        self.safety_envelope = SafetyEnvelope(
            max_altitude_m=30.0,
            min_altitude_m=1.0,
            geofence_radius_m=100.0,
            max_speed_mps=mission.safety.max_correction_mps + 5.0
        )

    async def initialize(self, timeout_s: float = 30.0) -> None:
        """Create and connect all DroneAgent instances in parallel."""
        conn_tasks = []
        for agent_conf in self.mission.agents:
            agent_id = agent_conf.id
            # Match connection address
            conn_conf = next((c for c in self.mission.connections if c.id == agent_id), None)
            conn_address = conn_conf.udp if conn_conf else f"udp://:14540"

            vehicle = self.vehicle_instances.get(agent_id)
            idx = next((i for i, a in enumerate(self.mission.agents) if a.id == agent_id), 0)
            grpc_port = 50051 + idx

            agent = DroneAgent(
                agent_id=agent_id,
                connection_address=conn_address,
                state_bus=self.state_bus,
                grpc_port=grpc_port,
                vehicle=vehicle
            )
            self.agents[agent_id] = agent
            conn_tasks.append(agent.connect(timeout_s=timeout_s))

        await asyncio.gather(*conn_tasks)
        logger.info("All swarm agents initialized and connected successfully.")

    async def run(self) -> None:
        """Execute the entire swarm mission lifecycle."""
        self._running = True
        try:
            # 1. Arm all vehicles in parallel
            logger.info("Arming all swarm vehicles...")
            await asyncio.gather(*(agent.arm() for agent in self.agents.values()))

            # 2. Takeoff sequence (all vehicles in parallel)
            takeoff_action = next((a for a in self.mission.leader_mission.actions if a.type == "takeoff"), None)
            takeoff_alt = takeoff_action.params.get("altitude_m", 10.0) if takeoff_action else 10.0
            
            logger.info(f"Commanding all vehicles to takeoff to {takeoff_alt}m...")
            await asyncio.gather(*(agent.takeoff(takeoff_alt) for agent in self.agents.values()))

            # Wait for all vehicles to reach at least 90% of takeoff altitude
            while True:
                reached = True
                for agent_id, agent in self.agents.items():
                    pos = self.state_bus.get_position(agent_id)
                    alt = -pos[2]  # Z is down, so altitude is -Z
                    if alt < takeoff_alt * 0.9:
                        reached = False
                        break
                if reached:
                    break
                await asyncio.sleep(0.1)
            logger.info("All vehicles successfully reached takeoff altitude.")

            # 3. Enter Offboard mode for all vehicles
            logger.info("Initializing offboard modes...")
            await asyncio.gather(*(agent.start_offboard() for agent in self.agents.values()))

            # 4. Start high-frequency follower control loop
            self._control_task = asyncio.create_task(self._follower_control_loop())

            # 5. Execute Leader Waypoint Actions
            leader_id = next(a.id for a in self.mission.agents if a.role == "leader")
            leader_agent = self.agents[leader_id]

            for action in self.mission.leader_mission.actions:
                if not self._running:
                    break
                if action.type in ("takeoff", "land", "return_to_launch"):
                    continue

                logger.info(f"Leader starting action: {action.type} with params {action.params}")
                if action.type == "goto":
                    await self._execute_goto(leader_agent, action)
                elif action.type == "loiter":
                    await asyncio.sleep(action.params.get("duration_s", 5.0))

            # 6. Land / Return To Launch Sequence
            self._running = False
            if self._control_task:
                self._control_task.cancel()
                try:
                    await self._control_task
                except asyncio.CancelledError:
                    pass

            await asyncio.gather(*(agent.stop_offboard() for agent in self.agents.values()))

            last_action = self.mission.leader_mission.actions[-1]
            if last_action.type == "return_to_launch":
                logger.info("Returning all agents to launch...")
                await asyncio.gather(*(agent.return_to_launch() for agent in self.agents.values()))
            else:
                logger.info("Landing all agents...")
                await asyncio.gather(*(agent.land() for agent in self.agents.values()))

            # Wait for disarm
            while True:
                disarmed = True
                for agent_id, agent in self.agents.items():
                    pos = self.state_bus.get_position(agent_id)
                    # Simple heuristic: altitude close to 0
                    if -pos[2] > 0.5:
                        disarmed = False
                        break
                if disarmed:
                    break
                await asyncio.sleep(0.1)

            logger.info("Swarm mission execution completed successfully.")

        except Exception as e:
            logger.error(f"Error during orchestrator execution: {e}")
            await self.emergency_land()
            raise e

    async def _execute_goto(self, leader_agent: DroneAgent, action: Action) -> None:
        """Guide the leader agent to a waypoint, updating heading dynamically."""
        target_n = action.params["north_m"]
        target_e = action.params["east_m"]
        target_d = -action.params["altitude_m"]
        speed = action.params["speed_mps"]

        while self._running:
            leader_pos = self.state_bus.get_position(leader_agent.agent_id)
            dn = target_n - leader_pos[0]
            de = target_e - leader_pos[1]
            dd = target_d - leader_pos[2]
            dist = math.sqrt(dn**2 + de**2)

            if dist <= WAYPOINT_TOLERANCE_M:
                break

            # Calculate movement vector scaled by desired speed
            angle = math.atan2(de, dn)
            heading_deg = math.degrees(angle) % 360

            vn = speed * math.cos(angle)
            ve = speed * math.sin(angle)
            vd = speed * (dd / dist) if dist > 0 else 0.0

            # Enforce safety envelope limits on the leader
            clamped_pos, clamped_vel = self.safety_envelope.enforce(
                (target_n, target_e, target_d), (vn, ve, vd)
            )

            await leader_agent.set_offboard_position_velocity(
                PositionNedYaw(clamped_pos[0], clamped_pos[1], clamped_pos[2], heading_deg),
                VelocityNedYaw(clamped_vel[0], clamped_vel[1], clamped_vel[2], heading_deg)
            )
            await asyncio.sleep(LOOP_INTERVAL_S)

    async def _follower_control_loop(self) -> None:
        """High-frequency control loop for followers (wingmen)."""
        leader_id = next(a.id for a in self.mission.agents if a.role == "leader")
        followers = [a for a in self.mission.agents if a.role in ("wingman_left", "wingman_right")]

        try:
            while self._running:
                leader_pos = self.state_bus.get_position(leader_id)
                leader_vel = self.state_bus.get_velocity(leader_id)
                leader_heading = self.state_bus.get_heading(leader_id)

                for follower_conf in followers:
                    fid = follower_conf.id
                    f_agent = self.agents[fid]
                    f_pos = self.state_bus.get_position(fid)

                    # Dynamic Regroup state machine logic
                    dist_to_leader = math.sqrt((f_pos[0] - leader_pos[0])**2 + (f_pos[1] - leader_pos[1])**2)
                    
                    if self._mode == "REGROUP":
                        # If close enough, switch back to formation
                        if dist_to_leader <= self.formation_controller.spacing_m * 1.5:
                            logger.info(f"Follower {fid} close enough. Switching from REGROUP to FORMATION.")
                            self._mode = "FORMATION"

                    # 1. Geometry Projection (Body-relative offsets)
                    if self._mode == "INDEPENDENT":
                        # Mirror waypoint sequence but apply a static offset in world frame
                        offset_role_multiplier = -1.0 if follower_conf.role == "wingman_left" else 1.0
                        target_n = leader_pos[0]
                        target_e = leader_pos[1] + (offset_role_multiplier * self.formation_controller.spacing_m)
                        target_d = leader_pos[2]
                    else:  # FORMATION or REGROUP
                        target_n, target_e, target_d = self.formation_controller.calculate_target_position(
                            leader_pos, leader_heading, follower_conf.role, follower_conf.slot or 1
                        )

                    # 2. Feedforward + Position tracking velocity calculation
                    vn, ve, vd = self.formation_controller.calculate_feedforward_velocity(
                        leader_vel, (target_n, target_e, target_d), f_pos, gain_kp=1.0,
                        max_correction_mps=self.mission.safety.max_correction_mps
                    )

                    # 3. Collision Avoidance (Artificial Potential Field + Altitude Stacking)
                    other_positions = self.state_bus.all_positions_except(fid)
                    
                    # Proximity check for altitude stacking override
                    for other_pos in other_positions:
                        h_dist = math.sqrt((f_pos[0] - other_pos[0])**2 + (f_pos[1] - other_pos[1])**2)
                        if h_dist < self.mission.safety.collision_radius_m:
                            # Apply altitude stacking
                            stack_offset = self.mission.safety.altitude_separation_m
                            if follower_conf.role == "wingman_left":
                                target_d -= stack_offset  # Fly higher
                            else:
                                target_d += stack_offset  # Fly lower

                    # Superimpose potential field forces onto velocities
                    vn, ve, vd = self.collision_avoidance.apply_avoidance_to_velocity(
                        (vn, ve, vd), f_pos, other_positions
                    )

                    # 4. Safety Envelope limits clamping
                    clamped_pos, clamped_vel = self.safety_envelope.enforce(
                        (target_n, target_e, target_d), (vn, ve, vd)
                    )

                    await f_agent.set_offboard_position_velocity(
                        PositionNedYaw(clamped_pos[0], clamped_pos[1], clamped_pos[2], leader_heading),
                        VelocityNedYaw(clamped_vel[0], clamped_vel[1], clamped_vel[2], leader_heading)
                    )

                await asyncio.sleep(LOOP_INTERVAL_S)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Follower control loop error: {e}")

    async def emergency_land(self) -> None:
        """Trigger immediate emergency landing on all connected vehicles."""
        logger.warning("EMERGENCY LANDING INITIATED FOR SWARM!")
        self._running = False
        if self._control_task:
            self._control_task.cancel()
        await asyncio.gather(*(agent.stop_offboard() for agent in self.agents.values()), return_exceptions=True)
        await asyncio.gather(*(agent.land() for agent in self.agents.values()), return_exceptions=True)
        logger.warning("Emergency land commands sent to all agents.")
