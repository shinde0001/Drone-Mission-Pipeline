import asyncio
import logging
import re
from typing import Optional, Tuple, Any
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, OffboardError
from swarm_backend.core.swarm_state import SwarmState

logger = logging.getLogger("drone_agent")

class DroneAgent:
    """
    Manages a single vehicle's MAVSDK connection, background telemetry streaming,
    and command execution interfaces.
    """
    def __init__(
        self,
        agent_id: str,
        connection_address: str,
        state_bus: SwarmState,
        grpc_port: Optional[int] = None,
        vehicle: Optional[Any] = None
    ):
        self.agent_id = agent_id
        self.connection_address = connection_address
        self.state_bus = state_bus
        self.grpc_port = grpc_port
        self.vehicle = vehicle
        self._telemetry_task = None
        self._running = False

    async def connect(self, timeout_s: float = 30.0) -> None:
        """Connect to the PX4 flight controller and wait for basic health checks."""
        if self.vehicle is None:
            # Enforce unique gRPC port per local MAVSDK instance to prevent port collisions
            if self.grpc_port is None:
                # Deduce dynamic port based on numbers found in agent_id
                num_match = re.search(r'\d+', self.agent_id)
                idx = int(num_match.group()) if num_match else 0
                self.grpc_port = 50051 + idx
            
            logger.info(f"Connecting to {self.agent_id} via {self.connection_address} on gRPC port {self.grpc_port}")
            self.vehicle = System(port=self.grpc_port)
            await self.vehicle.connect(system_address=self.connection_address)

        # Wait for connection and basic readiness
        async def wait_ready():
            async for state in self.vehicle.core.connection_state():
                if state.is_connected:
                    break
            async for health in self.vehicle.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok and health.is_armable:
                    break

        await asyncio.wait_for(wait_ready(), timeout=timeout_s)
        self._running = True
        self.start_telemetry_stream()
        logger.info(f"Agent {self.agent_id} fully connected and streaming telemetry.")

    def start_telemetry_stream(self) -> None:
        """Start background task for streaming telemetry to SwarmState."""
        if self._telemetry_task is None:
            self._telemetry_task = asyncio.create_task(self._telemetry_loop())

    async def _telemetry_loop(self) -> None:
        """High-frequency stream updating global SwarmState."""
        try:
            # Request high telemetry rates to minimize loop latency
            await self.vehicle.telemetry.set_rate_position_velocity_ned(20.0)
            await self.vehicle.telemetry.set_rate_heading(20.0)
        except Exception as e:
            logger.debug(f"Telemetry rate set failed for {self.agent_id}: {e}")

        async def stream_pos():
            async for pos_ned in self.vehicle.telemetry.position_velocity_ned():
                if not self._running:
                    break
                n = pos_ned.position.north_m
                e = pos_ned.position.east_m
                d = pos_ned.position.down_m
                vn = pos_ned.velocity.north_m_s
                ve = pos_ned.velocity.east_m_s
                vd = pos_ned.velocity.down_m_s
                self.state_bus.update_position(self.agent_id, (n, e, d))
                self.state_bus.update_velocity(self.agent_id, (vn, ve, vd))

        async def stream_heading():
            async for hdg in self.vehicle.telemetry.heading():
                if not self._running:
                    break
                self.state_bus.update_heading(self.agent_id, hdg.heading_deg)

        try:
            await asyncio.gather(stream_pos(), stream_heading())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Telemetry streaming loop encountered error for {self.agent_id}: {e}")

    async def arm(self) -> None:
        """Arm the vehicle."""
        logger.info(f"Arming {self.agent_id}")
        await self.vehicle.action.arm()

    async def disarm(self) -> None:
        """Disarm the vehicle."""
        logger.info(f"Disarming {self.agent_id}")
        await self.vehicle.action.disarm()

    async def takeoff(self, altitude_m: float) -> None:
        """Set takeoff altitude and execute takeoff command."""
        logger.info(f"Taking off {self.agent_id} to altitude {altitude_m}m")
        await self.vehicle.action.set_takeoff_altitude(altitude_m)
        await self.vehicle.action.takeoff()

    async def land(self) -> None:
        """Execute landing command."""
        logger.info(f"Landing {self.agent_id}")
        await self.vehicle.action.land()

    async def return_to_launch(self) -> None:
        """Return to launch position."""
        logger.info(f"Executing RTL for {self.agent_id}")
        await self.vehicle.action.return_to_launch()

    async def start_offboard(self) -> None:
        """Initialize offboard control with a dummy setpoint and start offboard mode."""
        try:
            # Send initial setpoint before starting to prevent failsafe
            await self.vehicle.offboard.set_position_velocity_ned(
                PositionNedYaw(0.0, 0.0, -10.0, 0.0),
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            )
            await self.vehicle.offboard.start()
            logger.info(f"Offboard mode active for {self.agent_id}")
        except OffboardError as e:
            if "already active" not in str(e).lower():
                raise e

    async def stop_offboard(self) -> None:
        """Stop offboard control mode."""
        try:
            await self.vehicle.offboard.stop()
            logger.info(f"Offboard mode stopped for {self.agent_id}")
        except OffboardError as e:
            logger.debug(f"Offboard stop error for {self.agent_id}: {e}")

    async def set_offboard_position_velocity(self, pos: PositionNedYaw, vel: VelocityNedYaw) -> None:
        """Send target position and velocity setpoints to the flight controller."""
        await self.vehicle.offboard.set_position_velocity_ned(pos, vel)

    async def stop(self) -> None:
        """Stop telemetry tasks and cleanup connection."""
        self._running = False
        if self._telemetry_task:
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass
            self._telemetry_task = None
        logger.info(f"DroneAgent {self.agent_id} stopped.")
