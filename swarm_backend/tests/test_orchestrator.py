import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
from swarm_backend.config.schema import Mission
from swarm_backend.core.orchestrator import SwarmOrchestrator

# Mock telemetry models
class MockPosition:
    def __init__(self, north_m, east_m, down_m):
        self.north_m = north_m
        self.east_m = east_m
        self.down_m = down_m
        self.relative_altitude_m = -down_m

class MockVelocity:
    def __init__(self, north_m_s, east_m_s, down_m_s):
        self.north_m_s = north_m_s
        self.east_m_s = east_m_s
        self.down_m_s = down_m_s

class MockPosNed:
    def __init__(self, n, e, d):
        self.position = MockPosition(n, e, d)
        self.velocity = MockVelocity(0.0, 0.0, 0.0)

class MockHeading:
    def __init__(self, heading_deg):
        self.heading_deg = heading_deg

class TestSwarmOrchestrator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create standard Mission configuration dictionary
        self.mission_data = {
            "mission_id": "test_swarm_001",
            "mode": "FORMATION",
            "formation": {
                "type": "wedge",
                "spacing_m": 5.0,
                "angle_deg": 135.0,
                "frame": "body_relative"
            },
            "safety": {
                "collision_radius_m": 1.0,
                "min_separation_m": 3.0,
                "altitude_separation_m": 1.0,
                "avoidance_gain": 0.6,
                "max_correction_mps": 2.0
            },
            "agents": [
                {"id": "leader", "role": "leader"},
                {"id": "drone_1", "role": "wingman_left", "slot": 1},
                {"id": "drone_2", "role": "wingman_right", "slot": 2}
            ],
            "leader_mission": {
                "actions": [
                    {"type": "takeoff", "params": {"altitude_m": 10.0}},
                    {"type": "goto", "params": {"north_m": 20.0, "east_m": 0.0, "altitude_m": 10.0, "speed_mps": 3.0}},
                    {"type": "land"}
                ]
            },
            "connections": [
                {"id": "leader", "udp": "udp://:14540"},
                {"id": "drone_1", "udp": "udp://:14541"},
                {"id": "drone_2", "udp": "udp://:14542"}
            ]
        }
        self.mission = Mission(**self.mission_data)

        # Mock vehicle instance dictionary
        self.vehicles = {}
        for agent_id in ("leader", "drone_1", "drone_2"):
            mock_vehicle = MagicMock()
            
            # Setup telemetry mocks
            mock_vehicle.telemetry.set_rate_position_velocity_ned = AsyncMock()
            mock_vehicle.telemetry.set_rate_heading = AsyncMock()
            
            # Use async generators that output starting locations
            async def pos_generator(mv=mock_vehicle):
                yield MockPosNed(0.0, 0.0, 0.0)
                while True:
                    await asyncio.sleep(0.05)
                    if mv.action.land.called or mv.action.return_to_launch.called:
                        yield MockPosNed(0.0, 0.0, 0.0)
                    elif mv.offboard.set_position_velocity_ned.called:
                        args = mv.offboard.set_position_velocity_ned.call_args[0][0]
                        yield MockPosNed(args.north_m, args.east_m, args.down_m)
                    else:
                        yield MockPosNed(0.0, 0.0, -10.0)  # Hovering at 10m

            async def heading_generator():
                while True:
                    yield MockHeading(0.0)
                    await asyncio.sleep(0.05)

            async def health_generator():
                # Yield home and global position OK
                mock_health = MagicMock()
                mock_health.is_global_position_ok = True
                mock_health.is_home_position_ok = True
                yield mock_health

            async def conn_generator():
                mock_conn = MagicMock()
                mock_conn.is_connected = True
                yield mock_conn

            mock_vehicle.telemetry.position_velocity_ned.return_value = pos_generator()
            mock_vehicle.telemetry.heading.return_value = heading_generator()
            mock_vehicle.telemetry.health.return_value = health_generator()
            mock_vehicle.core.connection_state.return_value = conn_generator()

            # Mock action methods
            mock_vehicle.action.arm = AsyncMock()
            mock_vehicle.action.disarm = AsyncMock()
            mock_vehicle.action.set_takeoff_altitude = AsyncMock()
            mock_vehicle.action.takeoff = AsyncMock()
            mock_vehicle.action.land = AsyncMock()
            mock_vehicle.action.return_to_launch = AsyncMock()

            # Mock offboard
            mock_vehicle.offboard.set_position_velocity_ned = AsyncMock()
            mock_vehicle.offboard.start = AsyncMock()
            mock_vehicle.offboard.stop = AsyncMock()

            self.vehicles[agent_id] = mock_vehicle

    async def test_full_orchestrator_flow(self):
        orchestrator = SwarmOrchestrator(self.mission, self.vehicles)
        await orchestrator.initialize()

        # Check agents were connected
        self.assertEqual(len(orchestrator.agents), 3)

        # Run the orchestrator mission in background task so we can cancel or monitor
        run_task = asyncio.create_task(orchestrator.run())

        # Let the mission loop run past initialization, arming, takeoff, and first waypoint
        await asyncio.sleep(0.3)

        # Verify takeoff and arm calls on all vehicles
        for agent_id, vehicle in self.vehicles.items():
            vehicle.action.arm.assert_called_once()
            vehicle.action.takeoff.assert_called_once()
            vehicle.offboard.start.assert_called()

        # Let the mission finish or stop
        orchestrator._running = False
        await run_task

        # Verify landing was commanded
        for agent_id, vehicle in self.vehicles.items():
            vehicle.action.land.assert_called_once()

    async def test_emergency_landing(self):
        orchestrator = SwarmOrchestrator(self.mission, self.vehicles)
        await orchestrator.initialize()

        # Trigger emergency land immediately
        await orchestrator.emergency_land()

        for agent_id, vehicle in self.vehicles.items():
            vehicle.action.land.assert_called_once()
            vehicle.offboard.stop.assert_called_once()
