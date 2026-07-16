import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
from swarm_backend.core.drone_agent import DroneAgent
from swarm_backend.core.swarm_state import SwarmState

# Mock telemetry structures corresponding to MAVSDK telemetry models
class MockPosition:
    def __init__(self, north_m, east_m, down_m):
        self.north_m = north_m
        self.east_m = east_m
        self.down_m = down_m

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

class TestDroneAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state_bus = SwarmState()
        self.mock_vehicle = MagicMock()
        
        # Setup mocks for telemetry streams
        self.mock_vehicle.telemetry.set_rate_position_velocity_ned = AsyncMock()
        self.mock_vehicle.telemetry.set_rate_heading = AsyncMock()
        
        # Setup async generator mocks for streaming telemetry
        async def mock_pos_stream():
            yield MockPosNed(10.0, 20.0, -15.0)
            await asyncio.sleep(0.1)
            yield MockPosNed(12.0, 22.0, -15.0)
            # block to keep stream active during test
            await asyncio.sleep(1.0)

        async def mock_heading_stream():
            yield MockHeading(45.0)
            await asyncio.sleep(0.1)
            yield MockHeading(50.0)
            await asyncio.sleep(1.0)

        self.mock_vehicle.telemetry.position_velocity_ned.return_value = mock_pos_stream()
        self.mock_vehicle.telemetry.heading.return_value = mock_heading_stream()
        
        # Mock actions
        self.mock_vehicle.action.arm = AsyncMock()
        self.mock_vehicle.action.disarm = AsyncMock()
        self.mock_vehicle.action.set_takeoff_altitude = AsyncMock()
        self.mock_vehicle.action.takeoff = AsyncMock()
        self.mock_vehicle.action.land = AsyncMock()
        self.mock_vehicle.action.return_to_launch = AsyncMock()
        
        # Mock offboard
        self.mock_vehicle.offboard.set_position_velocity_ned = AsyncMock()
        self.mock_vehicle.offboard.start = AsyncMock()
        self.mock_vehicle.offboard.stop = AsyncMock()

    async def test_drone_agent_telemetry_flow(self):
        agent = DroneAgent("drone_1", "udp://:14541", self.state_bus, vehicle=self.mock_vehicle)
        agent._running = True
        agent.start_telemetry_stream()

        # Let the async generator run and populate the state bus
        await asyncio.sleep(0.15)

        # Check that values have successfully propagated to the state bus
        self.assertEqual(self.state_bus.get_position("drone_1"), (12.0, 22.0, -15.0))
        self.assertEqual(self.state_bus.get_heading("drone_1"), 50.0)

        await agent.stop()

    async def test_drone_agent_commands(self):
        agent = DroneAgent("leader", "udp://:14540", self.state_bus, vehicle=self.mock_vehicle)
        
        # Test command delegations
        await agent.arm()
        self.mock_vehicle.action.arm.assert_called_once()

        await agent.takeoff(15.0)
        self.mock_vehicle.action.set_takeoff_altitude.assert_called_with(15.0)
        self.mock_vehicle.action.takeoff.assert_called_once()

        await agent.start_offboard()
        self.mock_vehicle.offboard.start.assert_called_once()

        await agent.stop()
