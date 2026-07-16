"""
test_e2e_pipeline.py — End-to-End Integration Smoke Suite for Swarm AI Pipeline

Tests the complete lifecycle of a multi-drone swarm mission:
1. Planning Layer (Stage 1-5): Natural language prompt -> plan_swarm_mission -> Pydantic Mission model
2. Guardrail Layer (Stage 2 Validation): validate_mission schema/safety/sanity verification
3. Execution Layer: SwarmOrchestrator initialization and concurrent multi-vehicle control loop execution
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from swarm_backend.config.schema import Mission
from swarm_backend.core.orchestrator import SwarmOrchestrator
from src.llm_planner import plan_swarm_mission
from src.mission_validator import validate_mission


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


class TestE2EPipeline(unittest.IsolatedAsyncioTestCase):
    def _create_mock_vehicles(self, agent_ids):
        vehicles = {}
        for agent_id in agent_ids:
            mock_vehicle = MagicMock()
            
            # Setup telemetry mocks
            mock_vehicle.telemetry.set_rate_position_velocity_ned = AsyncMock()
            mock_vehicle.telemetry.set_rate_heading = AsyncMock()
            
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
                        yield MockPosNed(0.0, 0.0, -16.0)

            async def heading_generator():
                while True:
                    yield MockHeading(0.0)
                    await asyncio.sleep(0.05)

            async def health_generator():
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

            vehicles[agent_id] = mock_vehicle
        return vehicles

    @patch("src.llm_planner._call_llm_json")
    async def test_e2e_formation_pipeline(self, mock_llm):
        """Test full Plan -> Validate -> Execute flow in FORMATION mode."""
        def mock_stages(system_prompt, user_prompt, examples, ai_engine, model_name, client, gemini_client, is_local):
            if "Classify" in system_prompt or "Stage 1" in system_prompt or "mode" in system_prompt.lower():
                return {"mode": "FORMATION", "formation_type": "wedge", "num_agents": 3, "needs_task_split": False}
            elif "Leader" in system_prompt or "Stage 2" in system_prompt or "actions" in system_prompt.lower():
                return [{"type": "takeoff", "params": {"altitude_m": 15.0}}, {"type": "goto", "params": {"north_m": 40.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0}}, {"type": "land", "params": {}}]
            elif "Formation" in system_prompt or "Stage 3" in system_prompt or "spacing" in system_prompt.lower():
                return {"spacing_m": 5.0, "angle_deg": 135.0}
            elif "Follower" in system_prompt or "Stage 4" in system_prompt:
                return {}
            return {}
        mock_llm.side_effect = mock_stages

        prompt = "Three drones takeoff to 15m and fly north 40m in wedge formation."
        
        # 1. PLAN
        plan_result = plan_swarm_mission(prompt, ai_engine="offline", num_drones=3)
        self.assertIn("llm_stages", plan_result)
        self.assertEqual(plan_result["vehicle_type"], "swarm")
        self.assertIn("classify", plan_result["llm_stages"])
        self.assertIn("leader_actions", plan_result["llm_stages"])
        
        # 2. VALIDATE
        validation_result = validate_mission(plan_result)
        self.assertTrue(validation_result.valid, f"Validation errors: {validation_result.errors}")
        mission_obj = Mission.model_validate(plan_result)
        self.assertEqual(mission_obj.mode.value, "FORMATION")
        self.assertEqual(len(mission_obj.agents), 3)

        # 3. EXECUTE
        vehicles = self._create_mock_vehicles(["leader", "follower_1", "follower_2"])
        orchestrator = SwarmOrchestrator(mission_obj, vehicles)
        await orchestrator.initialize()
        self.assertEqual(len(orchestrator.agents), 3)

        run_task = asyncio.create_task(orchestrator.run())
        await asyncio.sleep(0.3)

        # Assert all vehicles armed and took off
        for agent_id, vehicle in vehicles.items():
            vehicle.action.arm.assert_called_once()
            vehicle.action.takeoff.assert_called_once()
            vehicle.offboard.start.assert_called()

        orchestrator._running = False
        await run_task
        for agent_id, vehicle in vehicles.items():
            vehicle.action.land.assert_called_once()

    async def test_e2e_independent_pipeline(self):
        """Test full Plan -> Validate -> Execute flow in INDEPENDENT mode."""
        mission_data = {
            "mission_id": "e2e_independent_001",
            "vehicle_type": "swarm",
            "mode": "INDEPENDENT",
            "safety": {
                "collision_radius_m": 1.0,
                "min_separation_m": 3.0,
                "altitude_separation_m": 1.0,
                "avoidance_gain": 0.6,
                "max_correction_mps": 2.0
            },
            "agents": [
                {"id": "leader", "role": "leader"},
                {
                    "id": "follower_1",
                    "role": "independent",
                    "slot": 1,
                    "independent_actions": [
                        {"type": "takeoff", "params": {"altitude_m": 12.0}},
                        {"type": "goto", "params": {"north_m": 0.0, "east_m": 25.0, "altitude_m": 12.0, "speed_mps": 4.0}},
                        {"type": "land"}
                    ]
                },
                {
                    "id": "follower_2",
                    "role": "independent",
                    "slot": 2,
                    "independent_actions": [
                        {"type": "takeoff", "params": {"altitude_m": 14.0}},
                        {"type": "goto", "params": {"north_m": -20.0, "east_m": 0.0, "altitude_m": 14.0, "speed_mps": 4.0}},
                        {"type": "land"}
                    ]
                }
            ],
            "leader_mission": {
                "actions": [
                    {"type": "takeoff", "params": {"altitude_m": 10.0}},
                    {"type": "goto", "params": {"north_m": 30.0, "east_m": 0.0, "altitude_m": 10.0, "speed_mps": 4.0}},
                    {"type": "land"}
                ]
            },
            "connections": [
                {"id": "leader", "udp": "udp://:14540"},
                {"id": "follower_1", "udp": "udp://:14541"},
                {"id": "follower_2", "udp": "udp://:14542"}
            ]
        }

        # 1. VALIDATE
        validation_result = validate_mission(mission_data)
        self.assertTrue(validation_result.valid, f"Validation errors: {validation_result.errors}")
        mission_obj = Mission.model_validate(mission_data)
        self.assertEqual(mission_obj.mode.value, "INDEPENDENT")

        # 2. EXECUTE
        vehicles = self._create_mock_vehicles(["leader", "follower_1", "follower_2"])
        orchestrator = SwarmOrchestrator(mission_obj, vehicles)
        await orchestrator.initialize()
        self.assertEqual(len(orchestrator.agents), 3)

        run_task = asyncio.create_task(orchestrator.run())
        await asyncio.sleep(0.3)

        # Assert all vehicles armed and took off
        for agent_id, vehicle in vehicles.items():
            vehicle.action.arm.assert_called_once()
            vehicle.action.takeoff.assert_called_once()
            vehicle.offboard.start.assert_called()

        orchestrator._running = False
        await run_task
        for agent_id, vehicle in vehicles.items():
            vehicle.action.land.assert_called_once()


if __name__ == "__main__":
    unittest.main()
