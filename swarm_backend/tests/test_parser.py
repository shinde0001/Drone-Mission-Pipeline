import unittest
from unittest.mock import patch
from src.llm_planner import plan_swarm_mission

class TestParser(unittest.TestCase):
    @patch('src.llm_planner._call_llm_json')
    @patch('src.llm_planner._setup_llm_client')
    def test_plan_swarm_mission_success(self, mock_setup, mock_call):
        mock_setup.return_value = (None, None, "test-model", True)
        # Mock responses for the stages:
        # 1. Classify
        # 2. Leader Actions
        # 3. Formation Params
        mock_call.side_effect = [
            {"mode": "FORMATION", "formation_type": "wedge", "num_agents": 3, "needs_task_split": False},
            [
                {"type": "takeoff", "params": {"altitude_m": 15.0}},
                {"type": "goto", "params": {"north_m": 40.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0}},
                {"type": "land", "params": {}}
            ],
            {"spacing_m": 6.0, "angle_deg": 135.0}
        ]

        mission = plan_swarm_mission(
            prompt="fly northward 40m in a 6m wedge",
            ai_engine="offline",
            api_key=None,
            num_drones=3
        )

        self.assertIsInstance(mission, dict)
        self.assertEqual(mission["mode"], "FORMATION")
        self.assertEqual(mission["formation"]["type"], "wedge")
        self.assertEqual(mission["spacing_m"], 6.0)
        self.assertEqual(len(mission["actions"]), 3)
        self.assertEqual(len(mission["agents"]), 3)
        self.assertEqual(mission["agents"][0]["id"], "leader")
        self.assertEqual(mission["agents"][0]["role"], "leader")
        self.assertEqual(mission["agents"][1]["role"], "wingman_left")
        self.assertEqual(mission["agents"][1]["slot"], 1)
        self.assertEqual(mission["agents"][2]["role"], "wingman_right")
        self.assertEqual(mission["agents"][2]["slot"], 2)
        self.assertIn("llm_stages", mission)

    @patch('src.llm_planner._call_llm_json')
    @patch('src.llm_planner._setup_llm_client')
    @patch('src.llm_planner.plan_mission')
    def test_plan_swarm_mission_fallback(self, mock_plan, mock_setup, mock_call):
        mock_setup.return_value = (None, None, "test-model", True)
        mock_call.return_value = None
        mock_plan.return_value = {
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 15.0}},
                {"type": "goto", "params": {"north_m": 10.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0}},
                {"type": "land", "params": {}}
            ]
        }

        mission = plan_swarm_mission(
            prompt="broken command",
            ai_engine="offline",
            api_key=None,
            num_drones=3
        )

        self.assertIsInstance(mission, dict)
        self.assertEqual(mission["mode"], "FORMATION")
        self.assertEqual(mission["formation"]["type"], "wedge")
        self.assertEqual(mission["spacing_m"], 5.0)  # default
        self.assertEqual(len(mission["actions"]), 3)


if __name__ == "__main__":
    unittest.main()

