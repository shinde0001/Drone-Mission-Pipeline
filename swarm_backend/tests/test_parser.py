import unittest
from unittest.mock import patch
from swarm_backend.llm.mission_parser import parse_to_mission
from swarm_backend.config.schema import Mission

class TestParser(unittest.TestCase):
    @patch('swarm_backend.llm.mission_parser._call_llm')
    def test_parse_to_mission_success(self, mock_call):
        # Mock responses for the 3 stages:
        # 1. Mode/intent classification
        # 2. Formation params
        # 3. Leader mission actions
        mock_call.side_effect = [
            {"cleaned_prompt": "Three drones fly north 40m.", "mode": "FORMATION"},
            {"type": "wedge", "spacing_m": 6.0, "angle_deg": 135.0, "frame": "body_relative"},
            {
                "actions": [
                    {"type": "takeoff", "params": {"altitude_m": 15.0}},
                    {"type": "goto", "params": {"north_m": 40.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0}},
                    {"type": "land", "params": {}}
                ]
            }
        ]

        mission = parse_to_mission(
            prompt="fly northward 40m in a 6m wedge",
            ai_engine="offline",
            api_key=None,
            num_drones=3
        )

        self.assertIsInstance(mission, Mission)
        self.assertEqual(mission.mode, "FORMATION")
        self.assertEqual(mission.formation.type, "wedge")
        self.assertEqual(mission.formation.spacing_m, 6.0)
        self.assertEqual(len(mission.leader_mission.actions), 3)
        self.assertEqual(len(mission.agents), 3)
        self.assertEqual(mission.agents[0].id, "leader")
        self.assertEqual(mission.agents[0].role, "leader")
        self.assertEqual(mission.agents[1].role, "wingman_left")
        self.assertEqual(mission.agents[1].slot, 1)
        self.assertEqual(mission.agents[2].role, "wingman_right")
        self.assertEqual(mission.agents[2].slot, 1)
        self.assertEqual(len(mission.connections), 3)
        self.assertEqual(mission.connections[0].udp, "udp://:14540")
        self.assertEqual(mission.connections[1].udp, "udp://:14541")
        self.assertEqual(mission.connections[2].udp, "udp://:14542")

    @patch('swarm_backend.llm.mission_parser._call_llm')
    def test_parse_to_mission_fallback(self, mock_call):
        # Simulate LLM failing or returning None for all steps
        mock_call.return_value = None

        mission = parse_to_mission(
            prompt="broken command",
            ai_engine="offline",
            api_key=None,
            num_drones=3
        )

        self.assertIsInstance(mission, Mission)
        self.assertEqual(mission.mode, "FORMATION")
        self.assertEqual(mission.formation.type, "wedge")
        self.assertEqual(mission.formation.spacing_m, 5.0)  # default
        self.assertEqual(len(mission.leader_mission.actions), 3)  # default takeoff, goto, land
