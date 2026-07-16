import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_planner import plan_mission
from swarm_backend.llm.mission_parser import parse_to_mission

class TestOpenAIRemoval(unittest.TestCase):

    @patch('urllib.request.urlopen')
    def test_plan_mission_offline_local_urllib(self, mock_urlopen):
        """Test standard mission planning with a mocked local Ollama HTTP response."""
        # Mock Response Content
        mock_response_data = {
            "message": {
                "content": json.dumps({
                    "mission_name": "Mocked Mission",
                    "vehicle_type": "quadcopter",
                    "repeat_count": 1,
                    "actions": [
                        {"type": "takeoff", "params": {"altitude_m": 10.0}},
                        {"type": "goto", "params": {"north_m": 10.0, "east_m": 5.0, "altitude_m": 10.0, "speed_mps": 5.0}},
                        {"type": "land", "params": {}}
                    ]
                })
            }
        }
        
        def side_effect(req, *args, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            m = MagicMock()
            if "api/tags" in url:
                m.read.return_value = json.dumps({"models": [{"name": "llama3.2:1b"}]}).encode('utf-8')
            elif "api/chat" in url:
                m.read.return_value = json.dumps(mock_response_data).encode('utf-8')
            m.__enter__.return_value = m
            return m

        mock_urlopen.side_effect = side_effect

        # Call plan_mission
        mission = plan_mission(
            prompt="take off to 10m, fly to north 10m east 5m, then land",
            mode="standard",
            ai_engine="offline"
        )

        # Assertions
        self.assertEqual(mission["mission_name"], "Mocked Mission")
        self.assertEqual(len(mission["actions"]), 3)
        self.assertEqual(mission["actions"][0]["type"], "takeoff")
        self.assertEqual(mission["actions"][1]["type"], "goto")
        self.assertEqual(mission["actions"][2]["type"], "land")

    @patch('urllib.request.urlopen')
    def test_swarm_backend_mission_parser(self, mock_urlopen):
        """Test the mission parser in swarm_backend to ensure it works via mocked urllib."""
        # 3 calls: 1. mode classification, 2. formation params, 3. leader actions
        mock_mode_data = {"cleaned_prompt": "three drones wedge", "mode": "FORMATION"}
        mock_form_data = {"type": "wedge", "spacing_m": 5.0, "angle_deg": 135.0, "frame": "body_relative"}
        mock_leader_data = {
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 10.0}},
                {"type": "goto", "params": {"north_m": 20.0, "east_m": 0.0, "altitude_m": 10.0, "speed_mps": 5.0}},
                {"type": "land", "params": {}}
            ]
        }

        responses_queue = [
            {"message": {"content": json.dumps(mock_mode_data)}},
            {"message": {"content": json.dumps(mock_form_data)}},
            {"message": {"content": json.dumps(mock_leader_data)}}
        ]

        def side_effect(req, *args, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            m = MagicMock()
            if "api/tags" in url:
                m.read.return_value = json.dumps({"models": [{"name": "llama3.2:1b"}]}).encode('utf-8')
            elif "api/chat" in url:
                r = responses_queue.pop(0)
                m.read.return_value = json.dumps(r).encode('utf-8')
            m.__enter__.return_value = m
            return m

        mock_urlopen.side_effect = side_effect

        # Call parse_to_mission
        mission = parse_to_mission(
            prompt="Three drones takeoff to 10m and fly north 20m in wedge formation",
            ai_engine="offline"
        )

        # Assertions
        self.assertEqual(mission.mode, "FORMATION")
        self.assertEqual(mission.formation.type, "wedge")
        self.assertEqual(mission.formation.spacing_m, 5.0)
        self.assertEqual(len(mission.leader_mission.actions), 3)

if __name__ == '__main__':
    unittest.main()
