import unittest
from pydantic import ValidationError
from swarm_backend.config.schema import Mission

class TestSchema(unittest.TestCase):
    def setUp(self):
        self.valid_data = {
            "mission_id": "northward-flight-001",
            "mode": "FORMATION",
            "formation": {
                "type": "wedge",
                "spacing_m": 5.0,
                "angle_deg": 135.0,
                "frame": "body_relative"
            },
            "safety": {
                "collision_radius_m": 1.5,
                "min_separation_m": 3.0,
                "altitude_separation_m": 1.0,
                "avoidance_gain": 0.6,
                "max_correction_mps": 2.0
            },
            "agents": [
                { "id": "leader",   "role": "leader" },
                { "id": "drone_1",  "role": "wingman_left",  "slot": 1 },
                { "id": "drone_2",  "role": "wingman_right", "slot": 2 }
            ],
            "leader_mission": {
                "actions": [
                    { "type": "takeoff", "params": { "altitude_m": 15.0 } },
                    { "type": "goto", "params": { "north_m": 40.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0 } },
                    { "type": "land", "params": {} }
                ]
            },
            "connections": [
                { "id": "leader",  "udp": "udp://:14540" },
                { "id": "drone_1", "udp": "udp://:14541" },
                { "id": "drone_2", "udp": "udp://:14542" }
            ]
        }

    def test_valid_mission(self):
        mission = Mission(**self.valid_data)
        self.assertEqual(mission.mission_id, "northward-flight-001")
        self.assertEqual(len(mission.leader_mission.actions), 3)

    def test_invalid_takeoff_first(self):
        data = self.valid_data.copy()
        data["leader_mission"] = {
            "actions": [
                { "type": "goto", "params": { "north_m": 40.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0 } },
                { "type": "land", "params": {} }
            ]
        }
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("Mission must start with takeoff action", str(ctx.exception))

    def test_invalid_land_last(self):
        data = self.valid_data.copy()
        data["leader_mission"] = {
            "actions": [
                { "type": "takeoff", "params": { "altitude_m": 15.0 } },
                { "type": "goto", "params": { "north_m": 40.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0 } }
            ]
        }
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("Mission must end with land or return_to_launch action", str(ctx.exception))

    def test_invalid_separation_vs_collision(self):
        data = self.valid_data.copy()
        data["safety"] = {
            "collision_radius_m": 2.0,
            "min_separation_m": 3.0,  # Less than 2 * 2.0 = 4.0
            "altitude_separation_m": 1.0,
            "avoidance_gain": 0.6,
            "max_correction_mps": 2.0
        }
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("min_separation_m must be at least twice the collision_radius_m", str(ctx.exception))

    def test_invalid_spacing_vs_separation(self):
        data = self.valid_data.copy()
        data["formation"] = {
            "type": "wedge",
            "spacing_m": 2.5,  # Less than safety.min_separation_m (3.0)
            "angle_deg": 135.0,
            "frame": "body_relative"
        }
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("Formation spacing", str(ctx.exception))
        self.assertIn("must be greater than or equal to safety min_separation_m", str(ctx.exception))

    def test_missing_leader(self):
        data = self.valid_data.copy()
        data["agents"] = [
            { "id": "drone_1",  "role": "wingman_left",  "slot": 1 },
            { "id": "drone_2",  "role": "wingman_right", "slot": 2 }
        ]
        # Connections must match agents
        data["connections"] = [
            { "id": "drone_1", "udp": "udp://:14541" },
            { "id": "drone_2", "udp": "udp://:14542" }
        ]
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("exactly one agent with role 'leader'", str(ctx.exception))

    def test_mismatched_connections(self):
        data = self.valid_data.copy()
        data["connections"] = [
            { "id": "leader",  "udp": "udp://:14540" },
            { "id": "drone_1", "udp": "udp://:14541" }
            # drone_2 is missing
        ]
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("connections IDs must match exactly the agents IDs", str(ctx.exception))

    def test_duplicate_agent_ids(self):
        data = self.valid_data.copy()
        data["agents"] = [
            { "id": "leader",  "role": "leader" },
            { "id": "drone_1", "role": "wingman_left", "slot": 1 },
            { "id": "drone_1", "role": "wingman_right", "slot": 2 }
        ]
        data["connections"] = [
            { "id": "leader",  "udp": "udp://:14540" },
            { "id": "drone_1", "udp": "udp://:14541" },
            { "id": "drone_1", "udp": "udp://:14542" }
        ]
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("duplicate agent ids", str(ctx.exception))

    def test_invalid_altitude_exceeds_max(self):
        data = self.valid_data.copy()
        data["leader_mission"] = {
            "actions": [
                { "type": "takeoff", "params": { "altitude_m": 100.0 } },  # Exceeds max_altitude_m=50.0
                { "type": "land", "params": {} }
            ]
        }
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("exceeds max_altitude_m", str(ctx.exception))

    def test_invalid_speed_exceeds_max(self):
        data = self.valid_data.copy()
        data["leader_mission"] = {
            "actions": [
                { "type": "takeoff", "params": { "altitude_m": 15.0 } },
                { "type": "goto", "params": { "north_m": 40.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 30.0 } },  # Exceeds max_speed_mps=15.0
                { "type": "land", "params": {} }
            ]
        }
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("exceeds max_speed_mps", str(ctx.exception))

    def test_invalid_target_outside_geofence(self):
        data = self.valid_data.copy()
        data["leader_mission"] = {
            "actions": [
                { "type": "takeoff", "params": { "altitude_m": 15.0 } },
                { "type": "goto", "params": { "north_m": 2000.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0 } },  # Outside geofence
                { "type": "land", "params": {} }
            ]
        }
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("outside the configured geofence", str(ctx.exception))

    def test_exceeds_max_agents(self):
        data = self.valid_data.copy()
        agents = [{ "id": "leader", "role": "leader" }]
        connections = [{ "id": "leader", "udp": "udp://:14540" }]
        for i in range(1, 15):  # Exceeds max_agents (e.g. 10)
            agents.append({ "id": f"drone_{i}", "role": "wingman_left", "slot": i })
            connections.append({ "id": f"drone_{i}", "udp": f"udp://:{14540+i}" })
        data["agents"] = agents
        data["connections"] = connections
        with self.assertRaises(ValidationError) as ctx:
            Mission(**data)
        self.assertIn("exceeds max_agents", str(ctx.exception))

