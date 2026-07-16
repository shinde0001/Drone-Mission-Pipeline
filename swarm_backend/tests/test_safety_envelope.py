import unittest
import math
from swarm_backend.core.safety_envelope import SafetyEnvelope

class TestSafetyEnvelope(unittest.TestCase):
    def setUp(self):
        # max_alt=30m, min_alt=1m, geofence=100m, max_speed=8m/s
        self.envelope = SafetyEnvelope(
            max_altitude_m=30.0,
            min_altitude_m=1.0,
            geofence_radius_m=100.0,
            max_speed_mps=8.0
        )

    def test_altitude_clamping(self):
        # Under min altitude (0.5m -> Down = -0.5m)
        pos = (10.0, 10.0, -0.5)
        clamped_pos = self.envelope.clamp_position(pos)
        self.assertEqual(clamped_pos[2], -1.0)  # Clamped to min_altitude_m (1m)

        # Over max altitude (35m -> Down = -35m)
        pos = (10.0, 10.0, -35.0)
        clamped_pos = self.envelope.clamp_position(pos)
        self.assertEqual(clamped_pos[2], -30.0)  # Clamped to max_altitude_m (30m)

    def test_geofence_clamping(self):
        # Inside geofence
        pos = (50.0, 50.0, -10.0)
        clamped_pos = self.envelope.clamp_position(pos)
        self.assertEqual(clamped_pos, pos)

        # Outside geofence (North 120m, East 90m -> dist = 150m)
        pos = (120.0, 90.0, -10.0)
        clamped_pos = self.envelope.clamp_position(pos)
        
        dist = math.sqrt(clamped_pos[0]**2 + clamped_pos[1]**2)
        self.assertAlmostEqual(dist, 100.0)
        self.assertAlmostEqual(clamped_pos[0], 80.0)  # 120 * (100 / 150) = 80
        self.assertAlmostEqual(clamped_pos[1], 60.0)  # 90 * (100 / 150) = 60

    def test_speed_clamping(self):
        # Under limit
        vel = (4.0, 4.0, 0.0)
        clamped_vel = self.envelope.clamp_velocity(vel)
        self.assertEqual(clamped_vel, vel)

        # Over limit (10 m/s total speed)
        vel = (6.0, 8.0, 0.0)
        clamped_vel = self.envelope.clamp_velocity(vel)
        
        speed = math.sqrt(clamped_vel[0]**2 + clamped_vel[1]**2)
        self.assertAlmostEqual(speed, 8.0)
        self.assertAlmostEqual(clamped_vel[0], 4.8)  # 6.0 * (8 / 10) = 4.8
        self.assertAlmostEqual(clamped_vel[1], 6.4)  # 8.0 * (8 / 10) = 6.4
