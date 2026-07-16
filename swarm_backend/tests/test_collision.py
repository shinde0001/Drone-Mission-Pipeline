import unittest
from swarm_backend.core.collision_avoidance import CollisionAvoidance

class TestCollisionAvoidance(unittest.TestCase):
    def test_no_repulsion_when_safe(self):
        # min_separation = 3.0
        ca = CollisionAvoidance(min_separation_m=3.0, avoidance_gain=1.0)
        my_pos = (0.0, 0.0, 0.0)
        # Safe distance (5m)
        other_positions = [(5.0, 0.0, 0.0)]
        
        rep_n, rep_e, rep_d = ca.calculate_repulsion(my_pos, other_positions)
        self.assertEqual(rep_n, 0.0)
        self.assertEqual(rep_e, 0.0)
        self.assertEqual(rep_d, 0.0)

    def test_repulsion_triggered(self):
        ca = CollisionAvoidance(min_separation_m=3.0, avoidance_gain=1.0)
        my_pos = (0.0, 0.0, 0.0)
        # Violates separation (2m away to the North)
        other_positions = [(2.0, 0.0, 0.0)]
        
        # Should push to the South (negative North)
        rep_n, rep_e, rep_d = ca.calculate_repulsion(my_pos, other_positions)
        self.assertTrue(rep_n < 0.0)
        self.assertEqual(rep_e, 0.0)
        self.assertEqual(rep_d, 0.0)
        
        # Magnitude check: gain * (min_sep - dist) / dist * dn
        # dn = my_pos[0] - other_pos[0] = -2.0. dist = 2.0.
        # mag = 1.0 * (3.0 - 2.0) / 2.0 = 0.5.
        # rep_n = 0.5 * -2.0 = -1.0.
        self.assertAlmostEqual(rep_n, -1.0)

    def test_apply_avoidance_to_velocity(self):
        ca = CollisionAvoidance(min_separation_m=3.0, avoidance_gain=1.0)
        my_pos = (0.0, 0.0, 0.0)
        other_positions = [(2.0, 0.0, 0.0)]
        
        nominal_vel = (2.0, 2.0, 0.0)
        final_vn, final_ve, final_vd = ca.apply_avoidance_to_velocity(
            nominal_vel, my_pos, other_positions
        )
        # rep_n = -1.0. final_vn = 2.0 + (-1.0) = 1.0
        self.assertAlmostEqual(final_vn, 1.0)
        self.assertAlmostEqual(final_ve, 2.0)
        self.assertAlmostEqual(final_vd, 0.0)

    def test_co_location(self):
        ca = CollisionAvoidance(min_separation_m=3.0, avoidance_gain=0.6)
        my_pos = (0.0, 0.0, 0.0)
        other_positions = [(0.0, 0.0, 0.0)]
        
        rep_n, _, _ = ca.calculate_repulsion(my_pos, other_positions)
        self.assertAlmostEqual(rep_n, 0.6)
