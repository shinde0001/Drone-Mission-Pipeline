import unittest
import math
from swarm_backend.core.formation_controller import FormationController

class TestFormationController(unittest.TestCase):
    def test_wedge_body_offsets(self):
        # 135 degree wedge, spacing 10m
        controller = FormationController(formation_type="wedge", spacing_m=10.0, angle_deg=135.0)
        
        # Wingman left, slot 1
        dx_l, dy_l, dz_l = controller.calculate_body_offset("wingman_left", 1)
        self.assertAlmostEqual(dx_l, 10.0 * math.cos(math.radians(135.0)))
        self.assertAlmostEqual(dy_l, -10.0 * math.sin(math.radians(135.0)))
        self.assertEqual(dz_l, 0.0)

        # Wingman right, slot 1
        dx_r, dy_r, dz_r = controller.calculate_body_offset("wingman_right", 1)
        self.assertAlmostEqual(dx_r, 10.0 * math.cos(math.radians(135.0)))
        self.assertAlmostEqual(dy_r, 10.0 * math.sin(math.radians(135.0)))
        self.assertEqual(dz_r, 0.0)

    def test_line_body_offsets(self):
        controller = FormationController(formation_type="line", spacing_m=5.0)
        # Slot 3 (wingman_left rank 2)
        dx_l, dy_l, _ = controller.calculate_body_offset("wingman_left", 3)
        self.assertEqual(dx_l, 0.0)
        self.assertEqual(dy_l, -10.0)

    def test_heading_rotation(self):
        # Wedge with spacing 5m, angle 135 deg.
        controller = FormationController(formation_type="wedge", spacing_m=5.0, angle_deg=135.0, frame="body_relative")
        leader_pos = (100.0, 100.0, -10.0)
        dx, dy, _ = controller.calculate_body_offset("wingman_left", 1)
        
        # Heading 0 deg (facing North)
        # Rotating body offset (dx, dy) by 0 deg -> world offset is (dx, dy)
        target_n_0, target_e_0, target_d_0 = controller.calculate_target_position(leader_pos, 0.0, "wingman_left", 1)
        self.assertAlmostEqual(target_n_0, leader_pos[0] + dx)
        self.assertAlmostEqual(target_e_0, leader_pos[1] + dy)
        self.assertEqual(target_d_0, -10.0)

        # Heading 90 deg (facing East)
        # Rotating body offset by 90 deg -> world offset is (-dy, dx)
        target_n_90, target_e_90, target_d_90 = controller.calculate_target_position(leader_pos, 90.0, "wingman_left", 1)
        self.assertAlmostEqual(target_n_90, leader_pos[0] - dy)
        self.assertAlmostEqual(target_e_90, leader_pos[1] + dx)
        self.assertEqual(target_d_90, -10.0)

        # Verify target position strictly differs when heading rotates
        self.assertNotEqual(target_n_0, target_n_90)
        self.assertNotEqual((target_n_0, target_e_0), (target_n_90, target_e_90))

    def test_invalid_world_ned_frame(self):
        with self.assertRaises(ValueError) as ctx:
            FormationController(formation_type="wedge", spacing_m=5.0, angle_deg=135.0, frame="world_ned")
        self.assertIn("only supports 'body_relative' frame", str(ctx.exception))

    def test_feedforward_velocity_clamping(self):
        controller = FormationController()
        leader_vel = (2.0, 3.0, 0.0)
        target_pos = (10.0, 10.0, -10.0)
        follower_pos = (0.0, 0.0, -10.0)
        
        vn, ve, vd = controller.calculate_feedforward_velocity(
            leader_vel, target_pos, follower_pos, gain_kp=1.0, max_correction_mps=2.0
        )
        self.assertEqual(vn, 4.0)
        self.assertEqual(ve, 5.0)
        self.assertEqual(vd, 0.0)

if __name__ == "__main__":
    unittest.main()

