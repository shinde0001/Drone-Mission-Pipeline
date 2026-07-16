import unittest
import math
from swarm_backend.core.formation_controller import FormationController

class TestFormationController(unittest.TestCase):
    def test_wedge_body_offsets(self):
        # 135 degree wedge, spacing 10m
        controller = FormationController(formation_type="wedge", spacing_m=10.0, angle_deg=135.0)
        
        # Wingman left, slot 1
        # dx = 1 * 10 * cos(135 deg) = 10 * (-0.707) = -7.07
        # dy = -1 * 10 * sin(135 deg) = -10 * 0.707 = -7.07
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
        dx_l, dy_l, _ = controller.calculate_body_offset("wingman_left", 2)
        self.assertEqual(dx_l, 0.0)
        self.assertEqual(dy_l, -10.0)

    def test_heading_rotation(self):
        # Wedge with spacing 5m, angle 135 deg.
        controller = FormationController(formation_type="wedge", spacing_m=5.0, angle_deg=135.0, frame="body_relative")
        
        leader_pos = (100.0, 100.0, -10.0)
        
        # Heading 90 deg (facing East)
        # Leader is moving East. Wingman left (behind & left of leader) should be:
        # Behind leader = North offset is negative. Left of leader = East offset is negative.
        # Rotating body offset (-3.53, -3.53) by 90 deg:
        # offset_n = dx * cos(90) - dy * sin(90) = dx * 0 - dy * 1 = -dy = 3.53
        # offset_e = dx * sin(90) + dy * cos(90) = dx * 1 + dy * 0 = dx = -3.53
        # target_n = 100 + 3.53 = 103.53
        # target_e = 100 - 3.53 = 96.47
        dx, dy, _ = controller.calculate_body_offset("wingman_left", 1)
        target_n, target_e, target_d = controller.calculate_target_position(leader_pos, 90.0, "wingman_left", 1)
        
        expected_n = leader_pos[0] - dy
        expected_e = leader_pos[1] + dx
        self.assertAlmostEqual(target_n, expected_n)
        self.assertAlmostEqual(target_e, expected_e)
        self.assertEqual(target_d, -10.0)

    def test_world_ned_no_rotation(self):
        controller = FormationController(formation_type="wedge", spacing_m=5.0, angle_deg=135.0, frame="world_ned")
        leader_pos = (100.0, 100.0, -10.0)
        dx, dy, _ = controller.calculate_body_offset("wingman_left", 1)
        
        # With world_ned frame, target coordinates are leader_pos + raw body offsets, ignore heading
        target_n, target_e, target_d = controller.calculate_target_position(leader_pos, 270.0, "wingman_left", 1)
        self.assertEqual(target_n, leader_pos[0] + dx)
        self.assertEqual(target_e, leader_pos[1] + dy)

    def test_feedforward_velocity_clamping(self):
        controller = FormationController()
        leader_vel = (2.0, 3.0, 0.0)
        target_pos = (10.0, 10.0, -10.0)
        follower_pos = (0.0, 0.0, -10.0)
        
        # Pos errors = 10.0, 10.0, 0.0. Kp = 1.0. Corrections = 10.0, 10.0, 0.0.
        # Clamped corrections at max_correction_mps = 2.0.
        # Desired velocities: leader_vel + clamped corrections = (2.0 + 2.0, 3.0 + 2.0, 0.0 + 0.0) = (4.0, 5.0, 0.0)
        vn, ve, vd = controller.calculate_feedforward_velocity(
            leader_vel, target_pos, follower_pos, gain_kp=1.0, max_correction_mps=2.0
        )
        self.assertEqual(vn, 4.0)
        self.assertEqual(ve, 5.0)
        self.assertEqual(vd, 0.0)
