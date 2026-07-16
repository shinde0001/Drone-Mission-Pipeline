import math
from typing import Tuple, Literal

class FormationController:
    """
    Computes desired body-relative formation positions and velocity feedforward commands
    based on the leader's real-time position, velocity, and heading.
    """
    def __init__(
        self,
        formation_type: Literal["wedge", "line", "column"] = "wedge",
        spacing_m: float = 5.0,
        angle_deg: float = 135.0,
        frame: Literal["body_relative", "world_ned"] = "body_relative"
    ):
        self.formation_type = formation_type
        self.spacing_m = spacing_m
        self.angle_deg = angle_deg
        self.frame = frame

    def calculate_body_offset(self, role: str, slot: int) -> Tuple[float, float, float]:
        """
        Compute the raw offset (dx, dy, dz) in the leader's body frame.
        - Body X: forward (positive)
        - Body Y: right (positive)
        - Body Z: down (positive)
        """
        s = float(slot)
        if self.formation_type == "line":
            # Perpendicular to flight direction
            dx = 0.0
            dy = -s * self.spacing_m if role == "wingman_left" else s * self.spacing_m
            dz = 0.0
        elif self.formation_type == "column":
            # Inline behind the leader
            # Sort slots sequentially
            dx = -s * self.spacing_m
            dy = 0.0
            dz = 0.0
        else:  # wedge (default)
            # Symmetric wing shape pointing backward
            angle_rad = math.radians(self.angle_deg)
            dx = s * self.spacing_m * math.cos(angle_rad)
            dy = -s * self.spacing_m * math.sin(angle_rad) if role == "wingman_left" else s * self.spacing_m * math.sin(angle_rad)
            dz = 0.0
        return dx, dy, dz

    def calculate_target_position(
        self,
        leader_pos: Tuple[float, float, float],
        leader_heading_deg: float,
        role: str,
        slot: int
    ) -> Tuple[float, float, float]:
        """
        Calculate the target NED position (North, East, Down) for a wingman
        by projecting body-relative offsets into the world frame.
        """
        dx, dy, dz = self.calculate_body_offset(role, slot)

        if self.frame == "world_ned":
            # No rotation based on leader heading
            target_n = leader_pos[0] + dx
            target_e = leader_pos[1] + dy
            target_d = leader_pos[2] + dz
        else:
            # Rotate body offset by leader's heading (NED: North is X, East is Y)
            yaw_rad = math.radians(leader_heading_deg)
            cos_y = math.cos(yaw_rad)
            sin_y = math.sin(yaw_rad)

            offset_n = dx * cos_y - dy * sin_y
            offset_e = dx * sin_y + dy * cos_y

            target_n = leader_pos[0] + offset_n
            target_e = leader_pos[1] + offset_e
            target_d = leader_pos[2] + dz

        return target_n, target_e, target_d

    def calculate_feedforward_velocity(
        self,
        leader_vel: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        follower_pos: Tuple[float, float, float],
        gain_kp: float = 1.0,
        max_correction_mps: float = 2.0
    ) -> Tuple[float, float, float]:
        """
        Calculate the target velocity setpoint using velocity feedforward (leader's velocity)
        plus a proportional position error correction.
        """
        err_n = target_pos[0] - follower_pos[0]
        err_e = target_pos[1] - follower_pos[1]
        err_d = target_pos[2] - follower_pos[2]

        # Calculate proportional corrections
        corr_n = gain_kp * err_n
        corr_e = gain_kp * err_e
        corr_d = gain_kp * err_d

        # Clamp individual correction components to prevent extreme speed spikes
        corr_n = max(-max_correction_mps, min(max_correction_mps, corr_n))
        corr_e = max(-max_correction_mps, min(max_correction_mps, corr_e))
        corr_d = max(-max_correction_mps, min(max_correction_mps, corr_d))

        target_vn = leader_vel[0] + corr_n
        target_ve = leader_vel[1] + corr_e
        target_vd = leader_vel[2] + corr_d

        return target_vn, target_ve, target_vd
