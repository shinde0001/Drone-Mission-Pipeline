import math
from typing import Tuple, Literal, Union
from swarm_backend.config.schema import FormationType


class FormationController:
    """
    Computes desired body-relative formation positions and velocity feedforward commands
    based on the leader's real-time position, velocity, and heading.
    Enforces Rule #1: Formation offsets are body-frame, never world-frame.
    """
    def __init__(
        self,
        formation_type: Union[FormationType, str] = FormationType.WEDGE,
        spacing_m: float = 5.0,
        angle_deg: float = 135.0,
        frame: Literal["body_relative"] = "body_relative"
    ):
        if isinstance(formation_type, str):
            try:
                self.formation_type = FormationType(formation_type.lower())
            except ValueError:
                if formation_type.lower() == "line":
                    self.formation_type = FormationType.LINE_ABREAST
                else:
                    self.formation_type = FormationType.WEDGE
        else:
            self.formation_type = formation_type

        self.spacing_m = spacing_m
        self.angle_deg = angle_deg
        if frame != "body_relative":
            raise ValueError("FormationController only supports 'body_relative' frame (Rule #1).")
        self.frame = "body_relative"

    def calculate_body_offset(self, role: str, slot: int) -> Tuple[float, float, float]:
        """
        Compute the raw offset (dx, dy, dz) in the leader's body frame.
        - Body X: forward (positive), backward (negative)
        - Body Y: right (positive), left (negative)
        - Body Z: down (positive)
        """
        if slot <= 0 or role == "leader":
            return 0.0, 0.0, 0.0

        # Determine side (-1.0 for left, +1.0 for right)
        if role == "wingman_left" or (role in ("wingman", "independent") and slot % 2 != 0):
            side = -1.0
        else:
            side = 1.0

        rank = float((slot + 1) // 2)

        if self.formation_type in (FormationType.LINE_ABREAST, FormationType.LINE, "line"):
            # Perpendicular to flight direction
            dx = 0.0
            dy = side * rank * self.spacing_m
            dz = 0.0
        elif self.formation_type == FormationType.COLUMN:
            # Inline behind the leader sequentially
            dx = -float(slot) * self.spacing_m
            dy = 0.0
            dz = 0.0
        elif self.formation_type == FormationType.DIAMOND:
            # Diamond formation:
            # Slot 1 (left): (-spacing_m, -spacing_m, 0.0)
            # Slot 2 (right): (-spacing_m, +spacing_m, 0.0)
            # Slot 3 (tail behind leader): (-2 * spacing_m, 0.0, 0.0)
            # Slot 4+: outer diamond rings
            if slot == 1:
                dx, dy, dz = -self.spacing_m, -self.spacing_m, 0.0
            elif slot == 2:
                dx, dy, dz = -self.spacing_m, self.spacing_m, 0.0
            elif slot == 3:
                dx, dy, dz = -2.0 * self.spacing_m, 0.0, 0.0
            else:
                outer_rank = float((slot + 1) // 2)
                dx, dy, dz = -outer_rank * self.spacing_m, side * (outer_rank - 1.0) * self.spacing_m, 0.0
        else:  # WEDGE (default)
            angle_rad = math.radians(self.angle_deg)
            # cos(135 deg) is negative, so dx is negative (behind leader)
            dx = rank * self.spacing_m * math.cos(angle_rad)
            # sin(135 deg) is positive, side is -1 or +1
            dy = side * rank * self.spacing_m * math.sin(angle_rad)
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
        by rotating body-relative offsets into the world frame based on leader's heading.
        """
        dx, dy, dz = self.calculate_body_offset(role, slot)

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
