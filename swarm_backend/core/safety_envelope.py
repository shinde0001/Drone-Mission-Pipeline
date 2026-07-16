import math
from typing import Tuple

class SafetyEnvelope:
    """
    Enforces live physical limits (geofence, altitude limits, and max speeds)
    on position and velocity setpoints before they are sent to the flight controller.
    """
    def __init__(
        self,
        max_altitude_m: float = 30.0,
        min_altitude_m: float = 1.0,
        geofence_radius_m: float = 100.0,
        max_speed_mps: float = 8.0
    ):
        self.max_altitude_m = max_altitude_m
        self.min_altitude_m = min_altitude_m
        self.geofence_radius_m = geofence_radius_m
        self.max_speed_mps = max_speed_mps

    def clamp_position(self, pos_ned: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Enforces geofence and altitude limits on the target position.
        Note: NED coordinate frame has Down as Z (negative is upward altitude).
        """
        n, e, d = pos_ned

        # 1. Enforce Altitude Limits (Z-axis / Down)
        # altitude = -d
        max_d = -self.min_altitude_m  # lower altitude ceiling = closer to ground (0)
        min_d = -self.max_altitude_m  # upper altitude ceiling = further from ground
        
        clamped_d = max(min_d, min(max_d, d))

        # 2. Enforce Geofence Radius (Horizontal plane)
        horiz_dist = math.sqrt(n**2 + e**2)
        if horiz_dist > self.geofence_radius_m:
            # Project position back to boundary limit
            scale = self.geofence_radius_m / horiz_dist
            n *= scale
            e *= scale

        return n, e, clamped_d

    def clamp_velocity(self, vel_ned: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Enforces maximum speed limits on the target velocity vector.
        """
        vn, ve, vd = vel_ned
        speed = math.sqrt(vn**2 + ve**2 + vd**2)

        if speed > self.max_speed_mps:
            # Scale down the vector components while preserving direction
            scale = self.max_speed_mps / speed
            vn *= scale
            ve *= scale
            vd *= scale

        return vn, ve, vd

    def enforce(
        self,
        pos_ned: Tuple[float, float, float],
        vel_ned: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Enforce complete safety envelope on both position and velocity.
        """
        clamped_pos = self.clamp_position(pos_ned)
        clamped_vel = self.clamp_velocity(vel_ned)
        return clamped_pos, clamped_vel
