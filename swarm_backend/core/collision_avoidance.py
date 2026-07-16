import math
from typing import List, Tuple

class CollisionAvoidance:
    """
    Computes repulsion vectors using an Artificial Potential Field (APF) model
    to prevent inter-agent collisions when proximity thresholds are violated.
    """
    def __init__(self, min_separation_m: float = 3.0, avoidance_gain: float = 0.6):
        self.min_separation_m = min_separation_m
        self.avoidance_gain = avoidance_gain

    def calculate_repulsion(
        self,
        my_pos: Tuple[float, float, float],
        other_positions: List[Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        """
        Calculate a 3D repulsion velocity vector to resolve distance violations.
        - my_pos: current (North, East, Down) position of this drone
        - other_positions: list of (North, East, Down) positions of other drones
        """
        rep_n = 0.0
        rep_e = 0.0
        rep_d = 0.0

        for other_pos in other_positions:
            dn = my_pos[0] - other_pos[0]
            de = my_pos[1] - other_pos[1]
            dd = my_pos[2] - other_pos[2]
            dist = math.sqrt(dn**2 + de**2 + dd**2)

            if dist < 0.001:
                # Handle exact co-location to avoid division by zero
                # Apply a slight lateral repulsion nudge
                rep_n += self.avoidance_gain
                continue

            if dist < self.min_separation_m:
                # Repulsive force is proportional to the violation depth
                # and points radially outward (away from the obstacle)
                force_magnitude = self.avoidance_gain * (self.min_separation_m - dist) / dist
                rep_n += force_magnitude * dn
                rep_e += force_magnitude * de
                rep_d += force_magnitude * dd

        return rep_n, rep_e, rep_d

    def apply_avoidance_to_velocity(
        self,
        nominal_vel: Tuple[float, float, float],
        my_pos: Tuple[float, float, float],
        other_positions: List[Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        """
        Superimpose potential-field repulsion onto the nominal formation velocity command.
        """
        rep_n, rep_e, rep_d = self.calculate_repulsion(my_pos, other_positions)
        
        final_vn = nominal_vel[0] + rep_n
        final_ve = nominal_vel[1] + rep_e
        final_vd = nominal_vel[2] + rep_d

        return final_vn, final_ve, final_vd
