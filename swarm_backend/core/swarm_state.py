import threading
from typing import Dict, Tuple, List

class SwarmState:
    """
    Shared live telemetry bus for the swarm.
    Thread-safe and optimized for high-frequency updates (10-20 Hz).
    """
    def __init__(self):
        self._lock = threading.Lock()
        # Storage: agent_id -> Tuple[north_m, east_m, down_m]
        self._positions: Dict[str, Tuple[float, float, float]] = {}
        # Storage: agent_id -> Tuple[vn_mps, ve_mps, vd_mps]
        self._velocities: Dict[str, Tuple[float, float, float]] = {}
        # Storage: agent_id -> heading_deg
        self._headings: Dict[str, float] = {}

    def update_position(self, agent_id: str, position_ned: Tuple[float, float, float]) -> None:
        """Update the NED position of a drone."""
        with self._lock:
            self._positions[agent_id] = position_ned

    def update_velocity(self, agent_id: str, velocity_ned: Tuple[float, float, float]) -> None:
        """Update the velocity of a drone."""
        with self._lock:
            self._velocities[agent_id] = velocity_ned

    def update_heading(self, agent_id: str, heading_deg: float) -> None:
        """Update the heading of a drone."""
        with self._lock:
            self._headings[agent_id] = heading_deg

    def get_position(self, agent_id: str) -> Tuple[float, float, float]:
        """Get the NED position of a specific drone. Returns (0, 0, 0) if not set."""
        with self._lock:
            return self._positions.get(agent_id, (0.0, 0.0, 0.0))

    def get_velocity(self, agent_id: str) -> Tuple[float, float, float]:
        """Get the velocity of a specific drone. Returns (0, 0, 0) if not set."""
        with self._lock:
            return self._velocities.get(agent_id, (0.0, 0.0, 0.0))

    def get_heading(self, agent_id: str) -> float:
        """Get the heading of a specific drone. Returns 0.0 if not set."""
        with self._lock:
            return self._headings.get(agent_id, 0.0)

    def all_positions_except(self, agent_id: str) -> List[Tuple[float, float, float]]:
        """Get a list of positions for all drones except the specified one (for collision checks)."""
        with self._lock:
            return [pos for aid, pos in self._positions.items() if aid != agent_id]

    def reset(self) -> None:
        """Reset the telemetry states."""
        with self._lock:
            self._positions.clear()
            self._velocities.clear()
            self._headings.clear()
