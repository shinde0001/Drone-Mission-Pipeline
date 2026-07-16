"""
swarm_backend/core/task_manager.py — Task management and spatial decomposition helpers.
Provides professional lawn-mower grid search ('sweep') waypoint generation and multi-agent area splitting.
"""
import math
from typing import List, Tuple, Dict, Any
from swarm_backend.config.schema import Action, GotoParams, SweepParams


def generate_lawn_mower_waypoints(
    sweep: SweepParams,
    start_north_m: float = 0.0,
    start_east_m: float = 0.0
) -> List[Action]:
    """
    Generate a sequence of 'goto' actions representing a lawn-mower search pattern
    across the rectangle defined by corner_a and corner_b.
    """
    min_n = min(sweep.corner_a_north_m, sweep.corner_b_north_m)
    max_n = max(sweep.corner_a_north_m, sweep.corner_b_north_m)
    min_e = min(sweep.corner_a_east_m, sweep.corner_b_east_m)
    max_e = max(sweep.corner_a_east_m, sweep.corner_b_east_m)

    lane_spacing = max(1.0, sweep.lane_spacing_m)
    alt = sweep.altitude_m
    speed = sweep.speed_mps

    # Determine whether to run lanes North-South or East-West based on aspect ratio
    n_span = max_n - min_n
    e_span = max_e - min_e

    actions = []

    if n_span >= e_span:
        # Lanes run North-South, stepping East
        num_lanes = max(1, int(math.ceil(e_span / lane_spacing)) + 1)
        actual_spacing = e_span / max(1, num_lanes - 1) if num_lanes > 1 else 0.0

        for lane_idx in range(num_lanes):
            lane_east = min_e + lane_idx * actual_spacing
            if lane_idx % 2 == 0:
                # South to North
                actions.append(Action(type="goto", params=GotoParams(north_m=min_n, east_m=lane_east, altitude_m=alt, speed_mps=speed)))
                actions.append(Action(type="goto", params=GotoParams(north_m=max_n, east_m=lane_east, altitude_m=alt, speed_mps=speed)))
            else:
                # North to South
                actions.append(Action(type="goto", params=GotoParams(north_m=max_n, east_m=lane_east, altitude_m=alt, speed_mps=speed)))
                actions.append(Action(type="goto", params=GotoParams(north_m=min_n, east_m=lane_east, altitude_m=alt, speed_mps=speed)))
    else:
        # Lanes run East-West, stepping North
        num_lanes = max(1, int(math.ceil(n_span / lane_spacing)) + 1)
        actual_spacing = n_span / max(1, num_lanes - 1) if num_lanes > 1 else 0.0

        for lane_idx in range(num_lanes):
            lane_north = min_n + lane_idx * actual_spacing
            if lane_idx % 2 == 0:
                # West to East
                actions.append(Action(type="goto", params=GotoParams(north_m=lane_north, east_m=min_e, altitude_m=alt, speed_mps=speed)))
                actions.append(Action(type="goto", params=GotoParams(north_m=lane_north, east_m=max_e, altitude_m=alt, speed_mps=speed)))
            else:
                # East to West
                actions.append(Action(type="goto", params=GotoParams(north_m=lane_north, east_m=max_e, altitude_m=alt, speed_mps=speed)))
                actions.append(Action(type="goto", params=GotoParams(north_m=lane_north, east_m=min_e, altitude_m=alt, speed_mps=speed)))

    return actions


def split_sweep_area(
    sweep: SweepParams,
    num_sub_areas: int
) -> List[SweepParams]:
    """
    Split a bounding box into N equal sub-regions (e.g. for N independent followers)
    so each drone can perform its own distinct lawn-mower sweep without overlap.
    """
    if num_sub_areas <= 1:
        return [sweep]

    min_n = min(sweep.corner_a_north_m, sweep.corner_b_north_m)
    max_n = max(sweep.corner_a_north_m, sweep.corner_b_north_m)
    min_e = min(sweep.corner_a_east_m, sweep.corner_b_east_m)
    max_e = max(sweep.corner_a_east_m, sweep.corner_b_east_m)

    n_span = max_n - min_n
    e_span = max_e - min_e

    sub_sweeps = []

    if e_span >= n_span:
        # Split along East-West axis
        slice_width = e_span / num_sub_areas
        for i in range(num_sub_areas):
            sub_min_e = min_e + i * slice_width
            sub_max_e = min_e + (i + 1) * slice_width
            sub_sweeps.append(SweepParams(
                corner_a_north_m=min_n,
                corner_a_east_m=sub_min_e,
                corner_b_north_m=max_n,
                corner_b_east_m=sub_max_e,
                altitude_m=sweep.altitude_m,
                lane_spacing_m=sweep.lane_spacing_m,
                speed_mps=sweep.speed_mps
            ))
    else:
        # Split along North-South axis
        slice_height = n_span / num_sub_areas
        for i in range(num_sub_areas):
            sub_min_n = min_n + i * slice_height
            sub_max_n = min_n + (i + 1) * slice_height
            sub_sweeps.append(SweepParams(
                corner_a_north_m=sub_min_n,
                corner_a_east_m=min_e,
                corner_b_north_m=sub_max_n,
                corner_b_east_m=max_e,
                altitude_m=sweep.altitude_m,
                lane_spacing_m=sweep.lane_spacing_m,
                speed_mps=sweep.speed_mps
            ))

    return sub_sweeps
