"""
mission_validator.py — Stage 2: Schema & Safety Validation

This module is the GUARDRAIL LAYER. It validates the mission JSON produced
by the LLM planner against:
  1. JSON Schema — structural correctness
  2. Safety limits — altitude, speed, geofence, action whitelist
  3. Sanity checks — mission structure, duplicate waypoints, distance

The validator ensures that ONLY safe, well-formed missions reach the
deterministic executor. This is the critical safety barrier between
AI-generated plans and actual drone commands.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from .utils import (
    load_mission_schema,
    load_safety_limits,
    print_success,
    print_error,
    print_warning,
    setup_logger,
)
from swarm_backend.config.schema import Mission as SwarmMission, load_user_constraints

logger = setup_logger("validator")


@dataclass
class ValidationResult:
    """Result of mission validation."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def print_report(self) -> None:
        """Print a human-readable validation report."""
        if self.valid:
            print_success("Mission PASSED all validation checks")
        else:
            print_error("Mission FAILED validation")

        if self.errors:
            for err in self.errors:
                print_error(f"  ERROR: {err}")

        if self.warnings:
            for warn in self.warnings:
                print_warning(f"  WARNING: {warn}")

        if not self.errors and not self.warnings:
            print_success("  No errors or warnings")


def _validate_schema(mission: dict, schema: dict,
                     result: ValidationResult) -> None:
    """Layer 1: Validate mission JSON against the JSON schema."""
    try:
        jsonschema.validate(instance=mission, schema=schema)
        logger.info("Schema validation passed")
    except jsonschema.ValidationError as e:
        result.add_error(f"Schema violation: {e.message} "
                         f"(path: {'.'.join(str(p) for p in e.path)})")
    except jsonschema.SchemaError as e:
        result.add_error(f"Invalid schema: {e.message}")


def _validate_safety(mission: dict, limits: dict,
                     result: ValidationResult) -> None:
    """Layer 2: Check all safety constraints."""
    actions = mission.get("actions", [])
    allowed_types = set(limits.get("allowed_actions", []))

    for i, action in enumerate(actions):
        action_type = action.get("type", "unknown")
        params = action.get("params", {})

        # Check action type whitelist
        if action_type not in allowed_types:
            result.add_error(
                f"Action {i}: unknown action type '{action_type}'. "
                f"Allowed: {allowed_types}"
            )

        # Check altitude limits
        alt = params.get("altitude_m")
        if alt is not None:
            if alt > limits["max_altitude_m"]:
                result.add_error(
                    f"Action {i} ({action_type}): altitude {alt}m exceeds "
                    f"maximum {limits['max_altitude_m']}m"
                )
            if alt < limits["min_altitude_m"]:
                result.add_error(
                    f"Action {i} ({action_type}): altitude {alt}m below "
                    f"minimum {limits['min_altitude_m']}m"
                )

        # Check speed limits
        speed = params.get("speed_mps")
        if speed is not None:
            if speed > limits["max_speed_mps"]:
                result.add_error(
                    f"Action {i} ({action_type}): speed {speed} m/s exceeds "
                    f"maximum {limits['max_speed_mps']} m/s"
                )
            if speed < limits["min_speed_mps"]:
                result.add_error(
                    f"Action {i} ({action_type}): speed {speed} m/s below "
                    f"minimum {limits['min_speed_mps']} m/s"
                )
                
            # Professional: low-altitude speed check
            low_alt_thresh = limits.get("low_alt_threshold_m", 5.0)
            max_low_speed = limits.get("max_speed_at_low_alt_mps", 5.0)
            # Find the altitude for this action (default to min if not present to be safe)
            alt_for_speed = alt if alt is not None else limits["min_altitude_m"]
            if alt_for_speed <= low_alt_thresh and speed > max_low_speed:
                result.add_error(
                    f"Action {i} ({action_type}): speed {speed} m/s is too fast "
                    f"for low altitude ({alt_for_speed}m). Max speed below {low_alt_thresh}m "
                    f"is {max_low_speed} m/s."
                )

        # Check geofence (NED coordinates)
        north = params.get("north_m", 0)
        east = params.get("east_m", 0)
        distance_from_home = math.sqrt(north ** 2 + east ** 2)
        if distance_from_home > limits["geofence_radius_m"]:
            result.add_error(
                f"Action {i} ({action_type}): waypoint at "
                f"N={north}m E={east}m is {distance_from_home:.1f}m from home, "
                f"exceeds geofence radius {limits['geofence_radius_m']}m"
            )

        # Check loiter duration
        duration = params.get("duration_s")
        if duration is not None:
            if duration > limits["max_loiter_duration_s"]:
                result.add_error(
                    f"Action {i} ({action_type}): loiter duration {duration}s "
                    f"exceeds maximum {limits['max_loiter_duration_s']}s"
                )

    # Check repeat count
    repeat = mission.get("repeat_count", 1)
    if repeat > limits.get("max_repeat_count", 10):
        result.add_error(
            f"repeat_count {repeat} exceeds maximum "
            f"{limits['max_repeat_count']}"
        )

    # Check total action count
    max_actions = limits.get("max_actions_per_mission", 50)
    if len(actions) > max_actions:
        result.add_error(
            f"Mission has {len(actions)} actions, exceeds maximum "
            f"{max_actions}"
        )


def _validate_sanity(mission: dict, limits: dict,
                     result: ValidationResult) -> None:
    """Layer 3: Structural sanity checks."""
    actions = mission.get("actions", [])

    if not actions:
        result.add_error("Mission has no actions")
        return

    # Must start with takeoff
    if limits.get("require_takeoff_first", True):
        if actions[0].get("type") != "takeoff":
            result.add_error(
                f"Mission must start with 'takeoff', "
                f"but starts with '{actions[0].get('type')}'"
            )

    # Must end with land or return_to_launch
    if limits.get("require_landing_last", True):
        last_type = actions[-1].get("type")
        if last_type not in ("land", "return_to_launch"):
            result.add_error(
                f"Mission must end with 'land' or 'return_to_launch', "
                f"but ends with '{last_type}'"
            )

    # Must have at least one waypoint action between takeoff and landing
    middle_actions = [a for a in actions[1:-1]
                      if a.get("type") in ("goto", "loiter")]
    if not middle_actions:
        result.add_warning(
            "Mission has no 'goto' or 'loiter' actions between "
            "takeoff and landing — the drone won't go anywhere"
        )

    # Check for duplicate consecutive waypoints
    for i in range(1, len(actions)):
        prev = actions[i - 1]
        curr = actions[i]
        if (prev.get("type") == "goto" and curr.get("type") == "goto"):
            p_params = prev.get("params", {})
            c_params = curr.get("params", {})
            if (p_params.get("north_m") == c_params.get("north_m") and
                    p_params.get("east_m") == c_params.get("east_m")):
                result.add_warning(
                    f"Actions {i - 1} and {i} are duplicate consecutive "
                    f"goto waypoints (N={c_params.get('north_m')}m, "
                    f"E={c_params.get('east_m')}m)"
                )

    # Estimate total mission distance and flight time
    total_distance = 0.0
    estimated_time_s = 0.0
    prev_north, prev_east = 0.0, 0.0
    
    min_spacing = limits.get("min_waypoint_spacing_m", 1.0)
    max_leg = limits.get("max_leg_distance_m", 500.0)

    for i, action in enumerate(actions):
        atype = action.get("type")
        if atype == "takeoff":
            alt = action.get("params", {}).get("altitude_m", 10.0)
            estimated_time_s += alt / 2.0  # assume 2 m/s ascent
        elif atype in ("land", "return_to_launch"):
            estimated_time_s += 15.0  # approximate landing time
        elif atype == "loiter":
            estimated_time_s += action.get("params", {}).get("duration_s", 5)
        elif atype == "goto":
            params = action.get("params", {})
            n = params.get("north_m", 0)
            e = params.get("east_m", 0)
            speed = params.get("speed_mps", 5.0)
            
            leg_dist = math.sqrt((n - prev_north) ** 2 + (e - prev_east) ** 2)
            
            # Professional: VLOS check
            if leg_dist > max_leg:
                result.add_warning(
                    f"Action {i} (goto): Leg distance {leg_dist:.0f}m exceeds "
                    f"recommended VLOS (Visual Line of Sight) of {max_leg}m"
                )
                
            # Professional: Jitter/clutter check
            if 0 < leg_dist < min_spacing:
                result.add_warning(
                    f"Action {i} (goto): Waypoint is only {leg_dist:.1f}m from previous. "
                    f"Spacing below {min_spacing}m can cause drone jitter"
                )

            total_distance += leg_dist
            if speed > 0:
                estimated_time_s += leg_dist / speed
            prev_north, prev_east = n, e

    repeat = mission.get("repeat_count", 1)
    total_distance *= repeat
    estimated_time_s *= repeat

    max_dist = limits.get("max_total_distance_m", 5000)
    if total_distance > max_dist:
        result.add_error(
            f"Estimated total distance {total_distance:.0f}m "
            f"(with {repeat}x repeat) exceeds maximum {max_dist}m"
        )
    elif total_distance > max_dist * 0.8:
        result.add_warning(
            f"Estimated total distance {total_distance:.0f}m is "
            f"approaching the {max_dist}m limit"
        )
        
    # Professional: Battery capacity / flight time check
    max_time = limits.get("max_flight_time_s", 900)
    if estimated_time_s > max_time:
        result.add_error(
            f"Estimated flight time {estimated_time_s/60:.1f} minutes "
            f"exceeds battery limit ({max_time/60:.1f} mins)"
        )
    elif estimated_time_s > max_time * 0.8:
        result.add_warning(
            f"Estimated flight time {estimated_time_s/60:.1f} minutes "
            f"is approaching battery limit"
        )


def validate_mission(mission: dict) -> ValidationResult:
    """
    Stage 2: Validate a mission JSON against schema, safety, and sanity rules.

    This is the GUARDRAIL LAYER — the critical safety barrier between
    AI-generated plans and actual drone commands.

    Args:
        mission: Mission JSON dict from the LLM planner.

    Returns:
        ValidationResult with valid flag, errors list, and warnings list.
    """
    result = ValidationResult()
    limits = load_safety_limits()

    logger.info("Starting mission validation...")

    vehicle_type = mission.get("vehicle_type", "quadcopter")
    if vehicle_type == "swarm" or "agents" in mission or "mode" in mission and mission.get("mode") in ("FORMATION", "INDEPENDENT", "MIXED", "REGROUP"):
        logger.info(f"Using advanced validation path for vehicle type: {vehicle_type}")
        constraints = load_user_constraints()
        
        # Layer 1: Pydantic Schema Validation (if full structure present)
        if "agents" in mission and "mission_id" in mission:
            try:
                SwarmMission.model_validate(mission)
            except Exception as e:
                result.add_error(f"Pydantic Swarm Schema Validation Error: {e}")

        # Swarm-specific parameters validation
        spacing = float(mission.get("spacing_m", 5.0))
        if spacing < 2.0:
            result.add_warning(f"Swarm spacing ({spacing}m) is tight (<2.0m). Ensure collision guardrails are active.")
        if spacing < 1.0:
            result.add_error(f"Swarm spacing ({spacing}m) below minimum safe limit of 1.0m.")

        coll_rad = float(mission.get("collision_radius_m", 1.0))
        if coll_rad < 0.5:
            result.add_error(f"Collision avoidance radius ({coll_rad}m) below minimum 0.5m.")

        actions = mission.get("actions", [])
        if not actions and "leader_mission" in mission and isinstance(mission["leader_mission"], dict):
            actions = mission["leader_mission"].get("actions", [])

        if not actions:
            result.add_error("Mission has no actions")
            return result
        
        # Takeoff check
        if actions[0].get("type") != "takeoff":
            result.add_error("Mission must start with takeoff")

        # Gather all action lists (leader + all independent follower action sequences)
        all_actions_to_check = [("Leader", actions)]
        f_config = mission.get("follower_config", {})
        if isinstance(f_config, dict):
            for fkey, fval in f_config.items():
                if isinstance(fval, dict) and fval.get("independent_actions"):
                    all_actions_to_check.append((fkey.replace("_", " ").title(), fval["independent_actions"]))
        if "agents" in mission and isinstance(mission["agents"], list):
            for ag in mission["agents"]:
                if isinstance(ag, dict) and ag.get("independent_actions"):
                    role_lbl = ag.get("id", "agent").replace("_", " ").title()
                    if not any(role_lbl == existing[0] for existing in all_actions_to_check):
                        all_actions_to_check.append((role_lbl, ag["independent_actions"]))

        for role, act_list in all_actions_to_check:
            if not isinstance(act_list, list):
                continue
            for i, action in enumerate(act_list):
                if not isinstance(action, dict):
                    continue
                params = action.get("params", {})
                alt = params.get("altitude_m")
                if alt is not None:
                    if alt > constraints.max_altitude_m or alt > limits["max_altitude_m"]:
                        result.add_error(f"{role} Action {i}: altitude {alt}m exceeds maximum allowed limit")
                    if alt < constraints.min_altitude_m or alt < limits["min_altitude_m"]:
                        result.add_error(f"{role} Action {i}: altitude {alt}m below minimum allowed limit")
                
                speed = params.get("speed_mps")
                if speed is not None:
                    if speed > constraints.max_speed_mps or speed > limits["max_speed_mps"]:
                        result.add_error(f"{role} Action {i}: speed {speed} m/s exceeds maximum allowed limit")
                    if speed < limits["min_speed_mps"]:
                        result.add_error(f"{role} Action {i}: speed {speed} m/s below minimum allowed limit")

                n = params.get("north_m", 0.0)
                e = params.get("east_m", 0.0)
                if action.get("type") == "sweep":
                    n = max(abs(params.get("corner_a_north_m", 0.0)), abs(params.get("corner_b_north_m", 0.0)))
                    e = max(abs(params.get("corner_a_east_m", 0.0)), abs(params.get("corner_b_east_m", 0.0)))
                dist = math.sqrt(n**2 + e**2)
                total_dist = dist + spacing if role == "Leader" else dist
                if total_dist > limits["geofence_radius_m"]:
                    result.add_error(f"{role} Action {i}: destination + formation buffer exceeds geofence {limits['geofence_radius_m']}m")
    else:
        schema = load_mission_schema()
        # Layer 1: Schema validation
        _validate_schema(mission, schema, result)

        # Layer 2: Safety checks
        _validate_safety(mission, limits, result)

        # Layer 3: Sanity checks
        _validate_sanity(mission, limits, result)

    if result.valid:
        logger.info("Mission validation PASSED — all checks cleared")
    else:
        logger.warning(
            f"Mission validation FAILED — {len(result.errors)} error(s), "
            f"{len(result.warnings)} warning(s)"
        )

    return result

def validate_hardware_safety(mission: dict, telemetry: dict) -> ValidationResult:
    """
    Hardware-specific safety validation layer.
    Ensures the physical drone meets professional safety constraints before flight.
    """
    result = ValidationResult()
    limits = load_safety_limits()
    
    # 1. Battery Check
    bat = telemetry.get("battery_pct", 0)
    if bat < 30:
        result.add_error(f"Hardware Safety Violation: Battery is dangerously low ({bat}%). Minimum required is 30%.")
    elif bat < 45:
        result.add_warning(f"Hardware Safety: Battery is getting low ({bat}%).")

    # 2. GPS Fix Check
    # For this simplified dashboard telemetry we might not have exact satellite counts yet,
    # but we will check if telemetry implies a GPS lock or if the flight controller reported armable
    # In MAVSDK we wait for global position OK, here we warn if latitude/longitude are strictly 0.0
    lat = telemetry.get("latitude", 0.0)
    lon = telemetry.get("longitude", 0.0)
    if abs(lat) < 0.0001 and abs(lon) < 0.0001:
         result.add_error("Hardware Safety Violation: No valid GPS position detected.")

    # 3. Hardware Altitude Cap (typically stricter than sim)
    hw_max_alt = limits.get("hw_max_altitude_m", 30.0)
    
    actions = []
    if "actions" in mission:
        actions = mission.get("actions", [])
    elif "leader_mission" in mission and isinstance(mission["leader_mission"], dict):
        actions = mission["leader_mission"].get("actions", [])

    rtl_present = False
    for i, action in enumerate(actions):
        if not isinstance(action, dict): continue
        if action.get("type") in ("return_to_launch", "rtl"):
            rtl_present = True
            
        params = action.get("params", {})
        alt = params.get("altitude_m")
        if alt is not None and alt > hw_max_alt:
             result.add_error(f"Hardware Safety Violation: Action {i} requests altitude {alt}m which exceeds hardware cap of {hw_max_alt}m.")
             
        speed = params.get("speed_mps")
        if speed is not None and speed > 8.0:
             result.add_warning(f"Hardware Safety Advisory: Action {i} requests speed {speed}m/s. Ensure wind conditions permit this velocity safely.")
             
    if not rtl_present:
        result.add_warning("Hardware Safety Advisory: Mission does not explicitly end with a return-to-launch (RTL) command.")

    return result
