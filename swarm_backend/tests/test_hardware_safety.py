import pytest
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mission_validator import validate_hardware_safety

def get_base_telemetry():
    return {
        "battery_pct": 85.0,
        "latitude": 47.397742,
        "longitude": 8.545594,
    }

def get_base_mission():
    return {
        "actions": [
            {"type": "takeoff", "params": {"altitude_m": 10}},
            {"type": "goto", "params": {"north_m": 10, "east_m": 10, "altitude_m": 10}},
            {"type": "return_to_launch"}
        ]
    }

def test_hardware_safety_healthy():
    mission = get_base_mission()
    telemetry = get_base_telemetry()
    result = validate_hardware_safety(mission, telemetry)
    
    assert result.valid is True
    assert len(result.errors) == 0
    # Expected warning for failsafe advisory
    assert any("failsafe" in w.lower() for w in result.warnings)

def test_hardware_safety_low_battery():
    mission = get_base_mission()
    telemetry = get_base_telemetry()
    telemetry["battery_pct"] = 25.0
    result = validate_hardware_safety(mission, telemetry)
    
    assert result.valid is False
    assert any("dangerously low" in e.lower() for e in result.errors)

def test_hardware_safety_no_gps():
    mission = get_base_mission()
    telemetry = get_base_telemetry()
    telemetry["latitude"] = 0.0
    telemetry["longitude"] = 0.0
    result = validate_hardware_safety(mission, telemetry)
    
    assert result.valid is False
    assert any("no valid gps" in e.lower() for e in result.errors)

def test_hardware_safety_altitude_exceeded():
    mission = get_base_mission()
    mission["actions"][1]["params"]["altitude_m"] = 50.0  # hardware cap is 30m
    telemetry = get_base_telemetry()
    result = validate_hardware_safety(mission, telemetry)
    
    assert result.valid is False
    assert any("exceeds hardware cap" in e.lower() for e in result.errors)

def test_hardware_safety_geofence_exceeded():
    mission = get_base_mission()
    mission["actions"][1]["params"]["north_m"] = 300.0  # geofence cap is 200m
    telemetry = get_base_telemetry()
    result = validate_hardware_safety(mission, telemetry)
    
    assert result.valid is False
    assert any("exceeds geofence" in e.lower() for e in result.errors)

def test_hardware_safety_high_speed():
    mission = get_base_mission()
    mission["actions"][1]["params"]["speed_mps"] = 10.0  # limit is 8m/s for advisory
    telemetry = get_base_telemetry()
    result = validate_hardware_safety(mission, telemetry)
    
    assert result.valid is True
    assert any("speed" in w.lower() for w in result.warnings)

def test_hardware_safety_missing_rtl():
    mission = get_base_mission()
    mission["actions"].pop() # remove rtl
    telemetry = get_base_telemetry()
    result = validate_hardware_safety(mission, telemetry)
    
    assert result.valid is True
    assert any("explicitly end with a return-to-launch" in w.lower() for w in result.warnings)
