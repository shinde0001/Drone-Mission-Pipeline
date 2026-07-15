import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path("/home/parth/om/drone_pipeline")
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_planner import plan_mission
from src.mission_validator import validate_mission

# Define test cases: 10 commands for each mode
test_cases = {
    "standard": [
        "Take off to 10m, fly to 20m north and 10m east, loiter for 15s, then return to launch",
        "Fly the figure eight inspection pattern at 12 metres",
        "Take off to 15m, loiter for 10s, then land",
        "Execute the perimeter loop twice at 15m altitude and speed of 4m/s",
        "Take off to 8m, go east 40m, speed 3m/s, then land",
        "Fly to north 10m and east 20m at 14m altitude, loiter 5s, then return to launch",
        "Arm, take off to 5m, hover 3 seconds, fly north -15m and east -15m, land",
        "Fly a square pattern of 30m size at altitude 10m, speed 5m/s",
        "Linear sweep at 8m altitude with a speed of 3m/s",
        "Take off to 20m, fly east 50m, return to launch"
    ],
    "swarm": [
        "Three drones takeoff to 15m and fly north 40m in wedge formation.",
        "Plan a line formation swarm sweep to east 50m at 12m altitude.",
        "Fly 3 drones in column formation north 30m and east 30m, then land.",
        "Wedge formation takeoff to 10m, loiter 10 seconds, then return to launch.",
        "Three drones in column formation: fly to north 20m at 10m height.",
        "Swarm takeoff in wedge formation to 12m, move east 30m, land.",
        "Fly three drones in line formation to north 45m and east -10m at speed 3m/s.",
        "Squad of 3 drones takeoff to 8m, fly wedge formation to north -20m, then land.",
        "Column formation patrol to east 25m at 15m altitude.",
        "Three drones in wedge: fly north 35m, east 15m, loiter 5s, return home."
    ]
}

results = []
success_count = 0
total_count = 0

print("Running operating mode commands against local LLM pipeline...\n")

for mode, commands in test_cases.items():
    print(f"=== Testing Mode: {mode.upper()} ===")
    for idx, cmd in enumerate(commands, 1):
        total_count += 1
        print(f" [{idx}/10] Prompt: \"{cmd}\"")
        try:
            # Plan mission
            mission = plan_mission(cmd, mode=mode)
            # Validate mission
            val_res = validate_mission(mission)
            
            if val_res.valid:
                success_count += 1
                
            print(f"      Result: Valid={val_res.valid}, Actions={len(mission.get('actions', []))}")
        except Exception as e:
            print(f"      Result: FAILED to plan: {e}")
    print()

print(f"Test Complete: {success_count}/{total_count} missions successfully planned and validated!")
