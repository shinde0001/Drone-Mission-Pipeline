"""
llm_planner.py — Stage 1: LLM Intent Interpretation → Mission JSON

This module takes a natural-language prompt from the operator and uses
OpenAI GPT-4o to interpret the intent, then emits a structured mission
JSON that conforms to the mission schema.

KEY PRINCIPLE: The LLM *proposes* a plan. It never directly controls the
drone. The output goes through validation (Stage 2) before execution.
"""

import json
import os
import re
import time
import threading
import subprocess
import shutil
from pathlib import Path

from openai import OpenAI

from .utils import (
    load_mission_schema,
    load_waypoint_library,
    load_safety_limits,
    print_info,
    print_warning,
    print_error,
    print_success,
    setup_logger,
)

logger = setup_logger("llm_planner")

# ── Ollama Auto-Shutdown Watchdog ───────────────────────────────────────────
_last_ollama_use_time = 0
_ollama_watchdog_thread = None

def _ollama_watchdog():
    """Background thread that kills Ollama if inactive for > 60 seconds."""
    global _last_ollama_use_time
    while True:
        time.sleep(10)
        # Check if Ollama was used, and if 60s has passed since last use
        if _last_ollama_use_time > 0 and (time.time() - _last_ollama_use_time) > 60:
            try:
                # Use pkill to shut down the ollama serve process
                subprocess.run(["pkill", "-f", "ollama serve"], check=False)
                logger.info("Ollama inactive for > 60s. Auto-stopped to save CPU/RAM.")
                print_info("Ollama inactive for > 60s. Auto-stopped to save CPU/RAM.")
            except Exception as e:
                logger.error(f"Failed to auto-stop Ollama: {e}")
            _last_ollama_use_time = 0  # Reset so we don't spam kills

def _touch_ollama():
    """Update the last used timestamp and start the watchdog if needed."""
    global _last_ollama_use_time, _ollama_watchdog_thread
    _last_ollama_use_time = time.time()
    
    if _ollama_watchdog_thread is None or not _ollama_watchdog_thread.is_alive():
        _ollama_watchdog_thread = threading.Thread(target=_ollama_watchdog, daemon=True)
        _ollama_watchdog_thread.start()
        logger.info("Started Ollama inactivity watchdog thread.")

# ── System prompt template (full — for capable models like Gemini) ──────
SYSTEM_PROMPT = """You are a drone mission planner AI. Your job is to interpret 
natural-language commands from a human operator and produce a structured mission 
plan in JSON format.

## Your Role
- You PROPOSE a flight plan. You do NOT fly the drone.
- Your output will be validated against safety rules before execution.
- Be conservative: prefer lower altitudes and slower speeds when unspecified.
- Think geometrically: if the user asks to fly a letter shape, a pattern, or a path,
  calculate the waypoints (north_m, east_m) that trace that shape in the NED coordinate plane.

## Available Action Types
- "takeoff": Take off to a specified altitude. Params: altitude_m
- "goto": Fly to a waypoint. Params: north_m, east_m, altitude_m, speed_mps
- "loiter": Hover at current position. Params: duration_s
- "land": Land at current position. Params: (none needed)
- "return_to_launch": Return to the takeoff point and land. Params: (none needed)

## Coordinate System
All positions use NED (North-East-Down) offsets from the home/launch position:
- north_m: positive = north (up on map), negative = south (down on map)
- east_m: positive = east (right on map), negative = west (left on map)
- altitude_m: height above ground in metres (always positive)

## Safety Limits
{safety_limits}

## Mission Structure Rules
1. Every mission MUST start with a "takeoff" action
2. Every mission MUST end with either "land" or "return_to_launch"
3. Include at least one "goto" or "loiter" action between takeoff and landing
4. If the operator says "repeat N times" or "loop N times", set repeat_count=N
5. Default altitude: 10m if not specified
6. Default speed: 5 m/s if not specified
7. For shape/letter/pattern commands: calculate specific (north_m, east_m) waypoints that draw the shape

## Output Format
Return ONLY a valid JSON object. Do NOT include any text, explanation, or markdown fences.
"""

SYSTEM_PROMPT_SWARM = """You are a drone swarm mission planner AI. Your job is to interpret 
natural-language commands for a squad of {num_drones} drones.
You must produce a squad-level mission JSON.

Return ONLY a valid JSON object matching this structure:
{{
  "mission_name": "<descriptive name>",
  "vehicle_type": "swarm",
  "formation": "<wedge, line, or column>",
  "drone_count": {num_drones},
  "repeat_count": <repeat count>,
  "actions": [
    {{"type": "takeoff", "params": {{"altitude_m": <altitude>}}}},
    {{"type": "goto", "params": {{"north_m": <n>, "east_m": <e>, "altitude_m": <a>, "speed_mps": <s>}}}},
    {{"type": "land", "params": {{}}}}
  ]
}}
Do NOT include any text before or after the JSON. Output ONLY the JSON object.
"""

SYSTEM_PROMPT_VISION = """You are a vision-tracking drone planner AI. Your job is to interpret 
commands to find and track a specific target.
You must produce a mission JSON specifying the target class and search coordinates.

Return ONLY a valid JSON object matching this structure:
{{
  "mission_name": "<descriptive name>",
  "vehicle_type": "vision_drone",
  "target_class": "<object to follow, e.g. red_box, person, car>",
  "actions": [
    {{"type": "takeoff", "params": {{"altitude_m": <altitude>}}}},
    {{"type": "goto", "params": {{"north_m": <n>, "east_m": <e>, "altitude_m": <a>, "speed_mps": <s>}}}},
    {{"type": "search_and_follow", "params": {{"target": "<target_class>", "timeout_s": <seconds>}}}},
    {{"type": "return_to_launch", "params": {{}}}}
  ]
}}
Do NOT include any text before or after the JSON. Output ONLY the JSON object.
"""

SYSTEM_PROMPT_SLAM = """You are an autonomous SLAM exploration drone planner AI. Your job is to interpret 
commands to explore and map an unknown region.
You must produce a mission JSON specifying the boundary limits of the exploration.

Return ONLY a valid JSON object matching this structure:
{{
  "mission_name": "<descriptive name>",
  "vehicle_type": "slam_drone",
  "explore_area": {{"min_n": <min_n>, "max_n": <max_n>, "min_e": <min_e>, "max_e": <max_e>}},
  "actions": [
    {{"type": "takeoff", "params": {{"altitude_m": <altitude>}}}},
    {{"type": "explore", "params": {{"duration_s": <seconds>}}}},
    {{"type": "return_to_launch", "params": {{}}}}
  ]
}}
Do NOT include any text before or after the JSON. Output ONLY the JSON object.
"""

# ── Compact system prompt (for small local models like llama3.2:1b) ────
SYSTEM_PROMPT_COMPACT = """You are a drone mission planner. Convert the user command into a JSON mission plan.

Action types: takeoff (altitude_m), goto (north_m, east_m, altitude_m, speed_mps), loiter (duration_s), land (), return_to_launch ().
Coordinates: NED offsets in metres from home. north_m=forward, east_m=right. altitude_m is always positive.

Limits: max altitude {max_alt}m, max speed {max_speed}m/s, geofence {geofence}m.

Rules:
1. Start with takeoff, end with land or return_to_launch. Default altitude=10m, speed=5m/s.
2. For shapes/letters/patterns: calculate (north_m, east_m) waypoints that trace the shape.
3. Return ONLY valid JSON. No text, no explanation.
"""

# ══════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES — These teach the LLM the exact input→output format
# Each example is a (user_prompt, assistant_json_response) pair.
# The LLM sees these as prior conversation turns before the real prompt.
# ══════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES_FULL = [
    # Example 1: Letter shape — demonstrates geometric waypoint calculation
    {
        "user": "fly and patrol by path of letter P at 12 metres",
        "assistant": json.dumps({
            "mission_name": "Letter P Flight Path",
            "vehicle_type": "quadcopter",
            "repeat_count": 1,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 12}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 0, "altitude_m": 12, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 20, "east_m": 0, "altitude_m": 12, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 20, "east_m": 10, "altitude_m": 12, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 10, "east_m": 10, "altitude_m": 12, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 10, "east_m": 0, "altitude_m": 12, "speed_mps": 5}},
                {"type": "return_to_launch", "params": {}}
            ]
        }, indent=2)
    },
    # Example 2: Square pattern with repeat — demonstrates multi-loop
    {
        "user": "fly to 15 metres, do a square patrol of 30m size 2 times, then return to launch",
        "assistant": json.dumps({
            "mission_name": "Square Patrol 30m",
            "vehicle_type": "quadcopter",
            "repeat_count": 2,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 15}},
                {"type": "goto", "params": {"north_m": 30, "east_m": 0, "altitude_m": 15, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 30, "east_m": 30, "altitude_m": 15, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 30, "altitude_m": 15, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 0, "altitude_m": 15, "speed_mps": 5}},
                {"type": "return_to_launch", "params": {}}
            ]
        }, indent=2)
    },
    # Example 3: Simple goto + loiter — demonstrates hover/wait
    {
        "user": "take off to 8m, go north 20m and east 10m, hover for 10 seconds, then land",
        "assistant": json.dumps({
            "mission_name": "Goto And Hover",
            "vehicle_type": "quadcopter",
            "repeat_count": 1,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 8}},
                {"type": "goto", "params": {"north_m": 20, "east_m": 10, "altitude_m": 8, "speed_mps": 5}},
                {"type": "loiter", "params": {"duration_s": 10}},
                {"type": "land", "params": {}}
            ]
        }, indent=2)
    },
]

# Compact examples for small local models (shorter to save context window)
FEW_SHOT_EXAMPLES_COMPACT = [
    # Example 1: Letter shape
    {
        "user": "fly path of letter L at 10m",
        "assistant": json.dumps({
            "mission_name": "Letter L Path",
            "vehicle_type": "quadcopter",
            "repeat_count": 1,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 10}},
                {"type": "goto", "params": {"north_m": 20, "east_m": 0, "altitude_m": 10, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 0, "altitude_m": 10, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 15, "altitude_m": 10, "speed_mps": 5}},
                {"type": "return_to_launch", "params": {}}
            ]
        })
    },
    # Example 2: Simple movement
    {
        "user": "takeoff 8m, fly north 15m east 10m, hover 5 seconds, land",
        "assistant": json.dumps({
            "mission_name": "Goto Hover Land",
            "vehicle_type": "quadcopter",
            "repeat_count": 1,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 8}},
                {"type": "goto", "params": {"north_m": 15, "east_m": 10, "altitude_m": 8, "speed_mps": 5}},
                {"type": "loiter", "params": {"duration_s": 5}},
                {"type": "land", "params": {}}
            ]
        })
    },
]

FEW_SHOT_EXAMPLES_VISION = [
    {
        "user": "takeoff to 10m, search for the target person at north 15m east 15m and follow them for 30s, then return home",
        "assistant": json.dumps({
            "mission_name": "Person Follow Mission",
            "vehicle_type": "vision_drone",
            "target_class": "person",
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 10}},
                {"type": "goto", "params": {"north_m": 15, "east_m": 15, "altitude_m": 10, "speed_mps": 5}},
                {"type": "search_and_follow", "params": {"target": "person", "timeout_s": 30}},
                {"type": "return_to_launch", "params": {}}
            ]
        })
    }
]

FEW_SHOT_EXAMPLES_SLAM = [
    {
        "user": "takeoff to 8m, autonomously explore the area between north 0 to 40m and east -20m to 20m for 120 seconds, then return",
        "assistant": json.dumps({
            "mission_name": "Autonomous Area Exploration",
            "vehicle_type": "slam_drone",
            "explore_area": {"min_n": 0, "max_n": 40, "min_e": -20, "max_e": 20},
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 8}},
                {"type": "explore", "params": {"duration_s": 120}},
                {"type": "return_to_launch", "params": {}}
            ]
        })
    }
]


def _build_system_prompt(compact: bool = False, mode: str = "standard", num_drones: int = 3) -> str:
    """Build the full system prompt with current config data.
    
    Args:
        compact: If True, use the shorter prompt suitable for small
                 local models with limited context windows.
        mode: Operating mode (standard, swarm, slam, vision)
        num_drones: Number of drones in swarm mode
    """
    if mode == "swarm":
        return SYSTEM_PROMPT_SWARM.format(num_drones=num_drones)
    elif mode == "vision":
        return SYSTEM_PROMPT_VISION
    elif mode == "slam":
        return SYSTEM_PROMPT_SLAM

    safety = load_safety_limits()

    if compact:
        return SYSTEM_PROMPT_COMPACT.format(
            max_alt=safety['max_altitude_m'],
            max_speed=safety['max_speed_mps'],
            geofence=safety['geofence_radius_m'],
        )

    # Format safety limits for the prompt
    safety_lines = [
        f"- Maximum altitude: {safety['max_altitude_m']}m",
        f"- Maximum speed: {safety['max_speed_mps']} m/s",
        f"- Geofence radius: {safety['geofence_radius_m']}m from home",
        f"- Maximum loiter duration: {safety['max_loiter_duration_s']}s",
        f"- Maximum repeat count: {safety['max_repeat_count']}",
        f"- Maximum actions per mission: {safety['max_actions_per_mission']}",
    ]

    return SYSTEM_PROMPT.format(
        safety_limits="\n".join(safety_lines),
    )


def _build_few_shot_messages(system_prompt: str, user_prompt: str, compact: bool = False, mode: str = "standard", num_drones: int = 3) -> list:
    """Build the full message list with few-shot examples for the LLM.
    
    Structure sent to the LLM:
      1. System message (rules + format)
      2. Example user prompt #1
      3. Example assistant response #1 (correct JSON)
      4. Example user prompt #2
      5. Example assistant response #2 (correct JSON)
      ...
      N. Actual user prompt (the real command)
    
    This teaches the LLM by showing it concrete input→output pairs
    before it processes the real user command.
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    if mode == "swarm":
        examples = [
            {
                "user": f"{num_drones} drones fly in a wedge formation to north 20m, east 10m at altitude 15m, then land",
                "assistant": json.dumps({
                    "mission_name": f"{num_drones} Drone Wedge Formation",
                    "vehicle_type": "swarm",
                    "formation": "wedge",
                    "drone_count": num_drones,
                    "repeat_count": 1,
                    "actions": [
                        {"type": "takeoff", "params": {"altitude_m": 15}},
                        {"type": "goto", "params": {"north_m": 20, "east_m": 10, "altitude_m": 15, "speed_mps": 5}},
                        {"type": "land", "params": {}}
                    ]
                })
            }
        ]
    elif mode == "vision":
        examples = FEW_SHOT_EXAMPLES_VISION
    elif mode == "slam":
        examples = FEW_SHOT_EXAMPLES_SLAM
    else:
        examples = FEW_SHOT_EXAMPLES_COMPACT if compact else FEW_SHOT_EXAMPLES_FULL
    
    for example in examples:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    
    # Finally, add the actual user prompt
    messages.append({"role": "user", "content": user_prompt})
    
    return messages


def plan_mission(prompt: str, mode: str = "standard", ai_engine: str = "offline", api_key: str = None, num_drones: int = 3) -> dict:
    """
    Stage 1: Interpret a natural-language command and produce mission JSON.

    Args:
        prompt: Natural-language instruction from the operator.
        mode: The operating mode (standard, swarm, slam, vision)
        ai_engine: 'offline' or 'online'
        api_key: Optional API key for online engines.
        num_drones: Number of drones in swarm mode

    Returns:
        dict: Raw mission JSON (not yet validated).

    Raises:
        ValueError: If the LLM response cannot be parsed as JSON.
        RuntimeError: If the Ollama API call fails or is unavailable.
    """
    # Pre-process the prompt to help the small model map directions to NED coordinates
    # Small models often get confused and put forward distances into altitude_m
    import re
    p_lower = prompt.lower()
    p_lower = re.sub(r'\b(?:forward|front)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)?\b', r'north_m \1', p_lower)
    p_lower = re.sub(r'\b(?:right)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)?\b', r'east_m \1', p_lower)
    p_lower = re.sub(r'\b(?:back|backward|reverse)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)?\b', r'north_m -\1', p_lower)
    p_lower = re.sub(r'\b(?:left)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)?\b', r'east_m -\1', p_lower)
    # Fix common typos and grammar issues that confuse the small model
    p_lower = p_lower.replace("tlenght", "length")
    p_lower = p_lower.replace("lenght", "length")
    prompt = p_lower
    
    if ai_engine == "offline":
        # Ensure Ollama is running first (attempt to auto-start if not)
        import urllib.request
        import subprocess
        import time
        import shutil

        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

        def is_ollama_running():
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
                return True
            except Exception:
                return False

        def is_ollama_model_available(model):
            try:
                req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
                data = json.loads(req.read().decode())
                for m in data.get("models", []):
                    if m.get("name") == model or m.get("name") == model + ":latest":
                        return True
                return False
            except Exception:
                return False

        if not is_ollama_running():
            print_warning("Ollama is not running. Attempting to start it automatically...")
            logger.warning("Ollama is not running. Attempting to start it automatically...")
            try:
                ollama_path = shutil.which("ollama")
                if ollama_path:
                    subprocess.Popen([ollama_path, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for i in range(10):
                        time.sleep(1.0)
                        if is_ollama_running():
                            print_success("Ollama started successfully.")
                            logger.info("Ollama started successfully.")
                            break
                else:
                    logger.error("ollama command not found in PATH")
            except Exception as e:
                logger.error(f"Failed to start Ollama automatically: {e}")

        if not is_ollama_running():
            raise RuntimeError(
                "Ollama server is not running or unreachable on http://127.0.0.1:11434. "
                "Please start it manually using 'ollama serve'."
            )

        if not is_ollama_model_available(model_name):
            print_warning(f"Ollama model '{model_name}' not found locally. Pulling it now... This may take a while.")
            logger.warning(f"Ollama model '{model_name}' not found. Pulling...")
            try:
                ollama_path = shutil.which("ollama")
                subprocess.run([ollama_path, "pull", model_name], check=True)
                print_success(f"Model '{model_name}' pulled successfully.")
                logger.info(f"Model '{model_name}' pulled successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to pull Ollama model '{model_name}': {e}")

    if ai_engine == "online":
        print_info("Using online LLM via Gemini 3.5 Flash.")
        logger.info("Using online LLM via Gemini 3.5 Flash.")
        import google.generativeai as genai
        # Prefer the explicitly passed api_key, fallback to environment variable
        actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not actual_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Please provide it in the UI or set the environment variable.")
        genai.configure(api_key=actual_api_key)
        model_name = "gemini-3.5-flash"
        is_local = False
    else:
        print_info("Using local LLM via Ollama.")
        logger.info("Using local LLM via Ollama.")
        client = OpenAI(
            base_url='http://127.0.0.1:11434/v1',
            api_key='ollama',
        )
        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
        is_local = True

    # Use compact prompt for small local models to fit in limited context
    system_prompt = _build_system_prompt(compact=is_local, mode=mode, num_drones=num_drones)
    
    # Build few-shot messages (system + example pairs + real user prompt)
    # This teaches the LLM by showing concrete input→output examples
    few_shot_messages = _build_few_shot_messages(system_prompt, prompt, compact=is_local, mode=mode, num_drones=num_drones)
    
    print_info(f"Sending prompt to {model_name}: \"{prompt}\"")
    logger.info(f"Calling {model_name} for mission planning (with {len(few_shot_messages)//2} few-shot examples)...")

    if is_local:
        _touch_ollama()

    # Retry logic — small local models may need a few attempts
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print_warning(f"Retry attempt {attempt}/{max_attempts}...")
            logger.warning(f"Retry attempt {attempt}/{max_attempts}")

        try:
            if ai_engine == "online":
                # For Gemini: convert few-shot messages into multi-turn conversation
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    )
                )
                # Build Gemini conversation history from few-shot examples
                if mode == "swarm":
                    examples = [
                        {
                            "user": f"{num_drones} drones fly in a wedge formation to north 20m, east 10m at altitude 15m, then land",
                            "assistant": json.dumps({
                                "mission_name": f"{num_drones} Drone Wedge Formation",
                                "vehicle_type": "swarm",
                                "formation": "wedge",
                                "drone_count": num_drones,
                                "repeat_count": 1,
                                "actions": [
                                    {"type": "takeoff", "params": {"altitude_m": 15}},
                                    {"type": "goto", "params": {"north_m": 20, "east_m": 10, "altitude_m": 15, "speed_mps": 5}},
                                    {"type": "land", "params": {}}
                                ]
                            })
                        }
                    ]
                elif mode == "vision":
                    examples = FEW_SHOT_EXAMPLES_VISION
                elif mode == "slam":
                    examples = FEW_SHOT_EXAMPLES_SLAM
                else:
                    examples = FEW_SHOT_EXAMPLES_FULL
                
                gemini_history = []
                for ex in examples:
                    gemini_history.append({"role": "user", "parts": [ex["user"]]})
                    gemini_history.append({"role": "model", "parts": [ex["assistant"]]})
                
                chat = model.start_chat(history=gemini_history)
                response = chat.send_message(prompt)
                raw_content = response.text
            else:
                # For Ollama: pass the full few-shot message list directly
                api_kwargs = dict(
                    model=model_name,
                    messages=few_shot_messages,
                    temperature=0.1,  # Low temp for deterministic structured output
                    max_tokens=2000,
                    extra_body={"num_ctx": 4096},  # Increased context for few-shot examples
                    response_format={"type": "json_object"}
                )
                response = client.chat.completions.create(**api_kwargs)
                raw_content = response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt == max_attempts:
                raise RuntimeError(f"{model_name} API call failed: {e}") from e
            continue

        logger.info(f"Received response from {model_name}")
        logger.debug(f"Raw response: {raw_content[:500]}")
        
        if is_local:
            _touch_ollama() # Touch again after completion so it has a full 60s after generation ends

        # Parse the JSON response — for local models, extract JSON from
        # potentially noisy output (small models often add explanatory text)
        mission = _extract_mission_json(raw_content, is_local, mode=mode)
        if mission is not None:
            logger.info(
                f"Mission planned: '{mission.get('mission_name', 'unnamed')}' "
                f"with {len(mission['actions'])} actions"
            )
            return mission

        last_error = ValueError(
            f"Could not extract valid mission JSON from Ollama response "
            f"(attempt {attempt}): {raw_content[:300]}"
        )
        print_warning(f"Attempt {attempt}: Ollama did not return valid mission JSON.")
        print_info(f"Ollama raw response was:\n{raw_content}")

    raise last_error


def _extract_mission_json(raw_content: str, is_local: bool, mode: str = "standard") -> dict | None:
    """
    Extract a valid mission JSON dict from LLM output.

    Small local models often wrap JSON in explanatory text, markdown
    code fences, or produce minor formatting issues.  This function
    tries multiple strategies to pull out usable JSON.

    Returns the parsed dict if successful, or None if no valid mission
    JSON could be extracted.
    """
    if not raw_content or not raw_content.strip():
        return None

    strategies = []

    # Strategy 1: Direct parse (works for well-behaved models)
    strategies.append(raw_content.strip())

    # Strategy 2: Extract from markdown code fence ```json ... ```
    fence_match = re.search(
        r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw_content, re.DOTALL
    )
    if fence_match:
        strategies.append(fence_match.group(1).strip())

    # Strategy 3: Find the outermost { ... } block
    brace_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    if brace_match:
        strategies.append(brace_match.group(0).strip())

    # Strategy 4: Fallback for missing opening brace
    missing_brace_match = re.search(r'"mission_name"\s*:', raw_content)
    if missing_brace_match:
        start_idx = missing_brace_match.start()
        fixed_json = "{\n" + raw_content[start_idx:]
        if not fixed_json.strip().endswith("}"):
            fixed_json = fixed_json.rstrip() + "\n}"
        strategies.append(fixed_json)

    for candidate in strategies:
        try:
            # Clean single-line and multi-line comments
            cleaned = re.sub(r'//.*', '', candidate)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            # Clean trailing commas inside lists or objects
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(parsed, dict):
            continue

        if mode != "standard":
            # For advanced modes, if it has actions, return it without standard cleaning
            if "actions" in parsed and isinstance(parsed["actions"], list):
                return parsed

        # Must have an 'actions' list to be a valid mission
        if "actions" in parsed and isinstance(parsed["actions"], list):
            # Fill in defaults if the small model omitted them
            parsed.setdefault("mission_name", "llm_planned_mission")
            parsed.setdefault("vehicle_type", "quadcopter")
            parsed.setdefault("repeat_count", 1)

            # Post-process actions — fix common small-model mistakes

            # ── Synonym mapping for action types ──
            # Small models often invent action names; map them to valid ones
            ACTION_SYNONYMS = {
                # takeoff synonyms
                "up": "takeoff", "ascend": "takeoff", "launch": "takeoff",
                "take_off": "takeoff", "liftoff": "takeoff", "lift_off": "takeoff",
                "rise": "takeoff",
                # loiter / hover / wait synonyms
                "hover": "loiter", "wait": "loiter", "hold": "loiter",
                "stay": "loiter", "pause": "loiter", "delay": "loiter",
                "stop": "loiter", "idle": "loiter", "remain": "loiter",
                # goto / move / fly synonyms
                "forward": "goto", "move": "goto", "fly": "goto",
                "go": "goto", "travel": "goto", "navigate": "goto",
                "waypoint": "goto", "move_to": "goto", "fly_to": "goto",
                "go_to": "goto", "advance": "goto",
                # return_to_launch synonyms
                "return_to_home": "return_to_launch", "rtl": "return_to_launch",
                "return_home": "return_to_launch", "home": "return_to_launch",
                "rth": "return_to_launch", "go_home": "return_to_launch",
                "return": "return_to_launch",
                # land synonyms
                "descend": "land", "touch_down": "land", "touchdown": "land",
                "down": "land",
            }
            VALID_ACTIONS = {"takeoff", "goto", "loiter", "land", "return_to_launch"}

            for action in parsed["actions"]:
                # Remap action type via synonyms
                raw_type = action.get("type", "").lower().strip()
                if raw_type in ACTION_SYNONYMS:
                    action["type"] = ACTION_SYNONYMS[raw_type]

                # Ensure every action has a 'params' dict
                if "params" not in action:
                    action["params"] = {}

                # Move any top-level keys (other than type/params) into params
                extra_keys = [k for k in action if k not in ("type", "params")]
                for k in extra_keys:
                    if k not in action["params"]:
                        action["params"][k] = action.pop(k)
                    else:
                        action.pop(k)

                # ── Per-action param normalization ──
                atype = action["type"]
                params = action["params"]

                if atype == "loiter":
                    # Normalize duration keys: sec, seconds, time, time_s → duration_s
                    for alias in ("sec", "seconds", "time", "time_s", "wait_s"):
                        if alias in params and "duration_s" not in params:
                            params["duration_s"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    params.setdefault("duration_s", 5)
                    # Schema minimum is 1s; clamp invalid values
                    if params["duration_s"] < 1:
                        params["duration_s"] = 5

                elif atype == "goto":
                    # Normalize distance-only params → north_m
                    for alias in ("m", "distance", "distance_m", "forward_m",
                                  "front_m", "forward", "dist"):
                        if alias in params and "north_m" not in params:
                            params["north_m"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    # Normalize other direction aliases
                    for alias in ("right_m", "lateral_m"):
                        if alias in params and "east_m" not in params:
                            params["east_m"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    for alias in ("height_m", "alt_m", "alt", "height"):
                        if alias in params and "altitude_m" not in params:
                            params["altitude_m"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    for alias in ("speed", "velocity", "speed_m_s"):
                        if alias in params and "speed_mps" not in params:
                            params["speed_mps"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    # Ensure required goto params exist
                    params.setdefault("north_m", 0)
                    params.setdefault("east_m", 0)
                    params.setdefault("altitude_m", 10)
                    params.setdefault("speed_mps", 5)

                elif atype == "takeoff":
                    for alias in ("height_m", "alt_m", "alt", "height"):
                        if alias in params and "altitude_m" not in params:
                            params["altitude_m"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    params.setdefault("altitude_m", 10)

            # ── Final sanitization: strip unexpected params ──
            # The schema only allows specific params per action type.
            # Remove anything else so validation never fails on extra keys.
            ALLOWED_PARAMS = {
                "takeoff":          {"altitude_m"},
                "goto":             {"north_m", "east_m", "altitude_m", "speed_mps"},
                "loiter":           {"duration_s"},
                "land":             set(),
                "return_to_launch": set(),
            }
            for action in parsed["actions"]:
                allowed = ALLOWED_PARAMS.get(action["type"], set())
                unexpected = [k for k in action["params"] if k not in allowed]
                for k in unexpected:
                    action["params"].pop(k)

            # Remove duplicate consecutive takeoffs (model sometimes adds one
            # on top of our auto-inserted takeoff)
            cleaned = []
            for i, action in enumerate(parsed["actions"]):
                if action["type"] == "takeoff" and i > 0 and cleaned and cleaned[-1]["type"] == "takeoff":
                    # Keep the one with the higher altitude
                    if action["params"].get("altitude_m", 0) > cleaned[-1]["params"].get("altitude_m", 0):
                        cleaned[-1] = action
                    continue
                # Skip unknown action types that we couldn't remap
                if action["type"] not in VALID_ACTIONS:
                    continue
                cleaned.append(action)
            parsed["actions"] = cleaned

            # If all actions were stripped, this candidate is unusable
            if not parsed["actions"]:
                continue

            # Ensure the mission starts with takeoff
            if parsed["actions"][0].get("type") != "takeoff":
                takeoff_alt = 10
                for action in parsed["actions"]:
                    if "altitude_m" in action.get("params", {}):
                        takeoff_alt = action["params"]["altitude_m"]
                        break
                parsed["actions"].insert(0, {"type": "takeoff", "params": {"altitude_m": takeoff_alt}})

            # Remove any extra land/RTL commands at the end so there is only one
            while len(parsed["actions"]) > 1 and parsed["actions"][-1].get("type") in ("land", "return_to_launch") and parsed["actions"][-2].get("type") in ("land", "return_to_launch"):
                parsed["actions"].pop(-2)
                
            # Ensure the mission ends with land or RTL
            if parsed["actions"][-1].get("type") not in ("land", "return_to_launch"):
                parsed["actions"].append({"type": "land", "params": {}})

            return parsed

    return None
