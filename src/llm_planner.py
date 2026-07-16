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

SWARM_PROMPT_CLEANUP = """You are a drone command preprocessor and state manager. Do two things:
1. Fix spelling, grammar, and ambiguity in the user's command. Make it clear and concise.
2. Classify the intent into one of three operational modes:
   - FORMATION: Drones fly together in a coordinated shape (wedge, line, column, V-shape) following a leader route.
   - INDEPENDENT: Drones split up or take on independent tasks/routes (e.g. Alpha checks warehouse, Bravo/Charlie sweep perimeter).
   - REGROUP: Drones rally or converge to a single location from different positions.

Return ONLY a valid JSON object:
{"cleaned_prompt": "<clear natural language command>", "mode": "FORMATION|INDEPENDENT|REGROUP"}
Do NOT include any text or markdown before or after the JSON.
"""

SWARM_PROMPT_TASK_SPLIT = """You are a drone swarm commander specializing in Spatial Reasoning and Task Allocation for 3 drones: Leader (Alpha), Follower 1 (Bravo), Follower 2 (Charlie).

Based on the cleaned prompt and operational mode, output structured instructions:
- If FORMATION: Translate abstract shapes (wedge, line, column) into relative coordinate offsets for followers. Default spacing is 5m.
  Wedge = Follower 1 [-spacing_m, -spacing_m], Follower 2 [-spacing_m, +spacing_m]
  Line = Follower 1 [0, -spacing_m], Follower 2 [0, +spacing_m]
  Column = Follower 1 [-spacing_m, 0], Follower 2 [-2*spacing_m, 0]
- If INDEPENDENT: Assign specific tasks/routes to Leader, Follower 1, and Follower 2.
- If REGROUP: Define the rally coordinate offset and the formation to adopt once gathered.

Return ONLY a valid JSON object matching this structure:
{
  "leader_task": "<clear instruction for leader drone navigation path or actions>",
  "follower_task": "<clear instruction summarizing what followers should do>",
  "formation": "wedge|line|column",
  "spacing_m": 5,
  "mode": "FORMATION|INDEPENDENT|REGROUP",
  "follower_1_offsets": [-5, -5],
  "follower_2_offsets": [-5, 5],
  "independent_tasks": null
}
Do NOT include any text or markdown before or after the JSON.
"""

SWARM_PROMPT_FOLLOWER_CONFIG = """You are a follower drone configuration generator for a 3-drone squad.
The leader drone will execute: {leader_summary}
The follower task instructions are: {follower_task}

Generate the exact configuration for Follower 1 and Follower 2.
Followers maintain formation relative to the leader unless assigned independent actions.

Return ONLY a valid JSON object matching this structure:
{{
  "behavior": "maintain_formation|independent|regroup",
  "formation": "{formation}",
  "spacing_m": {spacing_m},
  "collision_radius_m": 1.0,
  "altitude_separation_m": 1.0,
  "follower_1": {{
    "role": "wingman_left",
    "offset_north_m": {f1_n},
    "offset_east_m": {f1_e},
    "independent_actions": null
  }},
  "follower_2": {{
    "role": "wingman_right",
    "offset_north_m": {f2_n},
    "offset_east_m": {f2_e},
    "independent_actions": null
  }}
}}
Do NOT include any text or markdown before or after the JSON.
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

def _build_system_prompt(compact: bool = False, mode: str = "standard", num_drones: int = 3) -> str:
    """Build the full system prompt with current config data.
    
    Args:
        compact: If True, use the shorter prompt suitable for small
                 local models with limited context windows.
        mode: Operating mode (standard, swarm)
        num_drones: Number of drones in swarm mode
    """
    if mode == "swarm":
        return SYSTEM_PROMPT_SWARM.format(num_drones=num_drones)

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
        mode: The operating mode (standard, swarm)
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
    
    if mode == "swarm":
        return plan_swarm_mission(prompt, ai_engine=ai_engine, api_key=api_key, num_drones=num_drones)

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

    if ai_engine.startswith("online"):
        if ai_engine == "online_lite":
            print_info("Using online LLM via Gemini 3.1 Flash Lite.")
            logger.info("Using online LLM via Gemini 3.1 Flash Lite.")
            model_name = "gemini-3.1-flash-lite"
        elif ai_engine == "online_pro":
            print_info("Using online LLM via Gemini 2.5 Pro.")
            logger.info("Using online LLM via Gemini 2.5 Pro.")
            model_name = "gemini-2.5-pro"
        else:
            print_info("Using online LLM via Gemini 3.5 Flash.")
            logger.info("Using online LLM via Gemini 3.5 Flash.")
            model_name = "gemini-3.5-flash"
            
        from google import genai
        from google.genai import types
        # Prefer the explicitly passed api_key, fallback to environment variable
        actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not actual_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Please provide it in the UI or set the environment variable.")
        gemini_client = genai.Client(api_key=actual_api_key)
        is_local = False
    else:
        print_info("Using local LLM via Ollama.")
        logger.info("Using local LLM via Ollama.")
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
            if ai_engine.startswith("online"):
                # For Gemini: convert few-shot messages into multi-turn conversation
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
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
                else:
                    examples = FEW_SHOT_EXAMPLES_FULL
                
                gemini_history = []
                for ex in examples:
                    gemini_history.append({"role": "user", "parts": [{"text": ex["user"]}]})
                    gemini_history.append({"role": "model", "parts": [{"text": ex["assistant"]}]})
                
                chat = gemini_client.chats.create(model=model_name, config=config, history=gemini_history)
                response = chat.send_message(prompt)
                raw_content = response.text
            else:
                # For Ollama: query directly via HTTP POST
                import urllib.request
                payload = {
                    "model": model_name,
                    "messages": few_shot_messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096
                    },
                    "format": "json"
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw_content = resp_data["message"]["content"]
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


def _setup_llm_client(ai_engine: str, api_key: str = None):
    """Setup and return (client, gemini_client, model_name, is_local)."""
    import urllib.request
    import subprocess
    import time
    import shutil

    if ai_engine == "offline":
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
            logger.warning("Ollama is not running. Attempting to start it automatically...")
            try:
                ollama_path = shutil.which("ollama")
                if ollama_path:
                    subprocess.Popen([ollama_path, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for _ in range(10):
                        time.sleep(1.0)
                        if is_ollama_running():
                            logger.info("Ollama started successfully.")
                            break
            except Exception as e:
                logger.error(f"Failed to start Ollama automatically: {e}")

        if not is_ollama_running():
            raise RuntimeError(
                "Ollama server is not running or unreachable on http://127.0.0.1:11434. "
                "Please start it manually using 'ollama serve'."
            )

        if not is_ollama_model_available(model_name):
            logger.warning(f"Ollama model '{model_name}' not found locally. Pulling it now...")
            try:
                ollama_path = shutil.which("ollama")
                subprocess.run([ollama_path, "pull", model_name], check=True)
                logger.info(f"Model '{model_name}' pulled successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to pull Ollama model '{model_name}': {e}")

        return None, None, model_name, True

    elif ai_engine.startswith("online"):
        if ai_engine == "online_lite":
            model_name = "gemini-3.1-flash-lite"
        elif ai_engine == "online_pro":
            model_name = "gemini-2.5-pro"
        else:
            model_name = "gemini-3.5-flash"

        from google import genai
        from google.genai import types
        actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not actual_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Please provide it in the UI or set the environment variable.")
        gemini_client = genai.Client(api_key=actual_api_key)
        return None, gemini_client, model_name, False
    else:
        raise ValueError(f"Unknown ai_engine: {ai_engine}")


def _extract_json_dict(raw_content: str) -> dict | None:
    """Extract a general JSON dict from raw string."""
    if not raw_content or not raw_content.strip():
        return None

    strategies = [
        raw_content.strip(),
    ]
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw_content, re.DOTALL)
    if fence_match:
        strategies.append(fence_match.group(1).strip())

    brace_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    if brace_match:
        strategies.append(brace_match.group(0).strip())

    for candidate in strategies:
        try:
            cleaned = re.sub(r'//.*', '', candidate)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _call_llm_json(system_prompt: str, user_prompt: str, few_shot_examples: list, ai_engine: str, model_name: str, client, gemini_client, is_local: bool) -> dict | None:
    """Executes a single LLM call with retries and extracts a JSON dictionary."""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if ai_engine.startswith("online"):
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
                )
                gemini_history = []
                for ex in few_shot_examples:
                    gemini_history.append({"role": "user", "parts": [{"text": ex["user"]}]})
                    gemini_history.append({"role": "model", "parts": [{"text": ex["assistant"]}]})
                
                chat = gemini_client.chats.create(model=model_name, config=config, history=gemini_history)
                response = chat.send_message(user_prompt)
                raw_content = response.text
            else:
                messages = [{"role": "system", "content": system_prompt}]
                for ex in few_shot_examples:
                    messages.append({"role": "user", "content": ex["user"]})
                    messages.append({"role": "assistant", "content": ex["assistant"]})
                messages.append({"role": "user", "content": user_prompt})

                if is_local:
                    _touch_ollama()

                import urllib.request
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096
                    },
                    "format": "json"
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw_content = resp_data["message"]["content"]

            if is_local:
                _touch_ollama()

            parsed = _extract_json_dict(raw_content)
            if parsed is not None:
                return parsed
            logger.warning(f"Attempt {attempt}: Could not extract JSON dict from response: {raw_content[:200]}")
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                logger.error(f"All {max_attempts} attempts failed for LLM call.")
    return None


def plan_swarm_mission(prompt: str, ai_engine: str = "offline", api_key: str = None, num_drones: int = 3) -> dict:
    """
    4-stage sequential LLM prompting chain for multi-agent swarm missions.
    All 4 calls use the same model selected by `ai_engine`.
    """
    logger.info(f"Starting 4-stage swarm mission planning for prompt: '{prompt}'")
    client, gemini_client, model_name, is_local = _setup_llm_client(ai_engine, api_key)

    # ── Call 1: Prompt Cleanup & Intent Classification ──
    logger.info("Swarm Stage 1/4: Prompt Cleanup & Intent Classification...")
    ex_cleanup = [
        {
            "user": "three drone wedge patrol north 40m east 10m",
            "assistant": json.dumps({"cleaned_prompt": "Three drones patrol in a wedge formation 40 meters north and 10 meters east.", "mode": "FORMATION"})
        },
        {
            "user": "alpha inspect roof, bravo and charlie sweep perimeter loop",
            "assistant": json.dumps({"cleaned_prompt": "Leader (Alpha) inspects the roof at altitude 15m. Follower 1 (Bravo) and Follower 2 (Charlie) sweep the perimeter loop.", "mode": "INDEPENDENT"})
        },
        {
            "user": "everyone rally and regroup at central tower north 20m east 20m",
            "assistant": json.dumps({"cleaned_prompt": "All three drones converge and rally at north 20 meters, east 20 meters in a wedge formation.", "mode": "REGROUP"})
        }
    ]
    res_cleanup = _call_llm_json(SWARM_PROMPT_CLEANUP, prompt, ex_cleanup, ai_engine, model_name, client, gemini_client, is_local)
    if res_cleanup:
        cleaned_prompt = res_cleanup.get("cleaned_prompt", prompt)
        mode = res_cleanup.get("mode", "FORMATION").upper()
        if mode not in ("FORMATION", "INDEPENDENT", "REGROUP"):
            mode = "FORMATION"
    else:
        cleaned_prompt = prompt
        mode = "FORMATION"
    logger.info(f"Swarm Stage 1 Result: mode={mode}, cleaned='{cleaned_prompt}'")

    # ── Call 2: Task Allocation & Spatial Reasoning ──
    logger.info("Swarm Stage 2/4: Task Allocation & Spatial Reasoning...")
    ex_split = [
        {
            "user": "Command: Three drones patrol in a wedge formation 40 meters north and 10 meters east.\nOperational Mode: FORMATION",
            "assistant": json.dumps({
                "leader_task": "Takeoff to 12m, fly north 40m east 10m at 5 m/s, loiter 5s, then land.",
                "follower_task": "Maintain wedge formation relative to the leader throughout the flight.",
                "formation": "wedge",
                "spacing_m": 5,
                "mode": "FORMATION",
                "follower_1_offsets": [-5, -5],
                "follower_2_offsets": [-5, 5],
                "independent_tasks": None
            })
        },
        {
            "user": "Command: Leader (Alpha) inspects the roof at altitude 15m. Follower 1 (Bravo) and Follower 2 (Charlie) sweep the perimeter loop.\nOperational Mode: INDEPENDENT",
            "assistant": json.dumps({
                "leader_task": "Takeoff to 15m, fly to north 20m east 0m and loiter 15 seconds, then land.",
                "follower_task": "Follower 1 flies north 0m east 25m; Follower 2 flies north 0m east -25m.",
                "formation": "wedge",
                "spacing_m": 5,
                "mode": "INDEPENDENT",
                "follower_1_offsets": [-5, -5],
                "follower_2_offsets": [-5, 5],
                "independent_tasks": [
                    {"drone": "follower_1", "waypoints": [{"north_m": 0, "east_m": 25, "altitude_m": 12, "speed_mps": 5}]},
                    {"drone": "follower_2", "waypoints": [{"north_m": 0, "east_m": -25, "altitude_m": 12, "speed_mps": 5}]}
                ]
            })
        }
    ]
    res_split = _call_llm_json(SWARM_PROMPT_TASK_SPLIT, f"Command: {cleaned_prompt}\nOperational Mode: {mode}", ex_split, ai_engine, model_name, client, gemini_client, is_local)
    if res_split:
        task_split = res_split
    else:
        task_split = {
            "leader_task": cleaned_prompt,
            "follower_task": f"Maintain wedge formation behind the leader.",
            "formation": "wedge",
            "spacing_m": 5,
            "mode": mode,
            "follower_1_offsets": [-5, -5],
            "follower_2_offsets": [-5, 5],
            "independent_tasks": None
        }
    logger.info(f"Swarm Stage 2 Result: formation={task_split.get('formation')}, spacing={task_split.get('spacing_m')}m")

    # ── Call 3: Leader Mission JSON Generator ──
    logger.info("Swarm Stage 3/4: Leader Mission JSON Generation...")
    leader_task_prompt = task_split.get("leader_task") or cleaned_prompt
    leader_mission = plan_mission(leader_task_prompt, mode="standard", ai_engine=ai_engine, api_key=api_key, num_drones=1)
    logger.info(f"Swarm Stage 3 Result: Leader mission generated with {len(leader_mission.get('actions', []))} actions.")

    # ── Call 4: Follower Config Generator ──
    logger.info("Swarm Stage 4/4: Follower Config Generation...")
    f_input = (
        f"Leader Mission Summary: {json.dumps(leader_mission.get('actions', []))}\n"
        f"Follower Task Instructions: {task_split.get('follower_task', '')}\n"
        f"Requested Mode: {mode}\n"
        f"Formation Shape: {task_split.get('formation', 'wedge')}\n"
        f"Spacing (m): {task_split.get('spacing_m', 5)}"
    )
    ex_fconfig = [
        {
            "user": "Leader Mission Summary: [{\"type\": \"takeoff\", \"params\": {\"altitude_m\": 12}}, {\"type\": \"goto\", \"params\": {\"north_m\": 40, \"east_m\": 10, \"altitude_m\": 12, \"speed_mps\": 5}}, {\"type\": \"land\", \"params\": {}}]\nFollower Task Instructions: Maintain wedge formation relative to the leader throughout the flight.\nRequested Mode: FORMATION\nFormation Shape: wedge\nSpacing (m): 5",
            "assistant": json.dumps({
                "behavior": "maintain_formation",
                "formation": "wedge",
                "spacing_m": 5,
                "collision_radius_m": 1.0,
                "altitude_separation_m": 1.0,
                "follower_1": {
                    "role": "wingman_left",
                    "offset_north_m": -5,
                    "offset_east_m": -5,
                    "independent_actions": None
                },
                "follower_2": {
                    "role": "wingman_right",
                    "offset_north_m": -5,
                    "offset_east_m": 5,
                    "independent_actions": None
                }
            })
        }
    ]
    res_fconfig = _call_llm_json(SWARM_PROMPT_FOLLOWER_CONFIG, f_input, ex_fconfig, ai_engine, model_name, client, gemini_client, is_local)
    if res_fconfig:
        follower_config = res_fconfig
    else:
        f1_off = task_split.get("follower_1_offsets", [-5, -5])
        f2_off = task_split.get("follower_2_offsets", [-5, 5])
        follower_config = {
            "behavior": "maintain_formation" if mode == "FORMATION" else mode.lower(),
            "formation": task_split.get("formation", "wedge"),
            "spacing_m": task_split.get("spacing_m", 5),
            "collision_radius_m": 1.0,
            "altitude_separation_m": 1.0,
            "follower_1": {
                "role": "wingman_left",
                "offset_north_m": f1_off[0] if isinstance(f1_off, list) and len(f1_off) >= 2 else -5,
                "offset_east_m": f1_off[1] if isinstance(f1_off, list) and len(f1_off) >= 2 else -5,
                "independent_actions": task_split.get("independent_tasks")
            },
            "follower_2": {
                "role": "wingman_right",
                "offset_north_m": f2_off[0] if isinstance(f2_off, list) and len(f2_off) >= 2 else -5,
                "offset_east_m": f2_off[1] if isinstance(f2_off, list) and len(f2_off) >= 2 else 5,
                "independent_actions": task_split.get("independent_tasks")
            }
        }
    logger.info("Swarm Stage 4 Result: Follower config complete.")

    return {
        "mission_name": leader_mission.get("mission_name", "Swarm Multi-Agent Mission"),
        "vehicle_type": "swarm",
        "drone_count": num_drones,
        "formation": task_split.get("formation", "wedge").lower(),
        "spacing_m": float(task_split.get("spacing_m", 5)),
        "collision_radius_m": float(follower_config.get("collision_radius_m", 1.0)),
        "mode": mode,
        "leader_mission": leader_mission,
        "follower_config": follower_config,
        "actions": leader_mission.get("actions", []),
        "llm_stages": {
            "cleanup": {"cleaned_prompt": cleaned_prompt, "mode": mode},
            "task_split": task_split,
            "leader_json": leader_mission,
            "follower_config": follower_config
        }
    }

